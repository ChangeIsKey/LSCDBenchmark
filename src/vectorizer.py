import os
import uuid
from dataclasses import dataclass
from typing import List, Tuple

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from transformers import AutoModel, AutoTokenizer, logging

import src.utils as utils
from src.config import Config
from src.use import Use

logging.set_verbosity_error()


@dataclass
class Vectorizer:
    config: Config

    def __post_init__(self):
        self.device = torch.device(
            "cpu" if self.config.model.gpu is None else f"cuda:{self.config.model.gpu}"
        )
        self._tokenizer = None
        self._model = None
        self._index = None
        self.index_dir = utils.path(".cache")
        self.index_dir.mkdir(exist_ok=True)

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model.name, use_fast=True, model_max_length=int(1e30)
            )
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            self._model = AutoModel.from_pretrained(
                self.config.model.name, output_hidden_states=True
            ).to(self.device)
            self._model.eval()
        return self._model

    @property
    def index(self):
        path = self.index_dir / "index"
        if self._index is None:
            if path.exists():
                self._index = pd.read_feather(path)
                self.clean_cache()
            else:
                self._index = DataFrame(
                    columns=["model", "target", "id", "preprocessing", "dataset"],
                )
        return self._index

    def row(self, target_name: str, id: str):
        return DataFrame(
            [
                {
                    "model": self.config.model.name,
                    "target": target_name,
                    "id": id,
                    "preprocessing": self.config.dataset.preprocessing.method_name,
                    "dataset": self.config.dataset.name,
                }
            ]
        )

    @index.setter
    def index(self, new: DataFrame):
        path = self.index_dir / "index"
        self._index = new
        self._index.to_feather(path)

    def clean_cache(self):
        existent_ids = self._index["id"].tolist()
        for filename in self.index_dir.iterdir():
            if filename.stem != "index":
                id_ = filename.stem.replace("-offset-mapping", "")
                if id_ not in existent_ids:
                    os.remove(filename)

    def truncate_input(
        self,
        subword_indices: np.array,
        input_ids: torch.Tensor,
        target_subword_indices: torch.Tensor,
        p_before: float = 0.95,
    ) -> Tuple[np.array, torch.Tensor]:

        n_target_subtokens = target_subword_indices.count(True)
        tokens_before = int(
            (self.model.config.max_position_embeddings - n_target_subtokens) * p_before
        )
        tokens_after = (
            self.model.config.max_position_embeddings
            - tokens_before
            - n_target_subtokens
        )

        # get index of the first target subword
        lindex_target = target_subword_indices.index(True)  
        # get index of the last target subword
        rindex_target = lindex_target + n_target_subtokens
        lindex = max(lindex_target - tokens_before, 0)
        rindex = rindex_target + tokens_after
        subword_indices = subword_indices[lindex:rindex]
        input_ids = input_ids[:, lindex:rindex]

        return subword_indices, input_ids

    def __call__(self, uses: List[Use]) -> torch.Tensor:
        hidden_states = {}
        subword_indices = {}

        row = self.index[
            (self.index.model == self.config.model.name)
            & (self.index.target == uses[0].target)
            & (
                self.index.preprocessing
                == self.config.dataset.preprocessing.method_name
            )
            & (self.index.dataset == self.config.dataset.name)
        ]

        target = uses[0].target
        if not row.empty:
            id_ = row["id"].iloc[0]
            hidden_states = np.load(self.index_dir / f"{id_}.npz", mmap_mode="r")
            subword_indices = np.load(
                self.index_dir / f"{id_}-offset-mapping.npz", mmap_mode="r"
            )
        else:
            for use in tqdm(
                uses, desc=f"Vectorizing uses of target '{target}'", leave=False
            ):
                encoded = self.tokenizer.encode_plus(
                    use.context_preprocessed,
                    return_tensors="pt",
                    add_special_tokens=True,
                    return_offsets_mapping=True,
                ).to(self.device)

                # TODO fix segment ids

                subword_indices[use.identifier] = (
                    encoded["offset_mapping"].squeeze(0).cpu().numpy()
                )
                input_ids = encoded["input_ids"].to(self.device)

                target_subword_indices = [
                    (
                        sub_start >= use.target_index_begin
                        and sub_end <= use.target_index_end
                    )
                    for sub_start, sub_end in subword_indices[use.identifier]
                ]

                if len(encoded.tokens()) > self.model.config.max_position_embeddings:
                    subword_indices[use.identifier], input_ids = self.truncate_input(
                        subword_indices[use.identifier],
                        input_ids,
                        target_subword_indices,
                    )

                segment_ids = torch.ones_like(input_ids).to(self.device)

                self.model.eval()
                with torch.no_grad():
                    outputs = self.model(input_ids, segment_ids)
                    hidden_states[use.identifier] = (
                        torch.stack(outputs[2], dim=0).cpu().numpy()
                    )

            id_ = str(uuid.uuid4())
            self.index = pd.concat(
                [self.index, self.row(target_name=use.target, id=id_)],
                ignore_index=True,
            )
            with open(file=self.index_dir / f"{id_}.npz", mode="wb") as f_hidden_states:
                np.savez(f_hidden_states, **hidden_states)
            with open(
                file=self.index_dir / f"{id_}-offset-mapping.npz", mode="wb"
            ) as f_subword_indices:
                np.savez(f_subword_indices, **subword_indices)

        target_vectors = []
        for use in tqdm(uses, desc=f"Processing uses of {target}", leave=False):
            layers = self.config.model.layer_aggregation(
                hidden_states[use.identifier][self.config.model.layers]
            )

            target_subword_indices = [
                (
                    sub_start >= use.target_index_begin
                    and sub_end <= use.target_index_end
                )
                for sub_start, sub_end in subword_indices[use.identifier]
            ]

            embeddings = layers.squeeze(0)[target_subword_indices]
            vec = self.config.model.subword_aggregation(vectors=embeddings)
            target_vectors.append(vec)

        return target_vectors
