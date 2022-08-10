import os
import uuid
from dataclasses import dataclass
from typing import List

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
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model.name)
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

        if not row.empty:
            id_ = row["id"].iloc[0]
            hidden_states = np.load(self.index_dir / f"{id_}.npz", mmap_mode="r")
            subword_indices = np.load(
                self.index_dir / f"{id_}-offset-mapping.npz", mmap_mode="r"
            )
        else:
            target = uses[0].target
            for use in tqdm(
                uses, desc=f"Vectorizing uses of target '{target}'", leave=False
            ):
                encoded = self.tokenizer(
                    use.context_preprocessed,
                    return_tensors="pt",
                    truncation=True,
                    add_special_tokens=True,
                    return_offsets_mapping=True,
                ).to(self.device)

                # TODO fix segment ids

                subword_indices[use.identifier] = (
                    encoded["offset_mapping"].squeeze(0).cpu().numpy()
                )

                try:
                    target_subword_indices = [
                        (
                            sub_start >= use.target_index_begin
                            and sub_end <= use.target_index_end
                        )
                        for sub_start, sub_end in subword_indices[use.identifier]
                    ]
                except ValueError as e:
                    print(use.identifier, use.target)
                    raise e

                lindex_target = target_subword_indices.index(True)
                rindex_target = lindex_target + target_subword_indices.count(True)
                lindex = max(rindex_target - 512, 0)
                subword_indices[use.identifier] = subword_indices[use.identifier][
                    lindex:
                ]

                input_ids = encoded["input_ids"][lindex:].to(self.device)
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
        for use in uses:
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
