import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import scipy.spatial.distance as distance
import torch
from pandas import DataFrame
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, logging

import src.utils as utils
from src.config import ID, Config, pairing, sampling
from src.distance_model import DistanceModel
from src.lscd import Target

logging.set_verbosity_error()


@dataclass
class VectorModel(DistanceModel):
    def __init__(self, config: Config, targets: List[Target]):
        self._device = None
        self._tokenizer = None
        self._model = None
        self._index = None
        self._vectors = None
        self._uses = None

        self.targets = targets
        self._distances = {
            target.name: {s: {p: dict() for p in pairing} for s in sampling}
            for target in self.targets
        }
        self.config = config
        self.index_dir = utils.path(".cache")
        self.index_dir.mkdir(exist_ok=True)

    @property
    def uses(self):
        if self._uses is None:
            self._uses = pd.concat([target.uses for target in self.targets], axis=0)
        return self._uses

    @property
    def device(self):
        if self._device is None:
            self._device = torch.device(
                "cpu"
                if self.config.model.gpu is None
                else f"cuda:{self.config.model.gpu}"
            )
        return self._device

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model.name, use_fast=True, model_max_length=int(1e30)
            )
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, other: str):
        self._tokenizer = AutoTokenizer.from_pretrained(
            other, use_fast=True, model_max_length=int(1e30)
        )

    @property
    def model(self):
        if self._model is None:
            self._model = AutoModel.from_pretrained(
                self.config.model.name, output_hidden_states=True
            ).to(self.device)
            self._model.eval()
        return self._model

    @model.setter
    def model(self, other: str):
        self._model = AutoModel.from_pretrained(other, output_hidden_states=True).to(
            self.device
        )
        self._model.eval()

    @property
    def index(self):
        if self._index is None:
            path = self.index_dir / "index.csv"
            if path.exists():
                self._index = pd.read_csv(path, engine="pyarrow")
            else:
                self._index = DataFrame(
                    columns=["model", "target", "id", "preprocessing", "dataset"],
                )
        self.clean_cache()
        return self._index

    def index_row(self, target_name: str, id: str):
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
        path = self.index_dir / "index.csv"
        self._index = new
        self._index.to_csv(path, index=False)

    def clean_cache(self):
        valid_ids = set(self._index.id.tolist())
        for file in utils.path(".cache").iterdir():
            if file.stem != "index" and file.stem not in valid_ids:
                file.unlink()

    def truncate_input(
        self,
        subword_indices: np.array,
        input_ids: torch.Tensor,
        target_subword_indices: List[bool],
        p_before: float = 0.95,
    ) -> Tuple[np.array, torch.Tensor, List[bool]]:

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
        target_subword_indices = target_subword_indices[lindex:rindex]
        return subword_indices, input_ids, target_subword_indices

    def distances(
        self,
        target: Target,
        sampling: sampling,
        pairing: pairing,
        method: Callable = distance.cosine,
        return_pairs: bool = False,
    ) -> Union[Tuple[List[Tuple[ID, ID]], List[float]], List[float]]:
        ids = sampling(pairing, target, **self.config.model.measure.sampling_params)
        for id_pair in ids:
            if id_pair not in self._distances[target.name][sampling][pairing]:
                self._distances[target.name][sampling][pairing][id_pair] = method(
                    self.vectors[id_pair[0]], self.vectors[id_pair[1]]
                )
        if return_pairs:
            return list(self._distances[target.name][sampling][pairing].keys()), list(
                self._distances[target.name][sampling][pairing].values()
            )
        else:
            return list(self._distances[target.name][sampling][pairing].values())

    @property
    def vectors(self) -> Dict[ID, np.array]:
        if self._vectors is None:
            self._vectors = {}

            pbar = tqdm(self.targets, leave=False)
            for target in pbar:
                pbar.set_description(f"Processing target {target.name}", refresh=False)
                token_embeddings = {}
                subword_indices = {}

                row = self.index[
                    (self.index.model == self.config.model.name)
                    & (self.index.target == target.name)
                    & (
                        self.index.preprocessing
                        == self.config.dataset.preprocessing.method_name
                    )
                    & (self.index.dataset == self.config.dataset.name)
                ]

                if not row.empty:
                    id_ = row["id"].iloc[0]
                    token_embeddings = np.load(
                        str(self.index_dir / f"{id_}.npz"), mmap_mode="r"
                    )
                else:
                    for _, use in target.uses.iterrows():
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
                            sub_start >= use.target_index_begin
                            and sub_end <= use.target_index_end
                            for sub_start, sub_end in subword_indices[use.identifier]
                        ]

                        if (
                            len(encoded.tokens())
                            > self.model.config.max_position_embeddings
                        ):
                            (
                                subword_indices[use.identifier],
                                input_ids,
                                target_subword_indices,
                            ) = self.truncate_input(
                                subword_indices[use.identifier],
                                input_ids,
                                target_subword_indices,
                            )

                        segment_ids = torch.ones_like(input_ids)

                        self.model.eval()
                        with torch.no_grad():
                            outputs = self.model(input_ids, segment_ids)
                            token_embeddings[use.identifier] = torch.stack(
                                outputs[2], dim=0
                            )
                            token_embeddings[use.identifier] = torch.squeeze(
                                token_embeddings[use.identifier], dim=1
                            )
                            token_embeddings[use.identifier] = token_embeddings[
                                use.identifier
                            ].permute(1, 0, 2)
                            token_embeddings[use.identifier] = token_embeddings[
                                use.identifier
                            ][torch.tensor(target_subword_indices), :, :]
                            token_embeddings[use.identifier] = (
                                token_embeddings[use.identifier].cpu().numpy()
                            )

                    id_ = str(uuid.uuid4())
                    while (self.index_dir / f"{id_}.npz").exists():
                        id_ = str(uuid.uuid4())

                    self.index = pd.concat(
                        [self.index, self.index_row(target_name=target.name, id=id_)],
                        ignore_index=True,
                    )
                    with open(
                        file=self.index_dir / f"{id_}.npz", mode="wb"
                    ) as f_target_embeddings:
                        np.savez(f_target_embeddings, **token_embeddings)

                for _, use in target.uses.iterrows():
                    target_subword_embeddings = token_embeddings[use.identifier]
                    target_embedding = self.config.model.subword_aggregation(
                        vectors=target_subword_embeddings
                    ).squeeze()
                    self._vectors[use.identifier] = self.config.model.layer_aggregation(
                        np.take(target_embedding, self.config.model.layers, axis=0)
                    )

        return self._vectors
