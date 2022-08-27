import logging
import uuid
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import scipy.spatial.distance as distance
import torch
from pandas import DataFrame, Series
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BatchEncoding
from transformers import logging as trans_logging

import src.utils as utils
from src.config import UseID, Config, pairing, sampling
from src.distance_model import DistanceModel
from src.lscd import Target

trans_logging.set_verbosity_error()

log = logging.getLogger(__name__)


@dataclass
class VectorModel(DistanceModel):
    def __init__(self, config: Config, targets: List[Target]):
        self._device = None
        self._tokenizer = None
        self._model = None
        self._index = None
        self._vectors = None

        self.targets = targets
        self._distances = {
            target.name: {s: {p: dict() for p in pairing} for s in sampling}
            for target in self.targets
        }
        self.config = config
        self.index_dir = utils.path(".cache")
        self.index_dir.mkdir(exist_ok=True)

    @property
    def device(self):
        if self._device is None:
            self._device = torch.device(
                "cpu"
                if self.config.gpu is None
                else f"cuda:{self.config.gpu}"
            )
        return self._device

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model, use_fast=True, model_max_length=int(1e30)
            )
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            self._model = AutoModel.from_pretrained(
                self.config.model, output_hidden_states=True
            ).to(self.device)
            self._model.eval()
        return self._model

    @property
    def index(self):
        if self._index is None:
            path = self.index_dir / "index.csv"
            if path.exists():
                self._index = pd.read_csv(path, engine="pyarrow")
            else:
                self._index = DataFrame(
                    columns=["model", "target", "id", "preprocessing", "dataset_name", "dataset_version"],
                )
            self.clean_cache()
        return self._index

    def index_row(self, target_name: str, id: str):
        return DataFrame(
            [
                {
                    "model": self.config.model,
                    "target": target_name,
                    "id": id,
                    "preprocessing": self.config.preprocessing.method,
                    "dataset_name": self.config.dataset.name,
                    "dataset_version": self.config.dataset.version,
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
        for file in self.index_dir.iterdir():
            if file.stem != "index" and file.stem not in valid_ids:
                file.unlink()

    def truncation_indices(
        self,
        target_subword_indices: List[bool],
        p_before: float = 0.95,
    ) -> Tuple[int, int]:

        max_tokens = 512
        n_target_subtokens = target_subword_indices.count(True)
        tokens_before = int(
            (max_tokens - n_target_subtokens) * p_before
        )
        tokens_after = max_tokens - tokens_before - n_target_subtokens

        # get index of the first target subword
        lindex_target = target_subword_indices.index(True)
        # get index of the last target subword
        rindex_target = lindex_target + n_target_subtokens + 1
        lindex = max(lindex_target - tokens_before, 0)
        rindex = rindex_target + tokens_after - 1
        return lindex, rindex

    def distances(
        self,
        target: Target,
        sampling: sampling,
        pairing: pairing,
        method: Callable = distance.cosine,
        return_pairs: bool = False,
    ) -> Union[Tuple[List[Tuple[UseID, UseID]], List[float]], List[float]]:

        ids = sampling(pairing, target, **self.config.measure.sampling_params)
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

    def retrieve_embeddings(self, target: Target) -> np.ndarray | None:
        mask = (
            (self.index.model == self.config.model)
            & (self.index.target == target.name)
            & (self.index.preprocessing == self.config.preprocessing.method)
            & (self.index.dataset_name == self.config.dataset.name)
            & (self.index.dataset_version == self.config.dataset.version)
        ) 
        row = self.index[mask]

        if not row.empty:
            assert len(row) == 1
            id_ = row.id.iloc[0]
            path = self.index_dir / f"{id_}.npz"
            return np.load(path, mmap_mode="r")

        return None

    def store_embeddings(self, target: Target, embeddings: Dict[str, np.ndarray]) -> None:
        ids = self.index.id.tolist()
        while True:
            id_ = str(uuid.uuid4())
            if id_ not in ids:
                with open(file=self.index_dir / f"{id_}.npz", mode="wb") as f:
                    np.savez(f, **embeddings)
                    log.info(f"Saved embeddings to disk as {id_}.npz")

                self.index = pd.concat(
                    [self.index, self.index_row(target_name=target.name, id=id_)],
                    ignore_index=True,
                )

                log.info("Logged record of new embedding file")

                break

        

    def tokenize(self, use: Series) -> BatchEncoding:
        return self.tokenizer.encode_plus(
            text=use.context_preprocessed, 
            return_tensors="pt", 
            add_special_tokens=True
        ).to(self.device)

    def aggregate(self, embedding: np.ndarray) -> np.ndarray:
        return self.config.layer_aggregation(
            self.config.subword_aggregation(embedding)
            .squeeze()
            .take(self.config.layers, axis=0)
        )
        

    @property
    def vectors(self) -> Dict[UseID, np.array]:
        if self._vectors is None:
            self._vectors = {}

            pbar = tqdm(self.targets, leave=False)

            for target in pbar:
                pbar.set_description(f"Processing target {target.name}", refresh=True)
                log.info(f"PROCESSING TARGET: {target.name}")

                
                token_embeddings = self.retrieve_embeddings(target)

                if token_embeddings is None:
                    token_embeddings = {}
                    for _, use in target.uses.iterrows():
                        log.info(
                            f"PROCESSING USE `{use.identifier}`: {use.context_preprocessed}"
                        )
                        
                        encoding = self.tokenize(use)
                        input_ids = encoding["input_ids"].to(self.device)
                        tokens = encoding.tokens()
                        subword_spans = [
                            encoding.token_to_chars(i) for i in range(len(tokens))
                        ]

                        log.info(f"Extracted {len(tokens)} tokens: {tokens}")

                        target_indices = [
                            span.start >= use.target_index_begin and 
                            span.end <= use.target_index_end
                            if span is not None else False
                            for span in subword_spans
                        ]

                        # truncate input if the model cannot handle it
                        if len(tokens) > 512:
                            lindex, rindex = self.truncation_indices(target_indices)
                            tokens = tokens[lindex:rindex]
                            input_ids = input_ids[:, lindex:rindex]
                            target_indices = target_indices[lindex:rindex]

                            log.info(f"Truncated input")
                            log.info(f"New tokens: {tokens}")

                        extracted_subwords = [tokens[i] for i, value in enumerate(target_indices) if value]

                        log.info(f"Selected subwords: {extracted_subwords}")
                        log.info(f"Size of input_ids: {input_ids.size()}")


                        with torch.no_grad():
                            outputs = self.model(input_ids, torch.ones_like(input_ids))
                        
                        token_embeddings[use.identifier] = (
                            # stack the layers
                            torch.stack(outputs[2], dim=0)  
                            # we don't vectorize in batches, so we can get rid of the batches dimension
                            .squeeze(dim=1)  
                            # swap the subwords and layers dimension
                            .permute(1, 0, 2)
                            # select the target's subwords' embeddings
                            [torch.tensor(target_indices), :, :]
                            # convert to numpy array
                            .cpu().numpy()
                        )
                        
                        log.info(f"Size of pre-subword-agregated tensor: {token_embeddings[use.identifier].shape}")

                    self.store_embeddings(target, token_embeddings)

                self._vectors.update({
                    identifier: self.aggregate(embedding)
                    for identifier, embedding in token_embeddings.items()
                })

        return self._vectors
