import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
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
from src.config.config import Config, UseID
from src.distance_model import DistanceModel
from src.target import Target, Sampling, Pairing
from src.use import Use

trans_logging.set_verbosity_error()

log = logging.getLogger(__name__)


@dataclass
class VectorModel(DistanceModel):
    def __init__(self, config: Config):
        self.config = config

        self._device = None
        self._tokenizer = None
        self._model = None
        self._index = None
        self._vectors = None

    @property
    def index_dir(self) -> Path:
        path = os.getenv("BENCHMARK_CACHE")
        path = Path(path) if path is not None else utils.path(".cache")
        path.mkdir(exist_ok=True)
        return path

    @property
    def device(self):
        if self._device is None:
            self._device = torch.device(
                f"cuda:{self.config.gpu}" if self.config.gpu is not None and torch.cuda.is_available() 
                else "cpu"
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
                self._index = DataFrame(columns=self.index_row(use=None, id=None))
            self.clean_cache()
        return self._index

    def index_row(self, use: Use, id: str):
        headers = [
            "model",
            "use",
            "id",
            "preprocessing",
            "dataset_name",
            "dataset_version",
            "tokens_before",
        ]
        if use is None and id is None:
            return headers
        
        values = [
            self.config.model,
            use.identifier,
            id,
            self.config.preprocessing.method,
            self.config.dataset.name,
            self.config.dataset.version,
            self.config.truncation.tokens_before,
        ]
        return DataFrame([dict(zip(headers, values))])

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
    ) -> Tuple[int, int]:

        max_tokens = 512
        n_target_subtokens = target_subword_indices.count(True)
        tokens_before = int(
            (max_tokens - n_target_subtokens) * self.config.truncation.tokens_before
        )
        tokens_after = max_tokens - tokens_before - n_target_subtokens

        # get index of the first target subword
        lindex_target = target_subword_indices.index(True)
        # get index of the last target subword
        rindex_target = lindex_target + n_target_subtokens + 1
        lindex = max(lindex_target - tokens_before, 0)
        rindex = rindex_target + tokens_after - 1
        return lindex, rindex

    def distances(self, use_pairs: list[tuple[Use, Use]]) -> dict[tuple[Use, Use], float]:
        distances = [
            distance.cosine(self.vectors[u1.identifier], self.vectors[u2.identifier]) 
            for u1, u2 in use_pairs
        ]
        return dict(zip(use_pairs, distances))

    def retrieve_embedding(self, use: Use) -> np.ndarray | None:
        mask = (
            (self.index.model == self.config.model.name)
            & (self.index.use == use.identifier)
            & (self.index.preprocessing == self.config.dataset.preprocessing.method)
            & (self.index.dataset_name == self.config.dataset.name)
            & (self.index.dataset_version == self.config.dataset.version)
            & (self.index.tokens_before == self.config.model.truncation.tokens_before)
        )
        row = self.index[mask]

        if not row.empty:
            assert len(row) == 1
            id_ = row.id.iat[0]
            path = self.index_dir / f"{id_}.npy"
            return np.load(path)

        return None

    def store_embedding(self, use: Use, embedding: np.ndarray) -> None:
        ids = self.index.id.tolist()
        while True:
            id_ = str(uuid.uuid4())
            if id_ not in ids:
                with open(file=self.index_dir / f"{id_}.npy", mode="wb") as f:
                    np.save(f, embedding)

                self.index = pd.concat(
                    [self.index, self.index_row(use=use, id=id_)],
                    ignore_index=True,
                )

                break

    def tokenize(self, use: Use) -> BatchEncoding:
        return self.tokenizer.encode_plus(
            text=use.context, return_tensors="pt", add_special_tokens=True
        ).to(self.device)

    def aggregate(self, embedding: np.ndarray) -> np.ndarray:
        return self.config.model.layer_aggregation(
            self.config.model.subword_aggregation(embedding)
            .squeeze()
            .take(self.config.model.layers, axis=0)
        )

    def encode(self, use: Use) -> np.ndarray:
        if self._vectors is None:
            self._vectors = {}

        embedding = self.retrieve_embedding(use)
        if embedding is None:
            log.info(
                f"PROCESSING USE `{use.identifier}`: {use.context_preprocessed}"
            )
            log.info(f"Target character indices: {use.indices}")
            log.info(f"Context slice corresponding to target indices: {use.context[use.indices[0]:use.indices[1]]}")
                        
            encoding = self.tokenize(use)
            input_ids = encoding["input_ids"].to(self.device)  # type: ignore
            tokens = encoding.tokens()
            subword_spans = [
                encoding.token_to_chars(i) for i in range(len(tokens))
            ]

            log.info(f"Extracted {len(tokens)} tokens: {tokens}")

            target_indices = [
                span.start >= use.indices[0] and span.end <= use.indices[1]
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
                        
            embedding = (
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
                        
            log.info(f"Size of pre-subword-agregated tensor: {embedding.shape}")
            self.store_embedding(use, embedding)

        embedding = self.aggregate(embedding) 
        self._vectors[use.identifier] = embedding

        return embedding
