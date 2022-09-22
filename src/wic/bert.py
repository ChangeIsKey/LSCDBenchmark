import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple

import numpy as np
import pandas as pd
import scipy.spatial.distance as distance
import torch
import torch.nn.functional as F
from pandas import DataFrame
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BatchEncoding
from transformers import logging as trans_logging

import src.utils as utils
from src.use import Use
from src.wic.model import WICModel as WicModel

trans_logging.set_verbosity_error()

if TYPE_CHECKING:
    from src.config.config import Config

log = logging.getLogger(__name__)


@dataclass
class ContextualEmbedderWIC(WicModel):
    def __init__(self, config: "Config"):
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
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def device(self) -> torch.device:
        if self._device is None:
            self._device = torch.device(
                f"cuda:{self.config.gpu}"
                if self.config.gpu is not None and torch.cuda.is_available()
                else "cpu"
            )
        return self._device

    @property
    def tokenizer(self) -> AutoTokenizer:
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model.wic.id, use_fast=True, model_max_length=int(1e30)
            )
        return self._tokenizer

    @property
    def model(self) -> AutoModel:
        if self._model is None:
            self._model = AutoModel.from_pretrained(
                self.config.model.wic.id, output_hidden_states=True
            ).to(self.device)
            self._model.eval()
        return self._model

    @property
    def index(self) -> DataFrame:
        if self._index is None:
            path = self.index_dir / "index.csv"
            if path.exists():
                self._index = pd.read_csv(path, engine="pyarrow")
            else:
                self._index = DataFrame(columns=self.index_row(use=None, id=None))
            self.clean_cache()
        return self._index

    def index_row(self, use: Use, id: str) -> DataFrame:
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
            self.config.model.wic.id,
            use.identifier,
            id,
            self.config.dataset.preprocessing.target,
            self.config.dataset.name,
            self.config.dataset.version,
            self.config.model.wic.truncation_tokens_before_target,
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

    def predict(self, use_pairs: list[tuple[Use, Use]]) -> list[float]:
        similarities = []
        for use_1, use_2 in tqdm(
            use_pairs, desc="Encoding and calculating distances", leave=False
        ):
            enc_1 = self.encode(use_1)
            enc_2 = self.encode(use_2)
            similarities.append(-self.config.model.wic.distance_metric(enc_1, enc_2))
        return similarities

    def retrieve_embedding(self, use: Use) -> np.ndarray | None:
        mask = (
            (self.index.model == self.config.model.wic.id)
            & (self.index.use == use.identifier)
            & (self.index.preprocessing == self.config.dataset.preprocessing.target)
            & (self.index.dataset_name == self.config.dataset.name)
            & (self.index.dataset_version == self.config.dataset.version)
            & (
                self.index.tokens_before
                == self.config.model.wic.truncation_tokens_before_target
            )
        )
        row = self.index[mask]

        if not row.empty:
            assert len(row) == 1
            id_ = row.id.iat[0]
            path = self.index_dir / f"{id_}.npy"
            return np.load(path, mmap="r")

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

    def aggregate(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.config.model.wic.layer_aggregation(
            self.config.model.wic.subword_aggregation(embedding)
            .squeeze()
            .index_select(index=self.config.model.wic.layers, dim=0)
        )

    def encode(self, use: Use) -> np.ndarray:
        if self._vectors is None:
            self._vectors = {}

        embedding = self._vectors.get(use.identifier)
        if embedding is None:
            log.info(f"PROCESSING USE `{use.identifier}`: {use.context}")
            log.info(f"Target character indices: {use.indices}")
            log.info(
                f"Context slice corresponding to target indices: \
                {use.context[use.indices[0]:use.indices[1]]}"
            )

            encoding = self.tokenize(use)
            input_ids = encoding["input_ids"].to(self.device)  # type: ignore
            tokens = encoding.tokens()
            subword_spans = [encoding.token_to_chars(i) for i in range(len(tokens))]

            log.info(f"Extracted {len(tokens)} tokens: {tokens}")

            target_indices = [
                span.start >= use.indices[0] and span.end <= use.indices[1]
                if span is not None
                else False
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

            extracted_subwords = [
                tokens[i] for i, value in enumerate(target_indices) if value
            ]
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
            )

            log.info(f"Size of pre-subword-agregated tensor: {embedding.shape}")

        self._vectors[use.identifier] = embedding
        embedding = self.aggregate(embedding).cpu().numpy()

        return embedding
