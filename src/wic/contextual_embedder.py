import logging
import signal
import os
import uuid
from collections import defaultdict
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import (
    Any,
    Callable,
    TypeAlias,
    TypedDict,
)

import numpy as np
import pandas as pd
import json
import torch
from pandas import DataFrame, Series
from pydantic import (
    BaseModel,
    Field,
    PositiveInt,
    PrivateAttr,
    conlist,
)
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    logging as trans_logging,
)

from src import utils
from src.use import (
    Use,
    UseID
)
from src.wic.model import Model

trans_logging.set_verbosity_error()

log = logging.getLogger(__name__)


class LayerAggregator(str, Enum):
    AVERAGE = "average"
    CONCAT = "concat"
    SUM = "sum"

    def __call__(
        self,
        layers: np.ndarray
    ) -> np.ndarray:
        match self:
            case self.AVERAGE:
                return np.mean(layers, axis=0)
            case self.SUM:
                return np.sum(layers, axis=0)
            case self.CONCAT:
                return np.ravel(layers)
            case _:
                raise ValueError


class SubwordAggregator(str, Enum):
    AVERAGE = "average"
    FIRST = "first"
    LAST = "last"
    SUM = "sum"

    def __call__(
        self,
        vectors: np.ndarray
    ) -> np.ndarray:
        match self:
            case self.AVERAGE:
                return np.mean(vectors, axis=0, keepdims=True)
            case self.SUM:
                return np.sum(vectors, axis=0, keepdims=True)
            case self.FIRST:
                return vectors[0]
            case self.LAST:
                return vectors[-1]
            case _:
                raise ValueError


class DatasetMetadata(BaseModel):
    name: str
    version: str
    preprocessing: str


class ContextualEmbedderMetadata(BaseModel):
    pre_target_tokens: float


class CacheParams(BaseModel):
    dataset: DatasetMetadata
    contextual_embedder: ContextualEmbedderMetadata

TargetName: TypeAlias = str

class Cache(BaseModel):
    metadata: CacheParams
    _cache: dict[TargetName, dict[UseID, np.ndarray]] = PrivateAttr(default_factory=dict)
    _targets_with_new_uses: set[TargetName] = PrivateAttr(default_factory=set)
    _index: DataFrame = PrivateAttr(default=None)
    _index_dir: Path = PrivateAttr(default=None)
    _index_path: Path = PrivateAttr(default=None)

    def __init__(self, **data):
        super().__init__(**data)
        self._index_dir = utils.path(os.getenv("CACHE_DIR") or ".cache")
        self._index_path = self._index_dir / "index.csv"
        try:
            self._index = pd.read_csv(filepath_or_buffer=self._index_path, engine="pyarrow")
            self.clean()
        except FileNotFoundError:
            self._index = DataFrame()

    def add_use(self, use: Use, embedding: np.ndarray, new: bool) -> None:
        if not use.target in self._cache:
            self._cache[use.target] = {}
        self._cache[use.target][use.identifier] = embedding
        if new:
            self._targets_with_new_uses.add(use.target)

    def retrieve(self, use: Use) -> np.ndarray | None:
        if not use.target in self._cache:
            loaded = self.load(use.target)
            if loaded is None:
                return None
            self._cache[use.target] = loaded
        return self._cache[use.target].get(use.identifier)  # this can still be None

    def mask(self, target: str):
        try:
            mask = self._index.target == target
            metadata = self.metadata.dict()
            for col in self._index.columns.tolist():
                if "." in col:
                    key, child = col.split(".")
                    mask &= self._index[col] == metadata[key][child]
            return mask
        except AttributeError:
            return []

    def load(self, target: str) -> dict[UseID, np.ndarray] | None:
        mask = self.mask(target)
        df = self._index[mask]
        assert len(df) < 2

        if df.empty:
            return None

        identifier = df.id.iloc[0]
        path = str(self._index_dir / f"{identifier}.npz")
        return dict(np.load(path, mmap_mode="r"))

    def _ids(self) -> set[UseID]:
        try:
            return set(self._index.id.tolist())
        except AttributeError:
            return set()

    def targets(self) -> set[TargetName]:
        return set(self._cache.keys())

    def clean(self):
        self._index.drop_duplicates(subset=[col for col in self._index.columns.tolist() if col != "id"], inplace=True, keep="last")
        valid_ids = self._ids()
        for file in self._index_dir.iterdir():
            if file.name != "index.csv" and file.stem not in valid_ids:
                file.unlink()


    def persist(self, target: str) -> None:
        while True:
            identifier = str(uuid.uuid4())
            if identifier not in self._ids():
                self._index_dir.mkdir(exist_ok=True, parents=True)
                with open(file=self._index_dir / f"{identifier}.npz", mode="wb") as f:
                    np.savez(f, **self._cache[target])
                    log.info(f"Saved embeddings to disk as {identifier}.npz")

                self._index = pd.concat([
                    self._index,
                    pd.json_normalize(self.metadata.dict()).assign(id=identifier, target=target)
                ], ignore_index=True)
                self._index.to_csv(path_or_buf=self._index_path, index=False)
                self._targets_with_new_uses.remove(target)
                log.info("Logged record of new embedding file")

                break

    def has_new_uses(self, target: str) -> bool:
        return target in self._targets_with_new_uses


class ContextualEmbedder(Model):
    layers: conlist(item_type=PositiveInt, unique_items=True)  # type: ignore
    layer_aggregation: LayerAggregator
    subword_aggregation: SubwordAggregator
    truncation_tokens_before_target: float
    similarity_metric: Callable[..., float]
    id: str
    gpu: int | None


    _device: torch.device = PrivateAttr(default=None)
    _tokenizer: PreTrainedTokenizerBase = PrivateAttr(default=None)
    _model: PreTrainedModel = PrivateAttr(default=None)
    _layer_mask: np.ndarray = PrivateAttr(default=None)

    # attributes for cleanup and vector caching
    cache: Cache
    _signal_received: Any = PrivateAttr(default=None)
    _old_exit_handler: Any = PrivateAttr(default=None)

    def __init__(
        self,
        **data
    ) -> None:
        super().__init__(**data)
        self._layer_mask = np.array(self.layers, dtype=int)

    def __enter__(self):
        self._signal_received = False
        self._old_exit_handler = signal.signal(signal.SIGINT, self.on_exit)

    def on_exit(self, sig, frame):
        self._signal_received = (sig, frame)
        targets = self.cache.targets()
        for target in targets:
            if self.cache.has_new_uses(target):
                self.cache.persist(target)

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.signal(signal.SIGINT, self._old_exit_handler)
        if self._signal_received:
            self._old_exit_handler(*self._signal_received)

    @property
    def device(
        self
    ) -> torch.device:
        if self._device is None:
            self._device = torch.device(
                f"cuda:{self.gpu}" if self.gpu is not None and torch.cuda.is_available() else "cpu"
            )
        return self._device

    @property
    def tokenizer(
        self
    ) -> PreTrainedTokenizerBase:
        if self._tokenizer is None:
            self._tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
                self.id, use_fast=True, model_max_length=int(1e30)
            )
        return self._tokenizer

    @property
    def model(
        self
    ) -> PreTrainedModel:
        if self._model is None:
            self._model = AutoModel.from_pretrained(
                self.id, output_hidden_states=True
            ).to(self.device)
            self._model.eval()
        return self._model

    def truncation_indices(
        self,
        target_subword_indices: list[bool]
    ) -> tuple[int, int]:

        max_tokens = 512
        n_target_subtokens = target_subword_indices.count(True)
        tokens_before = int(
            (max_tokens - n_target_subtokens) * self.truncation_tokens_before_target
        )
        tokens_after = max_tokens - tokens_before - n_target_subtokens

        # get index of the first target subword
        lindex_target = target_subword_indices.index(True)
        # get index of the last target subword
        rindex_target = lindex_target + n_target_subtokens + 1
        lindex = max(lindex_target - tokens_before, 0)
        rindex = rindex_target + tokens_after - 1
        return lindex, rindex

    def similarities(
        self,
        use_pairs: list[tuple[Use, Use]]
    ) -> list[float]:
        similarities = []
        for use_1, use_2 in use_pairs:
            enc_1 = self.encode(use_1)
            enc_2 = self.encode(use_2)
            print(enc_1.shape)
            similarities.append(self.similarity_metric(enc_1, enc_2))
        return similarities

    def tokenize(
        self,
        use: Use
    ) -> BatchEncoding:
        return self.tokenizer.encode_plus(
            text=use.context, return_tensors="pt", add_special_tokens=True
        ).to(self.device)

    def aggregate(
        self,
        embedding: np.ndarray
    ) -> np.ndarray:
        return self.layer_aggregation(
            self.subword_aggregation(embedding)
            .squeeze()
            .take(indices=self._layer_mask, axis=0)
        )

    def encode(
        self,
        use: Use
    ) -> np.ndarray:
<<<<<<< HEAD
        if self._vectors is None:
            self._vectors = {}
=======
>>>>>>> vector-cache

        is_new = False
        embedding = self.cache.retrieve(use)
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

            target_indices = [span.start >= use.indices[0] and span.end <= use.indices[1] if span is not None else False
                              for span in subword_spans]

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
                outputs = self.model(input_ids, torch.ones_like(input_ids))  # type: ignore

            embedding = (
                # stack the layers
                torch.stack(outputs[2], dim=0)
                # we don't vectorize in batches, so we can get rid of the batches dimension
                .squeeze(dim=1)  # swap the subwords and layers dimension
                .permute(1, 0, 2)  # select the target's subwords' embeddings
                [torch.tensor(target_indices), :, :]  # convert to numpy array
                .cpu().numpy()
            )

            log.info(f"Size of pre-subword-agregated tensor: {embedding.shape}")
            is_new = True

<<<<<<< HEAD
            self._vectors[use.identifier] = embedding

        embedding = self.aggregate(embedding)
        return embedding.cpu().numpy()
=======
        self.cache.add_use(use=use, embedding=embedding, new=is_new)
        return self.aggregate(embedding)
>>>>>>> vector-cache
