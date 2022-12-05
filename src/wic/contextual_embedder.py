import logging
import os
import uuid
from enum import Enum
from pathlib import Path
from tqdm import tqdm
from typing import Any, Callable, Iterable, Literal, Type, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from pandas import DataFrame
from pydantic import BaseModel, PositiveInt, PrivateAttr, conlist, Field
from transformers import (
    AutoModel,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers import logging as trans_logging

from src.use import Use, UseID
from src.utils import utils
from src.wic.model import WICModel

trans_logging.set_verbosity_error()

log = logging.getLogger(__name__)

T = TypeVar("T", torch.Tensor, np.ndarray)


class LayerAggregator(str, Enum):
    AVERAGE = "average"
    CONCAT = "concat"
    SUM = "sum"

    def __call__(self, tensor: torch.Tensor, layers: list[int]) -> torch.Tensor:
        tensor = tensor[:, torch.tensor(layers), :]
        match self:
            case self.AVERAGE:
                return tensor.mean(dim=1, keepdim=True)
            case self.SUM:
                return tensor.sum(dim=1, keepdim=True)
            case self.CONCAT:
                return tensor.ravel()
            case _:
                raise ValueError


class SubwordAggregator(str, Enum):
    AVERAGE = "average"
    FIRST = "first"
    LAST = "last"
    SUM = "sum"
    MAX = "max"
    MIN = "min"

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        match self:
            case self.AVERAGE:
                return tensor.mean(dim=0, keepdim=True)
            case self.SUM:
                return tensor.sum(dim=0, keepdim=True)
            case self.MAX:
                max_, _ = tensor.max(dim=0, keepdim=True)
                return max_
            case self.MIN:
                min_, _ = tensor.min(dim=0, keepdim=True)
                return min_
            case self.FIRST:
                return tensor[0]
            case self.LAST:
                return tensor[-1]
            case _:
                raise ValueError


TargetName: TypeAlias = str


class Cache(BaseModel):
    metadata: dict[Any, Any]
    _cache: dict[TargetName, dict[UseID, torch.Tensor]] = PrivateAttr(
        default_factory=dict
    )
    _targets_with_new_uses: set[TargetName] = PrivateAttr(default_factory=set)
    _index: DataFrame = PrivateAttr(default=None)
    _index_dir: Path = PrivateAttr(default=None)
    _index_path: Path = PrivateAttr(default=None)

    def __init__(self, **data):
        super().__init__(**data)
        self._index_dir = utils.path(os.getenv("CACHE_DIR") or ".bert")
        self._index_path = self._index_dir / "index.csv"

        try:
            self._index = pd.read_csv(
                filepath_or_buffer=self._index_path, engine="pyarrow"
            )
            self.clean()
        except FileNotFoundError:
            self._index = pd.json_normalize(self.metadata).assign(id=None, target=None)
            self._index = self._index.iloc[0:0]

    def add_use(self, use: Use, embedding: torch.Tensor) -> None:
        self._targets_with_new_uses.add(use.target)
        if not use.target in self._cache:
            self._cache[use.target] = {}
        self._cache[use.target][use.identifier] = embedding

    def retrieve(self, use: Use) -> torch.Tensor | None:
        if not use.target in self._cache:
            loaded = self.load(use.target)
            if loaded is None:
                return None
            self._cache[use.target] = loaded
        return self._cache[use.target].get(use.identifier)  # this can still be None

    def load(self, target: str) -> dict[UseID, torch.Tensor] | None:
        df = pd.json_normalize(self.metadata).assign(target=target).merge(self._index)
        assert len(df) < 2

        if df.empty:
            return None

        identifier = df.id.iloc[0]
        path = str(self._index_dir / f"{identifier}.pkl")
        return dict(torch.load(path))

    def _ids(self) -> set[UseID]:
        try:
            return set(self._index.id.tolist())
        except AttributeError:
            return set()

    def targets(self) -> set[TargetName]:
        return set(self._cache.keys())

    def clean(self):
        self._index.drop_duplicates(
            subset=[col for col in self._index.columns.tolist() if col != "id"],
            inplace=True,
            keep="last",
        )
        valid_ids = self._ids()
        for file in self._index_dir.iterdir():
            if file.name != "index.csv" and file.stem not in valid_ids:
                file.unlink()
        self._index.to_csv(path_or_buf=self._index_path, index=False)

    def persist(self, target: str) -> None:
        while True:
            identifier = str(uuid.uuid4())
            if identifier not in self._ids():
                self._index_dir.mkdir(exist_ok=True, parents=True)
                self._index = pd.concat(
                    [
                        self._index,
                        pd.json_normalize(self.metadata).assign(
                            id=identifier, target=target
                        ),
                    ],
                    ignore_index=True,
                )
                self._index.to_csv(path_or_buf=self._index_path, index=False)
                log.info("Logged record of new embedding file")

                with open(file=self._index_dir / f"{identifier}.pkl", mode="wb") as f:
                    torch.save(self._cache[target], f)
                    log.info(f"Saved embeddings to disk as {identifier}.pkl")

                self._targets_with_new_uses.remove(target)

                break

    def has_new_uses(self, target: str) -> bool:
        return target in self._targets_with_new_uses


class ContextualEmbedder(WICModel):
    layers: conlist(item_type=PositiveInt, unique_items=True)  # type: ignore
    truncation_tokens_before_target: float
    similarity_metric: Callable[..., float]
    normalization: None | Callable[[torch.Tensor], torch.Tensor]
    ckpt: str
    cache: Cache | None = Field(...)
    gpu: int | None = Field(...)
    layer_aggregator: LayerAggregator = Field(alias="layer_aggregation")
    subword_aggregator: SubwordAggregator = Field(alias="subword_aggregation")

    _device: torch.device = PrivateAttr(default=None)
    _tokenizer: PreTrainedTokenizerBase = PrivateAttr(default=None)
    _model: PreTrainedModel = PrivateAttr(default=None)

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cache is not None:
            targets = self.cache.targets()
            for target in targets:
                if self.cache.has_new_uses(target):
                    self.cache.persist(target)

    @property
    def device(self) -> torch.device:
        if self._device is None:
            self._device = torch.device(
                f"cuda:{self.gpu}"
                if self.gpu is not None and torch.cuda.is_available()
                else "cpu"
            )
        return self._device

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        if self._tokenizer is None:
            self._tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
                self.ckpt, use_fast=True, model_max_length=int(1e30)
            )
        return self._tokenizer

    @property
    def model(self) -> PreTrainedModel:
        if self._model is None:
            self._model = AutoModel.from_pretrained(
                self.ckpt, output_hidden_states=True
            ).to(self.device)
            self._model.eval()
        return self._model

    def truncation_indices(self, target_subword_indices: list[bool]) -> tuple[int, int]:

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

    def predict(
        self, use_pairs: Iterable[tuple[Use, Use]], type: Type[T] = np.ndarray
    ) -> list[float]:
        predictions = []
        with self:
            for use_pair in use_pairs:
                id_pair = (use_pair[0].identifier, use_pair[1].identifier)
                if id_pair in self.predictions:
                    # this will be true when this use pair has been previously
                    # processed in predict_all
                    predictions.append(self.predictions[id_pair])
                    continue
                enc_1 = self.encode(use_pair[0], type=type)
                enc_2 = self.encode(use_pair[1], type=type)
                predictions.append(self.similarity_metric(enc_1, enc_2))

        return predictions

    def tokenize(self, use: Use) -> BatchEncoding:
        return self.tokenizer.encode_plus(
            text=use.context, return_tensors="pt", add_special_tokens=True
        ).to(self.device)

    def aggregate(self, tensor: torch.Tensor, layers: list[int]) -> torch.Tensor:
        tensor = self.subword_aggregator(tensor)  # (1, layers, embedding)
        tensor = self.layer_aggregator(tensor, layers)  # (1, 1, embedding)
        return tensor.squeeze()

    def encode_all(self, uses: list[Use], type: Type[T]) -> list[T]:
        return [self.encode(use, type=type) for use in uses]

    def encode(self, use: Use, type: Type[T] = np.ndarray) -> T:
        embedding = None if self.cache is None else self.cache.retrieve(use)
        if embedding is None:
            log.info(f"PROCESSING USE `{use.identifier}`: {use.context}")
            log.info(f"Target character indices: {use.indices}")
            log.info(
                f"Context slice corresponding to target indices: {use.context[use.indices[0]:use.indices[1]+1]}"
            )

            encoding = self.tokenize(use)
            input_ids = encoding["input_ids"].to(self.device)  # type: ignore
            tokens = encoding.tokens()
            subword_spans = [encoding.token_to_chars(i) for i in range(len(tokens))]

            log.info(f"Extracted {len(tokens)} tokens: {tokens}")

            subwords_bool_mask = [
                span.start >= use.indices[0] and span.end <= (use.indices[1] + 1)
                if span is not None
                else False
                for span in subword_spans
            ]

            # truncate input if the model cannot handle it
            if len(tokens) > 512:
                lindex, rindex = self.truncation_indices(subwords_bool_mask)
                tokens = tokens[lindex:rindex]
                input_ids = input_ids[:, lindex:rindex]
                subwords_bool_mask = subwords_bool_mask[lindex:rindex]

                log.info(f"Truncated input")
                log.info(f"New tokens: {tokens}")

            extracted_subwords = [
                tokens[i] for i, value in enumerate(subwords_bool_mask) if value
            ]
            log.info(f"Selected subwords: {extracted_subwords}")
            log.info(f"Size of input_ids: {input_ids.size()}")

            with torch.no_grad():
                outputs = self.model(input_ids, torch.ones_like(input_ids))  # type: ignore

            embedding = (
                torch.stack(outputs[2], dim=0)  # (layer, batch, subword, embedding)
                .squeeze(dim=1)  # (layer, subword, embedding)
                .permute(1, 0, 2)[  # (subword, layer, embedding)
                    torch.tensor(subwords_bool_mask), :, :
                ]
            )

            log.info(f"Size of pre-subword-agregated tensor: {embedding.shape}")

            if self.cache is not None:
                self.cache.add_use(use=use, embedding=embedding)

        embedding = self.aggregate(embedding, layers=self.layers)
        if self.normalization is not None:
            embedding = self.normalization(embedding)

        if type == np.ndarray:
            return embedding.cpu().numpy()
        else:
            return embedding  # type: ignore
