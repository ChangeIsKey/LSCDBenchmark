import logging
from enum import Enum
from typing import Callable

from collections import defaultdict
import numpy as np
import torch
from pydantic import (
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

from src.use import (
    Use,
    UseID,
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
        layers: torch.Tensor
    ) -> torch.Tensor:
        match self:
            case self.AVERAGE:
                return torch.mean(layers, dim=0)
            case self.SUM:
                return torch.sum(layers, dim=0)
            case self.CONCAT:
                return torch.ravel(layers)
            case _:
                raise ValueError


class SubwordAggregator(str, Enum):
    AVERAGE = "average"
    FIRST = "first"
    LAST = "last"
    SUM = "sum"

    def __call__(
        self,
        vectors: torch.Tensor
    ) -> torch.Tensor:
        match self:
            case self.AVERAGE:
                return torch.mean(vectors, dim=0, keepdim=True)
            case self.SUM:
                return torch.sum(vectors, dim=0, keepdim=True)
            case self.FIRST:
                return vectors[0]
            case self.LAST:
                return vectors[-1]
            case _:
                raise ValueError


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
    _vectors: dict[str, torch.Tensor] = PrivateAttr(default=None)
    _layer_mask: torch.Tensor = PrivateAttr(default=None)

    def __init__(
        self,
        **data
    ) -> None:
        super().__init__(**data)
        self._layer_mask = torch.tensor(self.layers, dtype=torch.int32).to(self.device)

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
        for use_1, use_2 in tqdm(
                use_pairs, desc="Calculating use-pair distances", leave=False
        ):
            enc_1 = self.encode(use_1)
            enc_2 = self.encode(use_2)
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
        embedding: torch.Tensor
    ) -> torch.Tensor:
        return self.layer_aggregation(
            self.subword_aggregation(embedding).squeeze().index_select(index=self._layer_mask, dim=0)
        )

    def encode(
        self,
        use: Use
    ) -> np.ndarray:
        if self._vectors is None:
            self._vectors = defaultdict(dict)

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

            embedding = (  # stack the layers
                torch.stack(
                    outputs[2], dim=0
                )  # we don't vectorize in batches, so we can get rid of the batches dimension
                .squeeze(dim=1)  # swap the subwords and layers dimension
                .permute(1, 0, 2)  # select the target's subwords' embeddings
                [torch.tensor(target_indices), :, :]  # convert to numpy array
            )

            log.info(f"Size of pre-subword-agregated tensor: {embedding.shape}")

            self._vectors[use.identifier] = embedding
            embedding = self.aggregate(embedding)

        return embedding.cpu().numpy()