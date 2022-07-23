import numpy as np
import torch
from dataclasses import dataclass

from torch import Tensor
from transformers import BertTokenizerFast, BertModel, AutoTokenizer, AutoModelForMaskedLM, logging, BatchEncoding
from typing import List, Tuple, Protocol

from src.config import ModelConfig

logging.set_verbosity_error()


class Vectorizer(Protocol):
    def __call__(self, contexts: List[str], target_indices: List[Tuple[int, int]]) -> Tensor:
        ...


@dataclass
class Bert:
    config: ModelConfig

    def __post_init__(self):
        self.tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(self.config.model)
        self.model = BertModel.from_pretrained(self.config.model, output_hidden_states=True)

    def __call__(self, contexts: List[str], target_indices: List[Tuple[int, int]]) -> Tensor:
        # TODO investigate how to do this in batches
        # TODO don't change between tensors, numpy arrays and lists
        target_vectors = []
        for context, (target_begin, target_end) in zip(contexts, target_indices):
            encoded = self.tokenizer(context,
                                     return_tensors='pt',
                                     truncation=True,
                                     add_special_tokens=True,
                                     return_offsets_mapping=True)

            input_ids = encoded["input_ids"]
            tokens = self.tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
            subword_indices = encoded["offset_mapping"][0]

            target_subword_indices = [i for i, (sub_start, sub_end) in enumerate(subword_indices)
                                      if sub_start >= target_begin and sub_end <= target_end]

            segments_ids = torch.ones(1, len(tokens))  # TODO nikolai: should these be zeros?

            self.model.eval()
            with torch.no_grad():
                outputs = self.model(input_ids, segments_ids)
                target_token_vectors = [outputs[-1][0][0][i] for i in target_subword_indices]

                if self.config.subword_aggregation == "average":
                    target_vector = torch.mean(torch.stack(target_token_vectors), dim=0)
                elif self.config.subword_aggregation == "first":
                    target_vector = target_token_vectors[0]
                else:
                    raise NotImplementedError

                target_vectors.append(target_vector)

        return torch.stack(target_vectors)


@dataclass
class XLMR:
    config: ModelConfig

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model)
        self.model = AutoModelForMaskedLM.from_pretrained(self.config.model)

    def __call__(self, contexts: List[str], target_indices: List[Tuple[int, int]]) -> Tensor:
        target_vectors = []
        for context, (target_begin, target_end) in zip(contexts, target_indices):
            encoded = self.tokenizer(context,
                                     return_tensors='pt',
                                     truncation=True,
                                     add_special_tokens=True,
                                     return_offsets_mapping=True)

            input_ids = encoded["input_ids"]
            tokens = self.tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
            subword_indices = encoded["offset_mapping"][0]

            target_subword_indices = [i for i, (sub_start, sub_end) in enumerate(subword_indices)
                                      if sub_start >= target_begin and sub_end <= target_end]

            segments_ids = torch.ones(1, len(tokens))  # TODO nikolai: should these be zeros?

            self.model.eval()
            with torch.no_grad():
                outputs = self.model(input_ids, segments_ids)
                target_token_vectors = [outputs[-1][0][0][i] for i in target_subword_indices]

                if self.config.subword_aggregation == "average":
                    target_vector = torch.mean(torch.stack(target_token_vectors), dim=0)
                elif self.config.subword_aggregation == "first":
                    target_vector = target_token_vectors[0]
                else:
                    raise NotImplementedError

                target_vectors.append(target_vector)

        return torch.stack(target_vectors)
