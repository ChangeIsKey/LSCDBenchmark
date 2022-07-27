import numpy as np
import torch
from dataclasses import dataclass

from transformers import AutoTokenizer, AutoModel, logging
from typing import List, Tuple, Protocol

from src.config import ModelConfig

logging.set_verbosity_error()


@dataclass
class Vectorizer:
    config: ModelConfig

    def __post_init__(self):
        self.device = "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.name)
        self.model = AutoModel.from_pretrained(self.config.name, output_hidden_states=True)
        print(type(self.tokenizer))
        print(type(self.model))

        # Find a free GPU
        for i in range(torch.cuda.device_count()):
            self.device = f"cuda:{i}"
            try:
                self.model = self.model.to(self.device)
            except RuntimeError:
                continue
            else:
                break
            
    def __call__(self, contexts: List[str], target_indices: List[Tuple[int, int]]) -> torch.Tensor:
        # TODO investigate how to do this in batches
        # TODO don't change between tensors, numpy arrays and lists
        target_vectors = []

        for context, (target_begin, target_end) in zip(contexts, target_indices):
            encoded = self.tokenizer(context,
                                     return_tensors='pt',
                                     truncation=True,
                                     add_special_tokens=True,
                                     return_offsets_mapping=True).to(self.device)
            input_ids = encoded["input_ids"].to(self.device)
            tokens = self.tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
            subword_indices = encoded["offset_mapping"][0].to(self.device)

            target_subword_indices = [sub_start >= target_begin and sub_end <= target_end 
                                      for sub_start, sub_end in subword_indices]

            segments_ids = torch.ones(1, len(tokens)).to(self.device)  # TODO nikolai: should these be zeros?

            self.model.eval()
            with torch.no_grad():
                outputs = self.model(input_ids, segments_ids)
                target_token_vectors = outputs[-1][0][0][target_subword_indices]

                if self.config.subword_aggregation == "average":
                    target_vector = torch.mean(target_token_vectors, dim=0)
                elif self.config.subword_aggregation == "first":
                    target_vector = target_token_vectors[0]
                else:
                    raise NotImplementedError

                target_vectors.append(target_vector)

        return torch.stack(target_vectors).to(self.device)