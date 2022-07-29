from dataclasses import dataclass
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModel, logging

from src.config import ModelConfig

logging.set_verbosity_error()


@dataclass
class Vectorizer:
    config: ModelConfig

    def __post_init__(self):
        self.device = "cuda:1"
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.name)
        self.model = AutoModel.from_pretrained(self.config.name, output_hidden_states=True).to(self.device)

    def __call__(
        self, contexts: List[str], target_indices: List[Tuple[int, int]]
    ) -> torch.Tensor:
        target_vectors = []

        encoded = self.tokenizer(
            contexts,
            return_tensors="pt",
            truncation=True,
            add_special_tokens=True,
            return_offsets_mapping=True,
            padding=True,
        ).to(self.device)

        input_ids = encoded["input_ids"].to(self.device)
        segments_ids = torch.ones_like(input_ids).to(self.device)  

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids, segments_ids)
            hidden_states = torch.stack(outputs[2], dim=0)

        for i, (target_begin, target_end) in enumerate(target_indices):
            subword_indices = encoded["offset_mapping"][i].to(self.device)

            target_subword_indices = [
                sub_start >= target_begin and sub_end <= target_end
                for sub_start, sub_end in subword_indices
            ]

            layers = torch.mean(hidden_states[self.config.layers], dim=0)
            target_token_vectors = layers[i][target_subword_indices]
            vec = self.config.subword_aggregation(vectors=target_token_vectors)
            target_vectors.append(vec)

        return torch.stack(target_vectors).to(self.device)
