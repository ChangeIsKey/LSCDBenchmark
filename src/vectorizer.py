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
        target_vectors = []

        encoded = self.tokenizer(contexts,
                                 return_tensors='pt',
                                 truncation=True,
                                 add_special_tokens=True,
                                 return_offsets_mapping=True,
                                 padding=True).to(self.device)
        input_ids = encoded["input_ids"].to(self.device)
        segments_ids = torch.ones_like(input_ids).to(self.device)  # TODO nikolai: should these be zeros?
        # tokens = self.tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids, segments_ids)

        for i, (context, (target_begin, target_end)) in enumerate(zip(contexts, target_indices)):
            subword_indices = encoded["offset_mapping"][i].to(self.device)

            target_subword_indices = [sub_start >= target_begin and sub_end <= target_end
                                      for sub_start, sub_end in subword_indices]
            target_token_vectors = outputs[2][-1][i][target_subword_indices]
            target_vectors.append(self.config.subword_aggregation(vectors=target_token_vectors))

        return torch.stack(target_vectors).to(self.device)
