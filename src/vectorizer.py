from dataclasses import dataclass
from typing import List, Tuple

import uuid
import torch
import pandas as pd
import numpy as np
from pandas import DataFrame, Series

from transformers import AutoTokenizer, AutoModel, logging
from pathlib import Path

from src.config import Config, ID

logging.set_verbosity_error()


@dataclass
class Vectorizer:
    config: Config

    def __post_init__(self):
        self.device = torch.device("cpu" if self.config.model.gpu is None else f"cuda:{self.config.model.gpu}")
        self._tokenizer = None
        self._model = None
        self._index = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model.name)
        return self._tokenizer


    @property
    def model(self):
        if self._model is None:
            self._model = AutoModel.from_pretrained(self.config.model.name, output_hidden_states=True).to(self.device)
            self._model.eval()
        return self._model
    

    @property
    def index(self):
        path = Path("vectors").joinpath("index.csv")
        if path.exists():
            self._index = pd.read_csv(path, sep="\t")
        else:
            self._index = DataFrame(columns=["model", "target", "id", "preprocessing", "dataset"])
        return self._index

    def row(self, target_name: str, id: str):
        return DataFrame([{
            "model": self.config.model.name, 
            "target": target_name, 
            "id": id, 
            "preprocessing": str(self.config.dataset.preprocessing.method_name),
            "dataset": self.config.dataset.name
        }])


    @index.setter
    def index(self, new: DataFrame):
        path = Path("vectors").joinpath("index.csv")
        self._index = new
        self._index.to_csv(path, sep="\t", index=False)

    def __call__(
        self, contexts: List[str], target_indices: List[Tuple[int, int]], ids: List[ID], target_name: str
    ) -> torch.Tensor:
        hidden_states = {}
        subword_indices = {}
        cache = Path("vectors")
        cache.mkdir(exist_ok=True)

        row = self.index[
            (self.index.model == self.config.model.name) & 
            (self.index.target == target_name) & 
            (self.index.preprocessing == str(self.config.dataset.preprocessing.method_name)) & 
            (self.index.dataset == self.config.dataset.name)
        ]

        if not row.empty:
            id = row["id"].iloc[0]
            hidden_states = np.load(cache.joinpath(f"{id}.npz"), mmap_mode="r")
            subword_indices = np.load(cache.joinpath(f"{id}-offset-mapping.npz"), mmap_mode="r")
        else:
            for context, context_id in zip(contexts, ids):
                encoded = self.tokenizer(
                    context,
                    return_tensors="pt",
                    truncation=True,
                    add_special_tokens=True,
                    return_offsets_mapping=True
                ).to(self.device)




                # TODO fix segment ids
                input_ids = encoded["input_ids"].to(self.device)
                segment_ids = torch.ones_like(input_ids).to(self.device)  

                self.model.eval()
                with torch.no_grad():
                    outputs = self.model(input_ids, segment_ids)
                    hidden_states[context_id] = torch.stack(outputs[2], dim=0).cpu().numpy()
                    subword_indices[context_id] = encoded["offset_mapping"].squeeze(0).cpu().numpy()

            id = uuid.uuid4()
            self.index = pd.concat([self.index, self.row(target_name=target_name, id=id)], ignore_index=True)
            with open(file=cache.joinpath(f"{id}.npz"), mode="wb") as f_hidden_states:
                np.savez(f_hidden_states, **hidden_states)
            with open(file=cache.joinpath(f"{id}-offset-mapping.npz"), mode="wb") as f_subword_indices:
                np.savez(f_subword_indices, **subword_indices)

        target_vectors = []
        for context_id, (target_begin, target_end) in zip(ids, target_indices):
            layers = self.config.model.layer_aggregation(hidden_states[context_id][self.config.model.layers])

            target_subword_indices = [
                (sub_start >= target_begin and sub_end <= target_end)
                for sub_start, sub_end in subword_indices[context_id]
            ]

            embeddings = layers.squeeze(0)[target_subword_indices]
            vec = self.config.model.subword_aggregation(vectors=embeddings)
            target_vectors.append(vec)
        
        return target_vectors
