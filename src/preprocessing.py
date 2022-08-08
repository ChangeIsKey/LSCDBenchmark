import re
from typing import Tuple, List

import spacy
from pandas import Series
from thefuzz import process, fuzz

__all__ = ["toklem", "lemmatize", "tokenize"]

def char_indices(token_idx: int, tokens: List[str], target: str) -> Tuple[int, int]:
    char_idx = -1
    start, end = None, None
    for i, token in enumerate(tokens):
        if i == token_idx:
            # char_idx will be one index to the left of the target, so we need to add 1
            start = char_idx + 1  
            end = start + len(target)
            return start, end
        else:
            char_idx += len(token) + 1  # plus one space

def toklem(s: Series) -> Tuple[str, int, int]:
    tokens = s.context_tokenized.split()
    target = s.lemma.split("_")[0]
    start, end = char_indices(token_idx=s.indexes_target_token_tokenized, tokens=tokens, target=target)
    tokens[int(s.indexes_target_token_tokenized)] = target
    return " ".join(tokens), start, end


def lemmatize(s: Series) -> Tuple[str, int, int]:
    context_preprocessed = s.context_lemmatized
    tokens = context_preprocessed.split()
    
    # the spanish dataset has an index column for the lemmatized contexts, but all the others don't
    if "indexes_target_token_lemmatized" in s.columns:
        idx = s.indexes_target_token_lemmatized
    else:
        idx = s.indexes_target_token_tokenized

    target = tokens[idx]
    start, end = char_indices(token_idx=idx, tokens=tokens, target=target)
    return s.context_lemmatized, start, end


def tokenize(s: Series) -> Tuple[str, int, int]:
    tokens = s.context_tokenized.split()
    idx = s.indexes_target_token_tokenized
    start, end = char_indices(token_idx=idx, tokens=tokens, target=tokens[idx])
    return s.context_tokenized, start, end
