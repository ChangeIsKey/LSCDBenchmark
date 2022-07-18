import warnings
from typing import Tuple
from thefuzz import process, fuzz

from pandas import Series

__all__ = ["toklem", "lemmatize", "tokenize"]


def toklem(s: Series, cached: bool = True) -> str:
    if cached:
        i = int(s.indexes_target_token_tokenized)
        tokenized = s.context_tokenized.split()
        tokenized[i] = s.lemma
        return " ".join(tokenized)
    else:
        raise NotImplementedError


def lemmatize(s: Series, cached: bool = True) -> str:
    if cached:
        # TODO return indices of target word
        return s["context_lemmatized"]
    raise NotImplementedError


def tokenize(s: Series, cached: bool = True) -> str:
    if cached:
        col = "context_tokenized"
        if col not in s:
            warnings.warn("Precomputed lemmatization is not available. Please use 'cached=False' to lemmatize the text")
            return ""
        return s[col]
    raise NotImplementedError


def keep_intact(s: Series) -> str:
    return s.context