import re
from typing import Tuple

import spacy
from pandas import Series
from thefuzz import process, fuzz

__all__ = ["toklem", "lemmatize", "tokenize"]


def toklem(s: Series) -> Tuple[str, int, int]:
    tokens = s.context_tokenized.split()
    target_token_idx = s.indexes_target_token_tokenized
    lemma = s.lemma.split("_")[0]

    char_idx = -1
    start, end = None, None
    for i, token in enumerate(tokens):
        if i == target_token_idx:
            # char_idx will be one index to the left of the target, so we need to add 1
            start = char_idx + 1  
            end = start + len(lemma)
            break
        else:
            char_idx += len(token) + 1  # plus one space

    tokens[int(s.indexes_target_token_tokenized)] = lemma
    context_preprocessed = " ".join(tokens)

    return context_preprocessed, start, end


def lemmatize(s: Series) -> Tuple[str, int, int]:
    context_preprocessed = s.context_lemmatized
    match = re.search(s.lemma, context_preprocessed)
    return context_preprocessed, match.start(), match.end()


def tokenize(s: Series) -> Tuple[str, int, int]:
    tokens = s.context_tokenized.split()
    target = tokens[int(s.indexes_target_token_tokenized)]
    context_preprocessed = " ".join(tokens)
    match = re.search(target, context_preprocessed)
    return context_preprocessed, match.start(), match.end()
