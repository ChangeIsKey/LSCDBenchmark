import re
from typing import Tuple

import spacy
from thefuzz import process, fuzz

from pandas import Series

__all__ = ["toklem", "lemmatize", "tokenize"]


def toklem(s: Series, cached: bool = True, nlp: spacy.Language = None) -> Tuple[str, int, int]:
    if cached:
        tokens = s.context_tokenized.split()
        tokens[int(s.indexes_target_token_tokenized)] = s.lemma
    else:
        tokens = [token.text for token in nlp(s.context)]
        target, _ = process.extractOne(s.lemma, tokens, scorer=fuzz.token_sort_ratio)
        tokens[tokens.index(target)] = s.lemma

    context_preprocessed = " ".join(tokens)
    match = re.search(s.lemma, context_preprocessed)
    return context_preprocessed, match.start(), match.end()


def lemmatize(s: Series, cached: bool = True, nlp: spacy.Language = None) -> Tuple[str, int, int]:
    if cached:
        context_preprocessed = s.context_lemmatized
    else:
        context_preprocessed = " ".join([token.lemma_ for token in nlp(s.context)])
    match = re.search(s.lemma, context_preprocessed)
    return context_preprocessed, match.start(), match.end()


def tokenize(s: Series, cached: bool = True, nlp: spacy.Language = None) -> Tuple[str, int, int]:
    if cached:
        tokens = s.context_tokenized.split()
        target = tokens[int(s.indexes_target_token_tokenized)]
    else:
        tokens = [token.text for token in nlp(s.context)]
        target, _ = process.extractOne(s.lemma, tokens, scorer=fuzz.token_sort_ratio)
    context_preprocessed = " ".join(tokens)
    match = re.search(target, context_preprocessed)
    return context_preprocessed, match.start(), match.end()
