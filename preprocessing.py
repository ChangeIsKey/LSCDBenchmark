import warnings

from pandas import Series

__all__ = ["toklem", "lemmatize", "tokenize"]


def toklem(s: Series, cached: bool = True) -> str:
    if cached:
        i = int(s.indexes_target_token_tokenized)
        tokenized = s.context_tokenized.split()
    else:
        raise NotImplementedError
    tokenized[i] = s.lemma
    return " ".join(tokenized)


def lemmatize(s: Series, cached: bool = True) -> str:
    if cached:
        col = "context_lemmatized"
        if col not in s:
            warnings.warn("Precomputed lemmatization is not available. Please use 'cached=False' to lemmatize the text")
            return ""
        return s[col]
    raise NotImplementedError


def tokenize(s: Series, cached: bool = True) -> str:
    if cached:
        col = "context_tokenized"
        if col not in s:
            warnings.warn("Precomputed lemmatization is not available. Please use 'cached=False' to lemmatize the text")
            return ""
        return s[col]
    raise NotImplementedError


