from pandas import Series

__all__ = ["toklem", "lemmatize", "tokenize"]


def toklem(s: Series) -> str:
    i = int(s.indexes_target_token_tokenized)
    tokenized = s.context_tokenized.split()
    tokenized[i] = s.lemma
    return " ".join(tokenized)


def lemmatize(s: Series) -> str:
    return s.context_lemmatized


def tokenize(s: Series) -> str:
    return s.context_tokenized


