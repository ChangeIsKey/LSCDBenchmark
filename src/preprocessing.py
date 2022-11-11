import re
from abc import (
    ABC,
    abstractmethod,
)
from typing import Any

from pandas import Series
from pydantic import BaseModel
from logging import getLogger

log = getLogger(__name__)




class ContextPreprocessor(BaseModel, ABC):
    spelling_normalization: dict[str, str] | None

    def __init__(self, **data) -> None:
        super().__init__(**data)

        if self.spelling_normalization is not None:
            self.spelling_normalization = {
                k.replace("_", " "): v for k, v in self.spelling_normalization.items()
            }
        else:
            self.spelling_normalization = {}

    @staticmethod
    def start_char_index(token_index: int, tokens: list[str]) -> int:
        char_idx = -1
        for i, token in enumerate(tokens):
            if i == token_index:
                # char_idx will be one index to the left of the target, so we need to add 1
                start = char_idx + 1
                return start
            char_idx += len(token) + 1  # plus one space
        raise ValueError

    def normalize_spelling(self, context: str, start: int) -> tuple[str, int]:
        assert self.spelling_normalization is not None

        new_target_start = start
        new_context = context

        for key, replacement in self.spelling_normalization.items():
            spans = list(re.finditer(pattern=key, string=context))
            for span in spans:
                new_context = new_context.replace(key, replacement, 1)
                if span.end() < start:
                    new_target_start -= len(key) - len(replacement)

        return new_context, new_target_start

    @abstractmethod
    def fields_from_series(self, s: Series) -> dict[str, Any]:
        pass

    @abstractmethod
    def preprocess(self, *args, **kwargs) -> tuple[str, int, int]:
        pass

    def __call__(self, s: Series) -> Series:
        fields = self.fields_from_series(s)
        context, start, end = self.preprocess(**fields)
        return Series(
            {
                "context_preprocessed": context,
                "target_index_begin": start,
                "target_index_end": end,
            }
        )


class Toklem(ContextPreprocessor):
    def fields_from_series(self, s: Series) -> dict[str, str | int]:
        return {
            "context": s.context_tokenized,
            "lemma": s.lemma.split("_")[0],
            "index": int(s.indexes_target_token_tokenized),
        }

    def preprocess(self, context: str, index: int, lemma: str) -> tuple[str, int, int]:
        # extract tokens (in the DWUG datasets, each token is separated by space)
        # so no extra methods are needed
        tokens = context.split()
        tokens[index] = lemma
        # get the initial character indices, before spelling normalization is applied
        start = self.start_char_index(token_index=index, tokens=tokens)

        # if some spelling normalization table has been specified, apply it to the context
        # and recalculate the character indices (number of tokens may change)
        if self.spelling_normalization is not None:
            new_context, start = self.normalize_spelling(context, start)
            tokens = new_context.split()

        # adjust the end character index for possible changes in target length
        tokens[index] = lemma
        end = start + len(lemma) - 1

        return " ".join(tokens), start, end


class Raw(ContextPreprocessor):
    def fields_from_series(self, s: Series) -> dict[str, str | int]:
        start, end = tuple(map(int, s.indexes_target_token.split(":")))
        return {"context": s.context, "start": start, "end": end}

    def preprocess(self, context: str, start: int, end: int) -> tuple[str, int, int]:
        return context, start, end


class Lemmatize(ContextPreprocessor):
    def fields_from_series(self, s: Series) -> dict[str, str | int]:
        return {
            "context": s.context_lemmatized,
            # the spanish dataset has an index column for the lemmatized contexts, but all the others don't
            "index": s.get(
                key="indexes_target_token_lemmatized",
                default=s.indexes_target_token_tokenized,
            ),  # type: ignore
        }

    def preprocess(self, context: str, index: int) -> tuple[str, int, int]:
        tokens = context.split()
        start = self.start_char_index(token_index=index, tokens=tokens)
        if self.spelling_normalization is not None:
            context, start = self.normalize_spelling(context, start)

        end = start + len(tokens[index]) - 1
        return context, start, end


class Tokenize(ContextPreprocessor):
    def fields_from_series(self, s: Series) -> dict[str, str | int]:
        return {
            "context": s.context_tokenized,
            "index": int(s.indexes_target_token_tokenized),
        }

    def preprocess(self, context: str, index: int) -> tuple[str, int, int]:
        tokens = context.split()
        start = self.start_char_index(token_index=index, tokens=tokens)
        if self.spelling_normalization is not None:
            context, start = self.normalize_spelling(context, start)

        end = start + len(tokens[index]) - 1
        return context, start, end


class Normalize(ContextPreprocessor):
    default: str

    def fields_from_series(self, s: Series) -> dict[str, str | int]:
        context = s.get("context_normalized")
        if context is None:
            log.warn(f"(lemma={s.lemma}, use={s.identifier}) does not contain a pre-normalized context, {self.default} will be used")
        context = s[self.default]
        return {
            "context": context,
            "index": int(s.indexes_target_token_tokenized),
        }

    def preprocess(self, context: str, index: int) -> tuple[str, int, int]:
        tokens = context.split()
        start = self.start_char_index(token_index=index, tokens=tokens)

        if self.spelling_normalization is not None:
            context, start = self.normalize_spelling(context, start)

        try:
            end = context.index(" ", start) - 1
        except ValueError:
            end = len(context) - 1

        return context, start, end
