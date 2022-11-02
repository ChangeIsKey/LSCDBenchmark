import re
from abc import (
    ABC,
    abstractmethod,
)
from typing import Any

from pandas import Series
from pydantic import BaseModel


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
    def character_indices(
        token_index: int, tokens: list[str], target: str
    ) -> tuple[int, int]:
        char_idx = -1
        for i, token in enumerate(tokens):
            if i == token_index:
                # char_idx will be one index to the left of the target, so we need to add 1
                start = char_idx + 1
                end = start + len(target)
                return start, end
            char_idx += len(token) + 1  # plus one space
        raise ValueError

    def normalize_spelling(
        self, context: str, start: int, end: int
    ) -> tuple[str, int, int]:
        assert self.spelling_normalization is not None

        new_target_start, new_target_end = start, end
        new_context = context

        for key, replacement in self.spelling_normalization.items():
            spans = re.finditer(pattern=key, string=context)
            original_n_blanks = key.count(" ") + key.count("\t")
            later_n_blanks = replacement.count(" ") + replacement.count("\t")
            for span in spans:
                new_context = new_context.replace(key, replacement, 1)
                if span.end() < start:
                    new_target_start -= original_n_blanks - later_n_blanks
                    new_target_end -= original_n_blanks - later_n_blanks

        return new_context, new_target_start, new_target_end

    @staticmethod
    @abstractmethod
    def fields_from_series(s: Series) -> dict[str, Any]:
        pass

    @abstractmethod
    def preprocess(self, *args, **kwargs) -> tuple[str, int, int]:
        pass

    def __call__(self, s: Series) -> Series:
        fields = self.fields_from_series(s)
        context, start, end = self.preprocess(**fields.__dict__)
        return Series(
            {
                "context_preprocessed": context,
                "target_index_Begin": start,
                "target_index_end": end,
            }
        )


class Toklem(ContextPreprocessor):
    @staticmethod
    def fields_from_series(s: Series) -> dict[str, str | int]:
        return {
            "context": s.context_tokenized,
            "lemma": s.lemma.split("_")[0],
            "index": int(s.indexes_target_token_tokenized),
        }

    def preprocess(self, context: str, index: int, lemma: str) -> tuple[str, int, int]:
        # extract tokens (in the DWUG datasets, each token is separated by space)
        # so no extra methods are needed
        tokens = context.split()
        # get the initial character indices, before spelling normalization is applied
        start, end = self.character_indices(
            token_index=index, tokens=tokens, target=lemma
        )

        # if some spelling normalization table has been specified, apply it to the context
        # and recalculate the character indices (number of tokens may change)
        if self.spelling_normalization is not None:
            new_context, start, end = self.normalize_spelling(context, start, end)
            tokens = new_context.split()

        # adjust the end character index for possible changes in target length
        end -= abs(len(lemma) - len(tokens[index]))
        # replace target with lemma
        tokens[index] = lemma

        return " ".join(tokens), start, end


class KeepIntact(ContextPreprocessor):
    @staticmethod
    def fields_from_series(s: Series) -> dict[str, str | int]:
        start, end = tuple(map(int, s.indexes_target_token.split(":")))
        return {"context": s.context, "start": start, "end": end}

    def preprocess(self, context: str, start: int, end: int) -> tuple[str, int, int]:
        return context, start, end


class Lemmatize(ContextPreprocessor):
    @staticmethod
    def fields_from_series(s: Series) -> dict[str, str | int]:
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
        start, end = self.character_indices(
            token_index=index, tokens=tokens, target=tokens[index]
        )
        if self.spelling_normalization is not None:
            context, start, end = self.normalize_spelling(context, start, end)

        return context, start, end


class Tokenize(ContextPreprocessor):
    @staticmethod
    def fields_from_series(s: Series) -> dict[str, str | int]:
        return {
            "context": s.context_tokenized,
            "index": int(s.indexes_target_token_tokenized),
        }

    def preprocess(self, context: str, index: int) -> tuple[str, int, int]:
        tokens = context.split()
        start, end = self.character_indices(
            token_index=index, tokens=tokens, target=tokens[index]
        )
        if self.spelling_normalization is not None:
            context, start, end = self.normalize_spelling(context, start, end)

        return context, start, end
