from abc import (
    ABC,
    abstractmethod,
)

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
	def character_indices(token_index: int, tokens: list[str], target: str) -> tuple[int, int]:
		char_idx = -1
		for i, token in enumerate(tokens):
			if i == token_index:
				# char_idx will be one index to the left of the target, so we need to add 1
				start = char_idx + 1
				end = start + len(target)
				return start, end
			char_idx += len(token) + 1  # plus one space
		raise ValueError

    def normalize_spelling(self, context: str) -> str:
        if self.spelling_normalization is not None:
            for key, replacement in self.spelling_normalization.items():
                context = context.replace(key, replacement)
        return context

    @abstractmethod
    def preprocess_context(self, s: Series) -> tuple[str, int, int]:
        raise NotImplementedError

    def __call__(self, s: Series) -> Series:
        context, start, end = self.preprocess_context(s)
        return Series(
            {
                "context_preprocessed": context,
                "target_index_begin": start,
                "target_index_end": end,
            }
        )


class Toklem(ContextPreprocessor):
    def preprocess_context(self, s: Series) -> tuple[str, int, int]:
        tokens = self.normalize_spelling(s.context_tokenized).split()
        target = s.lemma.split("_")[0]
        start, end = self.character_indices(
            token_index=s.indexes_target_token_tokenized, tokens=tokens, target=target
        )
        tokens[int(s.indexes_target_token_tokenized)] = target
        return " ".join(tokens), start, end


class KeepIntact(ContextPreprocessor):
    def preprocess_context(self, s: Series) -> tuple[str, int, int]:
        start, end = tuple(map(int, s.indexes_target_token.split(":")))
        return s.context, start, end


class Lemmatize(ContextPreprocessor):
    def preprocess_context(self, s: Series) -> tuple[str, int, int]:
        context_preprocessed = self.normalize_spelling(s.context_lemmatized)
        tokens = context_preprocessed.split()

        # the spanish dataset has an index column for the lemmatized contexts, but all the others don't
        idx: int = s.get(key="indexes_target_token_lemmatized", default=s.indexes_target_token_tokenized)  # type: ignore
        start, end = self.character_indices(
            token_index=idx, tokens=tokens, target=tokens[idx]
        )
        return context_preprocessed, start, end


class Tokenize(ContextPreprocessor):
    def preprocess_context(self, s: Series) -> tuple[str, int, int]:
        tokens = self.normalize_spelling(s.context_tokenized).split()
        idx = s.indexes_target_token_tokenized
        start, end = self.character_indices(
            token_index=idx, tokens=tokens, target=tokens[idx]
        )
        return " ".join(tokens), start, end
