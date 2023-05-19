import re
from typing import Any

from pandas import Series
from pydantic import BaseModel, Field
from logging import getLogger

log = getLogger(__name__)


class ContextPreprocessor(BaseModel):
    """Base class for all kinds of context preprocessing strategies

    :param BaseModel: _description_
    :type BaseModel: _type_
    :raises ValueError: _description_
    :raises NotImplementedError: _description_
    :raises NotImplementedError: _description_
    :return: _description_
    :rtype: _type_
    """

    spelling_normalization: dict[str, str] | None = Field(...)
    """Dictionary of substring replacements to apply on the contexts"""

    def __init__(self, **data) -> None:
        """Creates a new context preprocessor and postprocesses 
        the spelling normalization table (replaces underscores with spaces)
        """

        super().__init__(**data)

        if self.spelling_normalization is not None:
            self.spelling_normalization = {
                k.replace("_", " "): v for k, v in self.spelling_normalization.items()
            }
        else:
            self.spelling_normalization = {}

    @staticmethod
    def start_char_index(token_index: int, tokens: list[str]) -> int:
        """Finds the index of the first character of the target token, i.e. `tokens[token_index]`

        :param token_index: the index of the target word in the list of tokens
        :type token_index: int
        :param tokens: the list of tokens
        :type tokens: list[str]
        :raises ValueError: If the token is not found
        :return: the start character index of the target word
        :rtype: int
        """
        char_idx = -1
        for i, token in enumerate(tokens):
            if i == token_index:
                # char_idx will be one index to the left of the target, so we need to add 1
                start = char_idx + 1
                return start
            char_idx += len(token) + 1  # plus one space
        raise ValueError

    def normalize_spelling(self, context: str, start: int) -> tuple[str, int]:
        """Applies the preprocessor's spelling normalization table and
        the new start character index of the target word after all modifications

        :param context: Context sentence of the target word
        :type context: str
        :param start: Start character index of the target word
        :type start: int
        :return: A tuple consisting of the modified string and the new start character index
        :rtype: tuple[str, int]
        """        
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

    def fields_from_series(self, s: Series) -> dict[str, Any]:
        """Selects fields from a pandas Series

        :param s: A row in a uses.csv file
        :type s: Series
        :raises NotImplementedError: _description_
        :return: A dictionary of parameters relevant to pass to the preprocess function
        :rtype: dict[str, Any]
        """
        raise NotImplementedError

    def preprocess(self, *args, **kwargs) -> tuple[str, int, int]:
        raise NotImplementedError

    def __call__(self, s: Series) -> Series:
        """Applies the preprocessing strategy based on a pandas.Series from a uses.csv file

        :param s: _description_
        :type s: Series
        :return: _description_
        :rtype: Series
        """
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
        """Applies the preprocessing strategy in a standalone manner

        :param context: The context sentence of the target word
        :type context: str
        :param index: The start character index of the target word
        :type index: int
        :param lemma: The lemma of the target word
        :type lemma: str
        :return: A tuple consisting of the modified string, and the start and end character indices of the target word
        :rtype: tuple[str, int, int]
        """
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
        """Returns the unmodified context and the character indices of the target word

        :param context: The context sentence of the target word
        :type context: str
        :param start: The start character index of the target word
        :type start: int
        :param end: The end character index of the target word
        :type end: int
        :return: A tuple consisting of the unmodified string, and the start and end character indices of the target word
        :rtype: tuple[str, int, int]
        """
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
        """Applies the preprocessing strategy in a standalone manner

        :param context: The context sentence of the target word
        :type context: str
        :param index: The start character index of the target word
        :type index: int
        :return: A tuple consisting of the modified string, and the start and end character indices of the target word
        :rtype: tuple[str, int, int]
        """
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
        """Applies the preprocessing strategy in a standalone manner

        :param context: The context sentence of the target word
        :type context: str
        :param index: The start character index of the target word
        :type index: int
        :return: A tuple consisting of the modified string, and the start and end character indices of the target word
        :rtype: tuple[str, int, int]
        """
        tokens = context.split()
        start = self.start_char_index(token_index=index, tokens=tokens)
        if self.spelling_normalization is not None:
            context, start = self.normalize_spelling(context, start)

        end = start + len(tokens[index]) - 1
        return context, start, end


class Normalize(ContextPreprocessor):
    default: str
    """Column to extract from a Series if a given use does not contain a pre-normalized context"""
    def fields_from_series(self, s: Series) -> dict[str, str | int]:
        context = s.get("context_normalized")
        if context is None:
            log.warn(
                f"(lemma={s.lemma}, use={s.identifier}) does not contain a pre-normalized context, {self.default} will be used"
            )
        context = s[self.default]
        return {
            "context": context,
            "index": int(s.indexes_target_token_tokenized),
        }

    def preprocess(self, context: str, index: int) -> tuple[str, int, int]:
        """Applies the preprocessing strategy in a standalone manner

        :param context: The context sentence of the target word
        :type context: str
        :param index: The start character index of the target word
        :type index: int
        :return: A tuple consisting of the modified string, and the start and end character indices of the target word
        :rtype: tuple[str, int, int]
        """
        tokens = context.split()
        start = self.start_char_index(token_index=index, tokens=tokens)

        if self.spelling_normalization is not None:
            context, start = self.normalize_spelling(context, start)

        end = start + len(tokens[index]) - 1
        return context, start, end
