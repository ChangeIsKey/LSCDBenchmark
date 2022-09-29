from typing import NewType

from pandas import Series
from pydantic import BaseModel

UseID = NewType("UseID", str)


class Use(BaseModel):
    # unique id for this specific use
    identifier: UseID
    # grouping id for the specific time period/dialect this use belongs to
    grouping: str
    # string representing one specific context of a word
    # (could be a preprocessed context, or a raw context)
    context: str
    # target word
    target: str
    # span of character indices in which the target word appears in `context`
    indices: tuple[int, int]

    @classmethod
    def from_series(cls, use: Series) -> "Use":
        return cls(
            identifier=use.identifier,
            grouping=use.grouping,
            context=use.context_preprocessed,
            target=use.lemma.split("_")[0],
            indices=(use.target_index_begin, use.target_index_end)
        )

    def __hash__(self) -> int:
        return hash(self.identifier)

    def __lt__(self, other: "Use") -> bool:
        return self.identifier < other.identifier
