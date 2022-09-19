from __future__ import annotations
from dataclasses import dataclass
from config.config import UseID

from target import Target

@dataclass
class Use:
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
    def from_df(cls, target: Target) -> dict[str, Use]:
        return {
            use.identifier: cls(
                identifier=use.identifier, 
                grouping=use.grouping, 
                context=use.context_preprocessed, 
                indices=(use.target_index_begin, use.target_index_end)
            ) for _, use in target.uses.iterrows()
        } 
        