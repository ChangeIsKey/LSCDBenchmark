from typing import TypeAlias

from pandas import Series
from pydantic import BaseModel

UseID: TypeAlias = str


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
    # part-of-speech
    pos: str

    @classmethod
    def from_series(cls, use: Series) -> "Use":
        """Return one use from type series with specific columns. The use is the Series includes 
        the folloing columns: lemma, pos, grouping, identifier, context_prepeocessed,
        target_index_begin, and target_index_end.

        :param use: use
        :type use: Series
        :return: Use object with specific columns.
        :rtype: Use
        """        
        return cls(
            identifier=use.identifier,
            grouping=use.grouping,
            pos=use.pos,
            context=use.context_preprocessed,
            target=use.lemma.split("_")[0],
            indices=(use.target_index_begin, use.target_index_end),
        )

    def __hash__(self) -> int:
        """Called by built-in function hash() and for operations on members of hashed collections.

        :return: The hash value of identifier of the Use.
        :rtype: int
        """        
        return hash(self.identifier)

    def __lt__(self, other: "Use") -> bool:
        """Compare the value of the identifier of self.use and the value of the identifier of 
        other use.

        :param other: The use to be compared.
        :type other: Use
        :return: Return Ture if the value of the identifier of self.use is less than the value of the identifier of 
        other use.
        :rtype: bool
        """        
        return self.identifier < other.identifier
