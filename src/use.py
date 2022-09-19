from dataclasses import dataclass

@dataclass
class Use:
    # string representing one specific context of a word 
    # (could be a preprocessed context, or a raw context)
    context: str
    # target word 
    target: str
    # span of character indices in which the target word appears in `context`
    indices: tuple[int, int]