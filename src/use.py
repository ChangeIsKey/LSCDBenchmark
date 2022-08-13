from dataclasses import dataclass


@dataclass
class Use:
    identifier: str
    context_preprocessed: str
    target_index_begin: int
    target_index_end: int
