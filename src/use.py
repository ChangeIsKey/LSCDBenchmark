from dataclasses import dataclass


@dataclass
class Use:
    target: str
    identifier: str
    context_preprocessed: str
    target_index_begin: int
    target_index_end: int

    # grouping: str
    # date: int
    # target_sentence_index_begin: int
    # target_sentence_index_end: int
