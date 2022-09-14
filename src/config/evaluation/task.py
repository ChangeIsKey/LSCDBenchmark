from enum import Enum
from pandera import Column, DataFrameSchema


class EvaluationTask(str, Enum):
    GRADED_CHANGE = "change_graded"
    BINARY_CHANGE = "change_binary"
    COMPARE = "COMPARE"
    SEMANTIC_PROXIMITY = "semantic_proximity"
    CLUSTERING = "clustering"
