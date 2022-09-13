from enum import Enum
import pandera as pa
from pandera import Column, DataFrameSchema


class EvaluationTask(str, Enum):
    GRADED_CHANGE = "change_graded"
    BINARY_CHANGE = "change_binary"
    COMPARE = "COMPARE"
    SEMANTIC_PROXIMITY = "semantic_proximity"
    CLUSTERING = "clustering"

    @property
    def schema(self):
        if self.value in {EvaluationTask.GRADED_CHANGE, EvaluationTask.BINARY_CHANGE, EvaluationTask.COMPARE}:
            return DataFrameSchema({
                self.value: Column(float)
            })
        return None
