from pydantic import BaseModel
from typing import Optional
from src.config.evaluation.task import EvaluationTask
from src.config.evaluation.binary_threshold import Threshold
    

class EvaluationConfig(BaseModel):
    task: Optional[EvaluationTask]
    binary_threshold: Threshold | None = None

