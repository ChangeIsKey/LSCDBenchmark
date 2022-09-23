from pydantic import BaseModel
from src.wic.model import WICModel
from src.use import Use


class DeepMistake(BaseModel):
    def predict(self, use_pairs: list[tuple[Use, Use]]) -> list[float]:
        raise NotImplementedError
