from src.wic.model import Model
from src.use import Use


class DeepMistake(Model):
    def predict(self, use_pairs: list[tuple[Use, Use]]) -> list[float]:
        raise NotImplementedError
