from src.wic.model import WICModel
from src.use import Use


class DeepMistakeWIC(WICModel):
    def predict(self, use_pairs: list[tuple[Use, Use]]) -> list[float]:
        pass
