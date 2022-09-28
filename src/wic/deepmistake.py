from src.wic.model import Model
from src.use import Use


class DeepMistake(Model):
    def similarities(
        self, use_pairs: list[tuple[Use, Use]]
    ) -> dict[tuple[str, str], float]:
        raise NotImplementedError
