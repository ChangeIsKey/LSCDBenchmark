from typing import Callable, Optional, Any
from src.config.custom_method import CustomMethod
from pandas import Series
from src.preprocessing import keep_intact

class Preprocessing(CustomMethod):
    def __init__(self, module: str, method: Optional[str], params: Optional[dict[str, Any]] = None) -> None:
        if params is None:
            params = {}
        super().__init__(module=module, method=method, params=params, default=keep_intact)

    def __call__(self, s: Series, translation_table: dict[str, str]) -> Series:
        context, start, end = self.func(s, translation_table)
        return Series(
            {
                "context_preprocessed": context,
                "target_index_begin": start,
                "target_index_end": end,
            }
        )

