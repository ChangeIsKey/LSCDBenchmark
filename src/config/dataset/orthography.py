from pydantic import BaseModel


class Orthography(BaseModel):
    translation_table: dict[str, dict[str, str]]
    normalize: bool = True

