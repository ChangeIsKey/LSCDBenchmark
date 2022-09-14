from pydantic import BaseModel

class Truncation(BaseModel):
    tokens_before: float

