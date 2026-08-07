from pydantic import BaseModel

class Config(BaseModel):
    VOCAB_SIZE: int = 32000