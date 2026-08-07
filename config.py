from pydantic import BaseModel

class Config(BaseModel):
    VOCAB_SIZE: int = 32000
    DEFAULT_TOKENIZER: str = "data/datasets/default_bpe.json"