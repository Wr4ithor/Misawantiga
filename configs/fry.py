from dataclasses import dataclass


@dataclass
class FryConfig:
    name: str = "Misawantiga-CCU-Fry"
    codename: str = "Fry"

    vocab_size: int = 40_000

    n_heads: int = 8
    n_layers: int = 8
    d_model: int = 512 # Embedding size
    context_length: int = 512
    ffn_dim: int = 4 * d_model

    dropout: float = 0.1