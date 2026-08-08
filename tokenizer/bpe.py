from __future__ import annotations

from typing import Sequence
from pathlib import Path

from .base import BaseTokenizer
from tokenizers import Tokenizer


class BPETokenizer(BaseTokenizer):
    """
    Class for BPE Tokenizer.
    """

    def __init__(self, tokenizer: Tokenizer):
        self._tokenizer = tokenizer


    def encode(self, text: str) -> list[int]:
        encoded = self._tokenizer.encode(text)
        return encoded.ids


    def decode(self, token_ids: Sequence[int]) -> str:
        decoded = self._tokenizer.decode(token_ids)
        return decoded


    def id_to_token(self, token_id: int) -> str | None:
        token = self._tokenizer.id_to_token(token_id)
        if token is None:
            raise KeyError(f"Unknown ID: {token}")
        
        return token


    def token_to_id(self, token: str) -> int | None:
        idx = self._tokenizer.token_to_id(token)
        if idx is None:
            raise KeyError(f"Unknown Token: {token}")
        
        return idx


    def get_vocab(self) -> dict[str, int]:
        return self._tokenizer.get_vocab()


    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()


    def save(self, path: str | Path) -> None:
        if not isinstance(path, Path):
            path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)

        self._tokenizer.save(str(path))


    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        if not isinstance(path, Path):
            path = Path(path)
        if not path.exists():
            raise FileNotFoundError("Tokenizer file doesn't exists")
        tokenizer = Tokenizer.from_file(str(path))
        return cls(tokenizer)

    @property
    def pad_id(self) -> int:
        return self._tokenizer.token_to_id("<PAD>")

    @property
    def unk_id(self) -> int:
        return self._tokenizer.token_to_id("<UNK>")

    @property
    def bos_id(self) -> int:
        return self._tokenizer.token_to_id("<BOS>")

    @property
    def eos_id(self) -> int:
        return self._tokenizer.token_to_id("<EOS>")
