"""
Misawantiga Tokenizer Base Interface

Defines the public API that every tokenizer implementation must follow.

Current implementation:
    - BPETokenizer

Future possibilities:
    - WordPieceTokenizer
    - UnigramTokenizer
    - CharacterTokenizer
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, List


class BaseTokenizer(ABC):
    """
    Abstract base class for all tokenizers.

    Every tokenizer implementation must provide methods for:
        - training
        - encoding
        - decoding
        -serialization
    """

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """
        Convert text into a sequence of token IDs.

        Args:
            text:
                Input string.

        Returns:
            List of token IDs.
        """

        pass


    @abstractmethod
    def decode(self, token_ids: Iterable[int]) -> str:
        """
        Convert token IDs back into text.

        Args:
            token_ids:
                Sequence of token IDs.

        Returns:
            Decoded text.
        """

        pass

    @abstractmethod
    def token_to_id(self, token: str) -> int:
        """
        Return the token represented by an integer ID
        """

        pass

    @abstractmethod
    def id_to_token(self, token_id: int) -> str:
        """
        Return the token represented by an integer ID.
        """

        pass

    @abstractmethod
    def get_vocab(self) -> dict[str, int]:
        """
        Return the vocabulary mapping.
        
        Returns:
            Dictionary mapping token -> id.    
        """

        pass


    @abstractmethod
    def vocab_size(self) -> int:
        """Return the vocabulary size"""

        pass

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save tokenizer to disk"""

        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "BaseTokenizer":
        """Load tokenizer from disk"""
        pass