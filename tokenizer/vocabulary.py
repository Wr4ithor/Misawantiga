from __future__ import annotations

import json
from pathlib import Path


class Vocabulary:
    """
    Lightweight vocabulary mapping for tokenizer training and inference.

    The vocabulary stores a bidirectional mapping between tokens and integer IDs,
    and can be persisted to disk for reuse across runs.
    """

    def __init__(self):
        """
        Initialize an empty vocabulary.
        """
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: list[str] = []

        self._frozen = False


    @property
    def frozen(self):
        return self._frozen


    def add_token(self, token: str):
        """
        Add a token to the vocabulary and assign it an ID.

        Args:
            token:
                Token to add.

        Returns:
            The token ID assigned to the token.
        """
        if self._frozen:
            raise ValueError(f"Vocabulary frozen. Cannot add token {token} to the vocabulary.")

        if token in self._token_to_id:
            return self._token_to_id[token]

        idx = len(self._token_to_id) + 1
        self._token_to_id[token] = idx

        return self._token_to_id.get(token)


    def token_to_id(self, token: str) -> int:
        """
        Convert a token into its integer ID.

        Args:
            token:
                Token to look up.

        Returns:
            Integer ID for the token.
        """
        idx = self._token_to_id.get(token)
        if idx is None:
            raise KeyError(f"Unknown token: {token}")

        return idx


    def id_to_token(self, idx: int) -> str:
        """
        Convert an integer ID back into a token.

        Args:
            idx:
                Token ID to look up.

        Returns:
            Token represented by the ID.
        """
        token = self._id_to_token[idx]
        if token is None:
            raise ValueError(f"ID '{idx}' not found in the vocabulary")

        return token


    def __contains__(self, item: str) -> bool:
        """
        Return whether a token exists in the vocabulary.
        """
        if not item in self._token_to_id:
            return False

        return True


    def __len__(self) -> int:
        """
        Return the number of tokens in the vocabulary.
        """
        return len(self._token_to_id)


    def __iter__(self):
        return iter(self._token_to_id)


    def save(self, path: str | Path) -> None:
        """
        Save the vocabulary mappings to disk.

        Args:
            idx_path:
                Path to save the ID-to-token mapping.

            tkn_path:
                Path to save the token-to-ID mapping.
        """
        if not isinstance(path, Path):
            path = Path(path)

        with open(path, "w") as f:
            json.dump(self._token_to_id, f, indent=4)


    @classmethod
    def load(cls, path: str | Path):
        """
        Load vocabulary mappings from disk.

        Args:
            idx_path:
                Path to the saved ID-to-token mapping.

            tkn_path:
                Path to the saved token-to-ID mapping.

        Returns:
            Tuple containing the loaded vocabulary mappings.
        """
        vocab = cls()

        if not isinstance(path, Path):
            path = Path(path)

            for token, _ in vocab._token_to_id.items():
                vocab._id_to_token.append(token)

        return vocab


    def freeze(self) -> None:
        self._frozen = True


    def unfreeze(self) -> None:
        self._frozen = False
