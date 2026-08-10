from __future__ import annotations

import hashlib
from pathlib import Path


CHUNK = 1 << 16


def hash_file(tokenizer_path: str | Path) -> str:
    """Return the SHA-256 digest of `path`"""
    h = hashlib.sha256()
    with open(tokenizer_path, "rb") as f:
        while True:
            block = f.read(CHUNK)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


class TokenizerMismatch(RuntimeError):
    """Raised when a checkpoint's tokenizer hash does not match the on-disk one."""


def verify_match(tokenizer_path: str | Path, expected: str | None) -> str:
    actual = hash_file(tokenizer_path=tokenizer_path)
    if expected is not None and expected != actual:
        raise TokenizerMismatch(
            f"Tokenizer mismatch:\n"
            f"  ckpt expected: {expected}\n"
            f"  on-disk      : {actual}\n"
            f"  path         : {tokenizer_path}\n"
            f"This ckpt was trained against a different tokenizer. Either point at "
            f"the right pirate_bpe.json or retrain."
        )
    return actual