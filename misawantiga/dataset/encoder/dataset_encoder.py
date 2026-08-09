import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
import random
from datetime import datetime, timezone
from typing import Optional, BinaryIO, Any

from tokenizer import BPETokenizer
from misawantiga.config.defaults import Default


class DatasetEncoder:
    def __init__(
        self, tokenizer: BPETokenizer
    ):
        self._tokenizer = tokenizer
        self._dtype = (
            np.uint16
            if self._tokenizer.vocab_size() <= np.iinfo(np.uint16).max
            else np.uint32
        )
        self._eos_id = self._tokenizer.eos_id

    # ------------------------------------------------------------------
    # Core writing helpers
    # ------------------------------------------------------------------

    def _encode_and_write(
        self, lines: list[str], out_f: BinaryIO, dtype, eos_id: int
    ) -> tuple[int, int]:
        """
        Encode each line independently and append EOS after every line.

        Returns
        -------
        n_tokens : int
            Total number of tokens written (including EOS tokens)
        n_examples : int
            Number of non-empty lines written
        """
        n_tokens = 0
        n_examples = 0

        for line in lines:
            line = line.rstrip("\n\r")
            if not line:
                continue

            tokens = self._tokenizer.encode(line)
            tokens.append(eos_id)  # <EOS> after each entry

            arr = np.asarray(tokens, dtype=dtype)
            arr.tofile(out_f)

            n_tokens += len(arr)
            n_examples += 1

        return n_tokens, n_examples

    def _get_split_paths(
        self, output_file: Path, val_ratio: float
    ) -> tuple[Path, Optional[Path]]:
        """Return (train_path, val_path). val_path is None when no split is requested."""
        if val_ratio <= 0.0:
            return output_file, None

        if not (0.0 < val_ratio < 1.0):
            raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")

        stem = output_file.stem
        suffix = output_file.suffix

        train_path = output_file.with_name(f"{stem}.train{suffix}")
        val_path = output_file.with_name(f"{stem}.val{suffix}")
        return train_path, val_path

    def _meta_path(self, bin_path: Path) -> Path:
        """Convention: tokens.train.bin → tokens.train.meta.json"""
        return bin_path.with_suffix(".meta.json")

    def _write_metadata(self, bin_path: Path, meta: dict[str, Any]) -> None:
        meta_path = self._meta_path(bin_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"Wrote meta  → {meta_path}")

    def _base_metadata(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """Common fields that appear in every metadata file."""
        meta = {
            "dtype": str(np.dtype(self._dtype)),
            "vocab_size": self._tokenizer.vocab_size(),
            "eos_id": int(self._eos_id),
            "created_at": datetime.now(timezone.utc).isoformat(),
            # Optional tokenizer version / name if the tokenizer exposes them
            "tokenizer": {
                "class": type(self._tokenizer).__name__,
            },
        }

        # Try to capture a few more tokenizer details if they exist
        for attr in ("name", "version", "model_name", "vocab_file"):
            if hasattr(self._tokenizer, attr):
                value = getattr(self._tokenizer, attr)
                if value is not None:
                    meta["tokenizer"][attr] = str(value)

        if extra:
            meta.update(extra)
        return meta

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode_file(
        self,
        input_file: Path,
        output_file: Path,
        chunk_size: int = 1_000_000,
        val_ratio: float = 0.0,
        seed: int = 42,
        source_name: str | None = None,
    ) -> None:
        """
        Encode a single text file into binary token format.

        If val_ratio > 0, two files are written:
            <stem>.train.bin  and  <stem>.val.bin
        plus corresponding .meta.json files.
        """
        if not input_file.exists():
            raise FileNotFoundError(f"Input file ({input_file}) doesn't exist")

        if not str(output_file).endswith(Default.ENCODED_DS_EXT):
            raise ValueError(f"Output file should be a {Default.ENCODED_DS_EXT} file")

        train_path, val_path = self._get_split_paths(output_file, val_ratio)

        with open(input_file, "r", encoding="utf-8", errors="replace") as f:
            total_lines = sum(1 for _ in f)

        rng = random.Random(seed)

        train_f = open(train_path, "wb")
        val_f: Optional[BinaryIO] = open(val_path, "wb") if val_path else None

        train_tokens = train_examples = 0
        val_tokens = val_examples = 0

        try:
            buffer_train: list[str] = []
            buffer_val: list[str] = []
            buffer_train_len = 0
            buffer_val_len = 0

            with open(input_file, "r", encoding="utf-8", errors="replace") as in_f:
                for line in tqdm(
                    in_f,
                    total=total_lines,
                    desc=f"Encoding {input_file.name}",
                    unit="lines",
                ):
                    if val_f is not None and rng.random() < val_ratio:
                        buffer_val.append(line)
                        buffer_val_len += len(line)
                        if buffer_val_len >= chunk_size:
                            n_tok, n_ex = self._encode_and_write(
                                buffer_val, val_f, self._dtype, self._eos_id
                            )
                            val_tokens += n_tok
                            val_examples += n_ex
                            buffer_val.clear()
                            buffer_val_len = 0
                    else:
                        buffer_train.append(line)
                        buffer_train_len += len(line)
                        if buffer_train_len >= chunk_size:
                            n_tok, n_ex = self._encode_and_write(
                                buffer_train, train_f, self._dtype, self._eos_id
                            )
                            train_tokens += n_tok
                            train_examples += n_ex
                            buffer_train.clear()
                            buffer_train_len = 0

                if buffer_train:
                    n_tok, n_ex = self._encode_and_write(
                        buffer_train, train_f, self._dtype, self._eos_id
                    )
                    train_tokens += n_tok
                    train_examples += n_ex
                if buffer_val and val_f is not None:
                    n_tok, n_ex = self._encode_and_write(
                        buffer_val, val_f, self._dtype, self._eos_id
                    )
                    val_tokens += n_tok
                    val_examples += n_ex
        finally:
            train_f.close()
            if val_f is not None:
                val_f.close()

        # ----- metadata -----
        common = {
            "source": str(input_file),
            "source_name": source_name or input_file.name,
            "val_ratio": val_ratio,
            "seed": seed,
        }

        self._write_metadata(
            train_path,
            self._base_metadata(
                {
                    **common,
                    "split": "train",
                    "n_tokens": train_tokens,
                    "n_examples": train_examples,
                    "bin_file": train_path.name,
                }
            ),
        )

        if val_path is not None:
            self._write_metadata(
                val_path,
                self._base_metadata(
                    {
                        **common,
                        "split": "val",
                        "n_tokens": val_tokens,
                        "n_examples": val_examples,
                        "bin_file": val_path.name,
                    }
                ),
            )

        print(f"Wrote train → {train_path}  ({train_tokens:,} tokens)")
        if val_path:
            print(f"Wrote val   → {val_path}  ({val_tokens:,} tokens)")

    def encode_dataset(
        self,
        input_dir: Path,
        output_file: Path,
        pattern: str = "*.md",
        recursive: bool = True,
        chunk_size: int = 1_000_000,
        val_ratio: float = 0.0,
        seed: int = 42,
        source_name: str | None = None,
    ) -> None:
        """
        Encode all matching files under input_dir into binary token format.

        If val_ratio > 0, two files are written:
            <stem>.train.bin  and  <stem>.val.bin
        plus corresponding .meta.json files.
        """
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory ({input_dir}) doesn't exist")

        if not input_dir.is_dir():
            raise NotADirectoryError(f"'{input_dir}' is not a directory")

        if not str(output_file).endswith(Default.ENCODED_DS_EXT):
            raise ValueError(f"Output file should be a {Default.ENCODED_DS_EXT} file")

        if recursive:
            files = sorted(input_dir.rglob(pattern))
        else:
            files = sorted(input_dir.glob(pattern))

        if not files:
            raise FileNotFoundError(
                f"No files matching '{pattern}' found in {input_dir}"
            )

        train_path, val_path = self._get_split_paths(output_file, val_ratio)
        rng = random.Random(seed)

        train_f = open(train_path, "wb")
        val_f: Optional[BinaryIO] = open(val_path, "wb") if val_path else None

        train_tokens = train_examples = 0
        val_tokens = val_examples = 0

        try:
            for file_path in tqdm(files, desc="Files", unit="file"):
                t_tok, t_ex, v_tok, v_ex = self._encode_one_file(
                    file_path,
                    train_f,
                    val_f,
                    self._dtype,
                    self._eos_id,
                    chunk_size,
                    val_ratio,
                    rng,
                )
                train_tokens += t_tok
                train_examples += t_ex
                val_tokens += v_tok
                val_examples += v_ex
        finally:
            train_f.close()
            if val_f is not None:
                val_f.close()

        # ----- metadata -----
        common = {
            "source": str(input_dir),
            "source_name": source_name or input_dir.name,
            "pattern": pattern,
            "recursive": recursive,
            "n_files": len(files),
            "val_ratio": val_ratio,
            "seed": seed,
        }

        self._write_metadata(
            train_path,
            self._base_metadata(
                {
                    **common,
                    "split": "train",
                    "n_tokens": train_tokens,
                    "n_examples": train_examples,
                    "bin_file": train_path.name,
                }
            ),
        )

        if val_path is not None:
            self._write_metadata(
                val_path,
                self._base_metadata(
                    {
                        **common,
                        "split": "val",
                        "n_tokens": val_tokens,
                        "n_examples": val_examples,
                        "bin_file": val_path.name,
                    }
                ),
            )

        print(f"Wrote train → {train_path}  ({train_tokens:,} tokens)")
        if val_path:
            print(f"Wrote val   → {val_path}  ({val_tokens:,} tokens)")

    def _encode_one_file(
        self,
        input_file: Path,
        train_f: BinaryIO,
        val_f: Optional[BinaryIO],
        dtype,
        eos_id: int,
        chunk_size: int,
        val_ratio: float,
        rng: random.Random,
    ) -> tuple[int, int, int, int]:
        """
        Encode a single file, optionally splitting lines into train/val.

        Returns
        -------
        train_tokens, train_examples, val_tokens, val_examples
        """
        buffer_train: list[str] = []
        buffer_val: list[str] = []
        buffer_train_len = 0
        buffer_val_len = 0

        train_tokens = train_examples = 0
        val_tokens = val_examples = 0

        with open(input_file, "r", encoding="utf-8", errors="replace") as in_f:
            total_lines = sum(1 for _ in in_f)
            in_f.seek(0)

            for line in tqdm(
                in_f,
                desc=input_file.name,
                unit="lines",
                leave=False,
                total=total_lines,
            ):
                if val_f is not None and rng.random() < val_ratio:
                    buffer_val.append(line)
                    buffer_val_len += len(line)
                    if buffer_val_len >= chunk_size:
                        n_tok, n_ex = self._encode_and_write(
                            buffer_val, val_f, dtype, eos_id
                        )
                        val_tokens += n_tok
                        val_examples += n_ex
                        buffer_val.clear()
                        buffer_val_len = 0
                else:
                    buffer_train.append(line)
                    buffer_train_len += len(line)
                    if buffer_train_len >= chunk_size:
                        n_tok, n_ex = self._encode_and_write(
                            buffer_train, train_f, dtype, eos_id
                        )
                        train_tokens += n_tok
                        train_examples += n_ex
                        buffer_train.clear()
                        buffer_train_len = 0

            if buffer_train:
                n_tok, n_ex = self._encode_and_write(
                    buffer_train, train_f, dtype, eos_id
                )
                train_tokens += n_tok
                train_examples += n_ex
            if buffer_val and val_f is not None:
                n_tok, n_ex = self._encode_and_write(
                    buffer_val, val_f, dtype, eos_id
                )
                val_tokens += n_tok
                val_examples += n_ex

        return train_tokens, train_examples, val_tokens, val_examples