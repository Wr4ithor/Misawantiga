import numpy as np
from tqdm import tqdm
from pathlib import Path
import random
from typing import Optional, BinaryIO

from tokenizer import BPETokenizer
from config.defaults import Default


class DatasetEncoder:
    def __init__(self, tokenizer: BPETokenizer):
        self._tokenizer = tokenizer
        self._dtype = (
            np.uint16
            if self._tokenizer.vocab_size() <= np.iinfo(np.uint16).max
            else np.uint32
        )
        self._eos_id = self._tokenizer.eos_id

    def _encode_and_write(
        self, lines: list[str], out_f: BinaryIO, dtype, eos_id: int
    ) -> None:
        """Encode each line independently and append EOS after every line."""
        for line in lines:
            line = line.rstrip("\n\r")
            if not line:
                continue

            tokens = self._tokenizer.encode(line)
            tokens.append(eos_id)  # <EOS> after each entry

            np.asarray(tokens, dtype=dtype).tofile(out_f)

    def _get_split_paths(
        self, output_file: Path, val_ratio: float
    ) -> tuple[Path, Optional[Path]]:
        """Return (train_path, val_path). val_path is None when no split is requested."""
        if val_ratio <= 0.0:
            return output_file, None

        if not (0.0 < val_ratio < 1.0):
            raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")

        stem = output_file.stem
        suffix = output_file.suffix  # should be Default.ENCODED_DS_EXT

        train_path = output_file.with_name(f"{stem}.train{suffix}")
        val_path = output_file.with_name(f"{stem}.val{suffix}")
        return train_path, val_path

    def encode_file(
        self,
        input_file: Path,
        output_file: Path,
        chunk_size: int = 1_000_000,
        val_ratio: float = 0.0,
        seed: int = 42,
    ) -> None:
        """
        Encode a single text file into binary token format.

        If val_ratio > 0, two files are written:
            <stem>.train.bin  and  <stem>.val.bin
        """
        if not input_file.exists():
            raise FileNotFoundError(f"Input file ({input_file}) doesn't exist")

        if not str(output_file).endswith(Default.ENCODED_DS_EXT):
            raise ValueError(f"Output file should be a {Default.ENCODED_DS_EXT} file")

        train_path, val_path = self._get_split_paths(output_file, val_ratio)

        # First pass just to count lines (nicer progress bar)
        with open(input_file, "r", encoding="utf-8", errors="replace") as f:
            total_lines = sum(1 for _ in f)

        rng = random.Random(seed)

        # Open files — only open val if we actually need it
        train_f = open(train_path, "wb")
        val_f: Optional[BinaryIO] = open(val_path, "wb") if val_path else None

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
                            self._encode_and_write(
                                buffer_val, val_f, self._dtype, self._eos_id
                            )
                            buffer_val.clear()
                            buffer_val_len = 0
                    else:
                        buffer_train.append(line)
                        buffer_train_len += len(line)
                        if buffer_train_len >= chunk_size:
                            self._encode_and_write(
                                buffer_train, train_f, self._dtype, self._eos_id
                            )
                            buffer_train.clear()
                            buffer_train_len = 0

                # Flush remaining buffers
                if buffer_train:
                    self._encode_and_write(
                        buffer_train, train_f, self._dtype, self._eos_id
                    )
                if buffer_val and val_f is not None:
                    self._encode_and_write(
                        buffer_val, val_f, self._dtype, self._eos_id
                    )
        finally:
            train_f.close()
            if val_f is not None:
                val_f.close()

        print(f"Wrote train → {train_path}")
        if val_path:
            print(f"Wrote val   → {val_path}")

    def encode_dataset(
        self,
        input_dir: Path,
        output_file: Path,
        pattern: str = "*.md",
        recursive: bool = True,
        chunk_size: int = 1_000_000,
        val_ratio: float = 0.0,
        seed: int = 42,
    ) -> None:
        """
        Encode all matching files under input_dir into binary token format.

        If val_ratio > 0, two files are written:
            <stem>.train.bin  and  <stem>.val.bin
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

        try:
            for file_path in tqdm(files, desc="Files", unit="file"):
                self._encode_one_file(
                    file_path,
                    train_f,
                    val_f,
                    self._dtype,
                    self._eos_id,
                    chunk_size,
                    val_ratio,
                    rng,
                )
        finally:
            train_f.close()
            if val_f is not None:
                val_f.close()

        print(f"Wrote train → {train_path}")
        if val_path:
            print(f"Wrote val   → {val_path}")

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
    ) -> None:
        """Encode a single file, optionally splitting lines into train/val."""
        buffer_train: list[str] = []
        buffer_val: list[str] = []
        buffer_train_len = 0
        buffer_val_len = 0

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
                        self._encode_and_write(buffer_val, val_f, dtype, eos_id)
                        buffer_val.clear()
                        buffer_val_len = 0
                else:
                    buffer_train.append(line)
                    buffer_train_len += len(line)
                    if buffer_train_len >= chunk_size:
                        self._encode_and_write(buffer_train, train_f, dtype, eos_id)
                        buffer_train.clear()
                        buffer_train_len = 0

            if buffer_train:
                self._encode_and_write(buffer_train, train_f, dtype, eos_id)
            if buffer_val and val_f is not None:
                self._encode_and_write(buffer_val, val_f, dtype, eos_id)