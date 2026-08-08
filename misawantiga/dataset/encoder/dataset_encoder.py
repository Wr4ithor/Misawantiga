import numpy as np
from tqdm import tqdm
from pathlib import Path

from tokenizer import BPETokenizer
from config.defaults import Default


class DatsetEncoder:
    def __init__(self, tokenizer: BPETokenizer):
        self._tokenizer = tokenizer
        self._dtype =(
            np.uint16
            if self._tokenizer.vocab_size() <= np.iinfo(np.uint16).max
            else np.uint32
        )
        self._eos_id = self._tokenizer.eos_id

    def _encode_and_write(self, lines: list[str], out_f, dtype, eos_id: int):
        """Encode each line independently and append EOS after every line."""
        for line in lines:
            line = line.rstrip("\n\r")
            if not line:
                continue

            tokens = self._tokenizer.encode(line)
            tokens.append(eos_id)               # <EOS> after each entry

            np.asarray(tokens, dtype=dtype).tofile(out_f)


    def encode_file(
        self,
        input_file: Path,
        output_file: Path,
        chunk_size: int = 1_000_000
    ):
        if not input_file.exists():
            raise FileNotFoundError(f"Input file ({input_file}) doesn't exist")

        if not str(output_file).endswith(Default.ENCODED_DS_EXT):
            raise ValueError(f"Output file should be a {Default.ENCODED_DS_EXT} file")

        # First pass just to count lines (nicer progress bar)
        with open(input_file, "r", encoding="utf-8", errors="replace") as f:
            total_lines = sum(1 for _ in f)

        with open(output_file, "wb") as out_f:
            buffer: list[str] = []
            buffer_len = 0

            with open(input_file, "r", encoding="utf-8", errors="replace") as in_f:
                for line in tqdm(in_f, total=total_lines, desc=f"Encoding {input_file.name}", unit="lines"):
                    buffer.append(line)
                    buffer_len += len(line)

                    if buffer_len >= chunk_size:
                        self._encode_and_write(buffer, out_f, self._dtype, self._eos_id)
                        buffer.clear()
                        buffer_len = 0

                if buffer:
                    self._encode_and_write(buffer, out_f, self._dtype, self._eos_id)


    def encode_dataset(self,
        input_dir: Path,
        output_file: Path,
        pattern: str = "*.md",
        recursive: bool = True,
        chunk_size: int = 1_000_000
    ) -> None:
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
            raise FileNotFoundError(f"No files matching '{pattern}' found in {input_dir}")

        with open(output_file, "wb") as out_f:
            for file_path in tqdm(files, desc="Files", unit="file"):
                self._encode_one_file(file_path, out_f, self._dtype, self._eos_id, chunk_size)


    def _encode_one_file(
        self,
        input_file: Path,
        out_f,
        dtype,
        eos_id: int,
        chunk_size: int,
    ):
        """Encode a single file (with optional chunking) into an already-open binary stream."""
        buffer = []
        buffer_len = 0

        with open(input_file, "r", encoding="utf-8", errors="replace") as in_f:
            total_lines = sum(1 for _ in in_f)
            in_f.seek(0)

            for line in tqdm(
                in_f,
                desc=input_file.name,
                unit="lines",
                leave=False,
                total=total_lines
            ):
                buffer.append(line)
                buffer_len += len(line)

                if buffer_len >= chunk_size:
                    self._encode_and_write(buffer, out_f, dtype, eos_id)
                    buffer.clear()
                    buffer_len = 0

            if buffer:
                self._encode_and_write(buffer, out_f, dtype, eos_id)
        