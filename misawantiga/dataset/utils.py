import json
import numpy as np
from pathlib import Path
from misawantiga.config.defaults import Config


def _get_dtype(bin_meta_path: str | Path) -> np.dtype:
    with open(bin_meta_path, "r") as f:
        data = json.load(f)
    return data["dtype"]


def count_bin(bin_path: str | Path, bin_meta: str | Path) -> None:
    config = Config()
    dtype_str = _get_dtype(bin_meta)
    dtype = config.dtype_map.get(dtype_str) # type: ignore
    data = np.memmap(bin_path, dtype=dtype, mode="r") # type: ignore
    total_tokens = len(data)
    print(f"Total tokens: {total_tokens:,}")


# count_bin(
#     bin_path=Path("data/processed/train.bin"),
#     bin_meta=Path("data/processed/train.meta.json")
# )