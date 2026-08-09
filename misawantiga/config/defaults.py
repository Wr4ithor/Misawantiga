import numpy as np
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Default:
    Name = "Misawantiga-CCU"
    Codename = "Fry"
    
    ENCODED_DS_EXT = ".bin"
    DATA_DIR = Path("data")
    DATASET_DIR = DATA_DIR / "datasets"

    DTYPE_MAP = {
        "uint16": np.uint16,
        "uint32": np.uint32,
        "int16": np.int16,
        "int32": np.int32
    }