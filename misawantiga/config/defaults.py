import numpy as np
from dataclasses import dataclass


@dataclass
class Default:
    Name = "Misawantiga-CCU"
    Codename = "Fry"
    
    ENCODED_DS_EXT = ".bin"

    DTYPE_MAP = {
        "uint16": np.uint16,
        "uint32": np.uint32,
        "int16": np.int16,
        "int32": np.int32
    }