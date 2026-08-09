from dataclasses import dataclass
from pathlib import Path


@dataclass
class Paths:
    DATA_DIR = Path("data")
    DATASET_DIR = DATA_DIR / "datasets"