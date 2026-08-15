from misawantiga.config.defaults import Config
from pathlib import Path


DATASET_DIR = Path("data/datasets/CCU/entries")


def make_config() -> Config:
    return Config(
        run_name="misawantiga-fry",
        model_name="fry",
        dataset_dir=DATASET_DIR,
        run_dir=Path("data/runs"),
        hf_model_repo="StarOfRuin/misawantiga-fry-38M",
        device="cpu",
        dtype="float32",
        compile = False,
        batch_size=8,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        max_iters=200,
        eval_interval=2,
        eval_iters=5,
        warmup_ratio=0.1
    )