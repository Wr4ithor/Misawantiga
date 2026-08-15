from dataclasses import dataclass
import importlib.util
from pathlib import Path
from numpy import uint16, uint32, uint64, int16, int32, int64
from torch import float16, float32, bfloat16


@dataclass
class Config:
    # ---- Run identity ----
    run_name: str = "misawantiga-fry"

    # --- Model dispatch (key into misawantiga.ai_models.MODEL_REGISTERY) [not implemented] ----
    model_name: str = "fry"

    # --- Default file extensions ----
    encoded_ds_ext: str = ".bin"

    # ---- Paths (Everything else derives from this) ----
    data_dir: Path = Path("data")
    run_dir: Path = data_dir / "runs"
    dataset_dir: Path = data_dir / "datasets/CCU/entries"

    # ---- Model architecture ----
    vocab_size: int = 8192
    block_size: int = 256
    n_layer: int = 8
    n_head: int = 8
    d_model: int = 512
    dropout: float = 0.1
    bias: bool = False
    ffn_dim: int = 4 * d_model
    context: int = 512

    # ---- Architecture flags (Cube+; Fry/Shugen leave all false) ----
    # Cube and Shugen are future model ideas
    use_rope: bool = False  # RoPE rotary pos-emb instead of learned wpe table
    use_swiglu: bool = False  # SwiGLU FFN (8/3 expansion) instead of GELU MLP
    use_rmsnorm: bool = False  # RMSNorm instead of LayerNorm
    use_qk_norm: bool = False  # per-head RMSNorm on Q and K (stabilizes depth)
    rope_theta: float = 10000.0  # RoPE base frequency

    # ---- Optimizer ----
    optimizer: str = "adamw" # "adamw" | "muon"
    lr: float = 3e-4  # AdamW peak LR (and fallback LR under Muon)
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    # Muon (only used when optimizer == "muon"); embeddings/head/norms stay AdamW.
    muon_lr: float = 0.02  # Muon peak LR — much higher than AdamW's
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5  # Newton-Schulz iterations per step

    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8

    # ---- LR schedule ----
    warmup_ratio: float = 0.01
    decay_ratio: float = 0.1
    min_lr: float = lr * 0.1

    # ---- Training loop ----
    epochs: float = 1.0  # passes over train.bin; resolves to max_iters at train start
    max_iters: int = 20000  # hard ceiling on iters (and the smoke-run cap)
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    eval_interval: int = 500
    eval_iters: int = 100
    log_interval: int = 1
    num_workers: int = 2
    pin_memory: bool = True
    drop_last: bool = True

    # ----- System -----
    device: str = "cpu"
    dtype: str = "float32"
    compile: bool = False
    seed: int = 1337

    # ----- HF Hub -----
    hf_model_repo: str = "StarOfRuin/fry"  # final published model repo
    hf_ckpt_repo: str | None = None  # rolling ckpt sync repo (private)
    pretrained_ckpt_repo: str | None = None  # SFT source: repo holding pretraining ckpt.pt
    hf_private: bool = True
    resume: bool = True

    # --- dtype mappings ----
    dtype_map = {
        "uint16": uint16,
        "uint32": uint32,
        "uint64": uint64,
        "int16": int16,
        "int32": int32,
        "int64": int64,
        "bfloat16": bfloat16,
        "float16": float16,
        "float32": float32
    }

    @property
    def train_bin(self) -> Path:
        return self.data_dir / "processed/train.bin"

    @property
    def train_meta(self) -> Path:
        return self.data_dir / "processed/train.meta.json"

    @property
    def val_bin(self) -> Path:
        return self.data_dir / "processed/val.bin"

    @property
    def val_meta(self) -> Path:
        return self.data_dir / "processed/val.meta.json"

    @property
    def tokenizer_path(self) -> Path:
        return self.data_dir / "datasets/CCU/ccu_bpe.json"

    @property
    def tokenizer_dir(self) -> Path:
        return self.data_dir / "datasets/CCU/"

    @property
    def ckpt_dir(self) -> Path:
        return self.run_dir

    @property
    def ckpt_path(self) -> Path:
        return self.run_dir / "ckpt.pt"


def load_config(path: str) -> Config:
    """
    Load a Config from a Python file that defines a`config` module-level variable
    or a `make_config()` function. E.g. `configs/sloop.py`.
    """
    p = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(p.stem, p)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config module from {path}")
    
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if hasattr(mod, "make_config"):
        return mod.make_config()
    if hasattr(mod, "config"):
        return mod.config
    raise AttributeError(
        f"{path} must define `config = Config(...)` or `def make_config() -> Config`"
    )