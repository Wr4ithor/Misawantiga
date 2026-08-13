import os
import torch
from pathlib import Path

import torch.nn as nn

from misawantiga.config.defaults import Config
from tokenizer.tokenizer_hash import hash_file


def try_resume(config: Config) -> dict | None:
    local_path = config.ckpt_path
    # if local_path is None:
    #     print("No local checkpoints found. Trying from HF...")

    if local_path.exists():
        print(f"Resuming from local chekpoint: {local_path}")
        return torch.load(local_path, map_location=config.device, weights_only=False)

    if not (config.resume and config.hf_ckpt_repo):
        return None

    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=config.hf_ckpt_repo,
            filename="ckpt.pt",
            token=os.environ.get("HF_TOKEN"),
            local_dir=config.run_dir
        )
        print(f"Resuming from Hub checkpoint: {config.hf_ckpt_repo}")
        return torch.load(path, map_location=config.device, weights_only=False)
    except Exception as e:
        print(f"No Hub checkpoint to resume from ({type(e).__name__}: {e})")
        return None

def ensure_hub_repo(config: Config):
    if not config.hf_ckpt_repo:
        return
    from huggingface_hub import HfApi

    HfApi().create_repo(
        repo_id=config.hf_ckpt_repo,
        private=config.hf_private,
        exist_ok=True,
        token=os.environ.get("HF_TOKEN"),
    )

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Config,
    iter_num: int,
    val_loss: float,
    best_val_loss: float,
    tag: str = "latest",
):
    raw_model: nn.Module = getattr(model, "_orig_mod", model)

    tokenizer_sha256 = None
    if os.path.exists(config.tokenizer_path):
        tokenizer_sha256 = hash_file(config.tokenizer_path)

    checkpoint = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "model_name": config.model_name,
        "iter_num": iter_num,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "tokenizer_sha256": tokenizer_sha256,
    }
    path = config.ckpt_path
    torch.save(checkpoint, path)
    print(f"  → saved checkpoint to {path} ({tag}, val {val_loss:.4f})")

    if config.hf_ckpt_repo:
        try:
            from huggingface_hub import HfApi

            HfApi().upload_file(
                path_or_fileobj=path,
                path_in_repo="ckpt.pt",
                repo_id=config.hf_ckpt_repo,
                token=os.environ.get("HF_TOKEN"),
                commit_message=f"iter {iter_num} | val {val_loss:.4f} ({tag})",
            )
            print(f"  → pushed to {config.hf_ckpt_repo}")
        except Exception as e:
            print(f"  ! Hub upload failed ({type(e).__name__}: {e}) — continuing")