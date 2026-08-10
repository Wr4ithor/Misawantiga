from dataclasses import dataclass

import torch.nn as nn

from misawantiga.config.defaults import Config

# from .fry import ARCH_FIELDS as FRY_FIELDS
from .fry import FryNetwork


@dataclass
class ModelSpec:
    dispatch_key: str # Config.model_name, ckpt key
    codename: str # display name
    hf_repo: str # final published HF model repo
    cls: type[nn.Module] # model class
    arch_fields: tuple[str, ...] # fields to persist into config.json


# ---- Single source of truth ----
MODEL_REGISTRY: dict[str, ModelSpec] = {
    "fry": ModelSpec(
        dispatch_key="fry",
        codename="Fry",
        hf_repo="tbp",
        cls=FryNetwork,
        arch_fields=()
    )
}


def build_model(cfg: Config) -> nn.Module:
    return MODEL_REGISTRY[cfg.model_name].cls(cfg)

def spec_for(cfg: Config) -> nn.Module:
    return MODEL_REGISTRY[cfg.model_name] # type: ignore