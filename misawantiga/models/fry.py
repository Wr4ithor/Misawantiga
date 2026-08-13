"""Misawantiga Fry — v1 architecture. Frozen: do not modify after shipping"""
from typing import Any

import torch
import torch.nn as nn

from misawantiga.config.defaults import Config
from training.loss import calculate_loss

ARCH_FIELDS = (
    "vocab_size",
    "block_size",
    "n_layer",
    "n_head",
    "d_model",
    "dropout",
    "bias"
)


class FryNetwork(nn.Module):
    pass