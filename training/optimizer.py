from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from misawantiga.config.defaults import Config


# Param-name substrings that must NOT go to Muon (embeddings + tied LM head).
_ADAMW_ONLY = ("wte", "wpe", "lm_head")


def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Orthogonalize G (2D) via a quintic Newton-Schulz iteration.

    Returns a matrix with roughly the same singular vectors as G but singular
    values pushed toward 1. Runs in bfloat16 — the iteration is robust to the
    reduced precision and it keeps the step cheap.
    """
    assert G.ndim == 2, "Newton-Schulz expects a 2D matrix"
    a, b, c = (3.4445, -4.7750, 2.0315)

    X = G.bfloat16()
    # X.norm() calculates the frobeninus norm of X
    # eps = 1e-7 = 10 riased to -7 (to prevent division by zero)
    X /= X.norm() + 1e-7

    if G.size(0) < G.size(1): # rows by col
        X = X.T

    for _ in range(steps):
        # Precomputing A = X_n * X transpose
        # so that the equation could be smth like this:
        # X_(n + 1) = aX_n + bAX_n + cA^2X_n
        A = X @ X.T
        # factor out X_n so that a temporary var B = bA + cA^2
        B = b * A + c * (A @ A)
        # now the whole expression simplifies to:
        # X_(n + 1) = aX_n + BX_n
        X = a * X + B @ X

    if G.size(0) < G.size(1):
        X = X.T

    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params: list[dict]):
        for g in params:
            if g.get("use_muon", False):
                g["use_muon"] = True
                g.setdefault("lr", 0.02)
                g.setdefault("momentum", 0.95)
                g.setdefault("ns_steps", 5)
                g.setdefault("nesterov", True)
                g.setdefault("weight_decay", 0.0)
            else:
                g["use_muon"] = False
                g.setdefault("lr", 3e-4)
                g.setdefault("betas", (0.9, 0.95))
                g.setdefault("eps", 1e-8)
                g.setdefault("weight_decay", 0.0)
        super().__init__(params, {})


    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                self._muon_step(group)
            else:
                self._adamw_step(group)
        return loss


    def _muon_step(self, group: dict) -> None:
        lr, momentum, wd = group["lr"], group["momentum"], group["weight_decay"]
        for p in group["params"]:
            if p.grad is None:
                continue

            grad = p.grad  # Current parameter gradients
            state = self.state[p]
            buffer = state.get("momentum_buffer")
            if buffer is None:
                buffer = state["momentum_buffer"] = torch.zeros_like(grad)

            buffer.mul_(momentum).add_(grad)

            grad = grad.add(buffer, alpha=momentum) if group["nesterov"] else buffer
            update = zeropower_via_newtonschulz5(grad, group["ns_steps"]).to(p.dtype)

            if wd != 0.0:
                p.data.mul_(1.0 - lr * wd)

            scale = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
            update.mul_(scale)

            p.data.add_(update, alpha=-lr)


    def _adamw_step(self, group: dict) -> None:
        lr, (b1, b2) = group["lr"], group["betas"]
        eps, wd = group["eps"], group["weight_decay"]

        for p in group["params"]:
            if p.grad is None:
                continue

            grad = p.grad
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1

            t = state["step"]

            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

            exp_avg.mul_(b1).add_(grad, alpha=1.0 - b1)
            exp_avg_sq.mul_(b2).addcmul_(grad, grad, value=1.0 - b2)

            bias1, bias2 = 1.0 - b1**t, 1.0 - b2**t

            denom = (exp_avg_sq.sqrt() / math.sqrt(bias2)).add_(eps)
            if wd != 0.0:
                p.data.mul_(1.0 - lr * wd)

            p.data.addcdiv_(exp_avg, denom, value=-lr / bias1)


def _classify_params(model: nn.Module) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
    muon, adam_decay, adam_no_decay =  [], [], []
    seen: set = set()

    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue

        seen.add(p)
        embed_or_head = any(k in name for k in _ADAMW_ONLY)

        if p.dim() >= 2 and not embed_or_head:
            muon.append(p)
        elif p.dim() >= 2:
            adam_decay.append(p)
        else:
            adam_no_decay.append(p)
    return (muon, adam_decay, adam_no_decay)


def _adamw_groups(model: nn.Module, config: Config) -> list[dict]:
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    return [
        {"params": decay, "weight_decay": config.weight_decay, "lr_ratio": 1.0},
        {"params": no_decay, "weight_decay": 0.0, "lr_ratio": 1.0},
    ]


def build_optimizer(model: nn.Module, config: Config) -> torch.optim.Optimizer:
    if config.optimizer == "muon":
        muon, adamw_decay, adamw_no_decay = _classify_params(model)
        ratio = config.muon_lr / config.lr

        groups = [
            {
                "params": muon,
                "use_muon": True,
                "lr": config.muon_lr,
                "momentum": config.muon_momentum,
                "ns_steps": config.muon_ns_steps,
                "weight_decay": config.weight_decay,
                "lr_ratio": ratio
            },
            {
                "params": adamw_decay,
                "use_muon": False,
                "lr": config.lr,
                "betas": (config.adam_beta1, config.adam_beta2),
                "eps": config.adam_eps,
                "weight_decay": config.weight_decay,
                "lr_ratio": 1.0
            },
            {
                "params": adamw_no_decay,
                "use_muon": False,
                "lr": config.lr,
                "betas": (config.adam_beta1, config.adam_beta2),
                "eps": config.adam_eps,
                "weight_decay": 0.0,
                "lr_ratio": 1.0
            }
        ]

        groups = [g for g in groups if g["params"]]  # drop empties
        optimizer = Muon(groups)

        n_muon = sum(p.numel() for p in muon)
        n_adam = sum(p.numel() for p in adamw_decay + adamw_no_decay)

        print(
            f"Optimizer: Muon (lr={config.muon_lr}, momentum={config.muon_momentum}, "
            f"ns_steps={config.muon_ns_steps}) + AdamW fallback (lr={config.lr})"
        )
        print(f"  Muon params:   {n_muon:,} ({len(muon)} tensors)")
        print(f"  AdamW params:  {n_adam:,} ({len(adamw_decay) + len(adamw_no_decay)} tensors)")

        return optimizer

    if config.optimizer != "adamw":
        raise ValueError(
            f"Unknown optimizer {config.optimizer!r}; expected 'adamw' or 'muon'"
        )

    groups = _adamw_groups(model, config)

    optimizer = torch.optim.AdamW(
        groups, lr=config.lr, betas=(config.adam_beta1, config.adam_beta2)
    )

    n_decay = sum(p.numel() for p in groups[0]["params"])
    n_no_decay = sum(p.numel() for p in groups[1]["params"])

    print(f"Optimizer: AdamW, lr={config.lr}, betas=({config.adam_beta1}, {config.adam_beta2})")
    print(f"  Decayed params:    {n_decay:,} ({len(groups[0]['params'])} tensors)")
    print(f"  No-decay params:   {n_no_decay:,} ({len(groups[1]['params'])} tensors)")

    return optimizer