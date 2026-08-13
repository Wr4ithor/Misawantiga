import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


class WSDScheduler(_LRScheduler):

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        stable_steps: int,
        decay_steps: int,
        peak_lr: float | None = None,
        min_lr: float = 0.0,
        decay_type: str = "cosine",
        last_epoch: int = -1,
    ):
        """Warmup-Stable-Decay (WSD) Learning Rate Scheduler for PyTorch.

        Supports 'linear', 'cosine', and 'sqrt' decay phases.
        """
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.decay_steps = decay_steps
        self.min_lr = min_lr
        self.decay_type = decay_type
        self.peak_lr = peak_lr

        # Base _LRScheduler init MUST be called after setting peak_lr logic
        super().__init__(optimizer, last_epoch)

    def _get_base_lrs(self):
        """Helper to determine peak learning rates per parameter group."""
        if self.peak_lr is not None:
            return [self.peak_lr] * len(self.optimizer.param_groups)
        return [group["lr"] for group in self.optimizer.param_groups]

    def get_lr(self):
        # self.last_epoch tracks current step (starts at 0)
        step = self.last_epoch

        lrs = []
        base_lrs = self._get_base_lrs()

        for base_lr in base_lrs:
            # 1. Warmup Phase
            if step < self.warmup_steps:
                lr = base_lr * (step / max(1, self.warmup_steps))

            # 2. Stable Phase
            elif step < (self.warmup_steps + self.stable_steps):
                lr = base_lr

            # 3. Decay Phase
            else:
                decay_progress = (
                    step - self.warmup_steps - self.stable_steps
                ) / max(1, self.decay_steps)
                decay_progress = min(1.0, max(0.0, decay_progress))

                if self.decay_type == "linear":
                    factor = 1.0 - decay_progress
                elif self.decay_type == "cosine":
                    factor = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
                elif self.decay_type == "sqrt":
                    factor = 1.0 - math.sqrt(decay_progress)
                else:
                    raise ValueError(
                        f"Unsupported decay_type: '{self.decay_type}'. Choose 'linear', 'cosine', or 'sqrt'."
                    )

                lr = self.min_lr + (base_lr - self.min_lr) * factor

            lrs.append(lr)

        return lrs