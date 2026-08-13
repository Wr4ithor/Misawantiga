import math
from torch.optim.lr_scheduler import _LRScheduler


class WSDScheduler(_LRScheduler):
    def __init__(
            self, optimizer, warmup_steps: int, stable_steps: int, decay_steps: int,
            peak_lr=None, min_lr=0.0, decay_type="linear", last_epoch=-1
    ):
        """
        decay_type: "linear", "sqrt", "cosine", ...
        If peak_lr is None, the optimizer's initial lr is treated as peak.
        """
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.decay_steps = decay_steps
        self.min_lr = min_lr
        self.decay_type = decay_type
        self.last_epoch = last_epoch
        super().__init__(optimizer, self.last_epoch) # type: ignore
        # store the peak lrs from the optimizer
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        if peak_lr is not None:
            self.base_lrs = [peak_lr] * len(self.base_lrs)


    def get_lr(self):
        step = self.last_epoch + 1
        lrs = []

        for base_lr in self.base_lrs:
            if step <= self.warmup_steps:
                lr = base_lr * step / max(1, self.warmup_steps)
            elif step <= self.warmup_steps + self.stable_steps:
                lr = base_lr
            else:
                progress = (step - self.warmup_steps - self.stable_steps) / max(1, self.decay_steps)
                progress = min(1.0, progress)

                if self.decay_type == "linear":
                    factor = 1.0 - progress
                elif self.decay_type == "cosine":
                    factor = 0.5 * (1.0 + math.cos(math.pi * progress))
                elif self.decay_type == "sqrt":
                    factor = 1.0 - math.sqrt(progress)
                else:
                    raise ValueError(f"Unknown decay_type {self.decay_type}")
                lr = self.min_lr + (base_lr - self.min_lr) * factor
            lrs.append(lr)
        return lrs
