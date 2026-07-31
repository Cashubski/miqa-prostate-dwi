"""Learning rate schedulers."""

import math
import torch


def build_scheduler(optimizer, name, epochs, warmup_epochs, steps_per_epoch):
    """Return a LambdaLR scheduler or None."""
    if name == "none":
        return None
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    def _lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
