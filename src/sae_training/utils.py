"""
Utility functions for SAE training on EEG data.

Includes learning rate schedulers, activation extraction helpers, and other utilities.
"""

import math
from typing import Optional

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


def get_scheduler(
    scheduler_name: Optional[str],
    optimizer: optim.Optimizer,
    **kwargs,
):
    """Create a learning rate scheduler.

    Args:
        scheduler_name: "constant" | "constantwithwarmup" | "linearwarmupdecay" |
                       "cosineannealing" | "cosineannealingwarmup"
        optimizer: The optimizer
        **kwargs: Additional arguments (warm_up_steps, training_steps, lr_end, num_cycles)
    """

    def get_warmup_lambda(warm_up_steps, training_steps):
        def lr_lambda(steps):
            if steps < warm_up_steps:
                return (steps + 1) / warm_up_steps
            else:
                return (training_steps - steps) / (training_steps - warm_up_steps)
        return lr_lambda

    def get_warmup_cosine_lambda(warm_up_steps, training_steps, lr_end):
        def lr_lambda(steps):
            if steps < warm_up_steps:
                return (steps + 1) / warm_up_steps
            else:
                progress = (steps - warm_up_steps) / (training_steps - warm_up_steps)
                return lr_end + 0.5 * (1 - lr_end) * (1 + math.cos(math.pi * progress))
        return lr_lambda

    if scheduler_name is None or scheduler_name.lower() == "constant":
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda steps: 1.0)

    elif scheduler_name.lower() == "constantwithwarmup":
        warm_up_steps = kwargs.get("warm_up_steps", 500)
        return lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda steps: min(1.0, (steps + 1) / warm_up_steps),
        )

    elif scheduler_name.lower() == "linearwarmupdecay":
        warm_up_steps = kwargs.get("warm_up_steps", 0)
        training_steps = kwargs.get("training_steps")
        lr_lambda = get_warmup_lambda(warm_up_steps, training_steps)
        return lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif scheduler_name.lower() == "cosineannealing":
        training_steps = kwargs.get("training_steps")
        eta_min = kwargs.get("lr_end", 0)
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_steps, eta_min=eta_min)

    elif scheduler_name.lower() == "cosineannealingwarmup":
        warm_up_steps = kwargs.get("warm_up_steps", 0)
        training_steps = kwargs.get("training_steps")
        eta_min = kwargs.get("lr_end", 0)
        lr_lambda = get_warmup_cosine_lambda(warm_up_steps, training_steps, eta_min)
        return lr_scheduler.LambdaLR(optimizer, lr_lambda)

    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
