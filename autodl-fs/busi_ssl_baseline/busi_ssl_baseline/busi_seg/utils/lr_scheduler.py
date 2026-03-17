"""Epoch-level learning-rate scheduler helpers.

Stage 1 supports only one scheduler recipe:
- linear warmup for the first `warmup_epochs`
- cosine decay afterwards down to `min_lr_ratio * base_lr`

Scheduler semantics are intentionally fixed to one clear policy:
- the scheduler is built before training begins
- PyTorch applies epoch index `0` during scheduler construction
- epoch 1 therefore starts with the warmup-adjusted learning rate
- trainers call `scheduler.step()` at each epoch end only to prepare the next
  epoch's learning rate
"""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


SUPPORTED_SCHEDULER_NAME = "cosine_with_linear_warmup"


def _cosine_with_linear_warmup_multiplier(
    epoch_index: int,
    *,
    total_epochs: int,
    warmup_epochs: int,
    min_lr_ratio: float,
) -> float:
    """Return the LR multiplier for a zero-based training epoch index."""

    safe_total_epochs = max(int(total_epochs), 1)
    safe_warmup_epochs = max(0, min(int(warmup_epochs), safe_total_epochs))
    safe_min_lr_ratio = float(min_lr_ratio)

    if safe_total_epochs == 1:
        return 1.0

    if safe_warmup_epochs > 0 and epoch_index < safe_warmup_epochs:
        return float(epoch_index + 1) / float(safe_warmup_epochs)

    if safe_warmup_epochs >= safe_total_epochs:
        return 1.0

    decay_epochs = safe_total_epochs - safe_warmup_epochs
    if decay_epochs <= 1:
        return safe_min_lr_ratio

    progress = float(epoch_index - safe_warmup_epochs) / float(decay_epochs - 1)
    clamped_progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * clamped_progress))
    return safe_min_lr_ratio + (1.0 - safe_min_lr_ratio) * cosine


def build_scheduler(
    optimizer: Optimizer,
    *,
    total_epochs: int,
    scheduler_config: dict | None,
) -> LambdaLR | None:
    """Build the minimal epoch-level scheduler or return None.

    Expected config shape:

    ```yaml
    scheduler:
      enabled: true
      name: cosine_with_linear_warmup
      warmup_epochs: 5
      min_lr_ratio: 0.01
    ```

    Returned scheduler semantics:
    - epoch index `0` maps to training epoch 1
    - epoch 1 uses the warmup-adjusted LR immediately after scheduler creation
    - later `scheduler.step()` calls should happen at epoch end to prepare the
      next epoch LR
    """

    if scheduler_config is None or not bool(scheduler_config.get("enabled", False)):
        return None

    scheduler_name = str(scheduler_config["name"]).lower()
    if scheduler_name != SUPPORTED_SCHEDULER_NAME:
        raise ValueError(
            "Stage 1 supports only scheduler.name='cosine_with_linear_warmup', "
            f"got: {scheduler_config['name']}"
        )

    warmup_epochs = int(scheduler_config.get("warmup_epochs", 5))
    min_lr_ratio = float(scheduler_config.get("min_lr_ratio", 0.01))

    if warmup_epochs < 0:
        raise ValueError(f"warmup_epochs must be >= 0, got {warmup_epochs}.")
    if not 0.0 < min_lr_ratio <= 1.0:
        raise ValueError(f"min_lr_ratio must be within (0, 1], got {min_lr_ratio}.")
    if int(total_epochs) <= 0:
        raise ValueError(f"total_epochs must be > 0, got {total_epochs}.")

    return LambdaLR(
        optimizer,
        lr_lambda=lambda epoch_index: _cosine_with_linear_warmup_multiplier(
            epoch_index,
            total_epochs=total_epochs,
            warmup_epochs=warmup_epochs,
            min_lr_ratio=min_lr_ratio,
        ),
    )
