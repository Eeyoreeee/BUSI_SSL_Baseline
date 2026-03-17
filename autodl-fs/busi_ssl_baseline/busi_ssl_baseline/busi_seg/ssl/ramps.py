"""Warmup helpers for the minimal SSL baseline."""

from __future__ import annotations


def linear_warmup_weight(
    *,
    current_epoch: int,
    warmup_epochs: int,
    target_weight: float,
) -> float:
    """Linearly warm `target_weight` using a 1-based training epoch convention.

    `current_epoch` is expected to be the human-readable epoch index starting at 1.
    """

    if target_weight < 0.0:
        raise ValueError(f"target_weight must be non-negative, got {target_weight}.")
    if warmup_epochs <= 0:
        return float(target_weight)

    progress = min(max(current_epoch, 0), warmup_epochs) / float(warmup_epochs)
    return float(target_weight) * progress
