"""Binary soft Dice helpers for single-channel segmentation."""

from __future__ import annotations

import torch
from torch import nn


def binary_soft_dice_score(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    from_logits: bool = True,
    smooth: float = 1.0,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compute mean soft Dice score for `N x 1 x H x W` tensors.

    If `from_logits=True`, `prediction` is interpreted as raw logits and
    `sigmoid` is applied internally. Otherwise `prediction` must already be a
    probability map in `[0, 1]`.
    """

    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target must have the same shape, "
            f"got {tuple(prediction.shape)} vs {tuple(target.shape)}."
        )
    if smooth < 0.0:
        raise ValueError(f"smooth must be non-negative, got {smooth}.")
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}.")

    probs = prediction.sigmoid() if from_logits else prediction
    probs = probs.float().clamp(0.0, 1.0)
    target = target.float().clamp(0.0, 1.0)

    probs = probs.flatten(start_dim=1)
    target = target.flatten(start_dim=1)

    intersection = (probs * target).sum(dim=1)
    denominator = probs.sum(dim=1) + target.sum(dim=1)
    score = (2.0 * intersection + smooth) / (denominator + smooth).clamp_min(eps)
    return score.mean()


class BinarySoftDiceLoss(nn.Module):
    """Binary soft Dice loss with explicit logits/probabilities convention."""

    def __init__(
        self,
        *,
        from_logits: bool = True,
        smooth: float = 1.0,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        score = binary_soft_dice_score(
            prediction,
            target,
            from_logits=self.from_logits,
            smooth=self.smooth,
            eps=self.eps,
        )
        return 1.0 - score
