"""Supervised segmentation loss: BCEWithLogits plus soft Dice."""

from __future__ import annotations

import torch
from torch import nn

from .dice import BinarySoftDiceLoss


class BCEPlusDiceLoss(nn.Module):
    """Combine BCEWithLogits and Dice for binary segmentation logits."""

    def __init__(
        self,
        *,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        dice_smooth: float = 1.0,
        dice_eps: float = 1e-7,
    ) -> None:
        super().__init__()
        if bce_weight < 0.0 or dice_weight < 0.0:
            raise ValueError("bce_weight and dice_weight must be non-negative.")
        if bce_weight == 0.0 and dice_weight == 0.0:
            raise ValueError("At least one of bce_weight or dice_weight must be > 0.")

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = BinarySoftDiceLoss(
            from_logits=True,
            smooth=dice_smooth,
            eps=dice_eps,
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        total_loss = logits.new_tensor(0.0)

        if self.bce_weight > 0.0:
            total_loss = total_loss + self.bce_weight * self.bce(logits, target)
        if self.dice_weight > 0.0:
            total_loss = total_loss + self.dice_weight * self.dice(logits, target)

        return total_loss


def build_supervised_loss(config: dict) -> nn.Module:
    """Build the stage-1 supervised loss from config."""

    loss_cfg = config["loss"]
    supervised_name = str(loss_cfg["supervised_name"]).lower()
    if supervised_name != "bce_plus_dice":
        raise ValueError(
            "Stage 1 supports only loss.supervised_name='bce_plus_dice', "
            f"got: {loss_cfg['supervised_name']}"
        )

    return BCEPlusDiceLoss(
        bce_weight=float(loss_cfg["bce_weight"]),
        dice_weight=float(loss_cfg["dice_weight"]),
    )
