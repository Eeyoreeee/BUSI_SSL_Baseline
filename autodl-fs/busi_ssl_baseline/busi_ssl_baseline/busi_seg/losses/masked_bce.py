"""Masked BCEWithLogits loss for fixed-threshold pseudo-label training."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MaskedBCEWithLogitsLoss(nn.Module):
    """Apply BCEWithLogits only on pixels selected by `selected_mask`.

    Expected tensor shapes are `N x 1 x H x W` for logits, pseudo labels, and mask.
    When `selected_mask` contains no positive entries, the returned loss is a stable
    scalar zero on the same device/dtype family as the logits.
    """

    def forward(
        self,
        logits: torch.Tensor,
        pseudo: torch.Tensor,
        selected_mask: torch.Tensor,
    ) -> torch.Tensor:
        if logits.shape != pseudo.shape or logits.shape != selected_mask.shape:
            raise ValueError(
                "logits, pseudo, and selected_mask must have the same shape, "
                f"got {tuple(logits.shape)}, {tuple(pseudo.shape)}, "
                f"{tuple(selected_mask.shape)}."
            )

        mask = selected_mask.to(dtype=logits.dtype)
        selected_count = mask.sum()
        if float(selected_count.item()) <= 0.0:
            return logits.new_zeros(())

        loss_map = F.binary_cross_entropy_with_logits(
            logits,
            pseudo.float(),
            reduction="none",
        )
        return (loss_map * mask).sum() / selected_count


def build_masked_unlabeled_loss(config: dict) -> nn.Module:
    """Build the minimal unlabeled loss from `ssl.unlabeled_loss.name`."""

    ssl_cfg = config["ssl"]
    unlabeled_cfg = ssl_cfg.get("unlabeled_loss", {})
    loss_name = str(unlabeled_cfg.get("name", "masked_bce")).lower()

    if loss_name != "masked_bce":
        raise ValueError(
            "Step 4 supports only ssl.unlabeled_loss.name='masked_bce', "
            f"got: {loss_name}"
        )

    return MaskedBCEWithLogitsLoss()
