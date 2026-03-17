"""Epoch-level detached statistics for pseudo-label diagnostics.

This collector is analysis-only:
- it accepts only detached tensors
- it maintains running sums and counts
- it never returns signals used to change training decisions
"""

from __future__ import annotations

import torch


class StatsCollector:
    """Accumulate detached pseudo-label statistics without caching full batches."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total_count = 0.0
        self.selected_count = 0.0
        self.rejected_count = 0.0
        self.pseudo_fg_count = 0.0
        self.pseudo_bg_count = 0.0
        self.selected_fg_count = 0.0
        self.selected_bg_count = 0.0
        self.teacher_entropy_selected_sum = 0.0
        self.teacher_entropy_rejected_sum = 0.0

    def update(
        self,
        prob: torch.Tensor,
        pseudo: torch.Tensor,
        selected_mask: torch.Tensor,
    ) -> None:
        """Update running sums from detached tensors only.

        `prob`, `pseudo`, and `selected_mask` are expected to share shape
        `N x 1 x H x W`. The caller must pass detached tensors so analysis stays
        fully outside the training graph.
        """

        if prob.shape != pseudo.shape or prob.shape != selected_mask.shape:
            raise ValueError(
                "prob, pseudo, and selected_mask must have the same shape, "
                f"got {tuple(prob.shape)}, {tuple(pseudo.shape)}, "
                f"{tuple(selected_mask.shape)}."
            )
        if prob.requires_grad or pseudo.requires_grad or selected_mask.requires_grad:
            raise ValueError("StatsCollector.update expects detached tensors only.")

        safe_prob = prob.float().clamp(1e-6, 1.0 - 1e-6)
        pseudo_bool = pseudo >= 0.5
        selected_bool = selected_mask.bool()
        rejected_bool = torch.logical_not(selected_bool)
        entropy = -(
            safe_prob * torch.log(safe_prob)
            + (1.0 - safe_prob) * torch.log(1.0 - safe_prob)
        )

        total_count = float(safe_prob.numel())
        selected_count = float(selected_bool.sum().item())
        pseudo_fg_count = float(pseudo_bool.sum().item())
        pseudo_bg_count = total_count - pseudo_fg_count

        selected_fg_bool = torch.logical_and(selected_bool, pseudo_bool)
        selected_bg_bool = torch.logical_and(selected_bool, torch.logical_not(pseudo_bool))
        selected_fg_count = float(selected_fg_bool.sum().item())
        selected_bg_count = float(selected_bg_bool.sum().item())

        self.total_count += total_count
        self.selected_count += selected_count
        self.rejected_count += total_count - selected_count
        self.pseudo_fg_count += pseudo_fg_count
        self.pseudo_bg_count += pseudo_bg_count
        self.selected_fg_count += selected_fg_count
        self.selected_bg_count += selected_bg_count
        self.teacher_entropy_selected_sum += float(entropy[selected_bool].sum().item())
        self.teacher_entropy_rejected_sum += float(entropy[rejected_bool].sum().item())

    def compute(self) -> dict[str, float]:
        """Return stable epoch-level scalar diagnostics.

        Empty selected/rejected sets produce entropy 0. Empty denominators in
        ratios are guarded with `max(count, 1)` so the result stays finite.
        """

        total_count = self.total_count
        selected_count = self.selected_count
        pseudo_fg_count = self.pseudo_fg_count
        pseudo_bg_count = self.pseudo_bg_count
        selected_fg_count = self.selected_fg_count
        selected_bg_count = self.selected_bg_count

        return {
            "total_count": total_count,
            "selected_count": selected_count,
            "selected_ratio": selected_count / max(total_count, 1.0),
            "pre_filter_fg_ratio": pseudo_fg_count / max(total_count, 1.0),
            "post_filter_fg_ratio": selected_fg_count / max(total_count, 1.0),
            "selected_fg_purity": selected_fg_count / max(selected_count, 1.0),
            "fg_keep_rate": selected_fg_count / max(pseudo_fg_count, 1.0),
            "bg_keep_rate": selected_bg_count / max(pseudo_bg_count, 1.0),
            "teacher_entropy_selected": self.teacher_entropy_selected_sum
            / max(selected_count, 1.0),
            "teacher_entropy_rejected": self.teacher_entropy_rejected_sum
            / max(self.rejected_count, 1.0),
        }
