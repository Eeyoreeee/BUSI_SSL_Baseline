"""Fixed-threshold pseudo-labeling for the minimal SSL baseline."""

from __future__ import annotations

import torch


class FixedThresholdPseudoLabeler:
    """Generate pseudo labels with a single global confidence threshold."""

    def __init__(self, *, tau: float) -> None:
        if not 0.0 <= tau <= 1.0:
            raise ValueError(f"tau must be within [0, 1], got {tau}.")
        self.tau = float(tau)

    @classmethod
    def from_config(cls, config: dict) -> "FixedThresholdPseudoLabeler":
        return cls(tau=float(config["ssl"]["tau"]))

    def __call__(
        self,
        prediction: torch.Tensor,
        *,
        from_logits: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Build pseudo-label tensors from teacher logits or probabilities."""

        prob = prediction.sigmoid() if from_logits else prediction
        prob = prob.float().clamp(0.0, 1.0)
        pseudo = (prob >= 0.5).to(dtype=torch.float32)
        conf = torch.maximum(prob, 1.0 - prob)
        selected_mask = conf >= self.tau
        return {
            "prob": prob,
            "pseudo": pseudo,
            "conf": conf,
            "selected_mask": selected_mask,
        }
