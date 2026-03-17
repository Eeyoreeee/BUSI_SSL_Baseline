"""Validation and test evaluator for binary BUSI segmentation."""

from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn
from torch.utils.data import DataLoader


def _safe_divide(numerator: float, denominator: float, eps: float = 1e-7) -> float:
    return float(numerator / max(denominator, eps))


class SegmentationEvaluator:
    """Evaluate a segmentation model independently from the trainer."""

    def __init__(self, *, threshold: float) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be within [0, 1], got {threshold}.")
        self.threshold = float(threshold)

    @classmethod
    def from_config(cls, config: dict) -> "SegmentationEvaluator":
        return cls(threshold=float(config["eval"]["threshold"]))

    def evaluate(
        self,
        *,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        criterion: nn.Module | None = None,
        amp: bool = False,
    ) -> dict[str, float]:
        if len(loader) == 0:
            raise ValueError("Cannot evaluate with an empty dataloader.")

        was_training = model.training
        model.eval()

        tp = 0.0
        fp = 0.0
        fn = 0.0
        tn = 0.0
        loss_sum = 0.0
        sample_count = 0
        use_amp = bool(amp and device.type == "cuda")

        with torch.no_grad():
            for images, masks, _ in loader:
                images = images.to(device, non_blocking=device.type == "cuda")
                masks = masks.to(device, non_blocking=device.type == "cuda")

                autocast_context = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if use_amp
                    else nullcontext()
                )
                with autocast_context:
                    logits = model(images)
                    batch_loss = criterion(logits, masks) if criterion is not None else None

                probs = logits.float().sigmoid()
                preds = probs >= self.threshold
                target = masks >= 0.5

                tp += torch.logical_and(preds, target).sum().item()
                fp += torch.logical_and(preds, torch.logical_not(target)).sum().item()
                fn += torch.logical_and(torch.logical_not(preds), target).sum().item()
                tn += torch.logical_and(
                    torch.logical_not(preds), torch.logical_not(target)
                ).sum().item()

                batch_size = int(images.shape[0])
                sample_count += batch_size
                if batch_loss is not None:
                    loss_sum += float(batch_loss.item()) * batch_size

        if was_training:
            model.train()

        metrics = {
            "dice": _safe_divide(2.0 * tp, 2.0 * tp + fp + fn),
            "iou": _safe_divide(tp, tp + fp + fn),
            "precision": _safe_divide(tp, tp + fp),
            "recall": _safe_divide(tp, tp + fn),
            "specificity": _safe_divide(tn, tn + fp),
        }
        if criterion is not None:
            metrics["loss"] = _safe_divide(loss_sum, float(sample_count))
        return metrics
