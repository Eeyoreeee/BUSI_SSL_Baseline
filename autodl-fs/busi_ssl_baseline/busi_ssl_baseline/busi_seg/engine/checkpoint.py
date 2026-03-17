"""Minimal checkpoint saving for stage-1 training.

Checkpoints are stored under `outputs/<exp_name>/checkpoints/`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


CHECKPOINT_DIRNAME = "checkpoints"


class CheckpointManager:
    """Save `best.pth` and `last.pth` with minimal reproducibility metadata."""

    def __init__(
        self,
        *,
        output_dir: str | Path,
        primary_metric_name: str,
        resolved_config_path: str | Path,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / CHECKPOINT_DIRNAME
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.primary_metric_name = primary_metric_name
        self.resolved_config_path = str(Path(resolved_config_path).resolve())
        self.best_path = self.checkpoint_dir / "best.pth"
        self.last_path = self.checkpoint_dir / "last.pth"

    def _build_payload(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        best_metric: float | None,
        scaler: torch.cuda.amp.GradScaler | None = None,
        current_metric: float | None = None,
        extra_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": int(epoch),
            "best_metric": None if best_metric is None else float(best_metric),
            "primary_metric_name": self.primary_metric_name,
            "current_metric": None if current_metric is None else float(current_metric),
            "resolved_config_path": self.resolved_config_path,
        }
        if scaler is not None:
            payload["scaler_state_dict"] = scaler.state_dict()
        if extra_state:
            payload.update(extra_state)
        return payload

    def save_last(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        best_metric: float | None,
        scaler: torch.cuda.amp.GradScaler | None = None,
        extra_state: dict[str, Any] | None = None,
    ) -> Path:
        payload = self._build_payload(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_metric=best_metric,
            scaler=scaler,
            extra_state=extra_state,
        )
        torch.save(payload, self.last_path)
        return self.last_path

    def maybe_save_best(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metric_value: float | None,
        best_metric: float | None,
        scaler: torch.cuda.amp.GradScaler | None = None,
        extra_state: dict[str, Any] | None = None,
    ) -> tuple[bool, float | None, Path]:
        if metric_value is None:
            return False, best_metric, self.best_path

        is_better = best_metric is None or metric_value > best_metric
        if not is_better:
            return False, best_metric, self.best_path

        payload = self._build_payload(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_metric=metric_value,
            scaler=scaler,
            current_metric=metric_value,
            extra_state=extra_state,
        )
        torch.save(payload, self.best_path)
        return True, metric_value, self.best_path
