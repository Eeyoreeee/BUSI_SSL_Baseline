"""Pure supervised trainer for the stage-1 BUSI baseline."""

from __future__ import annotations

from contextlib import nullcontext
import time

import torch
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from busi_seg.logging.logger import ExperimentLogger

from .checkpoint import CheckpointManager
from .evaluator import SegmentationEvaluator


class SupervisedTrainer:
    """Minimal supervised trainer with AMP, val evaluation, and checkpoints."""

    def __init__(
        self,
        *,
        config: dict,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: LRScheduler | None,
        criterion: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        evaluator: SegmentationEvaluator,
        logger: ExperimentLogger,
        checkpoint_manager: CheckpointManager,
        device: torch.device,
    ) -> None:
        self.config = config
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.evaluator = evaluator
        self.logger = logger
        self.checkpoint_manager = checkpoint_manager
        self.device = device

        train_cfg = config["train"]
        eval_cfg = config["eval"]
        self.epochs = int(train_cfg["epochs"])
        self.log_interval = max(1, int(train_cfg["log_interval"]))
        self.val_interval = max(1, int(train_cfg["val_interval"]))
        self.grad_clip_norm = train_cfg["grad_clip_norm"]
        self.primary_metric = str(eval_cfg["primary_metric"])
        self.amp_enabled = bool(train_cfg["amp"] and device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
        self.global_step = 0
        self.best_metric: float | None = None

    def fit(self) -> dict[str, float | None]:
        if len(self.train_loader) == 0:
            raise ValueError("Cannot train with an empty train dataloader.")

        self.logger.info(
            f"Starting supervised training for {self.epochs} epochs on {self.device}."
        )
        self.logger.info(
            f"AMP enabled: {self.amp_enabled}; primary metric: {self.primary_metric}."
        )
        if self.scheduler is not None:
            self.logger.info(
                f"scheduler | epoch=1 | current_epoch_lr={self.optimizer.param_groups[0]['lr']:.6g}"
            )

        for epoch in range(1, self.epochs + 1):
            train_metrics = self._train_one_epoch(epoch)
            self.logger.log_metrics(
                split="train",
                epoch=epoch,
                step=self.global_step,
                metrics=train_metrics,
            )

            val_metrics = None
            if self.val_loader is not None and epoch % self.val_interval == 0:
                val_metrics = self._run_validation(epoch)

            last_path = self.checkpoint_manager.save_last(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                best_metric=self.best_metric,
                scaler=self.scaler,
            )
            self.logger.info(f"Saved last checkpoint to {last_path}.")

            if val_metrics is not None:
                metric_value = val_metrics.get(self.primary_metric)
                if metric_value is None:
                    raise KeyError(
                        f"Primary metric '{self.primary_metric}' missing from evaluator output."
                    )
                improved, best_metric, best_path = self.checkpoint_manager.maybe_save_best(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    metric_value=float(metric_value),
                    best_metric=self.best_metric,
                    scaler=self.scaler,
                )
                if improved:
                    self.best_metric = best_metric
                    self.logger.info(
                        f"Saved new best checkpoint to {best_path} "
                        f"with {self.primary_metric}={best_metric:.4f}."
                    )

            if self.scheduler is not None and epoch < self.epochs:
                self.scheduler.step()
                self.logger.info(
                    f"scheduler | next_epoch={epoch + 1} | next_epoch_lr={self.optimizer.param_groups[0]['lr']:.6g}"
                )

        if self.best_metric is None and self.val_loader is not None:
            val_metrics = self._run_validation(self.epochs)
            metric_value = val_metrics.get(self.primary_metric)
            improved, best_metric, best_path = self.checkpoint_manager.maybe_save_best(
                model=self.model,
                optimizer=self.optimizer,
                epoch=self.epochs,
                metric_value=None if metric_value is None else float(metric_value),
                best_metric=self.best_metric,
                scaler=self.scaler,
            )
            if improved:
                self.best_metric = best_metric
                self.logger.info(
                    f"Saved fallback best checkpoint to {best_path} "
                    f"with {self.primary_metric}={best_metric:.4f}."
                )

        if self.best_metric is None:
            self.logger.info("Training finished without a validation-derived best metric.")
        else:
            self.logger.info(
                f"Training finished with best {self.primary_metric}={self.best_metric:.4f}."
            )

        return {"best_metric": self.best_metric}

    def _train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()

        running_loss = 0.0
        sample_count = 0
        epoch_start = time.time()
        last_lr = float(self.optimizer.param_groups[0]["lr"])

        for batch_index, (images, masks, _) in enumerate(self.train_loader, start=1):
            images = images.to(self.device, non_blocking=self.device.type == "cuda")
            masks = masks.to(self.device, non_blocking=self.device.type == "cuda")

            self.optimizer.zero_grad(set_to_none=True)
            autocast_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if self.amp_enabled
                else nullcontext()
            )
            with autocast_context:
                logits = self.model(images)
                loss = self.criterion(logits, masks)

            if self.amp_enabled:
                self.scaler.scale(loss).backward()
                if self.grad_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=float(self.grad_clip_norm),
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=float(self.grad_clip_norm),
                    )
                self.optimizer.step()

            batch_size = int(images.shape[0])
            running_loss += float(loss.item()) * batch_size
            sample_count += batch_size
            self.global_step += 1
            last_lr = float(self.optimizer.param_groups[0]["lr"])

            if batch_index % self.log_interval == 0 or batch_index == len(self.train_loader):
                avg_loss = running_loss / max(sample_count, 1)
                self.logger.info(
                    f"train_iter | epoch={epoch}/{self.epochs} "
                    f"| batch={batch_index}/{len(self.train_loader)} "
                    f"| step={self.global_step} "
                    f"| loss={avg_loss:.4f} "
                    f"| lr={last_lr:.6g}"
                )

        epoch_seconds = time.time() - epoch_start
        average_loss = running_loss / max(sample_count, 1)
        self.logger.info(f"epoch={epoch} finished in {epoch_seconds:.1f}s.")
        return {
            "loss": average_loss,
            "lr": last_lr,
        }

    def _run_validation(self, epoch: int) -> dict[str, float]:
        metrics = self.evaluator.evaluate(
            model=self.model,
            loader=self.val_loader,
            device=self.device,
            criterion=self.criterion,
            amp=self.amp_enabled,
        )
        self.logger.log_metrics(
            split="val",
            epoch=epoch,
            step=self.global_step,
            metrics=metrics,
        )
        return metrics
