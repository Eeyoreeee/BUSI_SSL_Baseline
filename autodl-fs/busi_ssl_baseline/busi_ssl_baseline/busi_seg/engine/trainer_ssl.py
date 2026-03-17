"""Minimal teacher-student SSL trainer for the BUSI baseline."""

from __future__ import annotations

from contextlib import nullcontext
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from busi_seg.analysis.stats_collector import StatsCollector
from busi_seg.logging.logger import ExperimentLogger
from busi_seg.ssl.ema import update_ema
from busi_seg.ssl.ramps import linear_warmup_weight

from .checkpoint import CheckpointManager
from .evaluator import SegmentationEvaluator


def _infinite_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


class SSLTrainer:
    """Minimal SSL trainer with EMA teacher, fixed-threshold pseudo labels, and AMP."""

    def __init__(
        self,
        *,
        config: dict,
        student_model: nn.Module,
        teacher_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        supervised_criterion: nn.Module,
        unlabeled_criterion: nn.Module,
        labeled_loader: DataLoader,
        unlabeled_loader: DataLoader,
        val_loader: DataLoader | None,
        evaluator: SegmentationEvaluator,
        logger: ExperimentLogger,
        checkpoint_manager: CheckpointManager,
        pseudo_labeler,
        device: torch.device,
        stats_collector: StatsCollector | None = None,
    ) -> None:
        self.config = config
        self.student_model = student_model.to(device)
        self.teacher_model = teacher_model.to(device)
        self.optimizer = optimizer
        self.supervised_criterion = supervised_criterion
        self.unlabeled_criterion = unlabeled_criterion
        self.labeled_loader = labeled_loader
        self.unlabeled_loader = unlabeled_loader
        self.val_loader = val_loader
        self.evaluator = evaluator
        self.logger = logger
        self.checkpoint_manager = checkpoint_manager
        self.pseudo_labeler = pseudo_labeler
        self.device = device
        self.stats_collector = stats_collector

        train_cfg = config["train"]
        eval_cfg = config["eval"]
        ssl_cfg = config["ssl"]
        self.epochs = int(train_cfg["epochs"])
        self.log_interval = max(1, int(train_cfg["log_interval"]))
        self.val_interval = max(1, int(train_cfg["val_interval"]))
        self.grad_clip_norm = train_cfg["grad_clip_norm"]
        self.primary_metric = str(eval_cfg["primary_metric"])
        self.amp_enabled = bool(train_cfg["amp"] and device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
        self.global_step = 0
        self.best_metric: float | None = None
        self.lambda_u_target = float(ssl_cfg["lambda_u"])
        self.lambda_u_warmup_epochs = int(ssl_cfg["lambda_u_warmup_epochs"])
        self.ema_decay = float(ssl_cfg["ema_decay"])

    def fit(self) -> dict[str, float | None]:
        if len(self.labeled_loader) == 0:
            raise ValueError("Cannot run SSL training with an empty labeled dataloader.")
        if len(self.unlabeled_loader) == 0:
            raise ValueError("Cannot run SSL training with an empty unlabeled dataloader.")

        self.logger.info(
            f"Starting minimal SSL training for {self.epochs} epochs on {self.device}."
        )
        self.logger.info(
            f"AMP enabled: {self.amp_enabled}; primary metric: {self.primary_metric}; "
            f"tau={self.pseudo_labeler.tau:.3f}; ema_decay={self.ema_decay:.4f}."
        )
        if self.stats_collector is not None:
            self.logger.info(
                "Detached analysis logging is enabled for the unlabeled teacher outputs."
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
                model=self.student_model,
                optimizer=self.optimizer,
                epoch=epoch,
                best_metric=self.best_metric,
                scaler=self.scaler,
                extra_state=self._checkpoint_extra_state(),
            )
            self.logger.info(f"Saved last checkpoint to {last_path}.")

            if val_metrics is not None:
                metric_value = val_metrics.get(self.primary_metric)
                if metric_value is None:
                    raise KeyError(
                        f"Primary metric '{self.primary_metric}' missing from evaluator output."
                    )
                improved, best_metric, best_path = self.checkpoint_manager.maybe_save_best(
                    model=self.student_model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    metric_value=float(metric_value),
                    best_metric=self.best_metric,
                    scaler=self.scaler,
                    extra_state=self._checkpoint_extra_state(),
                )
                if improved:
                    self.best_metric = best_metric
                    self.logger.info(
                        f"Saved new best checkpoint to {best_path} "
                        f"with {self.primary_metric}={best_metric:.4f}."
                    )

        if self.best_metric is None and self.val_loader is not None:
            val_metrics = self._run_validation(self.epochs)
            metric_value = val_metrics.get(self.primary_metric)
            improved, best_metric, best_path = self.checkpoint_manager.maybe_save_best(
                model=self.student_model,
                optimizer=self.optimizer,
                epoch=self.epochs,
                metric_value=None if metric_value is None else float(metric_value),
                best_metric=self.best_metric,
                scaler=self.scaler,
                extra_state=self._checkpoint_extra_state(),
            )
            if improved:
                self.best_metric = best_metric
                self.logger.info(
                    f"Saved fallback best checkpoint to {best_path} "
                    f"with {self.primary_metric}={best_metric:.4f}."
                )

        if self.best_metric is None:
            self.logger.info("SSL training finished without a validation-derived best metric.")
        else:
            self.logger.info(
                f"SSL training finished with best {self.primary_metric}={self.best_metric:.4f}."
            )

        return {"best_metric": self.best_metric}

    def _checkpoint_extra_state(self) -> dict[str, object]:
        return {
            "teacher_state_dict": self.teacher_model.state_dict(),
            "ema_decay": self.ema_decay,
            "tau": self.pseudo_labeler.tau,
        }

    def _train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.student_model.train()
        self.teacher_model.eval()

        if self.stats_collector is not None:
            self.stats_collector.reset()

        lambda_u = linear_warmup_weight(
            current_epoch=epoch,
            warmup_epochs=self.lambda_u_warmup_epochs,
            target_weight=self.lambda_u_target,
        )
        num_steps = max(len(self.labeled_loader), len(self.unlabeled_loader))
        labeled_iter = _infinite_loader(self.labeled_loader)
        unlabeled_iter = _infinite_loader(self.unlabeled_loader)

        running = {
            "loss": 0.0,
            "loss_sup": 0.0,
            "loss_unsup": 0.0,
            "lambda_u": 0.0,
            "selected_ratio_batch": 0.0,
            "selected_count_batch": 0.0,
        }
        epoch_start = time.time()
        last_lr = float(self.optimizer.param_groups[0]["lr"])

        for batch_index in range(1, num_steps + 1):
            labeled_images, labeled_masks, _ = next(labeled_iter)
            weak_images, strong_images, _ = next(unlabeled_iter)

            labeled_images = labeled_images.to(
                self.device, non_blocking=self.device.type == "cuda"
            )
            labeled_masks = labeled_masks.to(
                self.device, non_blocking=self.device.type == "cuda"
            )
            weak_images = weak_images.to(
                self.device, non_blocking=self.device.type == "cuda"
            )
            strong_images = strong_images.to(
                self.device, non_blocking=self.device.type == "cuda"
            )

            self.optimizer.zero_grad(set_to_none=True)
            autocast_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if self.amp_enabled
                else nullcontext()
            )

            with torch.no_grad():
                with autocast_context:
                    teacher_logits = self.teacher_model(weak_images)
                pseudo_outputs = self.pseudo_labeler(teacher_logits, from_logits=True)

            if self.stats_collector is not None:
                self.stats_collector.update(
                    pseudo_outputs["prob"].detach(),
                    pseudo_outputs["pseudo"].detach(),
                    pseudo_outputs["selected_mask"].detach(),
                )

            with autocast_context:
                labeled_logits = self.student_model(labeled_images)
                strong_logits = self.student_model(strong_images)
                loss_sup = self.supervised_criterion(labeled_logits, labeled_masks)
                loss_unsup = self.unlabeled_criterion(
                    strong_logits,
                    pseudo_outputs["pseudo"],
                    pseudo_outputs["selected_mask"],
                )
                total_loss = loss_sup + lambda_u * loss_unsup

            if self.amp_enabled:
                self.scaler.scale(total_loss).backward()
                if self.grad_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.student_model.parameters(),
                        max_norm=float(self.grad_clip_norm),
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                if self.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.student_model.parameters(),
                        max_norm=float(self.grad_clip_norm),
                    )
                self.optimizer.step()

            update_ema(
                student=self.student_model,
                teacher=self.teacher_model,
                decay=self.ema_decay,
            )

            selected_mask = pseudo_outputs["selected_mask"]
            selected_ratio = float(selected_mask.float().mean().item())
            selected_count = float(selected_mask.sum().item())

            running["loss"] += float(total_loss.item())
            running["loss_sup"] += float(loss_sup.item())
            running["loss_unsup"] += float(loss_unsup.item())
            running["lambda_u"] += lambda_u
            running["selected_ratio_batch"] += selected_ratio
            running["selected_count_batch"] += selected_count

            self.global_step += 1
            last_lr = float(self.optimizer.param_groups[0]["lr"])

            if batch_index % self.log_interval == 0 or batch_index == num_steps:
                denom = float(batch_index)
                self.logger.info(
                    f"ssl_train_iter | epoch={epoch}/{self.epochs} "
                    f"| batch={batch_index}/{num_steps} "
                    f"| step={self.global_step} "
                    f"| loss={running['loss'] / denom:.4f} "
                    f"| loss_sup={running['loss_sup'] / denom:.4f} "
                    f"| loss_unsup={running['loss_unsup'] / denom:.4f} "
                    f"| lambda_u={running['lambda_u'] / denom:.4f} "
                    f"| selected_ratio={running['selected_ratio_batch'] / denom:.4f} "
                    f"| lr={last_lr:.6g}"
                )

        epoch_seconds = time.time() - epoch_start
        self.logger.info(f"ssl_epoch={epoch} finished in {epoch_seconds:.1f}s.")

        denom = float(num_steps)
        metrics = {
            "loss": running["loss"] / denom,
            "loss_sup": running["loss_sup"] / denom,
            "loss_unsup": running["loss_unsup"] / denom,
            "lambda_u": running["lambda_u"] / denom,
            "selected_ratio_batch": running["selected_ratio_batch"] / denom,
            "selected_count_batch": running["selected_count_batch"] / denom,
            "lr": last_lr,
        }
        if self.stats_collector is not None:
            analysis_metrics = self.stats_collector.compute()
            self.logger.info(
                "ssl_analysis | "
                f"epoch={epoch} | "
                f"selected_ratio={analysis_metrics['selected_ratio']:.4f} | "
                f"pre_filter_fg_ratio={analysis_metrics['pre_filter_fg_ratio']:.4f} | "
                f"teacher_entropy_selected={analysis_metrics['teacher_entropy_selected']:.4f} | "
                f"teacher_entropy_rejected={analysis_metrics['teacher_entropy_rejected']:.4f}"
            )
            metrics.update(analysis_metrics)
            self.stats_collector.reset()
        return metrics

    def _run_validation(self, epoch: int) -> dict[str, float]:
        metrics = self.evaluator.evaluate(
            model=self.student_model,
            loader=self.val_loader,
            device=self.device,
            criterion=self.supervised_criterion,
            amp=self.amp_enabled,
        )
        self.logger.log_metrics(
            split="val",
            epoch=epoch,
            step=self.global_step,
            metrics=metrics,
        )
        return metrics
