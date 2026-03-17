"""Minimal experiment logger with a fixed output layout.

The baseline writes:
- `train.log`
- `metrics.csv`
- optional `tb/`
"""

from __future__ import annotations

import csv
from datetime import datetime
import logging as py_logging
from pathlib import Path
import sys


TRAIN_LOG_FILENAME = "train.log"
METRICS_FILENAME = "metrics.csv"
TENSORBOARD_DIRNAME = "tb"

CSV_FIELDS = [
    "timestamp",
    "split",
    "epoch",
    "step",
    "lr",
    "loss",
    "loss_sup",
    "loss_unsup",
    "lambda_u",
    "selected_ratio_batch",
    "selected_count_batch",
    "total_count",
    "selected_count",
    "selected_ratio",
    "pre_filter_fg_ratio",
    "post_filter_fg_ratio",
    "selected_fg_purity",
    "fg_keep_rate",
    "bg_keep_rate",
    "teacher_entropy_selected",
    "teacher_entropy_rejected",
    "dice",
    "iou",
    "precision",
    "recall",
    "specificity",
]


class ExperimentLogger:
    """Small logger wrapper for file/stdout text logs and flat metric rows."""

    def __init__(
        self,
        *,
        output_dir: str | Path,
        use_tensorboard: bool,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.log_path = self.output_dir / TRAIN_LOG_FILENAME
        self.metrics_path = self.output_dir / METRICS_FILENAME
        self.tensorboard_writer = None

        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tensorboard_writer = SummaryWriter(
                    log_dir=str(self.output_dir / TENSORBOARD_DIRNAME)
                )
            except ImportError:
                self.tensorboard_writer = None

        logger_name = f"busi_seg.experiment.{self.output_dir.resolve()}"
        self.logger = py_logging.getLogger(logger_name)
        self.logger.setLevel(py_logging.INFO)
        self.logger.propagate = False
        for handler in list(self.logger.handlers):
            self.logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

        formatter = py_logging.Formatter(
            fmt="%(asctime)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.stream_handler = py_logging.StreamHandler(sys.stdout)
        self.stream_handler.setFormatter(formatter)
        self.file_handler = py_logging.FileHandler(self.log_path, encoding="utf-8")
        self.file_handler.setFormatter(formatter)
        self.logger.addHandler(self.stream_handler)
        self.logger.addHandler(self.file_handler)

        file_exists = self.metrics_path.exists()
        self.metrics_handle = self.metrics_path.open("a", encoding="utf-8", newline="")
        self.metrics_writer = csv.DictWriter(self.metrics_handle, fieldnames=CSV_FIELDS)
        if not file_exists:
            self.metrics_writer.writeheader()
            self.metrics_handle.flush()

    def info(self, message: str) -> None:
        self.logger.info(message)

    def log_metrics(
        self,
        *,
        split: str,
        epoch: int,
        step: int,
        metrics: dict[str, float],
    ) -> None:
        timestamp = datetime.now().isoformat(timespec="seconds")
        row = {field: "" for field in CSV_FIELDS}
        row["timestamp"] = timestamp
        row["split"] = split
        row["epoch"] = int(epoch)
        row["step"] = int(step)

        for key, value in metrics.items():
            if key in row and value is not None:
                row[key] = float(value)

        self.metrics_writer.writerow(row)
        self.metrics_handle.flush()

        scalar_parts: list[str] = []
        for key, value in metrics.items():
            if value is None:
                continue
            numeric_value = float(value)
            scalar_parts.append(f"{key}={numeric_value:.4f}")
            if self.tensorboard_writer is not None:
                self.tensorboard_writer.add_scalar(f"{split}/{key}", numeric_value, step)

        summary = ", ".join(scalar_parts) if scalar_parts else "no metrics"
        self.info(f"{split} | epoch={epoch} | step={step} | {summary}")

    def close(self) -> None:
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.flush()
            self.tensorboard_writer.close()

        self.metrics_handle.flush()
        self.metrics_handle.close()

        self.logger.removeHandler(self.file_handler)
        self.logger.removeHandler(self.stream_handler)
        self.file_handler.close()


def build_experiment_logger(config: dict, *, output_dir: str | Path) -> ExperimentLogger:
    """Build the minimal experiment logger from config."""

    logging_cfg = config["logging"]
    return ExperimentLogger(
        output_dir=output_dir,
        use_tensorboard=bool(logging_cfg["use_tensorboard"]),
    )
