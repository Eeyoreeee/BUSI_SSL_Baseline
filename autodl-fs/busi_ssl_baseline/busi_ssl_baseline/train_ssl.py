"""Runnable entrypoint for minimal teacher-student BUSI SSL training."""

from __future__ import annotations

import argparse
from pathlib import Path
import random

import numpy as np
import torch

from busi_seg.analysis.stats_collector import StatsCollector
from busi_seg.config import dump_config, load_config
from busi_seg.data.loaders import build_ssl_eval_loader, build_ssl_train_loaders
from busi_seg.engine.checkpoint import CheckpointManager
from busi_seg.engine.evaluator import SegmentationEvaluator
from busi_seg.engine.trainer_ssl import SSLTrainer
from busi_seg.logging.logger import build_experiment_logger
from busi_seg.losses.masked_bce import build_masked_unlabeled_loss
from busi_seg.losses.supervised_loss import build_supervised_loss
from busi_seg.models.builder import build_model
from busi_seg.ssl.ema import copy_student_to_teacher
from busi_seg.ssl.pseudo_labeler import FixedThresholdPseudoLabeler
from busi_seg.utils.lr_scheduler import build_scheduler


CONFIG_DUMP_FILENAME = "config_dump.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the BUSI minimal SSL baseline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/ssl_tau095.yaml"),
        help="Path to an SSL experiment config.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override, for example 'cuda' or 'cpu'.",
    )
    return parser.parse_args()


def _project_root_from_config(config_path: str | Path) -> Path:
    resolved = Path(config_path).resolve()
    for parent in [resolved.parent, *resolved.parents]:
        if parent.name == "configs":
            return parent.parent
    return Path.cwd().resolve()


def resolve_output_dir(config: dict) -> Path:
    project_root = _project_root_from_config(config["_config_path"])
    output_root = Path(config["experiment"]["output_root"])
    if not output_root.is_absolute():
        output_root = project_root / output_root
    return (output_root / config["experiment"]["name"]).resolve()


def resolve_device(device_arg: str | None, config: dict) -> torch.device:
    config_device = config["train"].get("device")
    if device_arg:
        return torch.device(device_arg)
    if config_device:
        return torch.device(config_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_optimizer(config: dict, model: torch.nn.Module) -> torch.optim.Optimizer:
    optimizer_cfg = config["optimizer"]
    optimizer_name = str(optimizer_cfg["name"]).lower()
    if optimizer_name != "adamw":
        raise ValueError(
            f"Step 4 supports only optimizer.name='adamw', got: {optimizer_cfg['name']}"
        )

    return torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_cfg["lr"]),
        weight_decay=float(optimizer_cfg["weight_decay"]),
    )


def load_checkpoint_weights(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def build_stats_collector(config: dict) -> StatsCollector | None:
    analysis_cfg = config.get("analysis", {})
    if not bool(analysis_cfg.get("enabled", False)):
        return None
    return StatsCollector()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    output_dir = resolve_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dump_path = output_dir / CONFIG_DUMP_FILENAME
    dump_config(config, config_dump_path)

    seed = int(config["train"]["seed"])
    set_seed(seed)
    device = resolve_device(args.device, config)

    logger = build_experiment_logger(config, output_dir=output_dir)
    try:
        logger.info(f"Loaded config from {Path(config['_config_path']).resolve()}.")
        logger.info(f"Config dump written to {config_dump_path}.")
        logger.info(f"Experiment outputs will be written under {output_dir}.")
        logger.info(f"Using device {device} with seed={seed}.")

        labeled_loader, unlabeled_loader = build_ssl_train_loaders(config)
        val_loader = build_ssl_eval_loader(config, split_key="val_split")
        logger.info(
            "Loader sizes | "
            f"labeled_batches={len(labeled_loader)} | "
            f"unlabeled_batches={len(unlabeled_loader)} | "
            f"val_batches={len(val_loader)}."
        )

        student_model = build_model(config)
        teacher_model = build_model(config)
        copy_student_to_teacher(student_model, teacher_model)
        logger.info("Teacher initialized from student weights and frozen for EMA updates.")

        optimizer = build_optimizer(config, student_model)
        scheduler = build_scheduler(
            optimizer,
            total_epochs=int(config["train"]["epochs"]),
            scheduler_config=config.get("scheduler"),
        )
        supervised_criterion = build_supervised_loss(config)
        unlabeled_criterion = build_masked_unlabeled_loss(config)
        evaluator = SegmentationEvaluator.from_config(config)
        pseudo_labeler = FixedThresholdPseudoLabeler.from_config(config)
        stats_collector = build_stats_collector(config)
        if stats_collector is None:
            logger.info("Detached analysis logging is disabled.")
        if scheduler is None:
            logger.info("Epoch-level LR scheduler is disabled for SSL training.")
        else:
            logger.info(
                "Using epoch-level scheduler: "
                f"{config['scheduler']['name']} "
                f"(warmup_epochs={config['scheduler']['warmup_epochs']}, "
                f"min_lr_ratio={config['scheduler']['min_lr_ratio']}); "
                "epoch 1 starts from the warmup-adjusted learning rate."
            )

        checkpoint_manager = CheckpointManager(
            output_dir=output_dir,
            primary_metric_name=str(config["eval"]["primary_metric"]),
            resolved_config_path=config_dump_path,
        )
        trainer = SSLTrainer(
            config=config,
            student_model=student_model,
            teacher_model=teacher_model,
            optimizer=optimizer,
            scheduler=scheduler,
            supervised_criterion=supervised_criterion,
            unlabeled_criterion=unlabeled_criterion,
            labeled_loader=labeled_loader,
            unlabeled_loader=unlabeled_loader,
            val_loader=val_loader,
            evaluator=evaluator,
            logger=logger,
            checkpoint_manager=checkpoint_manager,
            pseudo_labeler=pseudo_labeler,
            device=device,
            stats_collector=stats_collector,
        )
        trainer.fit()

        test_loader = build_ssl_eval_loader(config, split_key="test_split")
        checkpoint_to_test = (
            checkpoint_manager.best_path
            if checkpoint_manager.best_path.is_file()
            else checkpoint_manager.last_path
        )
        checkpoint = load_checkpoint_weights(trainer.student_model, checkpoint_to_test, device)
        logger.info(
            "Loaded checkpoint for test evaluation: "
            f"{checkpoint_to_test} "
            f"(epoch={checkpoint['epoch']}, best_metric={checkpoint.get('best_metric')})."
        )
        test_metrics = evaluator.evaluate(
            model=trainer.student_model,
            loader=test_loader,
            device=device,
            criterion=supervised_criterion,
            amp=trainer.amp_enabled,
        )
        logger.log_metrics(
            split="test",
            epoch=int(checkpoint["epoch"]),
            step=trainer.global_step,
            metrics=test_metrics,
        )
    finally:
        logger.close()


if __name__ == "__main__":
    main()
