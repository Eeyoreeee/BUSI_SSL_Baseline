"""Runnable entrypoint for pure supervised BUSI segmentation training."""

from __future__ import annotations

import argparse
from pathlib import Path
import random

import numpy as np
import torch

from busi_seg.config import dump_config, load_config
from busi_seg.data.loaders import (
    build_supervised_eval_loader,
    build_supervised_train_loader,
)
from busi_seg.engine.checkpoint import CheckpointManager
from busi_seg.engine.evaluator import SegmentationEvaluator
from busi_seg.engine.trainer_sup import SupervisedTrainer
from busi_seg.logging.logger import build_experiment_logger
from busi_seg.losses.supervised_loss import build_supervised_loss
from busi_seg.models.builder import build_model
from busi_seg.utils.lr_scheduler import build_scheduler


CONFIG_DUMP_FILENAME = "config_dump.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the BUSI supervised baseline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/sup_full.yaml"),
        help="Path to a supervised experiment config.",
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


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
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
            f"Stage 1 supports only optimizer.name='adamw', got: {optimizer_cfg['name']}"
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


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    output_dir = resolve_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dump_path = output_dir / CONFIG_DUMP_FILENAME
    dump_config(config, config_dump_path)

    seed = int(config["train"]["seed"])
    set_seed(seed)
    device = resolve_device(args.device)

    logger = build_experiment_logger(config, output_dir=output_dir)
    try:
        logger.info(f"Loaded config from {Path(config['_config_path']).resolve()}.")
        logger.info(f"Config dump written to {config_dump_path}.")
        logger.info(f"Experiment outputs will be written under {output_dir}.")
        logger.info(f"Using device {device} with seed={seed}.")

        train_loader = build_supervised_train_loader(config)
        val_loader = build_supervised_eval_loader(config, split_key="val_split")
        logger.info(
            f"Loader sizes | train_batches={len(train_loader)} | val_batches={len(val_loader)}."
        )

        model = build_model(config)
        optimizer = build_optimizer(config, model)
        scheduler = build_scheduler(
            optimizer,
            total_epochs=int(config["train"]["epochs"]),
            scheduler_config=config.get("scheduler"),
        )
        criterion = build_supervised_loss(config)
        evaluator = SegmentationEvaluator.from_config(config)
        if scheduler is None:
            logger.info("Epoch-level LR scheduler is disabled.")
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
        trainer = SupervisedTrainer(
            config=config,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            train_loader=train_loader,
            val_loader=val_loader,
            evaluator=evaluator,
            logger=logger,
            checkpoint_manager=checkpoint_manager,
            device=device,
        )
        trainer.fit()

        test_loader = build_supervised_eval_loader(config, split_key="test_split")
        checkpoint_to_test = (
            checkpoint_manager.best_path
            if checkpoint_manager.best_path.is_file()
            else checkpoint_manager.last_path
        )
        checkpoint = load_checkpoint_weights(trainer.model, checkpoint_to_test, device)
        logger.info(
            "Loaded checkpoint for test evaluation: "
            f"{checkpoint_to_test} "
            f"(epoch={checkpoint['epoch']}, best_metric={checkpoint.get('best_metric')})."
        )
        test_metrics = evaluator.evaluate(
            model=trainer.model,
            loader=test_loader,
            device=device,
            criterion=criterion,
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
