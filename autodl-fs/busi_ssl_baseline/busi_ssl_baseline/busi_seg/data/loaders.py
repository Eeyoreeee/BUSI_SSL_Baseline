"""Dataloader builders for supervised and minimal SSL experiments."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from .busi_dataset import BUSISegDataset
from .manifest import read_manifest, read_split_ids
from .ssl_dataset import BUSISSLDataset
from .transforms import SSLViewTransform, SupervisedSegTransform


def _project_root_from_config(config: dict) -> Path:
    config_path = Path(config["_config_path"]).resolve()
    for parent in [config_path.parent, *config_path.parents]:
        if parent.name == "configs":
            return parent.parent
    return Path.cwd()


def _resolve_data_paths(config: dict) -> dict[str, Path]:
    project_root = _project_root_from_config(config)
    data_cfg = config["data"]
    return {
        "dataset_root": (project_root / data_cfg["dataset_root"]).resolve(),
        "manifest_path": (project_root / data_cfg["manifest_path"]).resolve(),
        "merged_mask_root": (project_root / data_cfg["merged_mask_root"]).resolve(),
        "split_root": (project_root / data_cfg["split_root"]).resolve(),
    }


def _read_records_for_split(config: dict, split_filename: str):
    paths = _resolve_data_paths(config)
    manifest = read_manifest(paths["manifest_path"])
    split_ids = read_split_ids(paths["split_root"] / split_filename)
    return manifest.subset(split_ids), paths


def _build_supervised_transform(config: dict, *, is_train: bool) -> SupervisedSegTransform:
    data_cfg = config["data"]
    mean = tuple(data_cfg["image_mean"])
    std = tuple(data_cfg["image_std"])

    return SupervisedSegTransform(
        output_size=data_cfg["image_size"],
        mean=mean,
        std=std,
        augment=is_train,
        hflip_prob=data_cfg["train_hflip_prob"] if is_train else 0.0,
        vflip_prob=data_cfg["train_vflip_prob"] if is_train else 0.0,
        max_rotate_deg=data_cfg["train_max_rotate_deg"] if is_train else 0.0,
        scale_min=data_cfg["train_scale_min"] if is_train else 1.0,
        scale_max=data_cfg["train_scale_max"] if is_train else 1.0,
        crop_min_ratio=data_cfg["train_crop_min_ratio"] if is_train else 1.0,
        color_jitter_strength=data_cfg["train_color_jitter_strength"] if is_train else 0.0,
        blur_prob=data_cfg["train_blur_prob"] if is_train else 0.0,
        noise_std=data_cfg["train_noise_std"] if is_train else 0.0,
    )


def _build_ssl_transform(config: dict) -> SSLViewTransform:
    data_cfg = config["data"]
    ssl_cfg = config["ssl"]
    return SSLViewTransform(
        output_size=data_cfg["image_size"],
        mean=tuple(data_cfg["image_mean"]),
        std=tuple(data_cfg["image_std"]),
        hflip_prob=ssl_cfg["geometry_hflip_prob"],
        vflip_prob=ssl_cfg["geometry_vflip_prob"],
        max_rotate_deg=ssl_cfg["geometry_max_rotate_deg"],
        scale_min=ssl_cfg["geometry_scale_min"],
        scale_max=ssl_cfg["geometry_scale_max"],
        crop_min_ratio=ssl_cfg["geometry_crop_min_ratio"],
        weak_color_jitter_strength=ssl_cfg["weak_color_jitter_strength"],
        strong_color_jitter_strength=ssl_cfg["strong_color_jitter_strength"],
        weak_blur_prob=ssl_cfg["weak_blur_prob"],
        strong_blur_prob=ssl_cfg["strong_blur_prob"],
        weak_noise_std=ssl_cfg["weak_noise_std"],
        strong_noise_std=ssl_cfg["strong_noise_std"],
    )


def _build_loader(dataset, *, batch_size: int, shuffle: bool, drop_last: bool, config: dict):
    data_cfg = config["data"]
    num_workers = int(data_cfg["num_workers"])
    persistent_workers = bool(num_workers > 0 and data_cfg["persistent_workers"])
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=bool(data_cfg["pin_memory"]),
        persistent_workers=persistent_workers,
    )


def build_supervised_train_loader(config: dict) -> DataLoader:
    records, paths = _read_records_for_split(config, config["data"]["train_split"])
    dataset = BUSISegDataset(
        records=records,
        dataset_root=paths["dataset_root"],
        merged_mask_root=paths["merged_mask_root"],
        transform=_build_supervised_transform(config, is_train=True),
    )
    return _build_loader(
        dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=True,
        drop_last=True,
        config=config,
    )


def build_supervised_eval_loader(config: dict, *, split_key: str = "val_split") -> DataLoader:
    split_filename = config["data"][split_key]
    records, paths = _read_records_for_split(config, split_filename)
    dataset = BUSISegDataset(
        records=records,
        dataset_root=paths["dataset_root"],
        merged_mask_root=paths["merged_mask_root"],
        transform=_build_supervised_transform(config, is_train=False),
    )
    return _build_loader(
        dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=False,
        drop_last=False,
        config=config,
    )


def build_ssl_train_loaders(config: dict) -> tuple[DataLoader, DataLoader]:
    labeled_records, paths = _read_records_for_split(config, config["data"]["labeled_split"])
    unlabeled_records, _ = _read_records_for_split(config, config["data"]["unlabeled_split"])

    labeled_dataset = BUSISegDataset(
        records=labeled_records,
        dataset_root=paths["dataset_root"],
        merged_mask_root=paths["merged_mask_root"],
        transform=_build_supervised_transform(config, is_train=True),
    )
    unlabeled_dataset = BUSISSLDataset(
        records=unlabeled_records,
        dataset_root=paths["dataset_root"],
        transform=_build_ssl_transform(config),
    )

    labeled_loader = _build_loader(
        labeled_dataset,
        batch_size=int(config["train"]["labeled_batch_size"]),
        shuffle=True,
        drop_last=True,
        config=config,
    )
    unlabeled_loader = _build_loader(
        unlabeled_dataset,
        batch_size=int(config["train"]["unlabeled_batch_size"]),
        shuffle=True,
        drop_last=True,
        config=config,
    )
    return labeled_loader, unlabeled_loader


def build_ssl_eval_loader(config: dict, *, split_key: str = "val_split") -> DataLoader:
    split_filename = config["data"][split_key]
    records, paths = _read_records_for_split(config, split_filename)
    dataset = BUSISegDataset(
        records=records,
        dataset_root=paths["dataset_root"],
        merged_mask_root=paths["merged_mask_root"],
        transform=_build_supervised_transform(config, is_train=False),
    )
    return _build_loader(
        dataset,
        batch_size=int(config["train"]["labeled_batch_size"]),
        shuffle=False,
        drop_last=False,
        config=config,
    )
