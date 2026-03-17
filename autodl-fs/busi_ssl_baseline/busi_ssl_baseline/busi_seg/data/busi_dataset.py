"""Supervised BUSI dataset that reads images and offline merged masks only.

`merged_mask_relpath` is resolved relative to `merged_mask_root`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .manifest import SampleRecord


def _load_busi_image_as_rgb(image_path: Path) -> Image.Image:
    array = np.array(Image.open(image_path))

    if array.ndim == 2:
        array = array[:, :, None]
    if array.ndim != 3:
        raise ValueError(f"Unsupported image shape at {image_path}: {array.shape}")

    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    elif array.shape[2] >= 3:
        array = array[:, :, :3]
    else:
        raise ValueError(f"Unsupported channel count at {image_path}: {array.shape}")

    return Image.fromarray(array.astype(np.uint8), mode="RGB")


def _load_mask(mask_path: Path) -> Image.Image:
    return Image.open(mask_path).convert("L")


class BUSISegDataset(Dataset):
    """Dataset for supervised, validation, and test segmentation samples."""

    def __init__(
        self,
        *,
        records: list[SampleRecord],
        dataset_root: str | Path,
        merged_mask_root: str | Path,
        transform,
    ) -> None:
        self.records = records
        self.dataset_root = Path(dataset_root)
        self.merged_mask_root = Path(merged_mask_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image_path = self.dataset_root / record.image_relpath
        # `merged_mask_relpath` is always relative to `merged_mask_root`.
        mask_path = self.merged_mask_root / record.merged_mask_relpath

        image = _load_busi_image_as_rgb(image_path)
        mask = _load_mask(mask_path)
        image_tensor, mask_tensor = self.transform(image, mask)

        meta = {
            "sample_id": record.sample_id,
            "label_group": record.label_group,
            "image_sha1": record.image_sha1,
        }
        return image_tensor, mask_tensor, meta
