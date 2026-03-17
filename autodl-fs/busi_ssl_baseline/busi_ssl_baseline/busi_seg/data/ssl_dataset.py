"""Unlabeled BUSI dataset that returns weak/strong views with shared geometry."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset

from .busi_dataset import _load_busi_image_as_rgb
from .manifest import SampleRecord


class BUSISSLDataset(Dataset):
    """Dataset for unlabeled SSL samples with optional geometry debug metadata."""

    def __init__(
        self,
        *,
        records: list[SampleRecord],
        dataset_root: str | Path,
        transform,
        include_geometry_in_meta: bool = False,
    ) -> None:
        self.records = records
        self.dataset_root = Path(dataset_root)
        self.transform = transform
        self.include_geometry_in_meta = include_geometry_in_meta

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image_path = self.dataset_root / record.image_relpath
        image = _load_busi_image_as_rgb(image_path)

        weak_tensor, strong_tensor, geometry = self.transform(image)
        meta = {
            "sample_id": record.sample_id,
            "label_group": record.label_group,
            "image_sha1": record.image_sha1,
        }
        if self.include_geometry_in_meta:
            meta["geometry"] = {
                "do_hflip": geometry.do_hflip,
                "do_vflip": geometry.do_vflip,
                "angle_deg": geometry.angle_deg,
                "scale": geometry.scale,
                "crop_top": geometry.crop_top,
                "crop_left": geometry.crop_left,
                "crop_size": geometry.crop_size,
                "output_size": geometry.output_size,
            }
        return weak_tensor, strong_tensor, meta
