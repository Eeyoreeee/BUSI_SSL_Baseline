"""Manifest helpers for BUSI sample indexing and split I/O.

`image_relpath` is relative to `dataset_root`.
`merged_mask_relpath` is always relative to `merged_mask_root`, even when the
stored value is just a filename such as `<sample_id>.png`.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


MANIFEST_COLUMNS = [
    "sample_id",
    "label_group",
    "image_relpath",
    "merged_mask_relpath",
    "mask_count",
    "component_mask_relpaths",
    "image_sha1",
]


@dataclass(frozen=True)
class SampleRecord:
    """Single BUSI sample record loaded from manifest.csv.

    Path semantics are strict:
    - `image_relpath` is relative to `dataset_root`
    - `merged_mask_relpath` is relative to `merged_mask_root`
    """

    sample_id: str
    label_group: str
    image_relpath: str
    merged_mask_relpath: str
    mask_count: int
    component_mask_relpaths: tuple[str, ...]
    image_sha1: str

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "SampleRecord":
        component_paths = tuple(json.loads(row["component_mask_relpaths"]))
        return cls(
            sample_id=row["sample_id"],
            label_group=row["label_group"],
            image_relpath=row["image_relpath"],
            merged_mask_relpath=row["merged_mask_relpath"],
            mask_count=int(row["mask_count"]),
            component_mask_relpaths=component_paths,
            image_sha1=row["image_sha1"],
        )

    def to_row(self) -> dict[str, str]:
        return {
            "sample_id": self.sample_id,
            "label_group": self.label_group,
            "image_relpath": self.image_relpath,
            "merged_mask_relpath": self.merged_mask_relpath,
            "mask_count": str(self.mask_count),
            "component_mask_relpaths": json.dumps(
                list(self.component_mask_relpaths), ensure_ascii=False
            ),
            "image_sha1": self.image_sha1,
        }


class ManifestIndex:
    """SampleRecord collection indexed by stable sample_id."""

    def __init__(self, records: Iterable[SampleRecord]) -> None:
        self.records = list(records)
        self.by_sample_id = {record.sample_id: record for record in self.records}
        if len(self.by_sample_id) != len(self.records):
            raise ValueError("Duplicate sample_id detected while building manifest index.")

    def __len__(self) -> int:
        return len(self.records)

    def get(self, sample_id: str) -> SampleRecord:
        return self.by_sample_id[sample_id]

    def subset(self, sample_ids: Iterable[str]) -> list[SampleRecord]:
        return [self.by_sample_id[sample_id] for sample_id in sample_ids]

    def label_map(self) -> dict[str, str]:
        return {record.sample_id: record.label_group for record in self.records}


def read_manifest(manifest_path: str | Path) -> ManifestIndex:
    path = Path(manifest_path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in MANIFEST_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"Manifest {path} is missing required columns: {', '.join(missing)}"
            )
        records = [SampleRecord.from_row(row) for row in reader]
    return ManifestIndex(records)


def write_manifest(records: Iterable[SampleRecord], manifest_path: str | Path) -> None:
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_row())


def read_split_ids(split_path: str | Path) -> list[str]:
    path = Path(split_path)
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def write_split_ids(split_path: str | Path, sample_ids: Iterable[str]) -> None:
    path = Path(split_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample_id in sample_ids:
            handle.write(f"{sample_id}\n")
