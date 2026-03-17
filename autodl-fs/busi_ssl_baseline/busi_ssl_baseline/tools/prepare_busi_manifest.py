"""Prepare a stable BUSI manifest and offline merged lesion masks."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import re
from pathlib import Path
import sys

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from busi_seg.data.manifest import SampleRecord, write_manifest


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MASK_STEM_PATTERN = re.compile(r"_mask(?:_\d+)?$")
REQUIRED_LABEL_GROUPS = ("benign", "malignant")
OPTIONAL_LABEL_GROUPS = ("normal",)


@dataclass(frozen=True)
class PendingSample:
    """Intermediate BUSI sample collected before final sample_id assignment."""

    label_group: str
    image_path: Path
    image_relpath: str
    component_mask_paths: tuple[Path, ...]
    component_mask_relpaths: tuple[str, ...]
    image_sha1: str
    base_sample_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare BUSI manifest and offline merged masks."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to the BUSI dataset root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data_meta"),
        help="Output directory for manifest.csv, merged_masks/, and splits/.",
    )
    parser.add_argument(
        "--include-normal",
        action="store_true",
        help="Include BUSI normal samples. Stage 1 defaults to excluding them.",
    )
    parser.add_argument(
        "--overwrite-merged-masks",
        action="store_true",
        help="Allow overwriting existing merged mask files.",
    )
    return parser.parse_args()


def normalize_stem(stem: str) -> str:
    normalized = stem.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    return normalized.strip("_")


def build_sample_id(label_group: str, image_stem: str, suffix: str | None = None) -> str:
    base = f"{label_group}__{normalize_stem(image_stem)}"
    if suffix:
        return f"{base}__{suffix}"
    return base


def compute_sha1(file_path: Path) -> str:
    sha1 = hashlib.sha1()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha1.update(chunk)
    return sha1.hexdigest()


def ensure_output_layout(output_root: Path) -> None:
    (output_root / "merged_masks").mkdir(parents=True, exist_ok=True)
    (output_root / "splits" / "seed0").mkdir(parents=True, exist_ok=True)


def resolve_scan_root(dataset_root: Path, include_normal: bool) -> Path:
    required_groups = REQUIRED_LABEL_GROUPS + OPTIONAL_LABEL_GROUPS if include_normal else REQUIRED_LABEL_GROUPS

    if all((dataset_root / label).is_dir() for label in required_groups):
        return dataset_root

    nested = dataset_root / "Dataset_BUSI_with_GT"
    if all((nested / label).is_dir() for label in required_groups):
        return nested

    raise FileNotFoundError(
        "Could not find BUSI label folders under dataset root. "
        f"Expected {', '.join(required_groups)} under {dataset_root} or {nested}."
    )


def iter_label_groups(include_normal: bool) -> tuple[str, ...]:
    return ("benign", "malignant", "normal") if include_normal else ("benign", "malignant")


def is_mask_file(path: Path) -> bool:
    return bool(MASK_STEM_PATTERN.search(path.stem))


def find_component_masks(label_dir: Path, image_path: Path) -> list[Path]:
    pattern = re.compile(rf"^{re.escape(image_path.stem)}_mask(?:_\d+)?$")
    mask_paths = [
        candidate
        for candidate in label_dir.iterdir()
        if candidate.is_file()
        and candidate.suffix.lower() in IMAGE_SUFFIXES
        and pattern.match(candidate.stem)
    ]
    return sorted(mask_paths)


def build_pending_samples(dataset_root: Path, scan_root: Path, include_normal: bool) -> list[PendingSample]:
    pending: list[PendingSample] = []

    for label_group in iter_label_groups(include_normal):
        label_dir = scan_root / label_group
        image_paths = [
            path
            for path in sorted(label_dir.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and not is_mask_file(path)
        ]

        for image_path in image_paths:
            component_mask_paths = find_component_masks(label_dir, image_path)
            if label_group != "normal" and not component_mask_paths:
                raise FileNotFoundError(f"No lesion mask found for BUSI image: {image_path}")

            pending.append(
                PendingSample(
                    label_group=label_group,
                    image_path=image_path,
                    image_relpath=image_path.relative_to(dataset_root).as_posix(),
                    component_mask_paths=tuple(component_mask_paths),
                    component_mask_relpaths=tuple(
                        mask_path.relative_to(dataset_root).as_posix()
                        for mask_path in component_mask_paths
                    ),
                    image_sha1=compute_sha1(image_path),
                    base_sample_id=build_sample_id(label_group, image_path.stem),
                )
            )

    if not pending:
        raise RuntimeError(f"No BUSI image files found under {scan_root}")

    return pending


def assign_sample_ids(pending_samples: list[PendingSample]) -> dict[Path, str]:
    """Resolve sample_id collisions with deterministic hash suffixes only."""

    grouped: dict[str, list[PendingSample]] = defaultdict(list)
    for sample in pending_samples:
        grouped[sample.base_sample_id].append(sample)

    assigned: dict[Path, str] = {}
    for base_id, samples in grouped.items():
        if len(samples) == 1:
            assigned[samples[0].image_path] = base_id
            continue

        sample_ids = [
            build_sample_id(sample.label_group, sample.image_path.stem, sample.image_sha1[:8])
            for sample in samples
        ]
        if len(set(sample_ids)) != len(samples):
            sample_ids = [
                build_sample_id(sample.label_group, sample.image_path.stem, sample.image_sha1[:12])
                for sample in samples
            ]

        if len(set(sample_ids)) != len(samples):
            raise ValueError(
                "sample_id collision remained after deterministic hash suffix extension. "
                f"Conflicting base sample_id: {base_id}"
            )

        for sample, sample_id in zip(samples, sample_ids, strict=True):
            assigned[sample.image_path] = sample_id

    return assigned


def export_merged_mask(
    *,
    image_path: Path,
    component_mask_paths: tuple[Path, ...],
    output_path: Path,
    overwrite: bool,
) -> int:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Merged mask already exists and overwrite is disabled: {output_path}"
        )

    image_array = np.array(Image.open(image_path))
    image_height, image_width = image_array.shape[:2]

    if component_mask_paths:
        merged_mask = np.zeros((image_height, image_width), dtype=bool)
        for mask_path in component_mask_paths:
            mask_array = np.array(Image.open(mask_path).convert("L"))
            if mask_array.shape != merged_mask.shape:
                raise ValueError(
                    "Mask shape does not match image shape: "
                    f"{mask_path} vs {image_path}"
                )
            merged_mask |= mask_array > 0
    else:
        merged_mask = np.zeros((image_height, image_width), dtype=bool)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((merged_mask.astype(np.uint8) * 255), mode="L").save(output_path)
    return int(len(component_mask_paths))


def create_manifest_records(
    pending_samples: list[PendingSample],
    sample_id_map: dict[Path, str],
    output_root: Path,
    overwrite_merged_masks: bool,
) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    merged_root = output_root / "merged_masks"

    for sample in sorted(pending_samples, key=lambda item: item.image_relpath):
        sample_id = sample_id_map[sample.image_path]
        merged_mask_filename = f"{sample_id}.png"
        merged_mask_path = merged_root / merged_mask_filename
        mask_count = export_merged_mask(
            image_path=sample.image_path,
            component_mask_paths=sample.component_mask_paths,
            output_path=merged_mask_path,
            overwrite=overwrite_merged_masks,
        )

        records.append(
            SampleRecord(
                sample_id=sample_id,
                label_group=sample.label_group,
                image_relpath=sample.image_relpath,
                # Stored as a path relative to `output_root/merged_masks`.
                merged_mask_relpath=merged_mask_filename,
                mask_count=mask_count,
                component_mask_relpaths=sample.component_mask_relpaths,
                image_sha1=sample.image_sha1,
            )
        )

    return records


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    ensure_output_layout(output_root)

    scan_root = resolve_scan_root(dataset_root, args.include_normal)
    pending_samples = build_pending_samples(
        dataset_root=dataset_root,
        scan_root=scan_root,
        include_normal=args.include_normal,
    )
    sample_id_map = assign_sample_ids(pending_samples)
    records = create_manifest_records(
        pending_samples=pending_samples,
        sample_id_map=sample_id_map,
        output_root=output_root,
        overwrite_merged_masks=args.overwrite_merged_masks,
    )
    write_manifest(records, output_root / "manifest.csv")


if __name__ == "__main__":
    main()
