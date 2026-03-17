"""Image and mask transforms with explicit shared geometry parameters.

Stage 1 intentionally uses a simple geometry contract:
- first resize to the fixed `output_size` canvas, which is `512` in this repo
- then apply shared geometry perturbations on that resized canvas

This is an engineering tradeoff for stability and implementation clarity in the
initial baseline. It is not presented here as a universally optimal
augmentation strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
import random

from PIL import Image
import torch
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode


@dataclass(frozen=True)
class GeometryParams:
    """Explicit geometry parameters shared after the fixed-size resize step."""

    output_size: int
    do_hflip: bool
    do_vflip: bool
    angle_deg: float
    scale: float
    crop_top: int
    crop_left: int
    crop_size: int


def _get_rng(rng: random.Random | None) -> random.Random:
    return rng if rng is not None else random


def sample_geometry_params(
    *,
    output_size: int,
    hflip_prob: float,
    vflip_prob: float,
    max_rotate_deg: float,
    scale_min: float,
    scale_max: float,
    crop_min_ratio: float,
    rng: random.Random | None = None,
) -> GeometryParams:
    """Sample one set of spatial parameters for reuse on the resized output canvas."""

    rng_obj = _get_rng(rng)
    safe_crop_min_ratio = min(max(crop_min_ratio, 0.1), 1.0)
    crop_ratio = rng_obj.uniform(safe_crop_min_ratio, 1.0)
    crop_size = max(1, min(output_size, int(round(output_size * crop_ratio))))
    max_offset = output_size - crop_size

    return GeometryParams(
        output_size=output_size,
        do_hflip=rng_obj.random() < hflip_prob,
        do_vflip=rng_obj.random() < vflip_prob,
        angle_deg=rng_obj.uniform(-max_rotate_deg, max_rotate_deg),
        scale=rng_obj.uniform(scale_min, scale_max),
        crop_top=rng_obj.randint(0, max_offset) if max_offset > 0 else 0,
        crop_left=rng_obj.randint(0, max_offset) if max_offset > 0 else 0,
        crop_size=crop_size,
    )


def apply_geometry(
    image: Image.Image,
    params: GeometryParams,
    *,
    is_mask: bool,
) -> Image.Image:
    """Apply stage-1 geometry after resizing first to the fixed output canvas."""

    interpolation = (
        InterpolationMode.NEAREST if is_mask else InterpolationMode.BILINEAR
    )
    resized = TF.resize(
        image,
        [params.output_size, params.output_size],
        interpolation=interpolation,
    )

    if params.do_hflip:
        resized = TF.hflip(resized)
    if params.do_vflip:
        resized = TF.vflip(resized)

    transformed = TF.affine(
        resized,
        angle=params.angle_deg,
        translate=[0, 0],
        scale=params.scale,
        shear=[0.0, 0.0],
        interpolation=interpolation,
        fill=0,
    )

    if params.crop_size < params.output_size:
        transformed = TF.crop(
            transformed,
            top=params.crop_top,
            left=params.crop_left,
            height=params.crop_size,
            width=params.crop_size,
        )
        transformed = TF.resize(
            transformed,
            [params.output_size, params.output_size],
            interpolation=interpolation,
        )

    return transformed


def image_to_normalized_tensor(
    image: Image.Image,
    *,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    noise_std: float = 0.0,
) -> torch.Tensor:
    tensor = TF.to_tensor(image)
    if noise_std > 0.0:
        tensor = (tensor + torch.randn_like(tensor) * noise_std).clamp(0.0, 1.0)
    return TF.normalize(tensor, mean=mean, std=std)


def mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    tensor = TF.to_tensor(mask)
    return (tensor > 0.5).to(dtype=torch.float32)


def apply_photometric_augment(
    image: Image.Image,
    *,
    color_jitter_strength: float,
    blur_prob: float,
    rng: random.Random | None = None,
) -> Image.Image:
    """Apply only photometric perturbations, keeping geometry untouched."""

    rng_obj = _get_rng(rng)
    augmented = image

    if color_jitter_strength > 0.0:
        brightness = 1.0 + rng_obj.uniform(-color_jitter_strength, color_jitter_strength)
        contrast = 1.0 + rng_obj.uniform(-color_jitter_strength, color_jitter_strength)
        saturation = 1.0 + rng_obj.uniform(
            -0.5 * color_jitter_strength, 0.5 * color_jitter_strength
        )
        augmented = TF.adjust_brightness(augmented, brightness)
        augmented = TF.adjust_contrast(augmented, contrast)
        augmented = TF.adjust_saturation(augmented, saturation)

    if blur_prob > 0.0 and rng_obj.random() < blur_prob:
        sigma = rng_obj.uniform(0.1, 1.0 + color_jitter_strength)
        augmented = TF.gaussian_blur(augmented, kernel_size=[5, 5], sigma=sigma)

    return augmented


class SupervisedSegTransform:
    """Transform for supervised train/val/test segmentation samples.

    Current stage-1 policy first resizes to `output_size`, then applies the
    shared geometry perturbation to image and mask together.
    """

    def __init__(
        self,
        *,
        output_size: int,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        augment: bool,
        hflip_prob: float = 0.0,
        vflip_prob: float = 0.0,
        max_rotate_deg: float = 0.0,
        scale_min: float = 1.0,
        scale_max: float = 1.0,
        crop_min_ratio: float = 1.0,
        color_jitter_strength: float = 0.0,
        blur_prob: float = 0.0,
        noise_std: float = 0.0,
    ) -> None:
        self.output_size = output_size
        self.mean = mean
        self.std = std
        self.augment = augment
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.max_rotate_deg = max_rotate_deg
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.crop_min_ratio = crop_min_ratio
        self.color_jitter_strength = color_jitter_strength
        self.blur_prob = blur_prob
        self.noise_std = noise_std

    def __call__(self, image: Image.Image, mask: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        if self.augment:
            geometry = sample_geometry_params(
                output_size=self.output_size,
                hflip_prob=self.hflip_prob,
                vflip_prob=self.vflip_prob,
                max_rotate_deg=self.max_rotate_deg,
                scale_min=self.scale_min,
                scale_max=self.scale_max,
                crop_min_ratio=self.crop_min_ratio,
            )
        else:
            geometry = GeometryParams(
                output_size=self.output_size,
                do_hflip=False,
                do_vflip=False,
                angle_deg=0.0,
                scale=1.0,
                crop_top=0,
                crop_left=0,
                crop_size=self.output_size,
            )

        image_geo = apply_geometry(image, geometry, is_mask=False)
        mask_geo = apply_geometry(mask, geometry, is_mask=True)

        if self.augment:
            image_geo = apply_photometric_augment(
                image_geo,
                color_jitter_strength=self.color_jitter_strength,
                blur_prob=self.blur_prob,
            )

        image_tensor = image_to_normalized_tensor(
            image_geo,
            mean=self.mean,
            std=self.std,
            noise_std=self.noise_std if self.augment else 0.0,
        )
        mask_tensor = mask_to_tensor(mask_geo)
        return image_tensor, mask_tensor


class SSLViewTransform:
    """Transform that guarantees weak/strong views reuse one GeometryParams.

    Weak and strong branches share the same post-resize geometry and differ
    only in photometric/noise perturbations.
    """

    def __init__(
        self,
        *,
        output_size: int,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        hflip_prob: float,
        vflip_prob: float,
        max_rotate_deg: float,
        scale_min: float,
        scale_max: float,
        crop_min_ratio: float,
        weak_color_jitter_strength: float,
        strong_color_jitter_strength: float,
        weak_blur_prob: float,
        strong_blur_prob: float,
        weak_noise_std: float,
        strong_noise_std: float,
    ) -> None:
        self.output_size = output_size
        self.mean = mean
        self.std = std
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.max_rotate_deg = max_rotate_deg
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.crop_min_ratio = crop_min_ratio
        self.weak_color_jitter_strength = weak_color_jitter_strength
        self.strong_color_jitter_strength = strong_color_jitter_strength
        self.weak_blur_prob = weak_blur_prob
        self.strong_blur_prob = strong_blur_prob
        self.weak_noise_std = weak_noise_std
        self.strong_noise_std = strong_noise_std

    def __call__(
        self, image: Image.Image
    ) -> tuple[torch.Tensor, torch.Tensor, GeometryParams]:
        geometry = sample_geometry_params(
            output_size=self.output_size,
            hflip_prob=self.hflip_prob,
            vflip_prob=self.vflip_prob,
            max_rotate_deg=self.max_rotate_deg,
            scale_min=self.scale_min,
            scale_max=self.scale_max,
            crop_min_ratio=self.crop_min_ratio,
        )

        shared_geometry_image = apply_geometry(image, geometry, is_mask=False)

        weak_image = apply_photometric_augment(
            shared_geometry_image,
            color_jitter_strength=self.weak_color_jitter_strength,
            blur_prob=self.weak_blur_prob,
        )
        strong_image = apply_photometric_augment(
            shared_geometry_image,
            color_jitter_strength=self.strong_color_jitter_strength,
            blur_prob=self.strong_blur_prob,
        )

        weak_tensor = image_to_normalized_tensor(
            weak_image,
            mean=self.mean,
            std=self.std,
            noise_std=self.weak_noise_std,
        )
        strong_tensor = image_to_normalized_tensor(
            strong_image,
            mean=self.mean,
            std=self.std,
            noise_std=self.strong_noise_std,
        )
        return weak_tensor, strong_tensor, geometry
