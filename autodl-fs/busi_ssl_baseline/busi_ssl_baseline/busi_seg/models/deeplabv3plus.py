"""DeepLabV3Plus wrapper for stage-1 BUSI segmentation.

This stage-1 wrapper uses a standard ImageNet-pretrained ResNet50 encoder and
is not claimed as official torchvision ResNet50 weights.
"""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
from torch import nn


class DeepLabV3PlusBinarySegModel(nn.Module):
    """Thin segmentation_models_pytorch wrapper that returns logits only."""

    def __init__(
        self,
        *,
        encoder_name: str = "resnet50",
        encoder_weights: str | None = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
    ) -> None:
        super().__init__()
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)
