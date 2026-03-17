"""Explicit model builder for the stage-1 BUSI baseline."""

from __future__ import annotations

from torch import nn

from .deeplabv3plus import DeepLabV3PlusBinarySegModel


def build_model(config: dict) -> nn.Module:
    """Build the only stage-1 supervised model explicitly, without a registry."""

    model_cfg = config["model"]
    model_name = str(model_cfg["name"]).lower()
    if model_name != "deeplabv3plus":
        raise ValueError(
            f"Stage 1 supports only model.name='deeplabv3plus', got: {model_cfg['name']}"
        )

    if model_cfg["encoder_name"] != "resnet50":
        raise ValueError(
            "Stage 1 supports only encoder_name='resnet50' for DeepLabV3Plus."
        )
    if model_cfg["encoder_weights"] != "imagenet":
        raise ValueError(
            "Stage 1 expects encoder_weights='imagenet' for the ResNet50 encoder."
        )
    if int(model_cfg["in_channels"]) != 3:
        raise ValueError("Stage 1 supports only in_channels=3.")
    if int(model_cfg["classes"]) != 1:
        raise ValueError("Stage 1 supports only classes=1 for binary segmentation.")
    if bool(model_cfg.get("claim_official_torchvision_equivalence", False)):
        raise ValueError(
            "This project does not claim official torchvision weight equivalence."
        )

    return DeepLabV3PlusBinarySegModel(
        encoder_name=model_cfg["encoder_name"],
        encoder_weights=model_cfg["encoder_weights"],
        in_channels=int(model_cfg["in_channels"]),
        classes=int(model_cfg["classes"]),
    )
