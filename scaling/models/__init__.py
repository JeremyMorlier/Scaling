"""Model zoo for the scaling study."""

from __future__ import annotations

import torch.nn as nn

from .resnet import ResNet50, ResNetConfig, resnet50
from .vit import ViTConfig, VisionTransformer, vit_small

__all__ = [
    "ResNet50", "ResNetConfig", "resnet50",
    "VisionTransformer", "ViTConfig", "vit_small",
    "build_model", "MODEL_AXES",
]

#: Per-architecture scaling axes and the base (unscaled) value of each.
MODEL_AXES: dict[str, dict[str, float | int]] = {
    "resnet50": {"width_mult": 1.0, "resolution": 224},
    "vit_small": {"embed_dim": 384, "depth": 12, "mlp_dim": 1536, "num_patches": 196},
}


def build_model(name: str, num_classes: int = 1000, **params) -> nn.Module:
    """Instantiate ``name`` with the base configuration overridden by ``params``."""
    if name not in MODEL_AXES:
        raise ValueError(f"unknown model {name!r}; expected one of {sorted(MODEL_AXES)}")
    config = dict(MODEL_AXES[name])
    unknown = set(params) - set(config)
    if unknown:
        raise ValueError(
            f"unknown parameter(s) {sorted(unknown)} for {name}; "
            f"valid axes are {sorted(config)}")
    config.update(params)
    if name == "resnet50":
        return resnet50(num_classes=num_classes, **config)
    return vit_small(num_classes=num_classes, **config)
