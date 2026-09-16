"""Shape and parameter-count checks for the scaled models.

Run with: python -m pytest tests -q   (or: python tests/test_models.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scaling.models import build_model
from scaling.models.resnet import make_divisible
from scaling.models.vit import ViTConfig


def test_resnet50_base_matches_reference_param_count():
    model = build_model("resnet50")
    # Canonical torchvision ResNet-50 with a 1000-class head.
    assert sum(p.numel() for p in model.parameters()) == 25_557_032


def test_vit_small_base_matches_reference_param_count():
    model = build_model("vit_small")
    # Canonical DeiT-S/16 / ViT-S/16 with a 1000-class head.
    assert sum(p.numel() for p in model.parameters()) == 22_050_664
    assert model.config.num_heads == 6
    assert model.config.seq_len == 197


def test_resnet_width_scales_every_channel_count():
    half = build_model("resnet50", width_mult=0.5)
    assert half.conv1.out_channels == make_divisible(64 * 0.5)
    assert half.fc.in_features == make_divisible(512 * 0.5) * 4
    # Parameters scale roughly quadratically in the width multiplier.
    full = sum(p.numel() for p in build_model("resnet50").parameters())
    ratio = sum(p.numel() for p in half.parameters()) / full
    assert 0.2 < ratio < 0.32


def test_resolution_axis_changes_only_the_input():
    for res in (96, 160, 224, 384):
        model = build_model("resnet50", resolution=res)
        out = model(torch.randn(2, *model.config.input_shape))
        assert out.shape == (2, 1000)
        assert model.config.input_shape == (3, res, res)


def test_vit_axes_are_independent():
    for axis, value in [("embed_dim", 512), ("depth", 4),
                        ("mlp_dim", 3072), ("num_patches", 64)]:
        model = build_model("vit_small", **{axis: value})
        cfg = model.config
        assert getattr(cfg, axis) == value
        assert model(torch.randn(2, *cfg.input_shape)).shape == (2, 1000)
    # Sequence length is driven by num_patches and nothing else.
    assert build_model("vit_small", num_patches=64).config.resolution == 128
    assert build_model("vit_small", embed_dim=768).config.resolution == 224


def test_vit_rejects_non_square_sequence_length():
    try:
        ViTConfig(num_patches=200)
    except ValueError as exc:
        assert "perfect square" in str(exc)
    else:
        raise AssertionError("expected ValueError for num_patches=200")


def test_unknown_axis_is_rejected():
    try:
        build_model("resnet50", depth=3)
    except ValueError as exc:
        assert "unknown parameter" in str(exc)
    else:
        raise AssertionError("expected ValueError for resnet50 depth")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
