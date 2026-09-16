"""ViT-S/16 with independently configurable width, depth, MLP size and sequence length.

The four scaling axes are decoupled on purpose:

* ``embed_dim``  -- token width.  ``num_heads`` defaults to ``embed_dim // head_dim``
  (head_dim = 64, the usual convention) so attention cost stays comparable.
* ``depth``      -- number of transformer blocks.
* ``mlp_dim``    -- hidden size of the feed-forward block (ViT-S: 4 x 384 = 1536).
* ``num_patches``-- input sequence length, excluding the class token.  The image
  resolution is derived from it so the model still consumes an ImageNet-like
  image tensor rather than a synthetic token tensor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int, patch_size: int, embed_dim: int,
                 in_channels: int = 3) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)  # (B, N, C)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"embed_dim {dim} is not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_drop = attn_drop
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # (B, heads, N, head_dim)
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop if self.training else 0.0)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Block(nn.Module):
    """Pre-norm transformer block (ViT / DeiT layout)."""

    def __init__(self, dim: int, num_heads: int, mlp_dim: int, qkv_bias: bool = True,
                 drop: float = 0.0, attn_drop: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, mlp_dim, drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


@dataclass
class ViTConfig:
    """Configuration of a ViT.  Defaults reproduce ViT-S/16 at 224px (196 patches)."""

    embed_dim: int = 384
    depth: int = 12
    mlp_dim: int = 1536
    num_patches: int = 196
    patch_size: int = 16
    num_heads: int | None = None  # defaults to embed_dim // head_dim
    head_dim: int = 64
    num_classes: int = 1000
    qkv_bias: bool = True
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0

    def __post_init__(self) -> None:
        grid = math.isqrt(self.num_patches)
        if grid * grid != self.num_patches:
            raise ValueError(
                f"num_patches={self.num_patches} must be a perfect square so that the "
                "model can consume a square image; try "
                f"{grid ** 2} or {(grid + 1) ** 2}")
        if self.num_heads is None:
            self.num_heads = max(1, self.embed_dim // self.head_dim)
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim={self.embed_dim} is not divisible by "
                f"num_heads={self.num_heads}")

    @property
    def grid_size(self) -> int:
        return math.isqrt(self.num_patches)

    @property
    def resolution(self) -> int:
        """Image side length implied by the requested sequence length."""
        return self.grid_size * self.patch_size

    @property
    def seq_len(self) -> int:
        """Sequence length actually seen by attention (patches + class token)."""
        return self.num_patches + 1

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return (3, self.resolution, self.resolution)


class VisionTransformer(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbed(config.resolution, config.patch_size,
                                      config.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.seq_len, config.embed_dim))
        self.pos_drop = nn.Dropout(config.drop_rate)
        self.blocks = nn.ModuleList([
            Block(config.embed_dim, config.num_heads, config.mlp_dim,
                  config.qkv_bias, config.drop_rate, config.attn_drop_rate)
            for _ in range(config.depth)
        ])
        self.norm = nn.LayerNorm(config.embed_dim, eps=1e-6)
        self.head = nn.Linear(config.embed_dim, config.num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = self.pos_drop(torch.cat((cls, x), dim=1) + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x)[:, 0])


def vit_small(embed_dim: int = 384, depth: int = 12, mlp_dim: int = 1536,
              num_patches: int = 196, **kwargs) -> VisionTransformer:
    return VisionTransformer(ViTConfig(embed_dim=embed_dim, depth=depth,
                                       mlp_dim=mlp_dim, num_patches=num_patches,
                                       **kwargs))
