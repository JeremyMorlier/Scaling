"""ResNet-50 with a uniform channel-width multiplier.

torchvision's ``width_per_group`` only rescales the 3x3 conv inside each
bottleneck.  Here every channel count -- stem, both bottleneck convs and the
expanded output -- is scaled by the same factor, which is what a clean width
scaling law requires.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def make_divisible(value: float, divisor: int = 8, min_value: int | None = None) -> int:
    """Round ``value`` to the nearest multiple of ``divisor`` (never below 90%)."""
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, planes: int, stride: int = 1,
                 downsample: nn.Module | None = None) -> None:
        super().__init__()
        out_channels = planes * self.expansion
        self.conv1 = nn.Conv2d(in_channels, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)


@dataclass
class ResNetConfig:
    """Configuration of a width-scaled ResNet-50.

    Attributes:
        width_mult: uniform multiplier applied to every channel count.
        resolution: side length of the square input image.
        num_classes: size of the classification head.
        base_width: stem width of the unscaled model (64 for ResNet-50).
        channel_divisor: scaled channel counts are rounded to this multiple.
    """

    width_mult: float = 1.0
    resolution: int = 224
    num_classes: int = 1000
    base_width: int = 64
    channel_divisor: int = 8

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return (3, self.resolution, self.resolution)


class ResNet50(nn.Module):
    """ResNet-50 (bottleneck layout [3, 4, 6, 3]) with uniform width scaling."""

    layers = (3, 4, 6, 3)

    def __init__(self, config: ResNetConfig) -> None:
        super().__init__()
        self.config = config
        div = config.channel_divisor
        stem = make_divisible(config.base_width * config.width_mult, div)
        planes = [make_divisible(config.base_width * (2 ** i) * config.width_mult, div)
                  for i in range(4)]

        self.in_channels = stem
        self.conv1 = nn.Conv2d(3, stem, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(stem)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(planes[0], self.layers[0], stride=1)
        self.layer2 = self._make_layer(planes[1], self.layers[1], stride=2)
        self.layer3 = self._make_layer(planes[2], self.layers[2], stride=2)
        self.layer4 = self._make_layer(planes[3], self.layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.in_channels, config.num_classes)

        self._init_weights()

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        out_channels = planes * Bottleneck.expansion
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = [Bottleneck(self.in_channels, planes, stride, downsample)]
        self.in_channels = out_channels
        layers += [Bottleneck(self.in_channels, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Zero-init the last BN of every residual branch (He et al., "Bag of Tricks").
        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.zeros_(m.bn3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = torch.flatten(self.avgpool(x), 1)
        return self.fc(x)


def resnet50(width_mult: float = 1.0, resolution: int = 224,
             num_classes: int = 1000, **kwargs) -> ResNet50:
    return ResNet50(ResNetConfig(width_mult=width_mult, resolution=resolution,
                                 num_classes=num_classes, **kwargs))
