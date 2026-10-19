"""Supported architectures. Every model takes inputs in [0, 1] and normalizes internally, so attacks operate in raw
input space and a checkpoint is self-describing.

    cnn_mlphead              six-convolution BN CNN with the head Linear -> BN1d -> ReLU -> Linear
    cnn_gap                  the same backbone with BN2d -> ReLU -> GAP -> Linear (fully convolutional)
    preact_resnet18          pre-activation ResNet-18 ending in BN2d -> ReLU -> GAP -> Linear
    preact_resnet18_mlphead  pre-activation ResNet-18 with the MLP head of cnn_mlphead
    preact_resnet18_bb       BackdoorBench's PreActResNet18 (parameter names as in its released checkpoints)

Register a new architecture with ``@register_arch("name")``.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn

ARCHS: dict[str, Callable[..., nn.Module]] = {}


def register_arch(name: str):
    def deco(fn):
        if name in ARCHS:
            raise KeyError(f"architecture {name!r} already registered")
        ARCHS[name] = fn
        return fn

    return deco


def build_model(arch: str, **kwargs) -> nn.Module:
    if arch not in ARCHS:
        raise KeyError(f"unknown architecture {arch!r}; available: {sorted(ARCHS)}")
    return ARCHS[arch](**kwargs)


def list_archs() -> list[str]:
    return sorted(ARCHS)


class Normalize(nn.Module):
    def __init__(self, mean, std, image: bool = True):
        super().__init__()
        shape = (1, -1, 1, 1) if image else (1, -1)
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32).reshape(shape))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32).reshape(shape))

    def forward(self, x):
        return (x - self.mean) / self.std


def _norm(in_channels, mean, std, image=True):
    mean = mean if mean is not None else [0.5] * in_channels
    std = std if std is not None else [0.25] * in_channels
    return Normalize(mean, std, image=image)


def _conv_bn_relu(cin, cout):
    return [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]


class CNNMLPHead(nn.Module):
    """Small BN CNN whose final block is exactly Linear -> BN1d -> ReLU -> Linear."""

    def __init__(self, num_classes=10, in_channels=3, widths=(64, 128, 256), head_width=256, mean=None, std=None):
        super().__init__()
        self.norm = _norm(in_channels, mean, std)
        layers: list[nn.Module] = []
        prev = in_channels
        for i, w in enumerate(widths):
            layers += _conv_bn_relu(prev, w) + _conv_bn_relu(w, w)
            if i < len(widths) - 1:
                layers.append(nn.MaxPool2d(2))
            prev = w
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head_fc = nn.Linear(prev, head_width)
        self.head_bn = nn.BatchNorm1d(head_width)
        self.head_relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(head_width, num_classes)

    def forward(self, x):
        x = self.pool(self.features(self.norm(x))).flatten(1)
        return self.fc(self.head_relu(self.head_bn(self.head_fc(x))))


@register_arch("cnn_mlphead")
def cnn_mlphead(num_classes=10, in_channels=3, widths=(64, 128, 256), head_width=256, mean=None, std=None):
    return CNNMLPHead(num_classes, in_channels, tuple(widths), head_width, mean, std)


class CNNGAP(CNNMLPHead):
    """cnn_mlphead's backbone (same modules and names, last BN2d features.18) with a plain GAP -> Linear classifier."""

    def __init__(self, num_classes=10, in_channels=3, widths=(64, 128, 256), mean=None, std=None):
        super().__init__(num_classes, in_channels, widths, 1, mean, std)
        del self.head_fc, self.head_bn, self.head_relu
        self.fc = nn.Linear(widths[-1], num_classes)

    def forward(self, x):
        return self.fc(self.pool(self.features(self.norm(x))).flatten(1))


@register_arch("cnn_gap")
def cnn_gap(num_classes=10, in_channels=3, widths=(64, 128, 256), mean=None, std=None):
    return CNNGAP(num_classes, in_channels, tuple(widths), mean, std)


class PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.shortcut = None
        if stride != 1 or cin != cout:
            self.shortcut = nn.Conv2d(cin, cout, 1, stride, bias=False)

    def forward(self, x):
        out = torch.relu(self.bn1(x))
        shortcut = self.shortcut(out) if self.shortcut is not None else x
        out = self.conv1(out)
        out = self.conv2(torch.relu(self.bn2(out)))
        return out + shortcut


class PreActResNet(nn.Module):
    """Pre-activation ResNet; tail BN2d -> ReLU -> GAP -> Linear (the BN input is a residual sum)."""

    def __init__(self, num_blocks=(2, 2, 2, 2), num_classes=10, in_channels=3, width=64, mean=None, std=None):
        super().__init__()
        self.norm = _norm(in_channels, mean, std)
        self.conv1 = nn.Conv2d(in_channels, width, 3, 1, 1, bias=False)
        layers = []
        cin = width
        for i, nb in enumerate(num_blocks):
            cout = width * 2**i
            for j in range(nb):
                layers.append(PreActBlock(cin, cout, stride=2 if (j == 0 and i > 0) else 1))
                cin = cout
        self.layers = nn.Sequential(*layers)
        self.bn = nn.BatchNorm2d(cin)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(cin, num_classes)

    def stem(self, x):
        """Normalisation and first convolution: the input of ``layers``."""
        return self.conv1(self.norm(x))

    def forward(self, x):
        out = self.layers(self.stem(x))
        return self.fc(torch.flatten(self.pool(self.relu(self.bn(out))), 1))


@register_arch("preact_resnet18")
def preact_resnet18(num_classes=10, in_channels=3, width=64, mean=None, std=None):
    return PreActResNet((2, 2, 2, 2), num_classes, in_channels, width, mean, std)


class PreActResNetMLPHead(PreActResNet):
    """PreAct-ResNet whose classifier is the MLP head GAP -> Linear -> BN1d -> ReLU -> Linear (as in cnn_mlphead)."""

    def __init__(self, num_blocks=(2, 2, 2, 2), num_classes=10, in_channels=3, width=64, head_width=256, mean=None, std=None):
        super().__init__(num_blocks, num_classes, in_channels, width, mean, std)
        channels = self.fc.in_features
        self.head_fc = nn.Linear(channels, head_width)
        self.head_bn = nn.BatchNorm1d(head_width)
        self.head_relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(head_width, num_classes)

    def forward(self, x):
        out = self.layers(self.stem(x))
        z = torch.flatten(self.pool(self.relu(self.bn(out))), 1)
        return self.fc(self.head_relu(self.head_bn(self.head_fc(z))))


@register_arch("preact_resnet18_mlphead")
def preact_resnet18_mlphead(num_classes=10, in_channels=3, width=64, head_width=256, mean=None, std=None):
    return PreActResNetMLPHead((2, 2, 2, 2), num_classes, in_channels, width, head_width, mean, std)


class PreActBlockBB(nn.Module):
    """BackdoorBench's PreActBlock (models/preact_resnet.py, SCLBD/BackdoorBench): parameter names as in its checkpoints."""
    expansion = 1

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False))

    def forward(self, x):
        out = torch.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(torch.relu(self.bn2(out)))
        return out + shortcut


class PreActResNetBB(nn.Module):
    """BackdoorBench's PreActResNet18: no BN -> ReLU after the last block, so the last BatchNorm2d (layer4[-1].bn2) sits
    inside the residual branch and reaches the classifier through ReLU -> conv2 (3x3) -> + skip -> GAP -> Linear.
    BackdoorBench normalises in its data transform; here the Normalize module (buffers, not in its checkpoints) does it,
    so inputs are in [0, 1] as for the other models."""

    def __init__(self, num_blocks=(2, 2, 2, 2), num_classes=10, in_channels=3, width=64, mean=None, std=None):
        super().__init__()
        self.norm = _norm(in_channels, mean, std)
        self.conv1 = nn.Conv2d(in_channels, width, 3, 1, 1, bias=False)
        cin = width
        for i, nb in enumerate(num_blocks):
            cout = width * 2**i
            blocks = []
            for j in range(nb):
                blocks.append(PreActBlockBB(cin, cout, stride=2 if (j == 0 and i > 0) else 1))
                cin = cout
            setattr(self, f'layer{i + 1}', nn.Sequential(*blocks))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(cin, num_classes)

    def forward(self, x):
        out = self.layer4(self.layer3(self.layer2(self.layer1(self.conv1(self.norm(x))))))
        return self.linear(torch.flatten(self.avgpool(out), 1))


@register_arch("preact_resnet18_bb")
def preact_resnet18_bb(num_classes=10, in_channels=3, width=64, mean=None, std=None):
    return PreActResNetBB((2, 2, 2, 2), num_classes, in_channels, width, mean, std)
