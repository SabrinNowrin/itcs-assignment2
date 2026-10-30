"""Backbones (always randomly initialised -- weights=None) and heads."""
import torch
import torch.nn as nn
import torchvision


class ResNetEncoder(nn.Module):
    """torchvision ResNet with weights=None and a stem adapted to small images.

    stem='cifar'    : 3x3 conv stride 1, no max-pool   (best for <=64px)
    stem='small'    : 3x3 conv stride 1, keep max-pool (good for ~96px)
    stem='imagenet' : original 7x7 stride-2 conv + max-pool (>=160px)
    """

    def __init__(self, name="resnet18", img_size=96, stem="auto"):
        super().__init__()
        net = getattr(torchvision.models, name)(weights=None)  # NO external pretraining
        if stem == "auto":
            stem = "cifar" if img_size <= 64 else ("small" if img_size <= 128 else "imagenet")
        if stem in ("cifar", "small"):
            net.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        if stem == "cifar":
            net.maxpool = nn.Identity()
        self.out_dim = net.fc.in_features
        net.fc = nn.Identity()
        self.net = net
        self.stem = stem

    def forward(self, x):
        return self.net(x)


class ViTEncoder(nn.Module):
    """A small Vision Transformer trained from scratch (for backbone comparisons)."""

    def __init__(self, img_size=96, patch=8, dim=192, depth=6, heads=3, mlp_ratio=4.0):
        super().__init__()
        assert img_size % patch == 0
        n = (img_size // patch) ** 2
        self.patch_embed = nn.Conv2d(3, dim, patch, patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n + 1, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, int(dim * mlp_ratio), dropout=0.0,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.out_dim = dim
        self.stem = f"patch{patch}"

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], 1) + self.pos
        return self.norm(self.blocks(x)[:, 0])


def build_encoder(arch, img_size, stem="auto"):
    if arch.startswith("resnet"):
        return ResNetEncoder(arch, img_size, stem)
    if arch == "vit_tiny":
        return ViTEncoder(img_size, patch=8 if img_size % 8 == 0 else 4)
    raise ValueError(arch)


class ProjectionHead(nn.Module):
    """SimCLR MLP head g(.): the loss is applied to g(h), the representation h is kept."""

    def __init__(self, in_dim, hidden=512, out=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden, bias=False), nn.BatchNorm1d(hidden),
                                 nn.ReLU(inplace=True), nn.Linear(hidden, out))

    def forward(self, x):
        return self.net(x)


class Classifier(nn.Module):
    def __init__(self, encoder, num_classes, dropout=0.0):
        super().__init__()
        self.encoder = encoder
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(encoder.out_dim, num_classes)

    def forward(self, x):
        return self.head(self.drop(self.encoder(x)))
