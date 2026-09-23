from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

BANDS = (
    "red",
    "green",
    "blue",
    "nir",
    "coastal",
    "rededge1",
    "rededge2",
    "rededge3",
    "nir08",
    "nir09",
    "swir16",
    "swir22",
)


class LayerScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(384))

    def forward(self, value):
        return value * self.gamma


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(384, 1152)
        self.proj = nn.Linear(384, 384)

    def forward(self, tokens):
        batch, count, _ = tokens.shape
        q, k, v = self.qkv(tokens).reshape(batch, count, 3, 6, 64).permute(2, 0, 3, 1, 4).unbind(0)
        value = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        return self.proj(value.transpose(1, 2).reshape(batch, count, 384))


class Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(384, 1536)
        self.fc2 = nn.Linear(1536, 384)

    def forward(self, value):
        return self.fc2(F.gelu(self.fc1(value)))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(384, eps=1e-6)
        self.attn = Attention()
        self.ls1 = LayerScale()
        self.norm2 = nn.LayerNorm(384, eps=1e-6)
        self.mlp = Mlp()
        self.ls2 = LayerScale()

    def forward(self, value):
        value = value + self.ls1(self.attn(self.norm1(value)))
        return value + self.ls2(self.mlp(self.norm2(value)))


class GeometryFiLM(nn.Module):
    def __init__(self, dropout: float = 0.5):
        super().__init__()
        if dropout != 0.5:
            raise ValueError("the reported geometry-dropout probability is 0.5")
        self.dropout = dropout
        self.reduce = nn.Sequential(nn.Linear(6, 16), nn.SiLU(), nn.Linear(16, 4))
        self.expand = nn.Linear(4, 384, bias=False)
        nn.init.zeros_(self.expand.weight)

    def forward(self, tokens, geometry):
        conditioned = geometry
        if self.training:
            keep = (torch.rand((len(geometry), 1), device=geometry.device) >= self.dropout).to(
                geometry.dtype
            )
            conditioned = geometry * keep
        vectors = conditioned[:, [0, 1, 2, 4, 5, 6]]
        available = (conditioned[:, [3, 7]].sum(1, keepdim=True) > 0).to(tokens.dtype)
        scale = 0.05 * torch.tanh(self.expand(self.reduce(vectors))) * available
        delta = tokens - tokens[:, :1]
        centered = delta - delta.mean(1, keepdim=True)
        return tokens + centered * scale[:, None, :]


class PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 384, 14, stride=14)

    def forward(self, value):
        return self.proj(value).flatten(2).transpose(1, 2)


class VesselEncoder(nn.Module):
    """Reported twelve-band DINOv2-S/14-derived encoder."""

    def __init__(self, position_tokens=1370):
        super().__init__()
        grid = math.isqrt(position_tokens - 1)
        if grid**2 != position_tokens - 1 or grid < 8:
            raise ValueError("position_tokens must be one plus a square grid")
        self.original_grid = grid
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 384))
        self.pos_embed = nn.Parameter(torch.zeros(1, position_tokens, 384))
        self.mask_token = nn.Parameter(torch.zeros(1, 384), requires_grad=False)
        self.patch_embed = PatchEmbed()
        self.spectral_patch = nn.Conv2d(9, 384, 14, stride=14, bias=False)
        nn.init.zeros_(self.spectral_patch.weight)
        self.geometry_film = GeometryFiLM()
        self.blocks = nn.ModuleList([Block() for _ in range(12)])
        self.norm = nn.LayerNorm(384, eps=1e-6)
        self.head = nn.Sequential(nn.LayerNorm(384), nn.Linear(384, 256))

    def positions(self):
        patch = self.pos_embed[:, 1:].reshape(1, self.original_grid, self.original_grid, 384)
        patch = patch.permute(0, 3, 1, 2)
        patch = F.interpolate(
            patch,
            scale_factor=((8.1 / self.original_grid),) * 2,
            mode="bicubic",
            align_corners=False,
        )
        if patch.shape[-2:] != (8, 8):
            raise ValueError("position interpolation did not produce the expected 8 x 8 grid")
        return torch.cat([self.pos_embed[:, :1], patch.flatten(2).transpose(1, 2)], dim=1)

    def forward(self, images, geometry=None):
        if images.ndim != 4 or tuple(images.shape[1:]) != (12, 64, 64):
            raise ValueError("expected [N,12,64,64] images")
        if geometry is None:
            geometry = images.new_zeros((len(images), 8))
        if geometry.shape != (len(images), 8):
            raise ValueError("expected [N,8] angle-conditioning vectors")
        if not torch.isfinite(images).all() or not torch.isfinite(geometry).all():
            raise ValueError("model inputs must be finite")
        if torch.any((images < 0) | (images > 1)):
            raise ValueError("image values must be scaled to [0,1]")
        rgb = torch.log1p(20 * images[:, :3]) / math.log(21)
        rgb = F.interpolate(rgb, (112, 112), mode="bilinear", align_corners=False)
        mean = rgb.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = rgb.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        tokens = self.patch_embed((rgb - mean) / std)
        extra = torch.log1p(20 * images[:, 3:]) / math.log(21)
        extra = F.interpolate(extra, (112, 112), mode="bilinear", align_corners=False)
        tokens = tokens + self.spectral_patch(extra).flatten(2).transpose(1, 2)
        tokens = self.geometry_film(tokens, geometry)
        tokens = (
            torch.cat([self.cls_token.expand(len(images), -1, -1), tokens], 1) + self.positions()
        )
        for block in self.blocks:
            tokens = block(tokens)
        features = self.norm(tokens)[:, 0]
        return F.normalize(self.head(features), dim=-1)


def load_model(
    checkpoint: str | Path,
    device: str | torch.device = "cpu",
    *,
    allow_missing_adapters: bool = False,
) -> VesselEncoder:
    """Load a plain state dict or a safe checkpoint containing a ``model`` key."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = (
        saved.get("model", saved.get("model_state", saved)) if isinstance(saved, dict) else saved
    )
    if not isinstance(state, dict):
        raise TypeError("checkpoint does not contain a model state_dict")
    position = state.get("backbone.pos_embed", state.get("pos_embed"))
    if not isinstance(position, torch.Tensor) or position.ndim != 3:
        raise ValueError("checkpoint has no compatible position embedding")
    model = VesselEncoder(position.shape[1])
    # Original checkpoints keep backbone parameters below a `backbone.` prefix.
    remapped = {}
    for key, value in state.items():
        key = key.removeprefix("module.")
        key = key.removeprefix("backbone.")
        remapped[key] = value
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    allowed_missing = {
        name
        for name in model.state_dict()
        if allow_missing_adapters and name.startswith(("spectral_patch.", "geometry_film."))
    }
    if unexpected or set(missing) - allowed_missing:
        raise ValueError(f"incompatible checkpoint; missing={missing}, unexpected={unexpected}")
    return model.to(device).eval()
