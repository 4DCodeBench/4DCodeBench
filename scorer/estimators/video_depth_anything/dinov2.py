"""DINOv2 vision transformer, trimmed to the inference path Video Depth Anything uses.

Adapted from facebookresearch/dinov2 (Apache-2.0) as vendored by the official
Video-Depth-Anything repository. Only the layers that carry checkpoint weights and
`get_intermediate_layers` are kept; training, chunked blocks and registers are dropped.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1.0):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        out = F.scaled_dot_product_attention(query, key, value)
        return self.proj(out.transpose(1, 2).reshape(batch, tokens, dim))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, init_values: float = 1.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.ls1 = LayerScale(dim, init_values)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.ls2 = LayerScale(dim, init_values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        return x + self.ls2(self.mlp(self.norm2(x)))


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class DinoVisionTransformer(nn.Module):
    """The plain (register-free) DINOv2 backbone, in evaluation mode only."""

    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        init_values: float = 1.0,
        interpolate_antialias: bool = False,
        interpolate_offset: float = 0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.patch_embed = PatchEmbed(patch_size, 3, embed_dim)
        patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, patches + 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio, init_values) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        stored = self.pos_embed.shape[1] - 1
        if npatch == stored and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        side = int(math.sqrt(stored))
        kwargs = {}
        if self.interpolate_offset:
            kwargs["scale_factor"] = (
                float(w0 + self.interpolate_offset) / side,
                float(h0 + self.interpolate_offset) / side,
            )
        else:
            kwargs["size"] = (w0, h0)
        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, side, side, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens(self, x: torch.Tensor) -> torch.Tensor:
        _, _, w, h = x.shape
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        return x + self.interpolate_pos_encoding(x, w, h)

    def get_intermediate_layers(
        self, x: torch.Tensor, n: list[int], return_class_token: bool = True
    ) -> tuple:
        """The normalised (patch tokens, class token) pairs of the requested blocks."""

        wanted = sorted(n)
        tokens = self.prepare_tokens(x)
        outputs = []
        for index, block in enumerate(self.blocks):
            tokens = block(tokens)
            if index in wanted:
                outputs.append(self.norm(tokens))
        patches = [out[:, 1:] for out in outputs]
        classes = [out[:, 0] for out in outputs]
        if return_class_token:
            return tuple(zip(patches, classes, strict=True))
        return tuple(patches)


CONFIGS = {
    "vits": dict(embed_dim=384, depth=12, num_heads=6),
    "vitb": dict(embed_dim=768, depth=12, num_heads=12),
    "vitl": dict(embed_dim=1024, depth=24, num_heads=16),
}


def DINOv2(model_name: str = "vitl") -> DinoVisionTransformer:
    return DinoVisionTransformer(**CONFIGS[model_name])
