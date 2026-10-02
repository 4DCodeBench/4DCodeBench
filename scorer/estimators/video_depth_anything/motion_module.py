"""The temporal attention module of Video Depth Anything, trimmed to inference.

Adapted from DepthAnything/Video-Depth-Anything (Apache-2.0), which in turn adapts
AnimateDiff's `VanillaTemporalModule`. Cross attention, group norm on the attention
input and added key/value projections never fire for the released checkpoint, so only
the temporal self-attention path is kept; the module names still match the weights.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class PositionalEncoding(nn.Module):
    """The fixed sinusoidal frame encoding, stored as a checkpoint buffer."""

    def __init__(self, d_model: int, max_len: int = 32):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden, gate = self.proj(x).chunk(2, dim=-1)
        return hidden * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        inner_dim = dim * mult
        self.net = nn.ModuleList([GEGLU(dim, inner_dim), nn.Identity(), nn.Linear(inner_dim, dim)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            x = module(x)
        return x


class VersatileAttention(nn.Module):
    """Temporal self attention over the frame axis of a flattened token grid."""

    def __init__(self, query_dim: int, heads: int, temporal_max_len: int):
        super().__init__()
        self.heads = heads
        self.head_dim = query_dim // heads
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(query_dim, query_dim, bias=False)
        self.to_v = nn.Linear(query_dim, query_dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(query_dim, query_dim), nn.Identity()])
        self.pos_encoder = PositionalEncoding(query_dim, max_len=temporal_max_len)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        return x.reshape(batch, tokens, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden_states: torch.Tensor, video_length: int) -> torch.Tensor:
        grid = hidden_states.shape[1]
        # (b f) d c -> (b d) f c
        x = hidden_states.unflatten(0, (-1, video_length)).permute(0, 2, 1, 3).flatten(0, 1)
        x = self.pos_encoder(x)

        query, key, value = self._heads(self.to_q(x)), self._heads(self.to_k(x)), self._heads(self.to_v(x))
        out = F.scaled_dot_product_attention(query, key, value)
        out = out.transpose(1, 2).flatten(2)
        for module in self.to_out:
            out = module(out)
        # (b d) f c -> (b f) d c
        return out.unflatten(0, (-1, grid)).permute(0, 2, 1, 3).flatten(0, 1)


class TemporalTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, attention_blocks: int, temporal_max_len: int):
        super().__init__()
        self.attention_blocks = nn.ModuleList(
            [VersatileAttention(dim, heads, temporal_max_len) for _ in range(attention_blocks)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(attention_blocks)])
        self.ff = FeedForward(dim)
        self.ff_norm = nn.LayerNorm(dim)

    def forward(self, hidden_states: torch.Tensor, video_length: int) -> torch.Tensor:
        for attention, norm in zip(self.attention_blocks, self.norms, strict=True):
            hidden_states = attention(norm(hidden_states), video_length) + hidden_states
        return self.ff(self.ff_norm(hidden_states)) + hidden_states


class TemporalTransformer3DModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        heads: int,
        num_layers: int,
        attention_blocks: int,
        temporal_max_len: int,
        norm_num_groups: int = 32,
    ):
        super().__init__()
        self.norm = nn.GroupNorm(norm_num_groups, in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Linear(in_channels, in_channels)
        self.transformer_blocks = nn.ModuleList(
            [
                TemporalTransformerBlock(in_channels, heads, attention_blocks, temporal_max_len)
                for _ in range(num_layers)
            ]
        )
        self.proj_out = nn.Linear(in_channels, in_channels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        video_length = hidden_states.shape[2]
        # b c f h w -> (b f) c h w
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).flatten(0, 1)
        batch, channels, height, width = hidden_states.shape
        residual = hidden_states

        x = self.norm(hidden_states).permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        x = self.proj_in(x)
        for block in self.transformer_blocks:
            x = block(x, video_length)
        x = self.proj_out(x)
        x = x.reshape(batch, height, width, channels).permute(0, 3, 1, 2).contiguous()

        out = x + residual
        # (b f) c h w -> b c f h w
        return out.unflatten(0, (-1, video_length)).permute(0, 2, 1, 3, 4)


class TemporalModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_attention_heads: int = 8,
        num_transformer_block: int = 1,
        num_attention_blocks: int = 2,
        temporal_max_len: int = 32,
    ):
        super().__init__()
        self.temporal_transformer = TemporalTransformer3DModel(
            in_channels=in_channels,
            heads=num_attention_heads,
            num_layers=num_transformer_block,
            attention_blocks=num_attention_blocks,
            temporal_max_len=temporal_max_len,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.temporal_transformer(hidden_states)
