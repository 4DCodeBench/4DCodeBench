"""Uni3D's point encoder, vendored from github.com/baaivision/Uni3D.

Trimmed to the forward path `encode_pc` takes, with `pointnet2_ops`' farthest
point sampling replaced by the same iterative choice in torch, so the scorer
carries no compiled CUDA extension.
"""

import torch
from torch import nn


def farthest_point_sample(xyz: torch.Tensor, count: int) -> torch.Tensor:
    """`count` indices per cloud, each the point farthest from those already taken."""

    batch, points, _ = xyz.shape
    rows = torch.arange(batch, device=xyz.device)
    index = torch.zeros(batch, count, dtype=torch.long, device=xyz.device)
    distance = torch.full((batch, points), torch.inf, device=xyz.device)
    taken = torch.zeros(batch, dtype=torch.long, device=xyz.device)
    for step in range(count):
        index[:, step] = taken
        offset = xyz - xyz[rows, taken][:, None, :]
        distance = torch.minimum(distance, (offset * offset).sum(dim=-1))
        taken = distance.argmax(dim=-1)
    return index


def fps(data: torch.Tensor, number: int) -> torch.Tensor:
    index = farthest_point_sample(data, number)
    return data.gather(1, index[..., None].expand(-1, -1, data.shape[-1]))


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += (src**2).sum(-1).unsqueeze(-1)
    dist += (dst**2).sum(-1).unsqueeze(-2)
    return dist


def knn_point(nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    return torch.topk(square_distance(new_xyz, xyz), nsample, dim=-1, largest=False, sorted=False)[1]


class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz, color):
        batch_size, num_points, _ = xyz.shape
        center = fps(xyz, self.num_group)
        idx = knn_point(self.group_size, xyz, center)
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = (idx + idx_base).view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        neighborhood_color = color.view(batch_size * num_points, -1)[idx, :]
        neighborhood_color = neighborhood_color.view(
            batch_size, self.num_group, self.group_size, 3).contiguous()
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center, torch.cat((neighborhood, neighborhood_color), dim=-1)


class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(6, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1),
        )

    def forward(self, point_groups):
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 6)
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        return torch.max(feature, dim=2, keepdim=False)[0].reshape(bs, g, self.encoder_channel)


class PointcloudEncoder(nn.Module):
    def __init__(self, point_transformer, args):
        super().__init__()
        self.trans_dim = args.pc_feat_dim
        self.embed_dim = args.embed_dim
        self.group_size = args.group_size
        self.num_group = args.num_group
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.encoder_dim = args.pc_encoder_dim
        self.encoder = Encoder(encoder_channel=self.encoder_dim)
        self.encoder2trans = nn.Linear(self.encoder_dim, self.trans_dim)
        self.trans2embed = nn.Linear(self.trans_dim, self.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.Linear(128, self.trans_dim))
        self.patch_dropout = nn.Identity()
        self.visual = point_transformer

    def forward(self, pts, colors):
        _, center, features = self.group_divider(pts, colors)
        group_input_tokens = self.encoder2trans(self.encoder(features))
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        x = x + torch.cat((cls_pos, self.pos_embed(center)), dim=1)
        x = self.visual.pos_drop(self.patch_dropout(x))
        for blk in self.visual.blocks:
            x = blk(x)
        x = self.visual.fc_norm(self.visual.norm(x[:, 0, :]))
        return self.trans2embed(x)
