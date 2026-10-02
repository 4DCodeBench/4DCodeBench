"""Uni3D, vendored from github.com/baaivision/Uni3D: the point tower alone."""

import numpy as np
import timm
import torch
from torch import nn

from .point_encoder import PointcloudEncoder


class Uni3D(nn.Module):
    def __init__(self, point_encoder):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.point_encoder = point_encoder

    def encode_pc(self, pc):
        return self.point_encoder(pc[:, :, :3].contiguous(), pc[:, :, 3:].contiguous())


def create_uni3d(args) -> Uni3D:
    point_transformer = timm.create_model(
        args.pc_model, checkpoint_path=args.pretrained_pc, drop_path_rate=args.drop_path_rate)
    return Uni3D(point_encoder=PointcloudEncoder(point_transformer, args))
