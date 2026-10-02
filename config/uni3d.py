"""Checkpoint path and build arguments of the Uni3D-L point tower.

The model is vendored under `scorer.metrics.uni3d` and embeds a cloud as a
1024-dimensional vector; the Uni3D metrics compare two embeddings by cosine.
"""

from pathlib import Path

from easydict import EasyDict

UNI3D_CHECKPOINT = Path("uni3d-l/model.pt")   # relative to the checkpoint store

UNI3D_ARGS = EasyDict(
    pc_model="eva02_large_patch14_448",
    pretrained_pc="",
    drop_path_rate=0.0,
    pc_feat_dim=1024,
    embed_dim=1024,
    group_size=64,
    num_group=512,
    pc_encoder_dim=512,
)

UNI3D_POINTS = 10000   # points per embedded cloud
UNI3D_COLOUR = 0.4     # constant RGB value given to every point
