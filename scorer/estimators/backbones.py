"""Image backbones that embed video frames: DINOv3 and TIPSv2.

Both feed the Semantic similarities; DINOv3 also feeds GeoPhys. Each embeds frames
independently and exposes:

    load(device, root)      the encoder, built once and cached
    preprocess(...)         uint8 frames -> the normalised batch the weights expect
    tokens(...)             global embeddings and their patch grids
    embed(...)              the global embeddings alone

`load` is an `lru_cache` per backbone; `backbone.load.cache_clear()` releases the
weights. Checkpoints load offline from the store named in `config.models`.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from transformers import AutoModel

from config import models

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Backbone:
    """A pretrained encoder with a cached, clearable `load`."""

    def __init__(self):
        self.load = lru_cache(maxsize=1)(self._load)

    def _load(self, device: str = DEVICE, root: str | None = None) -> torch.nn.Module:
        raise NotImplementedError


def _normalised(batch: torch.Tensor, mean, std, device: str) -> torch.Tensor:
    """Normalise a `(N, 3, H, W)` batch in `[0, 1]` in place by per-channel mean and std."""

    centre = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    spread = torch.tensor(std, device=device).view(1, 3, 1, 1)
    return batch.sub_(centre).div_(spread)


class DINOv3(Backbone):
    """DINOv3 ViT-L/16 frame embeddings, loaded with transformers' `AutoModel`.

    The global embedding is the final class token (`pooler_output`); the patch grid
    is the tokens after the class and register tokens.
    """

    IMAGE_SIZE = 224
    PATCH_SIZE = 16
    MEAN = IMAGENET_MEAN
    STD = IMAGENET_STD

    def _load(self, device: str = DEVICE, root: str | None = None) -> torch.nn.Module:
        model = AutoModel.from_pretrained(str(models.checkpoint(models.DINOV3, root)),
                                          local_files_only=True)
        return model.eval().to(device)

    def preprocess(self, frames: np.ndarray, device: str = DEVICE,
                   size: tuple[int, int] | None = None) -> torch.Tensor:
        """Resize RGB uint8 frames to `IMAGE_SIZE` (or `size`) and ImageNet-normalise them.

        With rotary position embeddings, any `size` whose sides are multiples of
        `PATCH_SIZE` is valid.
        """

        batch = torch.from_numpy(np.ascontiguousarray(frames)).to(device)
        batch = batch.permute(0, 3, 1, 2).float().div_(255.0)
        batch = torch.nn.functional.interpolate(
            batch, size or (self.IMAGE_SIZE, self.IMAGE_SIZE), mode="bilinear",
            align_corners=False, antialias=True
        )
        return _normalised(batch, self.MEAN, self.STD, device)

    @torch.no_grad()
    def tokens(
        self,
        frames: np.ndarray,
        device: str = DEVICE,
        size: tuple[int, int] | None = None,
        block: int | None = None,
        root: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return global embeddings `(N, D)` and patch grids `(N, gh, gw, D)`.

        `block` indexes transformer blocks from the end (`-1` is the last); the grid
        is taken at that block, passed through the final norm. The global
        embedding is always the final layer's.
        """

        model = self.load(device, root)
        pixels = self.preprocess(frames, device, size)
        output = model(pixel_values=pixels, output_hidden_states=block is not None)
        final = block is None or block == -1
        sequence = output.last_hidden_state if final else model.norm(output.hidden_states[block])
        prefix = 1 + int(model.config.num_register_tokens)
        grid = (pixels.shape[2] // self.PATCH_SIZE, pixels.shape[3] // self.PATCH_SIZE)
        patches = sequence[:, prefix:].unflatten(1, grid)
        return output.pooler_output.float(), patches.float()

    def embed(self, frames: np.ndarray, device: str = DEVICE,
              root: str | None = None) -> torch.Tensor:
        """Return the global embeddings `(N, D)` of RGB uint8 frames."""

        return self.tokens(frames, device, root=root)[0]


def pooled(backbone: Backbone, frames: np.ndarray, block: int | None = None,
           device: str = DEVICE, root: str | None = None) -> torch.Tensor:
    """Return `(N, D)`: the spatial mean of each frame's patch tokens at `block`.

    This is GeoPhys's per-frame vector `z̄ₜ = (1/N) Σₙ zₜ,ₙ`, taken at the readout
    layer (`config.models.block_for`).
    """

    grid = backbone.tokens(frames, device, block=block, root=root)[1]
    return grid.flatten(1, 2).mean(1)


class TIPSv2(Backbone):
    """TIPSv2 L/14 frame embeddings, loaded with `AutoModel(trust_remote_code=True)`.

    The model code ships with the checkpoint. Input pixels are in `[0, 1]` with no
    further normalisation. The global embedding is the class token; the patch grid
    is the patch tokens.
    """

    IMAGE_SIZE = 448
    PATCH_SIZE = 14

    def _load(self, device: str = DEVICE, root: str | None = None) -> torch.nn.Module:
        model = AutoModel.from_pretrained(str(models.checkpoint(models.TIPS, root)),
                                          local_files_only=True, trust_remote_code=True)
        return model.eval().to(device)

    def preprocess(self, frames: np.ndarray, device: str = DEVICE,
                   size: tuple[int, int] | None = None) -> torch.Tensor:
        """Resize RGB uint8 frames to `IMAGE_SIZE` (or `size`) as a `[0, 1]` batch.

        The encoder interpolates its position embeddings, so any `size` whose sides
        are multiples of `PATCH_SIZE` is valid.
        """

        batch = torch.from_numpy(np.ascontiguousarray(frames)).to(device)
        batch = batch.permute(0, 3, 1, 2).float().div_(255.0)
        return torch.nn.functional.interpolate(
            batch, size or (self.IMAGE_SIZE, self.IMAGE_SIZE), mode="bilinear",
            align_corners=False, antialias=True
        )

    @torch.no_grad()
    def tokens(
        self,
        frames: np.ndarray,
        device: str = DEVICE,
        size: tuple[int, int] | None = None,
        block: int | None = None,
        root: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return global embeddings `(N, D)` and patch grids `(N, gh, gw, D)`.

        `block` indexes transformer blocks from the end (`-1` is the last); a forward
        hook captures that block's output, which is passed through the final norm
        and stripped of class and register tokens. The global embedding is always
        the final layer's.
        """

        model = self.load(device, root)
        pixels = self.preprocess(frames, device, size)
        grid = (pixels.shape[2] // self.PATCH_SIZE, pixels.shape[3] // self.PATCH_SIZE)
        if block is None:
            output = model.encode_image(pixels)
            return output.cls_token[:, 0].float(), output.patch_tokens.unflatten(1, grid).float()

        encoder = model.vision_encoder
        caught: list[torch.Tensor] = []
        handle = encoder.blocks[block].register_forward_hook(lambda *args: caught.append(args[-1]))
        try:
            output = model.encode_image(pixels)
        finally:
            handle.remove()
        prefix = 1 + int(encoder.num_register_tokens)
        patches = encoder.norm(caught[0])[:, prefix:].unflatten(1, grid)
        return output.cls_token[:, 0].float(), patches.float()

    def embed(self, frames: np.ndarray, device: str = DEVICE,
              root: str | None = None) -> torch.Tensor:
        """Return the global embeddings `(N, D)` of RGB uint8 frames."""

        return self.tokens(frames, device, root=root)[0]


DINOV3 = DINOv3()
TIPS = TIPSv2()

__all__ = ["DEVICE", "DINOV3", "TIPS", "Backbone", "DINOv3", "TIPSv2", "pooled"]
