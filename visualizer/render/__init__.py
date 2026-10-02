"""One renderer per media group.

Each module except `features` exposes `render(target)`, which writes its group into
the target's output directory and returns one status line. A group whose input is
missing, or (except `bundle`) whose files already exist, writes nothing. If the run's
world gate is closed, only `features` is written; if its video gate is closed,
nothing is.

    bundle      bundle.json, bundle.bin -- the point metrics' arrays, packed for WebGL
    index       index.mp4, index_overlay.mp4 -- the per-pixel id map, alone and over the render
    masks       reference_masks.mp4, render_masks.mp4, masks_overlap.mp4 -- the DynamicIoU masks
    depth       reference_depth.mp4, render_depth.mp4, depth_error.mp4
    flow        reference_flow.mp4, render_flow.mp4, flow_error.mp4
    tracks      reference_tracks.mp4, render_tracks.mp4 -- the reference's movers, per side;
                matches.mp4 -- both sides' paths on the case video
    features    reference_<model>.mp4, render_<model>.mp4 -- the patch PCA, per model

`features.render` takes every run at once and loads each backbone once.
"""

from __future__ import annotations

import traceback

from .. import Target

MEDIA = ("bundle", "index", "masks", "depth", "flow", "tracks", "features")
RUN_MEDIA = MEDIA[:-1]          # rendered per run
FEATURES = MEDIA[-1]            # rendered across runs, one model at a time


def renderer(name: str):
    """Import one group's module on first use, so heavy dependencies load only when needed."""

    from importlib import import_module

    return import_module(f".{name}", __name__)


def render(target: Target, media: list[str] | None = None) -> list[str]:
    """Render one run's model-free groups; everything but `features` by default."""

    unknown = sorted(set(media or ()) - set(MEDIA))
    if unknown:
        raise ValueError(f"unknown media {unknown}; known: {list(MEDIA)}")
    wanted = [name for name in RUN_MEDIA if not media or name in media]
    video, world = target.gates
    if not world:
        return [f"{'world' if video else 'video'} gate closed"]
    done = []
    for name in wanted:
        try:
            status = renderer(name).render(target)
        except Exception as error:
            traceback.print_exc()
            done.append(f"{name}: FAILED {type(error).__name__}: {error}")
            continue
        done.append(status)
    return done


def features(targets: list[Target], media: list[str] | None = None) -> list[str]:
    """Render every video-gated run's patch PCA videos, loading each backbone once."""

    if media and FEATURES not in media:
        return []
    return renderer(FEATURES).render([target for target in targets if target.gates[0]])


__all__ = ["FEATURES", "MEDIA", "RUN_MEDIA", "features", "render", "renderer"]
