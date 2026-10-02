"""File layout of the reference-side estimates of a case's `reference.mp4`.

`python -m scorer.prepare` computes these once per case; the scorer loads them:

    data/<kind>/<case>/estimates/flow.h5     RAFT flow, `flow (F'-1, H, W, 2)`
                                              float16 in full-resolution pixels,
                                              `valid (F'-1, H, W)` bool, `stride`
    data/<kind>/<case>/estimates/tracks.npz   CoTracker3 tracks of a full-image
                                              grid on frame 0, `tracks (Q, F', 2)`
                                              float32 full-resolution pixels,
                                              `visible (Q, F')`, the start pixels
                                              `queries (Q, 3)` `(0, x, y)`,
                                              `step`, `shape`, `stride`
    data/<kind>/<case>/estimates/moge.npz     MoGe-3 point map of frame 0,
                                              `points (H, W, 3)` float32 in
                                              camera metres up to the model's own
                                              scale, `mask (H, W)` bool,
                                              `intrinsics (3, 3)` normalised to
                                              the unit image, `shape [H, W]`,
                                              `model`
    data/<kind>/<case>/estimates/semantic_<name>.npz
                                              one per semantic backbone, its
                                              `embeddings (n, D)` float32 and the
                                              `stride` they are on
    data/<kind>/<case>/estimates/geophys_<name>.npz
                                              `trajectory (T, 1024)` float32,
                                              `stride`, `layer`
    data/<kind>/<case>/estimates/depth.h5    Video Depth Anything `disparity
                                              (F', h, w)` float16 on the model's
                                              own output grid, `shape [F, H, W]`
                                              the video's own, `fps`, `stride`

All archives except the point map are on the case's sampled timeline
(`config.sampling`): `F'` is `ceil(F / stride)`, flow step `k` is the displacement
from sampled frame `k` to `k + 1`, and each archive stores its `stride`. The point
map covers frame 0 only.

The score of a case's reference world against itself (the per-metric ceiling):

    data/<kind>/<case>/sanity/results/        reward.json and the metrics' arrays
"""

ESTIMATES_DIRNAME = "estimates"
SANITY_DIRNAME = "sanity"
RESULTS_DIRNAME = "results"
REWARD_FILENAME = "reward.json"
VIZ_DIRNAME = "viz"

FLOW_FILENAME = "flow.h5"
TRACKS_FILENAME = "tracks.npz"
MOGE_FILENAME = "moge.npz"
SEMANTIC_FILENAME = "semantic_{name}.npz"
GEOPHYS_FILENAME = "geophys_{name}.npz"
DEPTH_FILENAME = "depth.h5"
RASTER_DEPTH_FILENAME = "raster_depth.h5"

MOGE_POINTS_KEY = "points"
MOGE_MASK_KEY = "mask"
