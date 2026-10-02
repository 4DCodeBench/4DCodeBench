"""A trimmed vendored copy of the Video Depth Anything network, inference only.

Upstream: https://github.com/DepthAnything/Video-Depth-Anything (Apache-2.0).
Only the modules a forward pass needs are kept. The assembly, the checkpoint
loading and the windowed inference loop are one level up, in
`scorer.estimators.video_depth`.
"""
