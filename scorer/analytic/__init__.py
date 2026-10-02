"""Submission-side optical flow and point tracks, computed from the world's geometry.

Flow and tracks are computed from a submission's meshes, material columns and
camera; estimators run only on the case's reference video. `advection` seeds
and advects the tracers of an identity-less liquid.
"""
