"""File layout of a real case's dynamic-object annotation.

`data/real/<case>/annotation/dynamic_mask.npz` holds `mask`, uint8 of shape
`(frames, height, width)`: 0 is background, 1 is dynamic.
"""

ANNOTATION_DIRNAME = "annotation"
MASK_FILENAME = "dynamic_mask.npz"
MASK_KEY = "mask"
