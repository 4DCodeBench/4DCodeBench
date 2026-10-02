"""Constants of the scoring rules."""

N = 16384          # target material sample count of the reference's frame 0
SEED = 0           # random seed of the metric samplers
BAND_FRACTION = 0.0025   # DynamicIoU ignored band along the gt boundary, fraction of diagonal
CLOSE_FRACTION = 0.0025  # real-case DynamicIoU closing radius of the submission mask, same unit
