"""Names and dtypes fixed by the 4DCodeBench world format."""

import numpy as np

CAMERA_FILENAME = "camera.json"
MESHES_DIRNAME = "meshes"
DYNAMICS_DIRNAME = "dynamics"
VIDEO_FILENAME = "render.mp4"
NPZ_SUFFIX = ".npz"

CAMERA_INTRINSICS_KEY = "intrinsics"
CAMERA_EXTRINSIC_KEY = "extrinsic"

# meshes/<id>.npz: one member pair per frame of the object's lifetime
MESH_VERTICES_KEY = "vertices_{frame:04d}"
MESH_FACES_KEY = "faces_{frame:04d}"

# dynamics/<name>.npz: one piece of material
DYNAMICS_POS_KEY = "pos"        # (F, N, 3) float32; NaN outside a column's lifetime
DYNAMICS_IDS_KEY = "ids"        # (M,) uint16; the mesh ids that display this material
DYNAMICS_FACES_KEY = "faces"    # (T, 3) int32; optional
DYNAMICS_TETS_KEY = "tets"      # (K, 4) int32; optional

# columns a liquid tracer archive may hold
TRACER_POINTS = 16384
TRACER_FILENAME = "tracers.npz"

POS_DTYPE = np.float32
FACES_DTYPE = np.int32
INDEX_DTYPE = np.uint16
OBJECT_ID_DTYPE = np.uint16
CAMERA_DTYPE = np.float64
