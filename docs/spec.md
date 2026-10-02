# World Format and Evaluation Specification

This document defines the 4D world package format and how a run is scored. It is not part of the agent image. Benchmark agents receive a self-contained task summary in `/task/` (`TASK.md`, `world-format.md`, and `render-settings.md`), `/input/reference.mp4`, and the public checker (`python -m checker`).

## World directory

```text
world/
├── camera.json
├── render.mp4
├── meshes/
│   └── <id>.npz
├── dynamics/
│   └── <name>.npz
└── solver/            (identity-less liquid only)
    ├── manifest.json
    └── <cache>/
```

| entry | contents |
|---|---|
| `camera.json` | `{"intrinsics": K, "extrinsic": E}` |
| `render.mp4` | the rendered video |
| `meshes/<id>.npz` | the geometry shown in `render.mp4`, per object: `vertices_TTTT` `(V_t, 3)` float32, `faces_TTTT` `(T_t, 3)` int32 |
| `dynamics/<name>.npz` | `pos` `(F, N, 3)` float32, `ids` `(M,)` uint16; optionally `faces` `(T, 3)` or `tets` `(K, 4)` int32 |
| `solver/` | `manifest.json` and the solver's cache, only for a liquid whose solver keeps no particle identity |

- Coordinates are world-space metres, Z-up.
- All frame-indexed data share a synchronized timeline of $F$ frames (the frame count of `render.mp4`): frame $t$ across `render.mp4`, `dynamics/`, and `meshes/` describes the same instant.

## `camera.json`

A UTF-8 JSON object with `intrinsics`, one 3×3 pinhole matrix, and `extrinsic`, one 4×4 camera-to-world transform constant across all frames (static camera). Pixel $(i, j)$ covers $[i, i+1) \times [j, j+1)$ with centre $(i+0.5,\, j+0.5)$; the centred principal point is $(W/2,\, H/2)$. Camera axes are +x right, +y down, +z forward. A world point $p$ projects to image coordinates via $K R^\top (p - t)$, where $R$ and $t$ are the rotation and translation components of `extrinsic`.

## `meshes/`

One archive per rendered object, named `<id>.npz` with `<id>` a positive integer unique within the world. An archive holds `vertices_TTTT` with shape `(V, 3)`, dtype `float32`, finite world-space metres, and `faces_TTTT` with shape `(T, 3)`, dtype `int32`, indexing that frame's vertices, for every frame `TTTT` (four-digit, zero-padded) of one contiguous interval of frames below $F$. A frame at which the object has no geometry holds empty arrays. Vertex count and topology are free to change between frames; an archive carries no cross-frame correspondence.

Any object ID not referenced in `dynamics/` is classified as static: its `vertices` and `faces` must be identical at every frame of its active interval.

## `dynamics/`

One archive per piece of material, under any file name unique within the directory. `pos` has shape `(F, N, 3)` and dtype `float32`; column $j$ traces one material point across the entire sequence, `pos[t, j]` its position at frame $t$, finite over one contiguous interval of at least one frame and NaN elsewhere. `ids` has shape `(M,)`, dtype `uint16`, $M \ge 1$, and lists the ids of the `meshes/` archives that display this material; an id may appear in several archives.

All moving geometry must be referenced in the `ids` of at least one `dynamics/` archive. An archive holds at most one connectivity member, `faces` or `tets`, fixed across all frames and indexing columns of `pos`:

| material | `pos` | connectivity |
|---|---|---|
| surface mesh (rigid body, cloth, shell, soft-body surface) | simulation mesh vertices | `faces` `(T, 3)` int32 |
| tetrahedral mesh (FEM) | mesh vertices | `tets` `(K, 4)` int32 |
| particles (MPM, SPH, granular, PBD) | every particle by slot, NaN where the slot is inactive | none |
| liquid without particle identity (e.g. Mantaflow, which reseeds every frame) | no archive; the solver bake goes under `solver/` | none |

For identity-less liquids, the scorer advects up to `TRACER_POINTS` (16,384) tracer columns from the simulation cache at scoring time. Its surface mesh is stored in `meshes/`, and all other moving objects require their own `dynamics/` archives.

## `solver/`

Present only for an identity-less liquid. `manifest.json` is one JSON object with seven members:

| member | value |
|---|---|
| `cache` | name of the solver cache directory beside `manifest.json` |
| `fps` | the domain's frame rate, positive and finite |
| `time_scale` | the domain's time scale, positive and finite; advances solver time by `time_scale / fps` seconds per frame |
| `metres_per_unit` | positive and finite |
| `domain_min` | three finite numbers: the corner of the domain object's bounding box in world metres (the domain object box, not the evaluated liquid surface) |
| `frames` | `[first, last]` frames covered, `first <= last` |
| `liquid_ids` | non-empty list of `meshes/` ids that render the liquid |

An ID in `liquid_ids` must not appear in any `dynamics/` archive and is exempt from the static-object consistency check.

## `render.mp4`

An H.264 MP4 rendered by Blender Cycles, with the frame count, resolution, and frame rate of the input `reference.mp4`. Benchmark reference videos use integer frame rates. Every object in the rendered scene is exported to `meshes/` under its ID, so that `meshes/`, seen through `camera.json`, matches `render.mp4` frame by frame, with nothing rendered that is not exported. Non-mesh elements (curves, point clouds, particles, hair, and volumes) must be converted to triangle meshes before rendering and exported as such.

## Evaluation

A reconstruction is scored against the reference video on every scene, and against the reference world on synthetic scenes. The main metrics form five families:

| family | reference | metrics |
|---|---|---|
| Perceptual | video | DINOv3 similarity |
| 2D Dynamics | video | Dynamic IoU; Flow and Track2D (real) |
| 2.5D Geometry | video | Depth error (real) |
| 3D Geometry | world | Scene 3D (synthetic) |
| 3D Dynamics | world | Trajectory DTW, EMD step (synthetic) |

The paper's Overall score averages the five families. [Additional metrics](#additional-metrics) and [geometry quality](#geometry-quality) are reported separately. The scorer writes per-metric readings only.

### Executability and scoring

Two validation gates determine scoring eligibility:

| gate | passes when | on failure |
|---|---|---|
| video | `render.mp4` decodes and matches the reference video's resolution, frame count and frame rate | scoring halts; all metrics receive `error` |
| world | every world entry and `solution/build.sh` exist, and the [checker](#checker) reports no violation | geometry metrics receive `error`; DINOv3, TIPS and GeoPhys, which read `render.mp4` only, are still computed |

Each metric yields a numerical score or one of four status states (in order of precedence):

| state | meaning |
|---|---|
| `not_applicable` | the metric does not apply to this kind of scene: reference-world metrics on real scenes; Flow, Track2D, Depth error and Uni3D MoGe on synthetic scenes |
| `excluded` | the scene's primary object is transparent (six real scenes) and the metric compares rasterised geometry with the video: Dynamic IoU, Flow, Track2D, Depth error, Uni3D MoGe |
| `error` | a validation gate failed or the reconstruction data is defective; `reward.detail.json` records the cause |
| `uncomputable` | the reference lacks the required ground-truth signal (e.g. no moving points); omitted across all models |

When computing aggregates, `error` is assigned the metric's worst possible value: 0 for similarities and 2 for Depth error.

### Sampled timeline

DINOv3, TIPS, GeoPhys, Flow, Track2D and Depth error read the frames $0, s, 2s, \dots$, with stride $s$ the smallest integer such that $\lceil F/s \rceil \le 300$; a video of at most 300 frames is read at every frame. $T$ denotes the number of frames read, and $k$ indexes them. The `prepare` stage computes the reference estimates on the same timeline.

### Metrics on all scenes

Let $x_k$ and $\hat x_k$ be frame $k$ of the reference and rendered videos.

**DINOv3 similarity** (`semantic_dinov3`). With $e_k$ and $\hat e_k$ the DINOv3 class-token embeddings of $x_k$ and $\hat x_k$, the mean frame-wise cosine similarity mapped to $[0, 1]$:
$$\mathrm{DINOv3}=\frac12\Big(1+\frac1T\sum_k \frac{e_k^\top \hat e_k}{\|e_k\|\,\|\hat e_k\|}\Big).$$

**Dynamic IoU** (`dynamic_iou`). $\hat M_k$ is the rasterised mask of the reconstruction's dynamic objects and $M_k$ the reference dynamic mask, rasterised from the reference world or annotated by hand for real videos. A thin band $B_k$ along the reference boundary is ignored, as in PASCAL VOC. On real scenes the reconstruction's mask is morphologically closed first, so granular material covers the region it fills.
$$\mathrm{DynIoU}=\frac1T\sum_k\frac{|(M_k\cap\hat M_k)\setminus B_k|}{|(M_k\cup\hat M_k)\setminus B_k|}.$$

### Metrics on real videos

The reference side is estimated from `reference.mp4` with off-the-shelf models; the reconstruction side is computed analytically from the 4D world, not estimated from its render.

**Depth error** (`depth_error`, lower is better). $\tilde d_k$ is the Video Depth Anything disparity of $x_k$ and $\tilde{\hat d}_k$ the disparity of the reconstruction's meshes rasterised through its camera, each normalised by its median and mean absolute deviation, as in MiDaS. On the covered pixels, $E_k$ is the mean absolute difference with the largest 1% of residuals discarded. Uncovered pixels count as error 1, so with coverage $c_k$:
$$\mathrm{DepthErr}=\frac1T\sum_k\big(c_kE_k+(1-c_k)\big)\in[0,2].$$

**Flow** (`flow_distribution`). $U_k$ is the RAFT flow of the reference and $\hat U_k$ the reconstruction's analytic flow between frames $k$ and $k+1$ of the sampled timeline, at pixels valid on both sides that move more than 0.5 px on either. $\mathcal K$ is the set of steps with such pixels. At each step the two flows are compared as unordered sets by sliced Wasserstein distance, normalised by the RMS magnitudes $m$ of both sides:
$$\mathrm{Flow}=1-\frac1{|\mathcal K|}\sum_{k\in\mathcal K}\min\Big(1,\frac{\mathrm{SW}(U_k,\hat U_k)}{m(U_k)+m(\hat U_k)}\Big).$$
A step where the reference exhibits motion but the reconstruction contains no geometry receives a penalty value of 1.

**Track2D** (`track2d_dtw`). CoTracker3 tracks a grid of query points through the reference video. On the reconstruction, the surface point under each query is followed through the 4D world and projected, so path $i$ on one side corresponds to path $i$ on the other. The $n$ reference paths kept are those with a textured seed that travel more than 2% of the image diagonal $\delta$ after camera-motion compensation (the grid's shared drift removed). $D_i$ is the per-step DTW distance between corresponding paths in units of $\delta$, capped at $\kappa=0.03$:
$$\mathrm{Track2D}=1-\frac1n\sum_i\frac{\min(D_i,\kappa)}{\kappa}.$$

### Metrics on synthetic scenes

The metrics below compare the reconstruction with the reference world in 3D, after registration.

**Registration.** $\mathcal P$ is the point cloud of surfaces visible at frame 0 of the reference world, with RMS radius $\sigma$; all 3D distances are in units of $\sigma$. Two similarity transforms are fitted by trimmed ICP, one on the two worlds' visible surfaces at frame 0 and one on their dynamic objects' surfaces over 16 frames. The transformation that minimizes registration error to $\mathcal P$ under the reference camera viewpoint is selected.

**Scene 3D** (`scene_3d`). $\hat{\mathcal P}$ is the registered reconstruction's frame-0 surface seen through the reference camera, and $D_\kappa$ the symmetric Chamfer distance with per-point distances capped at $\kappa=0.5$:
$$\mathrm{Scene3D}=1-D_\kappa(\mathcal P,\hat{\mathcal P})/\kappa.$$

**Trajectory DTW** (`trajectory_dtw`), Lagrangian view. Dynamic matter of both worlds is sampled at one voxel size and each sample is followed over its lifetime, giving up to 8192 3D paths $\{\tau_i\}$, $\{\hat\tau_j\}$ per side. A one-to-one assignment $\pi$ is solved by the Hungarian algorithm on DTW distances, capped at $\kappa=1$:
$$\mathrm{TrajDTW}=1-\frac1n\sum_i\frac{\min\big(\mathrm{DTW}(\tau_i,\hat\tau_{\pi(i)}),\kappa\big)}{\kappa}.$$

**EMD step** (`emd_step`), Eulerian view. $U_k$ and $\hat U_k$ are the one-step 3D displacements of the samples at frame $k$, compared as unordered sets by sliced Wasserstein distance. Normalising by the total motion over the video keeps near-static frames from dominating:
$$\mathrm{EMD\ step}=\max\Big(0,\,1-\frac{\sum_k\mathrm{SW}(U_k,\hat U_k)}{\sum_k\big(m(U_k)+m(\hat U_k)\big)}\Big).$$

### Additional metrics

**TIPS similarity** (`semantic_tips`, all scenes). DINOv3 similarity with TIPSv2 embeddings in place of DINOv3.

**GeoPhys DINOv3** (`geophys_dinov3`, all scenes). Each video becomes a trajectory of DINOv3 layer-18 patch tokens, spatially averaged. Five regularity descriptors $\phi$ of the trajectory (GeoPhys) are compared as $1-\frac15\sum_\phi|\phi-\hat\phi|/(\phi+\hat\phi)$.

**Uni3D point** (`uni3d_point_scene`, synthetic). Cosine similarity of the Uni3D embeddings of the registered frame-0 clouds of Scene 3D, mapped to $[0, 1]$ as $(1+\cos)/2$.

**Uni3D MoGe** (`uni3d_moge_scene`, real). Cosine similarity of the Uni3D embeddings of the reconstruction's frame-0 visible surface and the MoGe-3 point map of $x_0$, mapped to $[0, 1]$ as $(1+\cos)/2$.

**Occupancy DTW** (`occupancy_dtw`, synthetic). The two worlds' matter is sampled as point sets at 16 evenly spaced frames. Every reference frame is compared with every reconstruction frame by sliced Wasserstein distance capped at 0.5; the score is one minus the per-step DTW cost of that matrix divided by the cap. No particle identity is required.

### Geometry quality

Computed from the reconstructed world alone, over the runs that pass the world gate: the fraction of mesh components across all frames that are watertight, manifold, non-degenerate and free of self-intersections (`mesh_watertight`, `mesh_manifold`, `mesh_clean_faces`, `mesh_no_self_intersection`), and no interpenetration (`interpenetration`), one minus the fraction of vertices inside another object.

### Output

Each run's `results/` holds:

- `reward.json`: one entry per metric, a number or a state.
- `reward.detail.json`: intermediate readings, strides, the fitted registration, and the cause of every non-number.
- the arrays behind each metric (`.npz`, `.h5`).

## Checker

`python -m checker` inspects `/workspace/world` and `/workspace/solution`, takes no arguments, reports all format violations, and exits with a non-zero code on failure. It checks:

- the four entries exist and are readable;
- `camera.json`: shapes, finiteness, and a proper orthonormal rotation block in `extrinsic`;
- `meshes/`: member names, dtypes, shapes, face indices, finiteness and contiguous frame intervals;
- `dynamics/`: dtypes, shapes, index ranges and per-column lifetimes;
- every listed id has a mesh archive;
- $F$ agrees between `render.mp4` and every `pos`, and every mesh frame is below $F$;
- a static object's arrays are identical across its interval;
- `solver/`, if present: the seven `manifest.json` members are valid, and `cache` and `liquid_ids` resolve;
- `solution/` holds an executable `build.sh` and only `.py` / `.sh` files.

`python -m checker.preview` writes `/workspace/tmp/preview_0000.png`: frame 0 of `render.mp4` beside frame 0 of `meshes/` z-buffered through `camera.json` (one color per ID). Because it uses the scorer's projection, it visually verifies camera alignment before delivery.
