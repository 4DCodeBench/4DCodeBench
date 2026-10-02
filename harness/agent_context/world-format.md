# The World Package Format

The exact format of the world directory you return. TASK.md has the short
version and the rules that cost the most when they are wrong; this is the full
contract. `python -m checker` enforces all of it and prints every violation, and
a world that does not pass the checker scores zero.

Four entries: `camera.json`, `render.mp4`, `meshes/`, `dynamics/`. Plus
`solver/` for a liquid whose solver keeps no particle identity. Coordinates are
world-space metres, Z-up. All frame-indexed data share one timeline of `F`
frames, `F` being the frame count of `render.mp4`: frame `t` of the video, of
every `dynamics/` array and of every mesh archive is the same instant.

```
world/
  camera.json          {"intrinsics": K, "extrinsic": E}
  render.mp4           the rendered video
  meshes/<id>.npz      vertices_TTTT (V_t,3) float32, faces_TTTT (T_t,3) int32
  dynamics/<name>.npz  pos (F,N,3) float32, ids (M,) uint16; faces (T,3) int32 for a mesh, tets (K,4) int32 for a tetrahedral mesh
  solver/manifest.json  and the solver's own cache, for an identity-less liquid only
```

## `camera.json`

`camera.json` carries `intrinsics`, a 3×3 pinhole matrix in pixels, and
`extrinsic`, a 4×4 camera-to-world transform in metres. Both are shared by every
frame: the camera does not move. Camera axes are +x right, +y down, +z
forward, so a world point `p` projects proportionally to `K R^T (p - t)`; pixel
`(i, j)` has centre `(i+0.5, j+0.5)` and the principal point is `(W/2, H/2)`.
The scorer projects your geometry exactly this way, so getting it wrong shows up
as geometry sitting off the render.

## `meshes/`

One archive per rendered object, `<id>.npz`, `<id>` a positive integer unique
within the world. **Every rendered object needs one, including a liquid.** One
member pair per frame of the object's lifetime, which is one contiguous interval
below `F`: `vertices_TTTT` `(V, 3)` float32, finite world-space metres, and
`faces_TTTT` `(T, 3)` int32 indexing that frame's vertices; `TTTT` is the
zero-padded frame index. A frame where the object has no geometry holds empty
arrays. Vertex count and topology may change between frames; the archive
carries no cross-frame correspondence.

An object no `dynamics/` archive lists is static: identical `vertices` and
`faces` at every frame of its lifetime. The exception is an identity-less liquid,
which is listed by no archive and is not static; its ids go in
`solver/manifest.json` under `liquid_ids`.

## `dynamics/`

The state your description produced: one archive per piece of moving material,
any unique file name. Every rendered object that moves is listed in the `ids` of
some archive, except an identity-less liquid, which is named in
`solver/manifest.json` instead.

- `pos` `(F, N, 3)` float32. Column `j` is one piece of material through the whole
  video: `pos[t, j]` is its position at frame `t`, finite over one contiguous
  interval of at least one frame (its lifetime) and NaN outside it. A column is
  the same material at every frame it exists; it is not rebuilt per frame.
- `ids` `(M,)` uint16, `M ≥ 1`: the `meshes/` ids that display this material. One
  material may be displayed by several objects, and one object may display
  several materials.
- Connectivity, the same at every frame, indexing columns of `pos`. At most one
  member, `faces` or `tets`, according to how your dynamics represent the
  material:
  - surface mesh (rigid body, cloth, shell, soft-body surface): `pos` holds the
    dynamics mesh's vertices, `faces` `(T, 3)` int32 its triangles;
  - tetrahedral mesh (FEM): `pos` its vertices, `tets` `(K, 4)` int32 its
    tetrahedra;
  - particles (MPM, SPH, granular, PBD): `pos` holds every particle by slot, no
    connectivity member; a slot is NaN where the particle is not active;
  - a liquid whose solver keeps no particle identity, such as Blender's fluid
    (Mantaflow), which reseeds its particles every frame: write no archive and
    build no tracer columns. Ship the bake under `solver/` and the tracks are
    extracted from it afterwards. Its per-frame mesh still goes in `meshes/`
    like anything else, and every other moving object still needs its archive.

Every `pos` array has exactly `F` rows, and every mesh frame index falls below
`F`.

## `solver/`

The solver's own bake, so the tracks can be extracted afterwards. Required only
for an identity-less liquid; omit it otherwise.

```
world/solver/
  manifest.json        one JSON object, the seven members below
  <cache>/             the solver's cache directory, exactly as written
```

For Blender's `FLUID` domain: set `cache_type = "ALL"` before baking, or nothing
is written to disk and there is no bake to ship. Bake the full timeline, copy the
`cache_directory` whole with its `config/`, `data/` and `mesh/` subdirectories,
and do not clear it afterwards.

`manifest.json` carries the seven things the cache does not:

- `cache`, the directory name;
- `fps` and `time_scale`, the domain's own, so one frame is known to advance
  `time_scale / fps` seconds of solver time;
- `metres_per_unit`;
- `domain_min`, the domain object's own box corner in world metres. Take it from
  the object's box, not its evaluated mesh, which is the liquid surface and
  shrinks with the pool;
- `frames`, first and last covered;
- `liquid_ids`, the `meshes/` ids rendering it.

## `render.mp4`

An H.264 MP4 with the same frame count, resolution and frame rate as the input
video, rendered through `camera.json`. Those four are checked before anything
else is looked at, and a world that misses any of them is refused whole: not one
metric is read and the attempt scores nothing. So confirm what you actually
wrote, rather than what you meant to write:

```bash
ffprobe -v error -select_streams v:0 -count_frames \
        -show_entries stream=nb_read_frames,width,height,r_frame_rate \
        -of csv=p=0 world/render.mp4
```

and check it against the same reading of the input video. At every frame each
rendered object is exactly the triangle mesh stored under its id in `meshes/`,
and no geometry appears that is not in `meshes/`, so convert curves, volumes and
point clouds to triangles before rendering them. Lights, materials and world
shading are not geometry and are yours to choose.

## What the checker checks

`python -m checker` prints every violation of the format above: missing entries,
wrong dtypes and shapes, indices out of range, lifetimes that are not
contiguous, a rotation block that is not orthonormal, a frame count that
disagrees with the video, an id in `ids` with no mesh archive, a static object
whose arrays change.
