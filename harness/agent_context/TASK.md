`input.mp4` is a video of a dynamic scene.
Your task is to write graphics code that, when rendered, reproduces the observed 4D scene.

Your reconstruction should model the scene and its dynamics explicitly.
It should be both visually and physically consistent with the target.
Render the result in Blender, matching the input's dimensions, framerate, and camera.

## Scoring
Your solution will be scored frame-by-frame.
It will also be evaluated against the ground-truth 3D geometry.
Accordingly, solutions that represent motion or appearance in 2D or copy from the video instead of explicitly modeling the scene will be strongly penalized.
The autograder will fail submissions that copy pixels from the input video into the Blender scene for any purpose, such as to texture objects; all must be programmatically defined.

## Output
The output should contain two directories:
1. `{{paths.solution}}` is the program, an executable `build.sh`, and the Python it runs.
2. `{{paths.world}}` is its output, holding the camera, the render, the geometry at every frame, and the 3D state of the moving matter over time.

`build.sh` takes no arguments, does **not** read the video file, and writes all of `{{paths.world}}` from nothing.
Both are required to verify that the reconstruction is reproducible.

Run `python -m checker` to confirm that what you have written is in the correct format.
The packaged checker does not contain the privileged autograder.

## How to build the scene and its motion

One option is to do all of it in Blender: describe the 3D scene and its motion
through time there, and render that 4D sequence. You can also write external
graphics code or programs using the libraries provided to describe the motion of the scene
through time, and use Blender only to render the final 4D sequence. NumPy,
SciPy, PyTorch, Taichi, Warp and Trimesh are installed, and anything else on the
machine is fair game.

However the motion is computed, the evolution of the scene has to be rendered
with Blender, through the camera you export.

The rendered geometry in `meshes/` may change freely from frame to frame, vertex count and
topology included, as happens with liquids or fractures. `dynamics/` is where the
frame-to-frame correspondence lives: column `j` of `pos` is the same piece of
material at every frame, so `pos[t, j]` and `pos[t+1, j]` are that matter one
frame apart. Geometry rebuilt independently at each frame cannot give you that.

## Output format

The full format of `{{paths.world}}` is in `{{paths.task}}/world-format.md`.
It lists every file with its dtype and shape, the camera conventions, and what to return for a liquid.
Read it before you write your export.

### Program: `{{paths.solution}}`

`{{paths.solution}}` should contain an executable `build.sh` and the `.py` and `.sh` files it runs, in any layout.
Running `bash build.sh` should write the complete `{{paths.world}}`, replacing anything already there.
It cannot read the video, access the network, install packages, or load assets shipped alongside it.
Anyone with this environment should be able to rebuild your world by running it.

### World: `{{paths.world}}`

```
world/
  camera.json          {"intrinsics": K, "extrinsic": E}, the fixed camera
  render.mp4           the rendered video
  meshes/<id>.npz      each rendered object's triangles at every frame
  dynamics/<name>.npz  each piece of moving material's positions over time
  solver/              a liquid solver's own bake, if it keeps no particle identity
```

`meshes/` is the geometry that gets rendered, and its vertex count and topology may change from frame to frame.
`dynamics/` describes how the 3D geometry of the moving objects moves through time: `pos` has shape `(F, N, 3)`, and column `j` is the same piece of material at every frame.
Static objects need only a `meshes/` archive.
The points in `dynamics/` are the vertices of your simulation mesh for a rigid body, cloth or soft body, and the particles for MPM, SPH or sand.

All positions are in world-space metres, with Z up.
`render.mp4` must match `{{paths.video}}` in frame count, frame rate and resolution, or it will not be scored.
Every file in `meshes/` and `dynamics/` uses the same `F` frames.
Each frame of `render.mp4` should show exactly the triangles in `meshes/` and nothing else.

### Environment

The environment has FFmpeg, FFprobe, ImageMagick, Python and Blender.
You can run Blender as the `blender` executable or import it in Python as `bpy`.
NumPy, SciPy, PyTorch, Taichi, Warp, Trimesh, Pillow and the checker are available in Python.
The Blender Python API reference is in `/opt/blender-docs`.
`{{paths.task}}/render-settings.md` has the render settings we expect.

### Checking your work

`python -m checker` reports every format error, and a world that fails it scores zero.
`python -m checker.preview` saves `/workspace/tmp/preview_0000.png`, with frame 0 of `render.mp4` next to your `meshes/` projected through `camera.json`.
If the two don't line up, your camera or meshes don't match what you rendered.
Neither tool checks whether your world matches the video.
Put intermediate files in `/workspace/tmp`, not in `world/` or `solution/`.
