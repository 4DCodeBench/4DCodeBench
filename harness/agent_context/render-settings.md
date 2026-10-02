# Render settings this runtime expects

The task asks for `render.mp4` at the input video's frame count, resolution and
frame rate, rendered through the camera you export. How you get there is your
business; these are the settings this machine is set up for, and the ones the
reference worlds were rendered with.

## Final render

Cycles, at the input video's own width and height, `resolution_percentage = 100`,
128 samples, denoising on. Rendering smaller and scaling up changes what the
image-based metrics see, so render at the size you deliver.

## GPU

The runtime provides an NVIDIA GPU. Cycles falls back to the CPU silently when
OptiX is not configured, which turns a twenty-minute render into an overnight
one, so configure it explicitly and fail loudly if it is missing:

```python
scene.render.engine = "CYCLES"
preferences = bpy.context.preferences.addons["cycles"].preferences
preferences.compute_device_type = "OPTIX"
preferences.get_devices()
optix_devices = [device for device in preferences.devices if device.type == "OPTIX"]
if not optix_devices:
    raise RuntimeError("No OptiX device is available")
for device in preferences.devices:
    device.use = device.type == "OPTIX"
scene.cycles.device = "GPU"
scene.render.resolution_x = input_width
scene.render.resolution_y = input_height
scene.render.resolution_percentage = 100
scene.cycles.samples = 128
scene.cycles.use_denoising = True
```

## Writing the video

Two things about `render.mp4` that fail loudly only after a long render, and are
worth setting before you start one.

`file_format = "FFMPEG"` is rejected unless the media type is set first. Blender
5 splits `ImageFormatSettings` by media, and the default is `IMAGE`, so the video
formats are not in the enum until you ask for them:

```python
scene.render.image_settings.media_type = "VIDEO"     # or FFMPEG is not selectable
scene.render.image_settings.file_format = "FFMPEG"
scene.render.ffmpeg.format = "MPEG4"
scene.render.ffmpeg.codec = "H264"
scene.render.fps = round(input_fps)                  # and fps_base for a fractional rate
```

Blender names the file after the frame range, so a `filepath` of `.../render`
becomes `render0001-0251.mp4` and the delivery is missing the `render.mp4` the
format asks for. Spell the name out and turn the suffix off:

```python
scene.render.use_file_extension = False
scene.render.filepath = str(world / "render.mp4")
```

Check what you actually wrote before you deliver: `ffprobe -count_frames` on your
`render.mp4` and on the input should agree on frame count, width, height and
frame rate. The gate compares exactly those four and refuses the world outright
when they differ, before any metric is read.

## While you are iterating

EEVEE at half size is much faster and good enough to check framing, motion and
timing against the video:

```python
scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_percentage = 50
```

Switch back to the settings above for what you deliver. A long Cycles render is
worth waiting for; let the command run to completion rather than polling it.
