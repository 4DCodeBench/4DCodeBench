"""Extract the standard tracer samples from a world's Mantaflow UNI or VDB cache."""

from __future__ import annotations

import argparse
import fcntl
import gzip
import json
import struct
import subprocess
import tempfile
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from config.world import INDEX_DTYPE, POS_DTYPE, TRACER_FILENAME

from .validation import SOLVER_CONFIG_DIRNAME as CONFIG_DIRNAME, SOLVER_CONFIG_NAME as CONFIG_NAME
from .world import archive_paths, video_stream
from .paths import solver_cache
from .analytic.advection import MacSampler, log, sample_tracers


def configuration(path: Path):
    """Blender C01 cache: resolution, world voxel size, and accumulated Manta time."""

    raw = gzip.decompress(path.read_bytes())
    if len(raw) != 204 or raw[200:] != b"C01\0":
        raise ValueError(f"{path}: unsupported Mantaflow configuration")
    values = np.frombuffer(raw, "<f4")
    matrix = values[21:37].reshape(4, 4).T
    size = (values[9:12] - values[6:9]) * np.linalg.norm(matrix[:3, :3], axis=0)
    return struct.unpack_from("<3i", raw, 4), float(size.max() * values[4]), float(values[49])


def read_uni(path: Path, dim: tuple[int, int, int], components: int):
    """One gzipped Mantaflow MNT3 grid as an `(nx, ny, nz[, components])` float32 array."""

    raw = gzip.decompress(path.read_bytes())
    if raw[:4] != b"MNT3":
        raise ValueError(f"{path}: expected MNT3 grid")
    nx, ny, nz, _, _, width = struct.unpack_from("<6i", raw, 4)
    if (nx, ny, nz) != dim or width != 4 * components:
        raise ValueError(f"{path}: unexpected grid dimensions or element size")
    grid = np.frombuffer(raw, "<f4", offset=292).reshape(nz, ny, nx, components)
    grid = grid.transpose(2, 1, 0, 3)
    return grid[..., 0] if components == 1 else grid


class CachedFields:
    """Per-frame velocity and liquid level-set grids, in grid units.

    The constructor decodes every frame once into `.npy` files under `scratch`:
    velocity from the UNI or VDB cache, and `phi` as the signed distance of the
    liquid meshes (3.0 where a frame has none). `velocity` and `phi` memory-map them.
    """

    def __init__(self, world: Path, manifest, cache: Path, frames, dim, voxel, scratch: Path):
        self.frames, self.dim, self.scratch = frames, dim, scratch
        vdb = (cache / "data" / f"fluid_data_{frames[0]:04d}.vdb").is_file()
        paths = archive_paths(world)
        with ExitStack() as stack:
            meshes = [stack.enter_context(np.load(paths[oid]))
                      for oid in manifest["liquid_ids"]]
            if any(int(key[-4:]) >= len(frames) for mesh in meshes for key in mesh.files):
                raise ValueError(f"{world}: liquid mesh extends past the cached timeline")
            for slot, frame in enumerate(frames):
                if vdb:
                    path = cache / "data" / f"fluid_data_{frame:04d}.vdb"
                    array = self.grid("velocity", path, 3)
                else:
                    array = read_uni(cache / "data" / f"vel_{frame:04d}.uni", dim, 3)
                if not np.isfinite(array).all():
                    raise ValueError(f"frame {frame}: non-finite velocity")
                np.save(scratch / f"velocity_{slot}.npy", array)
                vertices, faces, offset = [], [], 0
                for mesh in meshes:
                    key = f"vertices_{slot:04d}"
                    if key not in mesh:
                        continue
                    v = (mesh[key] - manifest["domain_min"]) / (voxel * manifest["metres_per_unit"])
                    f = mesh[f"faces_{slot:04d}"]
                    vertices.append(v)
                    faces.append(f + offset)
                    offset += len(v)
                v = np.concatenate(vertices).astype(np.float32) if vertices else np.empty((0, 3), np.float32)
                f = np.concatenate(faces).astype(np.int32) if faces else np.empty((0, 3), np.int32)
                phi = np.full(dim, 3.0, np.float32)
                if len(f):
                    # Grid index i samples the cell centre i + 0.5, as Mantaflow phi does.
                    mesh_path = scratch / "surface.bin"
                    with mesh_path.open("wb") as stream:
                        np.asarray([len(v), len(f)], "<i4").tofile(stream)
                        (v - 0.5).astype("<f4").tofile(stream)
                        f.astype("<i4").tofile(stream)
                    phi = self.grid("surface", mesh_path, 1)
                np.save(scratch / f"phi_{slot}.npy", phi)
                if slot % 50 == 0:
                    log(f"decoded frame {slot + 1}/{len(frames)}")

    def grid(self, operation, source, components):
        """Run `scorer-vdb` (`vdb.cpp`) and return its dense output grid."""

        output = self.scratch / "grid.bin"
        command = ["scorer-vdb", operation, str(source), str(output), *map(str, self.dim)]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise ValueError(f"{source}: {result.stderr.strip()}")
        shape = self.dim + ((components,) if components > 1 else ())
        array = np.fromfile(output, "<f4").reshape(shape)
        output.unlink()
        return array

    def velocity(self, slot):
        return np.load(self.scratch / f"velocity_{slot}.npy", mmap_mode="r")

    def phi(self, slot):
        return np.load(self.scratch / f"phi_{slot}.npy", mmap_mode="r")



def cache_geometry(world: Path):
    """Read the grid and timeline of a world's liquid bake.

    Returns `(manifest, cache, frames, dim, voxel, dt, count)`: the cache frame
    numbers, the fixed grid shape and voxel size in solver units, the solver time of each step
    and the video frame count. Raises `ValueError` on a moving grid or a clock
    that does not increase.
    """

    manifest = json.loads((world / "solver" / "manifest.json").read_text())
    cache = solver_cache(world, manifest["cache"])
    first, last = manifest["frames"]
    # Exported video frame 0 can correspond to Blender cache frame 1.
    if first == 0 and not (cache / CONFIG_DIRNAME / CONFIG_NAME.format(frame=0)).is_file():
        first, last = 1, last + 1
    frames = list(range(first, last + 1))
    count, _, _, _ = video_stream(world / "render.mp4")
    if count < len(frames):
        raise ValueError(f"{world}: {len(frames)} cache frames for {count} video frames")
    configs = [configuration(cache / CONFIG_DIRNAME / CONFIG_NAME.format(frame=frame)) for frame in frames]
    dim, voxel, _ = configs[0]
    if any(shape != dim or not np.isclose(h, voxel) for shape, h, _ in configs):
        raise ValueError(f"{world}: moving or resized solver grid")
    times = np.array([time for _, _, time in configs])
    dt = np.diff(times)
    if not np.all(dt > 0):
        raise ValueError(f"{world}: cache times must increase")
    # the recorded solver clock already includes an animated time_scale
    return manifest, cache, frames, dim, voxel, dt, count


def extract(world: Path, output: Path, scratch: Path | None = None):
    """Advect the tracers and write them to `output` as a `dynamics/`-format archive.

    `pos` is `(F, N, 3)` world metres, NaN outside the cached frames and before a
    column's birth. Returns the frame and column counts.
    """

    manifest, cache, frames, dim, voxel, dt, count = cache_geometry(world)
    with tempfile.TemporaryDirectory(prefix="liquid-fields-", dir=scratch) as directory:
        fields = CachedFields(world, manifest, cache, frames, dim, voxel, Path(directory))
        sampler = MacSampler(np.zeros(3), 1.0, dim)
        pos = sample_tracers(fields, sampler, dt)
    tracks = np.full((count, pos.shape[1], 3), np.nan, POS_DTYPE)
    tracks[:len(pos)] = pos * voxel * manifest["metres_per_unit"] + manifest["domain_min"]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, pos=tracks, ids=np.asarray(manifest["liquid_ids"], INDEX_DTYPE))
    temporary.replace(output)
    return {"frames": count, "columns": pos.shape[1]}


def prepare(world: Path) -> Path:
    """The world's tracer file, extracted once under a lock."""

    output = world / "solver" / TRACER_FILENAME
    if output.is_file():
        return output
    with output.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not output.is_file():
            log(f"extracting liquid tracks: {world}")
            extract(world, output, scratch=output.parent)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("world", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--scratch", type=Path)
    args = parser.parse_args()
    output = args.out if args.out is not None else args.world / "solver" / TRACER_FILENAME
    print(json.dumps(extract(args.world, output, args.scratch)), flush=True)


if __name__ == "__main__":
    main()
