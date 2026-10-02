"""The viewer's HTTP server: serves what the scorer and the visualizer wrote.

    python viewer/server.py --port 8130 [--cases cases --data data --runs runs]

Read-only: nothing here scores, registers or renders. Numbers come from a world's
`results/` and media from its `viz/`; a missing file is served as absent.

    /worlds                                 every (case, world) pair
    /runs                                   every recorded run: run.json fields and rewards
    /metrics/<kind>/<case>/<slug>           the rewards and the video readings
    /series/<kind>/<case>/<slug>            the per-frame and per-path readings
    /media/<kind>/<case>/<slug>/<file>      one rendered file: the bundle or a video
    /source/<kind>/<case>/<slug>/<video>    the case's reference.mp4 or the world's render.mp4
    /mesh/<kind>/<case>/<slug>/<frame>      one frame of `meshes/`: the JSON manifest
    /meshblob/<kind>/<case>/<slug>/<frame>  the same frame's binary buffers

A pair is named `<kind>/<case>/<slug>`, e.g. `real/ball_ramp/gt`. The slug is `gt`
for the case's reference world and `<agent>-<trial>` for a run. Every root mirrors
`cases/`, and every path is derived from the pair name, so a request cannot reach
outside the scanned roots:

    world     data/<kind>/<case>/world            runs/<kind>/<case>/<agent>/<trial>/workspace/world
    results   data/<kind>/<case>/sanity/results   runs/<kind>/<case>/<agent>/<trial>/results
    media     data/<kind>/<case>/sanity/viz       runs/<kind>/<case>/<agent>/<trial>/viz

`results/` is the scorer's output: `reward.json`, `reward.detail.json` and the
metric arrays. `viz/` is the output of `python -m visualizer`. A case's reference
video is `cases/<kind>/<case>/reference.mp4`; for a synthetic case it is the
reference world's render.
"""

from __future__ import annotations

import argparse
import json
import math
import mimetypes
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from config.estimates import (
    RESULTS_DIRNAME,
    SANITY_DIRNAME,
    VIZ_DIRNAME,
)
import numpy as np

from config.world import CAMERA_FILENAME
from visualizer import elements
from scorer.storage import open_dense

DATA = REPO / "data"
RUNS = REPO / "runs"
CASES = REPO / "cases"

GT_SLUG = "gt"
CASE_VIDEO = "reference.mp4"

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
PORT = 8130
KINDS = {".json": "application/json", ".bin": "application/octet-stream", ".mp4": "video/mp4"}


@dataclass(frozen=True)
class Entry:
    """One (case, world) pair: the paths of its world, scorer results and media.

    `case` is kind-qualified (`<kind>/<case>`), so `name` is `<kind>/<case>/<slug>`.
    """

    case: str
    slug: str
    world: Path
    results: Path
    media: Path

    @property
    def name(self) -> str:
        return f"{self.case}/{self.slug}"

    @property
    def reference(self) -> bool:
        return self.slug == GT_SLUG

    def as_dict(self) -> dict:
        return {"world": self.name, "case": self.case, "slug": self.slug,
                "reference": self.reference}


def is_world(path: Path) -> bool:
    return (path / CAMERA_FILENAME).is_file()


def reference_video(case: str) -> Path | None:
    """The case's reference video, or None if the case has none."""

    video = CASES / case / CASE_VIDEO
    return video if video.is_file() else None


SCAN_TTL = 5.0            # seconds a scan of the roots is reused for
_scan: tuple[float, list["Entry"], dict[str, "Entry"]] = (0.0, [], {})
_scan_lock = threading.Lock()


def pairs() -> list[Entry]:
    """Every pair, grouped by case with the reference world first, then by slug.

    The scan is cached for `SCAN_TTL` seconds, since scrubbing issues several
    requests per frame; a newly finished run appears once the cache expires.
    """

    return scan()[0]


def scan() -> tuple[list[Entry], dict[str, Entry]]:
    """Return the cached scan and the pairs indexed by name, rescanning once stale."""

    global _scan
    if time.monotonic() - _scan[0] < SCAN_TTL:
        return _scan[1], _scan[2]
    with _scan_lock:
        if time.monotonic() - _scan[0] < SCAN_TTL:   # another thread just scanned
            return _scan[1], _scan[2]
        found = walk()
        _scan = (time.monotonic(), found, {entry.name: entry for entry in found})
    return _scan[1], _scan[2]


def walk() -> list[Entry]:
    """Scan the roots for every (case, world) pair on disk."""

    found: list[Entry] = []
    for world in sorted(DATA.glob("*/*/world")):       # data/<kind>/<case>/world
        if not is_world(world):
            continue
        case = str(world.parent.relative_to(DATA))
        sanity = world.parent / SANITY_DIRNAME
        found.append(Entry(case, GT_SLUG, world, sanity / RESULTS_DIRNAME, sanity / VIZ_DIRNAME))
    for workspace in sorted(RUNS.glob("*/*/*/*/workspace/world")):
        if not is_world(workspace):
            continue
        run = workspace.parents[1]                     # runs/<kind>/<case>/<agent>/<trial>
        case = "/".join(run.relative_to(RUNS).parts[:2])
        found.append(Entry(case, f"{run.parent.name}-{run.name}", workspace,
                           run / RESULTS_DIRNAME, run / VIZ_DIRNAME))
    return sorted(found, key=lambda entry: (entry.case, not entry.reference, entry.slug))


RUN_KEYS = ("agent", "model", "effort", "trial", "agent_exit", "delivered", "agent_seconds",
            "usage", "cost_usd", "steps", "tool_calls")


def runs() -> list[dict]:
    """Every run the harness recorded, one flat row each.

    A row holds the `run.json` fields in `RUN_KEYS`, the rewards in
    `results/reward.json`, the scorer status and the registration residual at the
    fit's start and end. Each reference world scored against itself is a row with
    agent `gt` and model `reference`. Nothing is averaged here.
    """

    def readings(results: Path) -> dict:
        reward = results / "reward.json"
        detail = results / "reward.detail.json"
        read = json.loads(detail.read_text(encoding="utf-8")) if detail.is_file() else {}
        return {"reward": json.loads(reward.read_text(encoding="utf-8")) if reward.is_file() else None,
                "status": read.get("status"),
                "residual": (read.get("alignment") or {}).get("residual"),
                "residual_start": (read.get("alignment") or {}).get("residual_start")}

    found = []
    # runs/<kind>/<case>/<agent>/<trial>/run.json
    for record_path in sorted(RUNS.glob("*/*/*/*/run.json")):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        run = record_path.parent
        kind, case = run.relative_to(RUNS).parts[:2]
        row = {"kind": kind, "case": case, "world": f"{kind}/{case}/{run.parent.name}-{run.name}"}
        row.update({key: record.get(key) for key in RUN_KEYS})
        row.update(readings(run / RESULTS_DIRNAME))
        found.append(row)
    # data/<kind>/<case>/sanity/results: the reference world scored against itself
    for reward in sorted(DATA.glob(f"*/*/{SANITY_DIRNAME}/{RESULTS_DIRNAME}/reward.json")):
        kind, case = reward.parents[2].relative_to(DATA).parts[:2]
        row = {"kind": kind, "case": case, "world": f"{kind}/{case}/{GT_SLUG}",
               "agent": GT_SLUG, "model": "reference", "effort": None, "trial": 1,
               "agent_exit": 0, "delivered": True}
        row.update({key: None for key in RUN_KEYS if key not in row})
        row.update(readings(reward.parent))
        found.append(row)
    return found


def reference_pair(entry: Entry) -> Entry | None:
    """The case's reference pair, or None if absent or if `entry` is that pair."""

    for other in pairs():
        if other.case == entry.case and other.reference and other.name != entry.name:
            return other
    return None


def find(name: str) -> Entry:
    """Return the pair a request names; raise KeyError for any other name."""

    entry = scan()[1].get(name)
    if entry is None:
        raise KeyError(f"no world named {name!r}")
    return entry


# detail keys read against the case video, served apart from the rewards
READING_PREFIXES = ("semantic_", "depth_")


def worlds() -> list[dict]:
    """Every pair, with the names of its rendered media files."""

    listed = []
    for entry in pairs():
        payload = entry.as_dict()
        payload["media"] = sorted(path.name for path in entry.media.glob("*")) \
            if entry.media.is_dir() else []
        listed.append(payload)
    return listed


def metrics(name: str) -> dict:
    """One pair's rewards, status, timeline and video readings; a missing file is a missing key.

    A real case has no reference world and so no reward for the reference-world
    metrics; a world whose Semantic or Depth reading failed lacks those keys.
    """

    entry = find(name)
    payload: dict = {"world": entry.name, "case": entry.case, "slug": entry.slug}
    reward = entry.results / "reward.json"
    if reward.is_file():
        payload["reward"] = json.loads(reward.read_text(encoding="utf-8"))
    detail = entry.results / "reward.detail.json"
    if detail.is_file():
        read = json.loads(detail.read_text(encoding="utf-8"))
        payload["status"] = read.get("status")
        payload["timeline"] = read.get("timeline") or []
        payload["video"] = {key: value for key, value in read.items()
                            if key.startswith(READING_PREFIXES) or key == "fps"}
    return payload


def source_path(name: str, kind: str) -> Path:
    """The path of the case's reference video or the world's render, served in place."""

    entry = find(name)
    if kind == "reference":
        video = reference_video(entry.case)
    elif kind == "render":
        video = entry.world / "render.mp4"
    else:
        raise KeyError(f"{kind!r} is neither the reference nor the render")
    if video is None or not video.is_file():
        raise FileNotFoundError(f"{name} has no {kind} video")
    return video


def media_path(name: str, filename: str) -> Path:
    """One rendered file of one pair; `filename` must be a bare name, not a path."""

    if "/" in filename or filename.startswith("."):
        raise FileNotFoundError(filename)
    target = find(name).media / filename
    if not target.is_file():
        raise FileNotFoundError(f"{name}/{filename} has not been rendered")
    return target


# ---------------------------------------------------------------- the series
#
# `/series` discovers the one-dimensional readings in `reward.detail.json` and the
# result archives instead of listing them. Each reading has one of two shapes:
#
#   frame      one value per sampled frame, plotted against time
#   path       one cost per path, plotted sorted

SERIES_POINTS = 256       # sorted values a path reading is thinned to
SERIES_MAX = 4096         # longest array served as a series

# Display names; an unlisted key is shown with its underscores replaced by spaces.
SERIES_LABELS = {
    "depth_per_frame": "depth error",
    "depth_valid_fraction": "depth coverage",
    "semantic_dinov3": "semantic (DINOv3)",
    "semantic_tips": "semantic (TIPS)",
    "distribution": "flow distribution gap",
    "cost_dtw": "DTW cost per path",
    "buried": "buried fraction per component",
}


def pretty(key: str) -> str:
    return SERIES_LABELS.get(key) or key.replace("_", " ")


def numbers(values) -> list | None:
    """The values as rounded floats with NaN as None; None if not a non-empty numeric list."""

    if not isinstance(values, list) or not values or len(values) > SERIES_MAX:
        return None
    out = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        out.append(None if value != value else round(float(value), 6))
    return out


def one_dimensional(bundle, key: str, dense: bool = False) -> tuple | None:
    """Read a member's shape and dtype without loading its array; None if unreadable."""

    if dense:
        member = bundle[key]
        return member.shape, member.dtype

    try:
        with bundle.zip.open(key + ".npy") as member:
            version = np.lib.format.read_magic(member)
            if version == (1, 0):
                shape, _, dtype = np.lib.format.read_array_header_1_0(member)
            elif version == (2, 0):
                shape, _, dtype = np.lib.format.read_array_header_2_0(member)
            else:
                return None
    except Exception:
        return None
    return shape, dtype


def timeline(detail: dict) -> set[int]:
    """The array lengths that count as a time series for this pair.

    A length qualifies if it equals a metric's sampled frame count, or is one
    shorter (a per-step reading). Other one-dimensional arrays, such as the
    per-component buried fractions, are served as per-path readings.
    """

    lengths = set()
    dimensions = detail.get("dimensions")
    if isinstance(dimensions, list) and dimensions and isinstance(dimensions[0], int):
        lengths.add(dimensions[0])
    for key, value in detail.items():
        if key.endswith("_frames") and isinstance(value, int) and value > 0:
            lengths.add(value)
    return lengths | {n - 1 for n in lengths if n > 1}


def arrays(results: Path, strides: dict, through_time: set[int]) -> list[dict]:
    """Every one-dimensional float array in this pair's result archives, as series entries."""

    found: list[dict] = []
    for path in sorted([*results.glob("*.npz"), *results.glob("*.h5")]):
        stem = path.stem
        if stem == "uni3d":
            continue
        dense = path.suffix == ".h5"
        if dense:
            bundle = open_dense(path)
        else:
            try:
                bundle = np.load(path)
            except Exception:
                continue                  # skip an unreadable archive
        with bundle:
            wanted = {}
            for key in bundle if dense else bundle.files:
                header = one_dimensional(bundle, key, dense)
                if header is None:
                    continue
                shape, dtype = header
                if len(shape) != 1 or dtype.kind != "f" or not 0 < shape[0] <= SERIES_MAX:
                    continue
                wanted[key] = shape[0]
            stride = strides.get(stem, 1)
            for key in sorted(wanted):
                # an unlisted key is labelled with its archive stem to keep labels distinct
                label = SERIES_LABELS.get(key) or pretty(f"{stem} {key}")
                values = read(bundle[key])
                if not key.startswith("cost_") and wanted[key] in through_time:
                    found.append({"key": f"{stem}.{key}", "label": label, "shape": "frame",
                                  "values": values, "stride": stride})
                else:
                    found.append({"key": f"{stem}.{key}", "label": label, "shape": "path",
                                  "values": thin(sorted(v for v in values if v is not None)),
                                  "paths": len(values)})
    return found


def read(array) -> list:
    return [None if v != v else round(float(v), 6) for v in np.asarray(array).ravel()]


def thin(values: list) -> list:
    """Evenly sample a sorted list down to `SERIES_POINTS` values."""

    if len(values) <= SERIES_POINTS:
        return values
    step = (len(values) - 1) / (SERIES_POINTS - 1)
    return [values[min(len(values) - 1, round(i * step))] for i in range(SERIES_POINTS)]


def series(name: str) -> dict:
    """Every per-frame and per-path reading of one pair.

    Read from `reward.detail.json` and the result archives; an absent reading is
    omitted.
    """

    entry = find(name)
    payload: dict = {"world": entry.name, "series": [], "fps": None}
    if not entry.results.is_dir():
        return payload
    detail = entry.results / "reward.detail.json"
    read_detail = json.loads(detail.read_text(encoding="utf-8")) if detail.is_file() else {}
    payload["fps"] = read_detail.get("fps")
    # `<metric>_stride` maps an index in that metric's arrays back to a frame number
    strides = {key[: -len("_stride")]: value for key, value in read_detail.items()
               if key.endswith("_stride") and isinstance(value, int) and value > 0}
    payload["strides"] = strides
    found: list[dict] = []
    for key, value in sorted(read_detail.items()):
        if not key.endswith("_per_frame"):
            continue
        stem = key.rsplit("_per_", 1)[0]
        stride = strides.get(stem, 1)
        plain = numbers(value)
        if plain is not None:
            found.append({"key": key, "label": pretty(key), "shape": "frame",
                          "values": plain, "stride": stride})
        elif isinstance(value, dict):
            # `semantic_per_frame` is one curve per model, under the model's name
            for inner, rows in sorted(value.items()):
                plain = numbers(rows)
                if plain is not None:
                    found.append({"key": f"{key}.{inner}", "label": pretty(inner), "shape": "frame",
                                  "values": plain, "stride": stride})
    found.extend(arrays(entry.results, strides, timeline(read_detail)))
    payload["series"] = found
    return payload


BUNDLE_MANIFEST = "bundle.json"


def bundle_manifest(name: str) -> bytes:
    """The rendered bundle manifest with the pair's names added.

    `python -m visualizer` writes no names into the manifest; this adds the pair's
    name, case and slug, and the name of the case's reference pair.
    """

    entry = find(name)
    payload = json.loads(media_path(name, BUNDLE_MANIFEST).read_text(encoding="utf-8"))
    other = reference_pair(entry) if payload.get("pair") else None
    payload.update({"world": entry.name, "case": entry.case, "slug": entry.slug,
                    "gt_world": other.name if other is not None else None})
    return as_json(payload)


# LRU cache of served mesh frames, keyed by (pair, frame). Each frame is requested
# twice (manifest and blob) and again on replay. The budget is in bytes because
# frame sizes vary widely between worlds.
FRAMES_BUDGET = 384 * 1024 * 1024
_frames: OrderedDict[tuple[str, int], tuple[dict, bytes]] = OrderedDict()
_frames_bytes = 0
_frames_lock = threading.Lock()


# Series are built once per pair and cached for the server's lifetime.
_series: dict[str, dict] = {}
_series_lock = threading.Lock()


def held_series(name: str) -> dict:
    with _series_lock:
        if name in _series:
            return _series[name]
    built = series(name)
    with _series_lock:
        _series[name] = built
    return built


def frame_geometry(entry: Entry, frame: int) -> tuple[dict, bytes]:
    key = (entry.name, frame)
    with _frames_lock:
        held = _frames.get(key)
        if held is not None:
            _frames.move_to_end(key)
            return held
    built = elements.geometry(entry.name, entry.world,
                              entry.results if entry.results.is_dir() else None, frame)
    global _frames_bytes
    with _frames_lock:
        if key not in _frames:
            _frames[key] = built
            _frames_bytes += len(built[1])
            while _frames_bytes > FRAMES_BUDGET and len(_frames) > 1:
                _frames_bytes -= len(_frames.popitem(last=False)[1][1])
    return built


def plain(value):
    """Replace non-finite floats with None, recursively.

    `json.dumps` writes NaN as a bare `NaN`, which `JSON.parse` rejects.
    """

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [plain(item) for item in value]
    return value


def as_json(payload) -> bytes:
    return json.dumps(plain(payload)).encode()


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC), **kwargs)

    def log_message(self, *args):
        pass

    def send_bytes(self, payload, kind):
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def send_error_json(self, error, code=500):
        payload = json.dumps({"error": f"{type(error).__name__}: {error}"}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        path = unquote(self.path.split("?")[0])
        try:
            if path == "/worlds":
                return self.send_bytes(as_json(worlds()), "application/json")
            if path == "/runs":
                return self.send_bytes(as_json(runs()), "application/json")
            if path.startswith("/series/"):
                return self.send_bytes(as_json(held_series(path[len("/series/"):])),
                                       "application/json")
            if path.startswith("/metrics/"):
                return self.send_bytes(as_json(metrics(path[len("/metrics/"):])),
                                       "application/json")
            if path.startswith("/source/"):
                name, _, kind = path[len("/source/"):].rpartition("/")
                return self.send_file(source_path(name, kind.removesuffix(".mp4")))
            if path.startswith("/media/"):
                name, _, filename = path[len("/media/"):].rpartition("/")
                if filename == BUNDLE_MANIFEST:
                    return self.send_bytes(bundle_manifest(name), "application/json")
                return self.send_file(media_path(name, filename))
            # /mesh serves one frame's manifest, /meshblob its buffers
            for prefix, want_json in (("/mesh/", True), ("/meshblob/", False)):
                if not path.startswith(prefix):
                    continue
                name, _, frame = path[len(prefix):].rpartition("/")
                manifest, binary = frame_geometry(find(name), int(frame))
                if want_json:
                    return self.send_bytes(as_json(manifest), "application/json")
                return self.send_bytes(binary, "application/octet-stream")
            if path.startswith("/vendor/"):
                target = (STATIC / path.lstrip("/")).resolve()
                if not target.is_relative_to(STATIC / "vendor"):
                    raise FileNotFoundError(path)
                kind = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
                return self.send_bytes(target.read_bytes(), kind)
        except (BrokenPipeError, ConnectionResetError):
            # the client closed the connection (a video seek, a source swap or a
            # page close); there is no one to respond to
            return
        except (FileNotFoundError, KeyError) as error:
            return self.send_error_json(error, code=404)
        except Exception as error:            # any other failure is a 500
            return self.send_error_json(error)
        return super().do_GET()

    def send_file(self, target: Path):
        """Serve a file, honouring a single `bytes=` Range header so videos can seek.

        A range request reads only the requested bytes from disk.
        """

        kind = KINDS.get(target.suffix, "application/octet-stream")
        size = target.stat().st_size
        span = self.headers.get("Range")
        if not span or not span.startswith("bytes="):
            return self.send_bytes(target.read_bytes(), kind)
        first, _, last = span[len("bytes="):].partition("-")
        start = int(first or 0)
        stop = min(int(last) if last else size - 1, size - 1)
        if start > stop:                      # unsatisfiable range
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        with target.open("rb") as handle:
            handle.seek(start)
            piece = handle.read(stop - start + 1)
        self.send_response(206)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Range", f"bytes {start}-{start + len(piece) - 1}/{size}")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(len(piece)))
        self.end_headers()
        self.wfile.write(piece)


class Server(ThreadingHTTPServer):
    """The threaded HTTP server, with a listen queue of 128.

    One page load opens many connections at once (media files and ranged video
    requests), more than the default queue of five holds.
    """

    daemon_threads = True
    request_queue_size = 128

    def handle_error(self, request, client_address):
        """Ignore errors from clients that closed the connection; report any other.

        A dropped connection can surface in `http.server`'s own flush after `do_GET`
        returns, so it is filtered here as well as in `do_GET`.
        """

        if isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


def main(argv=None) -> int:
    global CASES, DATA, RUNS
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--cases", type=Path, default=CASES, help="the cases root")
    parser.add_argument("--data", type=Path, default=DATA, help="the case data root")
    parser.add_argument("--runs", type=Path, default=RUNS, help="the runs root")
    arguments = parser.parse_args(argv)
    CASES, DATA, RUNS = arguments.cases.resolve(), arguments.data.resolve(), arguments.runs.resolve()
    server = Server(("0.0.0.0", arguments.port), Handler)
    print(f"visualizer on http://localhost:{arguments.port}", flush=True)
    print(f"{len(worlds())} worlds", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
