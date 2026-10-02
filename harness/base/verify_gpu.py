#!/usr/bin/env python3
"""GPU smoke test of the sim-base image; the image's default command.

Checks, in order: nvidia-smi runs; nvcc is CUDA 13.0; Blender Cycles renders a
frame on a CUDA or OptiX device; PyTorch 2.13.0 with CUDA 13.0 runs a matmul on
the GPU. Exits 0 when all pass, non-zero with a message at the first failure.
"""

import os
import shutil
import subprocess
import sys
import tempfile


EXPECTED_CUDA_VERSION = "13.0"
EXPECTED_TORCH_VERSION = "2.13.0"


def log_section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def verify_nvidia_smi() -> None:
    log_section("1. Verifying nvidia-smi and GPU visibility")
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            check=True,
        )
        print(result.stdout)
    except FileNotFoundError:
        print("[FAIL] 'nvidia-smi' executable not found on PATH.", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] 'nvidia-smi' exited with error code {e.returncode}:", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        sys.exit(1)
    print("[PASS] nvidia-smi executed successfully.")


def verify_cuda_toolkit() -> None:
    log_section("2. Verifying CUDA 13.0 Toolkit")
    result = subprocess.run(
        ["nvcc", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    print(result.stdout)
    if f"release {EXPECTED_CUDA_VERSION}" not in result.stdout:
        raise RuntimeError(
            f"nvcc is not from CUDA {EXPECTED_CUDA_VERSION}: {result.stdout.strip()}"
        )
    print(f"[PASS] CUDA Toolkit {EXPECTED_CUDA_VERSION} is available.")


def verify_blender_cycles_gpu() -> None:
    log_section("3. Verifying Blender 5.2 Cycles GPU Rendering (CUDA/OPTIX)")
    blender_bin = shutil.which("blender")
    if not blender_bin:
        print("[FAIL] 'blender' executable not found on PATH.", file=sys.stderr)
        sys.exit(1)

    blender_test_script = """
import bpy
import sys
import os

print(f"Blender Version: {bpy.app.version_string}")

# Access Cycles preferences
cpref = bpy.context.preferences.addons['cycles'].preferences
cpref.get_devices()

print("Probing available compute device types...")
chosen_backend = None
for dt in ('OPTIX', 'CUDA'):
    try:
        cpref.compute_device_type = dt
        cpref.get_devices()
        gpu_devs = [d for d in cpref.devices if d.type == dt]
        print(f"  Backend '{dt}': found {len(gpu_devs)} GPU device(s) -> {[d.name for d in gpu_devs]}")
        if gpu_devs and chosen_backend is None:
            chosen_backend = dt
    except Exception as e:
        print(f"  Backend '{dt}' probe failed: {e}")

if not chosen_backend:
    print("[FAIL] No CUDA or OPTIX GPU devices detected by Blender Cycles!", file=sys.stderr)
    sys.exit(1)

print(f"Configuring Blender Cycles to use {chosen_backend} backend...")
cpref.compute_device_type = chosen_backend
cpref.get_devices()

active_gpus = []
active_cpus = []
for dev in cpref.devices:
    if dev.type == chosen_backend:
        dev.use = True
        active_gpus.append(dev.name)
    else:
        dev.use = False
        if dev.type == 'CPU':
            active_cpus.append(dev.name)

assert len(active_gpus) > 0, f"No GPU devices enabled for backend {chosen_backend}"
print(f"Enabled GPU device(s): {active_gpus}")
print(f"Disabled CPU device(s): {active_cpus}")

# Configure scene for GPU render
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.device = 'GPU'
scene.render.resolution_x = 64
scene.render.resolution_y = 64
scene.cycles.samples = 4

temp_output = "/tmp/blender_smoke_test_render.png"
scene.render.filepath = temp_output

print("Rendering test frame using Cycles GPU...")
bpy.ops.render.render(write_still=True)

if not os.path.exists(temp_output) or os.path.getsize(temp_output) == 0:
    print(f"[FAIL] Rendered output file missing or empty: {temp_output}", file=sys.stderr)
    sys.exit(1)

# Verify device configuration
assert scene.render.engine == 'CYCLES', f"Engine is {scene.render.engine}, expected CYCLES"
assert scene.cycles.device == 'GPU', f"Cycles device is {scene.cycles.device}, expected GPU"
assert cpref.compute_device_type in ('OPTIX', 'CUDA'), f"Compute device type is {cpref.compute_device_type}"

# Cleanup
try:
    os.remove(temp_output)
except OSError:
    pass

print(f"[PASS] Blender Cycles GPU render succeeded on device(s): {active_gpus} (backend: {chosen_backend}).")
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(blender_test_script)
        script_path = f.name

    try:
        result = subprocess.run(
            [blender_bin, "-b", "--python", script_path],
            capture_output=True,
            text=True,
            check=True,
        )
        print(result.stdout)
        if result.stderr:
            print("Blender stderr (diagnostics):", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] Blender render test failed with exit code {e.returncode}:", file=sys.stderr)
        print(e.stdout, file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        sys.exit(1)
    finally:
        if os.path.exists(script_path):
            os.remove(script_path)


def verify_torch_gpu() -> None:
    log_section("4. Verifying PyTorch CUDA 13.0")
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"PyTorch CUDA runtime: {torch.version.cuda}")
        if torch.__version__.split("+", 1)[0] != EXPECTED_TORCH_VERSION:
            raise RuntimeError(
                f"PyTorch {torch.__version__} != expected {EXPECTED_TORCH_VERSION}"
            )
        if torch.version.cuda != EXPECTED_CUDA_VERSION:
            raise RuntimeError(
                f"PyTorch CUDA {torch.version.cuda} != expected {EXPECTED_CUDA_VERSION}"
            )
        print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
        if not torch.cuda.is_available():
            print("[FAIL] PyTorch cannot access CUDA GPU.", file=sys.stderr)
            sys.exit(1)
        print(f"PyTorch CUDA Device 0: {torch.cuda.get_device_name(0)}")

        x = torch.rand(1024, 1024, device="cuda")
        y = (x @ x).sum()
        assert y.is_cuda, "Matmul result tensor is not on CUDA device!"
        print(f"CUDA matmul checksum: {float(y):.3f}")
        print("[PASS] PyTorch CUDA tensors succeeded.")

    except Exception as e:
        import traceback
        print(f"[FAIL] PyTorch GPU verification encountered an exception: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


def main() -> None:
    print("Starting 4DCodeBench Base Environment GPU Verification Smoke Test...")
    verify_nvidia_smi()
    verify_cuda_toolkit()
    verify_blender_cycles_gpu()
    verify_torch_gpu()
    log_section("ALL GPU VERIFICATIONS PASSED SUCCESSFULLY")
    sys.exit(0)


if __name__ == "__main__":
    main()
