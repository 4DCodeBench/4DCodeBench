#!/usr/bin/env bash
# Install the scorer's Python packages into the active conda environment (environment.yml)
# and build `scorer-vdb`, the OpenVDB reader of liquid solver bakes. Run from anywhere.
#
# The CUDA extensions compile for the GPUs on this machine; set TORCH_CUDA_ARCH_LIST
# (e.g. "8.9;9.0") to compile for others.
set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "activate the conda environment from environment.yml first" >&2
  exit 1
fi
cd "$(dirname "$0")/.."
requirements=harness/scorer
install() { uv pip install --python "$CONDA_PREFIX/bin/python" "$@"; }

install --index-url https://download.pytorch.org/whl/cu130 torch==2.13.0 torchvision==0.28.0
install -r "$requirements/requirements.txt"
FORCE_CUDA=1 CUDA_HOME="$CONDA_PREFIX" install --no-build-isolation -r "$requirements/requirements-cuda.txt"
install --no-deps -r "$requirements/requirements-nodeps.txt"

g++ -O2 -std=c++17 scorer/vdb.cpp -I"$CONDA_PREFIX/include" -L"$CONDA_PREFIX/lib" \
    -Wl,-rpath,"$CONDA_PREFIX/lib" -lopenvdb -ltbb -o "$CONDA_PREFIX/bin/scorer-vdb"
echo "environment ready: $CONDA_PREFIX"
