# Stage 1 builds OpenVDB and the `scorer-vdb` helper. The runtime stage is the PyTorch
# CUDA devel image, whose toolkit the CUDA extensions below compile against.
FROM ubuntu:24.04 AS vdb
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates git cmake make g++ libtbb-dev libblosc-dev libboost-iostreams-dev zlib1g-dev
RUN git clone --depth 1 --branch v13.0.0 https://github.com/AcademySoftwareFoundation/openvdb.git /src && \
    cmake -S /src -B /build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/openvdb \
        -DOPENVDB_BUILD_BINARIES=OFF -DOPENVDB_CORE_STATIC=OFF -DUSE_EXPLICIT_INSTANTIATION=OFF && \
    cmake --build /build -j4 && cmake --install /build
COPY scorer/vdb.cpp /vdb.cpp
RUN g++ -O2 -std=c++17 /vdb.cpp -I/opt/openvdb/include -L/opt/openvdb/lib \
    -Wl,-rpath,/opt/openvdb/lib -lopenvdb -ltbb -o /opt/openvdb/scorer-vdb

FROM pytorch/pytorch:2.13.0-cuda13.0-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/usr/local/bin:$PATH" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;10.0;12.0+PTX"

# runtime libraries for the metrics and video decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglx-mesa0 \
    libopengl0 \
    libgomp1 \
    libspatialindex-dev \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python packages, shared with the conda environment (environment.yml).
COPY harness/scorer/requirements*.txt /tmp/requirements/
RUN pip install --no-cache-dir -r /tmp/requirements/requirements.txt && \
    FORCE_CUDA=1 pip install --no-cache-dir --no-build-isolation -r /tmp/requirements/requirements-cuda.txt && \
    pip install --no-cache-dir --no-deps -r /tmp/requirements/requirements-nodeps.txt

RUN apt-get update && apt-get install -y --no-install-recommends \
    libtbb12 libblosc1 libboost-iostreams1.83.0 && rm -rf /var/lib/apt/lists/*
COPY --from=vdb /opt/openvdb /opt/openvdb
RUN ln -s /opt/openvdb/scorer-vdb /usr/local/bin/scorer-vdb

# The stages of `harness/runtime/score.py`: `scorer.prepare`, `scorer` and `visualizer`.
COPY config /opt/4dcb/config
COPY scorer /opt/4dcb/scorer
COPY visualizer /opt/4dcb/visualizer
ENV PYTHONPATH=/opt/4dcb
# Model-loader and library caches go to /tmp; the home directory is read-only under
# `--containall --no-home`.
ENV HF_HOME=/tmp/hf-home \
    XDG_CACHE_HOME=/tmp/xdg-cache \
    TORCH_HOME=/tmp/torch-home \
    MPLCONFIGDIR=/tmp/mpl \
    TRITON_CACHE_DIR=/tmp/triton \
    FLEX_GEMM_AUTOTUNE_CACHE_PATH=/tmp/flex_gemm/autotune_cache.json

WORKDIR /workspace

ENTRYPOINT ["python", "-m", "scorer"]
