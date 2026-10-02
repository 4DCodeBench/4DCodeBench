FROM nvidia/cuda:13.0.2-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display \
    PATH="/opt/venv/bin:/opt/blender:$PATH"

# system packages: ffmpeg, ImageMagick and the headless GL, EGL and Vulkan libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    curl \
    xz-utils \
    ca-certificates \
    libgl1 \
    libglx-mesa0 \
    libegl1 \
    libgles2 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libxkbcommon0 \
    libxkbcommon-x11-0 \
    libxfixes3 \
    libxi6 \
    libxxf86vm1 \
    libxcursor1 \
    libxinerama1 \
    libxrandr2 \
    libvulkan1 \
    mesa-vulkan-drivers \
    imagemagick \
    && rm -rf /var/lib/apt/lists/*

# CPython 3.13 as python3, the version the `bpy` 5.2.0 wheel (Blender as a Python module)
# is built for.
RUN curl -fsSL "https://github.com/astral-sh/python-build-standalone/releases/download/20260901/cpython-3.13.15%2B20260901-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz" \
    | tar -xz -C /opt && ln -sf /opt/python/bin/python3.13 /usr/local/bin/python3

# Blender 5.2.0 as a standalone binary too: it bakes Mantaflow fluid, and its bundled Python
# has the OpenVDB, USD, OpenImageIO, MaterialX and OpenColorIO bindings the wheel lacks.
RUN mkdir -p /opt/blender && \
    curl -fsSL -H "User-Agent: Mozilla/5.0" \
    https://download.blender.org/release/Blender5.2/blender-5.2.0-linux-x64.tar.xz \
    | tar -xJ -C /opt/blender --strip-components=1

# /opt/venv: bpy, CUDA 13 PyTorch and the solution libraries, including Taichi and Warp
# for solutions that write their own simulator.
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir bpy==5.2.0 torch==2.13.0 scipy==1.18.0 trimesh==5.0.0 pillow==11.3.0 \
        taichi==1.7.4 warp-lang==1.17.0

# GPU smoke test, the default command
COPY harness/base/verify_gpu.py /usr/local/bin/verify_gpu.py
RUN chmod +x /usr/local/bin/verify_gpu.py

WORKDIR /workspace

CMD ["/usr/local/bin/verify_gpu.py"]
