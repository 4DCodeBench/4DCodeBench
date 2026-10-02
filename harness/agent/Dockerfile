FROM 4dcb/sim-base:dev

ENV DEBIAN_FRONTEND=noninteractive \
    NODE_MAJOR=22 \
    PYTHONPATH=/opt/4dcb

# The three agent CLIs: Claude Code, Codex and Antigravity. The image holds no
# credentials; each run mounts only its own CLI's credential file.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    procps \
    ripgrep \
    && curl -fsSL https://deb.nodesource.com/setup_${NODE_MAJOR}.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && npm install -g @anthropic-ai/claude-code @openai/codex \
    && curl -fsSL https://antigravity.google/cli/install.sh | bash -s -- --dir /usr/local/bin \
    && mkdir -p /root/.local/bin \
    && for cli in claude codex agy; do ln -sf "$(command -v $cli)" /usr/local/bin/$cli; ln -sf "$(command -v $cli)" /root/.local/bin/$cli; done \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir numpy==2.3.1

# Blender's Python API reference as offline HTML in /opt/blender-docs, without
# Sphinx's .doctrees build cache.
RUN curl -fsSL https://docs.blender.org/api/5.2/blender_python_reference_5_2.zip \
      -o /tmp/bpydocs.zip && \
    python3 -c "import zipfile; z = zipfile.ZipFile('/tmp/bpydocs.zip'); \
      z.extractall('/opt', [n for n in z.namelist() if '/.doctrees/' not in n])" && \
    mv /opt/blender_python_reference_5_2 /opt/blender-docs && \
    rm /tmp/bpydocs.zip
RUN mkdir -p /opt/4dcb/config
COPY config/__init__.py config/world.py /opt/4dcb/config/
COPY checker /opt/4dcb/checker

# Artificial Analysis' Stirrup agent loop, for the `stirrup` agent
RUN pip install --no-cache-dir stirrup==0.2.0

RUN mkdir -p \
    /root/.claude/projects \
    /root/.codex/sessions \
    /root/.gemini/antigravity-cli/brain \
    /root/.stirrup

WORKDIR /workspace
