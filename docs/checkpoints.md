# Model Checkpoints

Scoring loads all model checkpoints from one local directory, mounted read-only at `/checkpoints` in the scorer container. Inference needs none of them, so if you only run inference you can skip this page.

> **Note**: `facebook/dinov3-vitl16-pretrain-lvd1689m` is gated; ensure your Hugging Face account has access and is authenticated (`HF_TOKEN` or `hf auth login`) before downloading.

## 1. Download Checkpoints

Download and verify all required weights (~6.7 GB total):

```bash
scripts/download_checkpoints.sh             # default: ./checkpoints
scripts/download_checkpoints.sh /data/ckpt  # or custom directory
```

The script pins each model to the benchmark revision, validates SHA-256 checksums, and resumes interrupted downloads.

## 2. Configuration

In `runtime.toml` (template: `harness/runtime/templates/runtime.example.toml`):

```toml
[scorer]
checkpoints = "checkpoints"      # path relative to repo root, or absolute
```

When running the scoring stages directly, outside the harness:

```bash
python -m scorer.prepare --checkpoints <DIR> ...
python -m scorer         --checkpoints <DIR> ...
python -m visualizer     --checkpoints <DIR> ...
```

## Checkpoint Registry

Directory names must remain unchanged, as the scorer locates models by these paths (`config/models.py`, `config/uni3d.py`).

| Directory | Weight File | Source | Revision | SHA-256 |
|---|---|---|---|---|
| `dinov3-vitl16-pretrain-lvd1689m` | `model.safetensors` | [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) | `ea8dc28` | `dcb2e451…` |
| `tipsv2-l14` | `model.safetensors` | [google/tipsv2-l14](https://huggingface.co/google/tipsv2-l14) | `52847a7` | `c75707ea…` |
| `video-depth-anything-large` | `video_depth_anything_vitl.pth` | [depth-anything/Video-Depth-Anything-Large](https://huggingface.co/depth-anything/Video-Depth-Anything-Large) | `7aafbcb` | `43df27c6…` |
| `raft-large` | `raft_large_C_T_SKHT_V2.pth` | torchvision `Raft_Large_Weights.C_T_SKHT_V2` | -- | `ff5fadd5…` |
| `cotracker3` | `scaled_offline.pth` | [facebook/cotracker3](https://huggingface.co/facebook/cotracker3) | `bf55ea5` | `2670d456…` |
| `moge-3-vitl` | `model.pt` | [Ruicheng/moge-3-vitl](https://huggingface.co/Ruicheng/moge-3-vitl) | `184008f` | `9b41b7b9…` |
| `uni3d-l` | `model.pt` | [BAAI/Uni3D](https://huggingface.co/BAAI/Uni3D), `modelzoo/uni3d-l/model.pt` | `3d8233b` | `540a3caf…` |

### Models by Stage

| Stage | Models Used |
|---|---|
| `prepare` | RAFT (Flow), CoTracker3 (Track2D), Video Depth Anything (Depth), MoGe-3 (Uni3D MoGe), DINOv3 (Semantic, GeoPhys), TIPSv2 (Semantic) |
| `score` | DINOv3, TIPSv2, Uni3D |
| `viz` | DINOv3, TIPSv2 (feature visualizations) |
