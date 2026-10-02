#!/usr/bin/env bash
# Download every model checkpoint the scorer loads into one directory, and verify them.
#
#     scripts/download_checkpoints.sh [DIR]        # DIR defaults to ./checkpoints
#
# Needs `hf` (pip install -U "huggingface_hub[cli]"), `curl` and `sha256sum`, and a
# Hugging Face login (`hf auth login`) whose account has been granted access to the
# gated facebook/dinov3-vitl16-pretrain-lvd1689m. About 6.7 GB in all. Every download
# is pinned to the revision the paper's numbers were computed with, and each weight
# file is checked against its SHA-256 at the end.
set -euo pipefail

dir="${1:-checkpoints}"
mkdir -p "$dir"
dir="$(cd "$dir" && pwd)"

get() {   # get <repo> <revision> <subdirectory> [file ...]: a Hugging Face snapshot, or some of its files
    local repo="$1" revision="$2" target="$dir/$3"
    shift 3
    echo "== $repo -> $target"
    hf download "$repo" "$@" --revision "$revision" --local-dir "$target"
}

get facebook/dinov3-vitl16-pretrain-lvd1689m ea8dc2863c51be0a264bab82070e3e8836b02d51 dinov3-vitl16-pretrain-lvd1689m
get google/tipsv2-l14 52847a71eb02a3082c370fbcd98cd799687bd4d0 tipsv2-l14
get depth-anything/Video-Depth-Anything-Large 7aafbcb5c6af0bac741aad2b6471894fb4761afa video-depth-anything-large \
    video_depth_anything_vitl.pth
get facebook/cotracker3 bf55ea50d4390e1820a267f131cd6587240fb2c5 cotracker3 scaled_offline.pth
get Ruicheng/moge-3-vitl 184008f877d7ad1ad4c2cd2182a9bd1f63d0e5be moge-3-vitl model.pt

# Uni3D keeps its weights under modelzoo/; the scorer reads uni3d-l/model.pt
get BAAI/Uni3D 3d8233b76aa350d72f6213ecd2123c2026b42355 uni3d-l modelzoo/uni3d-l/model.pt
mv "$dir/uni3d-l/modelzoo/uni3d-l/model.pt" "$dir/uni3d-l/model.pt"
rm -rf "$dir/uni3d-l/modelzoo"

# torchvision's RAFT-large, Raft_Large_Weights.C_T_SKHT_V2
echo "== RAFT-large -> $dir/raft-large"
mkdir -p "$dir/raft-large"
curl -fL https://download.pytorch.org/models/raft_large_C_T_SKHT_V2-ff5fadd5.pth \
    -o "$dir/raft-large/raft_large_C_T_SKHT_V2.pth"

echo "== verifying"
cd "$dir"
sha256sum -c - <<'EOF'
dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179  dinov3-vitl16-pretrain-lvd1689m/model.safetensors
c75707eabca655e641a6c47a59dd26b347290ceadb993b34af184e7f654c5809  tipsv2-l14/model.safetensors
43df27c6b396042ba34ff7b798ab279f64d204d2e86d7a373968f8fa36d0e6fa  video-depth-anything-large/video_depth_anything_vitl.pth
ff5fadd56d26b40647388883af1547351ea17868b765c05b27231e72dd16a322  raft-large/raft_large_C_T_SKHT_V2.pth
2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834  cotracker3/scaled_offline.pth
9b41b7b9f65ad80aab7ad686f5e9cc0d1fd33f1964022618dfbcd52fc1fb7925  moge-3-vitl/model.pt
540a3cafac52c251cbda1844d109bc598f142b4f5a2118db87cd722b01800c20  uni3d-l/model.pt
EOF
echo "checkpoints ready in $dir"
