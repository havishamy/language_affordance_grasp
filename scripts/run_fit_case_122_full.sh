#!/usr/bin/env bash
set -euo pipefail

LAG_REPO=/home/dsj/code/language_affordance_grasp
LAG_CASE=/home/dsj/code/sam-3d-objects/fit_case_122
LAG_CONFIG="$LAG_REPO/configs/fit_case_122_full.yaml"
LAG_PYTHON=/home/dsj/miniconda3/envs/sam3d-objects/bin/python
LAG_MIN_FREE_MB="${LAG_MIN_FREE_MB:-22000}"
LAG_INSTRUCTION="${LAG_INSTRUCTION:-Where should I interact with the bottle to open the cap?}"
LAG_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LAG_OUTPUT_DIR="${2:-$LAG_REPO/runs/fit_case_122_full_$LAG_TIMESTAMP}"

if [[ -n "${1:-}" ]]; then
  LAG_GPU_INDEX="$1"
else
  LAG_GPU_ROW="$({ nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits; } | sort -t, -k2,2nr | head -n 1)"
  LAG_GPU_INDEX="${LAG_GPU_ROW%%,*}"
  LAG_GPU_INDEX="${LAG_GPU_INDEX//[[:space:]]/}"
fi

LAG_FREE_MB="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F, -v gpu_target="$LAG_GPU_INDEX" '
  {
    gpu_index = $1
    free_mb = $2
    gsub(/[[:space:]]/, "", gpu_index)
    gsub(/[[:space:]]/, "", free_mb)
    if (gpu_index == gpu_target) print free_mb
  }
')"

if [[ -z "$LAG_FREE_MB" ]]; then
  echo "Unable to find GPU index $LAG_GPU_INDEX." >&2
  exit 2
fi
if (( LAG_FREE_MB < LAG_MIN_FREE_MB )); then
  echo "GPU $LAG_GPU_INDEX has only ${LAG_FREE_MB} MiB free; SAM3D requires an almost empty 24 GB card." >&2
  echo "Choose another GPU or override the guard with LAG_MIN_FREE_MB=<MiB>." >&2
  exit 3
fi

mkdir -p "$LAG_OUTPUT_DIR" "$LAG_OUTPUT_DIR/.matplotlib" "$LAG_OUTPUT_DIR/.config"

export CUDA_VISIBLE_DEVICES="$LAG_GPU_INDEX"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export MPLCONFIGDIR="$LAG_OUTPUT_DIR/.matplotlib"
export XDG_CONFIG_HOME="$LAG_OUTPUT_DIR/.config"
export PATH="/home/dsj/miniconda3/bin:$PATH"

LAG_ARGS=(
  run
  --config "$LAG_CONFIG"
  --rgb "$LAG_CASE/rgb.png"
  --depth "$LAG_CASE/depth.png"
  --table-mask "$LAG_CASE/table_mask.png"
  --intrinsics-json "$LAG_CASE/intrinsics.json"
  --instruction "$LAG_INSTRUCTION"
  --yolo-class-id 4
  --output-dir "$LAG_OUTPUT_DIR"
)

if [[ -n "${LAG_T_BASE_CAM:-}" ]]; then
  LAG_ARGS+=(--t-base-cam "$LAG_T_BASE_CAM")
fi
if [[ "${LAG_DRY_RUN:-0}" == "1" ]]; then
  LAG_ARGS+=(--dry-run)
fi

echo "Using physical GPU $LAG_GPU_INDEX (${LAG_FREE_MB} MiB free)."
echo "Output directory: $LAG_OUTPUT_DIR"

cd "$LAG_REPO"
PYTHONPATH=src "$LAG_PYTHON" -m language_affordance_grasp "${LAG_ARGS[@]}" 2>&1 | tee "$LAG_OUTPUT_DIR/pipeline.log"
