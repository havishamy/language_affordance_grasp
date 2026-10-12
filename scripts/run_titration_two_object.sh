#!/usr/bin/env bash
set -euo pipefail

LAG_REPO=/home/dsj/code/language_affordance_grasp
LAG_YOLO_REPO=/home/dsj/code/yolo_3D
LAG_FASTSAM_REPO=/home/dsj/code/FastSAM
LAG_SAM3D_REPO=/home/dsj/code/sam-3d-objects
LAG_YOLO_EXPORTER="$LAG_REPO/src/language_affordance_grasp/yolo_mask_export.py"
LAG_YOLO_WEIGHTS="$LAG_YOLO_REPO/runs_edge_p2_sim/train-seg/exp/weights/best.pt"
LAG_YOLO_DATA=/home/dsj/dataset/mydata/server/mydata.yaml
LAG_FASTSAM_CHECKPOINT="$LAG_FASTSAM_REPO/affordance_runs/chem_hova_fullimg_e5/best.pt"
LAG_PIPELINE_PYTHON=/home/dsj/miniconda3/envs/sam3d-objects/bin/python
LAG_CONDA=/home/dsj/miniconda3/bin/conda
LAG_MIN_FREE_MB="${LAG_MIN_FREE_MB:-22000}"
LAG_DROPPER_CLASS_ID="${LAG_DROPPER_CLASS_ID:-2}"
LAG_BEAKER_CLASS_ID="${LAG_BEAKER_CLASS_ID:-6}"
LAG_DROPPER_INSTRUCTION="${LAG_DROPPER_INSTRUCTION:-Where should I interact with the Glass dropper bottle to dispense it?}"
LAG_BEAKER_INSTRUCTION="${LAG_BEAKER_INSTRUCTION:-Where should I interact with the beaker to pour it?}"
LAG_AFFORDANCE_METHOD="${LAG_AFFORDANCE_METHOD:-mask_others}"
LAG_AFFORDANCE_OTHER_DILATION="${LAG_AFFORDANCE_OTHER_DILATION:-12}"

LAG_RGB=
LAG_DEPTH=
LAG_INTRINSICS=
LAG_CALIBRATION=
LAG_WORLD_EE=
LAG_TABLE_MASK=
LAG_DEPTH_SCALE=0.001
LAG_GPU_INDEX=
LAG_OUTPUT_DIR=
LAG_DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  run_titration_two_object.sh \
    --rgb RGB.png \
    --depth DEPTH.png \
    --intrinsics INTRINSICS.json \
    --calibration CALIBRATION.{json,npy,txt} \
    [--world-ee T_WORLD_EE.{json,npy,txt}] \
    [--table-mask TABLE_MASK.png] \
    [--depth-scale 0.001] \
    [--gpu PHYSICAL_GPU_INDEX] \
    [--output-dir OUTPUT_DIR] \
    [--dry-run]

Calibration input can be either:
  1. A direct T_world_cam/T_base_cam 4x4 matrix; or
  2. An eye-in-hand JSON containing T_ee_cam and T_world_ee; or
  3. A T_ee_cam matrix plus --world-ee recorded at image capture time.

Environment overrides:
  LAG_DROPPER_CLASS_ID=2
  LAG_BEAKER_CLASS_ID=6
  LAG_DROPPER_INSTRUCTION="..."
  LAG_BEAKER_INSTRUCTION="..."
  LAG_MIN_FREE_MB=22000
  LAG_AFFORDANCE_METHOD=mask_others
  LAG_AFFORDANCE_OTHER_DILATION=12
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rgb) LAG_RGB="$2"; shift 2 ;;
    --depth) LAG_DEPTH="$2"; shift 2 ;;
    --intrinsics) LAG_INTRINSICS="$2"; shift 2 ;;
    --calibration) LAG_CALIBRATION="$2"; shift 2 ;;
    --world-ee) LAG_WORLD_EE="$2"; shift 2 ;;
    --table-mask) LAG_TABLE_MASK="$2"; shift 2 ;;
    --depth-scale) LAG_DEPTH_SCALE="$2"; shift 2 ;;
    --gpu) LAG_GPU_INDEX="$2"; shift 2 ;;
    --output-dir) LAG_OUTPUT_DIR="$2"; shift 2 ;;
    --dry-run) LAG_DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for LAG_REQUIRED_VALUE in "$LAG_RGB" "$LAG_DEPTH" "$LAG_INTRINSICS" "$LAG_CALIBRATION"; do
  if [[ -z "$LAG_REQUIRED_VALUE" ]]; then
    echo "--rgb, --depth, --intrinsics, and --calibration are required." >&2
    usage >&2
    exit 2
  fi
  if [[ ! -f "$LAG_REQUIRED_VALUE" ]]; then
    echo "Input file not found: $LAG_REQUIRED_VALUE" >&2
    exit 2
  fi
done
if [[ -n "$LAG_WORLD_EE" && ! -f "$LAG_WORLD_EE" ]]; then
  echo "World-to-end-effector transform not found: $LAG_WORLD_EE" >&2
  exit 2
fi
if [[ -n "$LAG_TABLE_MASK" && ! -f "$LAG_TABLE_MASK" ]]; then
  echo "Table mask not found: $LAG_TABLE_MASK" >&2
  exit 2
fi

if [[ -z "$LAG_GPU_INDEX" ]]; then
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
if [[ "$LAG_DRY_RUN" == 0 ]] && (( LAG_FREE_MB < LAG_MIN_FREE_MB )); then
  echo "GPU $LAG_GPU_INDEX has only ${LAG_FREE_MB} MiB free; SAM3D needs an almost empty 24 GB card." >&2
  exit 3
fi

if [[ -z "$LAG_OUTPUT_DIR" ]]; then
  LAG_OUTPUT_DIR="$LAG_REPO/runs/titration_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$LAG_OUTPUT_DIR" "$LAG_OUTPUT_DIR/.matplotlib" "$LAG_OUTPUT_DIR/.config"

export CUDA_VISIBLE_DEVICES="$LAG_GPU_INDEX"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export MPLCONFIGDIR="$LAG_OUTPUT_DIR/.matplotlib"
export XDG_CONFIG_HOME="$LAG_OUTPUT_DIR/.config"
export PATH="/home/dsj/miniconda3/bin:$PATH"
export PYTHONPATH="$LAG_REPO/src"

exec > >(tee "$LAG_OUTPUT_DIR/pipeline.log") 2>&1

run_stage() {
  echo
  echo "[STAGE] $1"
  shift
  if [[ "$LAG_DRY_RUN" == 1 ]]; then
    printf '[DRY_RUN]'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

LAG_DROPPER_SEG="$LAG_OUTPUT_DIR/00_segmentation/dropper"
LAG_BEAKER_SEG="$LAG_OUTPUT_DIR/00_segmentation/beaker"
LAG_DROPPER_AFF="$LAG_OUTPUT_DIR/01_affordance/dropper"
LAG_BEAKER_AFF="$LAG_OUTPUT_DIR/01_affordance/beaker"
LAG_DROPPER_3D="$LAG_OUTPUT_DIR/02_sam3d/dropper"
LAG_BEAKER_3D="$LAG_OUTPUT_DIR/02_sam3d/beaker"
LAG_DROPPER_PROJ="$LAG_OUTPUT_DIR/03_projection/dropper"
LAG_BEAKER_PROJ="$LAG_OUTPUT_DIR/03_projection/beaker"
LAG_TARGETS="$LAG_OUTPUT_DIR/04_targets"
LAG_DEPTH_CLOUDS="$LAG_OUTPUT_DIR/00_segmentation/depth_clouds"
LAG_CANONICAL_INTRINSICS="$LAG_OUTPUT_DIR/run_inputs/intrinsics_canonical.json"

echo "Physical GPU: $LAG_GPU_INDEX (${LAG_FREE_MB} MiB free at start)"
echo "Output directory: $LAG_OUTPUT_DIR"
echo "Dropper class id: $LAG_DROPPER_CLASS_ID; beaker class id: $LAG_BEAKER_CLASS_ID"
echo "Affordance isolation method: $LAG_AFFORDANCE_METHOD"

LAG_WORLD_EE_ARGS=()
if [[ -n "$LAG_WORLD_EE" ]]; then
  LAG_WORLD_EE_ARGS=(--world-ee "$LAG_WORLD_EE")
fi

run_stage "Normalize camera intrinsics" \
  "$LAG_PIPELINE_PYTHON" -m language_affordance_grasp.titration_targets normalize-intrinsics \
  --input "$LAG_INTRINSICS" --output "$LAG_CANONICAL_INTRINSICS"

run_stage "Validate and compose eye-in-hand calibration" \
  "$LAG_PIPELINE_PYTHON" -m language_affordance_grasp.titration_targets validate-calibration \
  --calibration "$LAG_CALIBRATION" "${LAG_WORLD_EE_ARGS[@]}"

run_stage "YOLO dropper mask" \
  "$LAG_CONDA" run --no-capture-output -n base python "$LAG_YOLO_EXPORTER" \
  --repo "$LAG_YOLO_REPO" --weights "$LAG_YOLO_WEIGHTS" --data "$LAG_YOLO_DATA" \
  --rgb "$LAG_RGB" --output-dir "$LAG_DROPPER_SEG" --class-id "$LAG_DROPPER_CLASS_ID" \
  --selection best --imgsz 640 --conf-thres 0.25 --iou-thres 0.45 --max-det 1000 --device 0 --mask-thresh 0.5

run_stage "YOLO beaker mask" \
  "$LAG_CONDA" run --no-capture-output -n base python "$LAG_YOLO_EXPORTER" \
  --repo "$LAG_YOLO_REPO" --weights "$LAG_YOLO_WEIGHTS" --data "$LAG_YOLO_DATA" \
  --rgb "$LAG_RGB" --output-dir "$LAG_BEAKER_SEG" --class-id "$LAG_BEAKER_CLASS_ID" \
  --selection best --imgsz 640 --conf-thres 0.25 --iou-thres 0.45 --max-det 1000 --device 0 --mask-thresh 0.5

if [[ "$LAG_DRY_RUN" == 0 ]]; then
  run_stage "Validate object masks" "$LAG_PIPELINE_PYTHON" -m language_affordance_grasp.titration_targets validate-mask \
    --mask "$LAG_DROPPER_SEG/mask.png" --rgb "$LAG_RGB" --label dropper --min-area 100
  run_stage "Validate beaker mask" "$LAG_PIPELINE_PYTHON" -m language_affordance_grasp.titration_targets validate-mask \
    --mask "$LAG_BEAKER_SEG/mask.png" --rgb "$LAG_RGB" --label beaker --min-area 100
fi

run_stage "Back-project two masked RGB-D point clouds" \
  "$LAG_PIPELINE_PYTHON" -m language_affordance_grasp.titration_targets backproject \
  --rgb "$LAG_RGB" --depth "$LAG_DEPTH" --intrinsics "$LAG_CANONICAL_INTRINSICS" \
  --dropper-mask "$LAG_DROPPER_SEG/mask.png" --beaker-mask "$LAG_BEAKER_SEG/mask.png" \
  --output-dir "$LAG_DEPTH_CLOUDS" --depth-scale "$LAG_DEPTH_SCALE" --min-points 50

run_stage "Prepare isolated dropper affordance input" \
  "$LAG_PIPELINE_PYTHON" -m language_affordance_grasp.affordance_roi prepare \
  --method "$LAG_AFFORDANCE_METHOD" --rgb "$LAG_RGB" \
  --target-mask "$LAG_DROPPER_SEG/mask.png" --other-mask "$LAG_BEAKER_SEG/mask.png" \
  --expansion 0.35 --other-dilation "$LAG_AFFORDANCE_OTHER_DILATION" --output-dir "$LAG_DROPPER_AFF"

run_stage "Dispense affordance for red rubber bulb" \
  env PYTHONPATH="$LAG_FASTSAM_REPO" "$LAG_CONDA" run --no-capture-output -n sam3d-objects python -m affordance.infer_chem_hova \
  --checkpoint "$LAG_FASTSAM_CHECKPOINT" --rgb "$LAG_DROPPER_AFF/model_input.png" \
  --instruction "$LAG_DROPPER_INSTRUCTION" --output-dir "$LAG_DROPPER_AFF/model_output" --device cuda

run_stage "Map dropper affordance back to full image" \
  "$LAG_PIPELINE_PYTHON" -m language_affordance_grasp.affordance_roi remap \
  --metadata "$LAG_DROPPER_AFF/prepare_metadata.json" \
  --local-heatmap "$LAG_DROPPER_AFF/model_output/affordance_heatmap.npy" \
  --rgb "$LAG_RGB" --output-dir "$LAG_DROPPER_AFF"
run_stage "Check dropper affordance target" \
  "$LAG_PIPELINE_PYTHON" -m language_affordance_grasp.affordance_roi evaluate \
  --method "$LAG_AFFORDANCE_METHOD" --target-type dropper --rgb "$LAG_RGB" \
  --target-mask "$LAG_DROPPER_SEG/mask.png" --other-mask "$LAG_BEAKER_SEG/mask.png" \
  --heatmap "$LAG_DROPPER_AFF/affordance_heatmap.npy" \
  --output "$LAG_DROPPER_AFF/quality_metrics.json" --require-valid

run_stage "Prepare isolated beaker affordance input" \
  "$LAG_PIPELINE_PYTHON" -m language_affordance_grasp.affordance_roi prepare \
  --method "$LAG_AFFORDANCE_METHOD" --rgb "$LAG_RGB" \
  --target-mask "$LAG_BEAKER_SEG/mask.png" --other-mask "$LAG_DROPPER_SEG/mask.png" \
  --expansion 0.35 --other-dilation "$LAG_AFFORDANCE_OTHER_DILATION" --output-dir "$LAG_BEAKER_AFF"

run_stage "Pour affordance for beaker mouth" \
  env PYTHONPATH="$LAG_FASTSAM_REPO" "$LAG_CONDA" run --no-capture-output -n sam3d-objects python -m affordance.infer_chem_hova \
  --checkpoint "$LAG_FASTSAM_CHECKPOINT" --rgb "$LAG_BEAKER_AFF/model_input.png" \
  --instruction "$LAG_BEAKER_INSTRUCTION" --output-dir "$LAG_BEAKER_AFF/model_output" --device cuda

run_stage "Map beaker affordance back to full image" \
  "$LAG_PIPELINE_PYTHON" -m language_affordance_grasp.affordance_roi remap \
  --metadata "$LAG_BEAKER_AFF/prepare_metadata.json" \
  --local-heatmap "$LAG_BEAKER_AFF/model_output/affordance_heatmap.npy" \
  --rgb "$LAG_RGB" --output-dir "$LAG_BEAKER_AFF"
run_stage "Check beaker affordance target" \
  "$LAG_PIPELINE_PYTHON" -m language_affordance_grasp.affordance_roi evaluate \
  --method "$LAG_AFFORDANCE_METHOD" --target-type beaker --rgb "$LAG_RGB" \
  --target-mask "$LAG_BEAKER_SEG/mask.png" --other-mask "$LAG_DROPPER_SEG/mask.png" \
  --heatmap "$LAG_BEAKER_AFF/affordance_heatmap.npy" \
  --output "$LAG_BEAKER_AFF/quality_metrics.json" --require-valid

LAG_TABLE_ARGS=(--table-mask "")
if [[ -n "$LAG_TABLE_MASK" ]]; then
  LAG_TABLE_ARGS=(--table-mask "$LAG_TABLE_MASK")
fi

run_stage "SAM3D dropper reconstruction and metric pose recovery" \
  "$LAG_CONDA" run --no-capture-output -n sam3d-objects python "$LAG_SAM3D_REPO/recover_glass_pose_notable_graspnet.py" \
  --rgb "$LAG_RGB" --depth "$LAG_DEPTH" --mask "$LAG_DROPPER_SEG/mask.png" \
  --intrinsics-json "$LAG_CANONICAL_INTRINSICS" --output-dir "$LAG_DROPPER_3D" \
  --sam3d-config "$LAG_SAM3D_REPO/checkpoints/hf/pipeline.yaml" --depth-scale "$LAG_DEPTH_SCALE" \
  --device cuda --seed 42 --optimize-steps 120 --combined-point-samples 120000 --table-point-ratio 0.35 \
  "${LAG_TABLE_ARGS[@]}"

run_stage "SAM3D beaker reconstruction and metric pose recovery" \
  "$LAG_CONDA" run --no-capture-output -n sam3d-objects python "$LAG_SAM3D_REPO/recover_glass_pose_notable_graspnet.py" \
  --rgb "$LAG_RGB" --depth "$LAG_DEPTH" --mask "$LAG_BEAKER_SEG/mask.png" \
  --intrinsics-json "$LAG_CANONICAL_INTRINSICS" --output-dir "$LAG_BEAKER_3D" \
  --sam3d-config "$LAG_SAM3D_REPO/checkpoints/hf/pipeline.yaml" --depth-scale "$LAG_DEPTH_SCALE" \
  --device cuda --seed 43 --optimize-steps 120 --combined-point-samples 120000 --table-point-ratio 0.35 \
  "${LAG_TABLE_ARGS[@]}"

run_stage "Project dispense affordance onto dropper mesh" \
  "$LAG_PIPELINE_PYTHON" -m language_affordance_grasp project \
  --heatmap "$LAG_DROPPER_AFF/affordance_heatmap.npy" \
  --mesh "$LAG_DROPPER_3D/optimized_mesh_camera.ply" --intrinsics-json "$LAG_CANONICAL_INTRINSICS" \
  --output-dir "$LAG_DROPPER_PROJ"

run_stage "Project pour affordance onto beaker mesh" \
  "$LAG_PIPELINE_PYTHON" -m language_affordance_grasp project \
  --heatmap "$LAG_BEAKER_AFF/affordance_heatmap.npy" \
  --mesh "$LAG_BEAKER_3D/optimized_mesh_camera.ply" --intrinsics-json "$LAG_CANONICAL_INTRINSICS" \
  --output-dir "$LAG_BEAKER_PROJ"

run_stage "Compute world targets and export colored PLY" \
  "$LAG_PIPELINE_PYTHON" -m language_affordance_grasp.titration_targets finalize \
  --dropper-region "$LAG_DROPPER_PROJ/affordance_region_camera.npz" \
  --beaker-region "$LAG_BEAKER_PROJ/affordance_region_camera.npz" \
  --dropper-mesh "$LAG_DROPPER_3D/optimized_mesh_camera.ply" \
  --beaker-mesh "$LAG_BEAKER_3D/optimized_mesh_camera.ply" \
  --table-mesh "$LAG_DROPPER_3D/table_camera.ply" \
  --dropper-depth-cloud "$LAG_DEPTH_CLOUDS/dropper_depth_camera.ply" \
  --beaker-depth-cloud "$LAG_DEPTH_CLOUDS/beaker_depth_camera.ply" \
  --calibration "$LAG_CALIBRATION" "${LAG_WORLD_EE_ARGS[@]}" \
  --rgb "$LAG_RGB" --intrinsics "$LAG_CANONICAL_INTRINSICS" \
  --output-dir "$LAG_TARGETS" --top-quantile 0.5 --marker-radius 0.008 --model-point-count 50000

if [[ "$LAG_DRY_RUN" == 0 ]]; then
  echo
  echo "Done. Final world coordinates:"
  cat "$LAG_TARGETS/target_centers_world.txt"
fi
