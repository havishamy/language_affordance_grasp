#!/usr/bin/env bash
set -euo pipefail

LAG_REPO=/home/dsj/code/language_affordance_grasp
LAG_YOLO_REPO=/home/dsj/code/yolo_3D
LAG_FASTSAM_REPO=/home/dsj/code/FastSAM
LAG_SAM3D_REPO=/home/dsj/code/sam-3d-objects
LAG_YOLO_EXPORTER="$LAG_REPO/src/language_affordance_grasp/yolo_mask_export.py"
LAG_YOLO_WEIGHTS="${LAG_YOLO_WEIGHTS:-$LAG_YOLO_REPO/runs_edge_p2_sim/train-seg/exp/weights/best.pt}"
LAG_YOLO_DATA="${LAG_YOLO_DATA:-/home/dsj/dataset/mydata/server/mydata.yaml}"
LAG_FASTSAM_CHECKPOINT="${LAG_FASTSAM_CHECKPOINT:-$LAG_FASTSAM_REPO/affordance_runs/chem_hova_fullimg_e5/best.pt}"
LAG_PIPELINE_PYTHON="${LAG_PIPELINE_PYTHON:-/home/dsj/miniconda3/envs/sam3d-objects/bin/python}"
LAG_CONDA="${LAG_CONDA:-/home/dsj/miniconda3/bin/conda}"
LAG_MIN_FREE_MB="${LAG_MIN_FREE_MB:-22000}"

# Dataset class IDs from /home/dsj/dataset/mydata/server/mydata.yaml.
LAG_DROPPER_CLASS_ID="${LAG_DROPPER_CLASS_ID:-2}"
LAG_BEAKER_CLASS_ID="${LAG_BEAKER_CLASS_ID:-6}"
LAG_WIDE_MOUTH_BOTTLE_CLASS_ID="${LAG_WIDE_MOUTH_BOTTLE_CLASS_ID:-14}"

LAG_STOPPER_INSTRUCTION="${LAG_STOPPER_INSTRUCTION:-Where should I interact with the jar to open it?}"
LAG_BODY_GRASP_INSTRUCTION="${LAG_BODY_GRASP_INSTRUCTION:-Where should I interact with the jar to hold it?}"
LAG_BEAKER_MOUTH_INSTRUCTION="${LAG_BEAKER_MOUTH_INSTRUCTION:-Where should I interact with the beaker to pour it?}"
LAG_BULB_INSTRUCTION="${LAG_BULB_INSTRUCTION:-Where should I interact with the Glass dropper bottle to dispense it?}"
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
LAG_CAPTURE_ID=
LAG_JOB_ID=
LAG_REQUESTED_OBJECTS_JSON='["wide_mouth_bottle","beaker","dropper"]'
LAG_REQUESTED_AFFORDANCE_REGIONS_JSON='{"wide_mouth_bottle":["stopper","body_grasp"],"beaker":["mouth"],"dropper":["bulb"]}'
LAG_DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  run_complex_demo_three_object.sh \
    --rgb RGB.png \
    --depth DEPTH.png \
    --intrinsics camera.json \
    --calibration T_EE_CAM.json \
    --world-ee T_WORLD_EE.json \
    --capture-id CAPTURE_ID \
    [--job-id JOB_ID] \
    [--requested-objects-json JSON] \
    [--requested-affordance-regions-json JSON] \
    [--table-mask TABLE_MASK.png] \
    [--depth-scale 0.001] \
    [--gpu PHYSICAL_GPU_INDEX] \
    [--output-dir OUTPUT_DIR] \
    [--dry-run]

The supported request is intentionally strict:
  objects: ["wide_mouth_bottle", "beaker", "dropper"]
  regions: {"wide_mouth_bottle":["stopper","body_grasp"],"beaker":["mouth"],"dropper":["bulb"]}

Environment overrides:
  LAG_WIDE_MOUTH_BOTTLE_CLASS_ID=14
  LAG_BEAKER_CLASS_ID=6
  LAG_DROPPER_CLASS_ID=2
  LAG_STOPPER_INSTRUCTION="..."
  LAG_BODY_GRASP_INSTRUCTION="..."
  LAG_BEAKER_MOUTH_INSTRUCTION="..."
  LAG_BULB_INSTRUCTION="..."
  LAG_MIN_FREE_MB=22000
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rgb) LAG_RGB="$2"; shift 2 ;;
    --depth) LAG_DEPTH="$2"; shift 2 ;;
    --intrinsics) LAG_INTRINSICS="$2"; shift 2 ;;
    --calibration) LAG_CALIBRATION="$2"; shift 2 ;;
    --world-ee) LAG_WORLD_EE="$2"; shift 2 ;;
    --capture-id) LAG_CAPTURE_ID="$2"; shift 2 ;;
    --job-id) LAG_JOB_ID="$2"; shift 2 ;;
    --requested-objects-json) LAG_REQUESTED_OBJECTS_JSON="$2"; shift 2 ;;
    --requested-affordance-regions-json) LAG_REQUESTED_AFFORDANCE_REGIONS_JSON="$2"; shift 2 ;;
    --table-mask) LAG_TABLE_MASK="$2"; shift 2 ;;
    --depth-scale) LAG_DEPTH_SCALE="$2"; shift 2 ;;
    --gpu) LAG_GPU_INDEX="$2"; shift 2 ;;
    --output-dir) LAG_OUTPUT_DIR="$2"; shift 2 ;;
    --dry-run) LAG_DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for LAG_REQUIRED_VALUE in "$LAG_RGB" "$LAG_DEPTH" "$LAG_INTRINSICS" "$LAG_CALIBRATION" "$LAG_WORLD_EE"; do
  if [[ -z "$LAG_REQUIRED_VALUE" ]]; then
    echo "--rgb, --depth, --intrinsics, --calibration, and --world-ee are required." >&2
    usage >&2
    exit 2
  fi
  if [[ ! -f "$LAG_REQUIRED_VALUE" ]]; then
    echo "Input file not found: $LAG_REQUIRED_VALUE" >&2
    exit 2
  fi
done
if [[ -z "$LAG_CAPTURE_ID" ]]; then
  echo "--capture-id is required." >&2
  exit 2
fi
if [[ -n "$LAG_TABLE_MASK" && ! -f "$LAG_TABLE_MASK" ]]; then
  echo "Table mask not found: $LAG_TABLE_MASK" >&2
  exit 2
fi
if [[ -z "$LAG_JOB_ID" ]]; then
  LAG_JOB_ID="standalone-$(date +%Y%m%d-%H%M%S)"
fi

if ! LAG_GPU_INVENTORY="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null)"; then
  if [[ "$LAG_DRY_RUN" == 1 ]]; then
    LAG_GPU_INDEX="${LAG_GPU_INDEX:-0}"
    LAG_FREE_MB=unknown
  else
    echo "Unable to query GPUs with nvidia-smi." >&2
    exit 3
  fi
else
  if [[ -z "$LAG_GPU_INDEX" ]]; then
    LAG_GPU_ROW="$(printf '%s\n' "$LAG_GPU_INVENTORY" | sort -t, -k2,2nr | head -n 1)"
    LAG_GPU_INDEX="${LAG_GPU_ROW%%,*}"
    LAG_GPU_INDEX="${LAG_GPU_INDEX//[[:space:]]/}"
  fi
  LAG_FREE_MB="$(printf '%s\n' "$LAG_GPU_INVENTORY" | awk -F, -v gpu_target="$LAG_GPU_INDEX" '
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
fi
if [[ "$LAG_DRY_RUN" == 0 ]] && (( LAG_FREE_MB < LAG_MIN_FREE_MB )); then
  echo "GPU $LAG_GPU_INDEX has only ${LAG_FREE_MB} MiB free; SAM3D needs an almost empty 24 GB card." >&2
  exit 3
fi

if [[ -z "$LAG_OUTPUT_DIR" ]]; then
  LAG_OUTPUT_DIR="$LAG_REPO/runs/complex_demo_$(date +%Y%m%d_%H%M%S)"
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

LAG_CURRENT_STAGE=initialization
LAG_CURRENT_OBJECT=
LAG_CURRENT_REGION=
LAG_CURRENT_CODE=PIPELINE_COMMAND_FAILED

on_error() {
  local exit_code=$?
  set +e
  "$LAG_PIPELINE_PYTHON" -m language_affordance_grasp.complex_demo_export write-failure \
    --output "$LAG_OUTPUT_DIR/failure.json" \
    --stage "$LAG_CURRENT_STAGE" \
    --object "$LAG_CURRENT_OBJECT" \
    --region "$LAG_CURRENT_REGION" \
    --code "$LAG_CURRENT_CODE" \
    --message "Stage '$LAG_CURRENT_STAGE' exited with code $exit_code; inspect pipeline.log for details."
  exit "$exit_code"
}
trap on_error ERR

run_stage() {
  LAG_CURRENT_STAGE="$1"
  LAG_CURRENT_OBJECT="$2"
  LAG_CURRENT_REGION="$3"
  LAG_CURRENT_CODE="$4"
  shift 4
  echo
  echo "[STAGE] $LAG_CURRENT_STAGE"
  if [[ "$LAG_DRY_RUN" == 1 ]]; then
    printf '[DRY_RUN]'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

LAG_WIDE_SEG="$LAG_OUTPUT_DIR/00_segmentation/wide_mouth_bottle"
LAG_BEAKER_SEG="$LAG_OUTPUT_DIR/00_segmentation/beaker"
LAG_DROPPER_SEG="$LAG_OUTPUT_DIR/00_segmentation/dropper"
LAG_STOPPER_AFF="$LAG_OUTPUT_DIR/01_affordance/wide_mouth_bottle/stopper"
LAG_BODY_AFF="$LAG_OUTPUT_DIR/01_affordance/wide_mouth_bottle/body_grasp"
LAG_BEAKER_MOUTH_AFF="$LAG_OUTPUT_DIR/01_affordance/beaker/mouth"
LAG_BULB_AFF="$LAG_OUTPUT_DIR/01_affordance/dropper/bulb"
LAG_WIDE_3D="$LAG_OUTPUT_DIR/02_sam3d/wide_mouth_bottle"
LAG_BEAKER_3D="$LAG_OUTPUT_DIR/02_sam3d/beaker"
LAG_DROPPER_3D="$LAG_OUTPUT_DIR/02_sam3d/dropper"
LAG_STOPPER_PROJ="$LAG_OUTPUT_DIR/03_projection/wide_mouth_bottle/stopper"
LAG_BODY_PROJ="$LAG_OUTPUT_DIR/03_projection/wide_mouth_bottle/body_grasp"
LAG_BEAKER_MOUTH_PROJ="$LAG_OUTPUT_DIR/03_projection/beaker/mouth"
LAG_BULB_PROJ="$LAG_OUTPUT_DIR/03_projection/dropper/bulb"
LAG_CANONICAL_INTRINSICS="$LAG_OUTPUT_DIR/run_inputs/intrinsics_canonical.json"

echo "Physical GPU: $LAG_GPU_INDEX (${LAG_FREE_MB} MiB free at start)"
echo "Output directory: $LAG_OUTPUT_DIR"
echo "Classes: wide-mouth bottle=$LAG_WIDE_MOUTH_BOTTLE_CLASS_ID, beaker=$LAG_BEAKER_CLASS_ID, dropper=$LAG_DROPPER_CLASS_ID"

run_stage "input_validation" "" "" "INVALID_INPUT" \
  "$LAG_PIPELINE_PYTHON" -m language_affordance_grasp.complex_demo_export validate-inputs \
  --rgb "$LAG_RGB" --depth "$LAG_DEPTH" --intrinsics "$LAG_INTRINSICS" --world-ee "$LAG_WORLD_EE" \
  --depth-scale "$LAG_DEPTH_SCALE" --capture-id "$LAG_CAPTURE_ID" \
  --requested-objects-json "$LAG_REQUESTED_OBJECTS_JSON" \
  --requested-affordance-regions-json "$LAG_REQUESTED_AFFORDANCE_REGIONS_JSON" \
  --output "$LAG_OUTPUT_DIR/run_inputs/input_validation.json"

run_stage "intrinsics_normalization" "" "" "INVALID_INTRINSICS" \
  "$LAG_PIPELINE_PYTHON" -m language_affordance_grasp.titration_targets normalize-intrinsics \
  --input "$LAG_INTRINSICS" --output "$LAG_CANONICAL_INTRINSICS"

run_stage "calibration_validation" "" "" "INVALID_CALIBRATION_CHAIN" \
  "$LAG_PIPELINE_PYTHON" -m language_affordance_grasp.titration_targets validate-calibration \
  --calibration "$LAG_CALIBRATION" --world-ee "$LAG_WORLD_EE"

run_stage "object_segmentation" "wide_mouth_bottle" "" "OBJECT_SEGMENTATION_FAILED" \
  "$LAG_CONDA" run --no-capture-output -n base python "$LAG_YOLO_EXPORTER" \
  --repo "$LAG_YOLO_REPO" --weights "$LAG_YOLO_WEIGHTS" --data "$LAG_YOLO_DATA" \
  --rgb "$LAG_RGB" --output-dir "$LAG_WIDE_SEG" --class-id "$LAG_WIDE_MOUTH_BOTTLE_CLASS_ID" \
  --selection best --imgsz 640 --conf-thres 0.25 --iou-thres 0.45 --max-det 1000 --device 0 --mask-thresh 0.5

run_stage "object_segmentation" "beaker" "" "OBJECT_SEGMENTATION_FAILED" \
  "$LAG_CONDA" run --no-capture-output -n base python "$LAG_YOLO_EXPORTER" \
  --repo "$LAG_YOLO_REPO" --weights "$LAG_YOLO_WEIGHTS" --data "$LAG_YOLO_DATA" \
  --rgb "$LAG_RGB" --output-dir "$LAG_BEAKER_SEG" --class-id "$LAG_BEAKER_CLASS_ID" \
  --selection best --imgsz 640 --conf-thres 0.25 --iou-thres 0.45 --max-det 1000 --device 0 --mask-thresh 0.5

run_stage "object_segmentation" "dropper" "" "OBJECT_SEGMENTATION_FAILED" \
  "$LAG_CONDA" run --no-capture-output -n base python "$LAG_YOLO_EXPORTER" \
  --repo "$LAG_YOLO_REPO" --weights "$LAG_YOLO_WEIGHTS" --data "$LAG_YOLO_DATA" \
  --rgb "$LAG_RGB" --output-dir "$LAG_DROPPER_SEG" --class-id "$LAG_DROPPER_CLASS_ID" \
  --selection best --imgsz 640 --conf-thres 0.25 --iou-thres 0.45 --max-det 1000 --device 0 --mask-thresh 0.5

if [[ "$LAG_DRY_RUN" == 0 ]]; then
  run_stage "mask_validation" "wide_mouth_bottle" "" "INVALID_OBJECT_MASK" \
    "$LAG_PIPELINE_PYTHON" -m language_affordance_grasp.titration_targets validate-mask \
    --mask "$LAG_WIDE_SEG/mask.png" --rgb "$LAG_RGB" --label wide_mouth_bottle --min-area 100
  run_stage "mask_validation" "beaker" "" "INVALID_OBJECT_MASK" \
    "$LAG_PIPELINE_PYTHON" -m language_affordance_grasp.titration_targets validate-mask \
    --mask "$LAG_BEAKER_SEG/mask.png" --rgb "$LAG_RGB" --label beaker --min-area 100
  run_stage "mask_validation" "dropper" "" "INVALID_OBJECT_MASK" \
    "$LAG_PIPELINE_PYTHON" -m language_affordance_grasp.titration_targets validate-mask \
    --mask "$LAG_DROPPER_SEG/mask.png" --rgb "$LAG_RGB" --label dropper --min-area 100
fi

run_affordance() {
  local object_name="$1"
  local region_name="$2"
  local target_mask="$3"
  local first_other_mask="$4"
  local second_other_mask="$5"
  local instruction="$6"
  local affordance_dir="$7"

  run_stage "affordance_input_preparation" "$object_name" "$region_name" "AFFORDANCE_PREPARATION_FAILED" \
    "$LAG_PIPELINE_PYTHON" -m language_affordance_grasp.affordance_roi prepare \
    --method "$LAG_AFFORDANCE_METHOD" --rgb "$LAG_RGB" --target-mask "$target_mask" \
    --other-mask "$first_other_mask" --other-mask "$second_other_mask" \
    --expansion 0.35 --other-dilation "$LAG_AFFORDANCE_OTHER_DILATION" --output-dir "$affordance_dir"

  run_stage "affordance_inference" "$object_name" "$region_name" "AFFORDANCE_INFERENCE_FAILED" \
    env PYTHONPATH="$LAG_FASTSAM_REPO" "$LAG_CONDA" run --no-capture-output -n sam3d-objects \
    python -m affordance.infer_chem_hova --checkpoint "$LAG_FASTSAM_CHECKPOINT" \
    --rgb "$affordance_dir/model_input.png" --instruction "$instruction" \
    --output-dir "$affordance_dir/model_output" --device cuda

  run_stage "affordance_remap" "$object_name" "$region_name" "AFFORDANCE_REMAP_FAILED" \
    "$LAG_PIPELINE_PYTHON" -m language_affordance_grasp.affordance_roi remap \
    --metadata "$affordance_dir/prepare_metadata.json" \
    --local-heatmap "$affordance_dir/model_output/affordance_heatmap.npy" \
    --rgb "$LAG_RGB" --output-dir "$affordance_dir"

  if [[ "$LAG_DRY_RUN" == 0 ]]; then
    run_stage "affordance_validation" "$object_name" "$region_name" "INVALID_AFFORDANCE" \
      "$LAG_PIPELINE_PYTHON" -m language_affordance_grasp.complex_demo_export validate-affordance \
      --heatmap "$affordance_dir/affordance_heatmap.npy" --target-mask "$target_mask" \
      --other-mask "$first_other_mask" --other-mask "$second_other_mask" \
      --object "$object_name" --region "$region_name" --output "$affordance_dir/quality_metrics.json"
  fi
}

run_affordance wide_mouth_bottle stopper "$LAG_WIDE_SEG/mask.png" "$LAG_BEAKER_SEG/mask.png" "$LAG_DROPPER_SEG/mask.png" "$LAG_STOPPER_INSTRUCTION" "$LAG_STOPPER_AFF"
run_affordance wide_mouth_bottle body_grasp "$LAG_WIDE_SEG/mask.png" "$LAG_BEAKER_SEG/mask.png" "$LAG_DROPPER_SEG/mask.png" "$LAG_BODY_GRASP_INSTRUCTION" "$LAG_BODY_AFF"
run_affordance beaker mouth "$LAG_BEAKER_SEG/mask.png" "$LAG_WIDE_SEG/mask.png" "$LAG_DROPPER_SEG/mask.png" "$LAG_BEAKER_MOUTH_INSTRUCTION" "$LAG_BEAKER_MOUTH_AFF"
run_affordance dropper bulb "$LAG_DROPPER_SEG/mask.png" "$LAG_WIDE_SEG/mask.png" "$LAG_BEAKER_SEG/mask.png" "$LAG_BULB_INSTRUCTION" "$LAG_BULB_AFF"

if [[ "$LAG_DRY_RUN" == 0 ]]; then
  run_stage "affordance_separation_validation" "wide_mouth_bottle" "stopper_vs_body_grasp" "AFFORDANCE_REGIONS_NOT_SEPARATED" \
    "$LAG_PIPELINE_PYTHON" -m language_affordance_grasp.complex_demo_export validate-separation \
    --first-heatmap "$LAG_STOPPER_AFF/affordance_heatmap.npy" \
    --second-heatmap "$LAG_BODY_AFF/affordance_heatmap.npy" \
    --target-mask "$LAG_WIDE_SEG/mask.png" --object wide_mouth_bottle \
    --first-region stopper --second-region body_grasp --top-fraction 0.15 \
    --max-iou 0.45 --min-vertical-gap 0.12 \
    --output "$LAG_OUTPUT_DIR/01_affordance/wide_mouth_bottle/separation_metrics.json"
fi

LAG_TABLE_ARGS=(--table-mask "")
if [[ -n "$LAG_TABLE_MASK" ]]; then
  LAG_TABLE_ARGS=(--table-mask "$LAG_TABLE_MASK")
fi

run_reconstruction() {
  local object_name="$1"
  local mask_path="$2"
  local object_dir="$3"
  local seed="$4"
  run_stage "object_reconstruction" "$object_name" "" "OBJECT_RECONSTRUCTION_FAILED" \
    "$LAG_CONDA" run --no-capture-output -n sam3d-objects python \
    "$LAG_SAM3D_REPO/recover_glass_pose_notable_graspnet.py" \
    --rgb "$LAG_RGB" --depth "$LAG_DEPTH" --mask "$mask_path" \
    --intrinsics-json "$LAG_CANONICAL_INTRINSICS" --output-dir "$object_dir" \
    --sam3d-config "$LAG_SAM3D_REPO/checkpoints/hf/pipeline.yaml" --depth-scale "$LAG_DEPTH_SCALE" \
    --device cuda --seed "$seed" --optimize-steps 120 --combined-point-samples 120000 --table-point-ratio 0.35 \
    --scale-search-steps 19 --scale-search-min-ratio 0.60 --scale-search-max-ratio 1.05 \
    --scale-only-steps 15 --lr-shift 0.0002 --max-tangent-shift 0.005 \
    --shift-reg-weight 0.05 --shift-reg-reference 0.005 \
    "${LAG_TABLE_ARGS[@]}"
}

run_reconstruction wide_mouth_bottle "$LAG_WIDE_SEG/mask.png" "$LAG_WIDE_3D" 44
run_reconstruction beaker "$LAG_BEAKER_SEG/mask.png" "$LAG_BEAKER_3D" 43
run_reconstruction dropper "$LAG_DROPPER_SEG/mask.png" "$LAG_DROPPER_3D" 42

run_projection() {
  local object_name="$1"
  local region_name="$2"
  local heatmap="$3"
  local mesh="$4"
  local output_dir="$5"
  run_stage "affordance_projection" "$object_name" "$region_name" "AFFORDANCE_PROJECTION_FAILED" \
    "$LAG_PIPELINE_PYTHON" -m language_affordance_grasp project \
    --heatmap "$heatmap" --mesh "$mesh" --intrinsics-json "$LAG_CANONICAL_INTRINSICS" \
    --output-dir "$output_dir"
}

run_projection wide_mouth_bottle stopper "$LAG_STOPPER_AFF/affordance_heatmap.npy" "$LAG_WIDE_3D/optimized_mesh_camera.ply" "$LAG_STOPPER_PROJ"
run_projection wide_mouth_bottle body_grasp "$LAG_BODY_AFF/affordance_heatmap.npy" "$LAG_WIDE_3D/optimized_mesh_camera.ply" "$LAG_BODY_PROJ"
run_projection beaker mouth "$LAG_BEAKER_MOUTH_AFF/affordance_heatmap.npy" "$LAG_BEAKER_3D/optimized_mesh_camera.ply" "$LAG_BEAKER_MOUTH_PROJ"
run_projection dropper bulb "$LAG_BULB_AFF/affordance_heatmap.npy" "$LAG_DROPPER_3D/optimized_mesh_camera.ply" "$LAG_BULB_PROJ"

run_stage "result_export" "" "" "RESULT_EXPORT_FAILED" \
  "$LAG_PIPELINE_PYTHON" -m language_affordance_grasp.complex_demo_export export-results \
  --output-dir "$LAG_OUTPUT_DIR" --job-id "$LAG_JOB_ID" --capture-id "$LAG_CAPTURE_ID" \
  --requested-objects-json "$LAG_REQUESTED_OBJECTS_JSON" \
  --requested-affordance-regions-json "$LAG_REQUESTED_AFFORDANCE_REGIONS_JSON" \
  --calibration "$LAG_CALIBRATION" --world-ee "$LAG_WORLD_EE" \
  --rgb "$LAG_RGB" --depth "$LAG_DEPTH" --intrinsics "$LAG_CANONICAL_INTRINSICS" --depth-scale "$LAG_DEPTH_SCALE" \
  --wide-mouth-bottle-mask "$LAG_WIDE_SEG/mask.png" --beaker-mask "$LAG_BEAKER_SEG/mask.png" \
  --dropper-mask "$LAG_DROPPER_SEG/mask.png" \
  --wide-mouth-bottle-mesh "$LAG_WIDE_3D/optimized_mesh_camera.ply" \
  --beaker-mesh "$LAG_BEAKER_3D/optimized_mesh_camera.ply" \
  --dropper-mesh "$LAG_DROPPER_3D/optimized_mesh_camera.ply" \
  --wide-mouth-bottle-pose "$LAG_WIDE_3D/pose_result.json" \
  --beaker-pose "$LAG_BEAKER_3D/pose_result.json" --dropper-pose "$LAG_DROPPER_3D/pose_result.json" \
  --stopper-region "$LAG_STOPPER_PROJ/affordance_region_camera.npz" \
  --body-grasp-region "$LAG_BODY_PROJ/affordance_region_camera.npz" \
  --beaker-mouth-region "$LAG_BEAKER_MOUTH_PROJ/affordance_region_camera.npz" \
  --bulb-region "$LAG_BULB_PROJ/affordance_region_camera.npz" \
  --stopper-heatmap "$LAG_STOPPER_AFF/affordance_heatmap.npy" \
  --body-grasp-heatmap "$LAG_BODY_AFF/affordance_heatmap.npy" \
  --beaker-mouth-heatmap "$LAG_BEAKER_MOUTH_AFF/affordance_heatmap.npy" \
  --bulb-heatmap "$LAG_BULB_AFF/affordance_heatmap.npy"

if [[ "$LAG_DRY_RUN" == 0 ]]; then
  echo
  echo "Done. Contract result: $LAG_OUTPUT_DIR/result.json"
fi
