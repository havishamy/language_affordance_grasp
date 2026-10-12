#!/usr/bin/env bash
set -euo pipefail

LAG_REPO=/home/dsj/code/language_affordance_grasp
LAG_YOLO_REPO=/home/dsj/code/yolo_3D
LAG_YOLO_EXPORTER="$LAG_REPO/src/language_affordance_grasp/yolo_mask_export.py"
LAG_YOLO_WEIGHTS="${LAG_YOLO_WEIGHTS:-$LAG_YOLO_REPO/runs_edge_p2_sim/train-seg/exp/weights/best.pt}"
LAG_YOLO_DATA="${LAG_YOLO_DATA:-/home/dsj/dataset/mydata/server/mydata.yaml}"
LAG_PIPELINE_PYTHON="${LAG_PIPELINE_PYTHON:-/home/dsj/miniconda3/envs/sam3d-objects/bin/python}"
LAG_CONDA="${LAG_CONDA:-/home/dsj/miniconda3/bin/conda}"
LAG_GLASS_ROD_CLASS_ID="${LAG_GLASS_ROD_CLASS_ID:-12}"
LAG_YOLO_DEVICE="${LAG_STIRRING_YOLO_DEVICE:-cpu}"
LAG_CONF_THRESHOLD="${LAG_STIRRING_CONF_THRESHOLD:-0.25}"

LAG_RGB=
LAG_DEPTH=
LAG_INTRINSICS=
LAG_CALIBRATION=
LAG_WORLD_EE=
LAG_DEPTH_SCALE=0.001
LAG_CAPTURE_ID=
LAG_JOB_ID=
LAG_OUTPUT_DIR=
LAG_ROD_RADIUS_M=0.0025
LAG_GRASP_INSET_M=0.025
LAG_MIN_LENGTH_M=0.08
LAG_MAX_LENGTH_M=0.40

usage() {
  cat <<'EOF'
Usage:
  run_glass_rod_stirring.sh \
    --rgb RGB.png \
    --depth DEPTH.png \
    --intrinsics camera.json \
    --calibration T_EE_CAM.json \
    --world-ee T_WORLD_EE.json \
    --capture-id CAPTURE_ID \
    --job-id JOB_ID \
    --output-dir OUTPUT_DIR \
    [--depth-scale 0.001] \
    [--rod-radius-m 0.0025] \
    [--grasp-inset-m 0.025] \
    [--min-length-m 0.08] \
    [--max-length-m 0.40]

This pipeline is CPU-first and does not run SAM3D. It assumes the glass rod lies
on the table and returns two non-semantic endpoints plus two grasp candidates.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rgb) LAG_RGB="$2"; shift 2 ;;
    --depth) LAG_DEPTH="$2"; shift 2 ;;
    --intrinsics) LAG_INTRINSICS="$2"; shift 2 ;;
    --calibration) LAG_CALIBRATION="$2"; shift 2 ;;
    --world-ee) LAG_WORLD_EE="$2"; shift 2 ;;
    --depth-scale) LAG_DEPTH_SCALE="$2"; shift 2 ;;
    --capture-id) LAG_CAPTURE_ID="$2"; shift 2 ;;
    --job-id) LAG_JOB_ID="$2"; shift 2 ;;
    --output-dir) LAG_OUTPUT_DIR="$2"; shift 2 ;;
    --rod-radius-m) LAG_ROD_RADIUS_M="$2"; shift 2 ;;
    --grasp-inset-m) LAG_GRASP_INSET_M="$2"; shift 2 ;;
    --min-length-m) LAG_MIN_LENGTH_M="$2"; shift 2 ;;
    --max-length-m) LAG_MAX_LENGTH_M="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for LAG_REQUIRED_VALUE in "$LAG_RGB" "$LAG_DEPTH" "$LAG_INTRINSICS" "$LAG_CALIBRATION" "$LAG_WORLD_EE"; do
  if [[ -z "$LAG_REQUIRED_VALUE" || ! -f "$LAG_REQUIRED_VALUE" ]]; then
    echo "Required input file is missing: $LAG_REQUIRED_VALUE" >&2
    exit 2
  fi
done
if [[ -z "$LAG_CAPTURE_ID" || -z "$LAG_JOB_ID" || -z "$LAG_OUTPUT_DIR" ]]; then
  echo "--capture-id, --job-id, and --output-dir are required." >&2
  exit 2
fi

mkdir -p "$LAG_OUTPUT_DIR/00_segmentation/glass_rod" "$LAG_OUTPUT_DIR/.matplotlib" "$LAG_OUTPUT_DIR/.config"
export PYTHONPATH="$LAG_REPO/src"
export MPLCONFIGDIR="$LAG_OUTPUT_DIR/.matplotlib"
export XDG_CONFIG_HOME="$LAG_OUTPUT_DIR/.config"
export PATH="/home/dsj/miniconda3/bin:$PATH"
exec > >(tee "$LAG_OUTPUT_DIR/pipeline.log") 2>&1

LAG_CURRENT_STAGE=initialization
LAG_CURRENT_CODE=PIPELINE_COMMAND_FAILED
on_error() {
  local exit_code=$?
  set +e
  "$LAG_PIPELINE_PYTHON" -m language_affordance_grasp.complex_demo_export write-failure \
    --output "$LAG_OUTPUT_DIR/failure.json" \
    --stage "$LAG_CURRENT_STAGE" --object glass_rod --code "$LAG_CURRENT_CODE" \
    --message "Stage '$LAG_CURRENT_STAGE' exited with code $exit_code; inspect pipeline.log for details."
  exit "$exit_code"
}
trap on_error ERR

run_stage() {
  LAG_CURRENT_STAGE="$1"
  LAG_CURRENT_CODE="$2"
  shift 2
  echo
  echo "[STAGE] $LAG_CURRENT_STAGE"
  "$@"
}

LAG_ROD_DIR="$LAG_OUTPUT_DIR/00_segmentation/glass_rod"

run_stage "calibration_validation" "INVALID_CALIBRATION_CHAIN" \
  "$LAG_PIPELINE_PYTHON" -m language_affordance_grasp.titration_targets validate-calibration \
  --calibration "$LAG_CALIBRATION" --world-ee "$LAG_WORLD_EE"

run_stage "glass_rod_segmentation" "GLASS_ROD_NOT_DETECTED" \
  "$LAG_CONDA" run --no-capture-output -n base python "$LAG_YOLO_EXPORTER" \
  --repo "$LAG_YOLO_REPO" --weights "$LAG_YOLO_WEIGHTS" --data "$LAG_YOLO_DATA" \
  --rgb "$LAG_RGB" --output-dir "$LAG_ROD_DIR" --class-id "$LAG_GLASS_ROD_CLASS_ID" \
  --selection best --imgsz 640 --conf-thres "$LAG_CONF_THRESHOLD" --iou-thres 0.45 \
  --max-det 1000 --device "$LAG_YOLO_DEVICE" --mask-thresh 0.5

run_stage "glass_rod_mask_validation" "INVALID_GLASS_ROD_MASK" \
  "$LAG_PIPELINE_PYTHON" -m language_affordance_grasp.titration_targets validate-mask \
  --mask "$LAG_ROD_DIR/mask.png" --rgb "$LAG_RGB" --label glass_rod --min-area 100

run_stage "glass_rod_geometry" "GLASS_ROD_GEOMETRY_FAILED" \
  "$LAG_PIPELINE_PYTHON" -m language_affordance_grasp.glass_rod_geometry \
  --rgb "$LAG_RGB" --depth "$LAG_DEPTH" --mask "$LAG_ROD_DIR/mask.png" \
  --detections "$LAG_ROD_DIR/detections.json" --intrinsics "$LAG_INTRINSICS" \
  --calibration "$LAG_CALIBRATION" --world-ee "$LAG_WORLD_EE" --depth-scale "$LAG_DEPTH_SCALE" \
  --capture-id "$LAG_CAPTURE_ID" --job-id "$LAG_JOB_ID" --output-dir "$LAG_OUTPUT_DIR" \
  --rod-radius-m "$LAG_ROD_RADIUS_M" --grasp-inset-m "$LAG_GRASP_INSET_M" \
  --min-length-m "$LAG_MIN_LENGTH_M" --max-length-m "$LAG_MAX_LENGTH_M"

echo
echo "Done. Stirring geometry result: $LAG_OUTPUT_DIR/result.json"
