#!/usr/bin/env bash
set -euo pipefail

LAG_REPO=/home/dsj/code/language_affordance_grasp
LAG_API_PYTHON="${LAG_API_PYTHON:-/home/dsj/miniconda3/envs/sam3d-objects/bin/python}"
LAG_API_HOST="${LAG_API_HOST:-127.0.0.1}"
LAG_API_PORT="${LAG_API_PORT:-8000}"

if [[ -z "${LAG_API_TOKEN:-}" ]]; then
  echo "LAG_API_TOKEN is required." >&2
  exit 2
fi
if [[ -z "${LAG_EYE_IN_HAND_CALIBRATION:-}" ]]; then
  echo "LAG_EYE_IN_HAND_CALIBRATION must point to the confirmed T_ee_cam file." >&2
  exit 2
fi
if [[ ! -f "$LAG_EYE_IN_HAND_CALIBRATION" ]]; then
  echo "Calibration file not found: $LAG_EYE_IN_HAND_CALIBRATION" >&2
  exit 2
fi

export PYTHONPATH="$LAG_REPO/src"
export LAG_TITRATION_RUN_ROOT="${LAG_TITRATION_RUN_ROOT:-${LAG_API_RUN_ROOT:-$LAG_REPO/runs/api_jobs}}"
export LAG_COMPLEX_DEMO_RUN_ROOT="${LAG_COMPLEX_DEMO_RUN_ROOT:-$LAG_REPO/runs/complex_demo_jobs}"

cd "$LAG_REPO"
exec "$LAG_API_PYTHON" -m uvicorn language_affordance_grasp.complex_demo_api:app \
  --host "$LAG_API_HOST" \
  --port "$LAG_API_PORT" \
  --workers 1
