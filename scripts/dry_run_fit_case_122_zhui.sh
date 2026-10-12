#!/usr/bin/env bash
set -euo pipefail
cd /home/dsj/code/language_affordance_grasp
PYTHONPATH=src python -m language_affordance_grasp run \
  --rgb /home/dsj/code/sam-3d-objects/fit_case_122_zhui/rgb.png \
  --depth /home/dsj/code/sam-3d-objects/fit_case_122_zhui/depth.png \
  --mask /home/dsj/code/sam-3d-objects/fit_case_122_zhui/mask.png \
  --table-mask /home/dsj/code/sam-3d-objects/fit_case_122_zhui/table_mask.png \
  --intrinsics-json /home/dsj/code/sam-3d-objects/fit_case_122_zhui/intrinsics.json \
  --instruction "where should i interact to open the jar lid?" \
  --output-dir /tmp/language_affordance_grasp_dry_run \
  --dry-run
