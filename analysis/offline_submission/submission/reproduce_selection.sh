#!/bin/sh
set -eu
if [ "$#" -ne 1 ]; then
  echo 'Usage from repository root: sh analysis/offline_submission/submission/reproduce_selection.sh NEW_OUTPUT_DIR' >&2
  exit 2
fi
env -u PYTHONPATH uv run --locked --group gui python analysis/offline_submission/select_candidates.py \
  --proposals 20261029-125052-optimize-baseline 20261029-125052-optimize-extratrees-default \
  20261029-125052-optimize-histgb-v32 20261029-125052-optimize-xgboost-v32 \
  runs/20261029-125621-optimize-offline-* \
  --judges histgb=20261029-125052-optimize-histgb-v32 xgboost=20261029-125052-optimize-xgboost-v32 \
  extratrees=20261029-125052-optimize-extratrees-default ridge2=20261029-125052-optimize-baseline \
  --evaluations histgb=20261029-125034-evaluate-histgb-v32 xgboost=20261029-125034-evaluate-xgboost-v32 \
  --out "$1"
