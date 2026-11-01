#!/bin/sh
set -eu
if [ "$#" -ne 2 ] && [ "$#" -ne 5 ]; then
  echo 'Usage from repository root: sh analysis/soft_weighting/delivery/reproduce.sh NEW_STUDY_DIR NEW_SUMMARY_DIR [HISTGB_REF XGBOOST_REF EXTRATREES_REF]' >&2
  exit 2
fi
env -u PYTHONPATH uv run --locked --group gui python analysis/soft_weighting/run_study.py --out "$1" --workers 2
env -u PYTHONPATH uv run --locked --group gui python analysis/soft_weighting/summarize.py \
  --manifest "$1/run_manifest.json" --out "$2" \
  --reference-histgb "${3:-20261029-125052-optimize-histgb-v32}" \
  --reference-xgboost "${4:-20261029-125052-optimize-xgboost-v32}" \
  --reference-extratrees "${5:-20261029-125052-optimize-extratrees-default}"
