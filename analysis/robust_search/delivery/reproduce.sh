#!/bin/sh
set -eu
if [ "$#" -ne 2 ]; then
  echo 'Usage from repository root: sh analysis/robust_search/delivery/reproduce.sh NEW_SCREEN_DIR NEW_AUDIT_DIR' >&2
  exit 2
fi
env -u PYTHONPATH uv run --locked --group gui python analysis/robust_search/study.py screen \
  --runs runs/20261029-132648-optimize-consensus-* runs/20261029-132649-optimize-consensus-* \
  --committee 20261029-132622-train-consensus-r075-2026 --out "$1"
env -u PYTHONPATH uv run --locked --group gui python analysis/robust_search/study.py audit \
  --screen-dir "$1" --out "$2"
