#!/bin/sh
# Run from repository root. Requires original ignored reference run directories:
# 20261029-160333-evaluate-soft-xgboost-step-2026
# 20261029-125052-optimize-{histgb-v32,xgboost-v32,extratrees-default}
# Fresh clones must recreate these and update BASE_EVAL/REFS in summarize.py.
set -eu
out=${1:-analysis/method_expansion/results/recheck-methods}
if [ -e "$out" ]; then echo "Output already exists: $out" >&2; exit 1; fi
mkdir -p "$out"
run() { env -u PYTHONPATH uv run --locked --group gui "$@"; }
run python analysis/method_expansion/run_study.py discover --out "$out/discovery" --workers 3
run python analysis/method_expansion/summarize.py --manifest "$out/discovery/run_manifest.json" --out "$out/screen"
run python analysis/method_expansion/run_study.py audit --manifest "$out/discovery/run_manifest.json" --selection "$out/screen/selection.json" --out "$out/audit" --workers 2
run python analysis/method_expansion/summarize.py --manifest "$out/discovery/run_manifest.json" --selection "$out/screen/selection.json" --audit "$out/audit/audit_manifest.json" --out "$out/final"
run python analysis/method_expansion/plot_results.py --summary "$out/final"
