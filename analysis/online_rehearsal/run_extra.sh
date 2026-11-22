#!/usr/bin/env bash
# Report experiment (2026-11-22): extra seeds for v2 and v4, and the v4j block-structure ablation, on the real
# solver.exe through the T37 rehearsal harness. Instructions: report/pending-experiments-2.md.
#
#   bash analysis/online_rehearsal/run_extra.sh            # from the repository root, in Git Bash
#   PAR=3 WORKERS=6 bash analysis/online_rehearsal/run_extra.sh
#
# PAR runs go at once, each with WORKERS solver processes; neither changes a run's batches. A finished run
# (its console.log ends with "final best") is skipped, and an interrupted one resumes after its last full call,
# so the script can simply be started again.
set -u
cd "$(dirname "$0")/../.."
PAR=${PAR:-3}
WORKERS=${WORKERS:-6}
# Order matters only if time runs out: the first nine give 3 v4j seeds and 6 seeds each for v2 and v4.
JOBS="v4j:2026 v2:1 v4:1 v4j:7 v2:2 v4:2 v4j:11 v2:3 v4:3 v4j:1 v4j:2 v4j:3"

run_one() {
  local v=${1%%:*} s=${1##*:}
  local name=${v}_s${s}
  local dir=analysis/online_rehearsal/rehearsals/$name
  if grep -q '^final best' "$dir/console.log" 2>/dev/null; then
    echo "skip  $name (finished)"
    return 0
  fi
  mkdir -p "$dir"
  echo "start $name $(date '+%m-%d %H:%M')"
  uv run python analysis/online_rehearsal/rehearse.py --evaluator solver --variant "$v" --seed "$s" \
    --run "$name" --workers "$WORKERS" >> "$dir/console.log" 2>&1
  echo "end   $name rc=$? $(date '+%m-%d %H:%M') | $(tail -n 1 "$dir/console.log")"
}
export -f run_one
export WORKERS

printf '%s\n' $JOBS | xargs -P "$PAR" -I{} bash -c 'run_one {}'
