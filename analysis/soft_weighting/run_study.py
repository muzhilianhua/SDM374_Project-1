"""Launch the prespecified weighting comparison through the sdm CLI, preserving every run ID."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from sdm.paths import REPO_ROOT


def case(family, strategy, out):
    label = f"{family}-{strategy}"
    config = REPO_ROOT / "experiments" / f"soft-{label}-2026.toml"
    result = {"family": family, "strategy": strategy, "optimize": []}
    cli = str(Path(sys.executable).with_name("sdm"))

    def call(kind, config, *extra):
        start = time.monotonic()
        process = subprocess.run([cli, kind, "-c", str(config), *extra], cwd=REPO_ROOT,
                                 text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        suffix = "-2027" if "2027.toml" in str(config) else "-2026"
        (out / f"{label}-{kind}{suffix}.log").write_text(process.stdout)
        if process.returncode:
            raise RuntimeError(f"{label} {kind} failed: {process.stdout[-2000:]}")
        run_id = Path(re.search(r"^run: (.+)$", process.stdout, re.M)[1]).name
        print(f"{label}: {kind} {run_id} ({time.monotonic()-start:.1f}s)", flush=True)
        return run_id

    result["evaluate"] = call("evaluate", config)
    result["train"] = call("train", config)
    for seed in (2026, 2027):
        config = REPO_ROOT / "experiments" / f"soft-{label}-{seed}.toml"
        result["optimize"].append(call("optimize", config, "-n", "50", "--model-run", result["train"]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    completed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(case, family, strategy, args.out)
                   for family in ("histgb", "xgboost")
                   for strategy in ("unweighted", "step", "trimmed-step", "soft")]
        for future in as_completed(futures):
            completed.append(future.result())
            completed.sort(key=lambda x: (x["family"], x["strategy"]))
            (args.out / "run_manifest.json").write_text(json.dumps(completed, indent=2) + "\n")


if __name__ == "__main__":
    main()
