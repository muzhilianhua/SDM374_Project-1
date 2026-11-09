"""Smoke test of the course evaluator `solver/solver.exe` (Windows only).

Standard library only, so it runs on a bare Windows Python (3.8+), including the course computer.
It calls the exe directly through a stdin pipe (no PowerShell), deletes `usage.count` before every
call so the 20-call development quota never blocks it, and rewrites the JSON report after every
check so a crash or a timeout still leaves partial evidence.

    python analysis/solver_smoke/smoke.py            # all checks
    python analysis/solver_smoke/smoke.py --quick    # skip the 10/50-set timing batches
    python analysis/solver_smoke/smoke.py --quota    # also probe what happens after 20 calls

Expected values: README example (all-ones) 114.4317428462673; baseline W* 119.0916 (TA, 4 decimals);
training row p01969 115.76481925917324; offline submission 1 119.71206678869203.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import platform
import random
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOLVER_DIR = ROOT / "solver"
EXE = SOLVER_DIR / "solver.exe"
MAX_BATCH = 50
ERROR_EXIT_CODE = 2

KNOWN = {
    "readme_all_ones": 114.4317428462673,
    "baseline_w_star": 119.0916,
    "train_p01969": 115.76481925917324,
    "submission_1": 119.71206678869203,
}
# Baseline W* is only known to 4 decimals; the others are full-precision measurements.
TOLERANCE = {"baseline_w_star": 6e-5}
DEFAULT_TOLERANCE = 1e-6


def flat(weights) -> list[float]:
    """Flatten a 3x7 matrix or a 21-vector into 21 floats in building-major order."""
    values = [float(v) for row in weights for v in row] if isinstance(weights[0], list) else [float(v) for v in weights]
    if len(values) != 21:
        raise ValueError(f"expected 21 weights, got {len(values)}")
    return values


def load_points() -> dict[str, list[float]]:
    with (ROOT / "dataset_same8" / "train.csv").open(newline="") as handle:
        row = next(r for r in csv.DictReader(handle) if r["candidate_id"] == "p01969")
    features = [f"b{b}_{m}" for b in (1, 2, 3)
                for m in ("cost", "carbon", "peak", "discomfort", "degradation", "ramping", "resilience")]
    return {
        "baseline_w_star": flat(json.loads((ROOT / "weights.json").read_text())),
        "train_p01969": [float(row[f]) for f in features],
        "submission_1": flat(json.loads((ROOT / "SUBMISSION_READY/01_FIRST_REAL_TEST/weights.json").read_text())),
    }


def random_simplex_points(n: int, seed: int) -> list[list[float]]:
    """Deterministic points with each building's 7 weights on the simplex (exponential trick)."""
    rng = random.Random(seed)
    points = []
    for _ in range(n):
        point = []
        for _ in range(3):
            draws = [rng.expovariate(1.0) for _ in range(7)]
            total = sum(draws)
            point.extend(d / total for d in draws)
        points.append(point)
    return points


def usage_files() -> dict[str, str | None]:
    found = {}
    for path in {SOLVER_DIR / "usage.count", Path.cwd() / "usage.count"}:
        found[str(path)] = path.read_text(errors="replace").strip() if path.exists() else None
    return found


def reset_usage() -> None:
    for path in (SOLVER_DIR / "usage.count", Path.cwd() / "usage.count"):
        if path.exists():
            path.unlink()


def call(payload, *, reset: bool = True, timeout: float = 7200.0) -> dict:
    """One evaluator call. Returns exit code, parsed R list (if any), stderr and wall time."""
    if reset:
        reset_usage()
    text = payload if isinstance(payload, str) else json.dumps(payload)
    start = time.perf_counter()
    proc = subprocess.run([str(EXE)], check=False, input=text.encode("utf-8"), capture_output=True,
                          cwd=str(SOLVER_DIR), timeout=timeout)
    seconds = time.perf_counter() - start
    stdout = proc.stdout.decode("utf-8", errors="replace").strip()
    result = {"exit_code": proc.returncode, "seconds": round(seconds, 3),
              "stderr": proc.stderr.decode("utf-8", errors="replace").strip()[-2000:]}
    try:
        result["R"] = json.loads(stdout)["R"]
    except (ValueError, KeyError, TypeError):
        result["stdout"] = stdout[-2000:]
    return result


def git_state() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], check=False, cwd=str(ROOT), capture_output=True, text=True,
                                  timeout=30).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain"))}


class Report:
    def __init__(self, path: Path):
        self.path = path
        self.data = {"started": dt.datetime.now().isoformat(timespec="seconds"), "checks": {}}

    def add(self, name: str, passed, **details) -> None:
        self.data["checks"][name] = {"passed": passed, **details}
        mark = {True: "PASS", False: "FAIL", None: "INFO"}[passed]
        print(f"[{mark}] {name}: " + json.dumps({k: v for k, v in details.items() if k != "calls"})[:400],
              flush=True)
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2) + "\n", encoding="utf-8")


def close(name: str, value: float) -> bool:
    return abs(value - KNOWN[name]) <= TOLERANCE.get(name, DEFAULT_TOLERANCE)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="skip the 10- and 50-set timing batches")
    parser.add_argument("--quota", action="store_true", help="probe the 20-call quota with 21 single-set calls")
    parser.add_argument("--exe", type=Path, help="evaluator to test (default solver/solver.exe); its folder is the cwd")
    parser.add_argument("--out", type=Path, help="report path (default analysis/solver_smoke/reports/<host>-<time>.json)")
    args = parser.parse_args()
    global EXE, SOLVER_DIR
    if args.exe:
        EXE = args.exe.resolve()
        SOLVER_DIR = EXE.parent

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = args.out or ROOT / "analysis" / "solver_smoke" / "reports" / f"{socket.gethostname()}-{stamp}.json"
    report = Report(out)
    print(f"report: {out}")

    expected_sha = (ROOT / "solver" / "CHECKSUMS.sha256").read_text().split()[0]
    actual_sha = hashlib.sha256(EXE.read_bytes()).hexdigest() if EXE.exists() else None
    report.data["environment"] = {
        "host": socket.gethostname(), "platform": platform.platform(), "machine": platform.machine(),
        "processor": platform.processor(), "cpu_count": os.cpu_count(), "python": sys.version,
        "exe": str(EXE), "git": git_state(), "usage_count_before": usage_files(),
    }
    report.add("exe_sha256", actual_sha == expected_sha, expected=expected_sha, actual=actual_sha)
    if actual_sha is None:
        print("solver/solver.exe is missing: copy it from the course solver.zip")
        return 1

    points = load_points()
    names = list(points)

    # 1. README example: two all-ones sets in one call.
    r = call({"weights": [[1.0] * 21, [1.0] * 21]})
    ok = r["exit_code"] == 0 and len(r.get("R", [])) == 2 and all(close("readme_all_ones", v) for v in r["R"])
    report.add("readme_example", ok, expected=KNOWN["readme_all_ones"], usage_after=usage_files(), **r)

    # 2. Known measured points in one batch.
    r = call({"weights": [points[n] for n in names]})
    known = dict(zip(names, r.get("R", [])))
    ok = r["exit_code"] == 0 and len(known) == len(names) and all(close(n, v) for n, v in known.items())
    report.add("known_points", ok, R=known, expected={n: KNOWN[n] for n in names},
               diff={n: v - KNOWN[n] for n, v in known.items()}, seconds=r["seconds"],
               exit_code=r["exit_code"], stderr=r["stderr"])

    # 3. Determinism and batch independence: same batch again, reversed order, each point alone.
    again = call({"weights": [points[n] for n in names]}).get("R")
    reverse = call({"weights": [points[n] for n in reversed(names)]}).get("R")
    alone = {n: (call({"weights": [points[n]]}).get("R") or [None])[0] for n in names}
    report.add("deterministic_repeat", again == r.get("R"), first=r.get("R"), repeat=again)
    report.add("batch_order_independent", reverse is not None and list(reversed(reverse)) == r.get("R"),
               reversed_call=reverse)
    report.add("batch_composition_independent", all(alone[n] == known.get(n) for n in names), alone=alone)

    # 4. Single-set input formats must agree.
    w = points["baseline_w_star"]
    matrix = [w[0:7], w[7:14], w[14:21]]
    formats = {"bare_21": w, "bare_3x7": matrix, "object_single": {"weights": matrix}}
    format_r = {k: call(v) for k, v in formats.items()}
    values = {k: (v.get("R") or [None])[0] for k, v in format_r.items()}
    report.add("single_set_formats", len(set(values.values())) == 1 and None not in values.values(),
               R=values, reference=known.get("baseline_w_star"))

    # 5. Renormalisation: scale each building by a different positive factor (informational).
    scaled = [v * f for v, f in zip(w, [3.0] * 7 + [0.5] * 7 + [10.0] * 7)]
    r_scaled = call({"weights": [scaled]})
    renormalises = r_scaled.get("R", [None])[0] == known.get("baseline_w_star")
    report.add("renormalises_per_building", None, equal_to_normalised=renormalises,
               R_scaled=r_scaled.get("R"), R_normalised=known.get("baseline_w_star"))
    r_course = call(json.loads((ROOT / "solver" / "weight.json").read_text()))
    report.add("course_weight_json", r_course["exit_code"] == 0, **r_course)

    # 6. Invalid inputs must fail with exit code 2 and no R.
    bad = {
        "batch_of_51": {"weights": random_simplex_points(MAX_BATCH + 1, seed=51)},
        "negative_weight": {"weights": [[-0.1] + w[1:]]},
        "building_all_zero": {"weights": [[0.0] * 7 + w[7:]]},
        "twenty_weights": {"weights": [w[:20]]},
        "not_json": "hello",
    }
    for name, payload in bad.items():
        res = call(payload)
        report.add(f"rejects_{name}", res["exit_code"] == ERROR_EXIT_CODE and "R" not in res, **res)

    # 7. Usage counter behaviour without resetting between calls.
    reset_usage()
    trail = []
    for _ in range(2):
        res = call({"weights": [points["train_p01969"]]}, reset=False)
        trail.append({"exit_code": res["exit_code"], "usage": usage_files()})
    call("hello", reset=False)
    trail.append({"after_invalid_call": usage_files()})
    report.add("usage_counter", None, trail=trail)

    # 8. Timing by batch size (distinct random simplex points).
    sizes = [1] if args.quick else [1, 10, MAX_BATCH]
    timing = {}
    for size in sizes:
        res = call({"weights": random_simplex_points(size, seed=1000 + size)})
        timing[size] = {"seconds": res["seconds"], "exit_code": res["exit_code"],
                        "n_R": len(res.get("R", [])), "min_R": min(res["R"]) if res.get("R") else None}
    report.add("timing", all(t["exit_code"] == 0 and t["n_R"] == s for s, t in timing.items()), by_batch_size=timing)

    # 9. Optional: what happens on call 21 without a reset.
    if args.quota:
        reset_usage()
        codes = []
        for i in range(21):
            res = call({"weights": [random_simplex_points(1, seed=5000 + i)[0]]}, reset=False)
            codes.append({"call": i + 1, "exit_code": res["exit_code"], "stderr": res["stderr"][-300:]})
        report.add("quota_after_20_calls", None, calls=codes, usage=usage_files())

    reset_usage()
    report.data["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    failed = [n for n, c in report.data["checks"].items() if c["passed"] is False]
    report.data["failed"] = failed
    report.save()
    print(f"\n{len(failed)} failed: {failed}" if failed else "\nall checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
