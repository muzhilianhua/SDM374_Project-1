"""Evaluate weights or verify the reconstructed solver against saved real results."""
from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing
import sys
import time
from pathlib import Path

from runtime import HERE, Solver, environment_info, verify_assets
from solver_cli import parse_weights


def parse_candidates(text):
    """Accept the original single-candidate and 1..50-candidate batch shapes."""
    value = json.loads(text)
    if isinstance(value, dict):
        if set(value) != {"weights"}:
            raise ValueError("The input object must contain only the weights key")
        value = value["weights"]
    try:
        return [parse_weights(json.dumps(value))]
    except ValueError:
        if not isinstance(value, list) or not 1 <= len(value) <= 50:
            raise ValueError("Expected one weight matrix or a batch of 1..50 candidates") from None
        return [parse_weights(json.dumps(candidate)) for candidate in value]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--weights", type=Path, help="Single or batch JSON file (default: best_weights.json)")
    mode.add_argument("--stdin", action="store_true", help="Read single or batch JSON from stdin")
    mode.add_argument("--check", action="store_true", help="Verify assets and import paths without simulation")
    mode.add_argument("--verify-reference", action="store_true", help="Re-run all three saved real-evaluator cases")
    parser.add_argument("--workers", type=int, default=8, choices=range(1, 9), metavar="1..8")
    parser.add_argument("--report", type=Path, help="Write a detailed report to a new JSON file")
    args = parser.parse_args()
    started = time.perf_counter()
    try:
        if sys.version_info[:2] != (3, 11):
            raise RuntimeError("Use Python 3.11 via the locked reverse_analysis uv project")
        if args.report and args.report.exists():
            raise ValueError("Report already exists; choose a new path to preserve evidence")
        checked = verify_assets()
        with contextlib.redirect_stdout(sys.stderr):
            environment = environment_info()
            expected = None
            if args.check:
                report = dict(assets_verified=checked, environment=environment)
            else:
                if args.verify_reference:
                    original = json.loads((HERE / "evidence/continued_real_validation.json").read_text())
                    candidates = [case["weights"] for case in original["candidates"]]
                    expected = [case["R_real"] for case in original["candidates"]]
                else:
                    text = sys.stdin.read() if args.stdin else (args.weights or HERE / "best_weights.json").read_text(encoding="utf-8-sig")
                    candidates = parse_candidates(text)
                records = []
                with Solver(args.workers) as solver:
                    for weights in candidates:
                        result = solver.evaluate(weights)
                        records.append(dict(weights=weights, R=result["R"], weekly_R=result["weekly_R"]))
                report = dict(R=[record["R"] for record in records], candidates=records,
                              environment=environment, assets_verified=checked,
                              seconds=time.perf_counter() - started)
                if expected is not None:
                    errors = [abs(record["R"] - actual) for record, actual in zip(records, expected)]
                    report.update(expected_R_real=expected, abs_errors=errors,
                                  max_abs_error=max(errors), matches_reference=max(errors) <= 1e-10)
        encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.report:
            with args.report.open("x", encoding="utf-8") as stream:
                stream.write(encoded)
        if args.check or args.verify_reference:
            print(encoded, end="")
        else:
            print(json.dumps(dict(R=report["R"]), allow_nan=False))
        return 1 if report.get("matches_reference") is False else 0
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
