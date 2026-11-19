# -*- coding: utf-8 -*-
"""Zero-solver verification for the v7 package.

    python verify_v7.py freeze   # replay calls 1-11 and compare with runs/pr25_repro (v4 reference)
    python verify_v7.py replay   # replay calls 12-20 from this package's own committed log
    python verify_v7.py audit    # rebuild the ten audit checks

All three commands are read-only with respect to the committed evidence: they never start
solver.exe and they write only into verify_out/, which is git-ignored. explore_v7.py stays
byte-identical to the runner that produced the real run, so its sha256 in
runs/explore_v7_s7/run_config.json keeps matching.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "policy"))

import explore_v7 as ex  # noqa: E402
from online_run import State, load_training  # noqa: E402

VERIFY = HERE / "verify_out"
LOG = HERE / "runs" / "explore_v7_s7" / "online_log.jsonl"
TOL = 1e-14


def read_calls(path: Path):
    """Return {call_no: [point records sorted by seq]} for one online log."""
    recs = [json.loads(x) for x in path.open(encoding="utf-8") if x.strip()]
    calls = {}
    for r in recs:
        if r.get("type") == "point":
            calls.setdefault(r["call"], []).append(r)
    return {c: sorted(rows, key=lambda r: r["seq"]) for c, rows in calls.items()}


def compare_batches(batch, logged):
    """Multiset comparison: every proposed row must appear once among the logged rows."""
    ref = np.array([e["W"] for e in logged])
    d = np.abs(batch[:, None, :] - ref[None, :, :]).max(axis=2)
    matched = int((d.min(axis=1) <= TOL).sum())
    return {"n": len(batch), "matched_rows": matched, "max_abs_diff": float(d.min(axis=1).max()),
            "pass": matched == len(batch) == len(ref)}


def write(name, payload):
    VERIFY.mkdir(parents=True, exist_ok=True)
    path = VERIFY / name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(HERE)}")


def cmd_freeze():
    """Replay the frozen phase with the reference run's own history and compare as a multiset."""
    X, y = load_training()
    ref = ex.load_reference()
    st = State(X, y)
    rows, ok = [], True
    for c in range(1, ex.FREEZE + 1):
        batch, _, _ = ex.propose(ex.make_policy(), st, X)
        row = {"call": c, **compare_batches(batch, ref[c - 1])}
        ok &= row["pass"]
        rows.append(row)
        print(f"  call {c:2d}: matched {row['matched_rows']}/{row['n']}  "
              f"max|dW|={row['max_abs_diff']:.2e}  {'PASS' if row['pass'] else 'FAIL'}", flush=True)
        st.calls.append([{"W": np.array(e["W"]).reshape(3, 7), "R": e["R"], "meta": e["meta"]}
                         for e in ref[c - 1]])
    write("freeze_check.json", {"freeze_calls": ex.FREEZE, "all_pass": ok,
                                "comparison": "multiset", "tolerance": TOL,
                                "solver_calls": 0, "per_call": rows})
    print(f"freeze: {'PASS' if ok else 'FAIL'} (no solver call)")
    return 0 if ok else 1


def cmd_replay():
    """Recompute calls 12-20 from the history recorded in this package's own log."""
    X, y = load_training()
    calls = read_calls(LOG)
    missing = [c for c in range(1, ex.CALLS + 1) if c not in calls]
    if missing:
        raise SystemExit(f"log has no points for calls {missing}")
    st = State(X, y)
    for c in range(1, ex.FREEZE + 1):
        st.calls.append([{"W": np.array(e["W"]).reshape(3, 7), "R": e["R"], "meta": e["meta"]}
                         for e in calls[c]])
    rows, ok = [], True
    for c in range(ex.FREEZE + 1, ex.CALLS + 1):
        batch, _, _ = ex.propose(ex.make_policy(), st, X)
        row = {"call": c, **compare_batches(batch, calls[c])}
        ok &= row["pass"]
        rows.append(row)
        print(f"  call {c:2d}: matched {row['matched_rows']}/{row['n']}  "
              f"max|dW|={row['max_abs_diff']:.2e}  {'PASS' if row['pass'] else 'FAIL'}", flush=True)
        st.calls.append([{"W": np.array(e["W"]).reshape(3, 7), "R": e["R"], "meta": e["meta"]}
                         for e in calls[c]])
    write("replay_check.json", {"replayed_calls": [ex.FREEZE + 1, ex.CALLS], "all_pass": ok,
                                "comparison": "multiset", "tolerance": TOL,
                                "solver_calls": 0, "per_call": rows})
    print(f"replay: {'PASS' if ok else 'FAIL'} (no solver call)")
    return 0 if ok else 1


def cmd_audit():
    """Run explore_v7's audit checks but write the report into verify_out/ with relative paths."""
    VERIFY.mkdir(parents=True, exist_ok=True)
    original = ex.AUDIT
    ex.AUDIT = VERIFY / "audit_report.json"
    try:
        report = ex.write_audit()
    finally:
        ex.AUDIT = original
    report["checks"]["9_freeze_and_solver"]["freeze_check"] = "runs/explore_v7_s7/freeze_check.json"
    (VERIFY / "audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(ex.canonical({"complete": report["complete_20_calls"],
                        "overall_pass": report["overall_pass"],
                        "report": "verify_out/audit_report.json"}))
    return 0 if report["overall_pass"] else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("freeze", "replay", "audit"))
    args = ap.parse_args()
    sys.exit({"freeze": cmd_freeze, "replay": cmd_replay, "audit": cmd_audit}[args.cmd]())


if __name__ == "__main__":
    main()
