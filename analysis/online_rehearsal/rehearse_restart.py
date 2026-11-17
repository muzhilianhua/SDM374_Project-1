"""Synthetic replay for V4 and the conservative restart variant.

This harness is intentionally separate from the formal submission runner.  It
checks trajectory mechanics and compares variants without consuming evaluator
calls.  The synthetic function is a deterministic additive stand-in with
plateaus; it is evidence for regressions, not a score prediction for solver.exe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
POLICY_DIR = REPO / "submission" / "policy"
sys.path.insert(0, str(POLICY_DIR))

from policy import State, norm_blocks  # noqa: E402
from policy_v4 import BlockDescentPolicyV4, ParamsV4  # noqa: E402
from policy_v4_restart import BlockDescentPolicyV4Restart, ParamsV4Restart  # noqa: E402


def load_training():
    csv = REPO / "dataset_same8" / "train.csv"
    t = pd.read_csv(csv, float_precision="round_trip", dtype={"candidate_id": str})
    t = t.sort_values("candidate_id", kind="stable").reset_index(drop=True)
    return t.iloc[:, 1:22].to_numpy(float), t["ramping"].to_numpy(float)


def synthetic(W):
    W = norm_blocks(W)
    out = []
    for w in W:
        total = 120.0
        for b, (rs, c) in enumerate([(0.27, 1.0), (0.24, 0.4), (0.25, 0.7)]):
            s = w[b, 4] + w[b, 5]
            r = w[b, 4] / s if s > 0 else 0.5
            minors = w[b, [0, 1, 2, 3, 6]] / max(s, 1e-9)
            total += c * (30 * (r - rs) ** 2 + 4 * np.log10(1 + 20 * minors.sum()) - 3 * s)
        h = int(hashlib.md5(np.round(w, 4).tobytes()).hexdigest()[:8], 16) / 2**32
        out.append(total + 0.5 * h)
    return np.asarray(out)


def run(variant, seed, calls, local_rows):
    X, y = load_training()
    if variant == "v4":
        policy = BlockDescentPolicyV4(ParamsV4(shift_bias=True, n_nav=3,
                                               local_rows=local_rows), seed=seed)
    else:
        policy = BlockDescentPolicyV4Restart(ParamsV4Restart(shift_bias=True, n_nav=3,
                                                              local_rows=local_rows), seed=seed)
    st = State(X, y)
    sent = set()
    curve = []
    dropped = []
    restart_points = []
    for call_no in range(1, calls + 1):
        props, info = policy.propose(st, 50)
        batch, metas = [], []
        for W, meta in props:
            flat = norm_blocks(W)[0]
            key = tuple(np.round(flat.reshape(-1), 12))
            if key in sent:
                continue
            sent.add(key)
            batch.append(flat)
            metas.append(meta)
        if len(batch) != 50:
            raise AssertionError(f"{variant} seed {seed} call {call_no}: {len(batch)} points")
        R = synthetic(np.asarray(batch))
        st.calls.append([{"W": W, "R": float(r), "meta": dict(m)}
                         for W, r, m in zip(batch, R, metas)])
        curve.append(float(min(np.min(R), *(x for x in curve)))) if curve else curve.append(float(np.min(R)))
        restart_points.append(int(info.get("restart_points", 0)))
        dropped.append(50 - len(batch))
    return {"variant": variant, "seed": seed, "curve": curve,
            "final": curve[-1], "restart_points": restart_points,
            "dropped": dropped}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 7, 11, 2026])
    ap.add_argument("--calls", type=int, default=20)
    ap.add_argument("--local-rows", type=int, default=600)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    rows = [run(v, s, args.calls, args.local_rows) for v in ("v4", "restart") for s in args.seeds]
    for row in rows:
        print(f"{row['variant']:7s} seed={row['seed']:4d} final={row['final']:.5f} "
              f"c5={row['curve'][min(4, args.calls - 1)]:.5f} "
              f"c10={row['curve'][min(9, args.calls - 1)]:.5f} "
              f"restarts={sum(row['restart_points'])}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

