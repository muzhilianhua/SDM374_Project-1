"""Check that the committed policy code reproduces the logged rehearsal batches (no solver calls).

For each chosen call c of a run, the policy gets the training rows and the logged results of calls 1..c-1,
proposes again, the same pre-submit screen runs, and the batch must equal the logged batch of call c
point by point. The evaluator is deterministic, so equal batches mean equal results.

    python check_replay.py v1_s2026 all
    python check_replay.py v2_s7 1 10 20
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from policy import BlockDescentPolicy, State, norm_blocks  # noqa: E402
from rehearse import Screen, load_training, variant_params  # noqa: E402


def main():
    run, calls = sys.argv[1], sys.argv[2:]
    folder = HERE / "rehearsals" / run
    cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    recs = [json.loads(l) for l in (folder / "online_log.jsonl").open(encoding="utf-8")]
    pts = [r for r in recs if r["type"] == "point"]
    n_calls = max(r["call"] for r in pts)
    todo = range(1, n_calls + 1) if calls == ["all"] else [int(c) for c in calls]
    X, y = load_training()
    variant = cfg.get("variant", "v1")      # the first run (v1, seed 2026) predates the variant field
    policy = BlockDescentPolicy(variant_params(variant), seed=cfg["seed"])
    screen = Screen(X)
    ok = True
    for c in todo:
        st = State(X, y)
        for k in range(1, c):
            st.calls.append([{"W": np.array(r["W"]).reshape(3, 7),
                              "R": float("nan") if r["R"] is None else r["R"], "meta": r["meta"]}
                             for r in pts if r["call"] == k])
        props, _ = policy.propose(st, 50)
        sent = np.array([e["W"].reshape(21) for cc in st.calls for e in cc]).reshape(-1, 21)
        batch = []
        for W, _meta in props:
            prev = np.vstack([sent] + [b.reshape(1, 21) for b in batch]) if batch else sent
            if screen.check(W, prev) is None:
                batch.append(norm_blocks(W)[0].reshape(21))
        logged = np.array([r["W"] for r in pts if r["call"] == c])
        same = len(batch) == len(logged) and np.max(np.abs(np.array(batch) - logged)) <= 1e-12
        ok &= bool(same)
        print(f"{run} call {c:2d}: {'identical' if same else 'DIFFERENT'} ({len(batch)} vs {len(logged)} points)",
              flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
