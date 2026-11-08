"""Score the submission-2 judges (additive ARD GP, local GP) on the frozen V3.3.2 protocol.

Uses the frozen release under analysis/v33_protocol/release_v2/ unchanged: same tasks, P1 holdout,
feedback stage and scorer as the boosted-tree recipes reported there. No optimizer, no simulator.

Usage (repository root): uv run python analysis/offline_submission2/v332_judges.py
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from importlib import metadata

import numpy as np
import pandas as pd

from sdm.config import REPO_ROOT
from sdm.registry import build_model
from sdm.runs import PACKAGES, git_state

RELEASE = REPO_ROOT / "analysis/v33_protocol/release_v2"
OUT = REPO_ROOT / "analysis/offline_submission2/delivery/v332"
JUDGES = {"ard": ("bayes-search", {"family": "gp", "kernel": "additive-ard", "lengths": [0.5],
                                   "noises": [0.03], "seed": 2026}),
          "local-gp": ("local-gp", {})}


def protocol():
    sys.path.insert(0, str(RELEASE))
    import protocol as frozen  # noqa: E402  (frozen release module, imported by path)
    return frozen


def rows_for(ids, data, p1, frozen):
    X = np.asarray([np.ravel(p1["weights"]) if cid == "P1" else data[cid]["x"] for cid in ids])
    y = np.asarray([frozen.P1_ACTUAL if cid == "P1" else data[cid]["y"] for cid in ids])
    return X, y


def fit_task(job):
    judge, feedback, index = job
    frozen = protocol()
    data, manifest, p1 = frozen.load_release()
    name, params = JUDGES[judge]
    if index is None:  # final fit on every allowed row; reported, never scored
        ids = sorted(data) + (["P1"] if feedback else [])
        model = build_model(name, params).fit(*rows_for(ids, data, p1, frozen))
        return judge, feedback, None, float(model.predict(np.asarray(p1["weights"]))[0][0])
    task = manifest["tasks"][index]
    ids = task["train_ids"] + (["P1"] if feedback else [])
    model = build_model(name, params).fit(*rows_for(ids, data, p1, frozen))
    test_ids = task["test_ids"] + ["P1"]
    X, y = rows_for(test_ids, data, p1, frozen)
    mean, std = model.predict(X)
    return judge, feedback, index, [{"split_id": task["split_id"], "candidate_id": cid, "actual": float(a),
                                      "predicted_mean": float(m), "predicted_std": float(s)}
                                     for cid, a, m, s in zip(test_ids, y, mean, std)]


def main():
    frozen = protocol()
    data, manifest, _ = frozen.load_release()
    jobs = [(judge, feedback, index) for judge in JUDGES for feedback in (False, True)
            for index in [*range(len(manifest["tasks"])), None]]
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(variable, "1")  # inherited by the spawned workers
    with ProcessPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(fit_task, jobs))

    summary = []
    for judge in JUDGES:
        for feedback in (False, True):
            stage = "feedback" if feedback else "primary"
            mine = [r for r in results if r[0] == judge and r[1] == feedback]
            predictions = [row for r in sorted((r for r in mine if r[2] is not None), key=lambda r: r[2])
                           for row in r[3]]
            final_p1 = next(r[3] for r in mine if r[2] is None)
            metrics, scores = frozen.evaluate_rows(predictions, manifest, data, feedback)
            out = OUT / judge / stage
            out.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(predictions).to_csv(out / "predictions.csv.gz", index=False)
            pd.DataFrame(scores).to_csv(out / "task_scores.csv", index=False)
            (out / "metrics.json").write_text(json.dumps({**metrics, "full_fit_p1_prediction": final_p1},
                                                         indent=2) + "\n")
            task_frame = pd.DataFrame(scores)
            stress = task_frame[task_frame.evaluation_mode == "breakthrough"]
            summary.append({"model": judge, "stage": stage, "score": metrics["score"],
                            "maximum_score": metrics["maximum_score"],
                            **{k: metrics[k] for k in metrics if k.startswith("points_")},
                            **{f"{f}_score": metrics["families"][f]["score"] for f in metrics["families"]},
                            "regret": metrics["regret"], "true_low_mae": metrics["true_low_mae"],
                            "mandatory_hidden_mae": metrics["mandatory_hidden_mae"],
                            "stress_breakthroughs": int(stress.actual_breakthrough.sum()),
                            "stress_false_breakthroughs": int(stress.false_breakthrough.sum()),
                            "stress_tasks": len(stress),
                            "task_p1_prediction_mean": metrics["p1_prediction"],
                            "full_fit_p1_prediction": final_p1})
    boosted = pd.read_csv(RELEASE / "delivery/summary.csv")
    for row in boosted.itertuples():
        scores = pd.read_csv(RELEASE / "delivery" / row.model / row.stage / "task_scores.csv")
        stress = scores[scores.evaluation_mode == "breakthrough"]
        full = json.loads((RELEASE / "delivery" / row.model / row.stage / "full_fit.json").read_text())
        summary.append({"model": row.model, "stage": row.stage, "score": row.score,
                        "maximum_score": row.maximum_score,
                        **{c: getattr(row, c) for c in boosted.columns if c.startswith("points_")},
                        "hidden_elite_score": row.hidden_elite_score, "local_region_score": row.local_region_score,
                        "supported_low_score": row.supported_low_score, "regret": row.regret,
                        "true_low_mae": row.true_low_mae, "mandatory_hidden_mae": row.mandatory_hidden_mae,
                        "stress_breakthroughs": int(stress.actual_breakthrough.astype(str).eq("True").sum()),
                        "stress_false_breakthroughs": int(stress.false_breakthrough.astype(str).eq("True").sum()),
                        "stress_tasks": len(stress), "task_p1_prediction_mean": row.p1_prediction,
                        "full_fit_p1_prediction": full["p1_prediction"]})
    table = pd.DataFrame(summary)
    table.to_csv(OUT / "summary.csv", index=False)
    provenance = {"created": datetime.now().isoformat(timespec="seconds"), "git": git_state(),
                  "packages": {package: metadata.version(package) for package in PACKAGES},
                  "protocol_id": frozen.PROTOCOL,
                  "release_checksums_sha256": frozen.sha256(RELEASE / "CHECKSUMS.json"),
                  "judges": {k: {"model": m, "params": p} for k, (m, p) in JUDGES.items()},
                  "fits": len(jobs), "optimizer_used": False, "simulator_calls": 0,
                  "boosted_rows": "copied from analysis/v33_protocol/release_v2/delivery/summary.csv"}
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    pd.set_option("display.width", 250)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
