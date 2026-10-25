"""Verify immutable packages and rescore supplied held-out predictions; never fit a model."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / "scoring"))
from surrogate_scoring_v3_2 import evaluate, load_final_policy, make_context


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf8")


def extract(archive, destination):
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if not (destination / name).resolve().is_relative_to(destination.resolve()):
                raise ValueError("Unsafe archive path")
        z.extractall(destination)


def main():
    out = HERE / "results" / (datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-release-verification")
    out.mkdir(parents=True)
    git = lambda *args: subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf8").strip()
    provenance = dict(analysis_id=out.name, created_utc=datetime.now(timezone.utc).isoformat(),
        commit=git("rev-parse", "HEAD"), branch=git("branch", "--show-current"),
        git_dirty=bool(git("status", "--porcelain")), python=platform.python_version(), numpy=np.__version__,
        source_sha256=sha(__file__), packages={p.name: sha(p) for p in sorted((HERE / "packages").glob("*.zip"))},
        fitting_performed=False, retuning_performed=False,
        evidence_origin="Provider's saved outer-test predictions; training membership declared by provider, not independently proven")
    save(out / "provenance.json", provenance)
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        extract(HERE / "packages/SDM374_data_splits_frozen_v1.zip", temp)
        extract(HERE / "packages/SDM374_final_models_v1.zip", temp)
        frozen = temp / "SDM374_data_splits_frozen_v1"
        spec = importlib.util.spec_from_file_location("frozen_split_api", frozen / "split_api.py")
        api = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(api)
        table, manifest = api.load_package(frozen)
        ids = np.array(table["candidate_ids"])
        y = np.array([r["ramping"] for r in table["records"]])
        splits = list(api.iter_nested_splits(manifest, table["candidate_ids"]))
        package = temp / "SDM374_final_models_v1"
        for line in (package / "checksums_sha256.txt").read_text("utf8").splitlines():
            if line.strip():
                digest, name = line.split("  ", 1)
                assert sha(package / name) == digest, name
        model_manifest = json.loads((package / "MANIFEST.json").read_text("utf8"))
        assert model_manifest["dataset_sha256"] == manifest["dataset"]["sha256"]
        assert model_manifest["feature_columns"] == api.WEIGHT_COLUMNS
        with (package / "evaluation/eval_predictions.csv").open(encoding="utf8", newline="") as f:
            predictions = list(csv.DictReader(f))
        assert len(predictions) == 3 * 4096
        policy = load_final_policy()
        folds = []
        summary = []
        for name in model_manifest["models"]:
            model_rows = []
            for split in splits:
                sid = split["split_id"]
                rows = [r for r in predictions if r["config"] == name and r["split_id"] == sid]
                row_ids = [r["candidate_id"] for r in rows]
                expected = ids[split["outer_test"]]
                assert len(rows) == len(expected) and len(set(row_ids)) == len(rows)
                assert set(row_ids) == set(expected)
                lookup = {r["candidate_id"]: r for r in rows}
                actual = np.array([float(lookup[i]["y_true"]) for i in expected])
                pred = np.array([float(lookup[i]["y_pred"]) for i in expected])
                np.testing.assert_allclose(actual, y[split["outer_test"]], rtol=0, atol=1e-12)
                result = evaluate(actual, pred, make_context(y[split["outer_train"]], policy), policy, expected)
                row = dict(model=name, split_id=sid, score=result["score"],
                    **{"p_" + k: v for k, v in result["points"].items()},
                    **{k: result[k] for k in ("local_mae", "global_mae", "local_bias", "global_bias", "top1_regret")})
                model_rows.append(row)
                folds.append(row)
            scores = np.array([r["score"] for r in model_rows])
            summary.append(dict(model=name, score=float(scores.mean()),
                score_sd_population=float(scores.std(ddof=0)), score_sd_sample=float(scores.std(ddof=1)),
                **{k: float(np.mean([r[k] for r in model_rows])) for k in model_rows[0] if k not in ("model", "split_id", "score")}))
        summary.sort(key=lambda r: r["score"], reverse=True)
        for rank, row in enumerate(summary, 1):
            row["rank"] = rank
        for filename, rows in (("full_config_scores.csv", summary), ("full_config_fold_scores.csv", folds)):
            with (out / filename).open("w", encoding="utf8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        save(out / "summary.json", dict(summary=summary, folds=folds))
        save(out / "validation.json", dict(status="passed", model_package_checksums=14,
            configurations=3, outer_folds=30, heldout_predictions=12288,
            exact_frozen_test_membership=True, training_membership="provider-declared",
            fitting_performed=False, retuning_performed=False))
        print(json.dumps(dict(output=str(out), summary=summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
