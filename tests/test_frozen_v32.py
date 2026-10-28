"""The frozen-v32 evaluate protocol: frozen splits are honoured, the approved scorer is used unchanged."""
import shutil

import numpy as np
import pandas as pd
import pytest

from sdm import catalog, frozen_v32
from sdm.cli import main
from sdm.config import load_config
from sdm.data import DataIntegrityError
from sdm.paths import REPO_ROOT
from sdm.runs import list_runs

V32_CONFIG = REPO_ROOT / "experiments" / "baseline-v32.toml"


@pytest.fixture(scope="module")
def splits(dataset):
    return frozen_v32.load_outer_splits(dataset)


def test_frozen_splits_partition_every_repeat(dataset, splits):
    assert [split.split_id for split in splits][:2] == ["r01_f01", "r01_f02"]
    rows = np.arange(len(dataset.y))
    for split in splits:
        assert np.array_equal(np.sort(np.concatenate([split.train, split.test])), rows)
    for repeat in (0, 1):
        tests = np.concatenate([split.test for split in splits if split.repeat == repeat])
        assert np.array_equal(np.sort(tests), rows)


def test_edited_release_files_are_rejected(dataset, tmp_path, monkeypatch):
    copy = tmp_path / "splits.zip"
    shutil.copyfile(frozen_v32.SPLIT_ZIP, copy)
    with open(copy, "ab") as file:
        file.write(b"\0")
    monkeypatch.setattr(frozen_v32, "SPLIT_ZIP", copy)
    with pytest.raises(DataIntegrityError, match="frozen release"):
        frozen_v32.load_outer_splits(dataset)

    scorer = tmp_path / "scorer.py"
    scorer.write_text(frozen_v32.SCORER.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    monkeypatch.setattr(frozen_v32, "SCORER", scorer)
    with pytest.raises(DataIntegrityError, match="frozen release"):
        frozen_v32.load_scorer()


def perfect_predictions(dataset, splits):
    frames = [pd.DataFrame({"split_id": split.split_id, "repeat": split.repeat,
                            "candidate_id": np.asarray(dataset.ids)[split.test], "actual": dataset.y[split.test],
                            "predicted_mean": dataset.y[split.test], "predicted_std": 1.0}) for split in splits]
    return pd.concat(frames, ignore_index=True)


def test_perfect_predictions_score_full_marks(dataset, splits):
    metrics, folds = frozen_v32.score_predictions(perfect_predictions(dataset, splits), dataset, splits)
    assert metrics["v32_score"] == pytest.approx(100.0)
    assert metrics["v32_top1_regret"] == 0.0 and metrics["mae"] == 0.0
    assert len(folds) == 10


def test_row_order_does_not_matter(dataset, splits):
    predictions = perfect_predictions(dataset, splits)
    predictions["predicted_mean"] += np.random.default_rng(0).normal(0, 5, len(predictions))
    shuffled = predictions.sample(frac=1, random_state=0)
    assert (frozen_v32.score_predictions(shuffled, dataset, splits)[0]
            == frozen_v32.score_predictions(predictions, dataset, splits)[0])


def _training_id(p, dataset, splits):
    p.loc[0, "candidate_id"] = dataset.ids[splits[0].train[0]]
    return p


def _relabel(p, dataset, splits):
    p.loc[0, ["actual", "predicted_mean"]] = 300.0
    return p


def _wrong_repeat(p, dataset, splits):
    p.loc[p.split_id == "r01_f01", "repeat"] = 1
    return p


def _missing_row(p, dataset, splits):
    return p.drop(index=0)


def _duplicate_row(p, dataset, splits):
    return pd.concat([p, p.iloc[[0]]], ignore_index=True)


def _missing_split(p, dataset, splits):
    return p[p.split_id != "r02_f05"]


def _unknown_split(p, dataset, splits):
    return pd.concat([p, p.iloc[[0]].assign(split_id="r09_f01")], ignore_index=True)


# Saved-prediction files that are wrongly joined or labelled must be rejected, not scored.
MALFORMED = {
    "test id replaced by a training id": (_training_id, "not in this test fold"),
    "label changed together with prediction": (_relabel, "actual values differ"),
    "wrong repeat": (_wrong_repeat, "repeat must be 0"),
    "missing test row": (_missing_row, "1 missing"),
    "duplicate test row": (_duplicate_row, "duplicate candidate_id"),
    "missing split": (_missing_split, r"missing \['r02_f05'\]"),
    "unknown split": (_unknown_split, r"unknown \['r09_f01'\]"),
}


@pytest.mark.parametrize("case", sorted(MALFORMED))
def test_malformed_predictions_are_rejected(dataset, splits, case):
    corrupt, message = MALFORMED[case]
    predictions = corrupt(perfect_predictions(dataset, splits), dataset, splits)
    with pytest.raises(ValueError, match=message):
        frozen_v32.score_predictions(predictions, dataset, splits)


def test_config_rejects_unknown_protocol_and_kfold_settings(tmp_path):
    assert load_config(V32_CONFIG).evaluate.protocol == "frozen-v32"
    path = tmp_path / "bad.toml"
    path.write_text('[model]\nname = "ridge2"\n[evaluate]\nprotocol = "holdout"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="protocol must be one of"):
        load_config(path)
    path.write_text('[model]\nname = "ridge2"\n[evaluate]\nprotocol = "frozen-v32"\nfolds = 3\n', encoding="utf-8")
    with pytest.raises(ValueError, match="remove folds and seed"):
        load_config(path)


def test_evaluate_run_records_v32_and_kfold_metrics(tmp_path):
    assert main(["--runs-dir", str(tmp_path), "evaluate", "-c", str(V32_CONFIG)]) == 0
    run = list_runs(tmp_path)[0]
    metrics = run.read_json("metrics.json")
    assert {"v32_score", "v32_points_local_order", "v32_top1_regret", "mae", "spearman", "top50_recall"} <= set(metrics)
    assert len(pd.read_csv(run.path / "predictions.csv")) == 2 * 2048
    assert len(pd.read_csv(run.path / "fold_scores.csv")) == 10
    assert main(["--runs-dir", str(tmp_path), "reproduce", run.id]) == 0


def test_leaderboard_ranks_protocols_separately():
    def row(run, protocol, **metrics):
        return {"run": run, "kind": "evaluate", "protocol": protocol, "complete": True, "metrics": metrics}
    records = pd.DataFrame([row("k-worse", "kfold", mae=7.0), row("v-low", "frozen-v32", v32_score=66.0, mae=6.9),
                            row("k-better", "kfold", mae=5.0), row("v-high", "frozen-v32", v32_score=68.0, mae=5.5)])
    assert list(catalog.leaderboard(records).run) == ["v-high", "v-low", "k-better", "k-worse"]
    only_kfold = records[records.protocol == "kfold"]
    assert list(catalog.leaderboard(only_kfold).run) == ["k-better", "k-worse"]
