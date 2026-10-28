"""The `frozen-v32` evaluate protocol: the team's frozen 2x5 outer splits scored with the approved v3.2 scorer.

Both inputs are member B's frozen releases (`frozen_data/`, `evaluation_v3_2/`). They are read in place and
checked against the hashes below, so any edit to them stops the evaluation instead of changing the numbers.
Only the outer splits are used; the inner 3-fold lists are for plugins that tune on v3.2 themselves.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
from types import ModuleType
from typing import Callable
import zipfile

import numpy as np
import pandas as pd

from .data import DataIntegrityError, Dataset, sha256_of
from .evaluate import score
from .interfaces import Surrogate
from .paths import REPO_ROOT

PROTOCOL = "frozen-v32"
SPLIT_PROTOCOL_ID = "sdm374.same8.splits.2x5x3.frozen.v1"
SPLIT_ZIP = REPO_ROOT / "frozen_data" / "SDM374_data_splits_frozen_v1.zip"
SPLIT_ZIP_SHA256 = "91fbd29f05d64ff2470e8b8a15ade78b7f234b234bf32f03a67fb4ca5d15a09d"
SPLIT_MANIFEST = "SDM374_data_splits_frozen_v1/split_manifest.json"
SCORER = REPO_ROOT / "evaluation_v3_2" / "surrogate_scoring_v3_2.py"
SCORER_SHA256 = "ae2f17917cfaeb3efae615fbfd15bb5cfbdd92913de20773a3b6d594353c1aa8"
POLICY = REPO_ROOT / "evaluation_v3_2" / "EVALUATION_POLICY_V3_2.json"
POLICY_SHA256 = "0bfb32636c746335250a8d2ae4743c07250bbaa8c369297b05e5aceb46db5d59"
COMPONENTS = ("local_order", "local_mae", "global_order", "global_mae", "top1_regret")
DIAGNOSTICS = ("local_pair_accuracy", "global_pair_accuracy", "local_mae", "global_mae", "top1_regret")


@dataclass(frozen=True)
class OuterSplit:
    split_id: str
    repeat: int
    train: np.ndarray  # row positions in Dataset order
    test: np.ndarray


def _verify(path, expected: str) -> None:
    digest = sha256_of(path)
    if digest != expected:
        raise DataIntegrityError(f"{path} sha256 {digest} differs from the frozen release {expected}")


def load_outer_splits(dataset: Dataset) -> list[OuterSplit]:
    """The 10 frozen outer splits as row positions, after checking the release against `dataset`."""
    _verify(SPLIT_ZIP, SPLIT_ZIP_SHA256)
    with zipfile.ZipFile(SPLIT_ZIP) as archive:
        manifest = json.loads(archive.read(SPLIT_MANIFEST).decode("utf-8"))
    if manifest["protocol_id"] != SPLIT_PROTOCOL_ID or manifest["dataset"]["sha256"] != dataset.sha256:
        raise DataIntegrityError("frozen splits were made for a different protocol or dataset")
    if manifest["canonical_candidate_ids"] != sorted(dataset.ids):
        raise DataIntegrityError("frozen split ids do not match the dataset's candidate_id column")

    position = {candidate: row for row, candidate in enumerate(dataset.ids)}
    splits, tested = [], {}
    for entry in manifest["outer_splits"]:
        train = np.array([position[i] for i in entry["outer_train_ids"]])
        test = np.array([position[i] for i in entry["outer_test_ids"]])
        if np.intersect1d(train, test).size or len(train) + len(test) != len(dataset.ids):
            raise DataIntegrityError(f"{entry['split_id']}: outer train/test do not partition the dataset")
        tested.setdefault(entry["repeat_index"], []).append(test)
        splits.append(OuterSplit(entry["split_id"], entry["repeat_index"], train, test))
    for repeat, tests in tested.items():
        if not np.array_equal(np.sort(np.concatenate(tests)), np.arange(len(dataset.ids))):
            raise DataIntegrityError(f"repeat {repeat}: outer test folds do not cover every row exactly once")
    if len(splits) != 10 or len(tested) != 2:
        raise DataIntegrityError(f"expected 2 repeats x 5 outer folds, found {len(splits)} splits")
    return splits


def load_scorer() -> tuple[ModuleType, object]:
    """The approved scorer module, imported from its frozen file, and its validated policy."""
    _verify(SCORER, SCORER_SHA256)
    _verify(POLICY, POLICY_SHA256)
    spec = importlib.util.spec_from_file_location("surrogate_scoring_v3_2", SCORER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, module.load_final_policy(POLICY)


def out_of_fold(build: Callable[[], Surrogate], dataset: Dataset,
                splits: list[OuterSplit]) -> pd.DataFrame:
    """One prediction per (outer split, test row): every row is predicted twice, once per repeat."""
    X, y = dataset.X, dataset.y
    frames = []
    for split in splits:
        mean, std = build().fit(X[split.train], y[split.train]).predict(X[split.test])
        frames.append(pd.DataFrame({"split_id": split.split_id, "repeat": split.repeat,
                                    "candidate_id": np.asarray(dataset.ids)[split.test],
                                    "actual": y[split.test], "predicted_mean": mean, "predicted_std": std}))
    return pd.concat(frames, ignore_index=True)


def _aligned(part: pd.DataFrame, split: OuterSplit, dataset: Dataset) -> pd.DataFrame:
    """This split's predictions in frozen test order, after checking they are exactly its test rows.

    Predictions may come from saved files, so a wrong join or label must fail here instead of scoring.
    """
    expected = np.asarray(dataset.ids)[split.test]
    ids = part.candidate_id.astype(str)
    if ids.duplicated().any():
        raise ValueError(f"{split.split_id}: duplicate candidate_id rows, e.g. {ids[ids.duplicated()].iloc[0]}")
    missing, extra = set(expected) - set(ids), set(ids) - set(expected)
    if missing or extra:
        raise ValueError(f"{split.split_id}: predictions must cover exactly its {len(expected)} test rows; "
                         f"{len(missing)} missing, {len(extra)} not in this test fold")
    if not (part["repeat"] == split.repeat).all():
        raise ValueError(f"{split.split_id}: repeat must be {split.repeat} on every row")
    part = part.set_index(ids).loc[expected].reset_index(drop=True)
    if not np.allclose(part.actual.to_numpy(dtype=float), dataset.y[split.test], rtol=0, atol=1e-9):
        raise ValueError(f"{split.split_id}: actual values differ from the dataset's ramping for the same ids")
    return part


def score_predictions(predictions: pd.DataFrame, dataset: Dataset, splits: list[OuterSplit], *,
                      top_k: int = 50) -> tuple[dict[str, float], pd.DataFrame]:
    """(metrics, per-split scores): v3.2 averaged over the outer splits, plus the k-fold metrics per repeat.

    `predictions` needs one row per (split, test row): split_id, repeat, candidate_id, actual,
    predicted_mean, predicted_std. Missing, extra or mislabelled rows raise ValueError.
    """
    scorer, policy = load_scorer()
    expected_splits = {split.split_id for split in splits}
    found = set(predictions.split_id)
    if found != expected_splits:
        raise ValueError(f"predictions must cover exactly the frozen splits; missing {sorted(expected_splits - found)}, "
                         f"unknown {sorted(found - expected_splits)}")
    by_split = dict(tuple(predictions.groupby("split_id", sort=False)))
    aligned, rows = [], []
    for split in splits:
        part = _aligned(by_split[split.split_id], split, dataset)
        aligned.append(part)
        # The scale s comes from the labels the model was allowed to train on: this split's outer train.
        context = scorer.make_context(dataset.y[split.train], policy)
        result = scorer.evaluate(part.actual.to_numpy(), part.predicted_mean.to_numpy(), context, policy,
                                 part.candidate_id.to_numpy())
        rows.append({"split_id": split.split_id, "score": result["score"],
                     **{f"points_{key}": result["points"][key] for key in COMPONENTS},
                     **{key: result[key] for key in DIAGNOSTICS},
                     "top1_candidate_id": result["top1_candidate_id"]})
    folds = pd.DataFrame(rows)

    metrics = {"v32_score": float(folds.score.mean()),
               # Spread across reused folds is descriptive only; it is not a confidence interval.
               "v32_score_std": float(folds.score.std(ddof=0)),
               **{f"v32_points_{key}": float(folds[f"points_{key}"].mean()) for key in COMPONENTS},
               **{f"v32_{key}": float(folds[key].mean()) for key in DIAGNOSTICS}}
    # Each repeat predicts every row once, so the k-fold metrics are computed per repeat and averaged.
    legacy = [score(part.actual.to_numpy(), part.predicted_mean.to_numpy(), part.predicted_std.to_numpy(),
                    top_k=top_k) for _, part in pd.concat(aligned).groupby("repeat")]
    metrics.update({key: float(np.mean([entry[key] for entry in legacy])) for key in legacy[0]})
    return metrics, folds
