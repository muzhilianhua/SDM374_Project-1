"""Rescore the saved outer-test predictions of member B's release with the team's frozen-v32 protocol."""
from pathlib import Path

import pandas as pd

from sdm import frozen_v32
from sdm.data import load_dataset
from sdm.paths import REPO_ROOT

EVIDENCE = Path(__file__).resolve().parent / "evidence"
# (file, group column, prediction column) for the two saved prediction sets.
SOURCES = {
    "selection_11_groups": ("selection_11_groups/heldout_predictions.csv.gz", "group", "predicted"),
    "full_data_configs": ("full_data_configs/outer_predictions.csv.gz", "config", "y_pred"),
}


def rescore(source: str) -> dict[str, float]:
    """v32_score per group, recomputed from the saved predictions; nothing is fitted."""
    path, group_column, prediction_column = SOURCES[source]
    dataset = load_dataset(REPO_ROOT / "dataset_same8")
    splits = frozen_v32.load_outer_splits(dataset)
    table = pd.read_csv(EVIDENCE / path, dtype={"candidate_id": str}, encoding="utf-8-sig")
    table = table.rename(columns={prediction_column: "predicted_mean", "y_true": "actual"})
    table["predicted_std"] = 0.0  # the saved artifacts give point predictions only
    table["repeat"] = table.split_id.str[1:3].astype(int) - 1  # "r01_f03" -> repeat 0
    return {group: frozen_v32.score_predictions(part, dataset, splits)[0]["v32_score"]
            for group, part in table.groupby(group_column, sort=False)}


if __name__ == "__main__":
    for source in SOURCES:
        for group, score in rescore(source).items():
            print(f"{source:20s} {group:32s} {score:.4f}")
