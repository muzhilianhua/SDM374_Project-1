"""Offline evaluation protocol shared by every surrogate: outer K-fold CV on the training table.

Model-internal scores (such as ridge2's alpha-selection CV) are tuned on the very folds they report,
so compare models with these numbers instead.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold

from .interfaces import Surrogate


def cross_validate(build: Callable[[], Surrogate], X: np.ndarray, y: np.ndarray, *,
                   folds: int = 5, seed: int = 2026, top_k: int = 50) -> dict[str, float]:
    mean, std = out_of_fold(build, X, y, folds=folds, seed=seed)
    return score(y, mean, std, top_k=top_k)


def out_of_fold(build: Callable[[], Surrogate], X: np.ndarray, y: np.ndarray, *,
                folds: int = 5, seed: int = 2026) -> tuple[np.ndarray, np.ndarray]:
    """Predicted (mean, std) for every row from a model that never saw that row."""
    mean, std = np.empty(len(y)), np.empty(len(y))
    for fit_rows, held_out in KFold(n_splits=folds, shuffle=True, random_state=seed).split(X):
        mean[held_out], std[held_out] = build().fit(X[fit_rows], y[fit_rows]).predict(X[held_out])
    return mean, std


def score(y: np.ndarray, mean: np.ndarray, std: np.ndarray, *, top_k: int = 50) -> dict[str, float]:
    # Optimization only needs the low-ramping end ranked correctly, hence the rank metrics.
    true_top = set(np.argsort(y, kind="stable")[:top_k])
    predicted_top = set(np.argsort(mean, kind="stable")[:top_k])
    return {
        "mae": float(mean_absolute_error(y, mean)),
        "rmse": float(np.sqrt(np.mean((mean - y) ** 2))),
        "r2": float(r2_score(y, mean)),
        "spearman": float(spearmanr(y, mean).correlation),
        f"top{top_k}_recall": len(true_top & predicted_top) / top_k,
        "mean_predicted_std": float(std.mean()),
        # Share of rows whose error lies within one predicted std; about 0.68 if std is calibrated.
        "std_coverage": float(np.mean(np.abs(mean - y) <= std)),
    }


def median_spacing(X: np.ndarray) -> float:
    """Median distance from each row of X to its nearest other row: the table's typical spacing."""
    squared = (X ** 2).sum(1)
    distances = squared[:, None] + squared[None, :] - 2 * X @ X.T
    np.fill_diagonal(distances, np.inf)
    return float(np.median(np.sqrt(np.maximum(distances, 0).min(axis=1))))


def nearest_distances(points: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Euclidean distance from each point to its nearest row of X; large values flag extrapolation."""
    squared = (points ** 2).sum(1)[:, None] + (X ** 2).sum(1)[None, :] - 2 * points @ X.T
    return np.sqrt(np.maximum(squared, 0).min(axis=1))
