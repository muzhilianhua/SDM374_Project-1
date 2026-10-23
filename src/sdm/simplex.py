"""Weight layout and the per-building simplex constraint (7 non-negative weights summing to 1)."""
from __future__ import annotations

import numpy as np

METRICS = ("cost", "carbon", "peak", "discomfort", "degradation", "ramping", "resilience")
N_BUILDINGS = 3
N_METRICS = len(METRICS)
DIM = N_BUILDINGS * N_METRICS
FEATURES = tuple(f"b{building}_{metric}" for building in range(1, N_BUILDINGS + 1) for metric in METRICS)
SUM_TOLERANCE = 1e-9


def as_batch(X) -> np.ndarray:
    """Return an (n, 21) float array from one weight set ((21,) or (3, 7)) or a batch of them."""
    X = np.asarray(X, dtype=float)
    if X.shape in ((DIM,), (N_BUILDINGS, N_METRICS)):
        return X.reshape(1, DIM)
    if X.ndim == 3 and X.shape[1:] == (N_BUILDINGS, N_METRICS):
        return X.reshape(-1, DIM)
    if X.ndim != 2 or X.shape[1] != DIM:
        raise ValueError(f"expected weights of shape (21,), (3, 7), (n, 21) or (n, 3, 7); got {X.shape}")
    return X


def check_weights(X) -> np.ndarray:
    """Validate the constraint and return the batch; raises ValueError on the first violation."""
    X = as_batch(X)
    if not np.isfinite(X).all():
        raise ValueError("weights must be finite")
    if (X < 0).any():
        raise ValueError("weights must be non-negative")
    sums = X.reshape(-1, N_BUILDINGS, N_METRICS).sum(axis=2)
    if not np.allclose(sums, 1, atol=SUM_TOLERANCE, rtol=0):
        raise ValueError("each building's 7 weights must sum to 1")
    return X


def project(X) -> np.ndarray:
    """Euclidean projection of each building's 7 weights onto the simplex; keeps the input shape."""
    X = np.asarray(X, dtype=float)
    if X.shape[-1] != DIM and X.shape[-2:] != (N_BUILDINGS, N_METRICS):
        raise ValueError(f"expected 21 weights per row; got shape {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError("weights must be finite")
    rows = X.reshape(-1, N_METRICS)
    ordered = np.sort(rows, axis=1)[:, ::-1]
    cumulative = np.cumsum(ordered, axis=1) - 1
    counts = np.arange(1, N_METRICS + 1)
    last_positive = np.sum(ordered - cumulative / counts > 0, axis=1) - 1
    theta = cumulative[np.arange(len(rows)), last_positive] / (last_positive + 1)
    return np.maximum(rows - theta[:, None], 0).reshape(X.shape)


def sample(n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw n weight sets uniformly from the product of three simplices."""
    return rng.dirichlet(np.ones(N_METRICS), size=(n, N_BUILDINGS)).reshape(n, DIM)
