"""Shared feasible directions and actual-radius neighborhood diagnostics."""
from __future__ import annotations

import numpy as np

from ..simplex import check_weights, project


def shared_directions(seed, count):
    raw = np.random.default_rng(seed).normal(size=(count, 3, 7))
    raw -= raw.mean(axis=2, keepdims=True)
    raw = raw.reshape(count, 21)
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)


def exact_neighbors(points, directions, radius):
    """Project rays, then solve for actual Euclidean distance (including boundary points)."""
    points = check_weights(points)
    directions = np.asarray(directions, dtype=float)
    if (radius <= 0 or not np.isfinite(radius) or directions.ndim != 2
            or directions.shape[1] != 21 or not np.isfinite(directions).all()):
        raise ValueError("positive finite radius and finite (n,21) directions required")
    centers = np.repeat(points, len(directions), axis=0)
    rays = np.tile(directions, (len(points), 1)).copy()
    stationary = np.linalg.norm(project(centers + radius*rays) - centers, axis=1) < 1e-12
    rays[stationary] *= -1
    lo, hi = np.zeros(len(centers)), np.full(len(centers), radius)
    for expansion in range(25):
        expand = np.linalg.norm(project(centers + hi[:, None]*rays) - centers, axis=1) < radius
        if not expand.any():
            break
        if expansion == 24:
            raise ValueError("direction cannot reach requested radius")
        hi[expand] *= 2
    # Keep the actual-radius contract even when bracketing requires a long ray.
    # Projection is 1-Lipschitz: half the remaining ray interval bounds distance error.
    ray_norm = np.linalg.norm(rays, axis=1)
    for step in range(80):
        mid = (lo + hi)/2
        small = np.linalg.norm(project(centers + mid[:, None]*rays) - centers, axis=1) < radius
        lo, hi = np.where(small, mid, lo), np.where(small, hi, mid)
        if step >= 41 and np.all(ray_norm * (hi-lo)/2 <= 1e-12):
            break
    result = check_weights(project(centers + ((lo+hi)/2)[:, None]*rays))
    error = np.abs(np.linalg.norm(result-centers, axis=1) - radius)
    roundoff = 32*np.finfo(float).eps*np.maximum(1., ray_norm*hi)
    tolerance = 1e-10 + ray_norm*(hi-lo)/2 + roundoff
    if np.any(error > tolerance):
        raise ValueError("actual neighborhood radius verification failed")
    return result.reshape(len(points), len(directions), 21)


def neighborhood_scores(model, points, *, radii, directions, risk=1., quantile=.8):
    """Upper-tail mean+error acquisition; never an estimate of true neighborhood labels."""
    points = check_weights(points)
    mean, scale = model.predict(points)
    central = mean + risk*scale
    columns = [central]
    for radius in radii:
        neighbors = exact_neighbors(points, directions, radius)
        m, s = model.predict(neighbors.reshape(-1, 21))
        columns.append(np.quantile((m+risk*s).reshape(len(points), -1), quantile, axis=1))
    return np.max(columns, axis=0), np.column_stack(columns)
