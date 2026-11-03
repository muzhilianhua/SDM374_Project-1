"""Budgeted trust search followed by shared-direction neighborhood-tail reranking."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..interfaces import Optimizer
from ..registry import register_optimizer
from ..simplex import check_weights
from ._neighborhood import neighborhood_scores, shared_directions
from .trust_dirichlet import TrustRegionDirichlet


@register_optimizer("neighborhood-trust")
class NeighborhoodTrust(Optimizer):
    """Trust search with optional actual-radius upper-tail reranking under an explicit prediction budget."""

    @dataclass(frozen=True)
    class Params:
        centers: int = 16
        iterations: int = 24
        samples_per_center: int = 25
        concentration: float = 200.
        growth: float = 1.15
        risk: float = 1.
        max_distance_ratio: float = .75
        min_separation_ratio: float = .25
        screening: int = 64
        directions: int = 16
        radii: tuple[float, ...] = ()
        quantile: float = .8
        prediction_budget: int = 12000
        seed: int = 2026

    def propose(self, model, X, y, n=1):
        p = self.params
        if (n < 1 or p.screening < n or p.directions < 1 or not 0 < p.quantile <= 1
                or p.centers < 1 or p.iterations < 1 or p.samples_per_center < 1
                or not np.isfinite((p.risk, p.max_distance_ratio, p.concentration,
                                   p.growth, p.min_separation_ratio, *p.radii)).all()
                or p.risk < 0 or p.max_distance_ratio <= 0 or p.concentration <= 0
                or p.growth <= 0 or p.min_separation_ratio <= 0
                or any(r <= 0 for r in p.radii)):
            raise ValueError("invalid neighborhood-trust parameters")
        requested = p.screening if p.radii else n
        upper_cost = p.centers*p.iterations*p.samples_per_center
        upper_cost += 1 + (1 + requested * (1 + len(p.radii)*p.directions) if p.radii else 0)
        if upper_cost > p.prediction_budget:
            raise ValueError(f"prediction budget exceeded: upper bound {upper_cost} > {p.prediction_budget}")
        values = {key: getattr(p, key) for key in
                  ("centers", "iterations", "samples_per_center", "concentration", "growth",
                   "risk", "max_distance_ratio", "min_separation_ratio", "seed")}
        base = TrustRegionDirichlet(TrustRegionDirichlet.Params(**values))
        pool = base.propose(model, X, y, n=requested)
        self.diagnostics = dict(base.diagnostics)
        if p.radii:
            scores, parts = neighborhood_scores(model, pool, radii=p.radii,
                directions=shared_directions(p.seed+4109, p.directions), risk=p.risk,
                quantile=p.quantile)
            order = np.argsort(scores, kind="stable")
            chosen = pool[order[:n]]
            selected_mean, selected_scale = model.predict(chosen[:1])
            self.diagnostics.update({"best_objective": float(scores[order[0]]),
                "best_mean": float(selected_mean[0]), "best_std": float(selected_scale[0]),
                "best_central_objective": float(parts[order[0], 0]),
                "screened_candidates": len(pool)})
        else:
            chosen = pool
        self.diagnostics["model_input_evaluations_upper_bound"] = upper_cost
        return check_weights(chosen)
