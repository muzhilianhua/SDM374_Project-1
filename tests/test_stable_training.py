"""Behavioral checks for honest calibration and neighborhood-aware selection."""
import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from sdm.models.stable_stack import checked_blend

from sdm.registry import build_model, build_optimizer
from sdm.simplex import check_weights
from sdm.optimizers._neighborhood import exact_neighbors, neighborhood_scores, shared_directions


def test_calibration_prediction_excludes_own_label(dataset):
    X, y = dataset.X[:60], dataset.y[:60].copy()
    params = {"iterations": 12, "bags": 2, "inner_folds": 2, "calib_folds": 3}
    original = build_model("stable-stack", params).fit(X, y)
    y[7] += 1000
    modified = build_model("stable-stack", params).fit(X, y)
    # Its own label may change a calibration residual, but cannot change its predictive fit.
    assert original.calibration_prediction_[7] == modified.calibration_prediction_[7]
    assert not np.allclose(original.predict(X)[0], modified.predict(X)[0])


def test_fit_scheduling_does_not_change_estimator(dataset):
    X, y = dataset.X[:60], dataset.y[:60]
    params = {"iterations": 8, "bags": 2, "inner_folds": 2, "calib_folds": 3}
    serial = build_model("stable-stack", {**params, "fit_workers": 1}).fit(X, y)
    parallel = build_model("stable-stack", {**params, "fit_workers": 3}).fit(X, y)
    for before, after in zip(serial.predict(X), parallel.predict(X)):
        np.testing.assert_array_equal(before, after)


def test_honest_stack_preserves_mean_and_excludes_calibration_label(dataset):
    X, y = dataset.X[:60], dataset.y[:60].copy()
    original = build_model("method-study", {"family": "ensemble", "ensemble_mode": "stack",
                                            "iterations": 10}).fit(X, y)
    honest = build_model("honest-stack", {"iterations": 10}).fit(X, y)
    np.testing.assert_array_equal(original.predict(X)[0], honest.predict(X)[0])
    y[7] += 1000
    changed = build_model("honest-stack", {"iterations": 10}).fit(X, y)
    assert honest.calibration_prediction_[7] == changed.calibration_prediction_[7]


def test_neighborhood_distance_on_simplex_boundary():
    center = np.tile([1., 0, 0, 0, 0, 0, 0], 3).reshape(1, 21)
    points = exact_neighbors(center, shared_directions(71, 24), .05)
    check_weights(points.reshape(-1, 21))
    np.testing.assert_allclose(np.linalg.norm(points-center[:, None], axis=2), .05,
                               atol=1e-10, rtol=0)


def test_neighborhood_ranking_rejects_narrow_false_minimum():
    centers = np.full((2, 21), 1/7)
    centers[1, :2] += [.1, -.1]

    class Spike:
        def predict(self, X):
            narrow = np.linalg.norm(X-centers[0], axis=1) < .001
            broad = np.linalg.norm(X-centers[1], axis=1) < .08
            return np.where(narrow, 0., np.where(broad, 3., 10.)), np.zeros(len(X))

    score, _ = neighborhood_scores(Spike(), centers, radii=(.02, .05),
                                   directions=shared_directions(81, 32))
    assert score[1] < score[0]


def test_neighborhood_budget_is_enforced(dataset, ridge2):
    optimizer = build_optimizer("neighborhood-trust", {"radii": [.02, .05], "prediction_budget": 10})
    with pytest.raises(ValueError, match="budget"):
        optimizer.propose(ridge2, dataset.X, dataset.y)


def test_prediction_budget_counts_neighbors_and_final_diagnostics(dataset, ridge2):
    class Counted:
        count = 0

        def predict(self, X):
            self.count += len(X)
            return ridge2.predict(X)

    model = Counted()
    optimizer = build_optimizer("neighborhood-trust", {"centers": 2, "iterations": 3,
        "samples_per_center": 4, "screening": 8, "directions": 4,
        "radii": [.02, .05], "prediction_budget": 98})
    optimizer.propose(model, dataset.X, dataset.y, n=3)
    assert model.count <= optimizer.diagnostics['model_input_evaluations_upper_bound'] == 98


def test_feasible_slsqp_non_success_is_used_and_recorded():
    prior = np.array([.5, .3, .2])
    candidate = np.array([.2, .3, .5])
    result = OptimizeResult(x=candidate, success=False, nit=200, message="Iteration limit reached")
    weights, info = checked_blend(result, prior, lambda w: np.square(w-candidate).sum())
    np.testing.assert_array_equal(weights, candidate)
    assert info["accepted_iterate"] and not info["used_prior"]
    assert info["message"] == "Iteration limit reached" and info["iterations"] == 200


@pytest.mark.parametrize("candidate", [[np.nan, .3, .7], [-.1, .3, .8], [.2, .3, .4],
                                       [np.inf, 0., 0.], [.1, .2, .7]])
def test_unusable_or_worse_slsqp_iterate_falls_back(candidate):
    prior = np.array([.5, .3, .2])
    result = OptimizeResult(x=candidate, success=False, message="linesearch stopped")
    weights, info = checked_blend(result, prior, lambda w: np.square(w-prior).sum())
    np.testing.assert_array_equal(weights, prior)
    assert info["used_prior"]


def test_slsqp_failure_does_not_abort_fit_and_is_in_metrics(dataset, monkeypatch):
    import sdm.models.stable_stack as module

    def stop(fun, initial, **kwargs):
        return OptimizeResult(x=initial, success=False, nit=200, message="Iteration limit reached")

    monkeypatch.setattr(module, "minimize", stop)
    model = build_model("stable-stack", {"iterations": 3, "bags": 2,
        "inner_folds": 2, "calib_folds": 2}).fit(dataset.X[:40], dataset.y[:40])
    assert all(np.isfinite(v).all() for v in model.predict(dataset.X[:2]))
    assert model.metrics["blend_optimization"]["message"] == "Iteration limit reached"
    assert len(model.metrics["calibration_blend_optimizations"]) == 2


def test_nearly_tangent_direction_reaches_radius_after_long_bracket():
    center = np.tile([1., 0, 0, 0, 0, 0, 0], 3).reshape(1, 21)
    ray = np.tile([0., 1e-5, -1., -1., -1., -1., -1.], 3).reshape(1, 3, 7)
    ray -= ray.mean(axis=2, keepdims=True)
    ray = ray.reshape(1, 21)
    ray /= np.linalg.norm(ray)
    result = exact_neighbors(center, ray, .05)
    check_weights(result.reshape(-1, 21))
    np.testing.assert_allclose(np.linalg.norm(result-center[:, None], axis=2), .05,
                               atol=1e-9, rtol=0)


def test_radius_is_checked_after_last_allowed_bracket_expansion():
    center = np.full((1, 21), 1/7)
    ray = np.zeros((1, 21))
    ray[0, :2] = [1, -1]
    ray /= np.linalg.norm(ray) * 2**23.5
    result = exact_neighbors(center, ray, .05)
    np.testing.assert_allclose(np.linalg.norm(result-center[:, None], axis=2), .05,
                               atol=1e-10, rtol=0)


def test_unreachable_radius_still_fails():
    with pytest.raises(ValueError, match="cannot reach"):
        exact_neighbors(np.full((1, 21), 1/7), np.zeros((1, 21)), .05)
