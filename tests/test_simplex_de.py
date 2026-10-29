"""DE must handle flat models, respect the simplex/trust region, and return reproducible new batches."""
import numpy as np
import pytest

from sdm.evaluate import median_spacing, nearest_distances
from sdm.registry import build_model, build_optimizer
from sdm.simplex import check_weights


def test_tree_search_is_feasible_ranked_distinct_and_deterministic(dataset):
    model = build_model("extratrees", {"n_estimators": 20}).fit(dataset.X, dataset.y)
    params = {"population": 48, "generations": 12, "risk": 2.0, "max_distance_ratio": 0.5}
    optimizer = build_optimizer("simplex-de", params)
    points = optimizer.propose(model, dataset.X, dataset.y, n=50)
    check_weights(points)
    assert points.shape == (50, 21)
    assert len(np.unique(np.round(points, 9), axis=0)) == 50
    distance = nearest_distances(points, dataset.X)
    assert np.all(distance <= 0.5 * median_spacing(dataset.X) + 1e-12)
    assert np.all(distance > 1e-6)
    mean, std = model.predict(points)
    assert np.all(np.diff(mean + 2 * std) >= -1e-9)
    again = build_optimizer("simplex-de", params).propose(model, dataset.X, dataset.y, n=50)
    np.testing.assert_array_equal(points, again)


def test_search_improves_a_known_objective_without_gradients(dataset):
    class LinearModel:
        def predict(self, points):
            return points[:, 0], np.zeros(len(points))

    optimizer = build_optimizer("simplex-de", {"population": 32, "generations": 10})
    points = optimizer.propose(LinearModel(), dataset.X, dataset.y, n=5)
    assert points[0, 0] < 0.01


@pytest.mark.parametrize("params", [{"population": 3}, {"risk": -1}, {"crossover": 2},
                                    {"max_distance_ratio": float("nan")}])
def test_invalid_search_parameters_fail(params, dataset, ridge2):
    with pytest.raises(ValueError, match="parameters"):
        build_optimizer("simplex-de", params).propose(ridge2, dataset.X, dataset.y)


def test_impossible_batch_fails_instead_of_repeating_points(dataset, ridge2):
    optimizer = build_optimizer("simplex-de", {"population": 4, "generations": 1})
    with pytest.raises(ValueError, match="distinct candidates"):
        optimizer.propose(ridge2, dataset.X, dataset.y, n=50)
