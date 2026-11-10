"""Tests for the deterministic online exploit/explore proposer and trust-radius state."""
import numpy as np
import pytest

from sdm.evaluate import nearest_distances
from sdm.registry import build_optimizer
from sdm.simplex import check_weights


def test_online_batch_is_feasible_distinct_and_new(dataset, ridge2):
    optimizer = build_optimizer("online-batch", {"pool_size": 1500})
    batch = optimizer.propose(ridge2, dataset.X, dataset.y, n=50)
    check_weights(batch)
    assert batch.shape == (50, 21)
    assert len(np.unique(np.round(batch, 9), axis=0)) == 50
    assert nearest_distances(batch, dataset.X).min() > 1e-8
    assert optimizer.diagnostics["exploit_count"] == 25
    assert optimizer.diagnostics["explore_count"] == 25


def test_online_batch_is_deterministic(dataset, ridge2):
    params = {"pool_size": 1200, "seed": 77}
    first = build_optimizer("online-batch", params).propose(ridge2, dataset.X, dataset.y, n=12)
    second = build_optimizer("online-batch", params).propose(ridge2, dataset.X, dataset.y, n=12)
    np.testing.assert_array_equal(first, second)


def test_second_batch_excludes_first_batch(dataset, ridge2):
    optimizer = build_optimizer("online-batch", {"pool_size": 1200, "seed": 77})
    first = optimizer.propose(ridge2, dataset.X, dataset.y, n=12)
    X = np.vstack((dataset.X, first))
    y = np.concatenate((dataset.y, ridge2.predict(first)[0]))
    second = optimizer.propose(ridge2, X, y, n=12)
    assert nearest_distances(second, X).min() > 1e-8
    assert not np.any(np.all(np.isclose(second[:, None, :], first[None, :, :], atol=1e-9), axis=2))


def test_trust_radius_update_and_resume():
    optimizer = build_optimizer("online-batch", {
        "radius_ratio": 1.0, "min_radius_ratio": 0.25, "max_radius_ratio": 1.5,
        "expand_factor": 1.5, "shrink_factor": 0.5,
    })
    assert optimizer.state_dict()["radius_ratio"] == 1.0
    optimizer.update(100.0, 99.0)
    assert optimizer.state_dict()["radius_ratio"] == 1.5
    optimizer.update(99.0, 99.0)
    assert optimizer.state_dict()["radius_ratio"] == 0.75
    state = optimizer.state_dict()
    restored = build_optimizer("online-batch")
    restored.load_state_dict(state)
    assert restored.state_dict() == state


@pytest.mark.parametrize("bad", [
    {"radius_ratio": 1.0, "proposal_index": 1.5, "success_streak": 0, "failure_streak": 0},
    {"radius_ratio": 1.0, "proposal_index": True, "success_streak": 0, "failure_streak": 0},
    {"radius_ratio": 1.0, "proposal_index": 0, "success_streak": -1, "failure_streak": 0},
])
def test_trust_radius_rejects_corrupt_checkpoint(bad):
    optimizer = build_optimizer("online-batch")
    with pytest.raises(ValueError, match="invalid online-batch state"):
        optimizer.load_state_dict(bad)


def test_online_batch_rejects_more_than_solver_batch_limit(dataset, ridge2):
    optimizer = build_optimizer("online-batch")
    with pytest.raises(ValueError, match="batch size"):
        optimizer.propose(ridge2, dataset.X, dataset.y, n=51)


def test_online_batch_still_proposes_after_radius_shrink(dataset, ridge2):
    optimizer = build_optimizer("online-batch", {
        "radius_ratio": 0.01, "min_radius_ratio": 0.01, "max_radius_ratio": 1.0,
        "pool_size": 256, "seed": 19,
    })
    batch = optimizer.propose(ridge2, dataset.X, dataset.y, n=50)
    check_weights(batch)
    assert batch.shape == (50, 21)
    assert nearest_distances(batch, dataset.X).max() <= 0.01 * 0.3772


@pytest.mark.parametrize("params", [
    {"pool_size": 63},
    {"seed": -1},
    {"success_patience": 1.5},
])
def test_online_batch_rejects_invalid_parameters(params):
    with pytest.raises(ValueError, match="invalid online-batch parameters"):
        build_optimizer("online-batch", params)
