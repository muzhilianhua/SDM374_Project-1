"""Tests for the evaluator-free sequential replay loop."""
import json

import numpy as np
import pytest

from sdm.interfaces import Optimizer, Surrogate
from sdm.replay import ReplayConfig, run_replay
from sdm.registry import build_optimizer
from sdm.simplex import sample


class QuadraticSurrogate(Surrogate):
    def fit(self, X, y):
        self.center = X[np.argmin(y)].copy()
        return self

    def predict(self, X):
        X = np.asarray(X)
        mean = np.sum((X - self.center) ** 2, axis=1)
        return mean, np.full(len(X), 0.05)


def _fixture():
    rng = np.random.default_rng(44)
    X = sample(24, rng)
    target = sample(1, np.random.default_rng(45))[0]
    y = np.sum((X - target) ** 2, axis=1)
    return X, y, target


def _run():
    X, y, target = _fixture()

    def evaluate(points):
        return np.sum((points - target) ** 2, axis=1)

    return run_replay(
        QuadraticSurrogate,
        build_optimizer("online-batch", {"pool_size": 400, "seed": 12}),
        X,
        y,
        evaluate,
        ReplayConfig(calls=3, batch_size=8),
    )


def test_replay_runs_sequential_batches_and_updates_best_curve():
    result = _run()

    assert result.best_curve.shape == (4,)
    assert np.all(np.diff(result.best_curve) <= 1e-12)
    assert len(result.rounds) == 3
    assert all(round_.points.shape == (8, 21) for round_ in result.rounds)
    assert len(np.unique(np.round(np.vstack([round_.points for round_ in result.rounds]), 12), axis=0)) == 24


def test_replay_is_deterministic_and_serializable():
    first, second = _run(), _run()
    np.testing.assert_array_equal(first.best_curve, second.best_curve)
    for left, right in zip(first.rounds, second.rounds):
        np.testing.assert_array_equal(left.points, right.points)
        np.testing.assert_array_equal(left.values, right.values)
    assert json.dumps(first.to_dict(), allow_nan=False) == json.dumps(second.to_dict(), allow_nan=False)


@pytest.mark.parametrize("config", [
    ReplayConfig(calls=21), ReplayConfig(batch_size=51), ReplayConfig(calls=0),
    ReplayConfig(batch_size=0), ReplayConfig(calls=True), ReplayConfig(batch_size=2.5),
])
def test_replay_rejects_invalid_budget(config):
    with pytest.raises(ValueError, match="replay budget"):
        config.validate()


class ScriptedOptimizer(Optimizer):
    def __init__(self, *batches):
        self.batches = iter(batches)

    def propose(self, model, X, y, n=1):
        return next(self.batches)


def test_replay_refits_with_previous_truth_and_updates_real_optimizer():
    X, y, _ = _fixture()
    y = np.full(len(y), 10.0)
    fits = []

    class RecordingModel(QuadraticSurrogate):
        def fit(self, X, y):
            fits.append((self, X.copy(), y.copy()))
            return super().fit(X, y)

    scores = iter(([11.0, 12.0], [9.0, 10.0]))
    optimizer = build_optimizer("online-batch", {"pool_size": 400})
    result = run_replay(RecordingModel, optimizer, X, y, lambda _: next(scores), ReplayConfig(2, 2))

    assert fits[0][0] is not fits[1][0]
    np.testing.assert_array_equal(fits[1][1], np.vstack([X, result.rounds[0].points]))
    np.testing.assert_array_equal(fits[1][2], np.r_[y, 11.0, 12.0])
    np.testing.assert_array_equal(result.best_curve, [10.0, 10.0, 9.0])
    assert result.rounds[0].batch_best == 11.0
    assert result.best_evaluated == 9.0
    assert result.rounds[0].optimizer_state["radius_ratio"] == pytest.approx(0.7)
    assert result.rounds[1].optimizer_state["radius_ratio"] == pytest.approx(1.05)


@pytest.mark.parametrize(("source", "message", "valid_calls"), [
    ("training", "previously observed", 0),
    ("earlier call", "previously observed", 1),
    ("within batch", "duplicate points", 0),
    ("short batch", "expected", 0),
    ("invalid weights", "sum to 1", 0),
])
def test_invalid_proposals_do_not_reach_evaluator(source, message, valid_calls):
    X, y, _ = _fixture()
    first, second, third, fourth = sample(4, np.random.default_rng(46))
    near_first = first.copy()
    near_first[:2] += (1e-9, -1e-9)
    invalid = {
        "training": [[X[0], second]],
        "earlier call": [[first, second], [near_first, third]],
        "within batch": [[first, near_first]],
        "short batch": [[first]],
        "invalid weights": [[first, 0.5 * fourth]],
    }
    calls = []

    def evaluate(points):
        calls.append(points)
        return [1.0, 2.0]

    with pytest.raises(ValueError, match=message):
        run_replay(QuadraticSurrogate, ScriptedOptimizer(*invalid[source]), X, y, evaluate, ReplayConfig(2, 2))
    assert len(calls) == valid_calls


def test_replay_rejects_invalid_truth_before_refitting():
    X, y, target = _fixture()
    optimizer = ScriptedOptimizer([target])
    with pytest.raises(ValueError, match="one finite value per proposed point"):
        run_replay(QuadraticSurrogate, optimizer, X, y, lambda _: [np.nan], ReplayConfig(2, 1))
