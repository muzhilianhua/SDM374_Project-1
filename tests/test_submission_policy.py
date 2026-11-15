"""Standalone T15 policy checks independent of the sdm registry."""

import numpy as np

from submission.policy import BlockDescentPolicy, Params, State
from submission.surrogate import Surrogate, features


def _state(dataset, calls=()):
    return State(X_train=dataset.X, y_train=dataset.y, calls=list(calls))


def test_submission_surrogate_uses_fixed_kernel(dataset):
    model = Surrogate(local_rows=80, alpha=1.0, length_scale=1.0, noise_level=0.1)
    model.fit(dataset.X, dataset.y, center=dataset.X[np.argmin(dataset.y)])
    mean, std = model.predict(dataset.X[:5])
    assert mean.shape == std.shape == (5,)
    assert np.isfinite(mean).all()
    assert np.isfinite(std).all() and (std >= 0).all()
    assert model.gp.optimizer is None


def test_submission_policy_returns_full_new_batch(dataset):
    params = Params(pool=400, local_rows=80)
    batch, metadata, info = BlockDescentPolicy(params, seed=7).propose_weights(_state(dataset), n=50)
    flat = batch.reshape(50, 21)
    assert flat.shape == (50, 21)
    assert len(metadata) == 50
    assert len({tuple(np.round(row, 9)) for row in flat}) == 50
    assert len(info["scale"]) == 3
    assert np.allclose(batch.sum(axis=2), 1.0, atol=1e-12)
    train = dataset.X.reshape(-1, 21)
    assert np.max(np.abs(flat[:, None, :] - train[None, :, :]), axis=2).min() > 1e-9


def test_submission_policy_excludes_failed_and_measured_points(dataset):
    first, _, _ = BlockDescentPolicy(Params(pool=400, local_rows=80), seed=7).propose_weights(
        _state(dataset), n=50
    )
    measured = first[:25]
    failed = first[25:30]
    call = [{"W": row, "R": float(110 + i), "meta": {"kind": "single", "building": i % 3}}
            for i, row in enumerate(measured)]
    call.extend({"W": row, "R": float("nan"), "meta": {"kind": "single", "building": i % 3}}
                for i, row in enumerate(failed))
    second, _, _ = BlockDescentPolicy(Params(pool=400, local_rows=80), seed=7).propose_weights(
        _state(dataset, calls=[call]), n=50
    )
    flat = second.reshape(50, 21)
    sent = np.vstack([first.reshape(50, 21)])
    assert np.max(np.abs(flat[:, None, :] - sent[None, :, :]), axis=2).min() > 1e-9


def test_submission_policy_is_deterministic_for_resume(dataset):
    state = _state(dataset)
    left, _, _ = BlockDescentPolicy(Params(pool=400, local_rows=80), seed=2026).propose_weights(state, n=50)
    right, _, _ = BlockDescentPolicy(Params(pool=400, local_rows=80), seed=2026).propose_weights(state, n=50)
    np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(features(left.reshape(50, 21)), features(right.reshape(50, 21)))


def test_submission_policy_batch_parity_across_three_calls(dataset):
    """A resumed runner emits the same batches as an uninterrupted runner, call by call."""
    params = Params(pool=180, local_rows=48)
    uninterrupted = _state(dataset)
    resumed = _state(dataset)
    left_policy = BlockDescentPolicy(params, seed=11)
    right_policy = BlockDescentPolicy(params, seed=11)
    for _ in range(3):
        left, _, _ = left_policy.propose_weights(uninterrupted, n=50)
        right, _, _ = right_policy.propose_weights(resumed, n=50)
        np.testing.assert_array_equal(left, right)
        records = [{"W": row, "R": float(100.0 + np.square(row).sum()),
                    "meta": {"kind": "single", "building": i % 3}}
                   for i, row in enumerate(left)]
        uninterrupted = _state(dataset, calls=uninterrupted.calls + [records])
        resumed = _state(dataset, calls=resumed.calls + [records])
