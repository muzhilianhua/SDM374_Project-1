"""Contract every registered plugin must satisfy. New plugins are picked up automatically."""
import json

import numpy as np
import pytest

from sdm import registry
from sdm.simplex import check_weights

FIT_ROWS = 200


@pytest.mark.parametrize("name", sorted(registry.models()))
def test_model_contract(name, dataset):
    model = registry.build_model(name)
    assert model.fit(dataset.X[:FIT_ROWS], dataset.y[:FIT_ROWS]) is model, "fit must return self"
    mean, std = model.predict(dataset.X[FIT_ROWS:FIT_ROWS + 50])
    assert np.shape(mean) == (50,) and np.shape(std) == (50,), "predict must return (mean, std) of shape (n,)"
    assert np.isfinite(mean).all() and np.isfinite(std).all()
    assert (np.asarray(std) >= 0).all(), "std must be non-negative"
    single_mean, _ = model.predict(dataset.X[0].reshape(3, 7))
    assert np.shape(single_mean) == (1,), "a single (3, 7) weight set must be accepted"
    json.dumps(model.metrics)
    # `sdm reproduce` relies on this: same params and data must give the same predictions.
    again = registry.build_model(name).fit(dataset.X[:FIT_ROWS], dataset.y[:FIT_ROWS])
    np.testing.assert_allclose(again.predict(dataset.X[FIT_ROWS:FIT_ROWS + 50])[0], mean, rtol=1e-9,
                               err_msg="fit must be deterministic for fixed params (seed your randomness)")


@pytest.mark.parametrize("name", sorted(registry.optimizers()))
def test_optimizer_contract(name, dataset, ridge2):
    optimizer = registry.build_optimizer(name)
    batch = optimizer.propose(ridge2, dataset.X, dataset.y, n=5)
    assert np.shape(batch) == (5, 21)
    check_weights(batch)
    # Optimizers order by their own objective (possibly mean + k * std); ridge2's std is constant,
    # so with this fixture every sensible objective orders like the mean.
    mean, _ = ridge2.predict(batch)
    assert np.all(np.diff(mean) >= -1e-9), "proposals must be ordered best (lowest mean) first"
    json.dumps(optimizer.diagnostics)


# pgd is frozen as the baseline and returns duplicates; new optimizers must not.
KNOWN_DUPLICATES = {"pgd"}


@pytest.mark.parametrize("name", [
    pytest.param(name, marks=pytest.mark.xfail(strict=True, reason="baseline pgd converges all starts to one point"))
    if name in KNOWN_DUPLICATES else name
    for name in sorted(registry.optimizers())
])
def test_optimizer_batches_are_distinct(name, dataset, ridge2):
    batch = registry.build_optimizer(name).propose(ridge2, dataset.X, dataset.y, n=10)
    assert len(np.unique(np.round(batch, 9), axis=0)) == len(batch), "online rounds need distinct points"
