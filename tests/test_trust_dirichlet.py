"""trust-dirichlet: trust region, no re-measured rows, separation, objective order, gradient-free use."""
import numpy as np
import pytest

from sdm.evaluate import median_spacing, nearest_distances
from sdm.registry import build_model, build_optimizer


@pytest.fixture(scope="module")
def extratrees(dataset):
    return build_model("extratrees", {"n_estimators": 50}).fit(dataset.X, dataset.y)


def test_batch_respects_trust_region_separation_and_objective_order(dataset, extratrees):
    optimizer = build_optimizer("trust-dirichlet", {"iterations": 8, "risk": 1.0})
    batch = optimizer.propose(extratrees, dataset.X, dataset.y, n=20)
    distance = nearest_distances(batch, dataset.X)
    assert (distance <= 1.5 * median_spacing(dataset.X) + 1e-12).all()
    assert (distance > 1e-6).all(), "observed rows must not be proposed again"
    pairwise = np.linalg.norm(batch[:, None] - batch[None, :], axis=-1)[np.triu_indices(len(batch), 1)]
    assert pairwise.min() >= optimizer.diagnostics["separation"] - 1e-12
    mean, std = extratrees.predict(batch)
    assert np.all(np.diff(mean + std) >= -1e-9), "best first by mean + risk * std"


def test_empty_trust_region_is_reported(dataset, extratrees):
    optimizer = build_optimizer("trust-dirichlet", {"iterations": 2, "max_distance_ratio": 1e-9})
    with pytest.raises(ValueError, match="trust region"):
        optimizer.propose(extratrees, dataset.X, dataset.y, n=1)


def test_pgd_cannot_search_a_piecewise_constant_model(dataset, extratrees):
    with pytest.raises(NotImplementedError):
        build_optimizer("pgd").propose(extratrees, dataset.X, dataset.y, n=1)
