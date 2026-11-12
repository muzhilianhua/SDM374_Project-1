import numpy as np
import pytest

from analysis.development_search.study import candidate_pool, key, select_batch
from sdm.simplex import check_weights


@pytest.mark.parametrize('sparse', [False, True])
def test_pool_and_selection_are_feasible_deterministic_and_exclude_history(sparse):
    anchor = np.tile([0, 0.05, 0, 0, 0.25, 0.69, 0.01], 3) if sparse else np.full(21, 1/7)
    pool, families = candidate_pool(anchor, np.random.default_rng(7), size=300)
    repeat, _ = candidate_pool(anchor, np.random.default_rng(7), size=300)
    np.testing.assert_array_equal(pool, repeat)
    check_weights(pool)
    X = np.concatenate([[anchor], pool[:30]])
    y = np.arange(len(X), dtype=float)
    W, _ = select_batch(pool, families, X, y, np.random.default_rng(8), n=15)
    assert len({key(w) for w in W}) == 15
    assert not {key(w) for w in X}.intersection(key(w) for w in W)
