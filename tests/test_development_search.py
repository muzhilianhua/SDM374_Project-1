import json

import numpy as np
import pytest

from analysis.development_search.study import candidate_pool, key, load_history, select_batch
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


def test_history_collapses_equivalent_building_scales_and_rejects_negative_weights(tmp_path):
    path = tmp_path / 'history.jsonl'
    w = np.full((3, 7), 1/7)
    scaled = w * np.array([2, 0.1, 10])[:, None]
    records = [{'weights': value.tolist(), 'R': 123.0} for value in (w, scaled)]
    path.write_text('\n'.join(json.dumps(row) for row in records), encoding='utf-8')
    X, y = load_history(path)
    assert X.shape == (1, 21)
    np.testing.assert_allclose(X[0], w.ravel())
    assert y.tolist() == [123.0]
    w[0, 0] = -0.1
    path.write_text(json.dumps({'weights': w.tolist(), 'R': 123.0}), encoding='utf-8')
    with pytest.raises(ValueError, match='invalid historical weights'):
        load_history(path)
