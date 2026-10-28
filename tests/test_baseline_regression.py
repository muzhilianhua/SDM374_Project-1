"""The ridge2 + pgd plugins must stay equivalent to the course baseline.py."""
import json

import numpy as np
import pytest

from sdm.paths import REPO_ROOT
from sdm.registry import build_optimizer

# Frozen from baseline.py on 2026-10-23 (numpy 1.24.4, scikit-learn 1.2.2, Python 3.11).
EXPECTED_ALPHA = 100.0
EXPECTED_CV_MAE = 6.840115206277481
EXPECTED_CV_R2 = 0.6476298671101501
EXPECTED_PREDICTED_MIN = 75.89570106874648


def test_ridge2_matches_frozen_cv_metrics(ridge2):
    assert ridge2.metrics["alpha"] == EXPECTED_ALPHA
    assert ridge2.metrics["cv_mae"] == pytest.approx(EXPECTED_CV_MAE, rel=1e-9)
    assert ridge2.metrics["cv_r2"] == pytest.approx(EXPECTED_CV_R2, rel=1e-9)


def test_ridge2_matches_baseline_model(dataset, ridge2, baseline):
    model, report = baseline.train_model(dataset.X, dataset.y)
    c, a, H = baseline.quadratic_form(model)
    assert report["alpha"] == ridge2.metrics["alpha"]
    np.testing.assert_allclose(ridge2.hessian, H, rtol=0, atol=1e-10)
    np.testing.assert_allclose(ridge2.predict(dataset.X)[0], model.predict(dataset.X), rtol=0, atol=1e-8)


def test_pgd_matches_baseline_search(dataset, ridge2, baseline):
    starts = baseline.search_starts(dataset.X, dataset.y)
    expected, _ = baseline.gradient_search(ridge2.offset, ridge2.linear, ridge2.hessian, starts)
    proposed = build_optimizer("pgd").propose(ridge2, dataset.X, dataset.y, n=1)
    np.testing.assert_allclose(proposed.reshape(3, 7), expected, rtol=0, atol=1e-12)


def test_pgd_reproduces_shipped_weights(dataset, ridge2):
    shipped = np.array(json.loads((REPO_ROOT / "weights.json").read_text(encoding="utf-8")))
    proposed = build_optimizer("pgd").propose(ridge2, dataset.X, dataset.y, n=1)
    # weights.json was produced on another machine; BLAS differences show up around 1e-12.
    np.testing.assert_allclose(proposed.reshape(3, 7), shipped, rtol=0, atol=1e-6)
    assert ridge2.predict(proposed)[0][0] == pytest.approx(EXPECTED_PREDICTED_MIN, rel=1e-9)


def test_pgd_returns_ranked_valid_batch(dataset, ridge2):
    proposed = build_optimizer("pgd").propose(ridge2, dataset.X, dataset.y, n=5)
    assert proposed.shape == (5, 21)
    mean, _ = ridge2.predict(proposed)
    assert np.all(np.diff(mean) >= 0)
