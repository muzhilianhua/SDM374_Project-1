"""The submission-2 study steps along simplex directions, so every step must keep each building's sum at 1."""
import importlib.util

import numpy as np

from sdm.paths import REPO_ROOT

spec = importlib.util.spec_from_file_location("offline_submission2_study",
                                              REPO_ROOT / "analysis/offline_submission2/study.py")
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def test_directions_are_unit_and_keep_building_sums(dataset):
    u = study.simplex_directions(500, np.random.default_rng(0))
    np.testing.assert_allclose(np.linalg.norm(u, axis=1), 1.0)
    np.testing.assert_allclose(u.reshape(-1, 3, 7).sum(axis=2), 0.0, atol=1e-12)
    step = dataset.X[0] + 0.01 * u
    np.testing.assert_allclose(step.reshape(-1, 3, 7).sum(axis=2), 1.0, atol=1e-12)


sign_spec = importlib.util.spec_from_file_location("offline_submission2_sign",
                                                   REPO_ROOT / "analysis/offline_submission2/sign_study.py")
sign = importlib.util.module_from_spec(sign_spec)
sign_spec.loader.exec_module(sign)


def test_pair_stats_match_each_judges_own_prediction(dataset):
    from sdm import registry
    X, y = dataset.X[:200], dataset.y[:200]
    judges = [registry.build_model("local-gp", {"neighbors": 100}).fit(X, y),
              registry.build_model("bayes-search", {"kernel": "matern", "lengths": [0.5], "noises": [0.03]}).fit(X, y)]
    Q = dataset.X[200:230]
    for model in judges:
        stats = sign.pair_stats(model, X[0], Q, chunk=7)
        mean, sd = model.predict(Q)
        np.testing.assert_allclose(stats["mean"], mean, rtol=1e-8)
        np.testing.assert_allclose(stats["sd"], sd, rtol=1e-6)
        np.testing.assert_allclose(stats.gain, mean - model.predict(X[:1])[0][0], rtol=1e-8, atol=1e-8)
        assert (stats.gain_sd >= 0).all()
