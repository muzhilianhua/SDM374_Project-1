"""local-gp fits only near the best row, so it must know where it is local and widen std away from it."""
import numpy as np

from sdm import registry


def test_fit_is_anchored_on_the_best_row(dataset):
    X, y = dataset.X[:300], dataset.y[:300]
    model = registry.build_model("local-gp", {"neighbors": 120}).fit(X, y)
    np.testing.assert_array_equal(model.anchor_, X[np.argmin(y)])
    assert model.metrics["local_rows"] == 120
    radius = np.sort(np.linalg.norm(X - model.anchor_, axis=1))[119]
    assert np.isclose(model.metrics["local_radius"], radius)


def test_uncertainty_grows_away_from_the_local_rows(dataset):
    X, y = dataset.X[:300], dataset.y[:300]
    model = registry.build_model("local-gp", {"neighbors": 120}).fit(X, y)
    distance = np.linalg.norm(X - model.anchor_, axis=1)
    near, far = X[np.argsort(distance)[:20]], X[np.argsort(distance)[-20:]]
    assert model.predict(far)[1].mean() > model.predict(near)[1].mean()
