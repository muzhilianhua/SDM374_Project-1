"""Boosted-tree plugins: the low-target weighting rule and the distance-stratified conformal std."""
import numpy as np
import pytest

from sdm import registry
from sdm.models._gbdt import low_target_weights

FIT_ROWS = 400


def test_low_target_weights_follow_the_xgb_w20x3_rule(dataset):
    weights = low_target_weights(dataset.y, 0.2, 3.0)
    assert int(np.sum(weights == 3.0)) == 409 and int(np.sum(weights == 1.0)) == 2048 - 409
    assert dataset.y[weights == 3.0].max() <= dataset.y[weights == 1.0].min()


@pytest.mark.parametrize("name", ["histgb", "xgboost"])
def test_std_grows_with_distance_from_the_data(name, dataset):
    model = registry.build_model(name).fit(dataset.X[:FIT_ROWS], dataset.y[:FIT_ROWS])
    assert np.all(np.diff(model.conformal.std_) >= 0)
    _, near = model.predict(dataset.X[:FIT_ROWS])
    # A simplex corner (all weight on cost in every building) is far from every measured row.
    corner = np.zeros((1, 21))
    corner[0, [0, 7, 14]] = 1.0
    _, far = model.predict(corner)
    assert far[0] == model.conformal.std_[-1] and np.all(near <= far[0])


def test_unit_low_weight_matches_unweighted_fit(dataset):
    X, y = dataset.X[:FIT_ROWS], dataset.y[:FIT_ROWS]
    weighted = registry.build_model("histgb", {"low_weight": 1.0, "low_fraction": 0.5}).fit(X, y)
    plain = registry.build_model("histgb", {"low_weight": 1.0, "low_fraction": 0.0}).fit(X, y)
    np.testing.assert_array_equal(weighted.predict(dataset.X[FIT_ROWS:])[0], plain.predict(dataset.X[FIT_ROWS:])[0])
