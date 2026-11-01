"""Gradient correctness, cross-fitted convex stacking and serializable alternative predictors."""
import pickle

import numpy as np
import pytest

from sdm.models.method_study import features, PairwiseSquaredLoss, convex_weights
from sdm.registry import build_model


def test_interactions_preserve_raw_columns_and_hellinger_geometry(dataset):
    X = dataset.X[:12]
    transformed = features(X,"interactions")
    assert transformed.shape == (12,105)
    np.testing.assert_array_equal(transformed[:,:21],X)
    np.testing.assert_allclose(features(X,"hellinger")**2,X)


def test_pairwise_gradient_and_diagonal_hessian_match_finite_differences():
    y, pred = np.array([-2.,-1.,0.,1.,2.]), np.array([.3,-.1,.2,.5,.7])
    objective = PairwiseSquaredLoss(.3,200,2026,1.,.01)
    gradient,hessian = objective(y,pred)
    epsilon = 1e-5
    for i in range(len(y)):
        delta = np.eye(len(y))[i] * epsilon
        derivative = (objective.value(y,pred+delta)-objective.value(y,pred-delta))/(2*epsilon)
        curvature = (objective(y,pred+delta)[0][i]-objective(y,pred-delta)[0][i])/(2*epsilon)
        assert derivative == pytest.approx(gradient[i],abs=1e-7)
        assert curvature == pytest.approx(hessian[i],abs=1e-7)
    assert (hessian>0).all()


def test_zero_rank_strength_reduces_to_weighted_squared_error():
    y,pred,w = np.array([1.,2.,3.]),np.array([3.,1.,2.]),np.array([1.,3.,1.])
    g,h = PairwiseSquaredLoss(0)(y,pred,w)
    np.testing.assert_array_equal(g,w*(pred-y))
    np.testing.assert_array_equal(h,w)


def test_convex_stacking_downweights_inaccurate_member():
    y = np.arange(20.)
    prediction = np.column_stack([y,y+10,y-15])
    w = convex_weights(prediction,y,np.ones(20),np.array([.5,.3,.2]))
    assert (w>=0).all() and w.sum()==pytest.approx(1)
    assert np.mean((prediction@w-y)**2) < 1e-3


@pytest.mark.parametrize("family,extra",[
    ("catboost",{}),("gbdt",{"loss":"huber","depth":3}),
    ("xgboost",{"loss":"pseudo-huber","depth":3,"learning_rate":.1}),
    ("pairwise",{"depth":3,"learning_rate":.1,"low_weight":1}),
    ("kernel",{"transform":"hellinger","low_weight":1}),
    ("ensemble",{"ensemble_mode":"stack"}),
])
def test_alternative_families_predict_finite_values_and_survive_pickle(dataset,family,extra):
    params = dict(family=family,iterations=12,calib_folds=2,pair_count=256,**extra)
    model = build_model("method-study",params).fit(dataset.X[:180],dataset.y[:180])
    mean,std = model.predict(dataset.X[180:200])
    assert mean.shape==std.shape==(20,) and np.isfinite(mean).all() and np.isfinite(std).all()
    assert (std>=0).all() and np.all(np.diff(model.conformal.std_)>=0)
    loaded = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(loaded.predict(dataset.X[180:200])[0],mean)


def test_stacking_members_use_only_fold_training_labels(dataset,monkeypatch):
    from sdm.models.method_study import MethodStudy
    calls=[]
    original=MethodStudy._core
    def record(self,X,y,family=None):
        calls.append((len(y),family))
        return original(self,X,y,family)
    monkeypatch.setattr(MethodStudy,"_core",record)
    model=build_model("method-study",dict(family="ensemble",ensemble_mode="stack",iterations=5,calib_folds=3))
    model.fit(dataset.X[:180],dataset.y[:180])
    assert sum(n==120 for n,_ in calls)==9 and sum(n==180 for n,_ in calls)==3
    assert model.blend_weights.sum()==pytest.approx(1)
