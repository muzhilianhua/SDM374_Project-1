"""Prevent duplicate budget use and silent feasibility relaxation in the round-one packet."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from sdm.simplex import check_weights, sample

spec=importlib.util.spec_from_file_location('online_round1',Path(__file__).parents[1]/'analysis/online_round1/prepare.py')
study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)


def test_rounding_preserves_simplex_near_vertices():
    rng=np.random.default_rng(18)
    points=np.concatenate([sample(30,rng),np.tile(np.eye(7)[0],(1,3)).reshape(1,21)])
    rounded=study.round_simplex(points)
    check_weights(rounded)
    assert np.max(np.abs(rounded-points))<4e-8
    np.testing.assert_array_equal(rounded[-1],points[-1])


def test_spread_excludes_repeats_and_rejects_infeasible_budget():
    points=sample(4,np.random.default_rng(2))
    points=np.vstack([points[0],points[0],points[1:]])
    chosen=study.spread_pick(points,list(range(5)),[0],3,.01)
    assert chosen==[2,3,4]
    with pytest.raises(ValueError,match='need 4'):
        study.spread_pick(points,list(range(5)),[0],4,.01)


def test_exploration_does_not_reselect_a_point_or_its_clone():
    points=sample(4,np.random.default_rng(7))
    points=np.vstack([points,points[0]])
    selected=study.exploration_pick(points,np.arange(5),[0],3,np.zeros(5),1,.01)
    assert set(selected)=={1,2,3}
    with pytest.raises(ValueError,match='Insufficient'):
        study.exploration_pick(points,np.arange(5),[0],4,np.zeros(5),1,.01)
