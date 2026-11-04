"""Checks for covariance validity and the new uncertainty-based search."""
import numpy as np
from sdm.models.bayes_search import BuildingKernel, BuildingARDKernel, BayesianSearchModel


def test_building_covariance_positive_semidefinite():
    rng = np.random.default_rng(2026)
    X = rng.dirichlet(np.ones(7), size=(40, 3)).reshape(40, 21)
    kernel = BuildingKernel(.4)
    matrix = kernel(X)
    np.testing.assert_allclose(np.diag(matrix), kernel.diag(X))
    np.testing.assert_allclose(matrix, matrix.T)
    assert np.linalg.eigvalsh(matrix).min() > -1e-9
    np.testing.assert_allclose(kernel(X[:10], X[10:]), matrix[:10, 10:])


def test_gp_uncertainty_and_input_preservation(dataset):
    X, y = dataset.X[:100].copy(), dataset.y[:100].copy()
    original = y.copy()
    model = BayesianSearchModel(BayesianSearchModel.Params(lengths=(.5,), noises=(.005,))).fit(X, y)
    _, near = model.predict(X[:1])
    corner = np.zeros((3, 7)); corner[:, 0] = 1
    _, far = model.predict(corner)
    assert far[0] > near[0]
    np.testing.assert_array_equal(original, y)


def test_log_gp_inverse_positive(dataset):
    params = BayesianSearchModel.Params(target='log', lengths=(.5,), noises=(.01,))
    model = BayesianSearchModel(params).fit(dataset.X[:100],dataset.y[:100])
    mean, std = model.predict(dataset.X[100:150])
    assert (mean > 0).all() and (std >= 0).all()
    assert np.isfinite(mean).all() and np.isfinite(std).all()


def test_ard_covariance_analytic_gradient():
    rng = np.random.default_rng(3)
    X = rng.dirichlet(np.ones(7),size=(8,3)).reshape(8,21)
    kernel = BuildingARDKernel()
    matrix, gradient = kernel(X,eval_gradient=True)
    assert np.linalg.eigvalsh(matrix).min() > -1e-9
    for j in range(7):
        upper, lower = kernel.theta.copy(), kernel.theta.copy()
        upper[j] += 1e-5; lower[j] -= 1e-5
        numeric = (kernel.clone_with_theta(upper)(X)-kernel.clone_with_theta(lower)(X))/2e-5
        np.testing.assert_allclose(gradient[:,:,j],numeric,atol=1e-7,rtol=1e-5)
