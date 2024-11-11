"""Tests for Sigma Ridge regression."""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from sklearn.exceptions import NotFittedError
import warnings
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from ..src.sigma_ridge import SigmaRidgeRegressor
from ..src.blockridge import SingularMatrixError, GroupedFeatures


def test_sigma_ridge_init():
    """Test SigmaRidgeRegressor initialization."""
    reg = SigmaRidgeRegressor()
    assert reg.sigma == 1.0
    assert reg.center is True
    assert reg.scale is True
    assert reg.tol == 1e-4
    assert reg.max_iter == 1000

    reg = SigmaRidgeRegressor(
        sigma=0.5, center=False, scale=False, tol=1e-5, max_iter=500
    )
    assert reg.sigma == 0.5
    assert reg.center is False
    assert reg.scale is False
    assert reg.tol == 1e-5
    assert reg.max_iter == 500


def test_sigma_ridge_parameter_validation():
    """Test parameter validation in SigmaRidgeRegressor."""
    X, y = make_regression(n_samples=10, n_features=5, random_state=42)

    with pytest.raises(ValueError, match="sigma must be positive"):
        reg = SigmaRidgeRegressor(sigma=-1.0)
        reg.fit(X, y)

    with pytest.raises(ValueError, match="tol must be positive"):
        reg = SigmaRidgeRegressor(tol=-1e-4)
        reg.fit(X, y)

    with pytest.raises(ValueError, match="max_iter must be positive"):
        reg = SigmaRidgeRegressor(max_iter=0)
        reg.fit(X, y)

    with pytest.raises(ValueError, match="decomposition must be one of"):
        reg = SigmaRidgeRegressor(decomposition="invalid")
        reg.fit(X, y)

    with pytest.raises(ValueError, match="optimization_method must be one of"):
        reg = SigmaRidgeRegressor(optimization_method="invalid")
        reg.fit(X, y)

    with pytest.raises(ValueError, match="sigma_range must be a tuple"):
        reg = SigmaRidgeRegressor(sigma_range=[1.0, 2.0])
        reg.fit(X, y)

    with pytest.raises(ValueError, match="sigma_range must be a tuple"):
        reg = SigmaRidgeRegressor(sigma_range=(2.0, 1.0))
        reg.fit(X, y)


def test_sigma_ridge_feature_groups():
    """Test feature group handling in SigmaRidgeRegressor."""
    X, y = make_regression(n_samples=10, n_features=4, random_state=42)

    reg = SigmaRidgeRegressor(feature_groups=[[0, 1], [2, 3]])
    reg.fit(X, y)
    assert reg.feature_groups_.num_groups == 2

    with pytest.raises(ValueError, match="Features cannot belong to multiple groups"):
        reg = SigmaRidgeRegressor(feature_groups=[[0, 1], [1, 2]])
        reg.fit(X, y)

    with pytest.raises(ValueError, match="All features must be assigned to a group"):
        reg = SigmaRidgeRegressor(feature_groups=[[0, 1]])
        reg.fit(X, y)

    with pytest.raises(ValueError, match="Empty groups are not allowed"):
        reg = SigmaRidgeRegressor(feature_groups=[[0, 1], []])
        reg.fit(X, y)

    with pytest.raises(ValueError, match="Invalid feature indices"):
        reg = SigmaRidgeRegressor(feature_groups=[[0, 1], [4, 5]])
        reg.fit(X, y)


def test_init_ridge_estimator_single_group():
    """Test that initial ridge estimator uses a single group."""
    X, y = make_regression(n_samples=50, n_features=10, random_state=42)

    reg1 = SigmaRidgeRegressor(feature_groups=[[i] for i in range(10)])
    reg2 = SigmaRidgeRegressor(feature_groups=[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    reg3 = SigmaRidgeRegressor()

    sigma_range = (0.001, 1.0)
    reg1.set_params(sigma_range=sigma_range)
    reg2.set_params(sigma_range=sigma_range)
    reg3.set_params(sigma_range=sigma_range)

    reg1.fit(X, y)
    reg2.fit(X, y)
    reg3.fit(X, y)

    initial_lambda1 = reg1.ridge_estimator_.alpha[0]
    initial_lambda2 = reg2.ridge_estimator_.alpha[0]
    initial_lambda3 = reg3.ridge_estimator_.alpha[0]

    assert np.abs(initial_lambda1 - initial_lambda2) < 1e-2
    assert np.abs(initial_lambda1 - initial_lambda3) < 1e-2
    assert np.abs(initial_lambda2 - initial_lambda3) < 1e-2


def test_init_ridge_estimator_loo_cv():
    """Test that initial ridge estimator uses LOO CV."""
    X, y = make_regression(n_samples=50, n_features=10, random_state=42)

    reg1 = SigmaRidgeRegressor(sigma_range=(0.001, 0.1))
    reg2 = SigmaRidgeRegressor(sigma_range=(1.0, 100.0))

    reg1.fit(X, y)
    reg2.fit(X, y)

    assert not np.allclose(
        reg1.ridge_estimator_.alpha, reg2.ridge_estimator_.alpha, rtol=1e-10
    )


def test_sigma_ridge_fit_predict():
    """Test basic fit and predict functionality."""
    X, y = make_regression(n_samples=100, n_features=20, random_state=42)
    X_test = X[:10]

    reg = SigmaRidgeRegressor()
    reg.fit(X, y)
    y_pred = reg.predict(X_test)
    assert y_pred.shape == (10,)

    groups = [[i, i + 1] for i in range(0, 20, 2)]
    reg = SigmaRidgeRegressor(feature_groups=groups)
    reg.fit(X, y)
    y_pred = reg.predict(X_test)
    assert y_pred.shape == (10,)

    reg1 = SigmaRidgeRegressor(
        optimization_method="grid_search", sigma_range=(0.001, 1.0)
    )
    reg1.fit(X, y)
    y_pred_grid = reg1.predict(X_test)

    reg2 = SigmaRidgeRegressor(optimization_method=None, sigma_range=(1.0, 1000.0))
    reg2.fit(X, y)
    y_pred_geom = reg2.predict(X_test)

    assert np.any(np.abs(y_pred_grid - y_pred_geom) > 1e-10)


def test_sigma_ridge_center_scale():
    """Test centering and scaling functionality."""
    X, y = make_regression(n_samples=100, n_features=20, random_state=42)
    X_test = X[:10]

    reg = SigmaRidgeRegressor(center=True, scale=True)
    reg.fit(X, y)
    y_pred_scaled = reg.predict(X_test)

    reg = SigmaRidgeRegressor(center=False, scale=False)
    reg.fit(X, y)
    y_pred_unscaled = reg.predict(X_test)

    assert not np.array_equal(y_pred_scaled, y_pred_unscaled)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    reg = SigmaRidgeRegressor(center=False, scale=False)
    reg.fit(X_scaled, y)
    y_pred_manual_scale = reg.predict(scaler.transform(X_test))

    assert_array_almost_equal(y_pred_scaled, y_pred_manual_scale, decimal=10)


def test_sigma_ridge_sigma_max():
    """Test σ_max grid optimization."""
    X, y = make_regression(n_samples=100, n_features=20, random_state=42)

    reg = SigmaRidgeRegressor()
    reg.fit(X, y)
    sigma_opt_auto = reg.sigma_opt_

    reg = SigmaRidgeRegressor(sigma_range=(0.1, 10.0))
    reg.fit(X, y)
    sigma_opt_custom = reg.sigma_opt_

    assert sigma_opt_auto != sigma_opt_custom

    reg = SigmaRidgeRegressor()
    reg.fit(X, y)
    moment_tuner = reg.moment_tuner_
    sigma_max = np.sqrt(moment_tuner.get_sigma_squared_max())
    assert 1e-3 * sigma_max <= reg.sigma_opt_ <= sigma_max


def test_sigma_ridge_edge_cases():
    """Test edge cases and numerical stability."""
    X = np.array([[1.0, 1.0], [1.1, 1.1], [0.9, 0.9]])
    y = np.array([1.0, 1.1, 0.9])

    reg = SigmaRidgeRegressor()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg.fit(X, y)
        reg.predict(X)

    X = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    y = np.array([1.0, 1.0, 1.0])

    reg = SigmaRidgeRegressor()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg.fit(X, y)
        y_pred = reg.predict(X)
        assert y_pred.shape == (3,)

    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([1.0, 2.0, 3.0])

    reg = SigmaRidgeRegressor()
    reg.fit(X, y)
    assert reg.coef_.shape == (1,)

    X = np.array([[1.0, 2.0]])
    y = np.array([1.0])

    reg = SigmaRidgeRegressor()
    with pytest.raises(ValueError):
        reg.fit(X, y)


def test_sigma_ridge_input_validation():
    """Test input validation and error handling."""
    X, y = make_regression(n_samples=100, n_features=20, random_state=42)
    reg = SigmaRidgeRegressor()

    with pytest.raises(NotFittedError):
        reg.predict(X)

    reg.fit(X, y)
    with pytest.raises(ValueError):
        reg.predict(X[:, :10])

    X_inf = X.copy()
    X_inf[0, 0] = np.inf
    with pytest.raises(ValueError):
        reg.fit(X_inf, y)

    X_nan = X.copy()
    X_nan[0, 0] = np.nan
    with pytest.raises(ValueError):
        reg.fit(X_nan, y)

    with pytest.raises(ValueError):
        reg.fit(X.reshape(-1), y)

    groups = [[0, 1], [2, 3]]
    reg = SigmaRidgeRegressor(feature_groups=groups)
    with pytest.raises(ValueError):
        reg.fit(X, y)


def test_sigma_ridge_decomposition_methods():
    """Test different decomposition methods."""
    X, y = make_regression(n_samples=100, n_features=20, random_state=42)

    reg = SigmaRidgeRegressor(decomposition="cholesky")
    reg.fit(X, y)
    y_pred_chol = reg.predict(X)

    X, y = make_regression(n_samples=20, n_features=100, random_state=42)

    reg = SigmaRidgeRegressor(decomposition="woodbury")
    reg.fit(X, y)
    y_pred_wood = reg.predict(X)

    reg = SigmaRidgeRegressor(decomposition="default")
    reg.fit(X, y)
    y_pred_auto = reg.predict(X)

    assert_array_almost_equal(y_pred_wood, y_pred_auto, decimal=10)


def test_sigma_ridge_get_set_params():
    """Test getting and setting parameters."""
    reg = SigmaRidgeRegressor(sigma=1.0, center=True)
    params = reg.get_params()

    assert params["sigma"] == 1.0
    assert params["center"] is True

    reg.set_params(sigma=2.0, center=False)
    assert reg.sigma == 2.0
    assert reg.center is False

    with pytest.raises(ValueError):
        reg.set_params(invalid_param=1.0)
