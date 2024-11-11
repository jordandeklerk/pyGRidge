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
    # Test default parameters
    reg = SigmaRidgeRegressor()
    assert reg.sigma == 1.0
    assert reg.center is True
    assert reg.scale is True
    assert reg.tol == 1e-4
    assert reg.max_iter == 1000

    # Test custom parameters
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

    # Test invalid sigma
    with pytest.raises(ValueError, match="sigma must be positive"):
        reg = SigmaRidgeRegressor(sigma=-1.0)
        reg.fit(X, y)

    # Test invalid tol
    with pytest.raises(ValueError, match="tol must be positive"):
        reg = SigmaRidgeRegressor(tol=-1e-4)
        reg.fit(X, y)

    # Test invalid max_iter
    with pytest.raises(ValueError, match="max_iter must be positive"):
        reg = SigmaRidgeRegressor(max_iter=0)
        reg.fit(X, y)

    # Test invalid decomposition
    with pytest.raises(ValueError, match="decomposition must be one of"):
        reg = SigmaRidgeRegressor(decomposition="invalid")
        reg.fit(X, y)

    # Test invalid optimization_method
    with pytest.raises(ValueError, match="optimization_method must be one of"):
        reg = SigmaRidgeRegressor(optimization_method="invalid")
        reg.fit(X, y)

    # Test invalid sigma_range
    with pytest.raises(ValueError, match="sigma_range must be a tuple"):
        reg = SigmaRidgeRegressor(sigma_range=[1.0, 2.0])
        reg.fit(X, y)

    with pytest.raises(ValueError, match="sigma_range must be a tuple"):
        reg = SigmaRidgeRegressor(sigma_range=(2.0, 1.0))
        reg.fit(X, y)


def test_sigma_ridge_feature_groups():
    """Test feature group handling in SigmaRidgeRegressor."""
    X, y = make_regression(n_samples=10, n_features=4, random_state=42)

    # Test valid feature groups
    reg = SigmaRidgeRegressor(feature_groups=[[0, 1], [2, 3]])
    reg.fit(X, y)
    assert reg.feature_groups_.num_groups == 2

    # Test overlapping groups
    with pytest.raises(ValueError, match="Features cannot belong to multiple groups"):
        reg = SigmaRidgeRegressor(feature_groups=[[0, 1], [1, 2]])
        reg.fit(X, y)

    # Test missing features
    with pytest.raises(ValueError, match="All features must be assigned to a group"):
        reg = SigmaRidgeRegressor(feature_groups=[[0, 1]])
        reg.fit(X, y)

    # Test empty groups
    with pytest.raises(ValueError, match="Empty groups are not allowed"):
        reg = SigmaRidgeRegressor(feature_groups=[[0, 1], []])
        reg.fit(X, y)

    # Test invalid feature indices
    with pytest.raises(ValueError, match="Invalid feature indices"):
        reg = SigmaRidgeRegressor(feature_groups=[[0, 1], [4, 5]])
        reg.fit(X, y)


def test_init_ridge_estimator_single_group():
    """Test that initial ridge estimator uses a single group."""
    X, y = make_regression(n_samples=50, n_features=10, random_state=42)

    # Create estimators with different feature groups
    reg1 = SigmaRidgeRegressor(
        feature_groups=[[i] for i in range(10)]
    )  # Each feature in its own group
    reg2 = SigmaRidgeRegressor(
        feature_groups=[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    )  # Two groups
    reg3 = SigmaRidgeRegressor()  # Default: each feature in its own group

    # Use same sigma range for all estimators to ensure comparable results
    sigma_range = (0.001, 1.0)
    reg1.set_params(sigma_range=sigma_range)
    reg2.set_params(sigma_range=sigma_range)
    reg3.set_params(sigma_range=sigma_range)

    reg1.fit(X, y)
    reg2.fit(X, y)
    reg3.fit(X, y)

    # The initial ridge estimators should have similar lambdas since they all
    # use a single group for initialization, despite having different final group structures
    initial_lambda1 = reg1.ridge_estimator_.alpha[0]  # Get the first lambda
    initial_lambda2 = reg2.ridge_estimator_.alpha[0]  # Get the first lambda
    initial_lambda3 = reg3.ridge_estimator_.alpha[0]  # Get the first lambda

    assert np.abs(initial_lambda1 - initial_lambda2) < 1e-2
    assert np.abs(initial_lambda1 - initial_lambda3) < 1e-2
    assert np.abs(initial_lambda2 - initial_lambda3) < 1e-2


def test_init_ridge_estimator_loo_cv():
    """Test that initial ridge estimator uses LOO CV."""
    X, y = make_regression(n_samples=50, n_features=10, random_state=42)

    # Create two estimators with different sigma ranges
    reg1 = SigmaRidgeRegressor(sigma_range=(0.001, 0.1))
    reg2 = SigmaRidgeRegressor(sigma_range=(1.0, 100.0))

    reg1.fit(X, y)
    reg2.fit(X, y)

    # The initial ridge estimators should have different lambdas due to
    # different sigma ranges in LOO CV
    assert not np.allclose(
        reg1.ridge_estimator_.alpha, reg2.ridge_estimator_.alpha, rtol=1e-10
    )


def test_sigma_ridge_fit_predict():
    """Test basic fit and predict functionality."""
    X, y = make_regression(n_samples=100, n_features=20, random_state=42)
    X_test = X[:10]

    # Test with default parameters
    reg = SigmaRidgeRegressor()
    reg.fit(X, y)
    y_pred = reg.predict(X_test)
    assert y_pred.shape == (10,)

    # Test with feature groups
    groups = [[i, i + 1] for i in range(0, 20, 2)]
    reg = SigmaRidgeRegressor(feature_groups=groups)
    reg.fit(X, y)
    y_pred = reg.predict(X_test)
    assert y_pred.shape == (10,)

    # Test with different optimization methods and significantly different sigma ranges
    reg1 = SigmaRidgeRegressor(
        optimization_method="grid_search", sigma_range=(0.001, 1.0)
    )
    reg1.fit(X, y)
    y_pred_grid = reg1.predict(X_test)

    reg2 = SigmaRidgeRegressor(optimization_method=None, sigma_range=(1.0, 1000.0))
    reg2.fit(X, y)
    y_pred_geom = reg2.predict(X_test)

    # Results should be different due to different sigma ranges
    assert np.any(np.abs(y_pred_grid - y_pred_geom) > 1e-10)


def test_sigma_ridge_center_scale():
    """Test centering and scaling functionality."""
    X, y = make_regression(n_samples=100, n_features=20, random_state=42)
    X_test = X[:10]

    # Test with centering and scaling
    reg = SigmaRidgeRegressor(center=True, scale=True)
    reg.fit(X, y)
    y_pred_scaled = reg.predict(X_test)

    # Test without centering and scaling
    reg = SigmaRidgeRegressor(center=False, scale=False)
    reg.fit(X, y)
    y_pred_unscaled = reg.predict(X_test)

    # Results should be different
    assert not np.array_equal(y_pred_scaled, y_pred_unscaled)

    # Test that centering and scaling is actually applied
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    reg = SigmaRidgeRegressor(center=False, scale=False)
    reg.fit(X_scaled, y)
    y_pred_manual_scale = reg.predict(scaler.transform(X_test))

    assert_array_almost_equal(y_pred_scaled, y_pred_manual_scale, decimal=10)


def test_sigma_ridge_sigma_max():
    """Test σ_max grid optimization."""
    X, y = make_regression(n_samples=100, n_features=20, random_state=42)

    # Test automatic sigma range based on σ_max
    reg = SigmaRidgeRegressor()
    reg.fit(X, y)
    sigma_opt_auto = reg.sigma_opt_

    # Test custom sigma range
    reg = SigmaRidgeRegressor(sigma_range=(0.1, 10.0))
    reg.fit(X, y)
    sigma_opt_custom = reg.sigma_opt_

    # Results should be different as ranges are different
    assert sigma_opt_auto != sigma_opt_custom

    # Test that sigma_opt is within the expected range
    reg = SigmaRidgeRegressor()
    reg.fit(X, y)
    moment_tuner = reg.moment_tuner_
    sigma_max = np.sqrt(moment_tuner.get_sigma_squared_max())
    assert 1e-3 * sigma_max <= reg.sigma_opt_ <= sigma_max


def test_sigma_ridge_edge_cases():
    """Test edge cases and numerical stability."""
    # Test with nearly singular matrix
    X = np.array([[1.0, 1.0], [1.1, 1.1], [0.9, 0.9]])
    y = np.array([1.0, 1.1, 0.9])

    reg = SigmaRidgeRegressor()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg.fit(X, y)  # Should handle near-singular matrix by increasing regularization
        reg.predict(X)  # Should be able to make predictions

    # Test with zero variance features
    X = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    y = np.array([1.0, 1.0, 1.0])

    reg = SigmaRidgeRegressor()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg.fit(X, y)  # Should handle zero variance features gracefully
        y_pred = reg.predict(X)  # Should be able to make predictions
        assert y_pred.shape == (3,)

    # Test with single feature
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([1.0, 2.0, 3.0])

    reg = SigmaRidgeRegressor()
    reg.fit(X, y)
    assert reg.coef_.shape == (1,)

    # Test with single sample
    X = np.array([[1.0, 2.0]])
    y = np.array([1.0])

    reg = SigmaRidgeRegressor()
    with pytest.raises(ValueError):  # Should raise error for insufficient samples
        reg.fit(X, y)


def test_sigma_ridge_input_validation():
    """Test input validation and error handling."""
    X, y = make_regression(n_samples=100, n_features=20, random_state=42)
    reg = SigmaRidgeRegressor()

    # Test unfitted predictor
    with pytest.raises(NotFittedError):
        reg.predict(X)

    # Test mismatched dimensions in predict
    reg.fit(X, y)
    with pytest.raises(ValueError):
        reg.predict(X[:, :10])

    # Test non-finite input
    X_inf = X.copy()
    X_inf[0, 0] = np.inf
    with pytest.raises(ValueError):
        reg.fit(X_inf, y)

    X_nan = X.copy()
    X_nan[0, 0] = np.nan
    with pytest.raises(ValueError):
        reg.fit(X_nan, y)

    # Test wrong dimensions
    with pytest.raises(ValueError):
        reg.fit(X.reshape(-1), y)

    # Test incompatible feature groups
    groups = [[0, 1], [2, 3]]  # Only covers 4 features
    reg = SigmaRidgeRegressor(feature_groups=groups)
    with pytest.raises(ValueError):
        reg.fit(X, y)  # X has 20 features


def test_sigma_ridge_decomposition_methods():
    """Test different decomposition methods."""
    # Test Cholesky (n_samples > n_features)
    X, y = make_regression(n_samples=100, n_features=20, random_state=42)

    reg = SigmaRidgeRegressor(decomposition="cholesky")
    reg.fit(X, y)
    y_pred_chol = reg.predict(X)

    # Test Woodbury (n_features > n_samples)
    X, y = make_regression(n_samples=20, n_features=100, random_state=42)

    reg = SigmaRidgeRegressor(decomposition="woodbury")
    reg.fit(X, y)
    y_pred_wood = reg.predict(X)

    # Test automatic selection
    reg = SigmaRidgeRegressor(decomposition="default")
    reg.fit(X, y)
    y_pred_auto = reg.predict(X)

    # Results should be similar to Woodbury for this case
    assert_array_almost_equal(y_pred_wood, y_pred_auto, decimal=10)


def test_sigma_ridge_get_set_params():
    """Test getting and setting parameters."""
    reg = SigmaRidgeRegressor(sigma=1.0, center=True)
    params = reg.get_params()

    assert params["sigma"] == 1.0
    assert params["center"] is True

    # Test setting parameters
    reg.set_params(sigma=2.0, center=False)
    assert reg.sigma == 2.0
    assert reg.center is False

    # Test setting invalid parameters
    with pytest.raises(ValueError):
        reg.set_params(invalid_param=1.0)
