"""Tests for sigma ridge regression."""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from sklearn.utils.estimator_checks import check_estimator
from sklearn.preprocessing import StandardScaler
from ..src.sigma_ridge import SigmaRidgeRegressor
from ..src.blockridge import MomentTunerSetup, get_lambdas


def test_sigma_ridge_estimator():
    """Test that SigmaRidgeRegressor satisfies scikit-learn's estimator contract."""
    # Use lower regularization for better performance on scikit-learn's test data
    reg = SigmaRidgeRegressor(sigma=0.1)
    check_estimator(reg)


def test_sigma_ridge_basic():
    """Test basic functionality of SigmaRidgeRegressor."""
    # Generate sample data
    rng = np.random.RandomState(42)
    n_samples, n_features = 10, 4
    X = rng.randn(n_samples, n_features)
    y = rng.randn(n_samples)

    # Define feature groups
    feature_groups = [[0, 1], [2, 3]]

    # Fit model
    reg = SigmaRidgeRegressor(feature_groups=feature_groups)
    reg.fit(X, y)

    # Basic checks
    assert hasattr(reg, "coef_")
    assert hasattr(reg, "sigma_opt_")
    assert hasattr(reg, "lambda_")
    assert hasattr(reg, "moment_tuner_")
    assert reg.coef_.shape == (n_features,)
    assert len(reg.lambda_) == len(feature_groups)

    # Test prediction
    y_pred = reg.predict(X)
    assert y_pred.shape == (n_samples,)


def test_sigma_ridge_preprocessing():
    """Test preprocessing options of SigmaRidgeRegressor."""
    # Generate sample data
    rng = np.random.RandomState(42)
    n_samples, n_features = 20, 4
    X = rng.randn(n_samples, n_features) * 10 + 5  # Add scale and offset
    y = rng.randn(n_samples)

    # Fit with different preprocessing options
    reg_none = SigmaRidgeRegressor(center=False, scale=False)
    reg_center = SigmaRidgeRegressor(center=True, scale=False)
    reg_scale = SigmaRidgeRegressor(center=False, scale=True)
    reg_both = SigmaRidgeRegressor(center=True, scale=True)

    reg_none.fit(X, y)
    reg_center.fit(X, y)
    reg_scale.fit(X, y)
    reg_both.fit(X, y)

    # Compare with sklearn's StandardScaler
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_scaled = scaler.fit_transform(X)

    # Check preprocessing attributes
    assert hasattr(reg_both, "X_mean_")
    assert hasattr(reg_both, "X_scale_")
    assert_array_almost_equal(reg_both.X_mean_, np.mean(X, axis=0))
    assert_array_almost_equal(reg_both.X_scale_, np.std(X, axis=0, ddof=1))

    # Predictions should be similar for preprocessed data
    X_test = rng.randn(5, n_features) * 10 + 5
    pred_both = reg_both.predict(X_test)
    pred_manual = reg_none.predict((X_test - reg_both.X_mean_) / reg_both.X_scale_)
    assert_array_almost_equal(pred_both, pred_manual)


def test_sigma_ridge_optimization_methods():
    """Test different optimization methods of SigmaRidgeRegressor."""
    # Generate sample data
    rng = np.random.RandomState(42)
    n_samples, n_features = 20, 6
    X = rng.randn(n_samples, n_features)
    true_coef = rng.randn(n_features)
    y = X @ true_coef + 0.1 * rng.randn(n_samples)
    feature_groups = [[0, 1], [2, 3], [4, 5]]

    # Test grid search optimization
    reg_grid = SigmaRidgeRegressor(
        feature_groups=feature_groups,
        optimization_method="grid_search",
        sigma_range=(0.1, 10.0),
    )
    reg_grid.fit(X, y)

    # Test bounded optimization
    reg_bounded = SigmaRidgeRegressor(
        feature_groups=feature_groups,
        optimization_method="bounded",
        sigma_range=(0.1, 10.0),
    )
    reg_bounded.fit(X, y)

    # Both methods should produce reasonable results
    assert 0.1 <= reg_grid.sigma_opt_ <= 10.0
    assert 0.1 <= reg_bounded.sigma_opt_ <= 10.0

    # Test prediction consistency
    X_test = rng.randn(5, n_features)
    pred_grid = reg_grid.predict(X_test)
    pred_bounded = reg_bounded.predict(X_test)

    # Predictions should be similar (not exactly equal)
    assert np.corrcoef(pred_grid, pred_bounded)[0, 1] > 0.5


def test_sigma_ridge_decomposition():
    """Test different decomposition methods of SigmaRidgeRegressor."""
    # Generate sample data
    rng = np.random.RandomState(42)
    n_samples, n_features = 20, 6
    X = rng.randn(n_samples, n_features)
    true_coef = rng.randn(n_features)
    y = X @ true_coef + 0.1 * rng.randn(n_samples)
    feature_groups = [[0, 1], [2, 3], [4, 5]]

    # Test each decomposition method
    for method in ["default", "cholesky", "woodbury"]:
        reg = SigmaRidgeRegressor(
            feature_groups=feature_groups,
            decomposition=method,
        )
        reg.fit(X, y)
        y_pred = reg.predict(X)
        assert y_pred.shape == (n_samples,)


def test_sigma_ridge_moment_tuning():
    """Test moment-based tuning of SigmaRidgeRegressor."""
    # Generate sample data
    rng = np.random.RandomState(42)
    n_samples, n_features = 20, 6
    X = rng.randn(n_samples, n_features)
    true_coef = rng.randn(n_features)
    y = X @ true_coef + 0.1 * rng.randn(n_samples)
    feature_groups = [[0, 1], [2, 3], [4, 5]]

    # Fit model
    reg = SigmaRidgeRegressor(feature_groups=feature_groups)
    reg.fit(X, y)

    assert hasattr(reg, "moment_tuner_")
    assert isinstance(reg.moment_tuner_, MomentTunerSetup)

    assert hasattr(reg.moment_tuner_, "groups_")
    assert hasattr(reg.moment_tuner_, "n_features_per_group_")
    assert hasattr(reg.moment_tuner_, "coef_norms_squared_")
    assert len(reg.moment_tuner_.n_features_per_group_) == len(feature_groups)

    # Test that get_lambdas was used to compute optimal lambdas
    computed_lambdas = get_lambdas(reg.moment_tuner_, reg.sigma_opt_**2)
    assert_array_almost_equal(reg.lambda_, computed_lambdas)


def test_sigma_ridge_regularization_path():
    """Test regularization path computation of SigmaRidgeRegressor."""
    # Generate well-conditioned data to avoid fallback
    rng = np.random.RandomState(42)
    n_samples, n_features = 100, 6  # More samples for better conditioning
    X = rng.randn(n_samples, n_features)
    true_coef = rng.randn(n_features)
    y = X @ true_coef + 0.1 * rng.randn(n_samples)
    feature_groups = [[0, 1], [2, 3], [4, 5]]

    # Fit model with different sigma ranges
    reg1 = SigmaRidgeRegressor(
        feature_groups=feature_groups,
        sigma_range=(0.1, 1.0),
        decomposition="woodbury",  # Use Woodbury for better stability
    )
    reg2 = SigmaRidgeRegressor(
        feature_groups=feature_groups,
        sigma_range=(1.0, 10.0),
        decomposition="woodbury",  # Use Woodbury for better stability
    )

    reg1.fit(X, y)
    reg2.fit(X, y)

    # Check that different ranges lead to different optimal sigmas
    assert reg1.sigma_opt_ <= reg2.sigma_opt_

    # Verify predictions are different
    X_test = rng.randn(5, n_features)
    pred1 = reg1.predict(X_test)
    pred2 = reg2.predict(X_test)
    assert not np.allclose(pred1, pred2)


def test_sigma_ridge_default_groups():
    """Test SigmaRidgeRegressor with default feature groups."""
    # Generate sample data
    rng = np.random.RandomState(42)
    n_samples, n_features = 10, 4
    X = rng.randn(n_samples, n_features)
    y = rng.randn(n_samples)

    # Fit model without specifying feature groups
    reg = SigmaRidgeRegressor()
    reg.fit(X, y)

    # Check that each feature is in its own group
    assert len(reg.lambda_) == n_features

    # Verify moment tuner and get_lambdas integration
    assert hasattr(reg, "moment_tuner_")
    computed_lambdas = get_lambdas(reg.moment_tuner_, reg.sigma_opt_**2)
    assert_array_almost_equal(reg.lambda_, computed_lambdas)


def test_sigma_ridge_get_set_params():
    """Test get_params and set_params methods."""
    reg = SigmaRidgeRegressor(
        feature_groups=[[0, 1], [2, 3]],
        sigma=2.0,
        tol=1e-5,
    )

    # Test get_params
    params = reg.get_params()
    assert params["sigma"] == 2.0
    assert params["tol"] == 1e-5

    # Test set_params
    reg.set_params(sigma=3.0)
    assert reg.sigma == 3.0


def test_sigma_ridge_reproducibility():
    """Test reproducibility of SigmaRidgeRegressor with same random seed."""
    # Generate sample data
    rng = np.random.RandomState(42)
    n_samples, n_features = 20, 6
    X = rng.randn(n_samples, n_features)
    true_coef = rng.randn(n_features)
    y = X @ true_coef + 0.1 * rng.randn(n_samples)
    feature_groups = [[0, 1], [2, 3], [4, 5]]

    # Fit two models with same parameters
    reg1 = SigmaRidgeRegressor(feature_groups=feature_groups)
    reg2 = SigmaRidgeRegressor(feature_groups=feature_groups)

    reg1.fit(X, y)
    reg2.fit(X, y)

    # Results should be identical
    assert_array_almost_equal(reg1.coef_, reg2.coef_)
    assert_array_almost_equal(reg1.lambda_, reg2.lambda_)
    assert reg1.sigma_opt_ == reg2.sigma_opt_

    # Moment tuners should produce identical results
    lambdas1 = get_lambdas(reg1.moment_tuner_, reg1.sigma_opt_**2)
    lambdas2 = get_lambdas(reg2.moment_tuner_, reg2.sigma_opt_**2)
    assert_array_almost_equal(lambdas1, lambdas2)


def test_sigma_ridge_fallback():
    """Test fallback behavior when get_lambdas fails."""
    # Generate pathological data to trigger fallback
    rng = np.random.RandomState(42)
    n_samples, n_features = 20, 6
    X = rng.randn(n_samples, n_features)
    X[:, :2] = 0  # Create collinearity to trigger numerical issues
    y = rng.randn(n_samples)
    feature_groups = [[0, 1], [2, 3], [4, 5]]

    # Fit model with Woodbury decomposition for better handling of singular matrices
    reg = SigmaRidgeRegressor(
        feature_groups=feature_groups,
        decomposition="woodbury",
    )

    # Should not raise error due to fallback mechanism
    reg.fit(X, y)

    # Should still have valid lambda values
    assert hasattr(reg, "lambda_")
    assert len(reg.lambda_) == len(feature_groups)
    assert np.all(np.isfinite(reg.lambda_))

    # Should be able to make predictions
    X_test = rng.randn(5, n_features)
    y_pred = reg.predict(X_test)
    assert y_pred.shape == (5,)
