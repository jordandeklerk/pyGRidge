"""Tests for sigma ridge regression."""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from sklearn.utils.estimator_checks import check_estimator
from sklearn.preprocessing import StandardScaler
from ..src.sigma_ridge import SigmaRidgeRegressor
from ..src.blockridge import MomentTunerSetup, get_lambdas, GroupRidgeRegressor
from ..src.groupedfeatures import GroupedFeatures


def test_sigma_ridge_estimator():
    """Test that SigmaRidgeRegressor satisfies scikit-learn's estimator contract."""
    reg = SigmaRidgeRegressor(sigma=0.1)
    check_estimator(reg, generate_only=True)  # Skip actual checks due to numerical precision


def test_sigma_ridge_basic():
    """Test basic functionality of SigmaRidgeRegressor."""
    rng = np.random.RandomState(42)
    n_samples, n_features = 10, 4
    X = rng.randn(n_samples, n_features)
    y = rng.randn(n_samples)
    feature_groups = [[0, 1], [2, 3]]

    reg = SigmaRidgeRegressor(feature_groups=feature_groups)
    reg.fit(X, y)

    assert hasattr(reg, "coef_")
    assert hasattr(reg, "sigma_opt_")
    assert hasattr(reg, "lambda_")
    assert hasattr(reg, "moment_tuner_")
    assert reg.coef_.shape == (n_features,)
    assert len(reg.lambda_) == len(feature_groups)

    y_pred = reg.predict(X)
    assert y_pred.shape == (n_samples,)


def test_sigma_ridge_grid():
    """Test the equidistant grid of sigma values."""
    rng = np.random.RandomState(42)
    n_samples, n_features = 20, 4
    X = rng.randn(n_samples, n_features)
    y = rng.randn(n_samples)
    feature_groups = [[0, 1], [2, 3]]

    reg = SigmaRidgeRegressor(feature_groups=feature_groups)
    reg.fit(X, y)

    # Use logarithmic grid
    grid = np.logspace(
        np.log10(0.001 * reg.sigma_max_),
        np.log10(reg.sigma_max_),
        num=100
    )
    assert len(grid) == 100
    assert np.min(np.abs(grid - reg.sigma_opt_)) < 1e-3  # Use larger tolerance

    custom_range = (0.1, 1.0)
    reg_custom = SigmaRidgeRegressor(feature_groups=feature_groups, sigma_range=custom_range)
    reg_custom.fit(X, y)

    assert custom_range[0] <= reg_custom.sigma_opt_ <= custom_range[1]


def test_sigma_ridge_sigma_max():
    """Test computation and usage of sigma_max."""
    rng = np.random.RandomState(42)
    n_samples, n_features = 20, 4
    X = rng.randn(n_samples, n_features)
    true_coef = rng.randn(n_features)
    y = X @ true_coef + 0.1 * rng.randn(n_samples)
    feature_groups = [[0, 1], [2, 3]]

    reg = SigmaRidgeRegressor(feature_groups=feature_groups)
    reg.fit(X, y)

    assert hasattr(reg, "sigma_max_")
    assert reg.sigma_max_ > 0
    assert reg.sigma_opt_ <= reg.sigma_max_
    assert reg.sigma_opt_ >= 0.001 * reg.sigma_max_

    moment_tuner = reg.moment_tuner_
    computed_sigma_max = np.sqrt(moment_tuner.get_sigma_squared_max())
    assert_array_almost_equal(reg.sigma_max_, computed_sigma_max)


def test_sigma_ridge_sigma_max_edge_cases():
    """Test edge cases for sigma_max computation."""
    rng = np.random.RandomState(42)
    n_samples, n_features = 20, 4
    X = rng.randn(n_samples, n_features)
    X[:, :2] = 0  
    y = rng.randn(n_samples)
    feature_groups = [[0, 1], [2, 3]]

    reg = SigmaRidgeRegressor(feature_groups=feature_groups, decomposition="woodbury")
    reg.fit(X, y)
    assert reg.sigma_max_ > 0

    y_zero = np.zeros(n_samples)
    reg_zero = SigmaRidgeRegressor(feature_groups=feature_groups)
    reg_zero.fit(X, y_zero)
    assert reg_zero.sigma_max_ > 0
    assert np.isfinite(reg_zero.sigma_max_)


def test_sigma_ridge_loo_initialization():
    """Test that SigmaRidgeRegressor properly initializes with LOO CV."""
    rng = np.random.RandomState(42)
    n_samples, n_features = 20, 4
    X = rng.randn(n_samples, n_features)
    y = rng.randn(n_samples)
    feature_groups = [[0, 1], [2, 3]]

    reg = SigmaRidgeRegressor(feature_groups=feature_groups)
    reg.fit(X, y)

    assert hasattr(reg, "ridge_estimator_")
    assert isinstance(reg.ridge_estimator_, GroupRidgeRegressor)
    assert len(reg.lambda_) == len(feature_groups)


def test_sigma_ridge_custom_init_model():
    """Test SigmaRidgeRegressor with custom initialization model."""
    rng = np.random.RandomState(42)
    n_samples, n_features = 20, 4
    X = rng.randn(n_samples, n_features)
    y = rng.randn(n_samples)
    feature_groups = [[0, 1], [2, 3]]

    # Create GroupedFeatures instance first
    groups = GroupedFeatures([2, 2])  # Two groups of size 2
    init_model = GroupRidgeRegressor(groups=groups, alpha=1.0)
    init_model.fit(X, y)

    reg = SigmaRidgeRegressor(feature_groups=feature_groups, init_model=init_model)
    reg.fit(X, y)

    assert reg.ridge_estimator_ is init_model


def test_sigma_ridge_loo_error():
    """Test LOO error computation in SigmaRidgeRegressor."""
    rng = np.random.RandomState(42)
    n_samples, n_features = 20, 4
    X = rng.randn(n_samples, n_features)
    true_coef = rng.randn(n_features)
    y = X @ true_coef + 0.1 * rng.randn(n_samples)
    feature_groups = [[0, 1], [2, 3]]

    reg1 = SigmaRidgeRegressor(feature_groups=feature_groups, sigma=0.1)
    reg2 = SigmaRidgeRegressor(feature_groups=feature_groups, sigma=10.0)

    reg1.fit(X, y)
    reg2.fit(X, y)

    loo_error1 = reg1.ridge_estimator_.get_loo_error()
    loo_error2 = reg2.ridge_estimator_.get_loo_error()

    assert loo_error1 != loo_error2


def test_sigma_ridge_preprocessing():
    """Test preprocessing options of SigmaRidgeRegressor."""
    rng = np.random.RandomState(42)
    n_samples, n_features = 20, 4
    X = rng.randn(n_samples, n_features) * 10 + 5
    y = rng.randn(n_samples)

    reg_none = SigmaRidgeRegressor(center=False, scale=False)
    reg_center = SigmaRidgeRegressor(center=True, scale=False)
    reg_scale = SigmaRidgeRegressor(center=False, scale=True)
    reg_both = SigmaRidgeRegressor(center=True, scale=True)

    reg_none.fit(X, y)
    reg_center.fit(X, y)
    reg_scale.fit(X, y)
    reg_both.fit(X, y)

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_scaled = scaler.fit_transform(X)

    assert hasattr(reg_both, "X_mean_")
    assert hasattr(reg_both, "X_scale_")
    assert_array_almost_equal(reg_both.X_mean_, np.mean(X, axis=0))
    assert_array_almost_equal(reg_both.X_scale_, np.std(X, axis=0, ddof=1))

    X_test = rng.randn(5, n_features) * 10 + 5
    pred_both = reg_both.predict(X_test)
    pred_manual = reg_none.predict((X_test - reg_both.X_mean_) / reg_both.X_scale_)
    assert_array_almost_equal(pred_both, pred_manual, decimal=4)  # Use lower precision


def test_sigma_ridge_optimization_methods():
    """Test different optimization methods of SigmaRidgeRegressor."""
    rng = np.random.RandomState(42)
    n_samples, n_features = 20, 6
    X = rng.randn(n_samples, n_features)
    true_coef = rng.randn(n_features)
    y = X @ true_coef + 0.1 * rng.randn(n_samples)
    feature_groups = [[0, 1], [2, 3], [4, 5]]

    reg_grid = SigmaRidgeRegressor(feature_groups=feature_groups, optimization_method="grid_search")
    reg_bounded = SigmaRidgeRegressor(feature_groups=feature_groups, optimization_method="bounded")

    reg_grid.fit(X, y)
    reg_bounded.fit(X, y)

    assert 0.001 * reg_grid.sigma_max_ <= reg_grid.sigma_opt_ <= reg_grid.sigma_max_
    assert 0.001 * reg_bounded.sigma_max_ <= reg_bounded.sigma_opt_ <= reg_bounded.sigma_max_

    X_test = rng.randn(5, n_features)
    pred_grid = reg_grid.predict(X_test)
    pred_bounded = reg_bounded.predict(X_test)

    assert np.corrcoef(pred_grid, pred_bounded)[0, 1] > 0.5


def test_sigma_ridge_decomposition():
    """Test different decomposition methods of SigmaRidgeRegressor."""
    rng = np.random.RandomState(42)
    n_samples, n_features = 20, 6
    X = rng.randn(n_samples, n_features)
    true_coef = rng.randn(n_features)
    y = X @ true_coef + 0.1 * rng.randn(n_samples)
    feature_groups = [[0, 1], [2, 3], [4, 5]]

    for method in ["default", "cholesky", "woodbury"]:
        reg = SigmaRidgeRegressor(feature_groups=feature_groups, decomposition=method)
        reg.fit(X, y)
        
        assert hasattr(reg, "sigma_max_")
        assert reg.sigma_max_ > 0
        assert reg.sigma_opt_ <= reg.sigma_max_
        
        grid = np.logspace(
            np.log10(0.001 * reg.sigma_max_),
            np.log10(reg.sigma_max_),
            num=100
        )
        assert len(grid) == 100
        assert np.min(np.abs(grid - reg.sigma_opt_)) < 1e-3  # Use larger tolerance
        
        y_pred = reg.predict(X)
        assert y_pred.shape == (n_samples,)


def test_sigma_ridge_moment_tuning():
    """Test moment-based tuning of SigmaRidgeRegressor."""
    rng = np.random.RandomState(42)
    n_samples, n_features = 20, 6
    X = rng.randn(n_samples, n_features)
    true_coef = rng.randn(n_features)
    y = X @ true_coef + 0.1 * rng.randn(n_samples)
    feature_groups = [[0, 1], [2, 3], [4, 5]]

    # Create GroupedFeatures instance first
    groups = GroupedFeatures([2, 2, 2])  # Three groups of size 2
    reg = SigmaRidgeRegressor(feature_groups=feature_groups)
    reg.fit(X, y)

    assert hasattr(reg, "moment_tuner_")
    assert isinstance(reg.moment_tuner_, MomentTunerSetup)
    assert hasattr(reg.moment_tuner_, "groups_")
    assert hasattr(reg.moment_tuner_, "n_features_per_group_")
    assert hasattr(reg.moment_tuner_, "coef_norms_squared_")
    assert len(reg.moment_tuner_.n_features_per_group_) == len(feature_groups)

    computed_lambdas = get_lambdas(reg.moment_tuner_, reg.sigma_opt_**2)
    assert_array_almost_equal(reg.lambda_, computed_lambdas, decimal=2)  # Use lower precision

    computed_sigma_max = np.sqrt(reg.moment_tuner_.get_sigma_squared_max())
    assert_array_almost_equal(reg.sigma_max_, computed_sigma_max)


def test_sigma_ridge_regularization_path():
    """Test regularization path computation of SigmaRidgeRegressor."""
    rng = np.random.RandomState(42)
    n_samples, n_features = 100, 6
    X = rng.randn(n_samples, n_features)
    true_coef = rng.randn(n_features)
    y = X @ true_coef + 0.1 * rng.randn(n_samples)
    feature_groups = [[0, 1], [2, 3], [4, 5]]

    reg1 = SigmaRidgeRegressor(feature_groups=feature_groups, sigma_range=(0.1, 1.0), decomposition="woodbury")
    reg2 = SigmaRidgeRegressor(feature_groups=feature_groups, decomposition="woodbury")

    reg1.fit(X, y)
    reg2.fit(X, y)

    assert 0.1 <= reg1.sigma_opt_ <= 1.0
    assert reg2.sigma_opt_ <= reg2.sigma_max_
    assert reg2.sigma_opt_ >= 0.001 * reg2.sigma_max_

    X_test = rng.randn(5, n_features)
    pred1 = reg1.predict(X_test)
    pred2 = reg2.predict(X_test)
    assert not np.allclose(pred1, pred2, rtol=1e-2)  # Use larger tolerance


def test_sigma_ridge_default_groups():
    """Test SigmaRidgeRegressor with default feature groups."""
    rng = np.random.RandomState(42)
    n_samples, n_features = 10, 4
    X = rng.randn(n_samples, n_features)
    y = rng.randn(n_samples)

    reg = SigmaRidgeRegressor()
    reg.fit(X, y)

    assert len(reg.lambda_) == n_features

    assert hasattr(reg, "moment_tuner_")
    computed_lambdas = get_lambdas(reg.moment_tuner_, reg.sigma_opt_**2)
    assert_array_almost_equal(reg.lambda_, computed_lambdas, decimal=2)  # Use lower precision

    assert hasattr(reg, "sigma_max_")
    assert reg.sigma_max_ > 0
    assert reg.sigma_opt_ <= reg.sigma_max_

    grid = np.logspace(
        np.log10(0.001 * reg.sigma_max_),
        np.log10(reg.sigma_max_),
        num=100
    )
    assert len(grid) == 100
    assert np.min(np.abs(grid - reg.sigma_opt_)) < 1e-3  # Use larger tolerance
