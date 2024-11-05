"""Tests for scikit-learn compatible Ridge regression estimators."""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from sklearn.utils.estimator_checks import check_estimator
from sklearn.exceptions import NotFittedError
from sklearn.datasets import make_regression
from ..src.blockridge import (
    GroupRidgeRegressor,
    CholeskyRidgePredictor,
    WoodburyRidgePredictor,
    ShermanMorrisonRidgePredictor,
    MomentTunerSetup,
    lambda_lolas_rule,
    get_lambdas,
    get_alpha_s_squared,
    InvalidDimensionsError,
    SingularMatrixError,
    NumericalInstabilityError,
)
from ..src.groupedfeatures import GroupedFeatures


def generate_test_matrix(n_samples, n_features, seed=42):
    """Generate a test matrix with good conditioning."""
    np.random.seed(seed)
    # Create a matrix with controlled singular values
    U = np.random.randn(n_samples, min(n_samples, n_features))
    U, _ = np.linalg.qr(U)

    V = np.random.randn(n_features, min(n_samples, n_features))
    V, _ = np.linalg.qr(V)

    s = np.linspace(0.1, 1.0, min(n_samples, n_features))
    X = U @ np.diag(s) @ V.T
    X = X + 1e-4 * np.random.randn(n_samples, n_features)
    X = X / np.sqrt(n_features)
    return X


@pytest.fixture
def sample_data():
    """Generate sample regression data."""
    X, y = make_regression(
        n_samples=100, n_features=20, n_informative=10, random_state=42
    )
    return X, y


@pytest.fixture
def sample_groups():
    """Create sample feature groups."""
    groups = GroupedFeatures([5, 5, 5, 5])  # 4 groups of 5 features each
    return groups


@pytest.fixture
def fitted_model(sample_data, sample_groups):
    """Create a fitted GroupRidgeRegressor."""
    X, y = sample_data
    groups = sample_groups
    groups.fit(X)  # Fit the groups first
    model = GroupRidgeRegressor(groups=groups)
    return model.fit(X, y)


class DynamicGroupedFeatures(GroupedFeatures):
    """A GroupedFeatures class that adapts to input data dimensions."""

    def fit(self, X):
        n_features = X.shape[1]
        # Create equal-sized groups based on number of features
        if n_features == 1:
            self.ps = [1]  # Single feature, single group
        else:
            group_size = n_features // 2  # Split features into 2 groups
            remainder = n_features % 2
            self.ps = [group_size] * 2
            if remainder:
                self.ps[-1] += remainder
        return super().fit(X)


def test_scikit_learn_compatibility():
    """Test scikit-learn estimator compatibility."""
    # Create a dummy model with dynamic groups that will adapt to any input
    groups = DynamicGroupedFeatures([1, 1])  # Initial groups don't matter
    model = GroupRidgeRegressor(groups=groups)

    # Run the scikit-learn compatibility checks
    check_estimator(model)


class TestGroupRidgeRegressor:
    """Test GroupRidgeRegressor functionality."""

    def test_init(self, sample_groups):
        """Test initialization."""
        model = GroupRidgeRegressor(groups=sample_groups)
        assert model.groups == sample_groups
        assert model.alpha is None

        alpha = np.ones(4)
        model = GroupRidgeRegressor(groups=sample_groups, alpha=alpha)
        assert_array_equal(model.alpha, alpha)

    def test_validate_params(self, sample_groups):
        """Test parameter validation."""
        # Valid case
        model = GroupRidgeRegressor(groups=sample_groups)
        model._validate_params()

        # Invalid cases
        with pytest.raises(ValueError):
            empty_groups = GroupedFeatures([])
            model = GroupRidgeRegressor(groups=empty_groups)
            model.groups_ = empty_groups  # Simulate fit
            model._validate_params()

        with pytest.raises(ValueError):
            zero_groups = GroupedFeatures([0, 1, 2])
            model = GroupRidgeRegressor(groups=zero_groups)
            model.groups_ = zero_groups  # Simulate fit
            model._validate_params()

    def test_fit(self, sample_data, sample_groups):
        """Test model fitting."""
        X, y = sample_data
        sample_groups.fit(X)  # Fit groups first
        model = GroupRidgeRegressor(groups=sample_groups)

        # Test successful fit
        model.fit(X, y)
        assert hasattr(model, "coef_")
        assert hasattr(model, "n_features_in_")
        assert model.n_features_in_ == X.shape[1]
        assert hasattr(model, "y_")
        assert hasattr(model, "gram_reg_inv_")

        # Test predictor selection with small dimensions
        X_small = X[:, :10]
        small_groups = GroupedFeatures([5, 5])
        small_groups.fit(X_small)  # Fit groups with the data
        model_small = GroupRidgeRegressor(groups=small_groups)
        model_small.fit(X_small, y)
        assert isinstance(model_small.predictor_, CholeskyRidgePredictor)

        # Test with larger dimensions
        n_samples, n_features = 20, 100
        X_large = generate_test_matrix(n_samples, n_features)
        y_large = np.random.randn(n_samples)

        large_groups = GroupedFeatures([50, 50])
        large_groups.fit(X_large)  # Fit groups with the data
        # Use stronger regularization
        model_large = GroupRidgeRegressor(
            groups=large_groups, alpha=np.array([100.0, 100.0])
        )
        model_large.fit(X_large, y_large)
        assert isinstance(model_large.predictor_, ShermanMorrisonRidgePredictor)

    def test_predict(self, fitted_model, sample_data):
        """Test prediction functionality."""
        X, _ = sample_data

        # Test prediction shape
        y_pred = fitted_model.predict(X)
        assert y_pred.shape == (X.shape[0],)

        # Test prediction before fit
        model = GroupRidgeRegressor(groups=sample_groups)
        with pytest.raises(AttributeError):
            model.predict(X)

    def test_get_set_params(self, sample_groups):
        """Test parameter getting and setting."""
        model = GroupRidgeRegressor(groups=sample_groups)
        params = model.get_params()
        assert "groups" in params
        assert "alpha" in params

        new_groups = GroupedFeatures([3, 3])
        new_groups.fit(np.zeros((1, 6)))  # Fit with dummy data
        model.set_params(groups=new_groups)
        assert model.groups == new_groups

    def test_error_metrics(self, fitted_model, sample_data):
        """Test error metric calculations."""
        X, y = sample_data
        X_test = X[80:, :]
        y_test = y[80:]

        mse = fitted_model.get_mse(X_test, y_test)
        assert isinstance(mse, float)
        assert mse >= 0

        loo_error = fitted_model.get_loo_error()
        assert isinstance(loo_error, float)
        assert loo_error >= 0


class TestMomentTuner:
    """Test moment-based tuning functionality."""

    def test_moment_tuner_setup(self, fitted_model):
        """Test MomentTunerSetup initialization and attributes."""
        tuner = MomentTunerSetup(fitted_model)

        assert hasattr(tuner, "groups_")
        assert hasattr(tuner, "n_features_per_group_")
        assert hasattr(tuner, "moment_matrix_")

        # Test shape consistency
        n_groups = len(tuner.n_features_per_group_)
        assert tuner.moment_matrix_.shape == (n_groups, n_groups)
        assert tuner.coef_norms_squared_.shape == (n_groups,)

    def test_get_lambdas(self, fitted_model):
        """Test lambda parameter computation."""
        tuner = MomentTunerSetup(fitted_model)
        sigma_sq = 0.1

        lambdas = get_lambdas(tuner, sigma_sq)
        assert len(lambdas) == len(tuner.n_features_per_group_)
        assert np.all(lambdas >= 0)

        # Test error cases
        with pytest.raises(ValueError):
            get_lambdas(tuner, -1.0)

    def test_get_alpha_squared(self, fitted_model):
        """Test alpha squared computation."""
        tuner = MomentTunerSetup(fitted_model)
        sigma_sq = 0.1

        alpha_squared = get_alpha_s_squared(tuner, sigma_sq)
        assert len(alpha_squared) == len(tuner.n_features_per_group_)
        assert np.all(alpha_squared >= 0)

        # Test error cases
        with pytest.raises(ValueError):
            get_alpha_s_squared(tuner, -1.0)


def test_lambda_lolas_rule(fitted_model):
    """Test Lolas rule for lambda selection."""
    lambda_val = lambda_lolas_rule(fitted_model)
    assert isinstance(lambda_val, float)
    assert lambda_val > 0

    # Test error cases
    with pytest.raises(ValueError):
        lambda_lolas_rule(fitted_model, multiplier=-1.0)


def test_edge_cases(sample_groups):
    """Test various edge cases and error conditions."""
    # Test with invalid input dimensions
    X_1d = np.random.randn(10)
    y_1d = np.random.randn(10)
    model = GroupRidgeRegressor(groups=sample_groups)

    with pytest.raises(ValueError):
        model.fit(X_1d, y_1d)

    # Test with NaN/inf values
    X_nan = np.random.randn(10, 20)
    X_nan[0, 0] = np.nan
    y_nan = np.random.randn(10)

    with pytest.raises(ValueError):
        model.fit(X_nan, y_nan)

    # Test with mismatched dimensions
    X = np.random.randn(10, 20)
    y_mismatched = np.random.randn(15)

    with pytest.raises(ValueError):
        model.fit(X, y_mismatched)

    # Test with singular matrix
    X_singular = np.zeros((10, 20))
    X_singular[:, 0] = 1
    X_singular[:, 1] = X_singular[:, 0]
    y = np.random.randn(10)

    model_singular = GroupRidgeRegressor(groups=sample_groups)
    with pytest.raises((SingularMatrixError, np.linalg.LinAlgError)):
        model_singular.fit(X_singular, y)


def test_numerical_stability():
    """Test numerical stability in extreme cases."""
    # Test with very large values
    X_large = np.random.randn(10, 5) * 1e10
    y_large = np.random.randn(10) * 1e10
    groups = GroupedFeatures([2, 3])
    groups.fit(X_large)  # Fit groups with the data
    model = GroupRidgeRegressor(groups=groups)

    # This should still work despite large values
    model.fit(X_large, y_large)

    # Test with very small values
    X_small = np.random.randn(10, 5) * 1e-10
    y_small = np.random.randn(10) * 1e-10
    groups.fit(X_small)  # Fit groups with the data

    # This should still work despite small values
    model.fit(X_small, y_small)
