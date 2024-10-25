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
    # Fit with dummy data to initialize internal state
    groups.fit(np.zeros((1, 20)))
    return groups


@pytest.fixture
def fitted_model(sample_data, sample_groups):
    """Create a fitted GroupRidgeRegressor."""
    X, y = sample_data
    model = GroupRidgeRegressor(groups=sample_groups)
    return model.fit(X, y)


def test_scikit_learn_compatibility():
    """Test scikit-learn estimator compatibility."""
    # Create data with matching dimensions
    n_samples, n_features = 10, 5
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)

    # Create and fit groups
    groups = GroupedFeatures([n_features])  # One group with all features
    groups.fit(X)  # Fit groups with the data
    alpha = np.array([1.0], dtype=float)

    # Create and fit model
    model = GroupRidgeRegressor(groups=groups, alpha=alpha)
    model.fit(X, y)

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
            empty_groups.fit(np.zeros((1, 0)))  # Fit with empty data
            GroupRidgeRegressor(groups=empty_groups)._validate_params()

        with pytest.raises(ValueError):
            zero_groups = GroupedFeatures([0, 1, 2])
            GroupRidgeRegressor(groups=zero_groups)._validate_params()

    def test_fit(self, sample_data, sample_groups):
        """Test model fitting."""
        X, y = sample_data
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
        n_samples, n_features = 20, 10
        X_large = np.random.randn(n_samples, n_features)
        y_large = np.random.randn(n_samples)
        large_groups = GroupedFeatures([5, 5])
        large_groups.fit(X_large)  # Fit groups with the data
        model_large = GroupRidgeRegressor(groups=large_groups)
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


class TestRidgePredictors:
    """Test individual Ridge predictor implementations."""

    @pytest.fixture
    def sample_matrices(self):
        """Generate sample matrices for testing."""
        X = np.random.randn(10, 5)
        groups = GroupedFeatures([2, 3])
        groups.fit(X)  # Fit groups with the data
        alpha = np.array([0.1, 0.2])
        return X, groups, alpha

    def test_cholesky_predictor(self, sample_matrices):
        """Test CholeskyRidgePredictor."""
        X, groups, alpha = sample_matrices
        predictor = CholeskyRidgePredictor(X)

        # Test initialization
        assert predictor.n_samples_ == X.shape[0]
        assert predictor.n_features_ == X.shape[1]
        assert predictor.lower_ is True

        # Test parameter updates
        predictor.set_params(groups, alpha)
        assert hasattr(predictor, "gram_reg_chol_")

        # Test error cases
        with pytest.raises(InvalidDimensionsError):
            CholeskyRidgePredictor(X.flatten())

        # Test with singular matrix
        X_singular = np.zeros((10, 5))
        X_singular[:, 0] = 1
        X_singular[:, 1] = X_singular[:, 0]
        with pytest.raises(SingularMatrixError):
            predictor_singular = CholeskyRidgePredictor(X_singular)

    def test_woodbury_predictor(self, sample_matrices):
        """Test WoodburyRidgePredictor."""
        X, groups, alpha = sample_matrices
        predictor = WoodburyRidgePredictor(X)

        # Test initialization
        assert predictor.n_samples_ == X.shape[0]
        assert predictor.n_features_ == X.shape[1]
        assert hasattr(predictor, "alpha_inv_")

        # Test parameter updates
        predictor.set_params(groups, alpha)

        # Test error cases
        with pytest.raises(ValueError):
            predictor.set_params(groups, np.array([-1.0, 0.1]))

    def test_sherman_morrison_predictor(self, sample_matrices):
        """Test ShermanMorrisonRidgePredictor."""
        X, groups, alpha = sample_matrices
        predictor = ShermanMorrisonRidgePredictor(X)

        # Test initialization
        assert predictor.n_samples_ == X.shape[0]
        assert predictor.n_features_ == X.shape[1]
        assert hasattr(predictor, "A_inv_")

        # Test parameter updates
        predictor.set_params(groups, alpha)

        # Test Sherman-Morrison formula
        u = np.random.randn(5)
        v = np.random.randn(5)
        result = predictor._sherman_morrison_formula(np.eye(5), u, v)
        assert result.shape == (5, 5)


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
