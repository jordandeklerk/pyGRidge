import numpy as np
import pytest
from ..src.lambda_max_lasso import lambda_max_lasso


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n, p = 100, 10
    y = np.random.randn(n)
    feature_weights = np.random.rand(p)
    beta = np.zeros(p)
    X = np.random.randn(n, p)
    return y, feature_weights, beta, X


def test_lambda_max_lasso_output(sample_data):
    """Test that lambda_max_lasso returns a positive float."""
    y, feature_weights, beta, X = sample_data
    result = lambda_max_lasso(y, feature_weights, beta, X)
    assert isinstance(result, float)
    assert result > 0


def test_lambda_max_lasso_with_zero_weights(sample_data):
    """Test lambda_max_lasso with some feature weights set to zero."""
    y, feature_weights, beta, X = sample_data
    feature_weights[:5] = 0  # Set first 5 weights to zero
    result = lambda_max_lasso(y, feature_weights, beta, X)
    assert isinstance(result, float)
    assert result > 0


def test_lambda_max_lasso_all_penalized():
    """Test lambda_max_lasso when all features are penalized."""
    n, p = 50, 5
    y = np.random.randn(n)
    feature_weights = np.ones(p)
    beta = np.zeros(p)
    X = np.random.randn(n, p)
    result = lambda_max_lasso(y, feature_weights, beta, X)
    assert isinstance(result, float)
    assert result > 0


def test_lambda_max_lasso_dimension_mismatch():
    """Test lambda_max_lasso raises ValueError for dimension mismatch."""
    n, p = 50, 5
    y = np.random.randn(n)
    feature_weights = np.ones(p)
    beta = np.zeros(p)
    X = np.random.randn(n, p + 1)  # Mismatched dimension
    with pytest.raises(ValueError):
        lambda_max_lasso(y, feature_weights, beta, X)


def test_lambda_max_lasso_numeric_correction():
    """Test that lambda_max_lasso result is corrected numerically."""
    n, p = 50, 5
    y = np.random.randn(n)
    feature_weights = np.ones(p)
    beta = np.zeros(p)
    X = np.random.randn(n, p)
    result = lambda_max_lasso(y, feature_weights, beta, X)
    uncorrected = result / 1.00001
    assert result > uncorrected
    assert np.isclose(result, uncorrected * 1.00001)
