import pytest
import numpy as np
from ..src.lambda_max_sparse_group_lasso import lambda_max_sparse_group_lasso


def test_lambda_max_sparse_group_lasso_basic():
    """Test basic functionality of lambda_max_sparse_group_lasso."""
    n, p = 100, 10
    alpha = 0.5
    vector_y = np.random.randn(n)
    vector_groups = np.repeat(np.arange(1, 6), 2)
    vector_weights_features = np.ones(p)
    vector_beta = np.zeros(p)
    matrix_x = np.random.randn(n, p)

    result = lambda_max_sparse_group_lasso(
        alpha, vector_y, vector_groups, vector_weights_features, vector_beta, matrix_x
    )
    assert result > 0


def test_lambda_max_sparse_group_lasso_unpenalized_features():
    """Test lambda_max_sparse_group_lasso with unpenalized features."""
    n, p = 100, 10
    alpha = 0.5
    vector_y = np.random.randn(n)
    vector_groups = np.repeat(np.arange(1, 6), 2)
    vector_weights_features = np.ones(p)
    vector_weights_features[:2] = 0  # First two features are unpenalized
    vector_beta = np.zeros(p)
    matrix_x = np.random.randn(n, p)

    result = lambda_max_sparse_group_lasso(
        alpha, vector_y, vector_groups, vector_weights_features, vector_beta, matrix_x
    )
    assert result > 0
    assert np.any(vector_beta[:2] != 0)  # Unpenalized features should be non-zero


def test_lambda_max_sparse_group_lasso_edge_cases():
    """Test lambda_max_sparse_group_lasso with edge case alpha values."""
    n, p = 100, 10
    vector_y = np.random.randn(n)
    vector_groups = np.repeat(np.arange(1, 6), 2)
    vector_weights_features = np.ones(p)
    vector_beta = np.zeros(p)
    matrix_x = np.random.randn(n, p)

    # Test with alpha close to 0 and 1
    result_low_alpha = lambda_max_sparse_group_lasso(
        0.001,
        vector_y,
        vector_groups,
        vector_weights_features,
        vector_beta.copy(),
        matrix_x,
    )
    result_high_alpha = lambda_max_sparse_group_lasso(
        0.999,
        vector_y,
        vector_groups,
        vector_weights_features,
        vector_beta.copy(),
        matrix_x,
    )

    assert result_low_alpha > 0
    assert result_high_alpha > 0
    assert result_low_alpha != result_high_alpha


def test_lambda_max_sparse_group_lasso_different_group_sizes():
    """Test lambda_max_sparse_group_lasso with varying group sizes."""
    n, p = 100, 15
    alpha = 0.5
    vector_y = np.random.randn(n)
    vector_groups = np.array([1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5])
    vector_weights_features = np.ones(p)
    vector_beta = np.zeros(p)
    matrix_x = np.random.randn(n, p)

    result = lambda_max_sparse_group_lasso(
        alpha, vector_y, vector_groups, vector_weights_features, vector_beta, matrix_x
    )
    assert result > 0


def test_lambda_max_sparse_group_lasso_zero_weight_group():
    """Test lambda_max_sparse_group_lasso with a group of zero weights."""
    n, p = 100, 10
    alpha = 0.5
    vector_y = np.random.randn(n)
    vector_groups = np.repeat(np.arange(1, 6), 2)
    vector_weights_features = np.ones(p)
    vector_weights_features[4:6] = 0  # Set weights for one group to zero
    vector_beta = np.zeros(p)
    matrix_x = np.random.randn(n, p)

    result = lambda_max_sparse_group_lasso(
        alpha, vector_y, vector_groups, vector_weights_features, vector_beta, matrix_x
    )
    assert result > 0


if __name__ == "__main__":
    pytest.main()
