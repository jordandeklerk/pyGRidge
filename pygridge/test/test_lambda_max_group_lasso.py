import numpy as np
import pytest
from ..src.lambda_max_group_lasso import lambda_max_group_lasso


@pytest.fixture
def sample_data():
    return {
        "y": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        "groups": np.array([1, 1, 2, 2, 3]),
        "feature_weights": np.array([0.5, 0.5, 1.0, 1.0, 2.0]),
        "beta": np.zeros(5),
        "X": np.eye(5),
    }


def test_basic_functionality(sample_data):
    """Test basic functionality of lambda_max_group_lasso."""
    result = lambda_max_group_lasso(**sample_data)
    assert result > 0, "Lambda max should be positive"
    assert result < 10, "Lambda max should be less than 10 for this sample data"


def test_different_input_sizes():
    """Test lambda_max_group_lasso with different input sizes."""
    n, p = 10, 8
    y = np.random.rand(n)
    groups = np.random.randint(1, 4, p)
    feature_weights = np.random.rand(p)
    beta = np.zeros(p)
    X = np.random.rand(n, p)

    result = lambda_max_group_lasso(y, groups, feature_weights, beta, X)
    assert result > 0, "Lambda max should be positive for different input sizes"


def test_single_group(sample_data):
    """Test lambda_max_group_lasso with a single group."""
    sample_data["groups"] = np.ones_like(sample_data["groups"])
    result = lambda_max_group_lasso(**sample_data)
    assert result > 0, "Lambda max should be positive for a single group"


def test_all_zero_weights(sample_data):
    """Test lambda_max_group_lasso when all feature weights are zero."""
    sample_data["feature_weights"] = np.zeros_like(sample_data["feature_weights"])
    result = lambda_max_group_lasso(**sample_data)
    assert result == 0, "Lambda max should be zero when all weights are zero"


def test_numerical_stability():
    """Test numerical stability of lambda_max_group_lasso with small weights."""
    n, p = 1000, 800
    y = np.random.rand(n)
    groups = np.random.randint(1, 11, p)
    feature_weights = np.random.rand(p) * 1e-6  # Very small weights
    beta = np.zeros(p)
    X = np.random.rand(n, p)

    result = lambda_max_group_lasso(y, groups, feature_weights, beta, X)
    assert np.isfinite(
        result
    ), "Lambda max should be finite even with very small weights"


def test_input_validation():
    """Test input validation for lambda_max_group_lasso."""
    # Test with mismatched input sizes
    result = lambda_max_group_lasso(
        np.array([1, 2]),
        np.array([1, 1]),
        np.array([1, 1]),
        np.array([0, 0]),
        np.eye(2),
    )
    assert result > 0, "Function should handle mismatched input sizes gracefully"

    # Test with mismatched beta size
    result = lambda_max_group_lasso(
        np.array([1, 2]), np.array([1, 1]), np.array([1, 1]), np.array([0]), np.eye(2)
    )
    assert result > 0, "Function should handle mismatched beta size gracefully"


def test_result_consistency(sample_data):
    """Test consistency of lambda_max_group_lasso results for the same input."""
    result1 = lambda_max_group_lasso(**sample_data)
    result2 = lambda_max_group_lasso(**sample_data)
    assert result1 == result2, "Lambda max should be consistent for the same input"


def test_scale_invariance(sample_data):
    """Test scale invariance of lambda_max_group_lasso."""
    result1 = lambda_max_group_lasso(**sample_data)

    # Scale the input
    scale_factor = 2
    sample_data["y"] *= scale_factor
    sample_data["X"] *= scale_factor

    result2 = lambda_max_group_lasso(**sample_data)
    assert np.isclose(result1, result2 / (scale_factor**2), rtol=1e-5), (
        f"Lambda max should scale quadratically with input. Got {result1} and"
        f" {result2/(scale_factor**2)}"
    )


if __name__ == "__main__":
    pytest.main()
