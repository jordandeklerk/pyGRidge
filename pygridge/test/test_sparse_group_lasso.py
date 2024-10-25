import numpy as np
import pytest
from ..src.sparse_group_lasso import sparse_group_lasso


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n, p = 100, 20
    y = np.random.randn(n)
    X = np.random.randn(n, p)
    feature_weights = np.ones(p)
    groups = np.repeat(np.arange(1, 6), 4)
    beta = np.zeros(p)
    index_permutation = np.arange(1, p + 1)
    return {
        "y": y,
        "X": X,
        "feature_weights": feature_weights,
        "groups": groups,
        "beta": beta,
        "index_permutation": index_permutation,
        "alpha": 0.5,
        "epsilon_convergence": 1e-6,
        "max_iterations": 1000,
        "gamma": 0.9,
        "lambda_max": 1.0,
        "proportion_xi": 0.01,
        "num_intervals": 10,
        "num_fixed_effects": 0,
        "trace_progress": False,
    }


def test_sparse_group_lasso_basic(sample_data):
    """Test basic functionality of sparse_group_lasso."""
    result = sparse_group_lasso(**sample_data)
    assert result is not None, "sparse_group_lasso returned None"
    assert isinstance(result, dict)
    assert "random_effects" in result
    assert "lambda" in result
    assert "iterations" in result


def test_sparse_group_lasso_dimensions(sample_data):
    """Test the output dimensions of sparse_group_lasso."""
    result = sparse_group_lasso(**sample_data)
    assert result is not None, "sparse_group_lasso returned None"
    n, p = sample_data["X"].shape
    num_intervals = sample_data["num_intervals"]
    assert result["random_effects"].shape == (num_intervals, p)
    assert result["lambda"].shape == (num_intervals,)
    assert result["iterations"].shape == (num_intervals,)


def test_sparse_group_lasso_fixed_effects(sample_data):
    """Test sparse_group_lasso with fixed effects included."""
    sample_data["num_fixed_effects"] = 5
    result = sparse_group_lasso(**sample_data)
    assert result is not None, "sparse_group_lasso returned None"
    assert "fixed_effects" in result
    assert "random_effects" in result
    assert result["fixed_effects"].shape == (sample_data["num_intervals"], 5)
    assert result["random_effects"].shape == (sample_data["num_intervals"], 15)


def test_sparse_group_lasso_convergence(sample_data):
    """Test convergence of iterations in sparse_group_lasso."""
    result = sparse_group_lasso(**sample_data)
    assert result is not None, "sparse_group_lasso returned None"
    assert np.all(result["iterations"] <= sample_data["max_iterations"])


def test_sparse_group_lasso_lambda_range(sample_data):
    """Test that lambda values are within the specified range."""
    result = sparse_group_lasso(**sample_data)
    assert result is not None, "sparse_group_lasso returned None"
    assert np.all(result["lambda"] <= sample_data["lambda_max"])
    assert np.all(
        result["lambda"] >= sample_data["lambda_max"] * sample_data["proportion_xi"]
    )


def test_sparse_group_lasso_trace_progress(sample_data, capsys):
    """Test progress tracing during sparse_group_lasso execution."""
    sample_data["trace_progress"] = True
    result = sparse_group_lasso(**sample_data)
    assert result is not None, "sparse_group_lasso returned None"
    captured = capsys.readouterr()
    assert "Loop: 10 of 10 finished" in captured.out


def test_sparse_group_lasso_alpha_extremes(sample_data):
    """Test sparse_group_lasso with extreme alpha values."""
    # Test with alpha close to 0 (more group lasso-like)
    sample_data["alpha"] = 0.01
    result_low_alpha = sparse_group_lasso(**sample_data)
    assert (
        result_low_alpha is not None
    ), "sparse_group_lasso returned None for low alpha"

    # Test with alpha close to 1 (more lasso-like)
    sample_data["alpha"] = 0.99
    result_high_alpha = sparse_group_lasso(**sample_data)
    assert (
        result_high_alpha is not None
    ), "sparse_group_lasso returned None for high alpha"

    assert not np.allclose(
        result_low_alpha["random_effects"], result_high_alpha["random_effects"]
    )


def test_sparse_group_lasso_input_validation(sample_data):
    """Test input validation for sparse_group_lasso."""
    with pytest.raises(ValueError):
        invalid_data = sample_data.copy()
        invalid_data["X"] = invalid_data["X"][:, :-1]  # Mismatch dimensions
        sparse_group_lasso(**invalid_data)


if __name__ == "__main__":
    pytest.main()
