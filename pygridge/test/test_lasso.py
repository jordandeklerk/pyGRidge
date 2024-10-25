import numpy as np
import pytest
from ..src.lasso import lasso


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n, p = 100, 20
    y = np.random.randn(n)
    X = np.random.randn(n, p)
    feature_weights = np.ones(p)
    beta = np.zeros(p)
    return {
        "y": y,
        "X": X,
        "feature_weights": feature_weights,
        "beta": beta,
        "epsilon_convergence": 1e-6,
        "max_iterations": 1000,
        "gamma": 0.9,
        "lambda_max": 1.0,
        "proportion_xi": 0.01,
        "num_intervals": 10,
        "num_fixed_effects": 0,
        "trace_progress": False,
    }


def test_lasso_basic(sample_data):
    """Test basic functionality of lasso."""
    result = lasso(**sample_data)
    assert isinstance(result, dict)
    assert "random_effects" in result
    assert "lambda" in result
    assert "iterations" in result


def test_lasso_dimensions(sample_data):
    """Test the dimensions of the output from lasso."""
    result = lasso(**sample_data)
    n, p = sample_data["X"].shape
    num_intervals = sample_data["num_intervals"]
    assert result["random_effects"].shape == (num_intervals, p)
    assert result["lambda"].shape == (num_intervals,)
    assert result["iterations"].shape == (num_intervals,)


def test_lasso_fixed_effects(sample_data):
    """Test lasso with fixed effects included."""
    sample_data["num_fixed_effects"] = 5
    result = lasso(**sample_data)
    assert "fixed_effects" in result
    assert "random_effects" in result
    assert result["fixed_effects"].shape == (sample_data["num_intervals"], 5)
    assert result["random_effects"].shape == (sample_data["num_intervals"], 15)


def test_lasso_convergence(sample_data):
    """Test convergence of iterations in lasso."""
    result = lasso(**sample_data)
    assert np.all(result["iterations"] <= sample_data["max_iterations"])


def test_lasso_lambda_range(sample_data):
    """Test that lambda values are within the specified range."""
    result = lasso(**sample_data)
    assert np.all(result["lambda"] <= sample_data["lambda_max"])
    assert np.all(
        result["lambda"] >= sample_data["lambda_max"] * sample_data["proportion_xi"]
    )


def test_lasso_trace_progress(sample_data, capsys):
    """Test progress tracing during lasso execution."""
    sample_data["trace_progress"] = True
    lasso(**sample_data)
    captured = capsys.readouterr()
    assert "Loop: 10 of 10 finished" in captured.out


def test_lasso_zero_lambda(sample_data):
    """Test lasso behavior with lambda_max set to zero."""
    sample_data["lambda_max"] = 0
    result = lasso(**sample_data)
    assert np.allclose(result["lambda"], 0)
    assert np.all(
        result["iterations"] == 1
    )  # Should converge in one iteration for lambda = 0

    # Check if the solution converges to OLS
    X = sample_data["X"]
    y = sample_data["y"]
    ols_solution = np.linalg.lstsq(X, y, rcond=None)[0]
    assert np.allclose(result["random_effects"][-1], ols_solution, atol=1e-5)


def test_lasso_large_lambda(sample_data):
    """Test lasso behavior with a large lambda_max value."""
    sample_data["lambda_max"] = 1e6
    result = lasso(**sample_data)
    assert np.all(
        np.abs(result["random_effects"]) < 1e-6
    )  # All coefficients should be close to zero


def test_lasso_single_iteration(sample_data):
    """Test lasso with a single interval."""
    sample_data["num_intervals"] = 1
    result = lasso(**sample_data)
    assert len(result["lambda"]) == 1
    assert len(result["iterations"]) == 1


def test_lasso_input_validation(sample_data):
    """Test input validation for lasso."""
    with pytest.raises(ValueError):
        invalid_data = sample_data.copy()
        invalid_data["X"] = invalid_data["X"][:, :-1]  # Mismatch dimensions
        lasso(**invalid_data)


if __name__ == "__main__":
    pytest.main()
