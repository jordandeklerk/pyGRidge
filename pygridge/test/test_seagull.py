"""Unit tests for seagull estimator."""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import warnings
from ..src.seagull import seagull


@pytest.fixture
def sample_data():
    """Create sample data for testing.
    
    Returns
    -------
    tuple
        Tuple containing (y, X, Z, weights_u, groups) where:
        - y is the target variable
        - X is the fixed effects design matrix
        - Z is the random effects design matrix
        - weights_u is the weights for random effects
        - groups is the group labels
    """
    rng = np.random.RandomState(42)
    n_samples = 5
    
    y = np.arange(1, n_samples + 1, dtype=np.float64)
    X = rng.randn(n_samples, 2).astype(np.float64)
    Z = np.column_stack([
        np.ones(n_samples),
        np.arange(1, n_samples + 1),
        np.arange(1, n_samples + 1) ** 2
    ]).astype(np.float64)
    weights_u = np.ones(3, dtype=np.float64)
    groups = np.array([1, 1, 2, 2, 3], dtype=np.float64)
    
    return y, X, Z, weights_u, groups


class TestSeagullInputValidation:
    """Test input validation in seagull."""
    
    def test_missing_y(self):
        """Test if error is raised when y is missing."""
        with pytest.raises(ValueError, match="Vector y is missing"):
            seagull(y=None, Z=np.array([[1, 2], [3, 4]]))

    def test_empty_y(self):
        """Test if error is raised when y is empty."""
        with pytest.raises(ValueError, match="Vector y is empty"):
            seagull(y=np.array([]), Z=np.array([[1, 2], [3, 4]]))

    def test_non_numeric_y(self):
        """Test if error is raised when y contains non-numeric values."""
        with pytest.raises(ValueError, match="Non-numeric values detected"):
            seagull(y=np.array(['a', 'b']), Z=np.array([[1, 2], [3, 4]]))

    def test_nan_inf_y(self):
        """Test if error is raised when y contains NaN or Inf."""
        with pytest.raises(ValueError, match="NA, NaN, or Inf detected"):
            seagull(y=np.array([1, np.nan, 3]), Z=np.array([[1, 2], [3, 4], [5, 6]]))

    def test_missing_Z(self):
        """Test if error is raised when Z is missing."""
        with pytest.raises(ValueError, match="Matrix Z is missing"):
            seagull(y=np.array([1, 2, 3]), Z=None)

    def test_non_numeric_Z(self):
        """Test if error is raised when Z contains non-numeric values."""
        with pytest.raises(ValueError, match="Non-numeric values detected"):
            seagull(y=np.array([1, 2]), Z=np.array([['a', 'b'], ['c', 'd']]))

    def test_nan_inf_Z(self):
        """Test if error is raised when Z contains NaN or Inf."""
        with pytest.raises(ValueError, match="NA, NaN, or Inf detected"):
            seagull(y=np.array([1, 2]), Z=np.array([[1, np.inf], [3, 4]]))

    def test_mismatching_dimensions(self):
        """Test if error is raised when dimensions don't match."""
        with pytest.raises(ValueError, match="Mismatching dimensions"):
            seagull(y=np.array([1, 2, 3]), Z=np.array([[1, 2], [3, 4]]))


class TestSeagullEstimator:
    """Test seagull estimator functionality."""

    def test_lasso_path(self, sample_data):
        """Test lasso path computation."""
        y, X, Z, weights_u, _ = sample_data
        result = seagull(y=y, X=X, Z=Z, weights_u=weights_u, alpha=1.0)
        
        assert result["result"] == "lasso"
        assert isinstance(result["lambda_values"], np.ndarray)
        assert isinstance(result["beta"], np.ndarray)
        assert result["beta"].shape[0] == X.shape[1] + Z.shape[1]

    def test_group_lasso_path(self, sample_data):
        """Test group lasso path computation."""
        y, X, Z, weights_u, groups = sample_data
        result = seagull(y=y, X=X, Z=Z, weights_u=weights_u, 
                        groups=groups, alpha=0.0)
        
        assert result["result"] == "group_lasso"
        assert isinstance(result["lambda_values"], np.ndarray)
        assert isinstance(result["beta"], np.ndarray)

    def test_sparse_group_lasso_path(self, sample_data):
        """Test sparse group lasso path computation."""
        y, X, Z, weights_u, groups = sample_data
        result = seagull(y=y, X=X, Z=Z, weights_u=weights_u, 
                        groups=groups, alpha=0.5)
        
        assert result["result"] == "sparse_group_lasso"
        assert isinstance(result["lambda_values"], np.ndarray)
        assert isinstance(result["beta"], np.ndarray)


class TestSeagullParameters:
    """Test seagull parameter validation and defaults."""

    def test_invalid_alpha(self, sample_data):
        """Test handling of invalid alpha parameter."""
        y, _, Z, _, _ = sample_data
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = seagull(y=y, Z=Z, alpha=2.0)
            assert len(w) == 1
            assert "alpha is out of range" in str(w[0].message)
            assert result["result"] == "lasso"

    def test_missing_groups_for_group_lasso(self, sample_data):
        """Test if error is raised when groups are missing for group lasso."""
        y, X, Z, weights_u, _ = sample_data
        with pytest.raises(ValueError, match="Vector groups is missing"):
            seagull(y=y, X=X, Z=Z, weights_u=weights_u, alpha=0.5)

    def test_parameter_validation(self, sample_data):
        """Test validation of numerical parameters."""
        y, _, Z, _, _ = sample_data
        
        # Test rel_acc validation
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            seagull(y=y, Z=Z, rel_acc=-0.1)
            assert any("rel_acc is non-positive" in str(warn.message) for warn in w)

        # Test max_iter validation
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            seagull(y=y, Z=Z, max_iter=0)
            assert any("max_iter is non-positive" in str(warn.message) for warn in w)

        # Test gamma_bls validation
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            seagull(y=y, Z=Z, gamma_bls=1.5)
            assert any("gamma_bls is out of range" in str(warn.message) for warn in w)


class TestSeagullEdgeCases:
    """Test seagull edge cases and corner cases."""

    def test_single_feature(self):
        """Test with single feature."""
        y = np.array([1., 2., 3.])
        Z = np.array([[1.], [2.], [3.]])
        result = seagull(y=y, Z=Z)
        assert result is not None
        assert isinstance(result, dict)

    def test_perfect_fit(self):
        """Test with perfectly correlated data."""
        y = np.array([1., 2., 3.])
        Z = np.array([[1., 0.], [2., 0.], [3., 0.]])
        result = seagull(y=y, Z=Z)
        assert result is not None
        assert isinstance(result, dict)

    def test_constant_feature(self):
        """Test with constant feature."""
        y = np.array([1., 2., 3.])
        Z = np.array([[1., 1.], [2., 1.], [3., 1.]])
        result = seagull(y=y, Z=Z)
        assert result is not None
        assert isinstance(result, dict)
