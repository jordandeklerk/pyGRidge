"""Tests for covariance matrix designs.

This module contains tests for the covariance matrix design implementations
in covariance_design.py.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from sklearn.utils._testing import assert_almost_equal
from sklearn.exceptions import NotFittedError

from ..src.covariance_design import (
    DiscreteNonParametric,
    AR1Design,
    IdentityCovarianceDesign,
    UniformScalingCovarianceDesign,
    ExponentialOrderStatsCovarianceDesign,
    BlockDiagonal,
    block_diag,
    BlockCovarianceDesign,
    MixtureModel,
    simulate_rotated_design,
    set_groups,
)
from ..src.groupedfeatures import GroupedFeatures


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 5)
    return X


def test_discrete_nonparametric_init():
    """Test initialization of DiscreteNonParametric."""
    # Valid initialization
    spec = DiscreteNonParametric([1.0, 2.0], [0.3, 0.7])
    assert_array_equal(spec.eigs, np.array([1.0, 2.0]))
    assert_array_equal(spec.probs, np.array([0.3, 0.7]))

    # Invalid inputs
    with pytest.raises(TypeError):
        DiscreteNonParametric(1.0, [0.5, 0.5])  # eigs not a list
    with pytest.raises(TypeError):
        DiscreteNonParametric([1.0, 2.0], 0.5)  # probs not a list
    with pytest.raises(ValueError):
        DiscreteNonParametric([1.0], [0.5, 0.5])  # length mismatch
    with pytest.raises(ValueError):
        DiscreteNonParametric([1.0, 2.0], [0.3, 0.3])  # probs don't sum to 1
    with pytest.raises(ValueError):
        DiscreteNonParametric([1.0, 2.0], [-0.1, 1.1])  # negative prob


def test_ar1_design():
    """Test AR1Design covariance matrix."""
    # Test initialization
    design = AR1Design(p=3, rho=0.5)
    design.fit()

    # Expected covariance matrix for AR(1) with rho=0.5
    expected = np.array([[1.0, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 1.0]])
    assert_array_almost_equal(design.covariance_, expected)

    # Test invalid parameters
    with pytest.raises(ValueError):
        AR1Design(p=3, rho=1.0)  # rho must be < 1
    with pytest.raises(ValueError):
        AR1Design(p=3, rho=-0.1)  # rho must be >= 0
    with pytest.raises(ValueError):
        AR1Design(p=-1, rho=0.5)  # p must be positive


def test_identity_covariance_design():
    """Test IdentityCovarianceDesign."""
    design = IdentityCovarianceDesign(p=3)
    design.fit()

    expected = np.eye(3)
    assert_array_equal(design.covariance_, expected)
    assert_array_equal(design.get_precision(), expected)

    # Test spectrum
    spectrum = design.spectrum()
    assert_array_equal(spectrum.eigs, np.ones(3))
    assert_array_equal(spectrum.probs, np.ones(3) / 3)


def test_uniform_scaling_covariance_design():
    """Test UniformScalingCovarianceDesign."""
    scaling = 2.0
    design = UniformScalingCovarianceDesign(scaling=scaling, p=3)
    design.fit()

    expected = scaling * np.eye(3)
    assert_array_equal(design.covariance_, expected)
    assert_array_equal(design.get_precision(), np.eye(3) / scaling)

    # Test invalid parameters
    with pytest.raises(ValueError):
        UniformScalingCovarianceDesign(scaling=-1.0, p=3)  # scaling must be positive


def test_exponential_order_stats_covariance_design():
    """Test ExponentialOrderStatsCovarianceDesign."""
    design = ExponentialOrderStatsCovarianceDesign(p=3, rate=1.0)
    design.fit()

    # Test that covariance is diagonal and ordered
    assert np.all(np.diag(design.covariance_) > 0)  # positive diagonal
    assert np.all(np.tril(design.covariance_, -1) == 0)  # lower triangular is zero
    assert np.all(np.triu(design.covariance_, 1) == 0)  # upper triangular is zero

    # Test ordering of eigenvalues
    eigs = np.diag(design.covariance_)
    assert np.all(np.diff(eigs) < 0)  # eigenvalues should be decreasing


def test_block_diagonal():
    """Test BlockDiagonal class and block_diag function."""
    A = np.array([[1, 0], [0, 1]])
    B = np.array([[2, 0], [0, 2]])

    # Test BlockDiagonal class
    block_diag_mat = BlockDiagonal([A, B])
    expected = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2]])
    assert_array_equal(block_diag_mat.get_Sigma(), expected)

    # Test block_diag function
    assert_array_equal(block_diag(A, B), expected)

    # Test invalid inputs
    with pytest.raises(ValueError):
        BlockDiagonal([A, np.array([1, 2])])  # non-square matrix
    with pytest.raises(TypeError):
        BlockDiagonal([A, "not a matrix"])  # invalid type


def test_mixture_model():
    """Test MixtureModel."""
    s1 = DiscreteNonParametric([1, 2], [0.5, 0.5])
    s2 = DiscreteNonParametric([3, 4], [0.3, 0.7])

    # Valid initialization
    mix = MixtureModel([s1, s2], [0.6, 0.4])
    assert len(mix.spectra) == 2
    assert_array_almost_equal(mix.mixing_prop, [0.6, 0.4])

    # Invalid inputs
    with pytest.raises(ValueError):
        MixtureModel([s1, s2], [0.5, 0.6])  # mixing props don't sum to 1
    with pytest.raises(ValueError):
        MixtureModel([s1], [0.5, 0.5])  # length mismatch


def test_block_covariance_design():
    """Test BlockCovarianceDesign."""
    block1 = IdentityCovarianceDesign(p=2)
    block2 = UniformScalingCovarianceDesign(scaling=2.0, p=2)

    # Test without groups
    design = BlockCovarianceDesign([block1, block2])
    design.fit()

    expected = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2]])
    assert_array_almost_equal(design.covariance_, expected)

    # Test with groups
    groups = GroupedFeatures(ps=[2, 2])
    groups.fit(np.zeros((1, 4)))  # Fit with dummy data
    design = BlockCovarianceDesign([block1, block2], groups=groups)
    design.fit()
    assert_array_almost_equal(design.covariance_, expected)


def test_simulate_rotated_design(sample_data):
    """Test simulate_rotated_design function."""
    design = AR1Design(p=5, rho=0.5)
    design.fit()  # Ensure design is fitted
    n_samples = 100

    # Test with default random measure
    X = simulate_rotated_design(design, n_samples)
    assert X.shape == (n_samples, 5)

    # Test with custom random measure
    def custom_measure(size):
        return np.ones(size)

    X = simulate_rotated_design(design, n_samples, rotated_measure=custom_measure)
    assert X.shape == (n_samples, 5)

    # Test with unfitted design
    unfitted_design = AR1Design(p=5, rho=0.5)
    with pytest.raises(NotFittedError):
        simulate_rotated_design(unfitted_design, n_samples)

    # Test other invalid inputs
    with pytest.raises(TypeError):
        simulate_rotated_design("not a design", n_samples)
    with pytest.raises(ValueError):
        simulate_rotated_design(design, -1)


def test_set_groups():
    """Test set_groups function."""
    # Test with integer input
    design = IdentityCovarianceDesign()
    set_groups(design, 5)
    assert design.p == 5

    # Test with GroupedFeatures input
    block1 = IdentityCovarianceDesign()
    block2 = UniformScalingCovarianceDesign(scaling=2.0)
    block_design = BlockCovarianceDesign([block1, block2])

    groups = GroupedFeatures(ps=[2, 3])
    groups.fit(np.zeros((1, 5)))  # Fit with dummy data
    set_groups(block_design, groups)
    assert block1.p == 2
    assert block2.p == 3

    # Test with unfitted GroupedFeatures
    unfitted_groups = GroupedFeatures(ps=[2, 3])
    with pytest.raises(NotFittedError):
        set_groups(block_design, unfitted_groups)

    # Test other invalid inputs
    with pytest.raises(TypeError):
        set_groups(design, "not an int or GroupedFeatures")
    with pytest.raises(ValueError):
        set_groups(design, -1)


def test_covariance_design_transform(sample_data):
    """Test transform method of CovarianceDesign."""
    design = AR1Design(p=5, rho=0.5)
    design.fit()

    # Test successful transform
    X_transformed = design.transform(sample_data)
    assert X_transformed.shape == sample_data.shape

    # Test transform before fit
    design_unfit = AR1Design(p=5, rho=0.5)
    with pytest.raises(NotFittedError):
        design_unfit.transform(sample_data)

    # Test with invalid input dimensions
    with pytest.raises(ValueError):
        design.transform(sample_data[:, :3])  # Wrong number of features


def test_covariance_design_score(sample_data):
    """Test score method of CovarianceDesign."""
    design = AR1Design(p=5, rho=0.5)
    design.fit()

    # Test successful score computation
    score = design.score(sample_data)
    assert isinstance(score, float)
    assert score <= 0  # Log-likelihood should be non-positive

    # Test score before fit
    design_unfit = AR1Design(p=5, rho=0.5)
    with pytest.raises(NotFittedError):
        design_unfit.score(sample_data)

    # Test with invalid input dimensions
    with pytest.raises(ValueError):
        design.score(sample_data[:, :3])  # Wrong number of features
