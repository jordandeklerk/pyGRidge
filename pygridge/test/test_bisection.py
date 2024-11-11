"""Tests for the bisection module."""

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from ..src.bisection import BisectionSolver, BisectionError, ConvergenceError


@pytest.fixture
def solver():
    """Create a BisectionSolver instance for testing."""
    return BisectionSolver()


def test_basic_root_finding(solver):
    """Test basic root finding functionality."""
    rows = 3
    alpha = 0.5
    left_border = 0.0
    right_border = 1.0
    group_weight = 1.0
    vector_weights = np.array([1.0, 1.0, 1.0])
    vector_in = np.array([0.5, 0.3, 0.2])

    root = solver.solve(
        rows, alpha, left_border, right_border, group_weight, vector_weights, vector_in
    )
    assert root >= left_border and root <= right_border
    assert solver.n_iter_ > 0


def test_invalid_alpha(solver):
    """Test that invalid alpha values raise ValueError."""
    rows = 3
    left_border = 0.0
    right_border = 1.0
    group_weight = 1.0
    vector_weights = np.array([1.0, 1.0, 1.0])
    vector_in = np.array([0.5, 0.3, 0.2])

    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        solver.solve(
            rows,
            0.0,
            left_border,
            right_border,
            group_weight,
            vector_weights,
            vector_in,
        )

    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        solver.solve(
            rows,
            1.0,
            left_border,
            right_border,
            group_weight,
            vector_weights,
            vector_in,
        )


def test_invalid_borders(solver):
    """Test that invalid border values raise ValueError."""
    rows = 3
    alpha = 0.5
    group_weight = 1.0
    vector_weights = np.array([1.0, 1.0, 1.0])
    vector_in = np.array([0.5, 0.3, 0.2])

    with pytest.raises(ValueError, match="left_border must be less than right_border"):
        solver.solve(
            rows, alpha, 1.0, 0.0, group_weight, vector_weights, vector_in
        )  # left > right

    with pytest.raises(ValueError, match="left_border must be less than right_border"):
        solver.solve(
            rows, alpha, 1.0, 1.0, group_weight, vector_weights, vector_in
        )  # left == right


def test_dimension_mismatch(solver):
    """Test that mismatched dimensions raise ValueError."""
    rows = 3
    alpha = 0.5
    left_border = 0.0
    right_border = 1.0
    group_weight = 1.0

    # Test with wrong vector_weights length
    vector_weights = np.array([1.0, 1.0])
    vector_in = np.array([0.5, 0.3, 0.2])
    with pytest.raises(ValueError, match="rows must match the length"):
        solver.solve(
            rows,
            alpha,
            left_border,
            right_border,
            group_weight,
            vector_weights,
            vector_in,
        )

    # Test with wrong vector_in length
    vector_weights = np.array([1.0, 1.0, 1.0])
    vector_in = np.array([0.5, 0.3])
    with pytest.raises(ValueError, match="rows must match the length"):
        solver.solve(
            rows,
            alpha,
            left_border,
            right_border,
            group_weight,
            vector_weights,
            vector_in,
        )


def test_convergence_failure():
    """Test that maximum iterations leads to ConvergenceError."""
    solver = BisectionSolver(max_iter=1)  # Set very low max_iter to force failure
    rows = 3
    alpha = 0.5
    left_border = 0.0
    right_border = 1.0
    group_weight = 1.0
    vector_weights = np.array([1.0, 1.0, 1.0])
    vector_in = np.array([0.5, 0.3, 0.2])

    with pytest.raises(ConvergenceError, match="Bisection did not converge"):
        solver.solve(
            rows,
            alpha,
            left_border,
            right_border,
            group_weight,
            vector_weights,
            vector_in,
        )


def test_tolerance_scaling(solver):
    """Test that tolerance scaling works correctly."""
    rows = 3
    alpha = 0.5
    left_border = 0.0
    right_border = 1000.0
    group_weight = 1.0
    vector_weights = np.array([1.0, 1.0, 1.0])
    vector_in = np.array([500.0, 300.0, 200.0])

    # With scaling
    solver_with_scaling = BisectionSolver(scale_tol=True)
    root_scaled = solver_with_scaling.solve(
        rows, alpha, left_border, right_border, group_weight, vector_weights, vector_in
    )

    # Without scaling
    solver_without_scaling = BisectionSolver(scale_tol=False)
    root_unscaled = solver_without_scaling.solve(
        rows, alpha, left_border, right_border, group_weight, vector_weights, vector_in
    )

    # The scaled version should converge in fewer iterations
    assert solver_with_scaling.n_iter_ <= solver_without_scaling.n_iter_


def test_get_params():
    """Test that get_params works correctly."""
    solver = BisectionSolver(tol=1e-6, max_iter=100, scale_tol=False)
    params = solver.get_params()

    assert params["tol"] == 1e-6
    assert params["max_iter"] == 100
    assert params["scale_tol"] is False


def test_set_params():
    """Test that set_params works correctly."""
    solver = BisectionSolver()
    solver.set_params(tol=1e-6, max_iter=100, scale_tol=False)

    assert solver.tol == 1e-6
    assert solver.max_iter == 100
    assert solver.scale_tol is False
