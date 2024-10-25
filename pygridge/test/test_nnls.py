import numpy as np
import pytest
from scipy.optimize import nnls as scipy_nnls
import time
from scipy.sparse import csr_matrix

from ..src.nnls import (
    nonneg_lsq,
    fnnls,
    fnnls_core,
    NNLSError,
    InvalidInputError,
    ConvergenceError,
)


@pytest.fixture
def simple_A_B():
    """Fixture providing a simple A and B for testing."""
    A = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    B = np.array([7, 8, 9], dtype=float)
    return A, B


@pytest.fixture
def random_A_B():
    """Fixture providing random A and B matrices for testing."""
    np.random.seed(0)
    A = np.random.rand(1000, 500)
    X_true = np.abs(np.random.randn(500, 3))  # Ensure non-negative
    B = A @ X_true + 0.1 * np.random.randn(1000, 3)
    return A, B


def test_nonneg_lsq_basic(simple_A_B):
    """Test basic functionality of nonneg_lsq."""
    A, B = simple_A_B
    X = nonneg_lsq(A, B, alg="fnnls")
    assert X.shape == (A.shape[1],), "Solution shape mismatch."
    assert np.all(X >= -1e-8), "Negative values in solution."


def test_nonneg_lsq_multiple_rhs(random_A_B):
    """Test nonneg_lsq with multiple right-hand sides."""
    A, B = random_A_B
    X = nonneg_lsq(A, B, alg="fnnls")
    assert X.shape == (
        A.shape[1],
        B.shape[1],
    ), "Solution shape mismatch for multiple RHS."
    assert np.all(X >= -1e-8), "Negative values in solution for multiple RHS."


def test_nonneg_lsq_correctness(simple_A_B):
    """Test correctness of nonneg_lsq against SciPy's nnls."""
    A, B = simple_A_B
    X = nonneg_lsq(A, B, alg="fnnls")
    residual = np.linalg.norm(A @ X - B)
    scipy_X, _ = scipy_nnls(A, B)
    scipy_residual = np.linalg.norm(A @ scipy_X - B)
    assert np.isclose(
        residual, scipy_residual, atol=1e-6
    ), "Residuals do not match SciPy's nnls."


def test_nonneg_lsq_scipy_comparison(random_A_B):
    """Compare nonneg_lsq results with SciPy's nnls for multiple RHS."""
    A, B = random_A_B
    X = nonneg_lsq(A, B, alg="fnnls")
    for i in range(B.shape[1]):
        scipy_X, scipy_res = scipy_nnls(A, B[:, i])
        np.testing.assert_allclose(
            X[:, i], scipy_X, atol=1e-6, err_msg=f"Mismatch in column {i}"
        )


def test_nonneg_lsq_gram_matrix(random_A_B):
    """Test nonneg_lsq using the Gram matrix."""
    A, B = random_A_B
    AtA = A.T @ A
    AtB = A.T @ B
    X = nonneg_lsq(AtA, AtB, alg="fnnls", gram=True)
    for i in range(B.shape[1]):
        scipy_X, scipy_res = scipy_nnls(A, B[:, i])
        np.testing.assert_allclose(
            X[:, i], scipy_X, atol=1e-6, err_msg=f"Mismatch in Gram matrix column {i}"
        )


def test_nonneg_lsq_parallel():
    """Test nonneg_lsq with parallel execution."""
    np.random.seed(0)
    A = np.random.rand(1000, 500)
    X_true = np.abs(np.random.randn(500, 10))
    B = A @ X_true + 0.1 * np.random.randn(1000, 10)

    start_time = time.time()
    X_parallel = nonneg_lsq(A, B, alg="fnnls", use_parallel=True)
    parallel_time = time.time() - start_time

    start_time = time.time()
    X_sequential = nonneg_lsq(A, B, alg="fnnls", use_parallel=False)
    sequential_time = time.time() - start_time

    np.testing.assert_allclose(
        X_parallel,
        X_sequential,
        atol=1e-6,
        err_msg="Parallel and sequential solutions differ.",
    )
    assert parallel_time < sequential_time * 2, "Parallel execution is not efficient."


def test_nonneg_lsq_invalid_algorithm(simple_A_B):
    """Test nonneg_lsq with an invalid algorithm."""
    A, B = simple_A_B
    with pytest.raises(
        ValueError, match="Specified algorithm 'invalid_alg' not recognized."
    ):
        nonneg_lsq(A, B, alg="invalid_alg")


def test_nonneg_lsq_zero_matrix():
    """Test nonneg_lsq with zero matrices."""
    A = np.zeros((5, 3))
    B = np.zeros(5)
    X = nonneg_lsq(A, B, alg="fnnls")
    assert np.all(X == 0), "Solution should be zero vector for zero matrices."


def test_nonneg_lsq_empty_matrix():
    """Test nonneg_lsq with empty matrices."""
    A = np.empty((0, 0))
    B = np.empty((0,))
    with pytest.raises(
        InvalidInputError, match="Input matrices A and B must not be empty."
    ):
        nonneg_lsq(A, B, alg="fnnls")


def test_nonneg_lsq_incompatible_shapes():
    """Test nonneg_lsq with incompatible shapes for A and B."""
    A = np.random.rand(5, 3)
    B = np.random.rand(4)
    with pytest.raises(InvalidInputError, match="Incompatible shapes:"):
        nonneg_lsq(A, B, alg="fnnls")


def test_nonneg_lsq_gram_incompatible_shapes():
    """Test nonneg_lsq with incompatible shapes when gram=True."""
    A = np.random.rand(5, 3)
    B = np.random.rand(4)
    with pytest.raises(
        InvalidInputError, match="When gram=True, A must be a square matrix."
    ):
        nonneg_lsq(A, B, alg="fnnls", gram=True)


def test_nonneg_lsq_non_square_gram():
    """Test nonneg_lsq with non-square A when gram=True."""
    A = np.random.rand(5, 3)
    B = np.random.rand(5)
    with pytest.raises(
        InvalidInputError, match="When gram=True, A must be a square matrix."
    ):
        nonneg_lsq(A, B, alg="fnnls", gram=True)


def test_nonneg_lsq_single_variable():
    """Test nonneg_lsq with a single variable."""
    A = np.array([[1], [2], [3]], dtype=float)
    B = np.array([1, 2, 3], dtype=float)
    X = nonneg_lsq(A, B, alg="fnnls")
    expected = np.array([1.0])
    np.testing.assert_allclose(
        X, expected, atol=1e-6, err_msg="Single variable solution incorrect."
    )


def test_nonneg_lsq_large_problem():
    """Test nonneg_lsq with a large problem setup."""
    A = np.random.rand(1000, 500)
    X_true = np.abs(np.random.randn(500))
    B = A @ X_true + 0.01 * np.random.randn(1000)
    X = nonneg_lsq(A, B, alg="fnnls", use_parallel=True)
    residual = np.linalg.norm(A @ X - B)
    scipy_X, scipy_res = scipy_nnls(A, B)
    assert np.all(X >= -1e-8), "Negative values in large problem solution."
    assert np.isclose(
        residual, scipy_res, atol=1e-4
    ), "Residuals do not match SciPy's nnls in large problem."


def test_nonneg_lsq_tolerance(simple_A_B):
    """Test nonneg_lsq with a specified tolerance."""
    A, B = simple_A_B
    X = nonneg_lsq(A, B, alg="fnnls", tol=1e-4)
    assert np.all(X >= -1e-4), "Solution violates tolerance."


def test_nonneg_lsq_max_iter(simple_A_B):
    """Test nonneg_lsq with max_iter set to force failure."""
    A, B = simple_A_B
    with pytest.raises(ConvergenceError, match="FNNLS failed to converge"):
        nonneg_lsq(
            A, B, alg="fnnls", max_iter=0
        )  # Set max_iter=0 to force convergence failure


def test_nonneg_lsq_non_unique_solution():
    """Test nonneg_lsq with a non-unique solution."""
    A = np.array([[1, 1], [2, 2], [3, 3]], dtype=float)
    B = np.array([6, 12, 18], dtype=float)
    X = nonneg_lsq(A, B, alg="fnnls")
    assert np.isclose(
        X[0] + X[1], 6, atol=1e-6
    ), "Non-unique solution does not satisfy sum constraint."
    assert np.all(X >= -1e-8), "Solution violates non-negativity."


def test_nonneg_lsq_no_solution():
    """Test nonneg_lsq when no solution exists."""
    A = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    B = np.array([-1, -2, -3], dtype=float)
    X = nonneg_lsq(A, B, alg="fnnls")
    expected = np.zeros(A.shape[1])
    np.testing.assert_allclose(
        X, expected, atol=1e-6, err_msg="Negative B should lead to zero solution."
    )


def test_nonneg_lsq_sparse_matrix():
    """Test nonneg_lsq with a sparse matrix."""
    A_sparse = csr_matrix([[1, 0], [0, 2], [3, 0]], dtype=float)
    B = np.array([1, 2, 3], dtype=float)
    X = nonneg_lsq(A_sparse.toarray(), B, alg="fnnls")
    expected = np.array([1.0, 1.0])
    np.testing.assert_allclose(
        X, expected, atol=1e-6, err_msg="Sparse matrix solution incorrect."
    )


def test_nonneg_lsq_with_kwargs(simple_A_B):
    """Test nonneg_lsq with additional unused keyword arguments."""
    A, B = simple_A_B
    X = nonneg_lsq(A, B, alg="fnnls", some_unused_kwarg=123)
    assert X.shape == (A.shape[1],), "Solution shape mismatch with kwargs."
    assert np.all(X >= -1e-8), "Negative values in solution with kwargs."


def test_nonneg_lsq_precision():
    """Test nonneg_lsq for precision with small values."""
    A = np.array([[1, 0], [0, 1]], dtype=float)
    B = np.array([1e-10, 1e-10], dtype=float)
    X = nonneg_lsq(A, B, alg="fnnls", tol=1e-12)
    expected = np.array([1e-10, 1e-10])
    np.testing.assert_allclose(
        X, expected, atol=1e-12, err_msg="Precision test failed."
    )


def test_nonneg_lsq_highly_correlated_columns():
    """Test nonneg_lsq with highly correlated columns in A."""
    A = np.array([[1, 1.0001], [2, 2.0002], [3, 3.0003]], dtype=float)
    B = np.array([4, 8, 12], dtype=float)
    X = nonneg_lsq(A, B, alg="fnnls")
    expected = np.array([0.0, 4.0])
    np.testing.assert_allclose(
        X, expected, atol=1e-3, err_msg="Highly correlated columns solution incorrect."
    )


def test_fnnls_core_individual_solution():
    """Test fnnls_core for an individual solution."""
    AtA = np.array([[4, 2], [2, 3]], dtype=float)
    Atb = np.array([2, 3], dtype=float)
    X = fnnls_core(AtA, Atb)
    expected = np.array([0.0, 1.0])
    np.testing.assert_allclose(
        X, expected, atol=1e-6, err_msg="fnnls_core individual solution incorrect."
    )


def test_fnnls_core_no_positive_gradients():
    """Test fnnls_core when no positive gradients exist."""
    AtA = np.array([[1, 0], [0, 1]], dtype=float)
    Atb = np.array([-1, -1], dtype=float)
    X = fnnls_core(AtA, Atb)
    expected = np.array([0.0, 0.0])
    np.testing.assert_allclose(
        X, expected, atol=1e-6, err_msg="fnnls_core should return zero vector."
    )


def test_fnnls_core_max_iter_reached():
    """Test fnnls_core when max_iter is reached."""
    AtA = np.array([[1, 0], [0, 1]], dtype=float)
    Atb = np.array([1, 1], dtype=float)
    with pytest.raises(ConvergenceError, match="FNNLS failed to converge"):
        fnnls_core(AtA, Atb, max_iter=0)


def test_fnnls_core_all_positive():
    """Test fnnls_core when all variables are positive."""
    AtA = np.array([[2, 0], [0, 2]], dtype=float)
    Atb = np.array([2, 2], dtype=float)
    X = fnnls_core(AtA, Atb)
    expected = np.array([1.0, 1.0])
    np.testing.assert_allclose(
        X,
        expected,
        atol=1e-6,
        err_msg=(
            "fnnls_core should return exact solution when all variables are positive."
        ),
    )


if __name__ == "__main__":
    pytest.main()
