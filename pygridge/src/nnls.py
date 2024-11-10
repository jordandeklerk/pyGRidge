"""Solve Non-Negative Least Squares (NNLS) problems efficiently."""

import numpy as np
from typing import Union, Optional
from functools import partial
from multiprocessing import cpu_count
from joblib import Parallel, delayed
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class NNLSError(Exception):
    """Base exception class for NNLS-related errors."""

    pass


class InvalidInputError(NNLSError):
    """Exception raised for invalid input to NNLS functions."""

    pass


class ConvergenceError(NNLSError):
    """Exception raised when the algorithm fails to converge."""

    pass


def nonneg_lsq(
    A: np.ndarray,
    B: Union[np.ndarray, np.ndarray],
    alg: str = "fnnls",
    gram: bool = False,
    use_parallel: bool = False,
    tol: float = 1e-8,
    max_iter: Optional[int] = None,
    **kwargs,
) -> np.ndarray:
    r"""Solve the non-negative least squares (NNLS) problem.

    Solves the optimization problem:

    .. math::
        \min_{\mathbf{X}} \| \mathbf{A} \mathbf{X} - \mathbf{B} \|_2^2 \quad \text{subject to} \quad \mathbf{X} \geq 0

    Parameters
    ----------
    A : array-like of shape (n_samples, n_features)
        The input matrix :math:`\mathbf{A}`.
    B : array-like of shape (n_samples,) or (n_samples, n_targets)
        The target matrix :math:`\mathbf{B}`. If :math:`\mathbf{B}` is a vector, it will be converted to a column matrix.
    alg : {'fnnls'}, default='fnnls'
        The algorithm to use. Currently, only 'fnnls' is supported.
    gram : bool, default=False
        If True, :math:`\mathbf{A}` and :math:`\mathbf{B}` are treated as Gram matrices (:math:`\mathbf{A}^T \mathbf{A}` and :math:`\mathbf{A}^T \mathbf{B}`).
    use_parallel : bool, default=False
        If True and multiple CPUs are available, computations for multiple columns of :math:`\mathbf{B}` are parallelized.
    tol : float, default=1e-8
        Tolerance for non-negativity constraints.
    max_iter : int, optional
        Maximum number of iterations. If None, set to :math:`30 \times` number of columns in :math:`\mathbf{A}^T \mathbf{A}`.
    **kwargs : dict
        Additional keyword arguments passed to the underlying algorithm.

    Returns
    -------
    X : ndarray of shape (n_features, n_targets)
        Solution matrix :math:`\mathbf{X}` that minimizes :math:`\| \mathbf{A} \mathbf{X} - \mathbf{B} \|_2` subject to :math:`\mathbf{X} \geq 0`.

    Raises
    ------
    InvalidInputError
        If the input matrices are invalid or incompatible.
    ValueError
        If the specified algorithm is not recognized.
    """

    # Input validation
    if not isinstance(A, np.ndarray) or not isinstance(B, np.ndarray):
        raise InvalidInputError("A and B must be numpy arrays.")

    if A.size == 0 or B.size == 0:
        raise InvalidInputError("Input matrices A and B must not be empty.")

    if B.ndim == 1:
        B = B[:, np.newaxis]

    if not gram and A.shape[0] != B.shape[0]:
        raise InvalidInputError(
            f"Incompatible shapes: A has {A.shape[0]} rows, B has {B.shape[0]} rows."
        )

    if gram and A.shape[0] != A.shape[1]:
        raise InvalidInputError("When gram=True, A must be a square matrix.")

    if gram and A.shape[0] != B.shape[0]:
        raise InvalidInputError(
            f"Incompatible shapes for gram matrices: A has {A.shape[0]} rows, B has"
            f" {B.shape[0]} rows."
        )

    if alg == "fnnls":
        return fnnls(
            A,
            B,
            gram=gram,
            use_parallel=use_parallel,
            tol=tol,
            max_iter=max_iter,
            **kwargs,
        )
    else:
        raise ValueError(f"Specified algorithm '{alg}' not recognized.")


def fnnls(
    A: np.ndarray,
    B: Union[np.ndarray, np.ndarray],
    gram: bool = False,
    use_parallel: bool = False,
    tol: float = 1e-8,
    max_iter: Optional[int] = None,
    **kwargs,
) -> np.ndarray:
    r"""Solve the non-negative least squares (NNLS) problem using the Fast NNLS (FNNLS) algorithm.

    Parameters
    ----------
    A : array-like of shape (n_samples, n_features)
        The input matrix :math:`\mathbf{A}` or Gram matrix :math:`\mathbf{A}^T \mathbf{A}` if `gram=True`.
    B : array-like of shape (n_samples,) or (n_samples, n_targets)
        The target matrix :math:`\mathbf{B}` or :math:`\mathbf{A}^T \mathbf{B}` if `gram=True`. If :math:`\mathbf{B}` is a vector, it will be converted to a column matrix.
    gram : bool, default=False
        If True, :math:`\mathbf{A}` and :math:`\mathbf{B}` are treated as Gram matrices (:math:`\mathbf{A}^T \mathbf{A}` and :math:`\mathbf{A}^T \mathbf{B}`).
    use_parallel : bool, default=False
        If True and multiple CPUs are available, computations for multiple columns of :math:`\mathbf{B}` are parallelized.
    tol : float, default=1e-8
        Tolerance for non-negativity constraints.
    max_iter : int, optional
        Maximum number of iterations. If None, set to :math:`30 \times` number of variables.
    **kwargs : dict
        Additional keyword arguments passed to the core FNNLS algorithm.

    Returns
    -------
    X : ndarray of shape (n_features, n_targets)
        Solution matrix :math:`\mathbf{X}` that minimizes :math:`\| \mathbf{A} \mathbf{X} - \mathbf{B} \|_2` subject to :math:`\mathbf{X} \geq 0`.

    Raises
    ------
    InvalidInputError
        If the input matrices are invalid or incompatible.
    ConvergenceError
        If the algorithm fails to converge within the maximum number of iterations.
    """

    if B.ndim == 1:
        B = B[:, np.newaxis]

    n, k = B.shape

    if gram:
        AtA = A
        AtB = B
    else:
        AtA = A.T @ A
        AtB = A.T @ B

    if max_iter is None:
        max_iter = 30 * AtA.shape[0]

    if use_parallel and cpu_count() > 1 and k > 1:
        solve_fn = partial(fnnls_core, AtA, tol=tol, max_iter=max_iter, **kwargs)
        X = np.column_stack(
            Parallel(n_jobs=-1)(delayed(solve_fn)(AtB[:, i]) for i in range(k))
        )
    else:
        X = np.zeros_like(AtB)
        for i in range(k):
            X[:, i] = fnnls_core(AtA, AtB[:, i], tol=tol, max_iter=max_iter, **kwargs)

    if B.shape[1] == 1:
        return X.ravel()
    return X


def fnnls_core(
    AtA: np.ndarray, Atb: np.ndarray, tol: float = 1e-8, max_iter: int = 300, **kwargs
) -> np.ndarray:
    r"""Core Fast Non-Negative Least Squares (FNNLS) algorithm to solve a single NNLS problem.

    Parameters
    ----------
    AtA : array-like of shape (n_features, n_features)
        The Gram matrix :math:`\mathbf{A}^T \mathbf{A}`.
    Atb : array-like of shape (n_features,)
        The product :math:`\mathbf{A}^T \mathbf{b}`.
    tol : float, default=1e-8
        Tolerance for non-negativity constraints.
    max_iter : int, default=300
        Maximum number of iterations.
    **kwargs : dict
        Additional keyword arguments (currently unused).

    Returns
    -------
    x : ndarray of shape (n_features,)
        Solution vector :math:`\mathbf{x}` that minimizes :math:`\| \mathbf{A} \mathbf{x} - \mathbf{b} \|_2` subject to :math:`\mathbf{x} \geq 0`.

    Raises
    ------
    ConvergenceError
        If the algorithm fails to converge within the maximum number of iterations.
    """

    n = AtA.shape[0]
    x = np.zeros(n, dtype=AtA.dtype)
    s = np.zeros(n, dtype=AtA.dtype)

    P = x > tol
    w = Atb - AtA @ x

    iter_count = 0

    while np.sum(P) < n and np.any(w[~P] > tol):
        # Mask w where P is False
        w_masked = np.where(~P, w, -np.inf)
        i = np.argmax(w_masked)
        if w_masked[i] == -np.inf:
            break  
        P[i] = True

        AtA_P = AtA[np.ix_(P, P)]
        Atb_P = Atb[P]
        try:
            s_P = np.linalg.solve(AtA_P, Atb_P)
        except np.linalg.LinAlgError:
            s_P = np.linalg.lstsq(AtA_P, Atb_P, rcond=None)[0]

        s[P] = s_P
        s[~P] = 0.0

        while np.any(s[P] <= tol):
            iter_count += 1
            if iter_count >= max_iter:
                raise ConvergenceError(
                    f"FNNLS failed to converge after {max_iter} iterations."
                )
            mask = (s <= tol) & P
            if not np.any(mask):
                break

            ind = np.where(mask)[0]
            with np.errstate(divide="ignore", invalid="ignore"):
                alpha = np.min(x[ind] / (x[ind] - s[ind]))
                alpha = np.minimum(alpha, 1.0)

            x += alpha * (s - x)
            x = np.maximum(x, 0.0)  
            P = x > tol

            AtA_P = AtA[np.ix_(P, P)]
            Atb_P = Atb[P]
            try:
                s_P = np.linalg.solve(AtA_P, Atb_P)
            except np.linalg.LinAlgError:
                s_P = np.linalg.lstsq(AtA_P, Atb_P, rcond=None)[0]

            s = np.zeros_like(s)
            s[P] = s_P

        x = s.copy()
        w = Atb - AtA @ x

    if iter_count >= max_iter:
        raise ConvergenceError(f"FNNLS failed to converge after {max_iter} iterations.")

    return x
