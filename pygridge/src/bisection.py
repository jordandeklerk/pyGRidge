"""Bisection algorithm for finding the smallest positive root of a polynomial."""

import numpy as np
from typing import Union, Optional
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array


class BisectionError(Exception):
    """Base exception class for bisection-related errors."""

    pass


class ConvergenceError(BisectionError):
    """Exception raised when the bisection algorithm fails to converge."""

    pass


class BisectionSolver(BaseEstimator):
    """Solver for finding the smallest positive root of a polynomial using bisection.

    Parameters
    ----------
    tol : float, default=1e-13
        The tolerance for convergence. The algorithm stops when the interval
        containing the root is smaller than tol.
    max_iter : int, default=10000
        The maximum number of iterations.
    scale_tol : bool, default=True
        Whether to scale the tolerance with the maximum value of the inputs.

    Attributes
    ----------
    n_iter_ : int
        Number of iterations performed.

    Examples
    --------
    >>> import numpy as np
    >>> from pygridge.src.bisection import BisectionSolver
    >>> solver = BisectionSolver()
    >>> rows = 3
    >>> alpha = 0.5
    >>> left_border = 0.0
    >>> right_border = 1.0
    >>> group_weight = 1.0
    >>> vector_weights = np.array([1.0, 1.0, 1.0])
    >>> vector_in = np.array([0.5, 0.3, 0.2])
    >>> root = solver.solve(rows, alpha, left_border, right_border,
    ...                    group_weight, vector_weights, vector_in)
    >>> print(f"Found root: {root:.6f}")
    Found root: 0.500000
    """

    def __init__(
        self,
        tol: float = 1e-13,
        max_iter: int = 10000,
        scale_tol: bool = True,
    ):
        self.tol = tol
        self.max_iter = max_iter
        self.scale_tol = scale_tol

    def solve(
        self,
        rows: int,
        alpha: float,
        left_border: float,
        right_border: float,
        group_weight: float,
        vector_weights: np.ndarray,
        vector_in: np.ndarray,
    ) -> float:
        r"""Find the smallest positive root of a polynomial using bisection.

        This method finds the smallest positive root of a polynomial of second degree in
        :math:`\lambda`. Bisection is an implicit algorithm, i.e., it calls itself until a
        certain precision is reached.

        Parameters
        ----------
        rows : int
            The length of the input vectors, :math:`n`.
        alpha : float
            Mixing parameter of the penalty terms. Must satisfy :math:`0 < \alpha < 1`.
            The penalty term is defined as:
            :math:`\alpha \times \text{"lasso penalty"} + (1 - \alpha) \times \text{"group lasso penalty"}`.
        left_border : float
            Value of the left border of the current interval that for sure harbors a root,
            :math:`\lambda_{\text{left}}`.
        right_border : float
            Value of the right border of the current interval that for sure harbors a root,
            :math:`\lambda_{\text{right}}`.
        group_weight : float
            A multiplicative scalar which is part of the polynomial, :math:`g`.
        vector_weights : ndarray of shape (rows,)
            An input vector of multiplicative scalars which are part of the polynomial,
            :math:`\mathbf{w}`. This vector is a subset of the vector of weights for features.
        vector_in : ndarray of shape (rows,)
            Another input vector which is required to compute the value of the polynomial,
            :math:`\mathbf{x}`.

        Returns
        -------
        float
            The smallest positive root of the polynomial, or the center point of
            the interval containing the root if a certain precision is reached.

        Raises
        ------
        ValueError
            If :math:`\alpha` is not between 0 and 1 (exclusive), if :math:`\text{left\_border}` is greater
            than or equal to :math:`\text{right\_border}`, or if the lengths of `vector_weights` and
            `vector_in` do not match the specified number of rows.
        ConvergenceError
            If the bisection algorithm does not converge after the maximum number
            of iterations.

        Notes
        -----
        The algorithm uses a bisection method to find the root of the polynomial:

        .. math::
            f(\lambda) = \sum_{i=1}^{n} \left( \max\left(0, |\mathbf{x}_i| - \alpha \lambda \mathbf{w}_i\right) \right)^2 - (1 - \alpha)^2 \lambda^2 g

        where:
        - :math:`\mathbf{x}_i` are the elements of `vector_in`,
        - :math:`\mathbf{w}_i` are the elements of `vector_weights`,
        - :math:`g` is the `group_weight`,
        - and :math:`\lambda` is the variable for which the root is sought.

        The algorithm iteratively narrows down the interval containing the root by evaluating
        the function at the midpoint and adjusting the borders based on the sign change.
        """
        if not (0 < alpha < 1):
            raise ValueError("alpha must be between 0 and 1 (exclusive)")
        if left_border >= right_border:
            raise ValueError("left_border must be less than right_border")

        vector_weights = check_array(
            vector_weights.reshape(1, -1), ensure_2d=True
        ).ravel()
        vector_in = check_array(vector_in.reshape(1, -1), ensure_2d=True).ravel()

        if rows != len(vector_weights) or rows != len(vector_in):
            raise ValueError(
                "rows must match the length of vector_weights and vector_in"
            )

        max_value = max(abs(left_border), abs(right_border), np.max(np.abs(vector_in)))
        tolerance = max(self.tol, self.tol * max_value) if self.scale_tol else self.tol

        self.n_iter_ = 0

        for _ in range(self.max_iter):
            mid_point = 0.5 * (left_border + right_border)
            func_left = 0.0
            func_mid = 0.0
            func_right = 0.0

            for i in range(rows):
                if vector_in[i] < 0.0:
                    temp_left = -vector_in[i] - alpha * left_border * vector_weights[i]
                    temp_mid = -vector_in[i] - alpha * mid_point * vector_weights[i]
                    temp_right = (
                        -vector_in[i] - alpha * right_border * vector_weights[i]
                    )
                else:
                    temp_left = vector_in[i] - alpha * left_border * vector_weights[i]
                    temp_mid = vector_in[i] - alpha * mid_point * vector_weights[i]
                    temp_right = vector_in[i] - alpha * right_border * vector_weights[i]

                func_left += max(0, temp_left) ** 2
                func_mid += max(0, temp_mid) ** 2
                func_right += max(0, temp_right) ** 2

            func_left -= (1.0 - alpha) ** 2 * left_border**2 * group_weight
            func_mid -= (1.0 - alpha) ** 2 * mid_point**2 * group_weight
            func_right -= (1.0 - alpha) ** 2 * right_border**2 * group_weight

            # Check for sign changes and update interval
            if func_left * func_mid <= 0.0:
                if abs(left_border - mid_point) <= tolerance:
                    self.n_iter_ += 1
                    return mid_point
                right_border = mid_point
            elif func_mid * func_right <= 0.0:
                if abs(mid_point - right_border) <= tolerance:
                    self.n_iter_ += 1
                    return mid_point
                left_border = mid_point
            else:
                # If no sign change, return the point with smallest absolute function value
                abs_func_left = abs(func_left)
                abs_func_mid = abs(func_mid)
                abs_func_right = abs(func_right)
                min_abs_func = min(abs_func_left, abs_func_mid, abs_func_right)

                self.n_iter_ += 1
                if min_abs_func == abs_func_left:
                    return left_border
                elif min_abs_func == abs_func_mid:
                    return mid_point
                else:
                    return right_border

            self.n_iter_ += 1

        raise ConvergenceError(
            f"Bisection did not converge after {self.max_iter} iterations"
        )
