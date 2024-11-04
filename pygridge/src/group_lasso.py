"""Group lasso algorithm."""

import numpy as np
from typing import Optional, Dict, Any
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.exceptions import NotFittedError
from .lambda_max_group_lasso import lambda_max_group_lasso


class GroupLassoError(Exception):
    """Base exception class for group lasso errors."""

    pass


class ConvergenceError(GroupLassoError):
    """Exception raised when the algorithm fails to converge."""

    pass


def group_lasso(
    y: np.ndarray,
    X: np.ndarray,
    feature_weights: np.ndarray,
    groups: np.ndarray,
    beta: np.ndarray,
    index_permutation: np.ndarray,
    epsilon_convergence: float = 1e-4,
    max_iterations: int = 1000,
    gamma: float = 0.8,
    lambda_max: Optional[float] = None,
    proportion_xi: float = 0.01,
    num_intervals: int = 50,
    num_fixed_effects: int = 0,
    trace_progress: bool = False,
) -> Dict[str, Any]:
    """Core group lasso algorithm.

    Parameters
    ----------
    y : ndarray of shape (n_samples,)
        Target values.
    X : ndarray of shape (n_samples, n_features)
        Training data.
    feature_weights : ndarray of shape (n_features,)
        Weights for the vectors of fixed and random effects.
    groups : ndarray of shape (n_features,)
        Integer vector specifying which effect belongs to which group.
    beta : ndarray of shape (n_features,)
        Initial coefficient values.
    index_permutation : ndarray of shape (n_features,)
        Permutation of feature indices.
    epsilon_convergence : float, default=1e-4
        Tolerance for convergence.
    max_iterations : int, default=1000
        Maximum number of iterations.
    gamma : float, default=0.8
        Line search parameter.
    lambda_max : float, optional
        Maximum value for the regularization parameter.
    proportion_xi : float, default=0.01
        Minimum value for lambda as a fraction of lambda_max.
    num_intervals : int, default=50
        Number of lambda values to use.
    num_fixed_effects : int, default=0
        Number of fixed effects.
    trace_progress : bool, default=False
        Whether to print progress information.

    Returns
    -------
    dict
        Dictionary containing the results of the optimization.
    """
    n, p = X.shape

    # Initialize variables
    num_groups = np.max(groups)
    index_start = np.zeros(num_groups, dtype=int)
    index_end = np.zeros(num_groups, dtype=int)
    group_sizes = np.zeros(num_groups, dtype=int)
    group_weights = np.zeros(num_groups)

    # Create vectors of group sizes, start indices, end indices, and group weights
    for i in range(num_groups):
        group_mask = groups == (i + 1)
        group_sizes[i] = np.sum(group_mask)
        index_start[i] = np.where(group_mask)[0][0]
        index_end[i] = np.where(group_mask)[0][-1]
        group_weights[i] = np.sqrt(np.sum(feature_weights[group_mask]))

    # Calculate X.T * y
    X_transp_y = X.T @ y

    # Initialize result matrices
    iterations = np.zeros(num_intervals, dtype=int)
    lambdas = np.zeros(num_intervals)
    solution = np.zeros((num_intervals, p))

    # Compute lambda_max if not provided
    if lambda_max is None:
        lambda_max = lambda_max_group_lasso(
            y=y,
            groups=groups,
            feature_weights=feature_weights,
            beta=beta.copy(),
            X=X,
        )

    for interval in range(num_intervals):
        accuracy_reached = False
        counter = 1
        if num_intervals > 1:
            lambda_val = lambda_max * np.exp(
                (interval / (num_intervals - 1)) * np.log(proportion_xi)
            )
        else:
            lambda_val = lambda_max

        while (not accuracy_reached) and (counter <= max_iterations):
            # Calculate gradient
            beta_col = beta.reshape(-1, 1)
            gradient = (X.T @ (X @ beta_col).flatten() - X_transp_y) / n

            criterion_fulfilled = False
            time_step = 1.0

            while not criterion_fulfilled:
                beta_new = np.zeros_like(beta)

                # Soft-scaling in groups
                for i in range(num_groups):
                    start, end = index_start[i], index_end[i] + 1
                    temp = beta[start:end] - time_step * gradient[start:end]
                    l2_norm = np.linalg.norm(temp)
                    threshold = time_step * lambda_val * group_weights[i]

                    if l2_norm > threshold:
                        scaling = 1.0 - threshold / l2_norm
                        beta_new[start:end] = scaling * temp

                # Check convergence
                beta_diff = beta - beta_new
                loss_old = 0.5 * np.sum((y - X @ beta) ** 2) / n
                loss_new = 0.5 * np.sum((y - X @ beta_new) ** 2) / n

                if loss_new <= loss_old - np.dot(gradient, beta_diff) + (
                    0.5 / time_step
                ) * np.sum(beta_diff**2):
                    if np.max(
                        np.abs(beta_diff)
                    ) <= epsilon_convergence * np.linalg.norm(beta):
                        accuracy_reached = True
                    beta = beta_new
                    criterion_fulfilled = True
                else:
                    time_step *= gamma

            counter += 1

        if trace_progress:
            print(f"Loop: {interval + 1} of {num_intervals} finished.")

        # Store solution
        solution[interval] = beta[index_permutation - 1]
        iterations[interval] = counter - 1
        lambdas[interval] = lambda_val

    # Prepare results
    if num_fixed_effects == 0:
        return {
            "random_effects": solution,
            "lambda": lambdas,
            "iterations": iterations,
            "rel_acc": epsilon_convergence,
            "max_iter": max_iterations,
            "gamma_bls": gamma,
            "xi": proportion_xi,
            "loops_lambda": num_intervals,
        }
    else:
        return {
            "fixed_effects": solution[:, :num_fixed_effects],
            "random_effects": solution[:, num_fixed_effects:],
            "lambda": lambdas,
            "iterations": iterations,
            "rel_acc": epsilon_convergence,
            "max_iter": max_iterations,
            "gamma_bls": gamma,
            "xi": proportion_xi,
            "loops_lambda": num_intervals,
        }


class GroupLasso(BaseEstimator, RegressorMixin):
    r"""Group Lasso regression.

    This estimator implements the group lasso algorithm for solving
    group lasso optimization problems.

    The optimization problem is defined as:

    .. math::
        \min_{\boldsymbol{\beta}} \frac{1}{2n} \| \mathbf{y} - \mathbf{X} \boldsymbol{\beta} \|_2^2 + \lambda P(\boldsymbol{\beta})

    where the penalty term :math:`P(\boldsymbol{\beta})` is:

    .. math::
        P(\boldsymbol{\beta}) = \sum_{g} \| \boldsymbol{\beta}_g \|_2

    where :math:`\boldsymbol{\beta}_g` represents the coefficients in group :math:`g`.

    Parameters
    ----------
    feature_weights : ndarray of shape (n_features,) or None, default=None
        Weights for the vectors of fixed and random effects, :math:`\mathbf{w}`.
        If None, all weights are set to 1.
    groups : ndarray of shape (n_features,), default=None
        Integer vector specifying which effect belongs to which group, :math:`\mathbf{G}`.
        If None, each feature is treated as its own group.
    tol : float, default=1e-4
        Tolerance for the optimization. The algorithm stops when the relative
        change in the parameters is less than tol.
    max_iter : int, default=1000
        Maximum number of iterations for each value of the penalty parameter :math:`\lambda`.
    gamma : float, default=0.8
        Multiplicative parameter to decrease the step size during backtracking
        line search, :math:`\gamma`. Must be between 0 and 1.
    lambda_max : float or None, default=None
        Maximum value for the penalty parameter, :math:`\lambda_{\text{max}}`.
        If None, it is computed from the data using lambda_max_group_lasso.
    proportion_xi : float, default=0.01
        Multiplicative parameter to determine the minimum value of :math:`\lambda` for the
        grid search, :math:`\xi`. Must be between 0 and 1.
    num_intervals : int, default=50
        Number of lambdas for the grid search, :math:`m`.
    num_fixed_effects : int, default=0
        Number of fixed effects present in the mixed model, :math:`p`.
    verbose : bool, default=False
        If True, prints progress information during fitting.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Parameter vector (:math:`\boldsymbol{\beta}`).
    n_iter_ : int
        Number of iterations run for the optimal lambda value.
    lambda_path_ : ndarray of shape (num_intervals,)
        The values of lambda used in the optimization path.
    coef_path_ : ndarray of shape (num_intervals, n_features)
        The values of the coefficients along the optimization path.
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during fit.

    Examples
    --------
    >>> import numpy as np
    >>> from pygridge.src.group_lasso import GroupLasso
    >>> X = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    >>> y = np.array([1, 2])
    >>> groups = np.array([1, 1, 2, 2])  # Two groups of two features each
    >>> clf = GroupLasso(groups=groups)
    >>> clf.fit(X, y)
    GroupLasso(...)
    """

    def __init__(
        self,
        feature_weights: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
        tol: float = 1e-4,
        max_iter: int = 1000,
        gamma: float = 0.8,
        lambda_max: Optional[float] = None,
        proportion_xi: float = 0.01,
        num_intervals: int = 50,
        num_fixed_effects: int = 0,
        verbose: bool = False,
    ):
        self.feature_weights = feature_weights
        self.groups = groups
        self.tol = tol
        self.max_iter = max_iter
        self.gamma = gamma
        self.lambda_max = lambda_max
        self.proportion_xi = proportion_xi
        self.num_intervals = num_intervals
        self.num_fixed_effects = num_fixed_effects
        self.verbose = verbose

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GroupLasso":
        """Fit the group lasso model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns self.
        """
        # Input validation
        X, y = check_X_y(X, y, accept_sparse=False)
        self._validate_params()

        n_samples, n_features = X.shape

        # Initialize or validate feature weights
        if self.feature_weights is None:
            feature_weights = np.ones(n_features)
        else:
            feature_weights = check_array(
                self.feature_weights.reshape(1, -1), ensure_2d=True
            ).ravel()
            if len(feature_weights) != n_features:
                raise ValueError(
                    f"feature_weights has wrong shape. Got {len(feature_weights)}"
                    f" features, expected {n_features}"
                )

        # Initialize or validate groups
        if self.groups is None:
            groups = np.arange(1, n_features + 1)  # Each feature is its own group
        else:
            groups = check_array(self.groups.reshape(1, -1), ensure_2d=True).ravel()
            if len(groups) != n_features:
                raise ValueError(
                    f"groups has wrong shape. Got {len(groups)} features, expected"
                    f" {n_features}"
                )

        # Initialize beta
        beta = np.zeros(n_features)
        index_permutation = np.arange(1, n_features + 1)

        # Run group lasso algorithm
        result = group_lasso(
            y=y,
            X=X,
            feature_weights=feature_weights,
            groups=groups,
            beta=beta,
            index_permutation=index_permutation,
            epsilon_convergence=self.tol,
            max_iterations=self.max_iter,
            gamma=self.gamma,
            lambda_max=self.lambda_max,
            proportion_xi=self.proportion_xi,
            num_intervals=self.num_intervals,
            num_fixed_effects=self.num_fixed_effects,
            trace_progress=self.verbose,
        )

        # Store results
        if self.num_fixed_effects == 0:
            self.coef_ = result["random_effects"][-1]  # Use last solution
            self.coef_path_ = result["random_effects"]
        else:
            fixed_effects = result["fixed_effects"][-1]
            random_effects = result["random_effects"][-1]
            self.coef_ = np.concatenate([fixed_effects, random_effects])
            self.coef_path_ = np.column_stack(
                [result["fixed_effects"], result["random_effects"]]
            )

        self.lambda_path_ = result["lambda"]
        self.n_iter_ = result["iterations"][-1]

        # Store feature names and number of features
        self.n_features_in_ = n_features
        self.feature_names_in_ = np.arange(n_features)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the group lasso model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Returns predicted values.

        Raises
        ------
        NotFittedError
            If the model is not fitted yet.
        ValueError
            If the input features do not match the training features.
        """
        check_is_fitted(self, ["coef_", "n_features_in_"])
        X = check_array(X, accept_sparse=False)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but GroupLasso is expecting"
                f" {self.n_features_in_} features as input"
            )

        return X @ self.coef_

    def _validate_params(self) -> None:
        """Validate parameters.

        Raises
        ------
        ValueError
            If any parameter is invalid.
        """
        if self.tol <= 0:
            raise ValueError("tol must be positive")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if not 0 < self.gamma < 1:
            raise ValueError("gamma must be between 0 and 1")
        if not 0 < self.proportion_xi <= 1:
            raise ValueError("proportion_xi must be between 0 and 1")
        if self.num_intervals <= 0:
            raise ValueError("num_intervals must be positive")
        if self.num_fixed_effects < 0:
            raise ValueError("num_fixed_effects must be non-negative")
        if self.lambda_max is not None and self.lambda_max <= 0:
            raise ValueError("lambda_max must be positive")
