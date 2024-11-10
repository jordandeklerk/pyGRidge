"""Main function to run lasso, group lasso, or sparse-group lasso."""

import numpy as np
import warnings
from .lambda_max_lasso import lambda_max_lasso
from .lambda_max_group_lasso import lambda_max_group_lasso
from .lambda_max_sparse_group_lasso import lambda_max_sparse_group_lasso
from .lasso import lasso
from .group_lasso import group_lasso
from .sparse_group_lasso import sparse_group_lasso


def seagull(
    y,
    X=None,
    Z=None,
    weights_u=None,
    groups=None,
    alpha=1.0,  
    rel_acc=0.0001,
    max_lambda=None,
    xi=0.01,
    loops_lambda=50,
    max_iter=1000,
    gamma_bls=0.8,
    trace_progress=False,
):
    r"""Mixed model fitting with lasso, group lasso, or sparse-group lasso regularization.

    This function fits a mixed model using lasso, group lasso, or sparse-group lasso
    regularization based on the provided parameters.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Target variable.
    X : array-like of shape (n_samples, n_features_fixed), optional
        Fixed effects design matrix.
    Z : array-like of shape (n_samples, n_features_random)
        Random effects design matrix.
    weights_u : array-like of shape (n_features_random), optional
        Weights for the random effects. If None, all weights are set to 1.
    groups : array-like of shape (n_features), optional
        Group labels for each feature. Required when alpha < 1.
    alpha : float, default=1.0
        Mixing parameter between lasso and group lasso. alpha=1 is lasso, alpha=0 is group lasso.
    rel_acc : float, default=0.0001
        Relative accuracy for convergence criterion.
    max_lambda : float, optional
        Maximum value for the regularization parameter lambda.
    xi : float, default=0.01
        Minimum value for lambda as a fraction of max_lambda.
    loops_lambda : int, default=50
        Number of lambda values to use in the regularization path.
    max_iter : int, default=1000
        Maximum number of iterations for each lambda value.
    gamma_bls : float, default=0.8
        Line search parameter for backtracking.
    trace_progress : bool, default=False
        If True, print progress information during fitting.

    Returns
    -------
    dict
        A dictionary containing the results of the fitting process.

    Notes
    -----
    The function solves the following optimization problem:

    .. math::
        \min_{\beta, u} \frac{1}{2} \|y - X\beta - Zu\|_2^2 + \lambda P_{\alpha}(\beta, u)

    where :math:`P_{\alpha}(\beta, u)` is the penalty term:

    .. math::
        P_{\alpha}(\beta, u) = \alpha \|\beta\|_1 + (1-\alpha) \sum_{g=1}^G \sqrt{p_g} \|\beta_g\|_2

    Here, :math:`\beta` are the fixed effects, :math:`u` are the random effects,
    :math:`G` is the number of groups, and :math:`p_g` is the size of group :math:`g`.
    """

    # Helper function to check if input is numeric
    def is_numeric(arr):
        return isinstance(arr, (list, np.ndarray)) and np.issubdtype(
            np.array(arr).dtype, np.number
        )

    # Check vector y
    if y is None:
        raise ValueError("Vector y is missing. Please provide y and try again.")
    y = np.asarray(y)
    if y.size == 0:
        raise ValueError(
            "Vector y is empty. Please provide a non-empty y and try again."
        )
    if y.ndim > 1 and not (y.shape[0] == 1 or y.shape[1] == 1):
        raise ValueError("Input y should be a one-dimensional array.")
    if not np.issubdtype(y.dtype, np.number):
        raise ValueError("Non-numeric values detected in vector y.")
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("NA, NaN, or Inf detected in vector y.")

    # Check matrix X
    if X is None:
        p1 = 0
        X = None
    else:
        if not is_numeric(X):
            raise ValueError("Non-numeric values detected in matrix X.")
        X = np.atleast_2d(X)
        p1 = X.shape[1]
        if X.shape[0] != y.size:
            raise ValueError("Mismatching dimensions of vector y and matrix X.")
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("NA, NaN, or Inf detected in matrix X.")

    # Check matrix Z
    if Z is None:
        raise ValueError("Matrix Z is missing. Please provide Z and try again.")
    if not is_numeric(Z):
        raise ValueError("Non-numeric values detected in matrix Z.")
    Z = np.atleast_2d(Z)
    if Z.ndim != 2:
        raise ValueError("Input Z should be a two-dimensional array")
    p2 = Z.shape[1]
    if Z.size == 0:
        raise ValueError(
            "Matrix Z is empty. Please provide a non-empty Z and try again."
        )
    if Z.shape[0] != y.size:
        raise ValueError("Mismatching dimensions of vector y and matrix Z.")
    if np.any(np.isnan(Z)) or np.any(np.isinf(Z)):
        raise ValueError("NA, NaN, or Inf detected in matrix Z.")

    # Check vector weights_u
    if weights_u is None:
        weights_u = np.ones(p2)
    else:
        if not is_numeric(weights_u):
            raise ValueError("Non-numeric values detected in vector weights_u.")
        weights_u = np.asarray(weights_u).flatten()
        if weights_u.ndim > 1:
            raise ValueError("Input weights_u should be a one-dimensional array.")
        if weights_u.size != p2:
            raise ValueError("Mismatching dimensions of matrix Z and vector weights_u.")
        if np.any(np.isnan(weights_u)) or np.any(np.isinf(weights_u)):
            raise ValueError("NA, NaN, or Inf detected in vector weights_u.")
        if np.any(weights_u <= 0):
            raise ValueError("Weights <= 0 detected in weights_u.")

    # Check alpha
    if alpha is None:
        warnings.warn("The parameter alpha is None. Reset to default value (=1.0).")
        alpha = 1.0
    else:
        try:
            alpha = float(alpha)
        except (TypeError, ValueError):
            warnings.warn(
                "The parameter alpha is non-numeric. Reset to default value (=1.0)."
            )
            alpha = 1.0
        else:
            if not (0.0 <= alpha <= 1.0):
                warnings.warn(
                    "The parameter alpha is out of range. Reset to default value (=1.0)."
                )
                alpha = 1.0

    # Check groups only if alpha < 1 (not when alpha = 1)
    if alpha < 1.0:
        if groups is None:
            raise ValueError(
                "Vector groups is missing. Please provide groups when alpha < 1."
            )
        if not is_numeric(groups):
            raise ValueError("Non-numeric values detected in vector groups.")
        groups = np.asarray(groups)
        if not np.issubdtype(groups.dtype, np.number):
            raise ValueError("Non-numeric values detected in vector groups.")
        groups = groups.astype(int).flatten()
        if groups.ndim > 1:
            raise ValueError("Input groups should be a one-dimensional array.")
        if np.any(np.isnan(groups)) or np.any(np.isinf(groups)):
            raise ValueError("NA, NaN, or Inf detected in vector groups.")

    # Check rel_acc
    if rel_acc is None:
        warnings.warn(
            "The parameter rel_acc is None. Reset to default value (=0.0001)."
        )
        rel_acc = 0.0001
    else:
        try:
            rel_acc = float(rel_acc)
        except (TypeError, ValueError):
            warnings.warn(
                "The parameter rel_acc is non-numeric. Reset to default value"
                " (=0.0001)."
            )
            rel_acc = 0.0001
        else:
            if rel_acc <= 0.0:
                warnings.warn(
                    "The parameter rel_acc is non-positive. Reset to default value"
                    " (=0.0001)."
                )
                rel_acc = 0.0001

    # Check max_iter
    if max_iter is None:
        warnings.warn("The parameter max_iter is None. Reset to default value (=1000).")
        max_iter = 1000
    else:
        try:
            max_iter = int(max_iter)
        except (TypeError, ValueError):
            warnings.warn(
                "The parameter max_iter is non-numeric. Reset to default value (=1000)."
            )
            max_iter = 1000
        else:
            if max_iter <= 0:
                warnings.warn(
                    "The parameter max_iter is non-positive. Reset to default value"
                    " (=1000)."
                )
                max_iter = 1000

    # Check gamma_bls
    if gamma_bls is None:
        warnings.warn("The parameter gamma_bls is None. Reset to default value (=0.8).")
        gamma_bls = 0.8
    else:
        try:
            gamma_bls = float(gamma_bls)
        except (TypeError, ValueError):
            warnings.warn(
                "The parameter gamma_bls is non-numeric. Reset to default value (=0.8)."
            )
            gamma_bls = 0.8
        else:
            if not (0.0 < gamma_bls < 1.0):
                warnings.warn(
                    "The parameter gamma_bls is out of range. Reset to default value"
                    " (=0.8)."
                )
                gamma_bls = 0.8

    # Create correct input
    if p1 == 0:
        weights_u_tilde = weights_u.copy()
        p = p2
    else:
        p = p1 + p2
        weights_u_tilde = np.zeros(p)
        weights_u_tilde[p1:] = weights_u
    if X is not None:
        X_tilde = np.hstack((X, Z))
    else:
        X_tilde = Z.copy()
    b_tilde = np.zeros(p)

    index_permutation = np.arange(1, p + 1)

    if alpha < 1.0:  
        # Assign all fixed effects to one group if not assigned
        if p1 > 0 and groups.size == p2:
            groups_temp = np.full(p1, groups.min() - 1, dtype=int)
            groups = np.concatenate((groups_temp, groups))
        # Ensure positivity of group assignments
        min_group = groups.min()
        if min_group <= 0:
            groups = groups - min_group + 1
        # Sort by group number
        index_permutation = np.argsort(groups) + 1
        if p > 1:
            X_tilde = X_tilde[:, index_permutation - 1]
            groups = groups[index_permutation - 1]
            weights_u_tilde = weights_u_tilde[index_permutation - 1]
        # Renumber groups
        unique_groups = np.unique(groups)
        group_mapping = {old: new for new, old in enumerate(unique_groups, start=1)}
        groups = np.array([group_mapping[g] for g in groups], dtype=int)

    # Check max_lambda
    if max_lambda is None:
        if alpha == 1.0:
            max_lambda = lambda_max_lasso(y, weights_u_tilde, b_tilde, X_tilde)
        elif alpha == 0.0:
            max_lambda = lambda_max_group_lasso(
                y, groups, weights_u_tilde, b_tilde, X_tilde
            )
        else:
            max_lambda = lambda_max_sparse_group_lasso(
                alpha, y, groups, weights_u_tilde, b_tilde, X_tilde
            )
    else:
        try:
            max_lambda = float(max_lambda)
        except (TypeError, ValueError):
            warnings.warn(
                "The parameter max_lambda is non-numeric. Using default algorithm to"
                " calculate max_lambda."
            )
            if alpha == 1.0:
                max_lambda = lambda_max_lasso(y, weights_u_tilde, b_tilde, X_tilde)
            elif alpha == 0.0:
                max_lambda = lambda_max_group_lasso(
                    y, groups, weights_u_tilde, b_tilde, X_tilde
                )
            else:
                max_lambda = lambda_max_sparse_group_lasso(
                    alpha, y, groups, weights_u_tilde, b_tilde, X_tilde
                )
        else:
            if max_lambda <= 0.0 or np.isnan(max_lambda) or np.isinf(max_lambda):
                warnings.warn(
                    "The parameter max_lambda is invalid. Using default algorithm to"
                    " calculate max_lambda."
                )
                if alpha == 1.0:
                    max_lambda = lambda_max_lasso(y, weights_u_tilde, b_tilde, X_tilde)
                elif alpha == 0.0:
                    max_lambda = lambda_max_group_lasso(
                        y, groups, weights_u_tilde, b_tilde, X_tilde
                    )
                else:
                    max_lambda = lambda_max_sparse_group_lasso(
                        alpha, y, groups, weights_u_tilde, b_tilde, X_tilde
                    )

    # Check xi
    if xi is None:
        warnings.warn("The parameter xi is None. Reset to default value (=0.01).")
        xi = 0.01
    else:
        try:
            xi = float(xi)
        except (TypeError, ValueError):
            warnings.warn(
                "The parameter xi is non-numeric. Reset to default value (=0.01)."
            )
            xi = 0.01
        else:
            if not (0.0 < xi <= 1.0):
                warnings.warn(
                    "The parameter xi is out of range. Reset to default value (=0.01)."
                )
                xi = 0.01

    # Check loops_lambda
    if xi == 1.0:
        warnings.warn(
            "Since the parameter xi = 1, the parameter loops_lambda will be set to 1."
        )
        loops_lambda = 1
    else:
        if loops_lambda is None:
            warnings.warn(
                "The parameter loops_lambda is None. Reset to default value (=50)."
            )
            loops_lambda = 50
        else:
            try:
                loops_lambda = int(loops_lambda)
            except (TypeError, ValueError):
                warnings.warn(
                    "The parameter loops_lambda is non-numeric. Reset to default value"
                    " (=50)."
                )
                loops_lambda = 50
            else:
                if loops_lambda <= 0:
                    warnings.warn(
                        "The parameter loops_lambda is non-positive. Reset to default"
                        " value (=50)."
                    )
                    loops_lambda = 50

    # Check trace_progress
    if not isinstance(trace_progress, bool):
        warnings.warn(
            "The parameter trace_progress is not boolean. Reset to default value"
            " (=False)."
        )
        trace_progress = False

    if alpha == 1.0:
        res = lasso(
            y=y,
            X=X_tilde,
            feature_weights=weights_u_tilde,
            beta=b_tilde,
            epsilon_convergence=rel_acc,
            max_iterations=max_iter,
            gamma=gamma_bls,
            lambda_max=max_lambda,
            proportion_xi=xi,
            num_intervals=loops_lambda,
            num_fixed_effects=p1,
            trace_progress=trace_progress,
        )
        res["result"] = "lasso"
        # Rename lambda to lambda_values for consistency
        res["lambda_values"] = res.pop("lambda")
        # Add beta key containing the coefficients
        if "fixed_effects" in res and "random_effects" in res:
            res["beta"] = np.concatenate(
                [res["fixed_effects"][-1], res["random_effects"][-1]]
            )
        else:
            res["beta"] = res["random_effects"][-1]
    elif alpha == 0.0:
        res = group_lasso(
            y=y,
            X=X_tilde,
            feature_weights=weights_u_tilde,
            groups=groups,
            beta=b_tilde,
            index_permutation=index_permutation,
            epsilon_convergence=rel_acc,
            max_iterations=max_iter,
            gamma=gamma_bls,
            lambda_max=max_lambda,
            proportion_xi=xi,
            num_intervals=loops_lambda,
            num_fixed_effects=p1,
            trace_progress=trace_progress,
        )
        res["result"] = "group_lasso"
        res["lambda_values"] = res.pop("lambda")
        if "fixed_effects" in res and "random_effects" in res:
            res["beta"] = np.concatenate(
                [res["fixed_effects"][-1], res["random_effects"][-1]]
            )
        else:
            res["beta"] = res["random_effects"][-1]
    else:
        res = sparse_group_lasso(
            y=y,
            X=X_tilde,
            feature_weights=weights_u_tilde,
            groups=groups,
            beta=b_tilde,
            index_permutation=index_permutation,
            alpha=alpha,
            epsilon_convergence=rel_acc,
            max_iterations=max_iter,
            gamma=gamma_bls,
            lambda_max=max_lambda,
            proportion_xi=xi,
            num_intervals=loops_lambda,
            num_fixed_effects=p1,
            trace_progress=trace_progress,
        )
        res["result"] = "sparse_group_lasso"
        res["lambda_values"] = res.pop("lambda")
        if "fixed_effects" in res and "random_effects" in res:
            res["beta"] = np.concatenate(
                [res["fixed_effects"][-1], res["random_effects"][-1]]
            )
        else:
            res["beta"] = res["random_effects"][-1]

    return res
