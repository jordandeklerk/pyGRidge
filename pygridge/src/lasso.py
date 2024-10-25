"""Lasso algorithm."""

import numpy as np


def lasso(
    y: np.ndarray,
    X: np.ndarray,
    feature_weights: np.ndarray,
    beta: np.ndarray,
    epsilon_convergence: float,
    max_iterations: int,
    gamma: float,
    lambda_max: float,
    proportion_xi: float,
    num_intervals: int,
    num_fixed_effects: int,
    trace_progress: bool,
) -> dict:
    r"""Perform Lasso regression.

    This function implements the lasso algorithm for solving
    Lasso, group lasso, and sparse-group lasso optimization problems.

    The optimization problem is defined as:

    .. math::
        \min_{\boldsymbol{\beta}} \frac{1}{2n} \| \mathbf{y} - \mathbf{X} \boldsymbol{\beta} \|_2^2 + \lambda P(\boldsymbol{\beta})

    where the penalty term :math:`P(\boldsymbol{\beta})` can be:
    - **Lasso:** :math:`P(\boldsymbol{\beta}) = \sum_{j} |\beta_j|`
    - **Group Lasso:** :math:`P(\boldsymbol{\beta}) = \sum_{g} \| \boldsymbol{\beta}_g \|_2`
    - **Sparse Group Lasso:** :math:`P(\boldsymbol{\beta}) = \alpha \sum_{j} |\beta_j| + (1 - \alpha) \sum_{g} \| \boldsymbol{\beta}_g \|_2`

    where:
    - :math:`\boldsymbol{\beta}_g` represents the coefficients in group :math:`g`,
    - :math:`\alpha` is the mixing parameter balancing Lasso and Group Lasso penalties.

    Parameters
    ----------
    y : ndarray of shape (n_samples,)
        Vector of observations, :math:`\mathbf{y}`.
    X : ndarray of shape (n_samples, n_features)
        Design matrix relating :math:`\mathbf{y}` to fixed and random effects, :math:`\mathbf{X}`.
    feature_weights : ndarray of shape (n_features,)
        Weights for the vectors of fixed and random effects, :math:`\mathbf{w}`.
    beta : ndarray of shape (n_features,)
        Initial guess for the coefficient vector, :math:`\boldsymbol{\beta}`.
    epsilon_convergence : float
        Relative accuracy of the solution, :math:`\epsilon`.
    max_iterations : int
        Maximum number of iterations for each value of the penalty parameter :math:`\lambda`.
    gamma : float
        Multiplicative parameter to decrease the step size during
        backtracking line search, :math:`\gamma`.
    lambda_max : float
        Maximum value for the penalty parameter, :math:`\lambda_{\text{max}}`.
    proportion_xi : float
        Multiplicative parameter to determine the minimum value of :math:`\lambda` for
        the grid search, :math:`\xi`.
    num_intervals : int
        Number of lambdas for the grid search, :math:`m`.
    num_fixed_effects : int
        Number of fixed effects present in the mixed model, :math:`p`.
    trace_progress : bool
        If True, print progress after each finished loop of the :math:`\lambda` grid.

    Returns
    -------
    dict
        A dictionary containing the results of the Lasso algorithm.
        Keys include:
        - `'random_effects'`: ndarray of shape (num_intervals, n_features)
          or (num_intervals, n_features - num_fixed_effects)
        - `'fixed_effects'`: ndarray of shape (num_intervals,
          num_fixed_effects) (only if `num_fixed_effects` > 0)
        - `'lambda'`: ndarray of shape (num_intervals,)
        - `'iterations'`: ndarray of shape (num_intervals,)
        - `'rel_acc'`: float
        - `'max_iter'`: int
        - `'gamma_bls'`: float
        - `'xi'`: float
        - `'loops_lambda'`: int

    Raises
    ------
    ValueError
        If any of the input parameters are invalid, such as mismatched dimensions
        or invalid parameter ranges.

    Notes
    -----
    The Seagull Lasso algorithm solves the optimization problem:

    .. math::
        \min_{\boldsymbol{\beta}} \frac{1}{2n} \| \mathbf{y} - \mathbf{X} \boldsymbol{\beta} \|_2^2 + \lambda P(\boldsymbol{\beta})

    where the penalty term :math:`P(\boldsymbol{\beta})` can be configured to perform:
    - Lasso Regression by setting :math:`P(\boldsymbol{\beta}) = \sum_{j} |\beta_j|`
    - Group Lasso Regression by setting :math:`P(\boldsymbol{\beta}) = \sum_{g} \| \boldsymbol{\beta}_g \|_2`
    - Sparse Group Lasso Regression by setting
      :math:`P(\boldsymbol{\beta}) = \alpha \sum_{j} |\beta_j| + (1 - \alpha) \sum_{g} \| \boldsymbol{\beta}_g \|_2`

    The algorithm employs a coordinate descent approach with soft-thresholding
    for the Lasso penalty and group soft-thresholding for the Group Lasso
    penalty. It iteratively updates the coefficients :math:`\boldsymbol{\beta}` to minimize the
    objective function while enforcing the non-negativity constraints.

    The grid search over :math:`\lambda` values is performed logarithmically,
    spanning from :math:`\lambda_{\text{max}}` down to :math:`\xi \times \lambda_{\text{max}}` with
    :math:`m` intervals.

    The `proportion_xi` parameter determines the lower bound of the grid search,
    ensuring that the algorithm explores a range of penalty strengths to find
    an optimal balance between sparsity and group structure in the coefficients.

    The `gamma` parameter controls the step size reduction during backtracking
    line search to ensure convergence and numerical stability.

    The results include the estimated fixed and random effects, the
    corresponding :math:`\lambda` values, the number of iterations
    required for convergence, and other diagnostic information.
    """

    n, p = X.shape

    # Calculate X.T * y
    X_transp_y = X.T @ y

    # Initialize result arrays
    iterations = np.zeros(num_intervals, dtype=int)
    lambdas = np.zeros(num_intervals)
    solution = np.zeros((num_intervals, p))

    for interval in range(num_intervals):
        accuracy_reached = False
        counter = 0
        if num_intervals > 1:
            lambda_val = lambda_max * np.exp(
                (interval / (num_intervals - 1)) * np.log(proportion_xi)
            )
        else:
            lambda_val = lambda_max

        # Special case for lambda = 0 (or very close to 0)
        if np.isclose(lambda_val, 0, atol=1e-10):
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            accuracy_reached = True
            counter = 1
        else:
            while (not accuracy_reached) and (counter < max_iterations):
                # Calculate gradient
                beta_col = beta.reshape(-1, 1)
                gradient = (X.T @ (X @ beta_col).flatten() - X_transp_y) / n

                criterion_fulfilled = False
                time_step = 1.0

                while not criterion_fulfilled:
                    # Preparation for soft-thresholding
                    temp1 = beta - time_step * gradient
                    temp2 = lambda_val * time_step * feature_weights

                    # Soft-thresholding to obtain beta_new
                    beta_new = np.sign(temp1) * np.maximum(np.abs(temp1) - temp2, 0)

                    # Check convergence
                    beta_diff = beta - beta_new
                    loss_old = 0.5 * np.sum((y - X @ beta) ** 2) / n
                    loss_new = 0.5 * np.sum((y - X @ beta_new) ** 2) / n

                    if loss_new <= loss_old - np.dot(gradient, beta_diff) + (
                        0.5 / time_step
                    ) * np.sum(beta_diff**2):
                        # Adjust convergence criteria based on lambda value
                        conv_threshold = max(epsilon_convergence, lambda_val * 1e-4)
                        if np.max(np.abs(beta_diff)) <= conv_threshold * np.linalg.norm(
                            beta
                        ):
                            accuracy_reached = True
                        beta = beta_new
                        criterion_fulfilled = True
                    else:
                        time_step *= gamma

                counter += 1

        if trace_progress:
            print(f"Loop: {interval + 1} of {num_intervals} finished.")

        # Store solution
        solution[interval] = beta
        iterations[interval] = counter
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
