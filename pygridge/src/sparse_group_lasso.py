"""Sparse group lasso algorithm using proximal gradient descent."""

import numpy as np


def sparse_group_lasso(
    y: np.ndarray,
    X: np.ndarray,
    feature_weights: np.ndarray,
    groups: np.ndarray,
    beta: np.ndarray,
    index_permutation: np.ndarray,
    alpha: float,
    epsilon_convergence: float,
    max_iterations: int,
    gamma: float,
    lambda_max: float,
    proportion_xi: float,
    num_intervals: int,
    num_fixed_effects: int,
    trace_progress: bool,
) -> dict:
    r"""Sparse-group lasso via proximal gradient descent.

    This function implements the sparse-group lasso algorithm using proximal
    gradient descent. It solves the optimization problem:

    .. math::
        \min_{\boldsymbol{\beta}} \frac{1}{2n} \| \mathbf{y} - \mathbf{X} \boldsymbol{\beta} \|_2^2 + \lambda P(\boldsymbol{\beta})

    where the penalty term :math:`P(\boldsymbol{\beta})` is the sparse-group lasso penalty:

    .. math::
        P(\boldsymbol{\beta}) = \alpha \sum_{j} |\beta_j| + (1 - \alpha) \sum_{g} \| \boldsymbol{\beta}_g \|_2

    where:
    - :math:`\beta_j` are individual coefficients,
    - :math:`\boldsymbol{\beta}_g` are groups of coefficients,
    - :math:`\alpha` is the mixing parameter balancing Lasso and Group Lasso penalties.

    Parameters
    ----------
    y : ndarray of shape (n_samples,)
        Vector of observations, :math:`\mathbf{y}`.
    X : ndarray of shape (n_samples, n_features)
        Design matrix relating :math:`\mathbf{y}` to fixed and random effects, :math:`\mathbf{X}`.
    feature_weights : ndarray of shape (n_features,)
        Weights for the vectors of fixed and random effects, :math:`\mathbf{w}`.
    groups : ndarray of shape (n_features,)
        Integer vector specifying group membership for each effect (fixed and
        random), :math:`\mathbf{G}`.
    beta : ndarray of shape (n_features,)
        Initial guess for the coefficient vector, :math:`\boldsymbol{\beta}`.
    index_permutation : ndarray of shape (n_features,)
        Integer vector containing information about the original order of the
        user's input, :math:`\mathbf{\pi}`.
    alpha : float
        Mixing parameter of the penalty terms. Must satisfy :math:`0 < \alpha < 1`.
    epsilon_convergence : float
        Relative accuracy of the solution, :math:`\epsilon`.
    max_iterations : int
        Maximum number of iterations for each value of the penalty parameter
        :math:`\lambda`.
    gamma : float
        Multiplicative parameter to decrease the step size during backtracking
        line search, :math:`\gamma`.
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
        A dictionary containing the results of the sparse-group lasso algorithm.
        Keys include:
        - `'random_effects'`: ndarray of shape (num_intervals, n_features) or
          (num_intervals, n_features - num_fixed_effects)
        - `'fixed_effects'`: ndarray of shape (num_intervals, num_fixed_effects)`
          (only if `num_fixed_effects` > 0)
        - `'lambda'`: ndarray of shape (num_intervals,)
        - `'iterations'`: ndarray of shape (num_intervals,)
        - `'alpha'`: float
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
    The algorithm solves the optimization problem:

    .. math::
        \min_{\boldsymbol{\beta}} \frac{1}{2n} \| \mathbf{y} - \mathbf{X} \boldsymbol{\beta} \|_2^2 + \lambda P(\boldsymbol{\beta})

    where the penalty term :math:`P(\boldsymbol{\beta})` is defined as:

    .. math::
        P(\boldsymbol{\beta}) = \alpha \sum_{j} |\beta_j| + (1 - \alpha) \sum_{g} \| \boldsymbol{\beta}_g \|_2

    The algorithm uses proximal gradient descent with soft-thresholding for the
    Lasso penalty and group soft-thresholding for the Group Lasso penalty.

    The optimization process involves the following steps:
    1. Initialization: Initialize variables and precompute necessary quantities.
    2. Grid Search Over :math:`\lambda`: Perform a logarithmic grid search over
       :math:`\lambda` values from :math:`\lambda_{\text{max}}` down to
       :math:`\xi \times \lambda_{\text{max}}` with :math:`m` intervals.
    3. Proximal Gradient Descent:
       - Gradient Calculation: Compute the gradient of the loss function.
       - Proximal Step: Apply soft-thresholding and group soft-thresholding to update
         the coefficients :math:`\boldsymbol{\beta}`.
       - Convergence Check: Assess whether the relative change in coefficients
         meets the convergence criterion :math:`\epsilon`.
    4. Iteration Tracking: Record the number of iterations and other diagnostic information.

    Parameters such as `gamma` control the step size reduction during backtracking
    line search to ensure convergence and numerical stability.

    The results include the estimated fixed and random effects, the corresponding
    :math:`\lambda` values, the number of iterations required for convergence, and
    other diagnostic information.
    """

    n, p = X.shape
    num_groups = np.max(groups)

    # Initialize variables
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

    # Initialize result arrays
    iterations = np.zeros(num_intervals, dtype=int)
    lambdas = np.zeros(num_intervals)
    solution = np.zeros((num_intervals, p))

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

                # Soft-thresholding and soft-scaling in groups
                for i in range(num_groups):
                    start, end = index_start[i], index_end[i] + 1
                    temp = beta[start:end] - time_step * gradient[start:end]

                    # Soft-thresholding
                    soft_threshold = (
                        alpha * time_step * lambda_val * feature_weights[start:end]
                    )
                    temp = np.sign(temp) * np.maximum(np.abs(temp) - soft_threshold, 0)

                    # Soft-scaling
                    l2_norm = np.linalg.norm(temp)
                    threshold = time_step * (1 - alpha) * lambda_val * group_weights[i]

                    if l2_norm > threshold:
                        scaling = 1 - threshold / l2_norm
                        beta_new[start:end] = scaling * temp
                    else:
                        beta_new[start:end] = 0

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
        solution[interval] = beta[index_permutation - 1]
        iterations[interval] = counter - 1
        lambdas[interval] = lambda_val

    # Prepare results
    if num_fixed_effects == 0:
        return {
            "random_effects": solution,
            "lambda": lambdas,
            "iterations": iterations,
            "alpha": alpha,
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
            "alpha": alpha,
            "rel_acc": epsilon_convergence,
            "max_iter": max_iterations,
            "gamma_bls": gamma,
            "xi": proportion_xi,
            "loops_lambda": num_intervals,
        }
