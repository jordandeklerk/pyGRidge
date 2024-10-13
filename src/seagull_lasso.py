"""
This module implements the seagull lasso algorithm for solving Lasso
regression problems.

It provides a single function, seagull_lasso, which performs the
optimization using coordinate descent with soft-thresholding. The
algorithm supports various penalty terms and can handle fixed and
random effects in mixed models.
"""

import numpy as np


def seagull_lasso(
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
    """Perform Lasso, group lasso, and sparse-group lasso regression.

    This function implements the seagull lasso algorithm for solving
    Lasso, group lasso, and sparse-group lasso problems.

    Parameters
    ----------
    y : ndarray of shape (n_samples,)
        Vector of observations.
    X : ndarray of shape (n_samples, n_features)
        Design matrix relating y to fixed and random effects [X Z].
    feature_weights : ndarray of shape (n_features,)
        Weights for the vectors of fixed and random effects [b^T, u^T]^T.
    beta : ndarray of shape (n_features,)
        Initial guess for the coefficient vector.
    epsilon_convergence : float
        Relative accuracy of the solution.
    max_iterations : int
        Maximum number of iterations for each value of the penalty
        parameter λ.
    gamma : float
        Multiplicative parameter to decrease the step size during
        backtracking line search.
    lambda_max : float
        Maximum value for the penalty parameter.
    proportion_xi : float
        Multiplicative parameter to determine the minimum value of λ for
        the grid search.
    num_intervals : int
        Number of lambdas for the grid search.
    num_fixed_effects : int
        Number of fixed effects present in the mixed model.
    trace_progress : bool
        If True, print progress after each finished loop of the λ grid.

    Returns
    -------
    dict
        A dictionary containing the results of the lasso algorithm.
        Keys include:
        - 'random_effects': ndarray of shape (num_intervals, n_features)
          or (num_intervals, n_features - num_fixed_effects)
        - 'fixed_effects': ndarray of shape (num_intervals,
          num_fixed_effects) (only if num_fixed_effects > 0)
        - 'lambda': ndarray of shape (num_intervals,)
        - 'iterations': ndarray of shape (num_intervals,)
        - 'rel_acc': float
        - 'max_iter': int
        - 'gamma_bls': float
        - 'xi': float
        - 'loops_lambda': int

    Notes
    -----
    The algorithm solves the optimization problem:

    min_{beta} 1/(2n) ||y - X beta||_2^2 + lambda * P(beta)

    where P(beta) is the penalty term, which can be:
    - Lasso: sum_j |beta_j|
    - Group Lasso: sum_g ||beta_g||_2
    - Sparse Group Lasso: alpha * sum_j |beta_j| + (1-alpha) *
      sum_g ||beta_g||_2

    The algorithm uses coordinate descent with soft-thresholding.
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
                gradient = (X.T @ (X @ beta) - X_transp_y) / n

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