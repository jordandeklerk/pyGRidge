"""Compute the maximum lambda value for sparse group lasso regularization."""

import numpy as np
from .bisection import BisectionSolver


def lambda_max_sparse_group_lasso(
    alpha: float,
    vector_y: np.ndarray,
    vector_groups: np.ndarray,
    vector_weights_features: np.ndarray,
    vector_beta: np.ndarray,
    matrix_x: np.ndarray,
) -> float:
    r"""Calculate the maximum value for the penalty parameter :math:`\lambda` in sparse group lasso.

    This function computes the maximum value for the penalty parameter :math:`\lambda`
    in the sparse group lasso optimization problem, ensuring that all coefficients
    are zero at this threshold.

    Parameters
    ----------
    alpha : float
        Mixing parameter of the penalty terms. Must satisfy :math:`0 < \alpha < 1`.
        The penalty term is defined as:
        :math:`\alpha \times \text{"lasso penalty"} + (1 - \alpha) \times \text{"group lasso penalty"}`.
    vector_y : ndarray of shape (n_samples,)
        Vector of observations, :math:`\mathbf{y}`.
    vector_groups : ndarray of shape (n_features,)
        Integer vector specifying group membership for each effect (fixed and random), :math:`\mathbf{G}`.
    vector_weights_features : ndarray of shape (n_features,)
        Vector of weights for the fixed and random effects, :math:`\mathbf{w}`.
    vector_beta : ndarray of shape (n_features,)
        Vector of features, :math:`\boldsymbol{\beta}`. Random effects are initialized to zero,
        and fixed effects are initialized via least squares.
    matrix_x : ndarray of shape (n_samples, n_features)
        Design matrix relating :math:`\mathbf{y}` to fixed and random effects, :math:`\mathbf{X}`.

    Returns
    -------
    float
        The maximum value for the penalty parameter :math:`\lambda`.

    Notes
    -----
    The sparse group lasso penalty is a combination of the lasso and group lasso penalties:

    .. math::
        P(\boldsymbol{\beta}) = \alpha \sum_{j} |\beta_j| + (1 - \alpha) \sum_{g} \|\boldsymbol{\beta}_g\|_2

    where:
    - :math:`\beta_j` are individual coefficients,
    - :math:`\boldsymbol{\beta}_g` are groups of coefficients,
    - :math:`\alpha` is the mixing parameter balancing lasso and group lasso penalties.

    The computation involves the following steps:
    1. Initialize variables specific to groups with unpenalized features.
    2. Handle unpenalized features if present.
    3. Scale :math:`\mathbf{X}^T \mathbf{r}` with weights and sample size.
    4. Determine :math:`\lambda_{\text{max}}` using a bisection method and apply a numeric correction factor.

    The formula for :math:`\lambda_{\text{max}}` is:

    .. math::
        \lambda_{\text{max}} = \max_{g} \lambda_g

    where each :math:`\lambda_g` for group :math:`g` is determined by:

    .. math::
        \lambda_g = \text{seagull\_bisection}\left(
            |\mathbf{G}_g|, \alpha, \text{left\_border}, \text{right\_border},
            w_g, \mathbf{w}_g, \frac{\mathbf{X}_g^T \mathbf{r}}{n}
        \right)

    Here:
    - :math:`|\mathbf{G}_g|` is the size of group :math:`g`,
    - :math:`w_g` is the weight for group :math:`g`,
    - :math:`\mathbf{w}_g` are the weights for features in group :math:`g`,
    - :math:`\mathbf{X}_g` is the submatrix of :math:`\mathbf{X}` corresponding to group :math:`g`,
    - :math:`\mathbf{r}` is the residual vector,
    - and :math:`n` is the number of samples.

    The `BisectionSolver` class is used to solve for :math:`\lambda_g` within specified borders.
    """

    n, p = matrix_x.shape
    number_groups = np.max(vector_groups)
    number_zeros_weights = np.sum(vector_weights_features == 0)

    vector_index_start = np.zeros(number_groups, dtype=int)
    vector_index_end = np.zeros(number_groups, dtype=int)
    vector_group_sizes = np.zeros(number_groups, dtype=int)
    vector_weights_groups = np.zeros(number_groups)

    for i in range(number_groups):
        group_mask = vector_groups == (i + 1)
        vector_group_sizes[i] = np.sum(group_mask)
        vector_index_start[i] = np.argmax(group_mask)
        vector_index_end[i] = vector_index_start[i] + vector_group_sizes[i] - 1
        vector_weights_groups[i] = np.sum(vector_weights_features[group_mask])

    if number_zeros_weights > 0:
        active_mask = vector_weights_features == 0
        matrix_x_active = matrix_x[:, active_mask]

        vector_beta_active = np.linalg.solve(
            matrix_x_active.T @ matrix_x_active, matrix_x_active.T @ vector_y
        )
        vector_beta[active_mask] = vector_beta_active
        vector_residual_active = vector_y - matrix_x_active @ vector_beta_active
        vector_x_transp_residual_active = matrix_x.T @ vector_residual_active
    else:
        vector_x_transp_residual_active = matrix_x.T @ vector_y

    vector_max_groups = np.zeros(number_groups)
    solver = BisectionSolver()

    for i in range(number_groups):
        if vector_weights_groups[i] == 0:
            vector_max_groups[i] = 0
        else:
            start, end = vector_index_start[i], vector_index_end[i] + 1
            vector_temp = vector_x_transp_residual_active[start:end] / (
                n * vector_weights_features[start:end]
            )
            vector_temp_absolute = np.abs(vector_temp)

            vector_temp = vector_x_transp_residual_active[start:end] / n

            left_border = 0
            right_border = np.max(vector_temp_absolute) / alpha

            vector_max_groups[i] = solver.solve(
                vector_group_sizes[i],
                alpha,
                left_border,
                right_border,
                vector_weights_groups[i],
                vector_weights_features[start:end],
                vector_temp,
            )

    lambda_max = np.max(vector_max_groups)
    return lambda_max * 1.00001
