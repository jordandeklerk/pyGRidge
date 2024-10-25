"""Compute the maximum lambda value for LASSO regularization."""

import numpy as np


def lambda_max_lasso(y, feature_weights, beta, X):
    r"""Compute the maximal lambda for LASSO regression.

    The maximal lambda value is the smallest value for which all coefficients become zero
    in the LASSO optimization problem.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Target vector, :math:`\mathbf{y}`.
    feature_weights : array-like of shape (n_features,)
        Weights for the feature vector :math:`\mathbf{w}`. The entries may be permuted
        corresponding to their group assignments.
    beta : array-like of shape (n_features,)
        Feature vector, :math:`\boldsymbol{\beta}`. Random effects are initialized to zero,
        fixed effects are initialized via least squares.
    X : array-like of shape (n_samples, n_features)
        Design matrix :math:`\mathbf{X}` relating :math:`\mathbf{y}` to fixed and random effects.

    Returns
    -------
    float
        The maximal lambda value for LASSO regression, :math:`\lambda_{\text{max}}`.

    Notes
    -----
    The maximal lambda is computed as:

    .. math::
        \lambda_{\text{max}} = \frac{\max_i \left| \mathbf{X}_i^T \mathbf{r} \right|}{n w_i}

    where:
    - :math:`\mathbf{X}_i` is the :math:`i`-th column of :math:`\mathbf{X}`,
    - :math:`\mathbf{r}` is the residual vector,
    - :math:`n` is the number of samples,
    - and :math:`w_i` is the weight for feature :math:`i`.

    The computation involves the following steps:
    1. Initialize variables specific to groups with unpenalized features.
    2. Handle unpenalized features if present.
    3. Scale :math:`\mathbf{X}^T \mathbf{r}` with weights and sample size.
    4. Determine :math:`\lambda_{\text{max}}` and apply a numeric correction factor.
    """

    n, p = X.shape

    # Check dimensions
    if y.shape[0] != n or feature_weights.shape[0] != p or beta.shape[0] != p:
        raise ValueError("Dimension mismatch in input vectors")

    num_zeros_weights = np.sum(feature_weights == 0)

    x_transp_residual_active = np.zeros(p, dtype=np.float64)

    if num_zeros_weights > 0:
        # Treatment if unpenalized features are involved
        X_active = X[:, feature_weights == 0]

        # Solve for beta_active in y = X_active * beta_active
        beta_active = np.linalg.solve(X_active.T @ X_active, X_active.T @ y)

        # Update beta with beta_active
        beta[feature_weights == 0] = beta_active

        # Calculate residual_active = y - X_active * beta_active
        residual_active = y - X_active @ beta_active

        # Calculate X^T * residual_active
        x_transp_residual_active = X.T @ residual_active
    else:
        # Treatment if only penalized features are involved
        x_transp_residual_active = X.T @ y

    # Scale X^T * residual_active with weight * n, if weight > 0
    mask = feature_weights > 0
    x_transp_residual_active[mask] /= feature_weights[mask] * n
    x_transp_residual_active[~mask] = 0

    # Determine lambda_max and perform numeric correction
    lambda_max = np.max(np.abs(x_transp_residual_active))
    return lambda_max * 1.00001
