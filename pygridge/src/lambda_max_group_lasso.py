"""Compute the maximum lambda value for group lasso regularization."""

import numpy as np
from sklearn.utils.validation import check_array, check_consistent_length


def lambda_max_group_lasso(y, groups, feature_weights, beta, X):
    r"""Compute the maximal lambda for group lasso regularization.

    The maximal lambda value is the smallest value for which all groups
    of coefficients are zero in the group lasso optimization problem.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Target vector.
    groups : array-like of shape (n_features,)
        Integer vector specifying group membership for each feature.
    feature_weights : array-like of shape (n_features,)
        Weights for the feature vector :math:`\mathbf{b}^T, \mathbf{u}^T`.
    beta : array-like of shape (n_features,)
        Feature vector. Random effects are initialized to zero,
        fixed effects are initialized via least squares.
    X : array-like of shape (n_samples, n_features)
        Design matrix :math:`[X \ Z]` relating :math:`y` to fixed and random effects.

    Returns
    -------
    float
        The maximal lambda value.

    Notes
    -----
    Computes the maximal lambda value for group lasso regularization through the following steps:
    1. Initialize group-specific variables.
    2. Handle unpenalized features if present.
    3. Scale :math:`X^T \mathbf{r}` in groups.
    4. Determine :math:`\lambda_{\text{max}}` and apply numeric correction.

    The formula for :math:`\lambda_{\text{max}}` is:

    .. math::
        \lambda_{\text{max}} = \frac{\max_{i} \left( \| X_i^T \mathbf{r} \|_2 \right)}{n w_i \sqrt{|G_i|}}

    where:
    - :math:`X_i` is the design matrix for group :math:`i`,
    - :math:`\mathbf{r}` is the residual vector,
    - :math:`n` is the number of samples,
    - :math:`w_i` is the weight for group :math:`i`,
    - and :math:`|G_i|` is the size of group :math:`i`.
    """
    # Input validation
    X = check_array(X, accept_sparse=False)
    y = check_array(y, ensure_2d=False)
    groups = check_array(groups, ensure_2d=False, dtype=np.int64)
    feature_weights = check_array(feature_weights, ensure_2d=False)
    beta = check_array(beta, ensure_2d=False)

    # Check consistent lengths
    check_consistent_length(X, groups)
    check_consistent_length(X, feature_weights)
    check_consistent_length(X, beta)
    n_samples = X.shape[0]
    check_consistent_length(y, np.zeros(n_samples))

    n, p = X.shape
    num_groups = np.max(groups)
    lambda_max = 0.0
    num_zeros_weights = np.sum(feature_weights == 0.0)

    index_start = np.zeros(num_groups, dtype=int)
    index_end = np.zeros(num_groups, dtype=int)
    group_weights = np.zeros(num_groups)
    l2_norm_groups = np.zeros(num_groups)

    # Create vectors of group sizes, start indices, end indices, and group weights
    for i in range(num_groups):
        group_mask = groups == (i + 1)
        group_indices = np.where(group_mask)[0]
        if len(group_indices) == 0:
            raise ValueError(f"Group {i + 1} is empty")
        index_start[i] = group_indices[0]
        index_end[i] = group_indices[-1]
        group_weights[i] = np.sum(feature_weights[group_mask])

    group_weights = np.sqrt(group_weights)

    # Treatment if unpenalized features are involved
    if num_zeros_weights > 0:
        active_mask = feature_weights == 0.0
        X_active = X[:, active_mask]
        try:
            beta_active = np.linalg.solve(X_active.T @ X_active, X_active.T @ y)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if matrix is singular
            beta_active = np.linalg.pinv(X_active.T @ X_active) @ X_active.T @ y
        beta[active_mask] = beta_active
        residual_active = y - X_active @ beta_active
        X_transp_residual_active = X.T @ residual_active
    else:
        X_transp_residual_active = X.T @ y

    # Scale X.T * residuals in groups with n * weight * sqrt(group_size), if weight > 0
    for i in range(num_groups):
        if group_weights[i] == 0.0:
            l2_norm_groups[i] = 0.0
        else:
            temp = n * group_weights[i]
            group_slice = slice(index_start[i], index_end[i] + 1)
            l2_norm_groups[i] = (
                np.linalg.norm(X_transp_residual_active[group_slice]) / temp
            )

    # Determine lambda_max and perform numeric correction
    lambda_max = np.max(np.abs(l2_norm_groups))
    return lambda_max * 1.00001
