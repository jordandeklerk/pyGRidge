r"""Sigma Ridge regression with accelerated leave-one-out cross-validation."""

import numpy as np
import warnings
from typing import Optional, Dict, Any
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from .blockridge import (
    GroupRidgeRegressor,
    MomentTunerSetup,
    sigma_squared_path,
    get_lambdas,
    SingularMatrixError,
)
from .groupedfeatures import GroupedFeatures


class SigmaRidgeRegressor(BaseEstimator, RegressorMixin):
    r"""Sigma Ridge regression with accelerated leave-one-out cross-validation.

    This estimator implements Sigma-Ridge regression, which uses an empirical Bayes
    approach to map a single hyperparameter :math:`\sigma` to group-specific regularization
    parameters :math:`\lambda(\sigma)`. The mapping is optimized using accelerated leave-one-out
    cross-validation.

    Parameters
    ----------
    feature_groups : list[list[int]], optional
        Each sublist contains indices of features belonging to a group.
        If None, each feature is treated as its own group.
    sigma : float, default=1.0
        Initial :math:`\sigma` value.
    decomposition : {'default', 'cholesky', 'woodbury'}, default='default'
        Method to solve the linear system:
        - 'default': Automatically choose based on problem dimensions
        - 'cholesky': Use Cholesky decomposition (efficient for n_samples > n_features)
        - 'woodbury': Use Woodbury identity (efficient for n_features > n_samples)
    center : bool, default=True
        Whether to center the features and target.
    scale : bool, default=True
        Whether to scale the features.
    init_model : object, optional
        Initial model for mapping :math:`\sigma` to :math:`\lambda`. Defaults to standard LOOCV Ridge Regressor.
    sigma_range : tuple of float, optional
        Range of :math:`\sigma` values for optimization. If None, determined automatically.
    optimization_method : {'bounded', 'grid_search'}, optional
        Optimization strategy for :math:`\sigma`.
    tol : float, default=1e-4
        Tolerance for optimization convergence.
    max_iter : int, default=1000
        Maximum iterations for optimization algorithms.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Estimated coefficients.
    intercept_ : float
        The intercept term.
    sigma_opt_ : float
        The optimal :math:`\sigma` value found during fitting.
    lambda_ : ndarray of shape (n_groups,)
        The optimal :math:`\lambda` values computed from :math:`\sigma_\text{opt}`.
    n_iter_ : int
        Number of iterations performed during optimization.
    feature_groups_ : GroupedFeatures
        The grouped features object used internally.
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during fit.

    Notes
    -----
    The Sigma-Ridge regression model solves the optimization problem:

    .. math::
        \min_{\beta} \|y - X\beta\|_2^2 + \sum_{g=1}^G \lambda_g(\sigma) \|\beta_g\|_2^2

    where:
    - :math:`\beta_g` are the coefficients for group g
    - :math:`\lambda_g(\sigma)` is the regularization parameter for group g
    - :math:`\sigma` is a single hyperparameter that controls all :math:`\lambda_g`

    Examples
    --------
    >>> import numpy as np
    >>> from pygridge.src.sigma_ridge import SigmaRidgeRegressor
    >>> X = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    >>> y = np.array([2, 4, 6])
    >>> groups = [[0, 1], [2, 3]]  # Two groups of two features each
    >>> reg = SigmaRidgeRegressor(feature_groups=groups)
    >>> reg.fit(X, y)
    SigmaRidgeRegressor(feature_groups=[[0, 1], [2, 3]])
    >>> reg.predict([[1, 2, 3, 4]])
    array([...])
    """

    def __init__(
        self,
        feature_groups: Optional[list[list[int]]] = None,
        sigma: float = 1.0,
        decomposition: str = "default",
        center: bool = True,
        scale: bool = True,
        init_model: Optional[object] = None,
        sigma_range: Optional[tuple] = None,
        optimization_method: Optional[str] = None,
        tol: float = 1e-4,
        max_iter: int = 1000,
    ):
        self.feature_groups = feature_groups
        self.sigma = sigma
        self.decomposition = decomposition
        self.center = center
        self.scale = scale
        self.init_model = init_model
        self.sigma_range = sigma_range
        self.optimization_method = optimization_method
        self.tol = tol
        self.max_iter = max_iter

    def _validate_params(self):
        """Validate parameters.

        Raises
        ------
        ValueError
            If parameters are invalid.
        """
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        if self.tol <= 0:
            raise ValueError("tol must be positive")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if self.decomposition not in ["default", "cholesky", "woodbury"]:
            raise ValueError(
                "decomposition must be one of ['default', 'cholesky', 'woodbury']"
            )
        if self.optimization_method is not None and self.optimization_method not in [
            "bounded",
            "grid_search",
        ]:
            raise ValueError(
                "optimization_method must be one of ['bounded', 'grid_search'] or None"
            )
        if self.sigma_range is not None:
            if (
                not isinstance(self.sigma_range, tuple)
                or len(self.sigma_range) != 2
                or self.sigma_range[0] >= self.sigma_range[1]
                or self.sigma_range[0] <= 0
            ):
                raise ValueError(
                    "sigma_range must be a tuple of two positive floats (min, max) where min < max"
                )

    def _init_feature_groups(self, n_features: int) -> GroupedFeatures:
        """Initialize feature groups.

        Parameters
        ----------
        n_features : int
            Number of features.

        Returns
        -------
        GroupedFeatures
            The initialized grouped features object.
        """
        if self.feature_groups is None:
            return GroupedFeatures([1] * n_features)
        else:
            all_features = set()
            group_sizes = []
            for group in self.feature_groups:
                if not isinstance(group, list):
                    raise ValueError("Each group must be a list of feature indices")
                if not group:
                    raise ValueError("Empty groups are not allowed")
                if not all(
                    isinstance(idx, int) and 0 <= idx < n_features for idx in group
                ):
                    raise ValueError(
                        f"Invalid feature indices in group {group}. Must be integers in [0, {n_features-1}]"
                    )
                if set(group) & all_features:
                    raise ValueError("Features cannot belong to multiple groups")
                all_features.update(group)
                group_sizes.append(len(group))

            if all_features != set(range(n_features)):
                raise ValueError("All features must be assigned to a group")

            return GroupedFeatures(group_sizes)

    def _init_ridge_estimator(self, X: np.ndarray, y: np.ndarray) -> None:
        """Initialize ridge estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Raises
        ------
        SingularMatrixError
            If fitting fails even with the largest regularization value.

        Notes
        -----
        The method tries regularization values from 0.001 to 100.0 in increasing order.
        For each value, it attempts to fit the GroupRidgeRegressor. If fitting succeeds,
        it stores the estimator and breaks the loop. If all values fail, it raises
        a SingularMatrixError.
        """
        regularization_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

        for alpha in regularization_values:
            try:
                self.ridge_estimator_ = GroupRidgeRegressor(
                    groups=self.feature_groups_,
                    alpha=np.ones(self.feature_groups_.num_groups) * alpha,
                )
                self.ridge_estimator_.fit(X, y)
                break
            except SingularMatrixError:
                if alpha == regularization_values[-1]:
                    raise

    def _optimize_sigma(self, X: np.ndarray, y: np.ndarray) -> float:
        """Optimize :math:`\sigma` by minimizing CV error.

        This method finds the optimal value of sigma by minimizing the leave-one-out
        cross-validation error. It uses either grid search or a geometric sequence
        approach based on the optimization_method parameter.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        float
            The optimal value of sigma that minimizes the cross-validation error.

        Notes
        -----
        The method follows these steps:
        1. Initializes a ridge estimator if not already done
        2. Sets up moment tuning for the estimator
        3. Determines the range of sigma values to evaluate
        4. Uses either grid search (20 points) or geometric sequence (10 points)
        5. Computes the path of sigma squared values and their corresponding errors
        6. Returns the square root of the sigma squared value with minimum error

        The sigma range is either specified by sigma_range parameter or automatically
        determined using \sigma_max from moment tuning.
        """
        if not hasattr(self, "ridge_estimator_"):
            self._init_ridge_estimator(X, y)

        moment_tuner = MomentTunerSetup(self.ridge_estimator_)

        if self.sigma_range is None:
            sigma_max = np.sqrt(moment_tuner.get_sigma_squared_max())
            # Define grid as [10^(-3) * sigma_max, sigma_max]
            sigma_range = (1e-3 * sigma_max, sigma_max)
        else:
            sigma_range = self.sigma_range

        if self.optimization_method == "grid_search":
            sigma_squared_values = np.logspace(
                2 * np.log10(sigma_range[0]),
                2 * np.log10(sigma_range[1]),
                num=20,
            )
        else:
            sigma_squared_values = np.geomspace(
                sigma_range[0] ** 2,
                sigma_range[1] ** 2,
                num=10,
            )

        path_results = sigma_squared_path(
            self.ridge_estimator_,
            moment_tuner,
            sigma_squared_values,
            max_iter=self.max_iter,
            tol=self.tol,
        )

        best_idx = np.argmin(path_results["errors"])
        sigma_opt = np.sqrt(sigma_squared_values[best_idx])
        self.moment_tuner_ = moment_tuner

        return sigma_opt

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SigmaRidgeRegressor":
        """Fit the :math:`\sigma`-Ridge regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, accept_sparse=False)
        self._validate_params()

        X = X.astype(np.float64)

        _, n_features = X.shape
        self.n_features_in_ = n_features
        self.feature_names_in_ = np.arange(n_features)
        self.feature_groups_ = self._init_feature_groups(n_features)

        if self.center:
            self.X_mean_ = np.mean(X, axis=0)
            X = X - self.X_mean_
        else:
            self.X_mean_ = np.zeros(n_features)

        if self.scale:
            self.X_scale_ = np.std(X, axis=0, ddof=1)
            self.X_scale_[self.X_scale_ == 0] = 1
            X = X / self.X_scale_
        else:
            self.X_scale_ = np.ones(n_features)

        self.X_ = X
        self.y_ = y

        try:
            self.sigma_opt_ = self._optimize_sigma(X, y)
            self.lambda_ = get_lambdas(self.moment_tuner_, self.sigma_opt_**2)
        except Exception as e:
            warnings.warn(
                f"Failed to compute optimal parameters: {str(e)}. "
                "Falling back to default regularization."
            )
            self.sigma_opt_ = self.sigma
            self.lambda_ = np.ones(self.feature_groups_.num_groups) * 0.001

        try:
            self.ridge_estimator_.set_params(alpha=self.lambda_)
            self.ridge_estimator_.fit(X, y)
        except SingularMatrixError:
            warnings.warn("Singular matrix detected. Using higher regularization.")
            self.lambda_ = np.maximum(self.lambda_, 10.0)
            self._init_ridge_estimator(X, y)

        self.coef_ = self.ridge_estimator_.coef_
        self.intercept_ = 0.0

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the :math:`\sigma`-Ridge regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Returns predicted values.
        """
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but SigmaRidgeRegressor is expecting "
                f"{self.n_features_in_} features as input"
            )

        X = X.astype(np.float64)
        if self.center:
            X = X - self.X_mean_
        if self.scale:
            X = X / self.X_scale_

        return self.ridge_estimator_.predict(X)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        params = super().get_params(deep=deep)
        if deep:
            if hasattr(self, "ridge_estimator_"):
                ridge_params = self.ridge_estimator_.get_params(deep=True)
                params.update(
                    {f"ridge__{key}": val for key, val in ridge_params.items()}
                )
        return params

    def set_params(self, **params) -> "SigmaRidgeRegressor":
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : object
            Estimator instance.
        """
        ridge_params = {}
        own_params = {}
        for key, value in params.items():
            if key.startswith("ridge__"):
                ridge_params[key.split("__", 1)[1]] = value
            else:
                own_params[key] = value

        if ridge_params and hasattr(self, "ridge_estimator_"):
            self.ridge_estimator_.set_params(**ridge_params)
        if own_params:
            super().set_params(**own_params)
        return self
