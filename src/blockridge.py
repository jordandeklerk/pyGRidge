"""Group Ridge regression estimators."""

from abc import ABC, abstractmethod
from typing import Union, TypeVar, Dict, Any
import numpy as np
from scipy.linalg import cho_solve
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array
from groupedfeatures import GroupedFeatures
from nnls import nonneg_lsq, NNLSError
import warnings

T = TypeVar("T")


class RidgeRegressionError(Exception):
    """Base exception class for Ridge regression errors."""
    pass


class InvalidDimensionsError(RidgeRegressionError):
    """Exception raised when matrix dimensions are incompatible."""
    pass


class SingularMatrixError(RidgeRegressionError):
    """Exception raised when a matrix is singular or nearly singular."""
    pass


class NumericalInstabilityError(RidgeRegressionError):
    """Exception raised when numerical instability is detected."""
    pass


class BaseRidgePredictor(ABC):
    r"""Abstract base class for Ridge regression predictors.

    This class defines the interface that all concrete Ridge predictors must
    implement, ensuring consistency in how regularization parameters are updated
    and how various mathematical operations related to Ridge regression are
    performed.

    Ridge regression seeks to minimize the following objective function:

    .. math::
        \min_{\beta} \|y - X\beta\|_2^2 + \alpha\|\beta\|_2^2

    where:
    - :math:`y` is the target vector with dimensions :math:`(n_{\text{samples}}, )`.
    - :math:`X` is the design matrix with dimensions :math:`(n_{\text{samples}}, n_{\text{features}})`.
    - :math:`\beta` is the coefficient vector with dimensions :math:`(n_{\text{features}}, )`.
    - :math:`\alpha` is the regularization parameter.

    The solution to this optimization problem is given by:

    .. math::
        \beta = (X^T X + \alpha I_p)^{-1} X^T y
    """

    @abstractmethod
    def set_params(self, groups: GroupedFeatures, alpha: np.ndarray):
        r"""Update the regularization parameters (:math:`\alpha`) based on feature groups.

        Parameters
        ----------
        groups : GroupedFeatures
            The grouped features object defining feature groupings.
        alpha : np.ndarray
            The new :math:`\alpha` values for each group.

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def _trace_gram_matrix(self) -> float:
        """Compute the trace of the Gram matrix :math:`X^T X`.

        Returns
        -------
        float
            The trace of the Gram matrix.
        """
        pass

    @abstractmethod
    def _solve_gram_system(self) -> np.ndarray:
        r"""Compute :math:`(X^T X + \alpha I)^{-1} X^T X`.

        Returns
        -------
        np.ndarray
            The result of the computation.
        """
        pass

    @abstractmethod
    def _solve_system(self, B: np.ndarray) -> np.ndarray:
        r"""Solve the linear system :math:`(X^T X + \alpha I) x = B`.

        Parameters
        ----------
        B : np.ndarray
            The right-hand side matrix or vector.

        Returns
        -------
        np.ndarray
            The solution vector :math:`x`.
        """
        pass


class CholeskyRidgePredictor(BaseRidgePredictor):
    r"""Ridge predictor using Cholesky decomposition for efficient matrix inversion.

    Suitable for scenarios where the number of features is less than the number
    of samples.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Training data.

    Attributes
    ----------
    n_samples_ : int
        Number of samples.
    n_features_ : int
        Number of features.
    gram_matrix_ : np.ndarray of shape (n_features, n_features)
        The Gram matrix :math:`X^T X` scaled by n_samples.
    gram_reg_ : np.ndarray of shape (n_features, n_features)
        The regularized Gram matrix :math:`X^T X + \alpha I`.
    gram_reg_chol_ : np.ndarray of shape (n_features, n_features)
        The Cholesky decomposition of gram_reg_.
    lower_ : bool
        Indicates if the Cholesky factor is lower triangular.
    """

    def __init__(self, X: np.ndarray):
        """Initialize the Cholesky Ridge predictor.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.

        Raises
        ------
        InvalidDimensionsError
            If X is not a 2D array.
        ValueError
            If X contains NaN or infinity values.
        """
        if X.ndim != 2:
            raise InvalidDimensionsError("X must be a 2D array.")
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("X contains NaN or infinity values.")

        self.n_samples_, self.n_features_ = X.shape
        self.gram_matrix_ = np.dot(X.T, X) / self.n_samples_
        self.gram_reg_ = self.gram_matrix_ + np.eye(self.n_features_)
        self._update_cholesky()
        self.lower_ = True

    def _update_cholesky(self):
        r"""Update the Cholesky decomposition of the regularized Gram matrix.

        Raises
        ------
        SingularMatrixError
            If the matrix is not positive definite.
        """
        try:
            self.gram_reg_chol_ = np.linalg.cholesky(self.gram_reg_)
        except np.linalg.LinAlgError:
            raise SingularMatrixError(
                "Failed to compute Cholesky decomposition. Matrix may not be positive "
                "definite."
            )

    def set_params(self, groups: GroupedFeatures, alpha: np.ndarray):
        r"""Update the regularization parameters and recompute the Cholesky decomposition.

        Parameters
        ----------
        groups : GroupedFeatures
            The grouped features object.
        alpha : np.ndarray
            The new :math:`\alpha` values for each group.

        Returns
        -------
        None
        """
        diag = groups.group_expand(alpha)
        self.gram_reg_ = self.gram_matrix_ + np.diag(diag)
        self._update_cholesky()

    def _trace_gram_matrix(self) -> float:
        """Compute the trace of the Gram matrix.

        Returns
        -------
        float
            The trace of the Gram matrix.
        """
        return np.trace(self.gram_matrix_)

    def _solve_gram_system(self) -> np.ndarray:
        r"""Compute :math:`(X^T X + \alpha I)^{-1} X^T X` using Cholesky decomposition.

        Returns
        -------
        np.ndarray
            The result of the computation.
        """
        return cho_solve((self.gram_reg_chol_, self.lower_), self.gram_matrix_)

    def _solve_system(self, B: np.ndarray) -> np.ndarray:
        r"""Solve the system :math:`(X^T X + \alpha I) x = B` using Cholesky decomposition.

        Parameters
        ----------
        B : np.ndarray
            The right-hand side of the equation.

        Returns
        -------
        np.ndarray
            The solution vector :math:`x`.
        """
        return cho_solve((self.gram_reg_chol_, self.lower_), B)


class WoodburyRidgePredictor(BaseRidgePredictor):
    r"""Ridge predictor using the Woodbury matrix identity for efficient matrix inversion.

    This class is suitable for scenarios where the number of features is greater
    than the number of samples.

    The Woodbury matrix identity states:

    .. math::
        (A + UCV)^{-1} = A^{-1} - A^{-1}U(C^{-1} + V A^{-1} U)^{-1} V A^{-1}

    In the context of Ridge regression:
        :math:`A = \alpha I` (diagonal matrix of regularization parameters)
        :math:`U = X^T`
        :math:`C = I` (identity matrix)
        :math:`V = X`

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Training data.

    Attributes
    ----------
    n_samples_ : int
        Number of samples.
    n_features_ : int
        Number of features.
    X_ : np.ndarray of shape (n_samples, n_features)
        The training data.
    gram_matrix_ : np.ndarray of shape (n_features, n_features)
        The Gram matrix :math:`X^T X`.
    alpha_inv_ : np.ndarray of shape (n_features, n_features)
        The inverse of :math:`\alpha I`.
    U_ : np.ndarray of shape (n_features, n_samples)
        Matrix :math:`U` in the Woodbury identity.
    V_ : np.ndarray of shape (n_samples, n_features)
        Matrix :math:`V` in the Woodbury identity.
    """

    def __init__(self, X: np.ndarray):
        """Initialize the Woodbury Ridge predictor.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.

        Raises
        ------
        InvalidDimensionsError
            If X is not a 2D array.
        ValueError
            If X contains NaN or infinity values.
        """
        if X.ndim != 2:
            raise InvalidDimensionsError("X must be a 2D array.")
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("X contains NaN or infinity values.")

        self.n_samples_, self.n_features_ = X.shape
        self.X_ = X
        self.gram_matrix_ = X.T @ X
        self.alpha_inv_ = np.eye(self.n_features_)
        self.U_ = X.T
        self.V_ = X

    def set_params(self, groups: GroupedFeatures, alpha: np.ndarray):
        r"""Update the regularization parameters and recompute the inverse matrix.

        Parameters
        ----------
        groups : GroupedFeatures
            The grouped features object.
        alpha : np.ndarray
            The new :math:`\alpha` values for each group.

        Raises
        ------
        ValueError
            If alpha contains non-positive values.

        Returns
        -------
        None
        """
        if np.any(alpha <= 0):
            raise ValueError("Alpha values must be positive.")

        diag = np.array(groups.group_expand(alpha))
        self.alpha_inv_ = np.diag(1 / diag)
        self._woodbury_update()

    def _woodbury_update(self):
        """Apply the Woodbury matrix identity to update the inverse matrix.

        Raises
        ------
        NumericalInstabilityError
            If numerical instability is detected during the update.
        """
        try:
            eye = np.eye(self.n_samples_)
            AU = self.alpha_inv_ @ self.U_
            inv_term = np.linalg.inv(eye + self.V_ @ AU)
            self.alpha_inv_ -= AU @ inv_term @ self.V_ @ self.alpha_inv_
        except np.linalg.LinAlgError:
            raise NumericalInstabilityError(
                "Numerical instability detected in Woodbury update."
            )

    def _trace_gram_matrix(self) -> float:
        """Compute the trace of the Gram matrix.

        Returns
        -------
        float
            The trace of the Gram matrix.
        """
        return np.trace(self.gram_matrix_)

    def _solve_gram_system(self) -> np.ndarray:
        r"""Compute :math:`(X^T X + \alpha I)^{-1} X^T X`.

        Returns
        -------
        np.ndarray
            The result of the computation.
        """
        return self.alpha_inv_ @ self.gram_matrix_

    def _solve_system(self, B: np.ndarray) -> np.ndarray:
        r"""Solve the system :math:`(X^T X + \alpha I) x = B`.

        Parameters
        ----------
        B : np.ndarray
            The right-hand side matrix or vector.

        Returns
        -------
        np.ndarray
            The solution vector :math:`x`.
        """
        return self.alpha_inv_ @ B


class ShermanMorrisonRidgePredictor(BaseRidgePredictor):
    r"""Ridge predictor using the Sherman-Morrison formula for efficient matrix updates.

    This class is suitable for scenarios where the number of features is much
    greater than the number of samples.

    The Sherman-Morrison formula states:

    .. math::
        (A + uv^T)^{-1} = A^{-1} - \frac{A^{-1}u v^T A^{-1}}{1 + v^T A^{-1} u}

    where :math:`A` is the current inverse, and :math:`uv^T` represents a rank-one update.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Training data.

    Attributes
    ----------
    n_samples_ : int
        Number of samples.
    n_features_ : int
        Number of features.
    X_ : np.ndarray of shape (n_samples, n_features)
        The training data.
    gram_matrix_ : np.ndarray of shape (n_features, n_features)
        The Gram matrix :math:`X^T X`.
    A_ : np.ndarray of shape (n_features, n_features)
        The matrix :math:`I + X^T X`.
    A_inv_ : np.ndarray of shape (n_features, n_features)
        The inverse of matrix :math:`A`.
    U_ : np.ndarray of shape (n_features, n_samples)
        Matrix :math:`U` used for efficiency in updates.
    V_ : np.ndarray of shape (n_samples, n_features)
        Matrix :math:`V` used for efficiency in updates.
    """

    def __init__(self, X: np.ndarray):
        """Initialize the Sherman-Morrison Ridge predictor.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.

        Raises
        ------
        InvalidDimensionsError
            If X is not a 2D array.
        ValueError
            If X contains NaN or infinity values.
        """
        if X.ndim != 2:
            raise InvalidDimensionsError("X must be a 2D array.")
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("X contains NaN or infinity values.")

        self.n_samples_, self.n_features_ = X.shape
        self.X_ = X
        self.gram_matrix_ = self.X_.T @ X / self.n_samples_
        self.A_ = np.eye(self.n_features_) + self.gram_matrix_
        self.A_inv_ = np.linalg.inv(self.A_)
        self.U_ = self.X_.T / np.sqrt(self.n_samples_)
        self.V_ = self.X_ / np.sqrt(self.n_samples_)

    def set_params(self, groups: GroupedFeatures, alpha: np.ndarray):
        r"""Update the regularization parameters and adjust A_inv_ accordingly.

        Parameters
        ----------
        groups : GroupedFeatures
            The grouped features object.
        alpha : np.ndarray
            The new :math:`\alpha` values for each group.

        Raises
        ------
        ValueError
            If alpha contains negative values.

        Returns
        -------
        None
        """
        if np.any(alpha < 0):
            raise ValueError("Alpha values must be non-negative.")

        diag = groups.group_expand(alpha)
        self.A_ = np.diag(diag) + self.gram_matrix_
        try:
            self.A_inv_ = np.linalg.inv(self.A_)
        except np.linalg.LinAlgError:
            raise SingularMatrixError("Failed to invert A. Matrix may be singular.")

    def _sherman_morrison_update(self, u: np.ndarray, v: np.ndarray):
        """Apply the Sherman-Morrison formula to update A_inv_ with a rank-one update.

        Parameters
        ----------
        u : np.ndarray
            Left vector for the rank-one update.
        v : np.ndarray
            Right vector for the rank-one update.

        Raises
        ------
        NumericalInstabilityError
            If numerical instability is detected during the update.

        Returns
        -------
        None
        """
        Au = self.A_inv_ @ u
        vA = v @ self.A_inv_
        denominator = 1.0 + v @ Au
        if abs(denominator) < 1e-10:
            raise NumericalInstabilityError(
                "Denominator in Sherman-Morrison update is close to zero."
            )
        self.A_inv_ -= np.outer(Au, vA) / denominator

    def _trace_gram_matrix(self) -> float:
        r"""Compute the trace of the Gram matrix.

        Returns
        -------
        float
            The trace of the Gram matrix.
        """
        return np.trace(self.gram_matrix_)

    def _solve_gram_system(self) -> np.ndarray:
        r"""Compute :math:`(X^T X + \alpha I)^{-1} X^T X`.

        Returns
        -------
        np.ndarray
            The result of the computation.
        """
        return self._solve_system(self.gram_matrix_)

    def _solve_system(self, B: np.ndarray) -> np.ndarray:
        r"""Solve the system :math:`(X^T X + \alpha I)^{-1} B` using the precomputed A_inv_.

        Parameters
        ----------
        B : np.ndarray
            The right-hand side matrix or vector.

        Returns
        -------
        np.ndarray
            The solution vector :math:`x`.
        """
        if B.ndim == 1:
            return self.A_inv_ @ B
        else:
            return self.A_inv_ @ B

    @staticmethod
    def _sherman_morrison_formula(
        A: np.ndarray, u: np.ndarray, v: np.ndarray
    ) -> np.ndarray:
        """Apply the Sherman-Morrison formula to matrix A with vectors u and v.

        Parameters
        ----------
        A : np.ndarray
            The current inverse matrix.
        u : np.ndarray
            Left vector for the rank-one update.
        v : np.ndarray
            Right vector for the rank-one update.

        Raises
        ------
        ValueError
            If the denominator in the update is zero.

        Returns
        -------
        np.ndarray
            The updated inverse matrix.
        """
        Au = A @ u
        vA = v @ A
        denominator = 1.0 + v @ Au
        if denominator == 0:
            raise ValueError("Denominator in Sherman-Morrison update is zero.")
        return A - np.outer(Au, vA) / denominator


class GroupRidgeRegressor(BaseEstimator, RegressorMixin):
    r"""Ridge regression with grouped features.

    This class implements Ridge regression with feature groups, automatically selecting
    the most efficient solver based on the problem dimensions. It supports Cholesky
    decomposition for n_samples > n_features, Woodbury identity for moderate-sized
    problems, and Sherman-Morrison updates for very high-dimensional problems.

    Parameters
    ----------
    groups : GroupedFeatures
        The grouped features object.
    alpha : Union[np.ndarray, dict], default=None
        The regularization parameters. Can be a numpy array or a dictionary
        mapping group names to :math:`\alpha` values. If None, uses ones.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during fit.
    n_samples_ : int
        Number of samples.
    n_features_ : int
        Number of features.
    groups_ : GroupedFeatures
        The grouped features object.
    predictor_ : BaseRidgePredictor
        The Ridge predictor being used.
    gram_target_ : np.ndarray of shape (n_features,)
        The :math:`X^T y` matrix scaled by n_samples.
    alpha_ : np.ndarray of shape (n_groups,)
        Regularization parameters for each group.
    coef_ : np.ndarray of shape (n_features,)
        Coefficient vector.
    leverage_ : np.ndarray of shape (n_samples,)
        Leverage scores for each sample.

    Notes
    -----
    The solver selection is based on the following rules:
    - If n_features <= n_samples: Use Cholesky decomposition
    - If n_samples < n_features < 4 * n_samples: Use Woodbury identity
    - If n_features >= 4 * n_samples: Use Sherman-Morrison updates
    """

    def __init__(self, groups: GroupedFeatures, alpha: Union[np.ndarray, dict] = None):
        self.groups = groups
        self.alpha = alpha

    def _validate_params(self):
        """Validate parameters.

        Raises
        ------
        ValueError
            If groups does not contain any groups or if any group has zero size.
        """
        if not self.groups.ps:
            raise ValueError("GroupedFeatures must contain at least one group.")
        if any(p == 0 for p in self.groups.ps):
            raise ValueError("GroupedFeatures groups must have non-zero sizes.")

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the Ridge regression model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        y : np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns self.
        """
        self._validate_params()
        X, y = check_X_y(X, y, accept_sparse=False)
        self.y_ = y  # Store y for later use

        self.n_features_in_ = X.shape[1]
        self.n_samples_, self.n_features_ = X.shape
        self.groups_ = self.groups

        # Initialize predictor based on problem dimensions
        if self.n_features_ <= self.n_samples_:
            self.predictor_ = CholeskyRidgePredictor(X)
        elif self.n_features_ > self.n_samples_ and self.n_features_ < 4 * self.n_samples_:
            self.predictor_ = WoodburyRidgePredictor(X)
        else:
            self.predictor_ = ShermanMorrisonRidgePredictor(X)

        self.gram_target_ = np.dot(X.T, y) / self.n_samples_

        # Set alpha values
        if self.alpha is None:
            self.alpha_ = np.ones(self.groups.num_groups)
        elif isinstance(self.alpha, dict):
            self.alpha_ = np.array(list(self.alpha.values()))
        else:
            self.alpha_ = np.asarray(self.alpha)

        self.predictor_.set_params(self.groups_, self.alpha_)
        self.coef_ = self.predictor_._solve_system(self.gram_target_)
        self.y_pred_ = np.dot(X, self.coef_)

        # Store gram_reg_inv_ for MomentTunerSetup
        self.gram_reg_inv_ = self.predictor_._solve_system(np.eye(self.n_features_))

        # Compute leverage scores
        self.gram_reg_inv_X_ = (
            self.predictor_._solve_system(X.T).T / self.n_samples_
        )
        self.leverage_ = np.sum(X * self.gram_reg_inv_X_, axis=1)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the Ridge regression model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples_new, n_features)
            New data.

        Returns
        -------
        np.ndarray of shape (n_samples_new,)
            Predicted values.
        """
        X = check_array(X, accept_sparse=False)
        return np.dot(X, self.coef_)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : Dict[str, Any]
            Parameter names mapped to their values.
        """
        return {"groups": self.groups, "alpha": self.alpha}

    def set_params(self, **params) -> "GroupRidgeRegressor":
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
        if "groups" in params:
            self.groups = params["groups"]
        if "alpha" in params:
            self.alpha = params["alpha"]
        return self

    def get_loo_error(self) -> float:
        """Compute the leave-one-out (LOO) error.

        Returns
        -------
        float
            The computed LOO error.
        """
        return np.mean((self.y_ - self.y_pred_) ** 2 / (1.0 - self.leverage_) ** 2)

    def get_mse(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Compute the mean squared error (MSE) on test data.

        Parameters
        ----------
        X_test : np.ndarray of shape (n_samples_test, n_features)
            Test design matrix.
        y_test : np.ndarray of shape (n_samples_test,)
            Test response vector.

        Returns
        -------
        float
            The computed MSE.
        """
        X_test = check_array(X_test, accept_sparse=False)
        return np.mean((y_test - self.predict(X_test)) ** 2)


def lambda_lolas_rule(rdg: GroupRidgeRegressor, multiplier: float = 0.1) -> float:
    r"""Compute the regularization parameter using the Panagiotis Lolas rule.

    The Lolas rule provides a heuristic for selecting the regularization parameter
    based on the model's degrees of freedom and the trace of :math:`X^T X`.

    The :math:`\alpha` is computed as:

    .. math::
        \alpha = \text{multiplier} \times \frac{n_{\text{features}}^2}{n_{\text{samples}} \times \text{trace}(X^T X)}

    Parameters
    ----------
    rdg : GroupRidgeRegressor
        The Ridge regression estimator.
    multiplier : float, default=0.1
        A scalar multiplier for the rule.

    Returns
    -------
    float
        The computed :math:`\alpha` value.

    Raises
    ------
    ValueError
        If multiplier is not positive or if trace(X^T X) is zero.
    """
    if multiplier <= 0:
        raise ValueError("Multiplier must be positive.")

    trace_gram = rdg.predictor_._trace_gram_matrix()
    if trace_gram == 0:
        raise ValueError("Trace of X^T X is zero, leading to division by zero.")

    return multiplier * rdg.n_features_**2 / rdg.n_samples_ / trace_gram


class MomentTunerSetup:
    r"""Setup for the moment-based tuning of regularization parameters.

    This class prepares and computes moment-based statistics required for tuning
    the regularization parameters (:math:`\lambda`) in Ridge regression. By leveraging
    moments of the coefficients and the design matrix, it facilitates principled
    selection of :math:`\lambda` values that balance bias and variance.

    Attributes
    ----------
    groups : GroupedFeatures
        The grouped features object.
    ps : np.ndarray
        Array of the number of features in each group.
    n : int
        Number of samples.
    beta_norms_squared : np.ndarray
        Squared norms of coefficients for each group.
    N_norms_squared : np.ndarray
        Squared norms of the :math:`N` matrix for each group.
    M_squared : np.ndarray
        :math:`M^2` matrix computed as (:math:`p_s \times p_s^T) / n^2`.
    """

    def __init__(self, rdg: GroupRidgeRegressor):
        self.groups_ = rdg.groups_
        self.n_features_per_group_ = np.array(rdg.groups_.ps)
        self.n_samples_ = rdg.n_samples_
        self.coef_norms_squared_ = np.array(
            rdg.groups_.group_summary(rdg.coef_, lambda x: np.sum(np.abs(x) ** 2))
        )
        gram_inv_matrix = rdg.gram_reg_inv_  # Use the (p, p) inverse matrix
        if gram_inv_matrix.shape[1] != self.n_features_per_group_.sum():
            raise ValueError(
                f"Length of gram_inv_matrix ({gram_inv_matrix.shape[1]}) does not match "
                f"number of features ({self.n_features_per_group_.sum()})"
            )
        self.gram_inv_norms_squared_ = np.array(
            rdg.groups_.group_summary(gram_inv_matrix, lambda x: np.sum(np.abs(x) ** 2))
        )
        self.moment_matrix_ = (
            np.outer(self.n_features_per_group_, self.n_features_per_group_)
            / self.n_samples_**2
        )


def sigma_squared_path(
    rdg: GroupRidgeRegressor, mom: MomentTunerSetup, sigma_s_squared: np.ndarray
):
    r"""Compute the regularization path for different values of :math:`\sigma^2`.

    This function evaluates how the Ridge regression coefficients and leave-one-out
    (LOO) errors change as :math:`\sigma^2` varies. By analyzing the regularization path,
    one can understand the impact of different levels of regularization on the
    model's performance.

    For each :math:`\sigma^2` in the provided array:

    1. Compute :math:`\alpha`:
        :math:`\alpha = \text{get_lambdas}(mom, \sigma^2)`

    2. Fit Ridge Model:
        Fit the Ridge regression model using the computed :math:`\alpha` and evaluate:
            LOO Error = rdg.fit(:math:`\alpha`)

    3. Store Coefficients:
        :math:`\beta = rdg.coef_`

    4. Aggregate Results:
        Collect the :math:`\alpha` values, LOO errors, and coefficients for each :math:`\sigma^2`.

    Parameters
    ----------
    rdg : GroupRidgeRegressor
        The Ridge regression estimator.
    mom : MomentTunerSetup
        The moment tuner setup containing necessary statistics.
    sigma_s_squared : np.ndarray of shape (n_sigmas,)
        An array of :math:`\sigma^2` values to evaluate.

    Returns
    -------
    dict
        A dictionary containing the regularization path information:
            - 'alphas' : np.ndarray of shape (n_sigmas, n_groups)
                Array of :math:`\alpha` values for each :math:`\sigma^2`.
            - 'errors' : np.ndarray of shape (n_sigmas,)
                Array of LOO errors corresponding to each :math:`\sigma^2`.
            - 'coefs' : np.ndarray of shape (n_sigmas, n_features)
                Array of coefficient vectors corresponding to each :math:`\sigma^2`.

    Raises
    ------
    ValueError
        If sigma_s_squared contains negative values.
    """
    if np.any(sigma_s_squared < 0):
        raise ValueError("sigma_s_squared values must be non-negative.")

    n_sigmas = len(sigma_s_squared)
    n_groups = rdg.get_n_groups()
    errors = np.zeros(n_sigmas)
    alphas = np.zeros((n_sigmas, n_groups))
    coefs = np.zeros((n_sigmas, rdg.groups_.p))

    for i, sigma_sq in enumerate(sigma_s_squared):
        try:
            alpha_tmp = get_lambdas(mom, sigma_sq)
            alphas[i, :] = alpha_tmp
            errors[i] = rdg.fit(alpha_tmp)
            coefs[i, :] = rdg.coef_
        except RidgeRegressionError as e:
            print(f"Error at :math:`\sigma^2 = {sigma_sq}`: {str(e)}")

    return {"alphas": alphas, "errors": errors, "coefs": coefs}


def get_lambdas(mom: MomentTunerSetup, sigma_sq: float) -> np.ndarray:
    r"""Compute :math:`\alpha` values for a given :math:`\sigma^2`.

    This function calculates the regularization parameters (:math:`\alpha`) for each
    feature group based on moment-based statistics. The computed :math:`\alpha` balances
    the regularization strength across different groups, ensuring that groups with
    more features or higher variance receive appropriate penalization.

    The steps are as follows:

    1. Compute :math:`\alpha^2` for each group:
        :math:`\alpha_g^2 = \max(\|\beta_g\|^2 - \sigma^2 \|N_g\|^2, 0) / p_g`

    2. Compute group ratios:
        :math:`r_g = p_g / n`

    3. Compute :math:`\alpha` for each group:
        :math:`\alpha_g = (\sigma^2 r_g) / \alpha_g^2`

    Parameters
    ----------
    mom : MomentTunerSetup
        The moment tuner setup containing necessary statistics.
    sigma_sq : float
        The :math:`\sigma^2` value for which to compute :math:`\alpha`.

    Returns
    -------
    np.ndarray of shape (n_groups,)
        The computed :math:`\alpha` values for each feature group.

    Raises
    ------
    ValueError
        If sigma_sq is negative.
    NumericalInstabilityError
        If division by zero occurs during alpha computation.
    """
    if sigma_sq < 0:
        raise ValueError("sigma_sq must be non-negative.")

    alpha_squared = get_alpha_s_squared(mom, sigma_sq)
    group_ratios = np.array(mom.n_features_per_group_) / mom.n_samples_

    LARGE_VALUE = 1e12

    with np.errstate(divide="ignore", invalid="ignore"):
        alphas = sigma_sq * group_ratios / alpha_squared
        zero_alpha = alpha_squared == 0
        if np.any(zero_alpha):
            warnings.warn(
                f"alpha_squared has zero values for groups: {np.where(zero_alpha)[0]}. "
                "Assigning large alpha values to these groups."
            )
        alphas = np.where(zero_alpha, LARGE_VALUE, alphas)

    return alphas


def get_alpha_s_squared(mom: MomentTunerSetup, sigma_sq: float) -> np.ndarray:
    r"""Compute :math:`\alpha^2` values for a given :math:`\sigma^2` using Non-Negative Least Squares
    (NNLS).

    This function calculates the :math:`\alpha^2` values required for determining the
    regularization parameters (:math:`\alpha`) in Ridge regression. The :math:`\alpha^2` values
    encapsulate the balance between the coefficient norms and the influence of
    the design matrix, adjusted by :math:`\sigma^2`.

    The steps are as follows:

    1. Compute the right-hand side (RHS):
        :math:`\text{RHS}_g = \|\beta_g\|^2 - \sigma^2 \|N_g\|^2`

    2. Solve the NNLS problem:
        :math:`\min \| M \times \alpha_{\text{per\_group}} - \text{RHS} \|_2^2`
        subject to :math:`\alpha_{\text{per\_group}} \geq 0`

    3. Compute :math:`\alpha^2`:
        :math:`\alpha^2 = \alpha_{\text{per\_group}} \times p_g`

    Parameters
    ----------
    mom : MomentTunerSetup
        The moment tuner setup containing necessary statistics.
    sigma_sq : float
        The :math:`\sigma^2` value for which to compute :math:`\alpha^2`.

    Returns
    -------
    np.ndarray of shape (n_groups,)
        The computed :math:`\alpha^2` values for each feature group.

    Raises
    ------
    ValueError
        If sigma_sq is negative or if any n_features_per_group is zero.
    NNLSError
        If the NNLS algorithm fails to converge.
    """
    if sigma_sq < 0:
        raise ValueError("sigma_sq must be non-negative.")
    if np.any(mom.n_features_per_group_ == 0):
        raise ValueError("All n_features_per_group values must be non-zero.")

    # Compute the right-hand side
    rhs = mom.coef_norms_squared_ - sigma_sq * mom.gram_inv_norms_squared_
    rhs = np.maximum(rhs, 0)

    # Solve the NNLS problem: moment_matrix * alpha_per_group ≈ rhs
    try:
        alpha_per_group = nonneg_lsq(mom.moment_matrix_, rhs, alg="fnnls")
    except NNLSError as e:
        raise NNLSError(f"Failed to compute alpha_squared: {str(e)}")

    alpha_squared = alpha_per_group * mom.n_features_per_group_

    return alpha_squared