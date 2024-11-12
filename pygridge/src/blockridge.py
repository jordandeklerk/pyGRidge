"""Group Ridge regression estimators."""

from abc import ABC, abstractmethod
from typing import Union, TypeVar, Dict, Any, Optional
import numpy as np
from scipy.linalg import cho_solve
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from .groupedfeatures import GroupedFeatures
from .nnls import nonneg_lsq, NNLSError
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

        # Add small regularization to handle near-singular matrices
        eps = np.finfo(X.dtype).eps
        reg_term = eps * np.trace(X.T @ X) * np.eye(self.n_features_)
        self.gram_matrix_ = (np.dot(X.T, X) / self.n_samples_) + reg_term

        # Check if gram matrix is nearly singular
        cond = np.linalg.cond(self.gram_matrix_)
        if cond > 1e15:
            # Add larger regularization for very ill-conditioned matrices
            reg_term = 1e-6 * np.trace(X.T @ X) * np.eye(self.n_features_)
            self.gram_matrix_ += reg_term
            warnings.warn(
                f"Added regularization term {1e-6:.2e} * tr(X^T X) * I to handle near-singular matrix"
            )

        # Handle zero variance features
        diag = np.diag(self.gram_matrix_)
        zero_var_mask = diag < eps
        if np.any(zero_var_mask):
            # For zero variance features, use a larger regularization
            reg_term = np.zeros_like(self.gram_matrix_)
            np.fill_diagonal(reg_term, zero_var_mask.astype(float))
            self.gram_matrix_ += reg_term

        self.gram_reg_ = self.gram_matrix_.copy()
        self._update_cholesky()
        self.lower_ = True

    def _update_cholesky(self):
        """Update the Cholesky decomposition of the regularized Gram matrix.

        Raises
        ------
        SingularMatrixError
            If the matrix is not positive definite.
        """
        try:
            self.gram_reg_chol_ = np.linalg.cholesky(self.gram_reg_)
        except np.linalg.LinAlgError:
            # Progressive regularization strategy
            regularization_factors = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0]
            trace_term = max(np.trace(self.gram_reg_), 1e-10)

            for factor in regularization_factors:
                try:
                    reg_matrix = self.gram_reg_.copy()
                    # Add regularization to diagonal
                    np.fill_diagonal(
                        reg_matrix, np.diag(reg_matrix) + factor * trace_term
                    )
                    self.gram_reg_chol_ = np.linalg.cholesky(reg_matrix)
                    self.gram_reg_ = reg_matrix
                    warnings.warn(f"Added regularization {factor:.2e} * tr(X^T X) * I")
                    return
                except np.linalg.LinAlgError:
                    continue

            # If all regularization attempts fail, try one last time with very large regularization
            try:
                reg_matrix = self.gram_reg_.copy()
                np.fill_diagonal(reg_matrix, np.diag(reg_matrix) + trace_term)
                self.gram_reg_chol_ = np.linalg.cholesky(reg_matrix)
                self.gram_reg_ = reg_matrix
                warnings.warn("Added maximum regularization")
                return
            except np.linalg.LinAlgError:
                pass

            raise SingularMatrixError(
                "Failed to compute Cholesky decomposition. Matrix may not be positive definite."
            )

    def set_params(self, groups: GroupedFeatures, alpha: np.ndarray):
        """Update the regularization parameters."""
        diag = groups.group_expand(alpha)
        if not isinstance(diag, np.ndarray):
            diag = np.array(diag)

        if len(diag) != self.n_features_:
            raise ValueError(
                f"Alpha expansion length ({len(diag)}) must match the number of features ({self.n_features_})"
            )

        eps = np.finfo(self.gram_matrix_.dtype).eps
        min_alpha = eps * np.trace(self.gram_matrix_)
        diag = np.maximum(diag, min_alpha)

        zero_var_mask = np.diag(self.gram_matrix_) < eps
        if np.any(zero_var_mask):
            diag[zero_var_mask] = max(1e-4 * np.trace(self.gram_matrix_), 1.0)

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

        # Check if gram matrix is nearly singular
        cond = np.linalg.cond(self.gram_matrix_)
        if cond > 1e15:
            raise SingularMatrixError(
                f"Gram matrix is nearly singular with condition number: {cond}"
            )

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
        if len(diag) != self.n_features_:
            raise ValueError(
                f"Alpha expansion length ({len(diag)}) must match the number of features ({self.n_features_})"
            )

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

        reg_term = 1e-6 * np.eye(self.n_features_)
        self.gram_matrix_ = (self.X_.T @ X / self.n_samples_) + reg_term

        cond = np.linalg.cond(self.gram_matrix_)
        if cond > 1e20:
            raise SingularMatrixError(
                f"Gram matrix is nearly singular with condition number: {cond}"
            )

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
        if len(diag) != self.n_features_:
            raise ValueError(
                f"Alpha expansion length ({len(diag)}) must match the number of features ({self.n_features_})"
            )

        self.A_ = np.diag(diag) + self.gram_matrix_
        try:
            # Check condition number before inversion
            cond = np.linalg.cond(self.A_)
            if cond > 1e15:
                raise SingularMatrixError(
                    f"Matrix is nearly singular with condition number: {cond}"
                )
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
    groups : Optional[GroupedFeatures], default=None
        The grouped features object. If None, a single group containing all features
        will be created during fit.
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

    def __init__(
        self,
        groups: Optional[GroupedFeatures] = None,
        alpha: Union[np.ndarray, dict, float] = None,
    ):
        self.groups = groups
        self.alpha = alpha

    def _validate_params(self):
        """Validate parameters.

        Raises
        ------
        ValueError
            If groups does not contain any groups or if any group has zero size.
        """
        if hasattr(self, "groups_") and self.groups_ is not None:
            if not isinstance(self.groups_, GroupedFeatures):
                raise ValueError("groups must be a GroupedFeatures instance")
            if not hasattr(self.groups_, "ps") or not self.groups_.ps:
                raise ValueError("GroupedFeatures must contain at least one group")
            if any(p == 0 for p in self.groups_.ps):
                raise ValueError("GroupedFeatures groups must have non-zero sizes")

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
        X, y = check_X_y(X, y, accept_sparse=False, multi_output=False)
        
        # Convert X and y to float64 if they are not already floating point
        if not np.issubdtype(X.dtype, np.floating):
            X = X.astype(np.float64)
        if not np.issubdtype(y.dtype, np.floating):
            y = y.astype(np.float64)

        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.arange(self.n_features_in_)
        self.n_samples_, self.n_features_ = X.shape

        if self.groups is None:
            self.groups_ = GroupedFeatures([self.n_features_])
        else:
            self.groups_ = clone(self.groups)

        self.groups_.fit(X)
        self.X_ = X
        self.y_ = y

        if self.n_features_ >= 4 * self.n_samples_:
            self.predictor_ = ShermanMorrisonRidgePredictor(X)
        elif self.n_features_ <= self.n_samples_:
            self.predictor_ = CholeskyRidgePredictor(X)
        else:
            self.predictor_ = WoodburyRidgePredictor(X)

        self.gram_target_ = np.dot(X.T, y) / self.n_samples_

        if self.alpha is None:
            self.alpha_ = np.ones(self.groups_.num_groups)
        elif isinstance(self.alpha, dict):
            self.alpha_ = np.array(
                [self.alpha.get(name, 1.0) for name in self.groups_.names]
            )
        elif isinstance(self.alpha, (int, float)):
            self.alpha_ = np.full(self.groups_.num_groups, self.alpha)
        else:
            self.alpha_ = np.asarray(self.alpha)
            if self.alpha_.size == 1:
                self.alpha_ = np.full(self.groups_.num_groups, self.alpha_[0])
            elif self.alpha_.size != self.groups_.num_groups:
                raise ValueError(
                    f"Alpha length ({self.alpha_.size}) must match number of groups ({self.groups_.num_groups})"
                )

        self.predictor_.set_params(self.groups_, self.alpha_)
        self.coef_ = self.predictor_._solve_system(self.gram_target_)
        self.y_pred_ = np.dot(X, self.coef_)

        self.gram_reg_inv_ = self.predictor_._solve_system(np.eye(self.n_features_))

        self.gram_reg_inv_X_ = self.predictor_._solve_system(X.T).T / self.n_samples_
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
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but {self.n_features_in_} features were seen during fit"
            )

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
        params = super().get_params(deep=deep)
        if deep and hasattr(self.groups, "get_params"):
            groups_params = self.groups.get_params(deep=True)
            params.update({f"groups__{key}": val for key, val in groups_params.items()})
        return params

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
        groups_params = {}
        own_params = {}
        for key, value in params.items():
            if key.startswith("groups__"):
                groups_params[key.split("__", 1)[1]] = value
            else:
                own_params[key] = value

        if groups_params and hasattr(self.groups, "set_params"):
            self.groups.set_params(**groups_params)
        if own_params:
            super().set_params(**own_params)
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


def lambda_lolas_rule(estimator: GroupRidgeRegressor, multiplier: float = 0.1) -> float:
    r"""Compute the regularization parameter using the Panagiotis Lolas rule.

    The Lolas rule provides a heuristic for selecting the regularization parameter
    based on the model's degrees of freedom and the trace of :math:`X^T X`.

    The :math:`\alpha` is computed as:

    .. math::
        \alpha = \text{multiplier} \times \frac{n_{\text{features}}^2}{n_{\text{samples}} \times \text{trace}(X^T X)}

    Parameters
    ----------
    estimator : GroupRidgeRegressor
        The fitted group ridge regression estimator.
    multiplier : float, default=0.1
        Scaling factor for the regularization parameter.

    Returns
    -------
    float
        The computed :math:`\alpha` value.

    Raises
    ------
    ValueError
        If multiplier is not positive or if the estimator is not fitted.
        If trace(X^T X) is zero.
    """
    if not hasattr(estimator, "predictor_"):
        raise ValueError("Estimator must be fitted before calling lambda_lolas_rule.")

    if not isinstance(multiplier, (int, float)):
        raise TypeError("multiplier must be a number.")
    if multiplier <= 0:
        raise ValueError("multiplier must be positive.")

    trace_gram = estimator.predictor_._trace_gram_matrix()
    if np.isclose(trace_gram, 0):
        raise ValueError("Trace of X^T X is zero, leading to division by zero.")

    return multiplier * estimator.n_features_**2 / estimator.n_samples_ / trace_gram


class MomentTunerSetup:
    """Setup for moment-based tuning of regularization parameters.

    This class prepares the necessary statistics for tuning regularization parameters
    using moment-based methods. It encapsulates the computation of coefficient norms,
    gram matrix norms, and moment matrices needed for parameter selection.

    Parameters
    ----------
    estimator : GroupRidgeRegressor
        The fitted group ridge regression estimator.

    Attributes
    ----------
    groups_ : GroupedFeatures
        The grouped features object.
    n_features_per_group_ : ndarray of shape (n_groups,)
        Number of features in each group.
    n_samples_ : int
        Number of samples in the training data.
    coef_norms_squared_ : ndarray of shape (n_groups,)
        Squared L2 norms of coefficients for each group.
    gram_inv_norms_squared_ : ndarray of shape (n_groups,)
        Squared Frobenius norms of gram matrix inverse blocks for each group.
    moment_matrix_ : ndarray of shape (n_groups, n_groups)
        Matrix of scaled outer products of group sizes.

    Raises
    ------
    ValueError
        If the estimator is not fitted or has incompatible dimensions.
    """

    def __init__(self, estimator: GroupRidgeRegressor):
        if not hasattr(estimator, "predictor_"):
            raise ValueError(
                "Estimator must be fitted before MomentTunerSetup initialization."
            )

        self.groups_ = estimator.groups_
        self.n_features_per_group_ = np.array(estimator.groups_.ps)
        self.n_samples_ = estimator.n_samples_

        self.coef_norms_squared_ = np.array(
            estimator.groups_.group_summary(
                estimator.coef_, lambda x: np.sum(np.abs(x) ** 2)
            )
        )

        gram_inv_matrix = estimator.gram_reg_inv_
        total_features = self.n_features_per_group_.sum()
        if gram_inv_matrix.shape[1] != total_features:
            raise ValueError(
                f"Gram inverse matrix dimension ({gram_inv_matrix.shape[1]}) "
                f"does not match total features ({total_features})"
            )

        self.gram_inv_norms_squared_ = np.array(
            estimator.groups_.group_summary(
                gram_inv_matrix, lambda x: np.sum(np.abs(x) ** 2)
            )
        )

        self.moment_matrix_ = (
            np.outer(self.n_features_per_group_, self.n_features_per_group_)
            / self.n_samples_**2
        )

    def get_sigma_squared_max(self) -> float:
        r"""Compute the maximum value of :math:`\sigma^2`.

        The maximum :math:`\sigma^2` is defined as:

        .. math::
            \sigma_{\max}^2 = \max_g \left\{ \frac{\hat{u}_g}{v_g} \right\}

        where for each group :math:`g`:

        .. math::
            \hat{u}_g &= \|\tilde{w}_{G_g}\|_2^2
            v_g &= \|N_{G_g,\cdot}\|_F^2/n

        Returns
        -------
        float
            The maximum value of :math:`\sigma^2`.
        """
        u_g = self.coef_norms_squared_
        v_g = self.gram_inv_norms_squared_

        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = u_g / v_g
            valid_ratios = ratios[np.isfinite(ratios)]

        if len(valid_ratios) == 0:
            return 1.0

        return np.max(valid_ratios)


def sigma_squared_path(
    estimator: GroupRidgeRegressor,
    moment_tuner: MomentTunerSetup,
    sigma_squared_values: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-4,
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
    estimator : GroupRidgeRegressor
        The group ridge regression estimator.
    moment_tuner : MomentTunerSetup
        The moment tuner setup containing necessary statistics.
    sigma_squared_values : ndarray of shape (n_values,)
        Array of noise variance values to evaluate.
    max_iter : int, default=100
        Maximum number of iterations for optimization.
    tol : float, default=1e-4
        Tolerance for optimization convergence.

    Returns
    -------
    dict
        Dictionary containing:
        - 'alphas': ndarray of shape (n_values, n_groups)
            Regularization parameters for each sigma squared value.
        - 'errors': ndarray of shape (n_values,)
            Leave-one-out errors for each sigma squared value.
        - 'coefs': ndarray of shape (n_values, n_features)
            Model coefficients for each sigma squared value.

    Raises
    ------
    ValueError
        If sigma_squared_values contains negative values or
        if max_iter or tol have invalid values.
    """
    if not isinstance(sigma_squared_values, np.ndarray):
        sigma_squared_values = np.asarray(sigma_squared_values)

    if np.any(sigma_squared_values < 0):
        raise ValueError("sigma_squared_values must be non-negative")

    if not isinstance(max_iter, int) or max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")

    if not isinstance(tol, float) or tol <= 0:
        raise ValueError("tol must be a positive float")

    n_values = len(sigma_squared_values)
    n_groups = len(moment_tuner.n_features_per_group_)
    n_features = moment_tuner.n_features_per_group_.sum()

    alphas = np.zeros((n_values, n_groups))
    errors = np.zeros(n_values)
    coefs = np.zeros((n_values, n_features))

    for i, sigma_sq in enumerate(sigma_squared_values):
        try:
            alpha_values = get_lambdas(moment_tuner, sigma_sq)
            alphas[i] = alpha_values

            estimator.set_params(alpha=alpha_values)
            estimator.fit(estimator.X_, estimator.y_)
            errors[i] = estimator.get_loo_error()
            coefs[i] = estimator.coef_

        except Exception as e:
            warnings.warn(f"Error at sigma_squared={sigma_sq}: {str(e)}")
            continue

    return {"alphas": alphas, "errors": errors, "coefs": coefs}


def get_lambdas(moment_tuner: MomentTunerSetup, sigma_squared: float) -> np.ndarray:
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
    moment_tuner : MomentTunerSetup
        The moment tuner setup containing necessary statistics.
    sigma_squared : float
        The noise variance value.

    Returns
    -------
    ndarray of shape (n_groups,)
        Optimal regularization parameters for each group.

    Raises
    ------
    ValueError
        If sigma_squared is negative.
    """
    if not isinstance(sigma_squared, (int, float)):
        raise TypeError("sigma_squared must be a number")
    if sigma_squared < 0:
        raise ValueError("sigma_squared must be non-negative")

    alpha_squared = get_alpha_s_squared(moment_tuner, sigma_squared)
    group_ratios = moment_tuner.n_features_per_group_ / moment_tuner.n_samples_

    LARGE_VALUE = 1e12
    with np.errstate(divide="ignore", invalid="ignore"):
        lambdas = sigma_squared * group_ratios / alpha_squared
        zero_mask = np.isclose(alpha_squared, 0)
        if np.any(zero_mask):
            warnings.warn(
                f"Assigning large regularization values to groups: {np.where(zero_mask)[0]}"
            )
        lambdas = np.where(zero_mask, LARGE_VALUE, lambdas)

    return lambdas


def get_alpha_s_squared(
    moment_tuner: MomentTunerSetup, sigma_squared: float
) -> np.ndarray:
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
    moment_tuner : MomentTunerSetup
        The moment tuner setup containing necessary statistics.
    sigma_squared : float
        The noise variance value.

    Returns
    -------
    ndarray of shape (n_groups,)
        Squared alpha values for each group.

    Raises
    ------
    ValueError
        If sigma_squared is negative or if group sizes are invalid.
    NNLSError
        If the non-negative least squares optimization fails.
    """
    if not isinstance(sigma_squared, (int, float)):
        raise TypeError("sigma_squared must be a number")
    if sigma_squared < 0:
        raise ValueError("sigma_squared must be non-negative")

    if np.any(moment_tuner.n_features_per_group_ <= 0):
        raise ValueError("All group sizes must be positive")

    rhs = (
        moment_tuner.coef_norms_squared_
        - sigma_squared * moment_tuner.gram_inv_norms_squared_
    )
    rhs = np.maximum(rhs, 0)

    try:
        # Solve non-negative least squares problem
        alpha_per_group = nonneg_lsq(moment_tuner.moment_matrix_, rhs, alg="fnnls")
    except NNLSError as e:
        raise NNLSError(f"Non-negative least squares optimization failed: {str(e)}")

    alpha_squared = alpha_per_group * moment_tuner.n_features_per_group_

    return alpha_squared
