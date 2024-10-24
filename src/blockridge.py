"""Regression estimators based on group ridge regression."""

from abc import ABC, abstractmethod
from typing import Union, TypeVar
import numpy as np
from scipy.linalg import cho_solve
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


class AbstractRidgePredictor(ABC):
    r"""Abstract base class for Ridge regression predictors.

    This class defines the interface that all concrete Ridge predictors must
    implement, ensuring consistency in how regularization parameters are updated
    and how various mathematical operations related to Ridge regression are
    performed.

    Ridge regression seeks to minimize the following objective function:

    .. math::
        \min_{\beta} \|Y - X\beta\|_2^2 + \lambda\|\beta\|_2^2

    where:
    - :math:`Y` is the response vector with dimensions :math:`(n, )`.
    - :math:`X` is the design matrix with dimensions :math:`(n, p)`.
    - :math:`\beta` is the coefficient vector with dimensions :math:`(p, )`.
    - :math:`\lambda` is the regularization parameter.

    The solution to this optimization problem is given by:

    .. math::
        \beta = (X^T X + \lambda I_p)^{-1} X^T Y
    """

    @abstractmethod
    def update_lambda_s(self, groups: GroupedFeatures, lambdas: np.ndarray):
        r"""Update the regularization parameters (:math:`\lambda`) based on feature groups.

        Parameters
        ----------
        groups : GroupedFeatures
            The grouped features object defining feature groupings.
        lambdas : np.ndarray
            The new :math:`\lambda` values for each group.

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def trace_XtX(self) -> float:
        """Compute the trace of :math:`X^T X`.

        Returns
        -------
        float
            The trace of :math:`X^T X`.
        """
        pass

    @abstractmethod
    def XtXp_lambda_ldiv_XtX(self) -> np.ndarray:
        r"""Compute :math:`(X^T X + \Lambda)^{-1} X^T X`.

        Returns
        -------
        np.ndarray
            The result of the computation.
        """
        pass

    @abstractmethod
    def ldiv(self, B: np.ndarray) -> np.ndarray:
        r"""Solve the linear system :math:`(X^T X + \Lambda) x = B`.

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


class CholeskyRidgePredictor(AbstractRidgePredictor):
    r"""Ridge predictor using Cholesky decomposition for efficient matrix inversion.

    Suitable for scenarios where the number of features (:math:`p`) is less than the number
    of samples (:math:`n`), leveraging the properties of Cholesky decomposition to solve
    the Ridge regression problem efficiently.

    Attributes
    ----------
    n : int
        Number of samples.
    p : int
        Number of features.
    XtX : np.ndarray
        The :math:`X^T X` matrix scaled by the number of samples.
    XtXp_lambda : np.ndarray
        The augmented matrix (:math:`X^T X + \Lambda`).
    XtXp_lambda_chol : np.ndarray
        The Cholesky decomposition of (:math:`X^T X + \Lambda`).
    lower : bool
        Indicates if the Cholesky factor is lower triangular.
    """

    def __init__(self, X: np.ndarray):
        """Initialize the Cholesky Ridge predictor.

        Parameters
        ----------
        X : np.ndarray
            The design matrix.

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

        self.n, self.p = X.shape
        self.XtX = np.dot(X.T, X) / self.n
        self.XtXp_lambda = self.XtX + np.eye(self.p)  # Initialize with identity matrix
        self.update_cholesky()
        self.lower = True

    def update_cholesky(self):
        r"""Update the Cholesky decomposition of (:math:`X^T X + \Lambda`).

        Raises
        ------
        SingularMatrixError
            If the matrix is not positive definite.
        """
        try:
            self.XtXp_lambda_chol = np.linalg.cholesky(self.XtXp_lambda)
        except np.linalg.LinAlgError:
            raise SingularMatrixError(
                "Failed to compute Cholesky decomposition. Matrix may not be positive "
                "definite."
            )

    def update_lambda_s(self, groups: GroupedFeatures, lambdas: np.ndarray):
        r"""Update the regularization parameters and recompute the Cholesky decomposition.

        Parameters
        ----------
        groups : GroupedFeatures
            The grouped features object.
        lambdas : np.ndarray
            The new :math:`\lambda` values for each group.

        Returns
        -------
        None
        """
        diag = groups.group_expand(lambdas)
        self.XtXp_lambda = self.XtX + np.diag(diag)
        self.update_cholesky()

    def trace_XtX(self) -> float:
        """Compute the trace of :math:`X^T X`.

        Returns
        -------
        float
            The trace of :math:`X^T X`.
        """
        return np.trace(self.XtX)

    def XtXp_lambda_ldiv_XtX(self) -> np.ndarray:
        r"""Compute :math:`(X^T X + \Lambda)^{-1} X^T X` using Cholesky decomposition.

        Returns
        -------
        np.ndarray
            The result of the computation.
        """
        return cho_solve((self.XtXp_lambda_chol, self.lower), self.XtX)

    def ldiv(self, B: np.ndarray) -> np.ndarray:
        r"""Solve the system :math:`(X^T X + \Lambda) x = B` using Cholesky decomposition.

        Parameters
        ----------
        B : np.ndarray
            The right-hand side of the equation.

        Returns
        -------
        np.ndarray
            The solution vector :math:`x`.
        """
        return cho_solve((self.XtXp_lambda_chol, self.lower), B)


class WoodburyRidgePredictor(AbstractRidgePredictor):
    r"""Ridge predictor using the Woodbury matrix identity for efficient matrix inversion.

    This class is suitable for scenarios where the number of features (:math:`p`) is greater
    than the number of samples (:math:`n`), leveraging the Woodbury matrix identity to solve
    the Ridge regression problem efficiently.

    The Woodbury matrix identity states:

    .. math::
        (A + UCV)^{-1} = A^{-1} - A^{-1}U(C^{-1} + V A^{-1} U)^{-1} V A^{-1}

    In the context of Ridge regression:
        :math:`A = \Lambda` (diagonal matrix of regularization parameters)
        :math:`U = X^T`
        :math:`C = I` (identity matrix)
        :math:`V = X`

    Attributes
    ----------
    n : int
        Number of samples.
    p : int
        Number of features.
    X : np.ndarray
        The design matrix.
    XtX : np.ndarray
        The :math:`X^T X` matrix.
    A_inv : np.ndarray
        The inverse of :math:`\Lambda`.
    U : np.ndarray
        Matrix :math:`U` in the Woodbury identity.
    V : np.ndarray
        Matrix :math:`V` in the Woodbury identity.
    """

    def __init__(self, X: np.ndarray):
        """Initialize the Woodbury Ridge predictor.

        Parameters
        ----------
        X : np.ndarray
            The design matrix.

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

        self.n, self.p = X.shape
        self.X = X
        self.XtX = X.T @ X
        self.A_inv = np.eye(self.p)
        self.U = X.T
        self.V = X

    def update_lambda_s(self, groups: GroupedFeatures, lambdas: np.ndarray):
        r"""Update the regularization parameters and recompute the inverse matrix.

        Parameters
        ----------
        groups : GroupedFeatures
            The grouped features object.
        lambdas : np.ndarray
            The new :math:`\lambda` values for each group.

        Raises
        ------
        ValueError
            If lambdas contain non-positive values.

        Returns
        -------
        None
        """
        if np.any(lambdas <= 0):
            raise ValueError("Lambda values must be positive.")

        diag = np.array(groups.group_expand(lambdas))
        self.A_inv = np.diag(1 / diag)
        self.woodbury_update()

    def woodbury_update(self):
        """Apply the Woodbury matrix identity to update the inverse matrix.

        Raises
        ------
        NumericalInstabilityError
            If numerical instability is detected during the update.
        """
        try:
            eye = np.eye(self.n)
            AU = self.A_inv @ self.U
            inv_term = np.linalg.inv(eye + self.V @ AU)
            self.A_inv -= AU @ inv_term @ self.V @ self.A_inv
        except np.linalg.LinAlgError:
            raise NumericalInstabilityError(
                "Numerical instability detected in Woodbury update."
            )

    def trace_XtX(self) -> float:
        """Compute the trace of :math:`X^T X`.

        Returns
        -------
        float
            The trace of :math:`X^T X`.
        """
        return np.trace(self.XtX)

    def XtXp_lambda_ldiv_XtX(self) -> np.ndarray:
        r"""Compute :math:`(X^T X + \Lambda)^{-1} X^T X`.

        Returns
        -------
        np.ndarray
            The result of the computation.
        """
        return self.A_inv @ self.XtX

    def ldiv(self, B: np.ndarray) -> np.ndarray:
        r"""Solve the system :math:`(X^T X + \Lambda) x = B`.

        Parameters
        ----------
        B : np.ndarray
            The right-hand side matrix or vector.

        Returns
        -------
        np.ndarray
            The solution vector :math:`x`.
        """
        return self.A_inv @ B


class ShermanMorrisonRidgePredictor(AbstractRidgePredictor):
    r"""Ridge predictor using the Sherman-Morrison formula for efficient matrix updates.

    This class is suitable for scenarios where the number of features (:math:`p`) is much
    greater than the number of samples (:math:`n`), leveraging the Sherman-Morrison formula
    to efficiently update the inverse of (:math:`X^T X + \Lambda`) as :math:`\Lambda` changes.

    The Sherman-Morrison formula states:

    .. math::
        (A + uv^T)^{-1} = A^{-1} - \frac{A^{-1}u v^T A^{-1}}{1 + v^T A^{-1} u}

    where :math:`A` is the current inverse, and :math:`uv^T` represents a rank-one update.

    Attributes
    ----------
    n : int
        Number of samples.
    p : int
        Number of features.
    X : np.ndarray
        The design matrix.
    XtX : np.ndarray
        The regularized :math:`X^T X` matrix.
    A : np.ndarray
        The matrix (:math:`I + X^T X`).
    A_inv : np.ndarray
        The inverse of matrix :math:`A`.
    U : np.ndarray
        Matrix :math:`U` used for efficiency in updates.
    V : np.ndarray
        Matrix :math:`V` used for efficiency in updates.
    """

    def __init__(self, X: np.ndarray):
        """Initialize the Sherman-Morrison Ridge predictor.

        Parameters
        ----------
        X : np.ndarray
            The design matrix.

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

        self.n, self.p = X.shape
        self.X = X
        self.XtX = self.X.T @ self.X / self.n
        self.A = np.eye(self.p) + self.XtX  # A = I + XtX
        self.A_inv = np.linalg.inv(self.A)  # Initialize A_inv as inv(A)
        self.U = self.X.T / np.sqrt(self.n)  # Precompute for efficiency
        self.V = self.X / np.sqrt(self.n)

    def update_lambda_s(self, groups: GroupedFeatures, lambdas: np.ndarray):
        r"""Update the regularization parameters (:math:`\lambda`) and adjust A_inv accordingly.

        Parameters
        ----------
        groups : GroupedFeatures
            The grouped features object.
        lambdas : np.ndarray
            The new :math:`\lambda` values for each group.

        Raises
        ------
        ValueError
            If lambdas contain negative values.

        Returns
        -------
        None
        """
        if np.any(lambdas < 0):
            raise ValueError("Lambda values must be non-negative.")

        diag = groups.group_expand(lambdas)
        self.A = np.diag(diag) + self.XtX  # Update A with new Lambda
        try:
            self.A_inv = np.linalg.inv(self.A)  # Recompute A_inv
        except np.linalg.LinAlgError:
            raise SingularMatrixError("Failed to invert A. Matrix may be singular.")

    def sherman_morrison(self, u: np.ndarray, v: np.ndarray):
        """Apply the Sherman-Morrison formula to update A_inv with a rank-one update.

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
        Au = self.A_inv @ u
        vA = v @ self.A_inv
        denominator = 1.0 + v @ Au
        if abs(denominator) < 1e-10:
            raise NumericalInstabilityError(
                "Denominator in Sherman-Morrison update is close to zero."
            )
        self.A_inv -= np.outer(Au, vA) / denominator

    def trace_XtX(self) -> float:
        r"""Compute the trace of :math:`X^T X`.

        Returns
        -------
        float
            The trace of :math:`X^T X`.
        """
        return np.trace(self.XtX)

    def XtXp_lambda_ldiv_XtX(self) -> np.ndarray:
        r"""Compute :math:`(X^T X + \Lambda)^{-1} X^T X`.

        Returns
        -------
        np.ndarray
            The result of the computation.
        """
        return self.ldiv(self.XtX)

    def ldiv(self, B: np.ndarray) -> np.ndarray:
        r"""Solve the system :math:`(X^T X + \Lambda)^{-1} B` using the precomputed A_inv.

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
            return self.A_inv @ B
        else:
            return self.A_inv @ B

    @staticmethod
    def sherman_morrison_formula(
        A: np.ndarray, u: np.ndarray, v: np.ndarray
    ) -> np.ndarray:
        """Apply the Sherman-Morrison formula to matrix :math:`A` with vectors :math:`u` and :math:`v`.

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


class BasicGroupRidgeWorkspace:
    r"""Workspace for performing group Ridge regression, including fitting and prediction.

    This class manages the entire workflow of group-wise Ridge regression, handling
    the fitting process, parameter updates, predictions, and evaluation metrics. It
    leverages Ridge predictors (`CholeskyRidgePredictor`, `WoodburyRidgePredictor`,
    `ShermanMorrisonRidgePredictor`) based on the dimensionality of the data to ensure
    computational efficiency.

    Attributes
    ----------
    X : np.ndarray
        The design matrix.
    Y : np.ndarray
        The target vector.
    groups : GroupedFeatures
        The grouped features object.
    n : int
        Number of samples.
    p : int
        Number of features.
    predictor : AbstractRidgePredictor
        The Ridge predictor being used.
    XtY : np.ndarray
        The :math:`X^T Y` matrix scaled by the number of samples.
    lambdas : np.ndarray
        Regularization parameters for each group.
    beta_current : np.ndarray
        Current coefficient estimates.
    Y_hat : np.ndarray
        Predicted values.
    XtXp_lambda_inv : np.ndarray
        Inverse of (:math:`X^T X + \Lambda`).
    moment_setup : MomentTunerSetup
        Moment setup for parameter tuning.
    leverage_store : np.ndarray
        Leverage scores for each sample.
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, groups: GroupedFeatures):
        """Initialize the BasicGroupRidgeWorkspace.

        Parameters
        ----------
        X : np.ndarray
            The design matrix.
        Y : np.ndarray
            The target vector.
        groups : GroupedFeatures
            The grouped features object.

        Raises
        ------
        ValueError
            If `groups` does not contain any groups or if any group has zero size.
        """
        if not groups.ps:
            raise ValueError("GroupedFeatures must contain at least one group.")
        if any(p == 0 for p in groups.ps):
            raise ValueError("GroupedFeatures groups must have non-zero sizes.")
        self.X = X
        self.Y = Y
        self.groups = groups
        self.n, self.p = X.shape

        # Initialize predictor based on p and n
        if self.p <= self.n:
            self.predictor = CholeskyRidgePredictor(X)
        elif self.p > self.n and self.p < 4 * self.n:
            self.predictor = WoodburyRidgePredictor(X)
        else:
            self.predictor = ShermanMorrisonRidgePredictor(X)

        self.XtY = np.dot(X.T, Y) / self.n
        self.lambdas = np.ones(groups.num_groups)
        self.update_lambda_s(self.lambdas)
        self.beta_current = self.predictor.ldiv(self.XtY)
        self.Y_hat = np.dot(X, self.beta_current)

        # Compute (:math:`X^T X + \Lambda`)^{-1}
        self.XtXp_lambda_inv = self.predictor.ldiv(np.eye(self.p))

        self.moment_setup = MomentTunerSetup(self)

    def update_lambda_s(self, lambdas: np.ndarray):
        r"""Update the regularization parameters (:math:`\lambda`).

        Parameters
        ----------
        lambdas : np.ndarray
            New :math:`\lambda` values for each group.

        Returns
        -------
        None
        """
        self.lambdas = lambdas
        self.predictor.update_lambda_s(self.groups, self.lambdas)
        self.XtXp_lambda_div_Xt = self.predictor.ldiv(self.X.T)

    def ngroups(self) -> int:
        """Get the number of feature groups.

        Returns
        -------
        int
            Number of groups.
        """
        return self.groups.ngroups()

    def coef(self) -> np.ndarray:
        """Get the current coefficient estimates (:math:`\beta`).

        Returns
        -------
        np.ndarray
            Coefficient vector :math:`\beta`.
        """
        return self.beta_current

    def islinear(self) -> bool:
        """Check if the model is linear.

        Returns
        -------
        bool
            Always returns True for Ridge regression.
        """
        return True

    def leverage(self) -> np.ndarray:
        """Get the leverage scores (:math:`h_i`).

        Returns
        -------
        np.ndarray
            Leverage scores vector.
        """
        return self.leverage_store

    def modelmatrix(self) -> np.ndarray:
        """Get the design matrix (:math:`X`).

        Returns
        -------
        np.ndarray
            Design matrix :math:`X`.
        """
        return self.X

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """Make predictions for new data.

        Parameters
        ----------
        X_new : np.ndarray
            New design matrix.

        Returns
        -------
        np.ndarray
            Predicted values.
        """
        return np.dot(X_new, self.coef())

    def response(self) -> np.ndarray:
        """Get the response variable (:math:`Y`).

        Returns
        -------
        np.ndarray
            Response vector :math:`Y`.
        """
        return self.Y

    def loo_error(self) -> float:
        """Compute the leave-one-out (LOO) error.

        Returns
        -------
        float
            The computed LOO error.
        """
        return np.mean((self.Y - self.Y_hat) ** 2 / (1.0 - self.leverage_store) ** 2)

    def mse_ridge(self, X_test: np.ndarray, Y_test: np.ndarray) -> float:
        """Compute the mean squared error (MSE) on test data.

        Parameters
        ----------
        X_test : np.ndarray
            Test design matrix.
        Y_test : np.ndarray
            Test response vector.

        Returns
        -------
        float
            The computed MSE.
        """
        return np.mean((Y_test - np.dot(X_test, self.beta_current)) ** 2)

    def fit(self, lambdas: Union[np.ndarray, dict]):
        r"""Fit the Ridge regression model with given regularization parameters.

        Parameters
        ----------
        lambdas : Union[np.ndarray, dict]
            The regularization parameters. Can be a numpy array or a dictionary
            mapping group names to :math:`\lambda` values.

        Returns
        -------
        float
            The leave-one-out (LOO) error after fitting.

        Raises
        ------
        ValueError
            If the provided lambdas are invalid.
        """
        if isinstance(lambdas, dict):
            lambdas = np.array(list(lambdas.values()))
        self.lambdas = lambdas
        self.predictor.update_lambda_s(self.groups, self.lambdas)
        self.beta_current = self.predictor.ldiv(self.XtY)
        self.Y_hat = np.dot(self.X, self.beta_current)
        self.XtXp_lambda_div_Xt = self.predictor.ldiv(self.X.T).T / self.n
        self.leverage_store = np.sum(self.X * self.XtXp_lambda_div_Xt, axis=1)
        return self.loo_error()


def lambda_lolas_rule(rdg: BasicGroupRidgeWorkspace, multiplier: float = 0.1) -> float:
    r"""Compute the regularization parameter :math:`\lambda` using the Panagiotis Lolas rule.

    The Lolas rule provides a heuristic for selecting the regularization parameter
    based on the model's degrees of freedom and the trace of :math:`X^T X`. This method
    balances the complexity of the model against its fit to the training data.

    The :math:`\lambda` is computed as:

    .. math::
        \lambda = \text{multiplier} \times \frac{p^2}{n \times \text{trace}(X^T X)}

    where:
    - multiplier : float, default=0.1
        A scalar multiplier for the rule.
    - :math:`p` : int
        Number of features.
    - :math:`n` : int
        Number of samples.
    - :math:`\text{trace}(X^T X)` : float
        The trace of the covariance matrix :math:`X^T X`.

    This formula scales the regularization parameter based on both the
    dimensionality of the data and the trace of the covariance matrix, ensuring
    that :math:`\lambda` is appropriately tuned for the given dataset.

    Parameters
    ----------
    rdg : BasicGroupRidgeWorkspace
        The Ridge regression workspace containing model parameters.
    multiplier : float, default=0.1
        A scalar multiplier for the rule.

    Returns
    -------
    float
        The computed :math:`\lambda` value.

    Raises
    ------
    ValueError
        If multiplier is not positive or if :math:`\text{trace}(X^T X)` is zero.
    """
    if multiplier <= 0:
        raise ValueError("Multiplier must be positive.")

    trace_XtX = rdg.predictor.trace_XtX()
    if trace_XtX == 0:
        raise ValueError("Trace of X^T X is zero, leading to division by zero.")

    return multiplier * rdg.p**2 / rdg.n / trace_XtX


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

    def __init__(self, rdg: BasicGroupRidgeWorkspace):
        """
        Initialize the MomentTunerSetup.

        Parameters
        ----------
        rdg : BasicGroupRidgeWorkspace
            An instance of BasicGroupRidgeWorkspace containing the model's current
            state.

        Raises
        ------
        ValueError
            If the length of N_matrix does not match the number of features.
        """
        self.groups = rdg.groups
        self.ps = np.array(rdg.groups.ps)
        self.n = rdg.n
        self.beta_norms_squared = np.array(
            rdg.groups.group_summary(rdg.beta_current, lambda x: np.sum(np.abs(x) ** 2))
        )
        N_matrix = rdg.XtXp_lambda_inv  # Use the (p, p) inverse matrix
        if N_matrix.shape[1] != self.ps.sum():
            raise ValueError(
                f"Length of N_matrix ({N_matrix.shape[1]}) does not match number of "
                f"features ({self.ps.sum()})"
            )
        self.N_norms_squared = np.array(
            rdg.groups.group_summary(N_matrix, lambda x: np.sum(np.abs(x) ** 2))
        )
        self.M_squared = np.outer(self.ps, self.ps) / self.n**2


def sigma_squared_path(
    rdg: BasicGroupRidgeWorkspace, mom: MomentTunerSetup, sigma_s_squared: np.ndarray
):
    r"""Compute the regularization path for different values of :math:`\sigma^2`.

    This function evaluates how the Ridge regression coefficients and leave-one-out
    (LOO) errors change as :math:`\sigma^2` varies. By analyzing the regularization path,
    one can understand the impact of different levels of regularization on the
    model's performance.

    For each :math:`\sigma^2` in the provided array:

    1. Compute :math:`\lambda`:
        :math:`\lambda = \text{get_lambdas}(mom, \sigma^2)`

    2. Fit Ridge Model:
        Fit the Ridge regression model using the computed :math:`\lambda` and evaluate:
            LOO Error = rdg.fit(:math:`\lambda`)

    3. Store Coefficients:
        :math:`\beta = rdg.beta_current`

    4. Aggregate Results:
        Collect the :math:`\lambda` values, LOO errors, and :math:`\beta` coefficients for each :math:`\sigma^2`.

    Parameters
    ----------
    rdg : BasicGroupRidgeWorkspace
        The Ridge regression workspace.
    mom : MomentTunerSetup
        The moment tuner setup containing necessary statistics.
    sigma_s_squared : np.ndarray
        An array of :math:`\sigma^2` values to evaluate.

    Returns
    -------
    dict
        A dictionary containing the regularization path information:
            - 'lambdas' : np.ndarray
                Array of :math:`\lambda` values for each :math:`\sigma^2`.
            - 'loos' : np.ndarray
                Array of LOO errors corresponding to each :math:`\sigma^2`.
            - 'betas' : np.ndarray
                Array of coefficient vectors corresponding to each :math:`\sigma^2`.

    Raises
    ------
    ValueError
        If sigma_s_squared contains negative values.
    """
    if np.any(sigma_s_squared < 0):
        raise ValueError("sigma_s_squared values must be non-negative.")

    n_sigma = len(sigma_s_squared)
    n_groups = rdg.ngroups()
    loos_hat = np.zeros(n_sigma)
    lambdas = np.zeros((n_sigma, n_groups))
    betas = np.zeros((n_sigma, rdg.groups.p))

    for i, sigma_sq in enumerate(sigma_s_squared):
        try:
            lambdas_tmp = get_lambdas(mom, sigma_sq)
            lambdas[i, :] = lambdas_tmp
            loos_hat[i] = rdg.fit(lambdas_tmp)
            betas[i, :] = rdg.beta_current
        except RidgeRegressionError as e:
            print(f"Error at :math:`\sigma^2 = {sigma_sq}`: {str(e)}")

    return {"lambdas": lambdas, "loos": loos_hat, "betas": betas}


def get_lambdas(mom: MomentTunerSetup, sigma_sq: float) -> np.ndarray:
    r"""Compute :math:`\lambda` values for a given :math:`\sigma^2`.

    This function calculates the regularization parameters (:math:`\lambda`) for each
    feature group based on moment-based statistics. The computed :math:`\lambda` balances
    the regularization strength across different groups, ensuring that groups with
    more features or higher variance receive appropriate penalization.

    The steps are as follows:

    1. Compute :math:`\alpha^2` for each group:
        :math:`\alpha_g^2 = \max(\|\beta_g\|^2 - \sigma^2 \|N_g\|^2, 0) / p_g`

    2. Compute :math:`\gamma_s` for each group:
        :math:`\gamma_g = p_g / n`

    3. Compute :math:`\lambda` for each group:
        :math:`\lambda_g = (\sigma^2 \gamma_g) / \alpha_g^2`

    Parameters
    ----------
    mom : MomentTunerSetup
        The moment tuner setup containing necessary statistics.
    sigma_sq : float
        The :math:`\sigma^2` value for which to compute :math:`\lambda`.

    Returns
    -------
    np.ndarray
        The computed :math:`\lambda` values for each feature group.

    Raises
    ------
    ValueError
        If sigma_sq is negative.
    NumericalInstabilityError
        If division by zero occurs during lambda computation.
    """
    if sigma_sq < 0:
        raise ValueError("sigma_sq must be non-negative.")

    alpha_sq = get_alpha_s_squared(mom, sigma_sq)
    gamma_s = np.array(mom.ps) / mom.n

    LARGE_VALUE = 1e12

    with np.errstate(divide="ignore", invalid="ignore"):
        lambdas = sigma_sq * gamma_s / alpha_sq
        zero_alpha = alpha_sq == 0
        if np.any(zero_alpha):
            warnings.warn(
                f"alpha_sq has zero values for groups: {np.where(zero_alpha)[0]}. "
                "Assigning large lambda values to these groups."
            )
        lambdas = np.where(zero_alpha, LARGE_VALUE, lambdas)

    return lambdas


def get_alpha_s_squared(mom: MomentTunerSetup, sigma_sq: float) -> np.ndarray:
    r"""Compute :math:`\alpha_s^2` values for a given :math:`\sigma^2` using Non-Negative Least Squares
    (NNLS).

    This function calculates the :math:`\alpha_s^2` values required for determining the
    regularization parameters (:math:`\lambda`) in Ridge regression. The :math:`\alpha_s^2` values
    encapsulate the balance between the coefficient norms and the influence of
    the design matrix, adjusted by :math:`\sigma^2`.

    The steps are as follows:

    1. Compute the right-hand side (RHS):
        :math:`\text{RHS}_g = \|\beta_g\|^2 - \sigma^2 \|N_g\|^2`

    2. Solve the NNLS problem:
        :math:`\min \| M_{\text{squared}} \times \alpha_{\text{sq\_by\_p}} - \text{RHS} \|_2^2`
        subject to :math:`\alpha_{\text{sq\_by\_p}} \geq 0`

    3. Compute :math:`\alpha_g^2`:
        :math:`\alpha_g^2 = \alpha_{\text{sq\_by\_p}} \times p_g`

    Parameters
    ----------
    mom : MomentTunerSetup
        The moment tuner setup containing necessary statistics.
    sigma_sq : float
        The :math:`\sigma^2` value for which to compute :math:`\alpha_s^2`.

    Returns
    -------
    np.ndarray
        The computed :math:`\alpha_s^2` values for each feature group.

    Raises
    ------
    ValueError
        If sigma_sq is negative or if any :math:`p_s` is zero.
    NNLSError
        If the NNLS algorithm fails to converge.
    """
    if sigma_sq < 0:
        raise ValueError("sigma_sq must be non-negative.")
    if np.any(mom.ps == 0):
        raise ValueError("All p_s values must be non-zero.")

    # Compute the right-hand side
    rhs = mom.beta_norms_squared - sigma_sq * mom.N_norms_squared
    rhs = np.maximum(rhs, 0)

    # Solve the NNLS problem: M_squared * alpha_sq_by_p ≈ rhs
    try:
        alpha_sq_by_p = nonneg_lsq(mom.M_squared, rhs, alg="fnnls")
    except NNLSError as e:
        raise NNLSError(f"Failed to compute alpha_s_squared: {str(e)}")

    alpha_s_squared = alpha_sq_by_p * mom.groups.ps

    return alpha_s_squared
