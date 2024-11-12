"""Covariance matrix designs for various statistical models."""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Callable, Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from .groupedfeatures import GroupedFeatures, fill


class DiscreteNonParametric:
    """Represents a discrete non-parametric spectrum.

    Parameters
    ----------
    eigs : list of float
        A list of eigenvalues.
    probs : list of float
        A list of probabilities corresponding to each eigenvalue.

    Attributes
    ----------
    eigs : ndarray of shape (n_eigenvalues,)
        Numpy array of eigenvalues.
    probs : ndarray of shape (n_eigenvalues,)
        Numpy array of probabilities for each eigenvalue.

    Examples
    --------
    >>> from PyGRidge.src.covariance_design import DiscreteNonParametric
    >>> spectrum = DiscreteNonParametric(eigs=[1.0, 2.0, 3.0], probs=[0.2, 0.3, 0.5])
    >>> spectrum.eigs
    array([1., 2., 3.])
    >>> spectrum.probs
    array([0.2, 0.3, 0.5])
    """

    def __init__(self, eigs: List[float], probs: List[float]):
        if not isinstance(eigs, list):
            raise TypeError(
                f"'eigs' must be a list of floats, got {type(eigs).__name__}"
            )
        if not all(isinstance(e, (int, float)) for e in eigs):
            raise ValueError("All elements in 'eigs' must be integers or floats.")
        if not isinstance(probs, list):
            raise TypeError(
                f"'probs' must be a list of floats, got {type(probs).__name__}"
            )
        if not all(isinstance(p, (int, float)) for p in probs):
            raise ValueError("All elements in 'probs' must be integers or floats.")
        if len(eigs) != len(probs):
            raise ValueError("'eigs' and 'probs' must be of the same length.")
        if not np.isclose(sum(probs), 1.0):
            raise ValueError("The probabilities in 'probs' must sum to 1.")
        if any(p < 0 for p in probs):
            raise ValueError("Probabilities in 'probs' must be non-negative.")

        self.eigs = np.array(eigs)
        self.probs = np.array(probs)


class CovarianceDesign(BaseEstimator, TransformerMixin):
    """Base class for covariance matrix designs.

    This class defines the interface for all covariance designs and implements
    the scikit-learn estimator interface.

    Parameters
    ----------
    store_precision : bool, default=True
        Whether to store the precision matrix.

    Attributes
    ----------
    covariance_ : ndarray of shape (n_features, n_features)
        Estimated covariance matrix.
    precision_ : ndarray of shape (n_features, n_features)
        Estimated precision matrix (inverse of covariance).
        Only stored if store_precision is True.
    n_features_in_ : int
        Number of features seen during fit.

    See Also
    --------
    AR1Design : AutoRegressive model of order 1.
    IdentityCovarianceDesign : Identity covariance matrix.
    UniformScalingCovarianceDesign : Uniform scaling on diagonal.
    ExponentialOrderStatsCovarianceDesign : Exponential order statistics.
    BlockCovarianceDesign : Block diagonal covariance matrix.
    """

    _parameter_constraints = {
        "store_precision": ["boolean"],
    }

    def __init__(self, *, store_precision=True):
        self.store_precision = store_precision

    @abstractmethod
    def get_Sigma(self) -> np.ndarray:
        """Construct and return the covariance matrix.

        Returns
        -------
        Sigma : ndarray of shape (n_features, n_features)
            The covariance matrix.
        """
        pass

    @abstractmethod
    def nfeatures(self) -> int:
        """Get the number of features.

        Returns
        -------
        n_features : int
            Number of features.
        """
        pass

    @abstractmethod
    def spectrum(self) -> DiscreteNonParametric:
        """Compute the spectral decomposition.

        Returns
        -------
        spectrum : DiscreteNonParametric
            Spectrum containing eigenvalues and probabilities.
        """
        pass

    def get_precision(self):
        """Get the precision matrix.

        Returns
        -------
        precision : ndarray of shape (n_features, n_features)
            The precision matrix.
        """
        check_is_fitted(self, ["covariance_"])
        if self.store_precision and hasattr(self, "precision_"):
            return self.precision_
        else:
            return np.linalg.pinv(self.covariance_)

    def fit(self, X=None, y=None):
        """Fit the covariance design.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default=None
            Training data. Not used, present for API consistency.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if X is not None:
            X = check_array(X)
            self.n_features_in_ = X.shape[1]

        self.covariance_ = self.get_Sigma()
        if self.store_precision:
            self.precision_ = np.linalg.pinv(self.covariance_)
        return self

    def transform(self, X):
        """Transform data using the covariance design.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            The transformed data.
        """
        check_is_fitted(self, ["covariance_"])
        X = check_array(X)
        return X @ self.covariance_

    def score(self, X, y=None):
        """Compute the log-likelihood of X under the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        score : float
            Log-likelihood of X under the model.
        """
        check_is_fitted(self, ["covariance_"])
        X = check_array(X)
        precision = self.get_precision()
        n_samples, n_features = X.shape
        log_likelihood = -np.sum(X @ precision @ X.T)
        log_likelihood += n_samples * np.log(np.linalg.det(precision))
        log_likelihood -= n_samples * n_features * np.log(2 * np.pi)
        log_likelihood /= 2.0
        return log_likelihood


class AR1Design(CovarianceDesign):
    r"""Implements an AutoRegressive model of order 1 (AR(1)) for covariance matrix design.

    Parameters
    ----------
    p : int, optional
        Number of features.
    rho : float, default=0.7
        AR(1) correlation coefficient. Must be in [0, 1).
    store_precision : bool, default=True
        Whether to store the precision matrix.

    Attributes
    ----------
    covariance_ : ndarray of shape (n_features, n_features)
        AR(1) covariance matrix.
    precision_ : ndarray of shape (n_features, n_features)
        Precision matrix (inverse of covariance).
    n_features_in_ : int
        Number of features.

    Examples
    --------
    >>> from PyGRidge.src.covariance_design import AR1Design
    >>> design = AR1Design(p=3, rho=0.5)
    >>> design.fit()
    >>> design.covariance_
    array([[1.  , 0.5 , 0.25],
           [0.5 , 1.  , 0.5 ],
           [0.25, 0.5 , 1.  ]])
    """

    _parameter_constraints = {
        **CovarianceDesign._parameter_constraints,
        "p": [int, None],
        "rho": [float],
    }

    def __init__(self, p=None, rho=0.7, *, store_precision=True):
        super().__init__(store_precision=store_precision)
        self.p = p
        self.rho = rho

        if p is not None:
            if not isinstance(p, int):
                raise TypeError(f"'p' must be an integer, got {type(p).__name__}")
            if p <= 0:
                raise ValueError("'p' must be a positive integer.")
        if not isinstance(rho, (int, float)):
            raise TypeError(f"'rho' must be a float, got {type(rho).__name__}")
        if not (0 <= rho < 1):
            raise ValueError("'rho' must be in the interval [0, 1).")

    def get_Sigma(self):
        """Construct the AR(1) covariance matrix.

        Returns
        -------
        Sigma : ndarray of shape (n_features, n_features)
            The AR(1) covariance matrix.
        """
        if self.p is None:
            raise ValueError("Number of features 'p' is not set.")
        if self.p <= 0:
            raise ValueError("'p' must be a positive integer.")

        indices = np.arange(self.p)
        Sigma = self.rho ** np.abs(np.subtract.outer(indices, indices))
        return Sigma

    def nfeatures(self):
        """Get the number of features.

        Returns
        -------
        n_features : int
            Number of features.
        """
        if self.p is None:
            raise ValueError("Number of features 'p' is not set.")
        return self.p

    def spectrum(self):
        """Compute the spectral decomposition.

        Returns
        -------
        spectrum : DiscreteNonParametric
            Spectrum containing eigenvalues and probabilities.
        """
        Sigma = self.get_Sigma()
        eigs = np.linalg.eigvals(Sigma)
        probs = [1.0 / len(eigs)] * len(eigs)
        return DiscreteNonParametric(eigs.tolist(), probs)


class DiagonalCovarianceDesign(CovarianceDesign):
    """Base class for diagonal covariance matrix designs.

    Parameters
    ----------
    p : int, optional
        Number of features.
    store_precision : bool, default=True
        Whether to store the precision matrix.

    Attributes
    ----------
    covariance_ : ndarray of shape (n_features, n_features)
        Diagonal covariance matrix.
    precision_ : ndarray of shape (n_features, n_features)
        Precision matrix (inverse of covariance).
    n_features_in_ : int
        Number of features.
    """

    _parameter_constraints = {
        **CovarianceDesign._parameter_constraints,
        "p": [int, None],
    }

    def __init__(self, p=None, *, store_precision=True):
        super().__init__(store_precision=store_precision)
        self.p = p

        if p is not None:
            if not isinstance(p, int):
                raise TypeError(f"'p' must be an integer, got {type(p).__name__}")
            if p <= 0:
                raise ValueError("'p' must be a positive integer.")

    def nfeatures(self):
        """Get the number of features.

        Returns
        -------
        n_features : int
            Number of features.
        """
        if self.p is None:
            raise ValueError("Number of features 'p' is not set.")
        return self.p


class IdentityCovarianceDesign(DiagonalCovarianceDesign):
    """Identity covariance matrix design.

    Parameters
    ----------
    p : int, optional
        Number of features.
    store_precision : bool, default=True
        Whether to store the precision matrix.

    Attributes
    ----------
    covariance_ : ndarray of shape (n_features, n_features)
        Identity matrix.
    precision_ : ndarray of shape (n_features, n_features)
        Identity matrix (same as covariance).
    n_features_in_ : int
        Number of features.

    Examples
    --------
    >>> from PyGRidge.src.covariance_design import IdentityCovarianceDesign
    >>> design = IdentityCovarianceDesign(p=3)
    >>> design.fit()
    >>> design.covariance_
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])
    """

    def get_Sigma(self):
        """Construct the identity covariance matrix.

        Returns
        -------
        Sigma : ndarray of shape (n_features, n_features)
            The identity matrix.
        """
        if self.p is None:
            raise ValueError("Number of features 'p' is not set.")
        return np.identity(self.p)

    def spectrum(self):
        """Compute the spectral decomposition.

        Returns
        -------
        spectrum : DiscreteNonParametric
            Spectrum containing eigenvalues (all 1) and equal probabilities.
        """
        if self.p is None:
            raise ValueError("Number of features 'p' is not set.")
        eigs = [1.0] * self.p
        probs = [1.0 / self.p] * self.p
        return DiscreteNonParametric(eigs, probs)


class UniformScalingCovarianceDesign(DiagonalCovarianceDesign):
    """Uniform scaling covariance matrix design.

    Parameters
    ----------
    scaling : float, default=1.0
        Scaling factor for diagonal entries.
    p : int, optional
        Number of features.
    store_precision : bool, default=True
        Whether to store the precision matrix.

    Attributes
    ----------
    covariance_ : ndarray of shape (n_features, n_features)
        Scaled identity matrix.
    precision_ : ndarray of shape (n_features, n_features)
        Inverse of scaled identity matrix.
    n_features_in_ : int
        Number of features.

    Examples
    --------
    >>> from PyGRidge.src.covariance_design import UniformScalingCovarianceDesign
    >>> design = UniformScalingCovarianceDesign(scaling=2.0, p=3)
    >>> design.fit()
    >>> design.covariance_
    array([[2., 0., 0.],
           [0., 2., 0.],
           [0., 0., 2.]])
    """

    _parameter_constraints = {
        **DiagonalCovarianceDesign._parameter_constraints,
        "scaling": [float],
    }

    def __init__(self, scaling=1.0, p=None, *, store_precision=True):
        super().__init__(p=p, store_precision=store_precision)
        self.scaling = scaling

        if not isinstance(scaling, (int, float)):
            raise TypeError(f"'scaling' must be a float, got {type(scaling).__name__}")
        if scaling <= 0:
            raise ValueError("'scaling' must be a positive number.")

    def get_Sigma(self):
        """Construct the scaled identity covariance matrix.

        Returns
        -------
        Sigma : ndarray of shape (n_features, n_features)
            The scaled identity matrix.
        """
        if self.p is None:
            raise ValueError("Number of features 'p' is not set.")
        return self.scaling * np.identity(self.p)

    def spectrum(self):
        """Compute the spectral decomposition.

        Returns
        -------
        spectrum : DiscreteNonParametric
            Spectrum containing eigenvalues (all equal to scaling) and equal probabilities.
        """
        if self.p is None:
            raise ValueError("Number of features 'p' is not set.")
        eigs = [self.scaling] * self.p
        probs = [1.0 / self.p] * self.p
        return DiscreteNonParametric(eigs, probs)


class ExponentialOrderStatsCovarianceDesign(DiagonalCovarianceDesign):
    """Exponential order statistics covariance matrix design.

    Parameters
    ----------
    p : int, optional
        Number of features.
    rate : float, default=1.0
        Rate parameter for exponential distribution.
    store_precision : bool, default=True
        Whether to store the precision matrix.

    Attributes
    ----------
    covariance_ : ndarray of shape (n_features, n_features)
        Diagonal matrix with exponential order statistics.
    precision_ : ndarray of shape (n_features, n_features)
        Inverse of covariance matrix.
    n_features_in_ : int
        Number of features.

    Examples
    --------
    >>> from PyGRidge.src.covariance_design import ExponentialOrderStatsCovarianceDesign
    >>> design = ExponentialOrderStatsCovarianceDesign(p=3, rate=1.0)
    >>> design.fit()
    >>> design.covariance_  # doctest: +SKIP
    array([[2.19..., 0.    , 0.    ],
           [0.    , 1.09..., 0.    ],
           [0.    , 0.    , 0.40...]])
    """

    _parameter_constraints = {
        **DiagonalCovarianceDesign._parameter_constraints,
        "rate": [float],
    }

    def __init__(self, p=None, rate=1.0, *, store_precision=True):
        super().__init__(p=p, store_precision=store_precision)
        self.rate = rate

        if not isinstance(rate, (int, float)):
            raise TypeError(f"'rate' must be a float, got {type(rate).__name__}")
        if rate <= 0:
            raise ValueError("'rate' must be a positive number.")

    def spectrum(self):
        """Compute the spectral decomposition.

        Returns
        -------
        spectrum : DiscreteNonParametric
            Spectrum containing exponential order statistics eigenvalues and equal probabilities.
        """
        if self.p is None:
            raise ValueError("Number of features 'p' is not set.")

        p = self.p
        rate = self.rate

        tmp = np.linspace(1 / (2 * p), 1 - 1 / (2 * p), p)
        eigs = (1 / rate) * np.log(1 / tmp)
        probs = [1.0 / p] * p

        return DiscreteNonParametric(eigs.tolist(), probs)

    def get_Sigma(self):
        """Construct the diagonal covariance matrix.

        Returns
        -------
        Sigma : ndarray of shape (n_features, n_features)
            Diagonal matrix with exponential order statistics.
        """
        spectrum = self.spectrum()
        return np.diag(spectrum.eigs)


class BlockDiagonal:
    """Block diagonal matrix.

    Parameters
    ----------
    blocks : list of ndarray
        List of square matrices to place on diagonal.

    Attributes
    ----------
    blocks : list of ndarray
        List of block matrices.

    Examples
    --------
    >>> import numpy as np
    >>> from PyGRidge.src.covariance_design import BlockDiagonal
    >>> A = np.array([[1, 0], [0, 1]])
    >>> B = np.array([[2, 0], [0, 2]])
    >>> block_diag = BlockDiagonal([A, B])
    >>> block_diag.get_Sigma()
    array([[1, 0, 0, 0],
           [0, 1, 0, 0],
           [0, 0, 2, 0],
           [0, 0, 0, 2]])
    """

    def __init__(self, blocks):
        if not isinstance(blocks, list):
            raise TypeError(
                f"'blocks' must be a list of numpy arrays, got {type(blocks).__name__}"
            )
        if not all(isinstance(block, np.ndarray) for block in blocks):
            raise TypeError("All blocks must be numpy.ndarray instances.")
        if not all(block.ndim == 2 for block in blocks):
            raise ValueError("All blocks must be 2-dimensional arrays.")
        if not all(block.shape[0] == block.shape[1] for block in blocks):
            raise ValueError("All blocks must be square matrices.")

        self.blocks = blocks

    def get_Sigma(self):
        """Construct the block diagonal matrix.

        Returns
        -------
        Sigma : ndarray
            The block diagonal matrix.
        """
        return block_diag(*self.blocks)


def block_diag(*arrs):
    """Construct a block diagonal matrix.

    Parameters
    ----------
    *arrs : ndarray
        Square arrays to place on diagonal.

    Returns
    -------
    out : ndarray
        The block diagonal matrix.

    Examples
    --------
    >>> import numpy as np
    >>> from PyGRidge.src.covariance_design import block_diag
    >>> A = np.array([[1, 0], [0, 1]])
    >>> B = np.array([[2, 0], [0, 2]])
    >>> block_diag(A, B)
    array([[1, 0, 0, 0],
           [0, 1, 0, 0],
           [0, 0, 2, 0],
           [0, 0, 0, 2]])
    """
    if not all(isinstance(a, np.ndarray) for a in arrs):
        raise TypeError("All arguments must be numpy.ndarray instances.")
    if not all(a.ndim == 2 for a in arrs):
        raise ValueError("All input arrays must be 2-dimensional.")
    if not all(a.shape[0] == a.shape[1] for a in arrs):
        raise ValueError("All input arrays must be square matrices.")

    if len(arrs) == 0:
        return np.array([[]])

    shapes = np.array([a.shape for a in arrs])
    out_shape = np.sum(shapes, axis=0)
    out = np.zeros(out_shape, dtype=arrs[0].dtype)
    r, c = 0, 0
    for a in arrs:
        out[r : r + a.shape[0], c : c + a.shape[1]] = a
        r += a.shape[0]
        c += a.shape[1]
    return out


class MixtureModel:
    """Mixture model of multiple spectra.

    Parameters
    ----------
    spectra : list of DiscreteNonParametric
        List of component spectra.
    mixing_prop : list of float
        Mixing proportions for each spectrum.

    Attributes
    ----------
    spectra : list of DiscreteNonParametric
        Component spectra.
    mixing_prop : list of float
        Mixing proportions.

    Examples
    --------
    >>> from PyGRidge.src.covariance_design import DiscreteNonParametric, MixtureModel
    >>> s1 = DiscreteNonParametric([1, 2], [0.5, 0.5])
    >>> s2 = DiscreteNonParametric([3, 4], [0.3, 0.7])
    >>> mix = MixtureModel([s1, s2], [0.6, 0.4])
    """

    def __init__(self, spectra, mixing_prop):
        if not isinstance(spectra, list):
            raise TypeError(
                "'spectra' must be a list of DiscreteNonParametric instances"
            )
        if not all(isinstance(s, DiscreteNonParametric) for s in spectra):
            raise TypeError(
                "All elements in 'spectra' must be DiscreteNonParametric instances"
            )
        if not isinstance(mixing_prop, list):
            raise TypeError("'mixing_prop' must be a list of floats")
        if not all(isinstance(p, (int, float)) for p in mixing_prop):
            raise ValueError("All elements in 'mixing_prop' must be numbers")
        if len(spectra) != len(mixing_prop):
            raise ValueError("'spectra' and 'mixing_prop' must have same length")
        if not np.isclose(sum(mixing_prop), 1.0):
            raise ValueError("'mixing_prop' must sum to 1")
        if any(p < 0 for p in mixing_prop):
            raise ValueError("'mixing_prop' must be non-negative")

        self.spectra = spectra
        self.mixing_prop = mixing_prop


class BlockCovarianceDesign(CovarianceDesign):
    """Block diagonal covariance matrix design.

    Parameters
    ----------
    blocks : list of CovarianceDesign
        List of covariance designs for each block.
    groups : GroupedFeatures, optional
        Feature groupings.
    store_precision : bool, default=True
        Whether to store the precision matrix.

    Attributes
    ----------
    covariance_ : ndarray of shape (n_features, n_features)
        Block diagonal covariance matrix.
    precision_ : ndarray of shape (n_features, n_features)
        Precision matrix (inverse of covariance).
    n_features_in_ : int
        Total number of features.

    Examples
    --------
    >>> from PyGRidge.src.covariance_design import BlockCovarianceDesign
    >>> from PyGRidge.src.covariance_design import IdentityCovarianceDesign
    >>> block1 = IdentityCovarianceDesign(p=2)
    >>> block2 = IdentityCovarianceDesign(p=3)
    >>> design = BlockCovarianceDesign([block1, block2])
    >>> design.fit()
    >>> design.covariance_
    array([[1., 0., 0., 0., 0.],
           [0., 1., 0., 0., 0.],
           [0., 0., 1., 0., 0.],
           [0., 0., 0., 1., 0.],
           [0., 0., 0., 0., 1.]])
    """

    _parameter_constraints = {
        **CovarianceDesign._parameter_constraints,
        "blocks": [list],
        "groups": [GroupedFeatures, None],
    }

    def __init__(self, blocks, groups=None, *, store_precision=True):
        super().__init__(store_precision=store_precision)
        self.blocks = blocks
        self.groups = groups

        if not isinstance(blocks, list):
            raise TypeError("'blocks' must be a list of CovarianceDesign instances")
        if not all(isinstance(block, CovarianceDesign) for block in blocks):
            raise TypeError("All blocks must be CovarianceDesign instances")
        if groups is not None:
            if not isinstance(groups, GroupedFeatures):
                raise TypeError("'groups' must be a GroupedFeatures instance")
            if len(groups.ps) != len(blocks):
                raise ValueError("Number of groups must match number of blocks")

    def get_Sigma(self):
        """Construct the block diagonal covariance matrix.

        Returns
        -------
        Sigma : ndarray of shape (n_features, n_features)
            The block diagonal covariance matrix.
        """
        block_matrices = [block.get_Sigma() for block in self.blocks]
        return block_diag(*block_matrices)

    def nfeatures(self):
        """Get the total number of features.

        Returns
        -------
        n_features : int
            Total number of features across all blocks.
        """
        return sum(block.nfeatures() for block in self.blocks)

    def spectrum(self):
        """Compute the spectral decomposition.

        Returns
        -------
        spectrum : MixtureModel
            Mixture model of block spectra.
        """
        spectra = [block.spectrum() for block in self.blocks]

        if self.groups is not None:
            total_p = self.groups.p
            mixing_prop = [ps / total_p for ps in self.groups.ps]
        else:
            mixing_prop = [1.0 / len(spectra)] * len(spectra)

        return MixtureModel(spectra, mixing_prop)


def simulate_rotated_design(cov, n, rotated_measure=None):
    """Simulate a rotated design matrix.

    Parameters
    ----------
    cov : CovarianceDesign
        Covariance design to use.
    n : int
        Number of samples.
    rotated_measure : callable, optional
        Random number generator.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Simulated design matrix.

    Examples
    --------
    >>> from PyGRidge.src.covariance_design import AR1Design, simulate_rotated_design
    >>> design = AR1Design(p=3, rho=0.5)
    >>> X = simulate_rotated_design(design, n=100)
    """
    if not isinstance(cov, CovarianceDesign):
        raise TypeError("'cov' must be a CovarianceDesign instance")
    if not isinstance(n, int):
        raise TypeError("'n' must be an integer")
    if n <= 0:
        raise ValueError("'n' must be positive")
    if rotated_measure is not None and not callable(rotated_measure):
        raise TypeError("'rotated_measure' must be callable")

    # Check if the design is fitted
    check_is_fitted(cov, ["covariance_"])

    if rotated_measure is None:
        rotated_measure = np.random.normal

    Sigma = cov.get_Sigma()
    try:
        L = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        raise ValueError("Covariance matrix is not positive definite")

    p = cov.nfeatures()
    Z = rotated_measure(size=(n, p))
    return Z @ L


def set_groups(design, groups_or_p):
    """Set the number of features for a covariance design.

    Parameters
    ----------
    design : CovarianceDesign
        The covariance design to modify.
    groups_or_p : int or GroupedFeatures
        Number of features or feature groupings.
    """
    if not isinstance(design, CovarianceDesign):
        raise TypeError("'design' must be a CovarianceDesign instance")

    if isinstance(groups_or_p, int):
        if groups_or_p <= 0:
            raise ValueError("'groups_or_p' as an integer must be a positive value.")

        if isinstance(design, BlockCovarianceDesign):
            n_blocks = len(design.blocks)
            size_per_block = groups_or_p // n_blocks
            remainder = groups_or_p % n_blocks
            sizes = [
                size_per_block + (1 if i < remainder else 0) for i in range(n_blocks)
            ]
            groups = GroupedFeatures(ps=sizes)

            for block, size in zip(design.blocks, sizes):
                block.p = size
            design.groups = groups
        else:
            design.p = groups_or_p
        return

    if groups_or_p.__class__.__name__ == "GroupedFeatures":
        # Check if the GroupedFeatures instance is fitted
        if not hasattr(groups_or_p, "is_fitted_") or not groups_or_p.is_fitted_:
            raise NotFittedError("GroupedFeatures instance is not fitted")

        if isinstance(design, BlockCovarianceDesign):
            if len(groups_or_p.ps) != len(design.blocks):
                raise ValueError(
                    "Number of groups must match number of blocks in BlockCovarianceDesign."
                )
            for block, group_size in zip(design.blocks, groups_or_p.ps):
                block.p = group_size
            design.groups = groups_or_p
        else:
            design.p = groups_or_p.n_features_in_
        return

    raise TypeError("groups_or_p must be an instance of GroupedFeatures or int.")
