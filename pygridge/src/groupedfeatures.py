"""Create and manage grouped feature structures for statistical modeling."""

from typing import Callable, Union, TypeVar, List, Optional
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

T = TypeVar("T")


class GroupedFeatures(BaseEstimator, TransformerMixin):
    r"""A class representing groups of features following scikit-learn's API.

    The first :math:`ps[0]` features are one group, the next :math:`ps[1]` features are the
    second group, and so forth.

    Parameters
    ----------
    ps : list of int
        List of positive integers representing the size of each group.
    group_operation : callable, default=None
        Optional function to apply to each group during transformation.
        If None, features are kept as is.

    Attributes
    ----------
    ps_ : list of int
        List of group sizes.
    n_features_in_ : int
        Total number of features, calculated as :math:`p = \sum ps`.
    n_groups_ : int
        Number of groups, denoted as :math:`G`.
    feature_groups_ : list of range
        List of range objects representing the indices for each group.

    Raises
    ------
    TypeError
        If `ps` is not a list or contains non-integer elements.
    ValueError
        If any group size is not positive.
    """

    def __init__(self, ps: List[int], group_operation: Optional[Callable] = None):
        self.ps = ps
        self.group_operation = group_operation

    @property
    def num_groups(self) -> int:
        """Get the number of groups.

        Returns
        -------
        int
            Number of groups.
        """
        return len(self.ps)

    def fit(self, X, y=None):
        """Fit the GroupedFeatures transformer.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.
        y : None
            Ignored. This parameter exists only for compatibility with
            scikit-learn's transformer interface.

        Returns
        -------
        self : object
            Returns self.
        """
        # Validate input
        X = check_array(X, accept_sparse=True)

        if not isinstance(self.ps, list):
            raise TypeError(
                f"ps must be a list of positive integers, got {type(self.ps).__name__}"
            )
        if not all(isinstance(p, int) for p in self.ps):
            raise TypeError("All group sizes in ps must be integers")
        if not all(p > 0 for p in self.ps):
            raise ValueError("All group sizes in ps must be positive integers")

        # Store attributes
        self.ps_ = self.ps
        self.n_features_in_ = sum(self.ps_)
        self.n_groups_ = len(self.ps_)

        # Validate feature dimensions
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but GroupedFeatures is expecting "
                f"{self.n_features_in_} features as per the ps parameter."
            )

        # Precompute group indices
        starts = np.cumsum([0] + self.ps_[:-1]).astype(int)
        ends = np.cumsum(self.ps_).astype(int)
        self.feature_groups_ = [range(start, end) for start, end in zip(starts, ends)]

        return self

    def transform(self, X):
        """Transform the data by applying the group operation if specified.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input data to be transformed.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features) or (n_samples, n_groups)
            Transformed data. If group_operation is None, returns the original data.
            Otherwise, returns the result of applying group_operation to each group.
        """
        check_is_fitted(self)
        X = check_array(X, accept_sparse=True)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but GroupedFeatures is expecting "
                f"{self.n_features_in_} features as per the fit method."
            )

        if self.group_operation is None:
            return X

        # Apply group operation to each group
        transformed_groups = []
        for group_range in self.feature_groups_:
            group_data = X[:, group_range]
            try:
                transformed = self.group_operation(group_data)
                if isinstance(transformed, np.ndarray):
                    transformed = transformed.reshape(X.shape[0], -1)
                transformed_groups.append(transformed)
            except Exception as e:
                raise RuntimeError(
                    f"Error applying group_operation to group {group_range}: {e}"
                )

        return np.column_stack(transformed_groups)

    def get_feature_names_out(self, feature_names_in=None):
        """Get output feature names for transformation.

        Parameters
        ----------
        feature_names_in : array-like of str or None, default=None
            Input feature names. If None, returns generated names.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Output feature names.
        """
        check_is_fitted(self)

        if feature_names_in is None:
            feature_names_in = [f"feature{i}" for i in range(self.n_features_in_)]

        if len(feature_names_in) != self.n_features_in_:
            raise ValueError(
                f"Length of feature_names_in ({len(feature_names_in)}) does not match "
                f"number of features ({self.n_features_in_})"
            )

        if self.group_operation is None:
            return np.array(feature_names_in)

        # If group operation is specified, generate group-based feature names
        group_names = []
        for i, group_range in enumerate(self.feature_groups_):
            group_names.append(f"group{i}")
        return np.array(group_names)

    def group_idx(self, i: int) -> range:
        """Get the range of feature indices for the i-th group.

        Parameters
        ----------
        i : int
            Index of the group (0-based).

        Returns
        -------
        range
            Range object representing the indices of the group.

        Raises
        ------
        TypeError
            If `i` is not an integer.
        IndexError
            If `i` is out of range [0, n_groups_ - 1].
        """
        check_is_fitted(self)

        if not isinstance(i, int):
            raise TypeError(f"Group index i must be an integer, got {type(i).__name__}")
        if not (0 <= i < self.n_groups_):
            raise IndexError(
                f"Group index i={i} is out of range [0, {self.n_groups_ - 1}]"
            )

        return self.feature_groups_[i]

    def group_summary(
        self,
        vec: Union[List[T], np.ndarray],
        f: Callable[[Union[List[T], np.ndarray]], T],
    ) -> List[T]:
        """Apply a summary function to each group of features.

        Parameters
        ----------
        vec : array-like of shape (n_features,) or (n_samples, n_features)
            List or ndarray of features.
        f : callable
            Function that takes a list or ndarray of features and returns a
            summary value.

        Returns
        -------
        list
            List of summary values, one per group.
        """
        check_is_fitted(self)

        if not callable(f):
            raise TypeError("f must be a callable function")

        if isinstance(vec, np.ndarray):
            vec = check_array(vec, ensure_2d=False)
            if vec.ndim == 1:
                if vec.shape[0] != self.n_features_in_:
                    raise ValueError(
                        f"Length of vec ({vec.shape[0]}) does not match number of "
                        f"features ({self.n_features_in_})"
                    )
                vec = vec.reshape(1, -1)
            elif vec.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"Length of vec ({vec.shape[1]}) does not match number of "
                    f"features ({self.n_features_in_})"
                )
        elif not isinstance(vec, list) or len(vec) != self.n_features_in_:
            raise ValueError(
                f"vec must be array-like with {self.n_features_in_} features"
            )

        summaries = []
        for group_range in self.feature_groups_:
            try:
                group_features = (
                    [vec[j] for j in group_range]
                    if isinstance(vec, list)
                    else vec[:, group_range]
                )
                summaries.append(f(group_features))
            except Exception as e:
                raise RuntimeError(
                    f"Error applying function to group {group_range}: {e}"
                )

        return summaries

    def group_expand(self, vec_or_num: Union[List[T], T, np.ndarray]) -> List[T]:
        """Expand a vector or number to a list of features.

        Parameters
        ----------
        vec_or_num : int, float, list or ndarray
            Either a single number or a list/ndarray with length equal to number
            of groups.

        Returns
        -------
        list
            Expanded list of features.
        """
        check_is_fitted(self)

        if isinstance(vec_or_num, (int, float)):
            return [vec_or_num] * self.n_features_in_

        if isinstance(vec_or_num, (list, np.ndarray)):
            if len(vec_or_num) != self.n_groups_:
                raise ValueError(
                    f"Length of vec_or_num ({len(vec_or_num)}) does not match number of "
                    f"groups ({self.n_groups_})"
                )

            if isinstance(vec_or_num, np.ndarray):
                vec_or_num = vec_or_num.tolist()

            expanded = []
            for val, group_range in zip(vec_or_num, self.feature_groups_):
                expanded.extend([val] * len(group_range))
            return expanded

        raise TypeError(
            "vec_or_num must be either a number (int or float) or a list/ndarray, got "
            f"{type(vec_or_num).__name__}"
        )

    @classmethod
    def from_group_size(cls, group_size: int, num_groups: int):
        """Create a GroupedFeatures instance with uniform group sizes.

        Parameters
        ----------
        group_size : int
            Size of each group.
        num_groups : int
            Number of groups.

        Returns
        -------
        GroupedFeatures
            Instance with uniform group sizes.
        """
        if not isinstance(group_size, int):
            raise TypeError(
                f"group_size must be an integer, got {type(group_size).__name__}"
            )
        if not isinstance(num_groups, int):
            raise TypeError(
                f"num_groups must be an integer, got {type(num_groups).__name__}"
            )
        if group_size <= 0:
            raise ValueError("group_size must be a positive integer")
        if num_groups <= 0:
            raise ValueError("num_groups must be a positive integer")

        return cls([group_size] * num_groups)


def fill(value: T, length: int) -> List[T]:
    """Fill a list with a given value repeated length times.

    Parameters
    ----------
    value : T
        Value to fill the list with.
    length : int
        Number of times to repeat the value.

    Returns
    -------
    list
        List containing the value repeated length times.
    """
    if not isinstance(length, int):
        raise TypeError(f"length must be an integer, got {type(length).__name__}")
    if length < 0:
        raise ValueError("length must be a non-negative integer")
    return [value] * length
