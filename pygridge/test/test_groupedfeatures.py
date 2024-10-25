"""Tests for GroupedFeatures class."""

import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from sklearn.utils.estimator_checks import check_estimator
from sklearn.exceptions import NotFittedError

from ..src.groupedfeatures import GroupedFeatures, fill


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 6)  # 6 features for 2 groups of 3
    return X


@pytest.fixture
def sample_groups():
    """Create sample GroupedFeatures instance."""
    return GroupedFeatures([3, 3])  # 2 groups of 3 features each


def test_scikit_learn_compatibility():
    """Test scikit-learn estimator compatibility."""
    groups = GroupedFeatures([2, 2])
    check_estimator(groups)


def test_initialization():
    """Test GroupedFeatures initialization."""
    # Valid initialization
    groups = GroupedFeatures([2, 3])
    assert groups.ps == [2, 3]

    # Invalid initializations
    with pytest.raises(TypeError):
        GroupedFeatures("not a list")
    with pytest.raises(TypeError):
        GroupedFeatures([1.5, 2])  # non-integer
    with pytest.raises(ValueError):
        GroupedFeatures([0, 1])  # zero group size
    with pytest.raises(ValueError):
        GroupedFeatures([-1, 2])  # negative group size


def test_from_group_size():
    """Test from_group_size class method."""
    groups = GroupedFeatures.from_group_size(2, 3)
    assert groups.ps == [2, 2, 2]

    with pytest.raises(TypeError):
        GroupedFeatures.from_group_size(1.5, 2)  # non-integer group_size
    with pytest.raises(TypeError):
        GroupedFeatures.from_group_size(2, 2.5)  # non-integer num_groups
    with pytest.raises(ValueError):
        GroupedFeatures.from_group_size(0, 2)  # zero group_size
    with pytest.raises(ValueError):
        GroupedFeatures.from_group_size(2, 0)  # zero num_groups


def test_fit_transform(sample_data, sample_groups):
    """Test fit and transform methods."""
    # Test fit
    sample_groups.fit(sample_data)
    assert hasattr(sample_groups, "n_features_in_")
    assert sample_groups.n_features_in_ == 6
    assert hasattr(sample_groups, "feature_groups_")
    assert len(sample_groups.feature_groups_) == 2

    # Test transform without group_operation (should return same data)
    X_transformed = sample_groups.transform(sample_data)
    assert_array_equal(X_transformed, sample_data)

    # Test transform with group_operation
    def mean_operation(group_data):
        return np.mean(group_data, axis=1, keepdims=True)

    groups_with_op = GroupedFeatures([3, 3], group_operation=mean_operation)
    groups_with_op.fit(sample_data)
    X_transformed = groups_with_op.transform(sample_data)
    assert X_transformed.shape == (100, 2)  # One value per group

    # Test transform before fit
    groups_unfit = GroupedFeatures([3, 3])
    with pytest.raises(NotFittedError):
        groups_unfit.transform(sample_data)


def test_get_feature_names_out(sample_data, sample_groups):
    """Test get_feature_names_out method."""
    sample_groups.fit(sample_data)

    # Test without input feature names
    feature_names = sample_groups.get_feature_names_out()
    assert len(feature_names) == 6
    assert all(isinstance(name, str) for name in feature_names)

    # Test with input feature names
    input_names = [f"feat_{i}" for i in range(6)]
    feature_names = sample_groups.get_feature_names_out(input_names)
    assert_array_equal(feature_names, input_names)

    # Test with group_operation (should return group names)
    def mean_operation(group_data):
        return np.mean(group_data, axis=1, keepdims=True)

    groups_with_op = GroupedFeatures([3, 3], group_operation=mean_operation)
    groups_with_op.fit(sample_data)
    feature_names = groups_with_op.get_feature_names_out()
    assert len(feature_names) == 2
    assert all("group" in name for name in feature_names)


def test_group_idx(sample_groups):
    """Test group_idx method."""
    sample_groups.fit(np.random.randn(10, 6))  # Fit with dummy data

    # Test valid indices
    idx0 = sample_groups.group_idx(0)
    assert list(idx0) == [0, 1, 2]
    idx1 = sample_groups.group_idx(1)
    assert list(idx1) == [3, 4, 5]

    # Test invalid indices
    with pytest.raises(TypeError):
        sample_groups.group_idx(1.5)  # non-integer
    with pytest.raises(IndexError):
        sample_groups.group_idx(2)  # out of range


def test_group_summary(sample_data, sample_groups):
    """Test group_summary method."""
    sample_groups.fit(sample_data)

    # Test with numpy array
    def mean_summary(x):
        return np.mean(x)

    summaries = sample_groups.group_summary(sample_data, mean_summary)
    assert len(summaries) == 2

    # Test with list
    data_list = sample_data.tolist()
    summaries_list = sample_groups.group_summary(data_list, mean_summary)
    assert len(summaries_list) == 2

    # Test invalid inputs
    with pytest.raises(TypeError):
        sample_groups.group_summary(sample_data, "not a function")


def test_group_expand(sample_groups):
    """Test group_expand method."""
    sample_groups.fit(np.random.randn(10, 6))  # Fit with dummy data

    # Test with single number
    expanded = sample_groups.group_expand(1.0)
    assert len(expanded) == 6
    assert all(x == 1.0 for x in expanded)

    # Test with list
    expanded = sample_groups.group_expand([1.0, 2.0])
    assert len(expanded) == 6
    assert expanded[:3] == [1.0, 1.0, 1.0]
    assert expanded[3:] == [2.0, 2.0, 2.0]

    # Test with numpy array
    expanded = sample_groups.group_expand(np.array([1.0, 2.0]))
    assert len(expanded) == 6
    assert all(expanded[:3] == 1.0)
    assert all(expanded[3:] == 2.0)

    # Test invalid inputs
    with pytest.raises(ValueError):
        sample_groups.group_expand([1.0])  # wrong length
    with pytest.raises(TypeError):
        sample_groups.group_expand("not a number or list")


def test_fill():
    """Test fill function."""
    filled = fill(1.0, 3)
    assert len(filled) == 3
    assert all(x == 1.0 for x in filled)

    with pytest.raises(TypeError):
        fill(1.0, 1.5)  # non-integer length
    with pytest.raises(ValueError):
        fill(1.0, -1)  # negative length
