"""Tests for the group lasso module."""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from sklearn.utils._testing import assert_raises
from sklearn.exceptions import NotFittedError
from ..src.group_lasso import GroupLasso, GroupLassoError, ConvergenceError


@pytest.fixture
def simple_problem():
    """Create a simple problem for testing."""
    X = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    y = np.array([1, 2, 3])
    groups = np.array([1, 1, 2, 2])  # Two groups of two features each
    return X, y, groups


def test_init_params():
    """Test parameter initialization."""
    clf = GroupLasso(tol=1e-3, max_iter=100)
    assert clf.tol == 1e-3
    assert clf.max_iter == 100
    assert clf.verbose is False


def test_invalid_params():
    """Test that invalid parameters raise appropriate errors."""
    with pytest.raises(ValueError, match="tol must be positive"):
        GroupLasso(tol=-1)

    with pytest.raises(ValueError, match="max_iter must be positive"):
        GroupLasso(max_iter=0)

    with pytest.raises(ValueError, match="gamma must be between 0 and 1"):
        GroupLasso(gamma=1.5)

    with pytest.raises(ValueError, match="proportion_xi must be between 0 and 1"):
        GroupLasso(proportion_xi=2.0)

    with pytest.raises(ValueError, match="num_intervals must be positive"):
        GroupLasso(num_intervals=0)

    with pytest.raises(ValueError, match="num_fixed_effects must be non-negative"):
        GroupLasso(num_fixed_effects=-1)

    with pytest.raises(ValueError, match="lambda_max must be positive"):
        GroupLasso(lambda_max=-1.0)


def test_fit_predict(simple_problem):
    """Test basic fitting and prediction."""
    X, y, groups = simple_problem
    clf = GroupLasso(groups=groups, max_iter=1000)
    clf.fit(X, y)

    assert hasattr(clf, "coef_")
    assert hasattr(clf, "n_iter_")
    assert hasattr(clf, "lambda_path_")
    assert hasattr(clf, "coef_path_")

    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape


def test_feature_weights(simple_problem):
    """Test custom feature weights."""
    X, y, groups = simple_problem
    feature_weights = np.array([0.5, 0.5, 1.0, 1.0])
    clf = GroupLasso(groups=groups, feature_weights=feature_weights)
    clf.fit(X, y)

    # Features with lower weights should have larger coefficients
    group1_norm = np.linalg.norm(clf.coef_[:2])
    group2_norm = np.linalg.norm(clf.coef_[2:])
    assert group1_norm > group2_norm


def test_groups_validation(simple_problem):
    """Test groups validation."""
    X, y, _ = simple_problem
    
    # Wrong number of groups
    groups = np.array([1, 1, 2])  # One feature missing
    clf = GroupLasso(groups=groups)
    with pytest.raises(ValueError, match="groups has wrong shape"):
        clf.fit(X, y)


def test_feature_weights_validation(simple_problem):
    """Test feature weights validation."""
    X, y, groups = simple_problem
    
    # Wrong number of weights
    feature_weights = np.array([0.5, 0.5, 1.0])  # One weight missing
    clf = GroupLasso(groups=groups, feature_weights=feature_weights)
    with pytest.raises(ValueError, match="feature_weights has wrong shape"):
        clf.fit(X, y)


def test_predict_without_fit(simple_problem):
    """Test prediction without fitting."""
    X, _, groups = simple_problem
    clf = GroupLasso(groups=groups)
    with pytest.raises(NotFittedError, match="This GroupLasso instance is not fitted yet"):
        clf.predict(X)


def test_predict_with_wrong_shape(simple_problem):
    """Test prediction with wrong input shape."""
    X, y, groups = simple_problem
    clf = GroupLasso(groups=groups)
    clf.fit(X, y)
    
    X_wrong = np.array([[1, 2]])  # Wrong number of features
    with pytest.raises(ValueError, match="X has .* features"):
        clf.predict(X_wrong)


def test_convergence(simple_problem):
    """Test convergence behavior."""
    X, y, groups = simple_problem
    
    # Should converge
    clf = GroupLasso(groups=groups, max_iter=1000, tol=1e-4)
    clf.fit(X, y)
    assert clf.n_iter_ < 1000

    # Should not converge
    clf_no_conv = GroupLasso(groups=groups, max_iter=1, tol=1e-10)
    clf_no_conv.fit(X, y)
    assert clf_no_conv.n_iter_ == 1


def test_lambda_path(simple_problem):
    """Test the regularization path."""
    X, y, groups = simple_problem
    clf = GroupLasso(groups=groups, num_intervals=5)
    clf.fit(X, y)
    
    assert len(clf.lambda_path_) == 5
    assert clf.lambda_path_[0] > clf.lambda_path_[-1]  # Should decrease


def test_fixed_effects(simple_problem):
    """Test with fixed effects."""
    X, y, groups = simple_problem
    clf = GroupLasso(groups=groups, num_fixed_effects=2)
    clf.fit(X, y)
    
    # First two coefficients should be for fixed effects
    fixed_effects = clf.coef_[:2]
    random_effects = clf.coef_[2:]
    assert len(fixed_effects) == 2
    assert len(random_effects) == 2


def test_verbose_output(simple_problem, capsys):
    """Test verbose output."""
    X, y, groups = simple_problem
    clf = GroupLasso(groups=groups, verbose=True, num_intervals=2)
    clf.fit(X, y)
    
    captured = capsys.readouterr()
    assert "Loop: 1 of 2 finished" in captured.out
    assert "Loop: 2 of 2 finished" in captured.out
