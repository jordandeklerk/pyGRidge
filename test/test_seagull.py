import pytest
import numpy as np
from ..src.seagull import seagull
from numpy.testing import assert_allclose
import warnings


# TODO: Some of these tests fail. Need to fix this.
@pytest.fixture(autouse=True)
def mock_imports(monkeypatch):
    def mock_lambda_max(*args, **kwargs):
        return 1.0

    def mock_lasso(*args, **kwargs):
        return {"result": "lasso", "lambda_values": [1.0]}

    def mock_group_lasso(*args, **kwargs):
        return {"result": "group_lasso", "lambda_values": [1.0]}

    def mock_sparse_group_lasso(*args, **kwargs):
        return {"result": "sparse_group_lasso", "lambda_values": [1.0]}

    monkeypatch.setattr("lambda_max_lasso.lambda_max_lasso", mock_lambda_max)
    monkeypatch.setattr("lambda_max_group_lasso.lambda_max_group_lasso", mock_lambda_max)
    monkeypatch.setattr(
        "lambda_max_sparse_group_lasso.lambda_max_sparse_group_lasso",
        mock_lambda_max,
    )
    monkeypatch.setattr("lasso.lasso", mock_lasso)
    monkeypatch.setattr("group_lasso.group_lasso", mock_group_lasso)
    monkeypatch.setattr("sparse_group_lasso.sparse_group_lasso", mock_sparse_group_lasso)


# Fixture for common test data
@pytest.fixture
def test_data():
    y = np.array([1, 2, 3, 4, 5])
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    Z = np.array([[1, 1, 1], [1, 2, 3], [1, 3, 6], [1, 4, 10], [1, 5, 15]])
    weights_u = np.array([1, 1, 1])
    groups = np.array([1, 1, 2, 2, 3])
    return y, X, Z, weights_u, groups


# Test input validation
def test_missing_y():
    with pytest.raises(ValueError, match="Vector y is missing"):
        seagull(y=None, Z=np.array([[1, 2], [3, 4]]))


def test_empty_y():
    with pytest.raises(ValueError, match="Vector y is empty"):
        seagull(y=np.array([]), Z=np.array([[1, 2], [3, 4]]))


def test_non_numeric_y():
    with pytest.raises(ValueError, match="Non-numeric values detected in vector y"):
        seagull(y=np.array(["a", "b"]), Z=np.array([[1, 2], [3, 4]]))


def test_nan_inf_y():
    with pytest.raises(ValueError, match="NA, NaN, or Inf detected in vector y"):
        seagull(y=np.array([1, np.nan, 3]), Z=np.array([[1, 2], [3, 4], [5, 6]]))


def test_missing_Z():
    with pytest.raises(ValueError, match="Matrix Z is missing"):
        seagull(y=np.array([1, 2, 3]), Z=None)


def test_non_numeric_Z():
    with pytest.raises(ValueError, match="Non-numeric values detected in matrix Z"):
        seagull(y=np.array([1, 2]), Z=np.array([["a", "b"], ["c", "d"]]))


def test_nan_inf_Z():
    with pytest.raises(ValueError, match="NA, NaN, or Inf detected in matrix Z"):
        seagull(y=np.array([1, 2]), Z=np.array([[1, np.inf], [3, 4]]))


def test_mismatching_dimensions_y_Z():
    with pytest.raises(
        ValueError, match="Mismatching dimensions of vector y and matrix Z"
    ):
        seagull(y=np.array([1, 2, 3]), Z=np.array([[1, 2], [3, 4]]))


# Test core functionality
def test_lasso(test_data):
    y, X, Z, weights_u, _ = test_data
    result = seagull(y=y, X=X, Z=Z, weights_u=weights_u, alpha=1.0)
    assert result["result"] == "lasso"


def test_group_lasso(test_data):
    y, X, Z, weights_u, groups = test_data
    result = seagull(y=y, X=X, Z=Z, weights_u=weights_u, groups=groups, alpha=0.0)
    assert result["result"] == "group_lasso"


def test_sparse_group_lasso(test_data):
    y, X, Z, weights_u, groups = test_data
    result = seagull(y=y, X=X, Z=Z, weights_u=weights_u, groups=groups, alpha=0.5)
    assert result["result"] == "sparse_group_lasso"


# Test edge cases and parameter validation
def test_invalid_alpha():
    y = np.array([1, 2, 3])
    Z = np.array([[1, 2], [3, 4], [5, 6]])
    with pytest.warns(UserWarning, match="The parameter alpha is out of range"):
        result = seagull(y, Z=Z, alpha=2.0)
        assert result["result"] == "lasso"  # Should default to lasso when alpha > 1


def test_missing_groups_for_group_lasso():
    y = np.array([1, 2, 3])
    Z = np.array([[1, 2], [3, 4], [5, 6]])
    with pytest.raises(ValueError, match="Vector groups is missing"):
        seagull(y, Z=Z, alpha=0.5)


def test_invalid_rel_acc():
    y = np.array([1, 2, 3])
    Z = np.array([[1, 2], [3, 4], [5, 6]])
    with pytest.warns(UserWarning, match="The parameter rel_acc is non-positive"):
        result = seagull(y, Z=Z, rel_acc=-0.1, alpha=1.0)
        assert result["result"] == "lasso"


def test_invalid_max_iter():
    y = np.array([1, 2, 3])
    Z = np.array([[1, 2], [3, 4], [5, 6]])
    with pytest.warns(UserWarning, match="The parameter max_iter is non-positive"):
        result = seagull(y, Z=Z, max_iter=0, alpha=1.0)
        assert result["result"] == "lasso"


def test_invalid_gamma_bls():
    y = np.array([1, 2, 3])
    Z = np.array([[1, 2], [3, 4], [5, 6]])
    with pytest.warns(UserWarning, match="The parameter gamma_bls is out of range"):
        result = seagull(y, Z=Z, gamma_bls=1.5, alpha=1.0)
        assert result["result"] == "lasso"
