# TODO: Remove this file soon 

import pytest
import numpy as np
import time
from scipy.linalg import cho_solve

from ..src.blockridge import (
    CholeskyRidgePredictor,
    WoodburyRidgePredictor,
    ShermanMorrisonRidgePredictor,
    GroupRidgeRegressor,
    lambda_lolas_rule,
    MomentTunerSetup,
    sigma_squared_path,
    get_lambdas,
    get_alpha_s_squared,
)

from ..src.groupedfeatures import GroupedFeatures
from ..src.nnls import NNLSError


@pytest.fixture
def X():
    """Generate a random matrix X for testing."""
    np.random.seed(0)
    return np.random.randn(100, 10)


@pytest.fixture
def Y(X):
    """Generate a response vector Y based on X for testing."""
    np.random.seed(1)
    beta_true = np.random.randn(X.shape[1])
    noise = np.random.randn(X.shape[0])
    return X @ beta_true + noise


# Group configurations
@pytest.fixture(
    params=[
        [10],  # Single group
        [5, 5],  # Two groups
        [2, 3, 5],  # Three groups
        [1] * 10,  # Ten groups
        [3, 2, 2, 3],  # Four groups with varying sizes
        [4, 6],  # Two groups with different sizes
        [2, 2, 2, 2, 2],  # Five groups
    ]
)
def groups(request):
    """Generate different group configurations for testing."""
    return GroupedFeatures(request.param)


def test_cholesky_ridge_predictor_initialization(X):
    """Test the initialization of CholeskyRidgePredictor."""
    predictor = CholeskyRidgePredictor(X)
    expected_XtX = np.dot(X.T, X) / X.shape[0]
    np.testing.assert_allclose(predictor.XtX, expected_XtX, atol=1e-6)
    assert (
        predictor.XtXp_lambda_chol is not None
    )  # Cholesky factor should be initialized
    expected_XtXp_lambda = expected_XtX + np.eye(X.shape[1])
    np.testing.assert_allclose(predictor.XtXp_lambda, expected_XtXp_lambda, atol=1e-6)


def test_cholesky_ridge_predictor_set_params(X, groups):
    """Test updating lambda values in CholeskyRidgePredictor."""
    predictor = CholeskyRidgePredictor(X)
    # Generate lambdas based on the number of groups
    lambdas = np.linspace(0.5, 1.5, groups.num_groups)
    predictor.set_params(groups, lambdas)
    expected_diag = groups.group_expand(lambdas)
    expected_XtXp_lambda = predictor.XtX + np.diag(expected_diag)
    np.testing.assert_allclose(predictor.XtXp_lambda, expected_XtXp_lambda, atol=1e-6)
    # Verify Cholesky factor by reconstructing
    reconstructed = predictor.XtXp_lambda_chol @ predictor.XtXp_lambda_chol.T
    np.testing.assert_allclose(reconstructed, expected_XtXp_lambda, atol=1e-6)


def test_cholesky_ridge_predictor_solve_system(X):
    """Test the _solve_system operation of CholeskyRidgePredictor."""
    predictor = CholeskyRidgePredictor(X)
    B = np.random.randn(X.shape[1], 3)
    result = predictor._solve_system(B)
    expected = np.linalg.solve(predictor.XtXp_lambda, B)
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_woodbury_ridge_predictor_initialization(X):
    """Test the initialization of WoodburyRidgePredictor."""
    predictor = WoodburyRidgePredictor(X)
    p = X.shape[1]
    assert predictor.A_inv.shape == (p, p)
    assert predictor.U.shape == X.T.shape
    assert predictor.V.shape == X.shape


def test_woodbury_ridge_predictor_set_params(X, groups):
    """Test updating lambda values in WoodburyRidgePredictor."""
    predictor = WoodburyRidgePredictor(X)
    # Generate lambdas based on the number of groups
    lambdas = np.linspace(0.5, 1.5, groups.num_groups)
    predictor.set_params(groups, lambdas)

    # Create the expected matrix
    expected_A = np.diag(groups.group_expand(lambdas)) + X.T @ X
    expected_A_inv = np.linalg.inv(expected_A)

    np.testing.assert_allclose(predictor.A_inv, expected_A_inv, atol=1e-6)

    # Test _solve_system operation
    B = np.random.randn(X.shape[1], 3)
    result = predictor._solve_system(B)
    expected = np.linalg.solve(expected_A, B)
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_group_ridge_regressor_initialization(X, Y, groups):
    """Test the initialization of GroupRidgeRegressor."""
    regressor = GroupRidgeRegressor(X=X, Y=Y, groups=groups)
    assert regressor.n == X.shape[0]
    assert regressor.p == X.shape[1]
    assert regressor.groups == groups
    assert regressor.XtY.shape == (X.shape[1],)
    assert regressor.lambdas.shape == (groups.num_groups,)
    assert regressor.beta_current.shape == (X.shape[1],)
    assert regressor.Y_hat.shape == (X.shape[0],)
    assert regressor.XtXp_lambda_div_Xt.shape == (X.shape[1], X.shape[0])


def test_group_ridge_regressor_fit(X, Y, groups):
    """Test the fit method of GroupRidgeRegressor."""
    regressor = GroupRidgeRegressor(X=X, Y=Y, groups=groups)
    # Generate lambdas based on the number of groups
    lambdas = np.linspace(1.0, 1.0, groups.num_groups)
    regressor.fit(lambdas)
    assert regressor.beta_current.shape == (groups.p,)
    expected_Y_hat = X @ regressor.beta_current
    np.testing.assert_allclose(regressor.Y_hat, expected_Y_hat, atol=1e-6)


def test_group_ridge_regressor_set_params(X, Y, groups):
    """Test updating lambda values in GroupRidgeRegressor."""
    regressor = GroupRidgeRegressor(X=X, Y=Y, groups=groups)
    new_lambdas = np.random.rand(groups.num_groups)
    regressor.set_params(new_lambdas)
    np.testing.assert_allclose(regressor.lambdas, new_lambdas)


def test_group_ridge_regressor_get_n_groups(X, Y, groups):
    """Test the get_n_groups method of GroupRidgeRegressor."""
    regressor = GroupRidgeRegressor(X=X, Y=Y, groups=groups)
    assert regressor.get_n_groups() == groups.num_groups


def test_group_ridge_regressor_get_coef(X, Y, groups):
    """Test the get_coef method of GroupRidgeRegressor."""
    regressor = GroupRidgeRegressor(X=X, Y=Y, groups=groups)
    assert np.allclose(regressor.get_coef(), regressor.beta_current)


def test_group_ridge_regressor_is_linear(X, Y, groups):
    """Test the is_linear method of GroupRidgeRegressor."""
    regressor = GroupRidgeRegressor(X=X, Y=Y, groups=groups)
    assert regressor.is_linear() == True


def test_group_ridge_regressor_get_leverage(X, Y, groups):
    """Test the get_leverage method of GroupRidgeRegressor."""
    regressor = GroupRidgeRegressor(X=X, Y=Y, groups=groups)
    regressor.fit(np.ones(groups.num_groups))
    leverage = regressor.get_leverage()
    assert leverage.shape == (X.shape[0],)
    assert np.all(leverage >= 0) and np.all(leverage <= 1)


def test_group_ridge_regressor_get_model_matrix(X, Y, groups):
    """Test the get_model_matrix method of GroupRidgeRegressor."""
    regressor = GroupRidgeRegressor(X=X, Y=Y, groups=groups)
    np.testing.assert_allclose(regressor.get_model_matrix(), X)


def test_group_ridge_regressor_get_response(X, Y, groups):
    """Test the get_response method of GroupRidgeRegressor."""
    regressor = GroupRidgeRegressor(X=X, Y=Y, groups=groups)
    np.testing.assert_allclose(regressor.get_response(), Y)


def test_group_ridge_regressor_fit_with_dict(X, Y, groups):
    """Test fitting GroupRidgeRegressor with a dictionary of lambda values."""
    regressor = GroupRidgeRegressor(X=X, Y=Y, groups=groups)
    lambda_dict = {
        f"group_{i}": val for i, val in enumerate(np.random.rand(groups.num_groups))
    }
    loo_error = regressor.fit(lambda_dict)
    assert isinstance(loo_error, float)
    assert loo_error >= 0


def test_group_ridge_regressor_predict_new_data(X, Y, groups):
    """Test predicting with new data using GroupRidgeRegressor."""
    regressor = GroupRidgeRegressor(X=X, Y=Y, groups=groups)
    regressor.fit(np.ones(groups.num_groups))
    X_new = np.random.randn(10, X.shape[1])
    predictions = regressor.predict(X_new)
    assert predictions.shape == (10,)


def test_lambda_lolas_rule(X, Y, groups):
    """Test the lambda LOLAS rule calculation."""
    regressor = GroupRidgeRegressor(X=X, Y=Y, groups=groups)
    regressor.fit(np.ones(groups.num_groups))
    mom = MomentTunerSetup(regressor)
    rule_lambda = lambda_lolas_rule(regressor, multiplier=0.1)
    expected_lambda = (
        0.1 * regressor.p**2 / regressor.n / regressor.predictor._trace_gram_matrix()
    )
    assert np.isclose(rule_lambda, expected_lambda, atol=1e-6)


def test_moment_tuner_setup(X, Y, groups):
    """Test the initialization of MomentTunerSetup."""
    regressor = GroupRidgeRegressor(X=X, Y=Y, groups=groups)
    mom = MomentTunerSetup(regressor)
    assert mom.ps.tolist() == groups.ps
    assert mom.n == X.shape[0]
    assert len(mom.beta_norms_squared) == groups.num_groups
    assert len(mom.N_norms_squared) == groups.num_groups
    assert mom.M_squared.shape == (groups.num_groups, groups.num_groups)


def test_sigma_squared_path(X, Y, groups):
    """Test the sigma squared path calculation."""
    regressor = GroupRidgeRegressor(X=X, Y=Y, groups=groups)
    lambdas = np.ones(groups.num_groups)
    regressor.fit(lambdas)
    mom = MomentTunerSetup(regressor)
    sigma_s_squared = np.linspace(0.1, 2.0, 10)
    path = sigma_squared_path(regressor, mom, sigma_s_squared)
    assert "lambdas" in path
    assert "loos" in path
    assert "betas" in path
    assert path["lambdas"].shape == (10, groups.num_groups)
    assert path["loos"].shape == (10,)
    assert path["betas"].shape == (10, groups.p)


def test_get_lambdas(X, Y, groups):
    """Test the get_lambdas function."""
    regressor = GroupRidgeRegressor(X=X, Y=Y, groups=groups)
    lambdas = np.linspace(1.0, 1.0, groups.num_groups)
    regressor.fit(lambdas)
    mom = MomentTunerSetup(regressor)
    sigma_sq = 1.0
    lambdas_out = get_lambdas(mom, sigma_sq)

    # Assert that lambdas_out does not contain LARGE_VALUE unless expected
    LARGE_VALUE = 1e12
    assert not np.any(
        lambdas_out > LARGE_VALUE
    ), "Lambda contains values exceeding LARGE_VALUE."


def test_get_alpha_s_squared(X, Y, groups):
    """Test the get_alpha_s_squared function."""
    regressor = GroupRidgeRegressor(X=X, Y=Y, groups=groups)
    lambdas = np.linspace(1.0, 1.0, groups.num_groups)
    regressor.fit(lambdas)
    mom = MomentTunerSetup(regressor)
    sigma_sq = 1.0
    try:
        alpha_sq = get_alpha_s_squared(mom, sigma_sq)
    except NNLSError as e:
        pytest.fail(f"NNLS failed with error: {e}")

    assert alpha_sq.shape == (groups.num_groups,)
    # Check that alpha_sq are non-negative
    assert np.all(alpha_sq >= 0), "alpha_sq contains negative values."


def test_predict(X, Y, groups):
    """Test the predict method of GroupRidgeRegressor."""
    regressor = GroupRidgeRegressor(X=X, Y=Y, groups=groups)
    lambdas = np.linspace(1.0, 1.0, groups.num_groups)
    regressor.fit(lambdas)
    predictions = regressor.predict(X)
    np.testing.assert_allclose(predictions, regressor.Y_hat, atol=1e-6)


def test_get_loo_error(X, Y, groups):
    """Test the leave-one-out error calculation."""
    regressor = GroupRidgeRegressor(X=X, Y=Y, groups=groups)
    lambdas = np.linspace(1.0, 1.0, groups.num_groups)
    loo_error = regressor.fit(lambdas)
    assert isinstance(loo_error, float)
    # Since it's a mean of squared errors, it should be non-negative
    assert loo_error >= 0


def test_get_mse(X, Y, groups):
    """Test the mean squared error calculation for ridge regression."""
    regressor = GroupRidgeRegressor(X=X, Y=Y, groups=groups)
    lambdas = np.linspace(1.0, 1.0, groups.num_groups)
    regressor.fit(lambdas)
    X_test = np.random.randn(50, X.shape[1])
    Y_test = X_test @ regressor.beta_current + np.random.randn(50)
    mse = regressor.get_mse(X_test, Y_test)
    assert isinstance(mse, float)
    assert mse >= 0


@pytest.fixture
def high_dim_data():
    """Generate high-dimensional data for testing."""
    np.random.seed(42)
    n = 100  # Number of observations
    p = 1000  # Number of features (p >> n)
    X = np.random.randn(n, p)
    beta_true = np.random.randn(p)
    Y = X @ beta_true + np.random.randn(n) * 0.1
    groups = GroupedFeatures([50] * 20)  # 20 groups of 50 features each
    return X, Y, groups


def test_high_dimensional_case(high_dim_data):
    """Test various components of the ridge regression in high-dimensional settings."""
    X, Y, groups = high_dim_data
    regressor = GroupRidgeRegressor(X=X, Y=Y, groups=groups)

    # Test initialization
    assert regressor.n == X.shape[0]
    assert regressor.p == X.shape[1]
    assert regressor.groups.num_groups == 20

    # Test fitting
    lambdas = np.ones(groups.num_groups)
    loo_error = regressor.fit(lambdas)
    assert isinstance(loo_error, float)
    assert loo_error >= 0

    # Test prediction
    predictions = regressor.predict(X)
    assert predictions.shape == (X.shape[0],)

    # Test MSE
    mse = regressor.get_mse(X, Y)
    assert isinstance(mse, float)
    assert mse >= 0

    # Test Cholesky predictor
    chol_predictor = CholeskyRidgePredictor(X)
    chol_predictor.set_params(groups, lambdas)
    assert chol_predictor.XtXp_lambda_chol.shape == (X.shape[1], X.shape[1])

    # Test Woodbury predictor
    wood_predictor = WoodburyRidgePredictor(X)
    wood_predictor.set_params(groups, lambdas)
    assert wood_predictor.A_inv.shape == (X.shape[1], X.shape[1])

    # Test MomentTunerSetup
    mom = MomentTunerSetup(regressor)
    assert mom.ps.shape == (groups.num_groups,)
    assert mom.M_squared.shape == (groups.num_groups, groups.num_groups)

    # Test sigma_squared_path
    sigma_s_squared = np.linspace(0.1, 2.0, 5)
    path = sigma_squared_path(regressor, mom, sigma_s_squared)
    assert path["lambdas"].shape == (5, groups.num_groups)
    assert path["loos"].shape == (5,)
    assert path["betas"].shape == (5, X.shape[1])


def test_high_dimensional_predictor_speed(high_dim_data):
    """Compare the speed of Cholesky and Woodbury predictors in high-dimensional settings."""
    X, Y, groups = high_dim_data
    lambdas = np.ones(groups.num_groups)
    n, p = X.shape

    print(f"Data dimensions: n={n}, p={p}")

    # Test Cholesky predictor speed
    chol_start = time.time()
    chol_predictor = CholeskyRidgePredictor(X)
    chol_predictor.set_params(groups, lambdas)
    chol_end = time.time()
    chol_time = chol_end - chol_start

    # Test Woodbury predictor speed
    wood_start = time.time()
    wood_predictor = WoodburyRidgePredictor(X)
    wood_predictor.set_params(groups, lambdas)
    wood_end = time.time()
    wood_time = wood_end - wood_start

    print(f"Cholesky predictor time: {chol_time:.4f} seconds")
    print(f"Woodbury predictor time: {wood_time:.4f} seconds")

    # Test matrix operations
    B = np.random.randn(p, 5)

    chol_op_start = time.time()
    chol_result = chol_predictor._solve_system(B)
    chol_op_end = time.time()
    chol_op_time = chol_op_end - chol_op_start

    wood_op_start = time.time()
    wood_result = wood_predictor._solve_system(B)
    wood_op_end = time.time()
    wood_op_time = wood_op_end - wood_op_start

    print(f"Cholesky operation time: {chol_op_time:.4f} seconds")
    print(f"Woodbury operation time: {wood_op_time:.4f} seconds")

    np.testing.assert_allclose(chol_result, wood_result, rtol=1e-1, atol=1e-1)


@pytest.fixture
def very_high_dim_data():
    """Generate very high-dimensional data for testing."""
    np.random.seed(42)
    n = 100  # Number of observations
    p = 10000  # Number of features (p >>> n)
    X = np.random.randn(n, p)
    beta_true = np.random.randn(p)
    Y = X @ beta_true + np.random.randn(n) * 0.1
    groups = GroupedFeatures([500] * 20)  # 20 groups of 500 features each
    return X, Y, groups


def test_very_high_dimensional_predictor_speed(very_high_dim_data):
    """Compare the speed of Cholesky, Woodbury, and Sherman-Morrison predictors in very high-dimensional settings."""
    X, Y, groups = very_high_dim_data
    lambdas = np.ones(groups.num_groups)
    n, p = X.shape

    print(f"Data dimensions: n={n}, p={p}")

    # Test Cholesky predictor speed
    chol_start = time.time()
    chol_predictor = CholeskyRidgePredictor(X)
    chol_predictor.set_params(groups, lambdas)
    chol_end = time.time()
    chol_time = chol_end - chol_start

    # Test Woodbury predictor speed
    wood_start = time.time()
    wood_predictor = WoodburyRidgePredictor(X)
    wood_predictor.set_params(groups, lambdas)
    wood_end = time.time()
    wood_time = wood_end - wood_start

    # Test Sherman-Morrison predictor speed
    sm_start = time.time()
    sm_predictor = ShermanMorrisonRidgePredictor(X)
    sm_predictor.set_params(groups, lambdas)
    sm_end = time.time()
    sm_time = sm_end - sm_start

    print(f"Cholesky predictor time: {chol_time:.4f} seconds")
    print(f"Woodbury predictor time: {wood_time:.4f} seconds")
    print(f"Sherman-Morrison predictor time: {sm_time:.4f} seconds")

    # Test matrix operations
    B = np.random.randn(p, 5)

    chol_op_start = time.time()
    chol_result = chol_predictor._solve_system(B)
    chol_op_end = time.time()
    chol_op_time = chol_op_end - chol_op_start

    wood_op_start = time.time()
    wood_result = wood_predictor._solve_system(B)
    wood_op_end = time.time()
    wood_op_time = wood_op_end - wood_op_start

    sm_op_start = time.time()
    sm_result = sm_predictor._solve_system(B)
    sm_op_end = time.time()
    sm_op_time = sm_op_end - sm_op_start

    print(f"Cholesky operation time: {chol_op_time:.4f} seconds")
    print(f"Woodbury operation time: {wood_op_time:.4f} seconds")
    print(f"Sherman-Morrison operation time: {sm_op_time:.4f} seconds")

    np.testing.assert_allclose(chol_result, sm_result, rtol=1e-1, atol=1e-1)
    np.testing.assert_allclose(wood_result, sm_result, rtol=1e-1, atol=1e-1)

    assert wood_time < chol_time, (
        "Woodbury predictor should be faster in very high-dimensional case. Woodbury:"
        f" {wood_time:.4f}, Cholesky: {chol_time:.4f}"
    )


def test_sherman_morrison_ridge_predictor_initialization(X):
    """Test the initialization of ShermanMorrisonRidgePredictor."""
    predictor = ShermanMorrisonRidgePredictor(X)
    expected_A = np.eye(X.shape[1]) + (X.T @ X) / X.shape[0]
    expected_A_inv = np.linalg.inv(expected_A)

    assert predictor.A_inv.shape == (X.shape[1], X.shape[1])
    np.testing.assert_allclose(predictor.A_inv, expected_A_inv, atol=1e-6)


def test_sherman_morrison_ridge_predictor_set_params(X, groups):
    """Test the set_params method of ShermanMorrisonRidgePredictor."""
    predictor = ShermanMorrisonRidgePredictor(X)
    lambdas = np.linspace(0.5, 1.5, groups.num_groups)
    predictor.set_params(groups, lambdas)

    # Create the expected matrix
    expected_A = np.diag(groups.group_expand(lambdas)) + X.T @ X / X.shape[0]
    expected_A_inv = np.linalg.inv(expected_A)

    np.testing.assert_allclose(predictor.A_inv, expected_A_inv, atol=1e-6)


def test_sherman_morrison_single_update(X):
    """Test a single Sherman-Morrison update."""
    predictor = ShermanMorrisonRidgePredictor(X)
    u = np.random.randn(X.shape[1])
    v = np.random.randn(X.shape[1])

    A_inv_initial = predictor.A_inv.copy()

    # Compute expected A_inv after Sherman-Morrison update manually
    denominator = 1.0 + v @ (A_inv_initial @ u)
    expected_A_inv = (
        A_inv_initial - np.outer(A_inv_initial @ u, v @ A_inv_initial) / denominator
    )

    # Apply Sherman-Morrison update
    predictor._sherman_morrison_update(u, v)

    # Assert that predictor.A_inv matches the manually computed inverse
    np.testing.assert_allclose(predictor.A_inv, expected_A_inv, atol=1e-6)


def test_sherman_morrison_ridge_predictor_solve_system(X):
    """Test the _solve_system method of ShermanMorrisonRidgePredictor."""
    predictor = ShermanMorrisonRidgePredictor(X)
    B = np.random.randn(X.shape[1], 3)
    result = predictor._solve_system(B)
    expected = predictor.A_inv @ B
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_sherman_morrison_formula(X):
    """Test the Sherman-Morrison formula implementation."""
    predictor = ShermanMorrisonRidgePredictor(X)

    # Initialize A and compute its inverse
    A = np.random.randn(X.shape[1], X.shape[1])
    A = A @ A.T  # Make A positive definite
    A_inv = np.linalg.inv(A)
    predictor.A_inv = A_inv.copy()

    u = np.random.randn(X.shape[1])
    v = np.random.randn(X.shape[1])

    denominator = 1.0 + v @ (A_inv @ u)
    expected_A_inv = A_inv - np.outer(A_inv @ u, v @ A_inv) / denominator

    predictor._sherman_morrison_update(u, v)
    np.testing.assert_allclose(predictor.A_inv, expected_A_inv, atol=1e-6)


def test_sherman_morrison_predictor_in_high_dimensional_case(high_dim_data):
    """Test ShermanMorrisonRidgePredictor in high-dimensional settings."""
    X, Y, groups = high_dim_data
    predictor = ShermanMorrisonRidgePredictor(X)

    # Test initialization
    assert predictor.A_inv.shape == (X.shape[1], X.shape[1])

    # Test updating lambda_s
    lambdas = np.ones(groups.num_groups)
    predictor.set_params(groups, lambdas)

    # Test _solve_system operation
    B = np.random.randn(X.shape[1], 5)
    result = predictor._solve_system(B)
    assert result.shape == B.shape


def test_very_high_dimensional_predictor_speed(very_high_dim_data):
    """Compare the speed of Cholesky, Woodbury, and Sherman-Morrison predictors in very high-dimensional settings."""
    X, Y, groups = very_high_dim_data
    lambdas = np.ones(groups.num_groups)
    n, p = X.shape

    print(f"Data dimensions: n={n}, p={p}")

    # Test Cholesky predictor speed
    chol_start = time.time()
    chol_predictor = CholeskyRidgePredictor(X)
    chol_predictor.set_params(groups, lambdas)
    chol_end = time.time()
    chol_time = chol_end - chol_start

    # Test Woodbury predictor speed
    wood_start = time.time()
    wood_predictor = WoodburyRidgePredictor(X)
    wood_predictor.set_params(groups, lambdas)
    wood_end = time.time()
    wood_time = wood_end - wood_start

    print(f"Cholesky predictor time: {chol_time:.4f} seconds")
    print(f"Woodbury predictor time: {wood_time:.4f} seconds")

    # Test Sherman-Morrison predictor speed
    sm_start = time.time()
    sm_predictor = ShermanMorrisonRidgePredictor(X)
    sm_predictor.set_params(groups, lambdas)
    sm_end = time.time()
    sm_time = sm_end - sm_start

    print(f"Sherman-Morrison predictor time: {sm_time:.4f} seconds")

    # Test matrix operations
    B = np.random.randn(p, 5)

    chol_op_start = time.time()
    chol_result = chol_predictor._trace_gram_matrix()
    chol_op_end = time.time()
    chol_op_time = chol_op_end - chol_op_start

    wood_op_start = time.time()
    wood_result = wood_predictor._trace_gram_matrix()
    wood_op_end = time.time()
    wood_op_time = wood_op_end - wood_op_start

    sm_op_start = time.time()
    sm_result = sm_predictor._trace_gram_matrix()
    sm_op_end = time.time()
    sm_op_time = sm_op_end - sm_op_start

    print(f"Cholesky operation time: {chol_op_time:.4f} seconds")
    print(f"Woodbury operation time: {wood_op_time:.4f} seconds")
    print(f"Sherman-Morrison operation time: {sm_op_time:.4f} seconds")

    np.testing.assert_allclose(chol_result, sm_result, rtol=1e-1, atol=1e-1)
    np.testing.assert_allclose(wood_result, sm_result, rtol=1e-1, atol=1e-1)
