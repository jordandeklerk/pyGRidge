import pytest
import numpy as np
from unittest.mock import Mock, patch

from ..src.covariance_design import (
    DiscreteNonParametric,
    CovarianceDesign,
    AR1Design,
    DiagonalCovarianceDesign,
    IdentityCovarianceDesign,
    UniformScalingCovarianceDesign,
    ExponentialOrderStatsCovarianceDesign,
    BlockDiagonal,
    MixtureModel,
    BlockCovarianceDesign,
    block_diag,
    simulate_rotated_design,
    set_groups,
)

from ..src.groupedfeatures import GroupedFeatures


def test_discrete_non_parametric_valid():
    """Test valid initialization of DiscreteNonParametric."""
    eigs = [1.0, 2.0, 3.0]
    probs = [0.2, 0.3, 0.5]
    spectrum = DiscreteNonParametric(eigs, probs)
    np.testing.assert_array_equal(spectrum.eigs, np.array(eigs))
    np.testing.assert_array_equal(spectrum.probs, np.array(probs))


def test_discrete_non_parametric_invalid_eigs_type():
    """Test TypeError for invalid eigenvalues type."""
    with pytest.raises(TypeError):
        DiscreteNonParametric(eigs="not_a_list", probs=[0.5, 0.5])


def test_discrete_non_parametric_invalid_eigs_elements():
    """Test ValueError for invalid elements in eigenvalues."""
    with pytest.raises(ValueError):
        DiscreteNonParametric(eigs=[1.0, "two", 3.0], probs=[0.3, 0.4, 0.3])


def test_discrete_non_parametric_invalid_probs_type():
    """Test TypeError for invalid probabilities type."""
    with pytest.raises(TypeError):
        DiscreteNonParametric(eigs=[1.0, 2.0], probs="not_a_list")


def test_discrete_non_parametric_probs_length_mismatch():
    """Test ValueError for mismatched lengths of eigenvalues and probabilities."""
    with pytest.raises(ValueError):
        DiscreteNonParametric(eigs=[1.0, 2.0], probs=[0.5])


def test_discrete_non_parametric_probs_sum_not_one():
    """Test ValueError for probabilities that do not sum to one."""
    with pytest.raises(ValueError):
        DiscreteNonParametric(eigs=[1.0, 2.0], probs=[0.6, 0.5])


def test_discrete_non_parametric_negative_probs():
    """Test ValueError for negative probabilities."""
    with pytest.raises(ValueError):
        DiscreteNonParametric(eigs=[1.0, 2.0], probs=[0.7, -0.7])


def test_ar1_design_valid():
    """Test valid initialization of AR1Design."""
    p = 5
    rho = 0.6
    ar1 = AR1Design(p=p, rho=rho)
    assert ar1.p == p
    assert ar1.rho == rho
    Sigma = ar1.get_Sigma()
    expected_Sigma = rho ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    np.testing.assert_array_almost_equal(Sigma, expected_Sigma)
    assert ar1.nfeatures() == p
    spectrum = ar1.spectrum()
    assert isinstance(spectrum, DiscreteNonParametric)
    assert len(spectrum.eigs) == p
    assert np.isclose(np.sum(spectrum.probs), 1.0)


def test_ar1_design_invalid_p_type():
    """Test TypeError for invalid p type in AR1Design."""
    with pytest.raises(TypeError):
        AR1Design(p="five", rho=0.6)


def test_ar1_design_invalid_p_value():
    """Test ValueError for invalid p value in AR1Design."""
    with pytest.raises(ValueError):
        AR1Design(p=0, rho=0.6)


def test_ar1_design_invalid_rho_type():
    """Test TypeError for invalid rho type in AR1Design."""
    with pytest.raises(TypeError):
        AR1Design(p=5, rho="0.6")


def test_ar1_design_invalid_rho_value():
    """Test ValueError for invalid rho value in AR1Design."""
    with pytest.raises(ValueError):
        AR1Design(p=5, rho=1.5)


def test_ar1_design_missing_p():
    """Test ValueError when p is None in AR1Design."""
    ar1 = AR1Design(p=None, rho=0.6)
    with pytest.raises(ValueError):
        ar1.get_Sigma()


def test_identity_covariance_design_valid():
    """Test valid initialization of IdentityCovarianceDesign."""
    p = 4
    identity = IdentityCovarianceDesign(p=p)
    Sigma = identity.get_Sigma()
    expected_Sigma = np.identity(p)
    np.testing.assert_array_equal(Sigma, expected_Sigma)
    assert identity.nfeatures() == p
    spectrum = identity.spectrum()
    assert isinstance(spectrum, DiscreteNonParametric)
    assert all(e == 1.0 for e in spectrum.eigs)
    assert np.allclose(spectrum.probs, np.full(p, 1.0 / p))


def test_identity_covariance_design_invalid_p_type():
    """Test TypeError for invalid p type in IdentityCovarianceDesign."""
    with pytest.raises(TypeError):
        IdentityCovarianceDesign(p="four")


def test_identity_covariance_design_invalid_p_value():
    """Test ValueError for invalid p value in IdentityCovarianceDesign."""
    with pytest.raises(ValueError):
        IdentityCovarianceDesign(p=-2)


def test_identity_covariance_design_missing_p():
    """Test ValueError when p is None in IdentityCovarianceDesign."""
    identity = IdentityCovarianceDesign()
    with pytest.raises(ValueError):
        identity.get_Sigma()
    with pytest.raises(ValueError):
        identity.nfeatures()
    with pytest.raises(ValueError):
        identity.spectrum()


def test_uniform_scaling_covariance_design_valid():
    """Test valid initialization of UniformScalingCovarianceDesign."""
    p = 3
    scaling = 2.5
    uniform = UniformScalingCovarianceDesign(scaling=scaling, p=p)
    Sigma = uniform.get_Sigma()
    expected_Sigma = scaling * np.identity(p)
    np.testing.assert_array_equal(Sigma, expected_Sigma)
    assert uniform.scaling == scaling
    assert uniform.nfeatures() == p
    spectrum = uniform.spectrum()
    assert isinstance(spectrum, DiscreteNonParametric)
    assert all(e == scaling for e in spectrum.eigs)
    assert np.allclose(spectrum.probs, np.full(p, 1.0 / p))


def test_uniform_scaling_covariance_design_invalid_scaling_type():
    """Test TypeError for invalid scaling type in UniformScalingCovarianceDesign."""
    with pytest.raises(TypeError):
        UniformScalingCovarianceDesign(scaling="2.5", p=3)


def test_uniform_scaling_covariance_design_invalid_scaling_value():
    """Test ValueError for invalid scaling value in UniformScalingCovarianceDesign."""
    with pytest.raises(ValueError):
        UniformScalingCovarianceDesign(scaling=-1.0, p=3)


def test_uniform_scaling_covariance_design_invalid_p_type():
    """Test TypeError for invalid p type in UniformScalingCovarianceDesign."""
    with pytest.raises(TypeError):
        UniformScalingCovarianceDesign(scaling=1.0, p="three")


def test_uniform_scaling_covariance_design_invalid_p_value():
    """Test ValueError for invalid p value in UniformScalingCovarianceDesign."""
    with pytest.raises(ValueError):
        UniformScalingCovarianceDesign(scaling=1.0, p=0)


def test_exponential_order_stats_covariance_design_valid():
    """Test valid initialization of ExponentialOrderStatsCovarianceDesign."""
    p = 4
    rate = 1.5
    exp_design = ExponentialOrderStatsCovarianceDesign(p=p, rate=rate)
    spectrum = exp_design.spectrum()
    assert isinstance(spectrum, DiscreteNonParametric)
    assert len(spectrum.eigs) == p
    assert np.all(spectrum.eigs > 0)
    assert np.isclose(np.sum(spectrum.probs), 1.0)
    Sigma = exp_design.get_Sigma()
    assert np.allclose(Sigma, np.diag(spectrum.eigs))


def test_exponential_order_stats_covariance_design_invalid_rate_type():
    """Test TypeError for invalid rate type in ExponentialOrderStatsCovarianceDesign."""
    with pytest.raises(TypeError):
        ExponentialOrderStatsCovarianceDesign(p=3, rate="1.0")


def test_exponential_order_stats_covariance_design_invalid_rate_value():
    """Test ValueError for invalid rate value in ExponentialOrderStatsCovarianceDesign."""
    with pytest.raises(ValueError):
        ExponentialOrderStatsCovarianceDesign(p=3, rate=0)


def test_exponential_order_stats_covariance_design_invalid_p_type():
    """Test TypeError for invalid p type in ExponentialOrderStatsCovarianceDesign."""
    with pytest.raises(TypeError):
        ExponentialOrderStatsCovarianceDesign(p="three", rate=1.0)


def test_exponential_order_stats_covariance_design_invalid_p_value():
    """Test ValueError for invalid p value in ExponentialOrderStatsCovarianceDesign."""
    with pytest.raises(ValueError):
        ExponentialOrderStatsCovarianceDesign(p=-1, rate=1.0)


def test_exponential_order_stats_covariance_design_missing_p():
    """Test ValueError when p is None in ExponentialOrderStatsCovarianceDesign."""
    exp_design = ExponentialOrderStatsCovarianceDesign(p=None, rate=1.0)
    with pytest.raises(ValueError):
        exp_design.get_Sigma()
    with pytest.raises(ValueError):
        exp_design.spectrum()
    with pytest.raises(ValueError):
        exp_design.nfeatures()


def test_block_diagonal_valid():
    """Test valid initialization of BlockDiagonal."""
    block1 = np.array([[1, 0], [0, 1]])
    block2 = np.array([[2, 0], [0, 2]])
    block_diag_obj = BlockDiagonal(blocks=[block1, block2])
    Sigma = block_diag_obj.get_Sigma()
    expected_Sigma = block_diag(block1, block2)
    np.testing.assert_array_equal(Sigma, expected_Sigma)


def test_block_diagonal_empty_blocks():
    """Test ValueError for empty blocks in BlockDiagonal."""
    block_diag_obj = BlockDiagonal(blocks=[])
    with pytest.raises(
        ValueError, match=r"Resulting covariance matrix :math:`\\Sigma` is empty\."
    ):
        Sigma = block_diag_obj.get_Sigma()


def test_block_diagonal_invalid_blocks_type():
    """Test TypeError for invalid blocks type in BlockDiagonal."""
    with pytest.raises(TypeError):
        BlockDiagonal(blocks="not_a_list")


def test_block_diagonal_invalid_blocks_elements():
    """Test TypeError for invalid elements in blocks of BlockDiagonal."""
    with pytest.raises(TypeError):
        BlockDiagonal(blocks=[np.array([[1, 0]]), "not_an_array"])


def test_block_diagonal_non_square_blocks():
    """Test ValueError for non-square blocks in BlockDiagonal."""
    with pytest.raises(ValueError):
        BlockDiagonal(blocks=[np.array([[1, 2, 3]]), np.array([[4, 5], [6, 7]])])


def test_mixture_model_valid():
    """Test valid initialization of MixtureModel."""
    eigs1 = [1.0, 2.0]
    probs1 = [0.5, 0.5]
    spectrum1 = DiscreteNonParametric(eigs=eigs1, probs=probs1)

    eigs2 = [3.0, 4.0]
    probs2 = [0.3, 0.7]
    spectrum2 = DiscreteNonParametric(eigs=eigs2, probs=probs2)

    mixture = MixtureModel(spectra=[spectrum1, spectrum2], mixing_prop=[0.6, 0.4])
    assert mixture.spectra == [spectrum1, spectrum2]
    assert mixture.mixing_prop == [0.6, 0.4]


def test_mixture_model_invalid_spectra_type():
    """Test TypeError for invalid spectra type in MixtureModel."""
    with pytest.raises(TypeError):
        MixtureModel(spectra="not_a_list", mixing_prop=[0.5, 0.5])


def test_mixture_model_invalid_spectra_elements():
    """Test TypeError for invalid elements in spectra of MixtureModel."""
    with pytest.raises(TypeError):
        MixtureModel(spectra=[Mock(), Mock()], mixing_prop=[0.5, 0.5])


def test_mixture_model_invalid_mixing_prop_type():
    """Test TypeError for invalid mixing_prop type in MixtureModel."""
    spectrum = DiscreteNonParametric(eigs=[1.0], probs=[1.0])
    with pytest.raises(TypeError):
        MixtureModel(spectra=[spectrum], mixing_prop="not_a_list")


def test_mixture_model_invalid_mixing_prop_elements():
    """Test ValueError for invalid elements in mixing_prop of MixtureModel."""
    spectrum = DiscreteNonParametric(eigs=[1.0], probs=[1.0])
    with pytest.raises(ValueError):
        MixtureModel(spectra=[spectrum], mixing_prop=[0.7, "0.3"])


def test_mixture_model_length_mismatch():
    """Test ValueError for length mismatch between spectra and mixing_prop in MixtureModel."""
    spectrum = DiscreteNonParametric(eigs=[1.0], probs=[1.0])
    with pytest.raises(ValueError):
        MixtureModel(spectra=[spectrum], mixing_prop=[0.5, 0.5])


def test_mixture_model_mixing_prop_sum_not_one():
    """Test ValueError for mixing_prop that does not sum to one in MixtureModel."""
    spectrum = DiscreteNonParametric(eigs=[1.0], probs=[1.0])
    with pytest.raises(ValueError):
        MixtureModel(spectra=[spectrum], mixing_prop=[0.6, 0.5])


def test_mixture_model_negative_mixing_prop():
    """Test ValueError for negative mixing_prop in MixtureModel."""
    spectrum = DiscreteNonParametric(eigs=[1.0], probs=[1.0])
    with pytest.raises(ValueError):
        MixtureModel(spectra=[spectrum], mixing_prop=[-0.5, 1.5])


def test_block_covariance_design_valid():
    """Test valid initialization of BlockCovarianceDesign."""
    # Create covariance designs
    block1 = IdentityCovarianceDesign(p=2)
    block2 = UniformScalingCovarianceDesign(scaling=3.0, p=3)

    # Define groups corresponding to each block
    groups = GroupedFeatures(ps=[2, 3])

    # Instantiate BlockCovarianceDesign with groups
    block_design = BlockCovarianceDesign(blocks=[block1, block2], groups=groups)

    # Get Sigma and construct expected Sigma using block_diag
    Sigma = block_design.get_Sigma()
    expected_Sigma = block_diag(block1.get_Sigma(), block2.get_Sigma())
    np.testing.assert_array_equal(Sigma, expected_Sigma)

    assert block_design.nfeatures() == 5
    spectrum = block_design.spectrum()

    assert isinstance(spectrum, MixtureModel)
    assert len(spectrum.spectra) == 2
    np.testing.assert_allclose(spectrum.mixing_prop, [2 / 5, 3 / 5])


def test_block_covariance_design_with_groups():
    """Test BlockCovarianceDesign with specified groups."""
    groups = GroupedFeatures(ps=[2, 3])

    block1 = IdentityCovarianceDesign(p=2)
    block2 = UniformScalingCovarianceDesign(scaling=3.0, p=3)
    block_design = BlockCovarianceDesign(blocks=[block1, block2], groups=groups)

    set_groups(block_design, groups)

    Sigma = block_design.get_Sigma()
    expected_Sigma = block_diag(block1.get_Sigma(), block2.get_Sigma())
    np.testing.assert_array_equal(Sigma, expected_Sigma)

    spectrum = block_design.spectrum()
    assert isinstance(spectrum, MixtureModel)
    assert len(spectrum.spectra) == 2
    assert spectrum.mixing_prop == [2 / 5, 3 / 5]


def test_block_covariance_design_invalid_blocks_type():
    """Test TypeError for invalid blocks type in BlockCovarianceDesign."""
    with pytest.raises(TypeError):
        BlockCovarianceDesign(blocks="not_a_list")


def test_block_covariance_design_invalid_blocks_elements():
    """Test TypeError for invalid elements in blocks of BlockCovarianceDesign."""
    with pytest.raises(TypeError):
        BlockCovarianceDesign(blocks=[IdentityCovarianceDesign(p=2), "not_a_design"])


def test_block_covariance_design_invalid_groups_type():
    """Test TypeError for invalid groups type in BlockCovarianceDesign."""
    block1 = IdentityCovarianceDesign(p=2)
    with pytest.raises(TypeError):
        BlockCovarianceDesign(blocks=[block1], groups="not_grouped")


def test_block_covariance_design_groups_blocks_mismatch():
    """Test ValueError for mismatch between groups and blocks in BlockCovarianceDesign."""
    groups = GroupedFeatures(ps=[2, 3, 1])  # 3 groups, but only 2 blocks
    block1 = IdentityCovarianceDesign(p=2)
    block2 = UniformScalingCovarianceDesign(scaling=3.0, p=3)
    with pytest.raises(
        ValueError,
        match="Number of groups must match number of blocks in BlockCovarianceDesign.",
    ):
        BlockCovarianceDesign(blocks=[block1, block2], groups=groups)


def test_block_covariance_design_spectrum_sum_not_one():
    """Test ValueError for incorrect mixing properties in BlockCovarianceDesign."""
    # Mock a BlockCovarianceDesign with incorrect mixing properties
    block1 = Mock(spec=CovarianceDesign)
    block1.spectrum.return_value = DiscreteNonParametric(eigs=[1.0], probs=[1.0])
    block2 = Mock(spec=CovarianceDesign)
    block2.spectrum.return_value = DiscreteNonParametric(eigs=[2.0], probs=[1.0])

    block_design = BlockCovarianceDesign(blocks=[block1, block2])

    with pytest.raises(ValueError):
        # Manually set mixing_prop to not sum to 1
        block_design.spectrum = Mock(
            return_value=MixtureModel(
                spectra=[block1.spectrum(), block2.spectrum()], mixing_prop=[0.6, 0.5]
            )
        )
        block_design.spectrum()


def test_block_diag_valid():
    """Test valid block diagonal construction."""
    A = np.array([[1, 0], [0, 1]])
    B = np.array([[2, 0], [0, 2]])
    result = block_diag(A, B)
    expected = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2]])
    np.testing.assert_array_equal(result, expected)


def test_block_diag_no_blocks():
    """Test block_diag with no blocks."""
    result = block_diag()
    expected = np.array([[]])
    np.testing.assert_array_equal(result, expected)


def test_block_diag_non_numpy_input():
    """Test TypeError for non-numpy input in block_diag."""
    A = [[1, 0], [0, 1]]
    B = np.array([[2, 0], [0, 2]])
    with pytest.raises(TypeError):
        block_diag(A, B)


def test_block_diag_non_2d_input():
    """Test ValueError for non-2D input in block_diag."""
    A = np.array([1, 2, 3])
    B = np.array([[2, 0], [0, 2]])
    with pytest.raises(ValueError):
        block_diag(A, B)


def test_block_diag_non_square_blocks():
    """Test ValueError for non-square blocks in block_diag."""
    A = np.array([[1, 0, 0], [0, 1, 0]])
    B = np.array([[2, 0], [0, 2]])
    with pytest.raises(ValueError):
        block_diag(A, B)


def test_block_diag_exception_handling():
    """Test exception handling in block_diag."""
    A = np.array([[1, 0], [0, 1]])
    B = np.array([[2, 0], [0, 2]])
    block_diag_obj = BlockDiagonal(blocks=[A, B])

    with patch(
        "PyGRidge.src.covariance_design.block_diag",
        side_effect=Exception("Mocked exception"),
    ):
        with pytest.raises(
            RuntimeError,
            match="Failed to construct block diagonal matrix: Mocked exception",
        ):
            Sigma = block_diag_obj.get_Sigma()


def test_simulate_rotated_design_valid():
    """Test valid simulation of rotated design."""
    p = 3
    n = 10
    rho = 0.5
    ar1 = AR1Design(p=p, rho=rho)
    X = simulate_rotated_design(cov=ar1, n=n)
    assert isinstance(X, np.ndarray)
    assert X.shape == (n, p)


def test_simulate_rotated_design_invalid_cov_type():
    """Test TypeError for invalid covariance type in simulate_rotated_design."""
    with pytest.raises(TypeError):
        simulate_rotated_design(cov="not_a_cov_design", n=10)


def test_simulate_rotated_design_invalid_n_type():
    """Test TypeError for invalid n type in simulate_rotated_design."""
    ar1 = AR1Design(p=3, rho=0.5)
    with pytest.raises(TypeError):
        simulate_rotated_design(cov=ar1, n="ten")


def test_simulate_rotated_design_invalid_n_value():
    """Test ValueError for invalid n value in simulate_rotated_design."""
    ar1 = AR1Design(p=3, rho=0.5)
    with pytest.raises(ValueError):
        simulate_rotated_design(cov=ar1, n=0)


def test_simulate_rotated_design_invalid_rotated_measure():
    """Test TypeError for invalid rotated measure in simulate_rotated_design."""
    ar1 = AR1Design(p=3, rho=0.5)
    with pytest.raises(TypeError):
        simulate_rotated_design(cov=ar1, n=10, rotated_measure="not_callable")


def test_simulate_rotated_design_sigma_not_positive_definite():
    """Test ValueError for non-positive definite Sigma in simulate_rotated_design."""
    # Create a covariance design that returns a non-positive definite matrix
    mock_cov = Mock(spec=CovarianceDesign)
    mock_cov.get_Sigma.return_value = np.array(
        [[1, 2], [2, 1]]
    )  # Not positive definite
    mock_cov.nfeatures.return_value = 2
    with pytest.raises(ValueError):
        simulate_rotated_design(cov=mock_cov, n=5)


def test_simulate_rotated_design_invalid_Sigma_type():
    """Test TypeError for invalid Sigma type in simulate_rotated_design."""
    mock_cov = Mock(spec=CovarianceDesign)
    mock_cov.get_Sigma.return_value = "not_an_array"
    mock_cov.nfeatures.return_value = 2
    with pytest.raises(TypeError):
        simulate_rotated_design(cov=mock_cov, n=5)


def test_simulate_rotated_design_invalid_Sigma_dimensions():
    """Test ValueError for invalid Sigma dimensions in simulate_rotated_design."""
    mock_cov = Mock(spec=CovarianceDesign)
    mock_cov.get_Sigma.return_value = np.array([1, 2, 3])
    mock_cov.nfeatures.return_value = 3
    with pytest.raises(ValueError):
        simulate_rotated_design(cov=mock_cov, n=5)


def test_simulate_rotated_design_invalid_Sigma_shape():
    """Test ValueError for invalid Sigma shape in simulate_rotated_design."""
    mock_cov = Mock(spec=CovarianceDesign)
    mock_cov.get_Sigma.return_value = np.array([[1, 2], [3, 4]])
    mock_cov.nfeatures.return_value = 3
    with pytest.raises(ValueError):
        simulate_rotated_design(cov=mock_cov, n=5)


def test_simulate_rotated_design_rotated_measure_output_invalid():
    """Test TypeError for invalid output from rotated measure in simulate_rotated_design."""

    def faulty_measure(size):
        return "not_an_array"

    ar1 = AR1Design(p=2, rho=0.5)
    with pytest.raises(TypeError):
        simulate_rotated_design(cov=ar1, n=5, rotated_measure=faulty_measure)


def test_simulate_rotated_design_rotated_measure_shape_mismatch():
    """Test ValueError for shape mismatch in rotated measure output in simulate_rotated_design."""

    def faulty_measure(size):
        return np.random.normal(size=(size[0], size[1] + 1))  # Incorrect shape

    ar1 = AR1Design(p=2, rho=0.5)
    with pytest.raises(ValueError):
        simulate_rotated_design(cov=ar1, n=5, rotated_measure=faulty_measure)


def test_set_groups_with_GroupedFeatures_non_BlockCovarianceDesign():
    """Test setting groups with GroupedFeatures for non-BlockCovarianceDesign."""
    groups = GroupedFeatures(ps=[2, 3])
    ar1 = AR1Design(p=None, rho=0.5)
    set_groups(ar1, groups)
    assert ar1.p == 5


def test_set_groups_with_GroupedFeatures_BlockCovarianceDesign():
    """Test setting groups with GroupedFeatures for BlockCovarianceDesign."""
    groups = GroupedFeatures(ps=[2, 3])
    block1 = IdentityCovarianceDesign(p=None)
    block2 = UniformScalingCovarianceDesign(scaling=3.0, p=None)
    block_design = BlockCovarianceDesign(blocks=[block1, block2])
    set_groups(block_design, groups)
    assert block_design.groups == groups
    assert block1.p == 2
    assert block2.p == 3


def test_set_groups_with_GroupedFeatures_BlockCovarianceDesign_mismatch():
    """Test ValueError for mismatch between groups and blocks in BlockCovarianceDesign."""
    groups = GroupedFeatures(ps=[2, 3, 1])  # 3 groups
    block1 = IdentityCovarianceDesign(p=None)
    block2 = UniformScalingCovarianceDesign(scaling=3.0, p=None)
    block_design = BlockCovarianceDesign(blocks=[block1, block2])
    with pytest.raises(ValueError):
        set_groups(block_design, groups)


def test_set_groups_with_int():
    """Test setting groups with an integer value."""
    ar1 = AR1Design(p=None, rho=0.5)
    set_groups(ar1, 4)
    assert ar1.p == 4


def test_set_groups_with_invalid_groups_or_p_type():
    """Test TypeError for invalid groups or p type in set_groups."""
    ar1 = AR1Design(p=None, rho=0.5)
    with pytest.raises(TypeError):
        set_groups(ar1, "invalid_type")


def test_set_groups_with_GroupedFeatures_invalid_group_sizes():
    """Test ValueError for invalid group sizes in GroupedFeatures."""
    with pytest.raises(
        ValueError, match="All group sizes in ps must be positive integers"
    ):
        GroupedFeatures(ps=[2, -3])  # Negative group size


def test_set_groups_with_int_invalid_value():
    """Test ValueError for invalid integer value in set_groups."""
    ar1 = AR1Design(p=None, rho=0.5)
    with pytest.raises(ValueError):
        set_groups(ar1, 0)
    with pytest.raises(ValueError):
        set_groups(ar1, -1)


def test_set_groups_with_BlockCovarianceDesign_non_GroupedFeatures():
    """Test setting groups with an integer for BlockCovarianceDesign."""
    block1 = IdentityCovarianceDesign(p=None)
    block2 = UniformScalingCovarianceDesign(scaling=3.0, p=None)
    block_design = BlockCovarianceDesign(blocks=[block1, block2])

    set_groups(block_design, 5)

    assert block_design.groups is None
    assert block1.p == 5
    assert block2.p == 5
