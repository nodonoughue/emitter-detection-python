import numpy as np
import pytest

from ewgeo.utils.utils import (
    parse_reference_sensor,
    resample_covariance_matrix,
    modulo2pi,
    remove_outliers,
    ensure_iterable,
    atleast_nd_trailing,
)


def equal_to_tolerance(x, y, tol=1e-9):
    if np.size(x) != np.size(y):
        return False
    return np.all(np.fabs(np.array(x, dtype=float) - np.array(y, dtype=float)) < tol)


# ===========================================================================
# parse_reference_sensor
# ===========================================================================

def test_parse_ref_none_uses_last_sensor():
    test_idx, ref_idx = parse_reference_sensor(None, num_sensors=4)
    assert np.array_equal(test_idx, [0, 1, 2])
    assert np.array_equal(ref_idx, [3, 3, 3])


def test_parse_ref_none_two_sensors():
    test_idx, ref_idx = parse_reference_sensor(None, num_sensors=2)
    assert np.array_equal(test_idx, [0])
    assert np.array_equal(ref_idx, [1])


def test_parse_ref_full_generates_all_pairs():
    test_idx, ref_idx = parse_reference_sensor('full', num_sensors=3)
    # Expected pairs: (0,1), (0,2), (1,2)
    expected_test = np.array([0, 0, 1])
    expected_ref  = np.array([1, 2, 2])
    assert np.array_equal(np.sort(test_idx), expected_test)
    assert np.array_equal(np.sort(ref_idx), expected_ref)


def test_parse_ref_scalar_uses_named_sensor():
    test_idx, ref_idx = parse_reference_sensor(1, num_sensors=3)
    # All sensors except 1 become test sensors, ref=1 everywhere
    assert np.array_equal(np.sort(test_idx), [0, 2])
    assert np.all(ref_idx == 1)


def test_parse_ref_scalar_first_sensor():
    test_idx, ref_idx = parse_reference_sensor(0, num_sensors=3)
    assert np.array_equal(np.sort(test_idx), [1, 2])
    assert np.all(ref_idx == 0)


def test_parse_ref_explicit_array():
    pairs = np.array([[0, 1], [2, 2]])  # test=[0,1], ref=[2,2]
    test_idx, ref_idx = parse_reference_sensor(pairs, num_sensors=3)
    assert np.array_equal(test_idx, [0, 1])
    assert np.array_equal(ref_idx, [2, 2])


def test_parse_ref_invalid_num_sensors_raises():
    with pytest.raises(ValueError):
        parse_reference_sensor(None, num_sensors=0)


def test_parse_ref_invalid_scalar_out_of_range_raises():
    with pytest.raises(ValueError):
        parse_reference_sensor(5, num_sensors=3)


def test_parse_ref_invalid_string_raises():
    with pytest.raises(ValueError):
        parse_reference_sensor('half', num_sensors=3)


def test_parse_ref_bad_array_shape_raises():
    with pytest.raises(ValueError):
        parse_reference_sensor(np.array([0, 1, 2]), num_sensors=3)  # 1-D, not 2xN


def test_parse_ref_returns_int64_dtype():
    test_idx, ref_idx = parse_reference_sensor(None, num_sensors=3)
    assert test_idx.dtype == np.int64
    assert ref_idx.dtype == np.int64


# ===========================================================================
# resample_covariance_matrix
# ===========================================================================

def test_resample_cov_diagonal_common_ref():
    """For a diagonal cov with common reference, off-diag of output = sigma^2."""
    sigma_sq = 4.0
    cov = sigma_sq * np.eye(3)
    test_idx = np.array([0, 1])
    ref_idx  = np.array([2, 2])
    cov_out = resample_covariance_matrix(cov, test_idx, ref_idx)

    # C[i,i] = sigma^2 + sigma^2 = 2*sigma^2
    # C[i,j] = sigma^2  (shared reference)
    expected = sigma_sq * np.array([[2., 1.], [1., 2.]])
    assert equal_to_tolerance(cov_out, expected)


def test_resample_cov_shape():
    cov = np.eye(4)
    test_idx = np.array([0, 1, 2])
    ref_idx  = np.array([3, 3, 3])
    cov_out = resample_covariance_matrix(cov, test_idx, ref_idx)
    assert cov_out.shape == (3, 3)


def test_resample_cov_identity_no_ref_diagonal():
    """Single-sensor difference (no common ref) with identity cov gives 2 on diag."""
    cov = np.eye(2)
    test_idx = np.array([0])
    ref_idx  = np.array([1])
    cov_out = resample_covariance_matrix(cov, test_idx, ref_idx)
    # C_out[0,0] = cov[0,0] + cov[1,1] - 0 - 0 = 2
    assert equal_to_tolerance(cov_out.ravel(), [2.0])


# ===========================================================================
# modulo2pi
# ===========================================================================

def test_modulo2pi_zero():
    assert equal_to_tolerance(modulo2pi(0.0), 0.0)


def test_modulo2pi_pi_wraps():
    # pi maps to -pi (edge case due to modulo)
    assert equal_to_tolerance(np.fabs(modulo2pi(np.pi)), np.pi)


def test_modulo2pi_two_pi_is_zero():
    assert equal_to_tolerance(modulo2pi(2 * np.pi), 0.0)


def test_modulo2pi_half_pi():
    assert equal_to_tolerance(modulo2pi(np.pi / 2), np.pi / 2)


def test_modulo2pi_three_halves_pi():
    # 3π/2 → -π/2
    assert equal_to_tolerance(modulo2pi(3 * np.pi / 2), -np.pi / 2)


def test_modulo2pi_large_positive():
    # 7π/2 = 3π + π/2 → should be -π/2
    assert equal_to_tolerance(modulo2pi(7 * np.pi / 2), -np.pi / 2)


def test_modulo2pi_result_in_range():
    x = np.linspace(-10, 10, 1000)
    result = modulo2pi(x)
    assert np.all(result >= -np.pi) and np.all(result <= np.pi)


def test_modulo2pi_vectorized():
    x = np.array([0., np.pi / 2, -np.pi / 2])
    result = modulo2pi(x)
    assert equal_to_tolerance(result, [0., np.pi / 2, -np.pi / 2])


# ===========================================================================
# remove_outliers
# ===========================================================================

def test_remove_outliers_clean_data_unchanged():
    data = np.array([1., 2., 3., 4., 5.])
    result = remove_outliers(data)
    assert result.shape[0] == data.shape[0]


def test_remove_outliers_removes_spike():
    data = np.array([1., 2., 2., 2., 2., 2., 100.])
    result = remove_outliers(data)
    assert result.shape[0] < data.shape[0]
    assert 100. not in result


def test_remove_outliers_output_type():
    data = np.array([1., 2., 3.])
    result = remove_outliers(data)
    assert isinstance(result, np.ndarray)


# ===========================================================================
# ensure_iterable
# ===========================================================================

def test_ensure_iterable_scalar_wrapped():
    result = ensure_iterable(5)
    assert list(result) == [5]


def test_ensure_iterable_list_unchanged():
    x = [1, 2, 3]
    result = ensure_iterable(x)
    assert list(result) == x


def test_ensure_iterable_array_unchanged():
    x = np.array([1, 2, 3])
    result = ensure_iterable(x)
    assert np.array_equal(result, x)


def test_ensure_iterable_string_unchanged():
    # strings are iterable, so returned as-is
    result = ensure_iterable("hello")
    assert result == "hello"


# ===========================================================================
# atleast_nd_trailing
# ===========================================================================

def test_atleast_nd_scalar_to_2d():
    x = np.array(3.0)
    result = atleast_nd_trailing(x, 2)
    assert result.ndim == 2
    assert result.shape == (1, 1)


def test_atleast_nd_1d_to_2d():
    x = np.array([1., 2., 3.])
    result = atleast_nd_trailing(x, 2)
    assert result.ndim == 2
    assert result.shape == (3, 1)


def test_atleast_nd_already_sufficient():
    x = np.ones((3, 4))
    result = atleast_nd_trailing(x, 2)
    assert result.shape == (3, 4)


def test_atleast_nd_adds_multiple_dims():
    x = np.array([1., 2.])
    result = atleast_nd_trailing(x, 4)
    assert result.ndim == 4
    assert result.shape == (2, 1, 1, 1)
