import numpy as np
import pytest

from ewgeo.utils.utils import (
    parse_reference_sensor,
    resample_covariance_matrix,
    modulo2pi,
    remove_outliers,
    ensure_iterable,
    atleast_nd_trailing,
    compute_sample_mean,
    compute_sample_mean_update,
    broadcast_backwards,
    sinc_derivative,
    make_taper,
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


# ===========================================================================
# compute_sample_mean
# ===========================================================================

def test_compute_sample_mean_shape():
    rng = np.random.default_rng(0)
    zeta = rng.standard_normal((3, 10))
    zeta_mean, zeta_mean_full = compute_sample_mean(zeta)
    assert zeta_mean.shape == (3,)
    assert zeta_mean_full.shape == (3, 10)


def test_compute_sample_mean_final_equals_last_column():
    rng = np.random.default_rng(1)
    zeta = rng.standard_normal((2, 20))
    zeta_mean, zeta_mean_full = compute_sample_mean(zeta)
    assert np.allclose(zeta_mean, zeta_mean_full[:, -1])


def test_compute_sample_mean_known_constant():
    """For constant rows the mean must equal that constant."""
    zeta = np.array([[3.0] * 5, [7.0] * 5])
    zeta_mean, _ = compute_sample_mean(zeta)
    assert np.allclose(zeta_mean, [3.0, 7.0])


def test_compute_sample_mean_first_column_equals_first_sample():
    zeta = np.array([[1., 2., 3., 4.], [5., 6., 7., 8.]])
    _, zeta_mean_full = compute_sample_mean(zeta)
    assert np.allclose(zeta_mean_full[:, 0], zeta[:, 0])


def test_compute_sample_mean_converges_to_true_mean():
    rng = np.random.default_rng(2)
    true_mean = np.array([5.0, -2.0])
    zeta = true_mean[:, np.newaxis] + rng.standard_normal((2, 2000))
    zeta_mean, _ = compute_sample_mean(zeta)
    assert np.allclose(zeta_mean, true_mean, atol=0.1)


# ===========================================================================
# compute_sample_mean_update
# ===========================================================================

def test_compute_sample_mean_update_matches_batch():
    """Incremental update must agree with the batch mean over all samples."""
    rng = np.random.default_rng(3)
    zeta1 = rng.standard_normal((2, 10))
    zeta2 = rng.standard_normal((2, 5))
    mean1, _ = compute_sample_mean(zeta1)
    mean_update, _ = compute_sample_mean_update(zeta2, mean1, 10)
    all_samples = np.concatenate([zeta1, zeta2], axis=1)
    mean_batch, _ = compute_sample_mean(all_samples)
    assert np.allclose(mean_update, mean_batch, atol=1e-10)


def test_compute_sample_mean_update_count():
    zeta = np.ones((2, 5))
    mean_prev = np.array([1.0, 1.0])
    _, n_new = compute_sample_mean_update(zeta, mean_prev, 10)
    assert n_new == 15


def test_compute_sample_mean_update_single_new_sample():
    """Adding one sample by update must equal batch mean over all."""
    rng = np.random.default_rng(4)
    zeta_prev = rng.standard_normal((3, 9))
    new_sample = rng.standard_normal((3, 1))
    mean_prev, _ = compute_sample_mean(zeta_prev)
    mean_update, n = compute_sample_mean_update(new_sample, mean_prev, 9)
    all_data = np.concatenate([zeta_prev, new_sample], axis=1)
    mean_batch, _ = compute_sample_mean(all_data)
    assert n == 10
    assert np.allclose(mean_update, mean_batch, atol=1e-10)


# ===========================================================================
# broadcast_backwards
# ===========================================================================

def test_broadcast_backwards_pads_trailing():
    """Shorter array should get singleton trailing dimensions appended."""
    a = np.ones((3,))    # (3,)
    b = np.ones((3, 4))  # (3, 4)
    out, _ = broadcast_backwards([a, b])
    assert out[0].shape == (3, 1)
    assert out[1].shape == (3, 4)


def test_broadcast_backwards_out_shape():
    a = np.ones((3,))
    b = np.ones((3, 4))
    _, out_shp = broadcast_backwards([a, b])
    assert out_shp == (3, 4)


def test_broadcast_backwards_do_broadcast_expands():
    """With do_broadcast=True both arrays should be fully broadcast."""
    a = np.ones((3, 1))
    b = np.ones((1, 4))
    out, _ = broadcast_backwards([a, b], do_broadcast=True)
    assert out[0].shape == (3, 4)
    assert out[1].shape == (3, 4)


def test_broadcast_backwards_same_shape_unchanged():
    a = np.ones((3, 4))
    b = np.ones((3, 4))
    out, out_shp = broadcast_backwards([a, b])
    assert out[0].shape == (3, 4)
    assert out[1].shape == (3, 4)
    assert out_shp == (3, 4)


def test_broadcast_backwards_start_dim_preserves_leading():
    """Leading dims before start_dim must be left untouched."""
    a = np.ones((2, 3))      # (batch, meas)
    b = np.ones((2, 3, 4))   # (batch, meas, sources)
    out, out_shp = broadcast_backwards([a, b], start_dim=2)
    assert out[0].shape == (2, 3, 1)   # singleton appended at trailing
    assert out_shp == (4,)             # broadcast shape starts at dim 2


# ===========================================================================
# sinc_derivative
# ===========================================================================

def test_sinc_derivative_at_zero():
    assert equal_to_tolerance(sinc_derivative(0.0), 0.0)


def test_sinc_derivative_finite_difference():
    """Analytic sinc_derivative should match numerical finite difference of sin(x)/x."""
    x = np.array([0.5, 1.0, 2.0, 3.0])
    dx = 1e-7
    def sinc_rad(xx):
        return np.sin(xx) / xx
    fd = (sinc_rad(x + dx) - sinc_rad(x - dx)) / (2 * dx)
    analytic = sinc_derivative(x)
    assert np.allclose(analytic, fd, atol=1e-6)


def test_sinc_derivative_vectorized():
    x = np.linspace(-3, 3, 61)
    result = sinc_derivative(x)
    assert result.shape == x.shape


def test_sinc_derivative_negative_input():
    """sinc_derivative should be defined (and finite) for negative x."""
    result = sinc_derivative(-1.5)
    assert np.isfinite(result)


# ===========================================================================
# make_taper
# ===========================================================================

def test_make_taper_uniform_all_ones():
    w, _ = make_taper(10, 'uniform')
    assert np.allclose(w, 1.0)


def test_make_taper_uniform_snr_loss_zero():
    _, snr_loss = make_taper(16, 'uniform')
    assert np.isclose(snr_loss, 0.0)


def test_make_taper_hann_shape():
    w, _ = make_taper(32, 'hann')
    assert w.shape == (32,)


def test_make_taper_hann_peak_is_one():
    w, _ = make_taper(32, 'hann')
    assert np.isclose(np.max(np.fabs(w)), 1.0)


def test_make_taper_hann_snr_loss_negative():
    """Non-uniform tapers should incur SNR loss (< 0 dB)."""
    _, snr_loss = make_taper(32, 'hann')
    assert snr_loss < 0.0


def test_make_taper_hamming_length():
    n = 25
    w, _ = make_taper(n, 'hamming')
    assert len(w) == n


def test_make_taper_invalid_type_raises():
    with pytest.raises(KeyError):
        make_taper(16, 'triangular')
