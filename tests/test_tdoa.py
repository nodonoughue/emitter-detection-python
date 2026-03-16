import numpy as np
import pytest

from ewgeo.tdoa import model, perf, solvers, TDOAPassiveSurveillanceSystem
from ewgeo.utils.covariance import CovarianceMatrix
from ewgeo.utils import SearchSpace
from ewgeo.utils.unit_conversions import db_to_lin
from ewgeo.utils.geo import calc_range_diff


def equal_to_tolerance(x, y, tol=1e-6):
    if np.size(x) != np.size(y):
        return False
    return np.all(np.fabs(np.array(x) - np.array(y)) < tol)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# 3 sensors forming a triangle (2D), source near the centroid
X_SENSOR = np.array([[0.0,  1000.0, 500.0],
                     [0.0,     0.0, 866.0]])   # shape (2, 3)
X_SOURCE = np.array([500.0, 300.0])

# Default ref_idx=None → last sensor (index 2) as reference → 2 RDOA measurements

# Covariance already in range units (m^2), variance_is_toa=False throughout
COV_10M = CovarianceMatrix(100.0 * np.eye(3))   # 10 m std dev per sensor (3x3, resampled to pair space)

# Chan-Ho source: X_SOURCE sits on the x=500 symmetry axis between sensors 0 and 1,
# making zeta[0]=zeta[1] (degenerate for x-estimation).  Use an off-axis position.
X_SOURCE_CH = np.array([400.0, 200.0])

# Chan-Ho requires n_sensors >= n_dims + 2 for Stage 1 to be fully determined
# (3 RDOA equations, 3 unknowns [x, y, r_ref]).  Add a 4th sensor.
X_SENSOR_CH = np.array([[0.0, 1000.0, 500.0,    0.0],
                         [0.0,    0.0, 866.0, 1000.0]])   # 4th sensor is reference
COV_CH = CovarianceMatrix(np.eye(4))   # 4x4 per-sensor, resampled to 3x3 pair space


# Two-sensor geometry where source lies on the perpendicular bisector → RDOA = 0
X_SENSOR_2 = np.array([[-500.0, 500.0],
                        [   0.0,   0.0]])        # sensors at (±500, 0)
X_SOURCE_EQUIDIST = np.array([0.0, 500.0])       # equidistant from both


# ===========================================================================
# model.measurement
# ===========================================================================

def test_tdoa_measurement_shape_single_source():
    rdoa = model.measurement(X_SENSOR, X_SOURCE)
    assert rdoa.shape == (2,), f"Expected (2,), got {rdoa.shape}"


def test_tdoa_measurement_shape_multi_source():
    x_sources = np.column_stack([X_SOURCE, X_SOURCE + 100.0])
    rdoa = model.measurement(X_SENSOR, x_sources)
    assert rdoa.shape == (2, 2), f"Expected (2, 2), got {rdoa.shape}"


def test_tdoa_measurement_equidistant_source_is_zero():
    """Source on the perpendicular bisector of the sensor pair → RDOA = 0."""
    rdoa = model.measurement(X_SENSOR_2, X_SOURCE_EQUIDIST)
    assert equal_to_tolerance(rdoa, [0.0]), f"Expected 0, got {rdoa}"


def test_tdoa_measurement_antisymmetry():
    """Swapping test and reference sensor negates the RDOA measurement."""
    rdoa_0ref = model.measurement(X_SENSOR[:, :2], X_SOURCE, ref_idx=1)
    rdoa_1ref = model.measurement(X_SENSOR[:, :2], X_SOURCE, ref_idx=0)
    assert equal_to_tolerance(rdoa_0ref, -rdoa_1ref), \
        f"Antisymmetry failed: {rdoa_0ref} != -{rdoa_1ref}"


def test_tdoa_measurement_zero_bias_matches_no_bias():
    """An all-zeros bias vector produces the same result as no bias."""
    bias = np.zeros(3)
    rdoa_no_bias = model.measurement(X_SENSOR, X_SOURCE)
    rdoa_biased  = model.measurement(X_SENSOR, X_SOURCE, bias=bias)
    assert equal_to_tolerance(rdoa_biased, rdoa_no_bias)


def test_tdoa_measurement_bias_test_sensor_only():
    """Bias on test sensor 0 shifts only measurement[0] (r[0]-r[2]) by +10 m."""
    bias = np.array([10.0, 0.0, 0.0])
    rdoa_no_bias = model.measurement(X_SENSOR, X_SOURCE)
    rdoa_biased  = model.measurement(X_SENSOR, X_SOURCE, bias=bias)
    assert equal_to_tolerance(rdoa_biased[0] - rdoa_no_bias[0], 10.0)
    assert equal_to_tolerance(rdoa_biased[1], rdoa_no_bias[1])


def test_tdoa_measurement_bias_ref_sensor_only():
    """Bias on the reference sensor (index 2) shifts all measurements by -bias[2]."""
    bias = np.array([0.0, 0.0, 5.0])
    rdoa_no_bias = model.measurement(X_SENSOR, X_SOURCE)
    rdoa_biased  = model.measurement(X_SENSOR, X_SOURCE, bias=bias)
    assert equal_to_tolerance(rdoa_biased - rdoa_no_bias, np.array([-5.0, -5.0]))


def test_tdoa_measurement_bias_all_sensors():
    """With bias on every sensor: shift for pair (i, ref) = bias[i] - bias[ref]."""
    bias = np.array([3.0, 7.0, 2.0])
    rdoa_no_bias = model.measurement(X_SENSOR, X_SOURCE)
    rdoa_biased  = model.measurement(X_SENSOR, X_SOURCE, bias=bias)
    expected_shift = np.array([bias[0] - bias[2], bias[1] - bias[2]])
    assert equal_to_tolerance(rdoa_biased - rdoa_no_bias, expected_shift)


def test_tdoa_measurement_dim_mismatch_raises():
    x_source_3d = np.array([500.0, 300.0, 0.0])
    with pytest.raises(TypeError):
        model.measurement(X_SENSOR, x_source_3d)


# ===========================================================================
# model.jacobian
# ===========================================================================

def test_tdoa_jacobian_shape_single_source():
    J = model.jacobian(X_SENSOR, X_SOURCE)
    assert J.shape == (2, 2), f"Expected (2, 2), got {J.shape}"


def test_tdoa_jacobian_shape_multi_source():
    x_sources = np.column_stack([X_SOURCE, X_SOURCE + 100.0])
    J = model.jacobian(X_SENSOR, x_sources)
    assert J.shape == (2, 2, 2), f"Expected (2, 2, 2), got {J.shape}"


def test_tdoa_jacobian_finite_difference():
    """Central-difference numeric Jacobian should match the analytic one."""
    eps = 0.1   # metres
    n_dim = 2

    J_analytic = model.jacobian(X_SENSOR, X_SOURCE)   # (2, 2)
    J_numeric  = np.zeros_like(J_analytic)

    for d in range(n_dim):
        delta = np.zeros(n_dim)
        delta[d] = eps
        m_plus  = model.measurement(X_SENSOR, X_SOURCE + delta)
        m_minus = model.measurement(X_SENSOR, X_SOURCE - delta)
        J_numeric[d, :] = (m_plus - m_minus) / (2.0 * eps)

    assert equal_to_tolerance(J_analytic, J_numeric, tol=1e-5), \
        f"Analytic Jacobian differs from numeric:\n{J_analytic}\nvs\n{J_numeric}"


# ===========================================================================
# model.grad_sensor_pos
# ===========================================================================

def test_tdoa_grad_sensor_pos_shape():
    """Shape must be (n_dim * n_sensor, n_measurement) = (6, 2)."""
    gp = model.grad_sensor_pos(X_SENSOR, X_SOURCE)
    assert gp.shape == (6, 2), f"Expected (6, 2), got {gp.shape}"


def test_tdoa_grad_sensor_pos_finite_diff():
    """Analytical gradient should match numerical finite differences."""
    eps = 1e-3
    n_dim, n_sensor = X_SENSOR.shape
    n_measurement = n_sensor - 1  # RDOA pairs

    gp_analytic = model.grad_sensor_pos(X_SENSOR, X_SOURCE)
    gp_numeric = np.zeros((n_dim * n_sensor, n_measurement))

    for k in range(n_dim * n_sensor):
        sen = k // n_dim
        dim = k % n_dim
        x_plus  = X_SENSOR.copy(); x_plus[dim, sen]  += eps
        x_minus = X_SENSOR.copy(); x_minus[dim, sen] -= eps
        m_plus  = model.measurement(x_plus,  X_SOURCE)
        m_minus = model.measurement(x_minus, X_SOURCE)
        gp_numeric[k, :] = (m_plus - m_minus) / (2 * eps)

    assert np.allclose(gp_analytic, gp_numeric, atol=1e-4), \
        f"Max error: {np.max(np.abs(gp_analytic - gp_numeric)):.2e}"


# ===========================================================================
# perf.compute_crlb
# ===========================================================================

def test_tdoa_crlb_returns_covariance_matrix():
    crlb = perf.compute_crlb(X_SENSOR, X_SOURCE, COV_10M,
                             variance_is_toa=False, do_resample=True)
    assert isinstance(crlb, CovarianceMatrix)


def test_tdoa_crlb_is_positive_definite():
    crlb = perf.compute_crlb(X_SENSOR, X_SOURCE, COV_10M,
                             variance_is_toa=False, do_resample=True)
    eigenvalues = np.linalg.eigvalsh(crlb.cov)
    assert np.all(eigenvalues > 0), f"CRLB not positive definite: {eigenvalues}"


def test_tdoa_crlb_shape():
    crlb = perf.compute_crlb(X_SENSOR, X_SOURCE, COV_10M,
                             variance_is_toa=False, do_resample=True)
    assert crlb.cov.shape == (2, 2)


def test_tdoa_crlb_decreases_with_tighter_noise():
    """Halving range error (quarter variance) should reduce CRLB trace."""
    cov_10m = CovarianceMatrix(100.0 * np.eye(3))
    cov_1m  = CovarianceMatrix(1.0   * np.eye(3))
    crlb_10 = perf.compute_crlb(X_SENSOR, X_SOURCE, cov_10m,
                                variance_is_toa=False, do_resample=True)
    crlb_1  = perf.compute_crlb(X_SENSOR, X_SOURCE, cov_1m,
                                variance_is_toa=False, do_resample=True)
    assert np.trace(crlb_1.cov) < np.trace(crlb_10.cov)


def test_tdoa_crlb_multi_source_returns_list():
    x_sources = np.column_stack([X_SOURCE, X_SOURCE + np.array([300, 0])])
    result = perf.compute_crlb(X_SENSOR, x_sources, COV_10M,
                               variance_is_toa=False, do_resample=True)
    assert isinstance(result, list) and len(result) == 2
    for item in result:
        assert isinstance(item, CovarianceMatrix)


# ===========================================================================
# solvers.chan_ho
# ===========================================================================

def test_tdoa_chan_ho_output_shape():
    zeta = model.measurement(X_SENSOR_CH, X_SOURCE_CH)
    x_est = solvers.chan_ho(X_SENSOR_CH, zeta, COV_CH,
                           variance_is_toa=False, do_resample=True)
    assert x_est.shape == (2,), f"Expected (2,), got {x_est.shape}"


def test_tdoa_chan_ho_exact_measurements_near_source():
    """With exact (noiseless) inputs, Chan-Ho should recover the source position."""
    zeta = model.measurement(X_SENSOR_CH, X_SOURCE_CH)
    x_est = solvers.chan_ho(X_SENSOR_CH, zeta, COV_CH,
                           variance_is_toa=False, do_resample=True)
    error = np.linalg.norm(x_est - X_SOURCE_CH)
    assert error < 1.0, f"Chan-Ho estimate too far from source: {error:.4f} m"


# ===========================================================================
# TDOAPassiveSurveillanceSystem
# ===========================================================================

def test_tdoa_pss_measurement_matches_model():
    pss = TDOAPassiveSurveillanceSystem(x=X_SENSOR, cov=COV_10M, variance_is_toa=False, do_resample=True)
    assert equal_to_tolerance(pss.measurement(X_SOURCE),
                              model.measurement(X_SENSOR, X_SOURCE))


def test_tdoa_pss_jacobian_matches_model():
    pss = TDOAPassiveSurveillanceSystem(x=X_SENSOR, cov=COV_10M, variance_is_toa=False, do_resample=True)
    assert equal_to_tolerance(pss.jacobian(X_SOURCE),
                              model.jacobian(X_SENSOR, X_SOURCE))


def test_tdoa_pss_num_measurements():
    pss = TDOAPassiveSurveillanceSystem(x=X_SENSOR, cov=COV_10M, variance_is_toa=False, do_resample=True)
    assert pss.num_measurements == 2


def test_tdoa_pss_noisy_measurement_shape():
    pss = TDOAPassiveSurveillanceSystem(x=X_SENSOR, cov=COV_10M, variance_is_toa=False, do_resample=True)
    zeta = pss.noisy_measurement(X_SOURCE, num_samples=50)
    assert zeta.shape == (2, 50), f"Expected (2, 50), got {zeta.shape}"


def test_tdoa_pss_log_likelihood_peaks_at_source():
    """Noiseless measurement evaluated at the true source should have higher LL."""
    pss = TDOAPassiveSurveillanceSystem(x=X_SENSOR, cov=COV_10M, variance_is_toa=False, do_resample=True)
    zeta = pss.measurement(X_SOURCE)
    ll_true  = pss.log_likelihood(zeta=zeta, x_source=X_SOURCE)
    ll_wrong = pss.log_likelihood(zeta=zeta, x_source=X_SOURCE + np.array([500.0, 500.0]))
    assert ll_true > ll_wrong


def test_tdoa_pss_chan_ho_method():
    pss = TDOAPassiveSurveillanceSystem(x=X_SENSOR_CH, cov=COV_CH, variance_is_toa=False, do_resample=True)
    zeta = pss.measurement(X_SOURCE_CH)
    x_est = pss.chan_ho(zeta)
    assert np.linalg.norm(x_est - X_SOURCE_CH) < 1.0


# ===========================================================================
# TDOAPassiveSurveillanceSystem iterative and grid solvers
# ===========================================================================

def test_tdoa_pss_least_square_near_source():
    """LS solver should converge close to the source given noiseless measurements."""
    pss = TDOAPassiveSurveillanceSystem(x=X_SENSOR, cov=COV_10M, variance_is_toa=False, do_resample=True)
    zeta = pss.measurement(X_SOURCE)
    x_est, _ = pss.least_square(zeta, x_init=X_SOURCE + np.array([50., 50.]))
    assert np.linalg.norm(x_est - X_SOURCE) < 1.0, \
        f"LS estimate error too large: {np.linalg.norm(x_est - X_SOURCE):.2f} m"


def test_tdoa_pss_gradient_descent_near_source():
    """GD solver should converge close to the source given noiseless measurements."""
    pss = TDOAPassiveSurveillanceSystem(x=X_SENSOR, cov=COV_10M, variance_is_toa=False, do_resample=True)
    zeta = pss.measurement(X_SOURCE)
    x_est, _ = pss.gradient_descent(zeta, x_init=X_SOURCE + np.array([50., 50.]))
    assert np.linalg.norm(x_est - X_SOURCE) < 1.0, \
        f"GD estimate error too large: {np.linalg.norm(x_est - X_SOURCE):.2f} m"


def test_tdoa_pss_max_likelihood_near_source():
    """ML grid search should locate the source within one grid-diagonal of the truth."""
    pss = TDOAPassiveSurveillanceSystem(x=X_SENSOR, cov=COV_10M, variance_is_toa=False, do_resample=True)
    zeta = pss.measurement(X_SOURCE)
    ss = SearchSpace(x_ctr=X_SOURCE, epsilon=10., max_offset=300.)
    x_est, _, _ = pss.max_likelihood(zeta, search_space=ss)
    assert np.linalg.norm(x_est - X_SOURCE) < 15., \
        f"ML estimate error too large: {np.linalg.norm(x_est - X_SOURCE):.2f} m"


# ===========================================================================
# model.grad_source
# ===========================================================================

def test_tdoa_grad_source_matches_jacobian():
    """grad_source is a wrapper for jacobian; results must be identical."""
    J = model.jacobian(X_SENSOR, X_SOURCE)
    G = model.grad_source(X_SENSOR, X_SOURCE)
    assert np.array_equal(G, J), "grad_source should return the same result as jacobian"


def test_tdoa_grad_source_shape():
    """Shape (n_dim, n_measurement) = (2, 2) for 3-sensor 2D geometry."""
    G = model.grad_source(X_SENSOR, X_SOURCE)
    assert G.shape == (2, 2), f"Expected (2, 2), got {G.shape}"


# ===========================================================================
# model.grad_bias
# ===========================================================================

def test_tdoa_grad_bias_shape():
    """Shape must be (n_sensor, n_measurement) = (3, 2)."""
    B = model.grad_bias(X_SENSOR, X_SOURCE)
    assert B.shape == (3, 2), f"Expected (3, 2), got {B.shape}"


def test_tdoa_grad_bias_structure():
    """Pair (i, ref): B[test, col]=+1, B[ref, col]=-1, all others=0."""
    # Default ref_idx=None → ref=2; pairs (0,2) and (1,2)
    B = model.grad_bias(X_SENSOR, X_SOURCE)
    # Column 0: test=0, ref=2
    assert B[0, 0] == 1,  "B[test=0, col=0] should be +1"
    assert B[2, 0] == -1, "B[ref=2,  col=0] should be -1"
    assert B[1, 0] == 0,  "B[1, 0] should be 0"
    # Column 1: test=1, ref=2
    assert B[1, 1] == 1,  "B[test=1, col=1] should be +1"
    assert B[2, 1] == -1, "B[ref=2,  col=1] should be -1"
    assert B[0, 1] == 0,  "B[0, 1] should be 0"


def test_tdoa_grad_bias_multi_source_shape():
    """For two sources the third axis should be 2."""
    x_sources = np.column_stack([X_SOURCE, X_SOURCE + 100.0])
    B = model.grad_bias(X_SENSOR, x_sources)
    assert B.shape == (3, 2, 2), f"Expected (3, 2, 2), got {B.shape}"


# ===========================================================================
# model.jacobian_uncertainty
# ===========================================================================

def test_tdoa_jacobian_uncertainty_no_flags_shape():
    """No optional flags → only grad_source → shape (2, 2)."""
    J = model.jacobian_uncertainty(X_SENSOR, X_SOURCE)
    assert J.shape == (2, 2), f"Expected (2, 2), got {J.shape}"


def test_tdoa_jacobian_uncertainty_do_bias_shape():
    """do_bias=True appends grad_bias (3, 2) → shape (5, 2)."""
    J = model.jacobian_uncertainty(X_SENSOR, X_SOURCE, do_bias=True)
    assert J.shape == (5, 2), f"Expected (5, 2), got {J.shape}"


def test_tdoa_jacobian_uncertainty_do_pos_error_shape():
    """do_pos_error=True appends grad_sensor_pos (6, 2) → shape (8, 2)."""
    J = model.jacobian_uncertainty(X_SENSOR, X_SOURCE, do_pos_error=True)
    assert J.shape == (8, 2), f"Expected (8, 2), got {J.shape}"


def test_tdoa_jacobian_uncertainty_both_flags_shape():
    """Both flags → (2 + 3 + 6, 2) = (11, 2)."""
    J = model.jacobian_uncertainty(X_SENSOR, X_SOURCE, do_bias=True, do_pos_error=True)
    assert J.shape == (11, 2), f"Expected (11, 2), got {J.shape}"


# ===========================================================================
# model.toa_error_peak_detection
# ===========================================================================

def test_toa_error_peak_detection_formula():
    """At 0 dB SNR (lin=1): error = 1/(2*1) = 0.5 s²."""
    result = model.toa_error_peak_detection(0.0)
    assert equal_to_tolerance(result, 0.5, tol=1e-10), \
        f"Expected 0.5 s², got {result}"


def test_toa_error_peak_detection_decreases_with_snr():
    """Higher SNR → lower timing error."""
    e_low  = model.toa_error_peak_detection(0.0)
    e_high = model.toa_error_peak_detection(10.0)
    assert e_high < e_low, "Timing error should decrease with SNR"


def test_toa_error_peak_detection_positive():
    assert model.toa_error_peak_detection(10.0) > 0


def test_toa_error_peak_detection_vectorized():
    snr_vec = np.array([0.0, 5.0, 10.0])
    result = model.toa_error_peak_detection(snr_vec)
    assert np.shape(result) == (3,), f"Expected shape (3,), got {np.shape(result)}"
    assert np.all(np.diff(result) < 0), "Error should decrease monotonically with SNR"


# ===========================================================================
# model.toa_error_cross_corr
# ===========================================================================

def test_toa_error_cross_corr_formula():
    """At 0 dB SNR, bw=1e6, T=1e-3, bw_rms=1e6: error = 1/(8π * 1e9)."""
    bw, T, bw_rms = 1e6, 1e-3, 1e6
    expected = 1.0 / (8 * np.pi * db_to_lin(0.0) * bw * T * bw_rms)
    result = model.toa_error_cross_corr(0.0, bw, T, bw_rms)
    assert equal_to_tolerance(result, expected, tol=1e-20), \
        f"Expected {expected:.3e} s², got {result:.3e}"


def test_toa_error_cross_corr_positive():
    assert model.toa_error_cross_corr(10.0, 1e6, 1e-3, 1e6) > 0


def test_toa_error_cross_corr_decreases_with_snr():
    e_low  = model.toa_error_cross_corr(0.0,  1e6, 1e-3, 1e6)
    e_high = model.toa_error_cross_corr(10.0, 1e6, 1e-3, 1e6)
    assert e_high < e_low


def test_toa_error_cross_corr_decreases_with_bandwidth():
    e_low  = model.toa_error_cross_corr(0.0, 1e6,  1e-3, 1e6)
    e_high = model.toa_error_cross_corr(0.0, 10e6, 1e-3, 1e6)
    assert e_high < e_low


def test_toa_error_cross_corr_vectorized():
    snr_vec = np.array([0.0, 5.0, 10.0])
    result = model.toa_error_cross_corr(snr_vec, 1e6, 1e-3, 1e6)
    assert np.shape(result) == (3,)
    assert np.all(np.diff(result) < 0)


# ===========================================================================
# model.generate_parameter_indices
# ===========================================================================

def test_generate_parameter_indices_with_bias():
    """2D, 3 sensors, do_bias=True: target=[0,1], bias=[2,3,4], sensor=[5..10]."""
    idx = model.generate_parameter_indices(X_SENSOR, do_bias=True)
    assert list(idx['target_pos']) == [0, 1]
    assert list(idx['bias'])       == [2, 3, 4]
    assert list(idx['sensor_pos']) == [5, 6, 7, 8, 9, 10]


def test_generate_parameter_indices_without_bias():
    """do_bias=False: bias=None, sensor_pos starts at n_dim."""
    idx = model.generate_parameter_indices(X_SENSOR, do_bias=False)
    assert list(idx['target_pos']) == [0, 1]
    assert idx['bias'] is None
    assert list(idx['sensor_pos']) == [2, 3, 4, 5, 6, 7]


def test_generate_parameter_indices_no_overlap():
    """target_pos, bias, and sensor_pos index ranges must not overlap."""
    idx = model.generate_parameter_indices(X_SENSOR, do_bias=True)
    all_indices = (list(idx['target_pos']) + list(idx['bias']) + list(idx['sensor_pos']))
    assert len(all_indices) == len(set(all_indices)), "Index ranges must not overlap"


# ===========================================================================
# model.error
# ===========================================================================

def test_tdoa_error_minimum_near_source():
    """Error surface minimum should be close to the true source position."""
    x_max = np.array([1000.0, 1000.0])
    num_pts = 11
    epsilon = model.error(X_SENSOR, COV_10M, X_SOURCE, x_max, num_pts,
                          variance_is_toa=False, do_resample=True)

    # Reconstruct the grid to find where the minimum lives
    x_vec = x_max[0] * np.linspace(-1, 1, num_pts)
    y_vec = x_max[1] * np.linspace(-1, 1, num_pts)
    xx, yy = np.meshgrid(x_vec, y_vec)
    idx_min = np.argmin(epsilon)
    x_min = np.array([xx.flatten()[idx_min], yy.flatten()[idx_min]])

    dist = np.linalg.norm(x_min - X_SOURCE)
    # Grid spacing is 200 m; allow 2 cells of error
    assert dist < 400.0, f"Error minimum at {x_min}, source at {X_SOURCE}, dist={dist:.1f} m"


def test_tdoa_error_output_length():
    """Output should have num_pts² elements."""
    x_max = np.array([1000.0, 1000.0])
    num_pts = 11
    epsilon = model.error(X_SENSOR, COV_10M, X_SOURCE, x_max, num_pts,
                          variance_is_toa=False, do_resample=True)
    assert np.size(epsilon) == num_pts ** 2, \
        f"Expected {num_pts**2} elements, got {np.size(epsilon)}"


# ===========================================================================
# model.draw_isochrone
# ===========================================================================

def test_draw_isochrone_output_shape():
    """Output arrays should each have 2*num_pts - 1 points."""
    x_ref  = np.array([0.0,    0.0])
    x_test = np.array([1000.0, 0.0])
    num_pts = 5
    x_iso, y_iso = model.draw_isochrone(x_ref, x_test, range_diff=200.0,
                                        num_pts=num_pts, max_ortho=500.0)
    expected_len = 2 * num_pts - 1
    assert len(x_iso) == expected_len, f"Expected {expected_len} points, got {len(x_iso)}"
    assert len(y_iso) == expected_len


def test_draw_isochrone_midpoint_satisfies_range_diff():
    """The apex of the isochrone (v=0 point) should satisfy the range difference."""
    x_ref  = np.array([0.0,    0.0])
    x_test = np.array([1000.0, 0.0])
    range_diff = 200.0
    num_pts = 5
    x_iso, y_iso = model.draw_isochrone(x_ref, x_test, range_diff=range_diff,
                                        num_pts=num_pts, max_ortho=500.0)

    # The v=0 point is at index num_pts-1 in the output (after prepending num_pts-1 mirror points)
    midpt = np.array([[x_iso[num_pts - 1]], [y_iso[num_pts - 1]]])
    rdiff = float(calc_range_diff(midpt, x_ref[:, np.newaxis], x_test[:, np.newaxis]))
    assert abs(rdiff - range_diff) < 1.0, \
        f"Expected range diff {range_diff:.1f} m at apex, got {rdiff:.2f} m"
