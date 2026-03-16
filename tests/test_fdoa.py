import numpy as np
import pytest

from ewgeo.fdoa import model, perf, FDOAPassiveSurveillanceSystem
from ewgeo.utils.covariance import CovarianceMatrix
from ewgeo.utils import SearchSpace


def equal_to_tolerance(x, y, tol=1e-6):
    if np.size(x) != np.size(y):
        return False
    return np.all(np.fabs(np.array(x) - np.array(y)) < tol)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# 3 sensors forming a triangle (same geometry as test_triang / test_tdoa)
X_SENSOR = np.array([[0.0,  1000.0, 500.0],
                     [0.0,     0.0, 866.0]])   # shape (2, 3)

# Sensor velocities — intentionally varied so all range rates are distinct
V_SENSOR = np.array([[100.0, -100.0,   0.0],
                     [  0.0,    0.0, 100.0]])  # shape (2, 3)

X_SOURCE = np.array([500.0, 300.0])            # stationary source

# Default ref_idx=None → last sensor (index 2) is reference → 2 RRDOA measurements

# Covariance in (m/s)^2 units; already in range-rate domain
COV_1MS = CovarianceMatrix(1.0 * np.eye(3))    # 1 m/s std dev per sensor (3x3, resampled to pair space)

# Two-sensor symmetric geometry where both sensors have the same velocity
# component along the source LOS → RRDOA = 0
X_SENSOR_2 = np.array([[-500.0, 500.0],
                        [   0.0,   0.0]])       # sensors at (±500, 0)
V_SENSOR_2 = np.array([[   0.0,   0.0],
                        [ 100.0, 100.0]])       # both moving at (0, 100) m/s
X_SOURCE_2 = np.array([0.0, 500.0])            # source at (0, 500)


# ===========================================================================
# model.measurement
# ===========================================================================

def test_fdoa_measurement_shape_single_source():
    rdoa = model.measurement(X_SENSOR, X_SOURCE, v_sensor=V_SENSOR)
    assert rdoa.shape == (2,), f"Expected (2,), got {rdoa.shape}"


def test_fdoa_measurement_shape_multi_source():
    x_sources = np.column_stack([X_SOURCE, X_SOURCE + 100.0])
    rdoa = model.measurement(X_SENSOR, x_sources, v_sensor=V_SENSOR)
    assert rdoa.shape == (2, 2), f"Expected (2, 2), got {rdoa.shape}"


def test_fdoa_measurement_equal_velocity_is_zero():
    """
    When both sensors move at the same velocity and the source is stationary,
    both sensors have the same range-rate projection → RRDOA = 0.
    """
    rdoa = model.measurement(X_SENSOR_2, X_SOURCE_2, v_sensor=V_SENSOR_2)
    assert equal_to_tolerance(rdoa, [0.0], tol=1e-9), f"Expected 0, got {rdoa}"


def test_fdoa_measurement_antisymmetry():
    """Swapping test and reference sensor negates the RRDOA measurement."""
    x2 = X_SENSOR[:, :2]
    v2 = V_SENSOR[:, :2]
    rdoa_0ref = model.measurement(x2, X_SOURCE, v_sensor=v2, ref_idx=1)
    rdoa_1ref = model.measurement(x2, X_SOURCE, v_sensor=v2, ref_idx=0)
    assert equal_to_tolerance(rdoa_0ref, -rdoa_1ref), \
        f"Antisymmetry failed: {rdoa_0ref} != -{rdoa_1ref}"


def test_fdoa_measurement_zero_bias_matches_no_bias():
    bias = np.zeros(3)
    rdoa_no_bias = model.measurement(X_SENSOR, X_SOURCE, v_sensor=V_SENSOR)
    rdoa_biased  = model.measurement(X_SENSOR, X_SOURCE, v_sensor=V_SENSOR, bias=bias)
    assert equal_to_tolerance(rdoa_biased, rdoa_no_bias)


def test_fdoa_measurement_bias_test_sensor_only():
    """Bias on test sensor 0 shifts only measurement[0] by that amount."""
    bias = np.array([2.0, 0.0, 0.0])
    rdoa_no_bias = model.measurement(X_SENSOR, X_SOURCE, v_sensor=V_SENSOR)
    rdoa_biased  = model.measurement(X_SENSOR, X_SOURCE, v_sensor=V_SENSOR, bias=bias)
    assert equal_to_tolerance(rdoa_biased[0] - rdoa_no_bias[0], 2.0)
    assert equal_to_tolerance(rdoa_biased[1], rdoa_no_bias[1])


def test_fdoa_measurement_bias_ref_sensor_only():
    """Bias on the reference sensor (index 2) shifts all measurements by -bias[2]."""
    bias = np.array([0.0, 0.0, 1.5])
    rdoa_no_bias = model.measurement(X_SENSOR, X_SOURCE, v_sensor=V_SENSOR)
    rdoa_biased  = model.measurement(X_SENSOR, X_SOURCE, v_sensor=V_SENSOR, bias=bias)
    assert equal_to_tolerance(rdoa_biased - rdoa_no_bias, np.array([-1.5, -1.5]))


def test_fdoa_measurement_bias_all_sensors():
    """Bias on every sensor: shift for pair (i, ref) = bias[i] - bias[ref]."""
    bias = np.array([1.0, 3.0, 0.5])
    rdoa_no_bias = model.measurement(X_SENSOR, X_SOURCE, v_sensor=V_SENSOR)
    rdoa_biased  = model.measurement(X_SENSOR, X_SOURCE, v_sensor=V_SENSOR, bias=bias)
    expected_shift = np.array([bias[0] - bias[2], bias[1] - bias[2]])
    assert equal_to_tolerance(rdoa_biased - rdoa_no_bias, expected_shift)


def test_fdoa_measurement_requires_velocity():
    """Calling measurement without any velocity should raise ValueError."""
    with pytest.raises(ValueError):
        model.measurement(X_SENSOR, X_SOURCE, v_sensor=None, v_source=None)


# ===========================================================================
# model.jacobian
# ===========================================================================

def test_fdoa_jacobian_shape_single_source():
    """FDOA Jacobian combines position and velocity rows: shape (2*n_dim, n_pair)."""
    J = model.jacobian(X_SENSOR, X_SOURCE, v_sensor=V_SENSOR)
    n_dim = X_SENSOR.shape[0]
    n_pair = X_SENSOR.shape[1] - 1
    assert J.shape == (2 * n_dim, n_pair), f"Expected ({2*n_dim}, {n_pair}), got {J.shape}"


def test_fdoa_jacobian_shape_multi_source():
    x_sources = np.column_stack([X_SOURCE, X_SOURCE + 100.0])
    J = model.jacobian(X_SENSOR, x_sources, v_sensor=V_SENSOR)
    n_dim = X_SENSOR.shape[0]
    n_pair = X_SENSOR.shape[1] - 1
    assert J.shape == (2 * n_dim, n_pair, 2), \
        f"Expected ({2*n_dim}, {n_pair}, 2), got {J.shape}"


def test_fdoa_jacobian_finite_difference():
    """
    Position rows of the analytic Jacobian should match a central-difference
    numeric Jacobian computed by perturbing the source position.
    """
    eps = 0.1   # metres
    n_dim = X_SENSOR.shape[0]

    J_analytic = model.jacobian(X_SENSOR, X_SOURCE, v_sensor=V_SENSOR)
    J_pos_analytic = J_analytic[:n_dim, :]   # (2, 2): position rows only

    J_numeric = np.zeros_like(J_pos_analytic)
    for d in range(n_dim):
        delta = np.zeros(n_dim)
        delta[d] = eps
        m_plus  = model.measurement(X_SENSOR, X_SOURCE + delta, v_sensor=V_SENSOR)
        m_minus = model.measurement(X_SENSOR, X_SOURCE - delta, v_sensor=V_SENSOR)
        J_numeric[d, :] = (m_plus - m_minus) / (2.0 * eps)

    assert equal_to_tolerance(J_pos_analytic, J_numeric, tol=1e-5), \
        f"Position Jacobian mismatch:\n{J_pos_analytic}\nvs\n{J_numeric}"


# ===========================================================================
# perf.compute_crlb
# ===========================================================================

def test_fdoa_crlb_returns_covariance_matrix():
    crlb = perf.compute_crlb(X_SENSOR, V_SENSOR, X_SOURCE, COV_1MS,
                             do_resample=True)
    assert isinstance(crlb, CovarianceMatrix)


def test_fdoa_crlb_is_positive_definite():
    crlb = perf.compute_crlb(X_SENSOR, V_SENSOR, X_SOURCE, COV_1MS,
                             do_resample=True)
    eigenvalues = np.linalg.eigvalsh(crlb.cov)
    assert np.all(eigenvalues > 0), f"CRLB not positive definite: {eigenvalues}"


def test_fdoa_crlb_shape():
    crlb = perf.compute_crlb(X_SENSOR, V_SENSOR, X_SOURCE, COV_1MS,
                             do_resample=True)
    assert crlb.cov.shape == (2, 2)


def test_fdoa_crlb_decreases_with_tighter_noise():
    cov_1ms  = CovarianceMatrix(1.0   * np.eye(3))
    cov_01ms = CovarianceMatrix(0.01  * np.eye(3))
    crlb_1   = perf.compute_crlb(X_SENSOR, V_SENSOR, X_SOURCE, cov_1ms,  do_resample=True)
    crlb_01  = perf.compute_crlb(X_SENSOR, V_SENSOR, X_SOURCE, cov_01ms, do_resample=True)
    assert np.trace(crlb_01.cov) < np.trace(crlb_1.cov)


def test_fdoa_crlb_multi_source_returns_list():
    x_sources = np.column_stack([X_SOURCE, X_SOURCE + np.array([300, 0])])
    result = perf.compute_crlb(X_SENSOR, V_SENSOR, x_sources, COV_1MS,
                               do_resample=True)
    assert isinstance(result, list) and len(result) == 2
    for item in result:
        assert isinstance(item, CovarianceMatrix)


# ===========================================================================
# FDOAPassiveSurveillanceSystem
# ===========================================================================

def test_fdoa_pss_measurement_matches_model():
    pss = FDOAPassiveSurveillanceSystem(x=X_SENSOR, vel=V_SENSOR, cov=COV_1MS, do_resample=True)
    assert equal_to_tolerance(pss.measurement(X_SOURCE),
                              model.measurement(X_SENSOR, X_SOURCE, v_sensor=V_SENSOR))


def test_fdoa_pss_jacobian_matches_model():
    pss = FDOAPassiveSurveillanceSystem(x=X_SENSOR, vel=V_SENSOR, cov=COV_1MS, do_resample=True)
    assert equal_to_tolerance(pss.jacobian(X_SOURCE),
                              model.jacobian(X_SENSOR, X_SOURCE, v_sensor=V_SENSOR))


def test_fdoa_pss_num_measurements():
    pss = FDOAPassiveSurveillanceSystem(x=X_SENSOR, vel=V_SENSOR, cov=COV_1MS, do_resample=True)
    assert pss.num_measurements == 2


def test_fdoa_pss_noisy_measurement_shape():
    pss = FDOAPassiveSurveillanceSystem(x=X_SENSOR, vel=V_SENSOR, cov=COV_1MS, do_resample=True)
    zeta = pss.noisy_measurement(X_SOURCE, num_samples=50)
    assert zeta.shape == (2, 50), f"Expected (2, 50), got {zeta.shape}"


def test_fdoa_pss_log_likelihood_peaks_at_source():
    """Noiseless measurement should yield higher LL at the true source."""
    pss = FDOAPassiveSurveillanceSystem(x=X_SENSOR, vel=V_SENSOR, cov=COV_1MS, do_resample=True)
    zeta = pss.measurement(X_SOURCE)
    ll_true  = pss.log_likelihood(zeta=zeta, x_source=X_SOURCE)
    ll_wrong = pss.log_likelihood(zeta=zeta, x_source=X_SOURCE + np.array([500.0, 500.0]))
    assert ll_true > ll_wrong


# ===========================================================================
# FDOAPassiveSurveillanceSystem iterative and grid solvers
# ===========================================================================

def test_fdoa_pss_least_square_near_source():
    """LS solver should converge close to the source given noiseless measurements."""
    pss = FDOAPassiveSurveillanceSystem(x=X_SENSOR, vel=V_SENSOR, cov=COV_1MS, do_resample=True)
    zeta = pss.measurement(X_SOURCE)
    x_est, _ = pss.least_square(zeta, x_init=X_SOURCE + np.array([50., 50.]))
    assert np.linalg.norm(x_est - X_SOURCE) < 1.0, \
        f"LS estimate error too large: {np.linalg.norm(x_est - X_SOURCE):.2f} m"


def test_fdoa_pss_gradient_descent_near_source():
    """GD solver should converge close to the source given noiseless measurements."""
    pss = FDOAPassiveSurveillanceSystem(x=X_SENSOR, vel=V_SENSOR, cov=COV_1MS, do_resample=True)
    zeta = pss.measurement(X_SOURCE)
    x_est, _ = pss.gradient_descent(zeta, x_init=X_SOURCE + np.array([50., 50.]))
    assert np.linalg.norm(x_est - X_SOURCE) < 1.0, \
        f"GD estimate error too large: {np.linalg.norm(x_est - X_SOURCE):.2f} m"


def test_fdoa_pss_max_likelihood_near_source():
    """ML grid search should locate the source within one grid-diagonal of the truth."""
    pss = FDOAPassiveSurveillanceSystem(x=X_SENSOR, vel=V_SENSOR, cov=COV_1MS, do_resample=True)
    zeta = pss.measurement(X_SOURCE)
    ss = SearchSpace(x_ctr=X_SOURCE, epsilon=10., max_offset=300.)
    x_est, _, _ = pss.max_likelihood(zeta, search_space=ss)
    assert np.linalg.norm(x_est - X_SOURCE) < 15., \
        f"ML estimate error too large: {np.linalg.norm(x_est - X_SOURCE):.2f} m"


# ===========================================================================
# model.grad_source
# ===========================================================================

def test_fdoa_grad_source_matches_jacobian():
    """grad_source is a thin wrapper for jacobian; outputs must be identical."""
    j  = model.jacobian(X_SENSOR, X_SOURCE, v_sensor=V_SENSOR)
    gs = model.grad_source(X_SENSOR, X_SOURCE, v_sensor=V_SENSOR)
    assert np.allclose(gs, j)


# ===========================================================================
# model.grad_bias
# ===========================================================================

def test_fdoa_grad_bias_shape():
    """grad_bias must be (n_sensor, n_measurement) for a single source."""
    gb = model.grad_bias(X_SENSOR, X_SOURCE)
    # 3 sensors, 2 measurement pairs (default ref = last sensor)
    assert gb.shape == (3, 2), f"Expected (3, 2), got {gb.shape}"


def test_fdoa_grad_bias_values():
    """For default ref_idx: col 0 → test=0 (+1), ref=2 (−1); col 1 → test=1 (+1), ref=2 (−1)."""
    gb = model.grad_bias(X_SENSOR, X_SOURCE)
    # Column 0: sensor 0 is test (+1), sensor 2 is ref (−1)
    assert gb[0, 0] ==  1, f"Expected +1 at [0,0], got {gb[0,0]}"
    assert gb[2, 0] == -1, f"Expected −1 at [2,0], got {gb[2,0]}"
    assert gb[1, 0] ==  0, f"Expected  0 at [1,0], got {gb[1,0]}"
    # Column 1: sensor 1 is test (+1), sensor 2 is ref (−1)
    assert gb[1, 1] ==  1, f"Expected +1 at [1,1], got {gb[1,1]}"
    assert gb[2, 1] == -1, f"Expected −1 at [2,1], got {gb[2,1]}"
    assert gb[0, 1] ==  0, f"Expected  0 at [0,1], got {gb[0,1]}"


def test_fdoa_grad_bias_multi_source_shape():
    n_src = 4
    x_multi = np.random.default_rng(0).standard_normal((2, n_src))
    gb = model.grad_bias(X_SENSOR, x_multi)
    assert gb.shape == (3, 2, n_src), f"Expected (3, 2, {n_src}), got {gb.shape}"


# ===========================================================================
# model.grad_sensor_pos
# ===========================================================================

def test_fdoa_grad_sensor_pos_shape():
    """Shape must be (n_dim * n_sensor, n_measurement) = (6, 2)."""
    gp = model.grad_sensor_pos(X_SENSOR, X_SOURCE, v_sensor=V_SENSOR)
    assert gp.shape == (6, 2), f"Expected (6, 2), got {gp.shape}"


def test_fdoa_grad_sensor_pos_finite_diff():
    """Analytical gradient should match numerical finite differences."""
    eps = 1e-3
    n_dim, n_sensor = X_SENSOR.shape
    n_measurement = 2

    gp_analytic = model.grad_sensor_pos(X_SENSOR, X_SOURCE, v_sensor=V_SENSOR)
    gp_numeric = np.zeros((n_dim * n_sensor, n_measurement))

    for k in range(n_dim * n_sensor):
        sen = k // n_dim
        dim = k % n_dim
        x_plus  = X_SENSOR.copy(); x_plus[dim, sen]  += eps
        x_minus = X_SENSOR.copy(); x_minus[dim, sen] -= eps
        m_plus  = model.measurement(x_plus,  X_SOURCE, v_sensor=V_SENSOR)
        m_minus = model.measurement(x_minus, X_SOURCE, v_sensor=V_SENSOR)
        gp_numeric[k, :] = (m_plus - m_minus) / (2 * eps)

    assert np.allclose(gp_analytic, gp_numeric, atol=1e-4), \
        f"Max error: {np.max(np.abs(gp_analytic - gp_numeric)):.2e}"


# ===========================================================================
# model.grad_sensor_vel
# ===========================================================================

def test_fdoa_grad_sensor_vel_shape():
    """Shape must be (n_dim * n_sensor, n_measurement) = (6, 2)."""
    gv = model.grad_sensor_vel(X_SENSOR, X_SOURCE, v_sensor=V_SENSOR)
    assert gv.shape == (6, 2), f"Expected (6, 2), got {gv.shape}"


def test_fdoa_grad_sensor_vel_finite_diff():
    """Analytical gradient should match numerical finite differences w.r.t. sensor velocity."""
    eps = 1e-3
    n_dim, n_sensor = V_SENSOR.shape
    n_measurement = 2

    gv_analytic = model.grad_sensor_vel(X_SENSOR, X_SOURCE, v_sensor=V_SENSOR)
    gv_numeric = np.zeros((n_dim * n_sensor, n_measurement))

    for k in range(n_dim * n_sensor):
        sen = k // n_dim
        dim = k % n_dim
        v_plus  = V_SENSOR.copy(); v_plus[dim, sen]  += eps
        v_minus = V_SENSOR.copy(); v_minus[dim, sen] -= eps
        m_plus  = model.measurement(X_SENSOR, X_SOURCE, v_sensor=v_plus)
        m_minus = model.measurement(X_SENSOR, X_SOURCE, v_sensor=v_minus)
        gv_numeric[k, :] = (m_plus - m_minus) / (2 * eps)

    assert np.allclose(gv_analytic, gv_numeric, atol=1e-4), \
        f"Max error: {np.max(np.abs(gv_analytic - gv_numeric)):.2e}"


# ===========================================================================
# model.jacobian_uncertainty
# ===========================================================================

def test_fdoa_jacobian_uncertainty_no_flags_shape():
    """Without optional flags, result equals grad_source (2*n_dim rows): shape (4, 2).
    grad_source wraps jacobian which stacks pos and vel rows: (n_dim + n_dim, n_meas)."""
    j = model.jacobian_uncertainty(X_SENSOR, X_SOURCE, v_sensor=V_SENSOR)
    assert j.shape == (4, 2), f"Expected (4, 2), got {j.shape}"


def test_fdoa_jacobian_uncertainty_with_bias_shape():
    """do_bias=True appends grad_bias (n_sensor rows): shape (2*n_dim + n_sensor, n_meas) = (7, 2)."""
    j = model.jacobian_uncertainty(X_SENSOR, X_SOURCE, v_sensor=V_SENSOR, do_bias=True)
    assert j.shape == (7, 2), f"Expected (7, 2), got {j.shape}"


def test_fdoa_jacobian_uncertainty_with_pos_error_shape():
    """do_pos_error=True appends grad_sensor_pos (n_dim*n_sensor rows):
    shape (2*n_dim + n_dim*n_sensor, n_meas) = (10, 2)."""
    j = model.jacobian_uncertainty(X_SENSOR, X_SOURCE, v_sensor=V_SENSOR, do_pos_error=True)
    assert j.shape == (10, 2), f"Expected (10, 2), got {j.shape}"


def test_fdoa_jacobian_uncertainty_with_all_flags_shape():
    """Both flags: shape (2*n_dim + n_sensor + n_dim*n_sensor, n_meas) = (13, 2)."""
    j = model.jacobian_uncertainty(X_SENSOR, X_SOURCE, v_sensor=V_SENSOR,
                                   do_bias=True, do_pos_error=True)
    assert j.shape == (13, 2), f"Expected (13, 2), got {j.shape}"


# ===========================================================================
# model.error
# ===========================================================================

def test_fdoa_error_shape():
    """error() should return a (num_pts, num_pts) grid and matching x/y vectors."""
    num_pts = 11
    epsilon, x_vec, y_vec = model.error(X_SENSOR, COV_1MS, X_SOURCE,
                                         x_max=np.array([1000., 1000.]),
                                         num_pts=num_pts, v_sensor=V_SENSOR,
                                         do_resample=True)
    assert epsilon.shape == (num_pts, num_pts)
    assert x_vec.shape == (num_pts,)
    assert y_vec.shape == (num_pts,)


def test_fdoa_error_minimum_near_source():
    """The minimum of the error surface should lie close to the true source position."""
    num_pts = 51
    x_max = np.array([1000., 1000.])
    epsilon, x_vec, y_vec = model.error(X_SENSOR, COV_1MS, X_SOURCE,
                                         x_max=x_max, num_pts=num_pts,
                                         v_sensor=V_SENSOR, do_resample=True)
    idx = np.unravel_index(np.argmin(epsilon), epsilon.shape)
    x_min = np.array([x_vec[idx[1]], y_vec[idx[0]]])
    grid_spacing = 2 * x_max / (num_pts - 1)
    assert np.linalg.norm(x_min - X_SOURCE) < 3 * np.linalg.norm(grid_spacing), \
        f"Error minimum at {x_min}, expected near {X_SOURCE}"


# ===========================================================================
# perf.freq_crlb
# ===========================================================================

def test_fdoa_freq_crlb_positive():
    sigma = perf.freq_crlb(sample_time=1e-3, num_samples=100, snr_db=10.0)
    assert float(sigma) > 0


def test_fdoa_freq_crlb_decreases_with_snr():
    s_low  = perf.freq_crlb(sample_time=1e-3, num_samples=100, snr_db=0.0)
    s_high = perf.freq_crlb(sample_time=1e-3, num_samples=100, snr_db=20.0)
    assert float(s_high) < float(s_low)


def test_fdoa_freq_crlb_decreases_with_time():
    s_short = perf.freq_crlb(sample_time=1e-4, num_samples=100, snr_db=10.0)
    s_long  = perf.freq_crlb(sample_time=1e-2, num_samples=100, snr_db=10.0)
    assert float(s_long) < float(s_short)


def test_fdoa_freq_crlb_known_value():
    """σ = sqrt(3 / (π² T² M (M²-1) snr_lin)); at T=1, M=2, snr=0dB: σ = sqrt(1/(2π²))."""
    expected = np.sqrt(1.0 / (2.0 * np.pi**2))
    sigma = perf.freq_crlb(sample_time=1.0, num_samples=2, snr_db=0.0)
    assert np.isclose(float(sigma), expected, rtol=1e-9)


# ===========================================================================
# perf.freq_diff_crlb
# ===========================================================================

def test_fdoa_freq_diff_crlb_positive():
    sigma = perf.freq_diff_crlb(time_s=1e-3, bw_hz=1e6, snr_db=10.0)
    assert float(sigma) > 0


def test_fdoa_freq_diff_crlb_decreases_with_snr():
    s_low  = perf.freq_diff_crlb(time_s=1e-3, bw_hz=1e6, snr_db=0.0)
    s_high = perf.freq_diff_crlb(time_s=1e-3, bw_hz=1e6, snr_db=20.0)
    assert float(s_high) < float(s_low)


def test_fdoa_freq_diff_crlb_decreases_with_bandwidth():
    s_narrow = perf.freq_diff_crlb(time_s=1e-3, bw_hz=1e5, snr_db=10.0)
    s_wide   = perf.freq_diff_crlb(time_s=1e-3, bw_hz=1e7, snr_db=10.0)
    assert float(s_wide) < float(s_narrow)


def test_fdoa_freq_diff_crlb_known_value():
    """σ = sqrt(3 / (4π² T³ B snr_lin)); at T=1, B=1, snr=0dB: σ = sqrt(3)/(2π)."""
    expected = np.sqrt(3.0) / (2.0 * np.pi)
    sigma = perf.freq_diff_crlb(time_s=1.0, bw_hz=1.0, snr_db=0.0)
    assert np.isclose(float(sigma), expected, rtol=1e-9)
