import numpy as np
import pytest

from ewgeo.fdoa import model, perf, FDOAPassiveSurveillanceSystem
from ewgeo.utils.covariance import CovarianceMatrix


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
COV_1MS = CovarianceMatrix(1.0 * np.eye(2))    # 1 m/s std dev

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
                             do_resample=False)
    assert isinstance(crlb, CovarianceMatrix)


def test_fdoa_crlb_is_positive_definite():
    crlb = perf.compute_crlb(X_SENSOR, V_SENSOR, X_SOURCE, COV_1MS,
                             do_resample=False)
    eigenvalues = np.linalg.eigvalsh(crlb.cov)
    assert np.all(eigenvalues > 0), f"CRLB not positive definite: {eigenvalues}"


def test_fdoa_crlb_shape():
    crlb = perf.compute_crlb(X_SENSOR, V_SENSOR, X_SOURCE, COV_1MS,
                             do_resample=False)
    assert crlb.cov.shape == (2, 2)


def test_fdoa_crlb_decreases_with_tighter_noise():
    cov_1ms  = CovarianceMatrix(1.0   * np.eye(2))
    cov_01ms = CovarianceMatrix(0.01  * np.eye(2))
    crlb_1   = perf.compute_crlb(X_SENSOR, V_SENSOR, X_SOURCE, cov_1ms,  do_resample=False)
    crlb_01  = perf.compute_crlb(X_SENSOR, V_SENSOR, X_SOURCE, cov_01ms, do_resample=False)
    assert np.trace(crlb_01.cov) < np.trace(crlb_1.cov)


def test_fdoa_crlb_multi_source_returns_list():
    x_sources = np.column_stack([X_SOURCE, X_SOURCE + np.array([300, 0])])
    result = perf.compute_crlb(X_SENSOR, V_SENSOR, x_sources, COV_1MS,
                               do_resample=False)
    assert isinstance(result, list) and len(result) == 2
    for item in result:
        assert isinstance(item, CovarianceMatrix)


# ===========================================================================
# FDOAPassiveSurveillanceSystem
# ===========================================================================

def test_fdoa_pss_measurement_matches_model():
    pss = FDOAPassiveSurveillanceSystem(x=X_SENSOR, vel=V_SENSOR, cov=COV_1MS, do_resample=False)
    assert equal_to_tolerance(pss.measurement(X_SOURCE),
                              model.measurement(X_SENSOR, X_SOURCE, v_sensor=V_SENSOR))


def test_fdoa_pss_jacobian_matches_model():
    pss = FDOAPassiveSurveillanceSystem(x=X_SENSOR, vel=V_SENSOR, cov=COV_1MS, do_resample=False)
    assert equal_to_tolerance(pss.jacobian(X_SOURCE),
                              model.jacobian(X_SENSOR, X_SOURCE, v_sensor=V_SENSOR))


def test_fdoa_pss_num_measurements():
    pss = FDOAPassiveSurveillanceSystem(x=X_SENSOR, vel=V_SENSOR, cov=COV_1MS, do_resample=False)
    assert pss.num_measurements == 2


def test_fdoa_pss_noisy_measurement_shape():
    pss = FDOAPassiveSurveillanceSystem(x=X_SENSOR, vel=V_SENSOR, cov=COV_1MS, do_resample=False)
    zeta = pss.noisy_measurement(X_SOURCE, num_samples=50)
    assert zeta.shape == (2, 50), f"Expected (2, 50), got {zeta.shape}"


def test_fdoa_pss_log_likelihood_peaks_at_source():
    """Noiseless measurement should yield higher LL at the true source."""
    pss = FDOAPassiveSurveillanceSystem(x=X_SENSOR, vel=V_SENSOR, cov=COV_1MS, do_resample=False)
    zeta = pss.measurement(X_SOURCE)
    ll_true  = pss.log_likelihood(zeta=zeta, x_source=X_SOURCE)
    ll_wrong = pss.log_likelihood(zeta=zeta, x_source=X_SOURCE + np.array([500.0, 500.0]))
    assert ll_true > ll_wrong
