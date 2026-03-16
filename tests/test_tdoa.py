import numpy as np
import pytest

from ewgeo.tdoa import model, perf, solvers, TDOAPassiveSurveillanceSystem
from ewgeo.utils.covariance import CovarianceMatrix


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
