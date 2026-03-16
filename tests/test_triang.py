import numpy as np
import pytest

from ewgeo.triang import model, perf, solvers, DirectionFinder
from ewgeo.utils.covariance import CovarianceMatrix
from ewgeo.utils import SearchSpace


def equal_to_tolerance(x, y, tol=1e-6):
    if np.size(x) != np.size(y):
        return False
    return np.all(np.fabs(np.array(x) - np.array(y)) < tol)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# 3 sensors arranged in a triangle (2D), source near the centroid
X_SENSOR = np.array([[0.0,  1000.0, 500.0],
                     [0.0,     0.0, 866.0]])   # shape (2, 3)
X_SOURCE = np.array([500.0, 300.0])             # shape (2,)

# Small measurement error covariance (1 deg^2 diagonal)
_sig_rad = np.deg2rad(1.0)
COV_1DEG = CovarianceMatrix((_sig_rad ** 2) * np.eye(3))

# Single-sensor geometry for known-value checks
X_SENSOR_1 = np.array([[0.0], [0.0]])           # sensor at origin, shape (2,1)
X_SOURCE_EAST  = np.array([1000.0,    0.0])     # directly east  -> az = 0
X_SOURCE_NORTH = np.array([   0.0, 1000.0])     # directly north -> az = pi/2
X_SOURCE_NW    = np.array([-500.0,  500.0])     # NW             -> az = 3*pi/4


# ===========================================================================
# model.measurement
# ===========================================================================

def test_measurement_shape_single_source():
    psi = model.measurement(X_SENSOR, X_SOURCE)
    assert psi.shape == (3,), f"Expected (3,), got {psi.shape}"


def test_measurement_shape_multi_source():
    x_sources = np.column_stack([X_SOURCE, X_SOURCE + 100.0])  # (2, 2)
    psi = model.measurement(X_SENSOR, x_sources)
    assert psi.shape == (3, 2), f"Expected (3, 2), got {psi.shape}"


def test_measurement_known_east():
    psi = model.measurement(X_SENSOR_1, X_SOURCE_EAST)
    assert equal_to_tolerance(psi, [0.0]), f"Expected 0, got {psi}"


def test_measurement_known_north():
    psi = model.measurement(X_SENSOR_1, X_SOURCE_NORTH)
    assert equal_to_tolerance(psi, [np.pi / 2]), f"Expected pi/2, got {psi}"


def test_measurement_known_northwest():
    psi = model.measurement(X_SENSOR_1, X_SOURCE_NW)
    expected = np.arctan2(500.0, -500.0)   # 3*pi/4
    assert equal_to_tolerance(psi, [expected]), f"Expected {expected}, got {psi}"


def test_measurement_zero_bias_matches_no_bias():
    """A bias vector of all zeros should produce the same result as no bias."""
    bias_zero = np.zeros(3)
    psi_no_bias = model.measurement(X_SENSOR, X_SOURCE)
    psi_biased  = model.measurement(X_SENSOR, X_SOURCE, bias=bias_zero)
    assert equal_to_tolerance(psi_biased, psi_no_bias)


def test_measurement_bias_first_sensor_only():
    """Non-zero bias on sensor 0 shifts only that measurement."""
    bias = np.array([0.1, 0.0, 0.0])
    psi_no_bias = model.measurement(X_SENSOR, X_SOURCE)
    psi_biased  = model.measurement(X_SENSOR, X_SOURCE, bias=bias)
    assert equal_to_tolerance(psi_biased[0] - psi_no_bias[0], 0.1)
    assert equal_to_tolerance(psi_biased[1:], psi_no_bias[1:])


def test_measurement_bias_third_sensor_only():
    """Non-zero bias on sensor 2 shifts only that measurement."""
    bias = np.array([0.0, 0.0, 0.2])
    psi_no_bias = model.measurement(X_SENSOR, X_SOURCE)
    psi_biased  = model.measurement(X_SENSOR, X_SOURCE, bias=bias)
    assert equal_to_tolerance(psi_biased[2] - psi_no_bias[2], 0.2)
    assert equal_to_tolerance(psi_biased[:2], psi_no_bias[:2])


def test_measurement_bias_all_sensors():
    """Independent bias on every sensor shifts every measurement by the correct amount."""
    bias = np.array([0.1, -0.05, 0.2])
    psi_no_bias = model.measurement(X_SENSOR, X_SOURCE)
    psi_biased  = model.measurement(X_SENSOR, X_SOURCE, bias=bias)
    assert equal_to_tolerance(psi_biased - psi_no_bias, bias)


def test_measurement_2d_aoa_shape():
    x_sensor_3d = np.vstack([X_SENSOR, np.zeros((1, 3))])   # (3, 3)
    x_source_3d = np.append(X_SOURCE, 100.0)                 # (3,)
    psi = model.measurement(x_sensor_3d, x_source_3d, do_2d_aoa=True)
    assert psi.shape == (6,), f"Expected (6,), got {psi.shape}"


def test_measurement_dim_mismatch_raises():
    x_sensor_2d = X_SENSOR                          # (2, 3)
    x_source_3d = np.array([500.0, 300.0, 0.0])    # (3,)
    with pytest.raises(TypeError):
        model.measurement(x_sensor_2d, x_source_3d)


# ===========================================================================
# model.jacobian
# ===========================================================================

def test_jacobian_shape_single_source():
    J = model.jacobian(X_SENSOR, X_SOURCE)
    assert J.shape == (2, 3), f"Expected (2, 3), got {J.shape}"


def test_jacobian_shape_multi_source():
    x_sources = np.column_stack([X_SOURCE, X_SOURCE + 100.0])
    J = model.jacobian(X_SENSOR, x_sources)
    assert J.shape == (2, 3, 2), f"Expected (2, 3, 2), got {J.shape}"


def test_jacobian_finite_difference():
    """
    Analytic Jacobian should match a central-difference numeric Jacobian.
    J[dim, sensor] = d(az_sensor)/d(x_dim)
    """
    eps = 0.1  # metres
    n_dim, n_sensor = X_SENSOR.shape

    J_analytic = model.jacobian(X_SENSOR, X_SOURCE)   # (2, 3)

    J_numeric = np.zeros_like(J_analytic)
    for d in range(n_dim):
        delta = np.zeros(n_dim)
        delta[d] = eps
        m_plus  = model.measurement(X_SENSOR, X_SOURCE + delta)
        m_minus = model.measurement(X_SENSOR, X_SOURCE - delta)
        J_numeric[d, :] = (m_plus - m_minus) / (2.0 * eps)

    assert equal_to_tolerance(J_analytic, J_numeric, tol=1e-5), \
        f"Analytic and numeric Jacobians differ:\n{J_analytic}\nvs\n{J_numeric}"


def test_jacobian_2d_aoa_shape():
    x_sensor_3d = np.vstack([X_SENSOR, np.zeros((1, 3))])
    x_source_3d = np.append(X_SOURCE, 100.0)
    J = model.jacobian(x_sensor_3d, x_source_3d, do_2d_aoa=True)
    # Should be (n_dim=3, 2*n_sensor=6) for single source
    assert J.shape == (3, 6), f"Expected (3, 6), got {J.shape}"


def test_jacobian_finite_difference_2d_aoa():
    """Analytic Jacobian for 2D AOA matches numeric Jacobian."""
    x_sensor_3d = np.vstack([X_SENSOR, np.zeros((1, 3))])   # (3, 3)
    x_source_3d = np.array([500.0, 300.0, 100.0])

    eps = 0.1
    n_dim = 3
    J_analytic = model.jacobian(x_sensor_3d, x_source_3d, do_2d_aoa=True)  # (3, 6)

    J_numeric = np.zeros_like(J_analytic)
    for d in range(n_dim):
        delta = np.zeros(n_dim)
        delta[d] = eps
        m_plus  = model.measurement(x_sensor_3d, x_source_3d + delta, do_2d_aoa=True)
        m_minus = model.measurement(x_sensor_3d, x_source_3d - delta, do_2d_aoa=True)
        J_numeric[d, :] = (m_plus - m_minus) / (2.0 * eps)

    assert equal_to_tolerance(J_analytic, J_numeric, tol=1e-5), \
        f"2D AOA Jacobians differ:\n{J_analytic}\nvs\n{J_numeric}"


# ===========================================================================
# perf.compute_crlb
# ===========================================================================

def test_crlb_returns_covariance_matrix():
    crlb = perf.compute_crlb(X_SENSOR, X_SOURCE, COV_1DEG)
    assert isinstance(crlb, CovarianceMatrix)


def test_crlb_is_positive_definite():
    crlb = perf.compute_crlb(X_SENSOR, X_SOURCE, COV_1DEG)
    eigenvalues = np.linalg.eigvalsh(crlb.cov)
    assert np.all(eigenvalues > 0), f"CRLB eigenvalues not all positive: {eigenvalues}"


def test_crlb_shape():
    crlb = perf.compute_crlb(X_SENSOR, X_SOURCE, COV_1DEG)
    assert crlb.cov.shape == (2, 2), f"Expected (2, 2), got {crlb.cov.shape}"


def test_crlb_decreases_with_tighter_noise():
    """Halving angle error (quarter variance) should reduce CRLB trace."""
    sig_1deg = np.deg2rad(1.0)
    sig_half = np.deg2rad(0.5)
    cov_1 = CovarianceMatrix(sig_1deg**2 * np.eye(3))
    cov_2 = CovarianceMatrix(sig_half**2 * np.eye(3))
    crlb_1 = perf.compute_crlb(X_SENSOR, X_SOURCE, cov_1)
    crlb_2 = perf.compute_crlb(X_SENSOR, X_SOURCE, cov_2)
    assert np.trace(crlb_2.cov) < np.trace(crlb_1.cov)


def test_crlb_multi_source_returns_list():
    x_sources = np.column_stack([X_SOURCE, X_SOURCE + np.array([500, 0])])
    result = perf.compute_crlb(X_SENSOR, x_sources, COV_1DEG)
    assert isinstance(result, list)
    assert len(result) == 2
    for item in result:
        assert isinstance(item, CovarianceMatrix)


# ===========================================================================
# solvers
# ===========================================================================

def test_centroid_exact_measurements_near_source():
    """
    With exact (noiseless) measurements all LOBs intersect at the true source,
    so the centroid of the degenerate triangle should equal the source to
    floating-point precision (~nm for a 1 km geometry).
    """
    psi = model.measurement(X_SENSOR, X_SOURCE)
    x_est = solvers.centroid(X_SENSOR, psi)
    error = np.linalg.norm(x_est - X_SOURCE)
    assert error < 1e-6, f"Centroid estimate too far from source: {error:.2e} m"


def test_angle_bisector_exact_measurements_near_source():
    """
    With exact measurements all LOBs intersect at the true source,
    so the angle bisector of the degenerate triangle should equal the source to
    floating-point precision (~nm for a 1 km geometry).
    """
    psi = model.measurement(X_SENSOR, X_SOURCE)
    x_est = solvers.angle_bisector(X_SENSOR, psi)
    error = np.linalg.norm(x_est - X_SOURCE)
    assert error < 1e-6, f"Angle bisector estimate too far from source: {error:.2e} m"


def test_centroid_output_shape():
    psi = model.measurement(X_SENSOR, X_SOURCE)
    x_est = solvers.centroid(X_SENSOR, psi)
    assert x_est.shape == (2,), f"Expected (2,), got {x_est.shape}"


def test_angle_bisector_output_shape():
    psi = model.measurement(X_SENSOR, X_SOURCE)
    x_est = solvers.angle_bisector(X_SENSOR, psi)
    assert x_est.shape == (2,), f"Expected (2,), got {x_est.shape}"


def test_solvers_require_2d_input():
    """Both solvers require 2D sensor positions; 3D should raise."""
    x_sensor_3d = np.vstack([X_SENSOR, np.zeros((1, 3))])
    psi = np.zeros(3)
    with pytest.raises(TypeError):
        solvers.centroid(x_sensor_3d, psi)
    with pytest.raises(TypeError):
        solvers.angle_bisector(x_sensor_3d, psi)


# ===========================================================================
# DirectionFinder (system.py)
# ===========================================================================

def test_direction_finder_measurement_matches_model():
    df = DirectionFinder(x=X_SENSOR, cov=COV_1DEG)
    psi_df    = df.measurement(X_SOURCE)
    psi_model = model.measurement(X_SENSOR, X_SOURCE)
    assert equal_to_tolerance(psi_df, psi_model)


def test_direction_finder_jacobian_matches_model():
    df = DirectionFinder(x=X_SENSOR, cov=COV_1DEG)
    J_df    = df.jacobian(X_SOURCE)
    J_model = model.jacobian(X_SENSOR, X_SOURCE)
    assert equal_to_tolerance(J_df, J_model)


def test_direction_finder_num_measurements_1d():
    df = DirectionFinder(x=X_SENSOR, cov=COV_1DEG, do_2d_aoa=False)
    assert df.num_measurements == 3


def test_direction_finder_num_measurements_2d():
    df = DirectionFinder(x=X_SENSOR, cov=COV_1DEG, do_2d_aoa=True)
    assert df.num_measurements == 6


def test_direction_finder_noisy_measurement_shape():
    df = DirectionFinder(x=X_SENSOR, cov=COV_1DEG)
    zeta = df.noisy_measurement(X_SOURCE, num_samples=50)
    assert zeta.shape == (3, 50), f"Expected (3, 50), got {zeta.shape}"


def test_direction_finder_noisy_measurement_mean_near_true():
    """Mean of many noisy samples should be close to the true measurement."""
    rng = np.random.default_rng(42)
    df = DirectionFinder(x=X_SENSOR, cov=COV_1DEG)
    psi_true = model.measurement(X_SENSOR, X_SOURCE)

    n = 10_000
    zeta = df.noisy_measurement(X_SOURCE, num_samples=n)   # (3, n)
    psi_mean = np.mean(zeta, axis=1)

    assert equal_to_tolerance(psi_mean, psi_true, tol=5 * _sig_rad / np.sqrt(n)), \
        f"Sample mean far from truth: diff = {psi_mean - psi_true}"


def test_direction_finder_log_likelihood_peaks_at_source():
    """Log-likelihood should be higher at the true source than at a wrong position."""
    df = DirectionFinder(x=X_SENSOR, cov=COV_1DEG)
    zeta = model.measurement(X_SENSOR, X_SOURCE)   # noiseless measurement

    x_wrong = X_SOURCE + np.array([500.0, 500.0])
    ll_true  = df.log_likelihood(x_source=X_SOURCE,  zeta=zeta)
    ll_wrong = df.log_likelihood(x_source=x_wrong,   zeta=zeta)

    assert ll_true > ll_wrong, \
        f"LL at true source ({ll_true:.4f}) not greater than at wrong pos ({ll_wrong:.4f})"


def test_direction_finder_centroid_method():
    df = DirectionFinder(x=X_SENSOR, cov=COV_1DEG)
    psi = model.measurement(X_SENSOR, X_SOURCE)
    x_est = df.centroid(psi)
    assert np.linalg.norm(x_est - X_SOURCE) < 1e-6


def test_direction_finder_angle_bisector_method():
    df = DirectionFinder(x=X_SENSOR, cov=COV_1DEG)
    psi = model.measurement(X_SENSOR, X_SOURCE)
    x_est = df.angle_bisector(psi)
    assert np.linalg.norm(x_est - X_SOURCE) < 1e-6


# ===========================================================================
# DirectionFinder iterative and grid solvers
# ===========================================================================

def test_direction_finder_least_square_near_source():
    """LS solver should converge close to the source given noiseless measurements."""
    df = DirectionFinder(x=X_SENSOR, cov=COV_1DEG)
    zeta = df.measurement(X_SOURCE)
    x_est, _ = df.least_square(zeta, x_init=X_SOURCE + np.array([50., 50.]))
    assert np.linalg.norm(x_est - X_SOURCE) < 1.0, \
        f"LS estimate error too large: {np.linalg.norm(x_est - X_SOURCE):.2f} m"


def test_direction_finder_gradient_descent_near_source():
    """GD solver should converge close to the source given noiseless measurements."""
    df = DirectionFinder(x=X_SENSOR, cov=COV_1DEG)
    zeta = df.measurement(X_SOURCE)
    x_est, _ = df.gradient_descent(zeta, x_init=X_SOURCE + np.array([50., 50.]))
    assert np.linalg.norm(x_est - X_SOURCE) < 1.0, \
        f"GD estimate error too large: {np.linalg.norm(x_est - X_SOURCE):.2f} m"


def test_direction_finder_max_likelihood_near_source():
    """ML grid search should locate the source within one grid-diagonal of the truth."""
    df = DirectionFinder(x=X_SENSOR, cov=COV_1DEG)
    zeta = df.measurement(X_SOURCE)
    ss = SearchSpace(x_ctr=X_SOURCE, epsilon=10., max_offset=300.)
    x_est, _, _ = df.max_likelihood(zeta, search_space=ss)
    assert np.linalg.norm(x_est - X_SOURCE) < 15., \
        f"ML estimate error too large: {np.linalg.norm(x_est - X_SOURCE):.2f} m"
