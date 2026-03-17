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


# ===========================================================================
# model.grad_sensor_pos
# ===========================================================================

def test_triang_grad_sensor_pos_shape():
    """Shape must be (n_dim * n_sensor, n_sensor) = (6, 3)."""
    gp = model.grad_sensor_pos(X_SENSOR, X_SOURCE)
    assert gp.shape == (6, 3), f"Expected (6, 3), got {gp.shape}"


def test_triang_grad_sensor_pos_finite_diff():
    """Analytical gradient should match numerical finite differences."""
    eps = 1e-3
    n_dim, n_sensor = X_SENSOR.shape
    n_measurement = n_sensor  # one angle per sensor

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


# ===========================================================================
# model.grad_source
# ===========================================================================

def test_triang_grad_source_matches_jacobian():
    """grad_source is a wrapper for jacobian; results must be identical."""
    J = model.jacobian(X_SENSOR, X_SOURCE)
    G = model.grad_source(X_SENSOR, X_SOURCE)
    assert np.array_equal(G, J), "grad_source should return the same result as jacobian"


def test_triang_grad_source_shape():
    """Shape (n_dim, n_measurement) = (2, 3) for 3-sensor 2D geometry."""
    G = model.grad_source(X_SENSOR, X_SOURCE)
    assert G.shape == (2, 3), f"Expected (2, 3), got {G.shape}"


# ===========================================================================
# model.grad_bias
# ===========================================================================

def test_triang_grad_bias_is_identity():
    """For 1D AOA, grad_bias = I_{n_sensor x n_sensor} = I_3."""
    B = model.grad_bias(X_SENSOR, X_SOURCE)
    assert B.shape == (3, 3), f"Expected (3, 3), got {B.shape}"
    assert equal_to_tolerance(B, np.eye(3)), "grad_bias should be the identity matrix"


def test_triang_grad_bias_2d_aoa_shape():
    """For 2D AOA, n_measurements = 2 * n_sensors → I_{6x6}."""
    B = model.grad_bias(X_SENSOR, X_SOURCE, do_2d_aoa=True)
    assert B.shape == (6, 6), f"Expected (6, 6) for 2D AOA, got {B.shape}"


# ===========================================================================
# model.jacobian_uncertainty
# ===========================================================================

def test_triang_jacobian_uncertainty_no_flags_shape():
    """No flags → only grad_source → shape (2, 3)."""
    J = model.jacobian_uncertainty(X_SENSOR, X_SOURCE)
    assert J.shape == (2, 3), f"Expected (2, 3), got {J.shape}"


def test_triang_jacobian_uncertainty_do_bias_shape():
    """do_bias=True appends grad_bias (3, 3) → shape (5, 3)."""
    J = model.jacobian_uncertainty(X_SENSOR, X_SOURCE, do_bias=True)
    assert J.shape == (5, 3), f"Expected (5, 3), got {J.shape}"


def test_triang_jacobian_uncertainty_do_pos_error_shape():
    """do_pos_error=True appends grad_sensor_pos (6, 3) → shape (8, 3)."""
    J = model.jacobian_uncertainty(X_SENSOR, X_SOURCE, do_pos_error=True)
    assert J.shape == (8, 3), f"Expected (8, 3), got {J.shape}"


def test_triang_jacobian_uncertainty_both_flags_shape():
    """Both flags → (2 + 3 + 6, 3) = (11, 3)."""
    J = model.jacobian_uncertainty(X_SENSOR, X_SOURCE, do_bias=True, do_pos_error=True)
    assert J.shape == (11, 3), f"Expected (11, 3), got {J.shape}"


# ===========================================================================
# model.error
# ===========================================================================

def test_triang_error_output_shape():
    """Output should be a 2D array of shape (num_pts, num_pts)."""
    x_max = np.array([1000.0, 1000.0])
    num_pts = 11
    epsilon = model.error(X_SENSOR, COV_1DEG, X_SOURCE, x_max, num_pts)
    assert epsilon.shape == (num_pts, num_pts), \
        f"Expected ({num_pts}, {num_pts}), got {epsilon.shape}"


def test_triang_error_minimum_near_source():
    """Error surface minimum should be close to the true source position."""
    x_max = np.array([1000.0, 1000.0])
    num_pts = 11
    epsilon = model.error(X_SENSOR, COV_1DEG, X_SOURCE, x_max, num_pts)

    x_vec = x_max[0] * np.linspace(-1, 1, num_pts)
    y_vec = x_max[1] * np.linspace(-1, 1, num_pts)
    xx, yy = np.meshgrid(x_vec, y_vec)
    idx_min = np.unravel_index(np.argmin(epsilon), epsilon.shape)
    x_min = np.array([xx[idx_min], yy[idx_min]])

    dist = np.linalg.norm(x_min - X_SOURCE)
    # Grid spacing is 200 m; allow 2 cells of error
    assert dist < 400.0, \
        f"Error minimum at {x_min}, source at {X_SOURCE}, dist={dist:.1f} m"


# ===========================================================================
# model.draw_lob
# ===========================================================================

def test_draw_lob_output_shape_single():
    """Single sensor, single angle -> shape (2, 2, 1)."""
    lob = model.draw_lob(X_SENSOR_1, psi=0.0)
    assert lob.shape == (2, 2, 1), f"Expected (2, 2, 1), got {lob.shape}"


def test_draw_lob_output_shape_multi():
    """Three sensors, three angles -> shape (2, 2, 3)."""
    psi = model.measurement(X_SENSOR, X_SOURCE)
    lob = model.draw_lob(X_SENSOR, psi)
    assert lob.shape == (2, 2, 3), f"Expected (2, 2, 3), got {lob.shape}"


def test_draw_lob_start_equals_sensor():
    """First column (start point) of each LOB must equal the sensor position."""
    psi = model.measurement(X_SENSOR, X_SOURCE)
    lob = model.draw_lob(X_SENSOR, psi)
    assert equal_to_tolerance(lob[:, 0, :], X_SENSOR)


def test_draw_lob_east_direction():
    """psi=0 from origin -> end point is directly east (positive x, zero y)."""
    lob = model.draw_lob(X_SENSOR_1, psi=0.0)
    end = lob[:, 1, 0]
    assert end[0] > 0, "End x should be positive (east) for psi=0"
    assert abs(end[1]) < 1e-10, "End y should be zero for psi=0"


def test_draw_lob_north_direction():
    """psi=pi/2 from origin -> end point is directly north (zero x, positive y)."""
    lob = model.draw_lob(X_SENSOR_1, psi=np.pi / 2)
    end = lob[:, 1, 0]
    assert abs(end[0]) < 1e-10, "End x should be zero for psi=pi/2"
    assert end[1] > 0, "End y should be positive (north) for psi=pi/2"


def test_draw_lob_scale_factor():
    """scale=2 should double the displacement from sensor to end point."""
    psi = np.pi / 4
    lob1 = model.draw_lob(X_SENSOR_1, psi=psi, scale=1)
    lob2 = model.draw_lob(X_SENSOR_1, psi=psi, scale=2)
    diff1 = lob1[:, 1, 0] - lob1[:, 0, 0]
    diff2 = lob2[:, 1, 0] - lob2[:, 0, 0]
    assert equal_to_tolerance(diff2, 2 * diff1)


def test_draw_lob_3d_azimuth_only():
    """3-D sensor, no elevation: output shape is (3, 2, 1), z displacement is zero."""
    x_sensor_3d = np.array([[0.], [0.], [500.]])   # sensor at z=500
    lob = model.draw_lob(x_sensor_3d, psi=0.0)
    assert lob.shape == (3, 2, 1)
    assert lob[2, 0, 0] == pytest.approx(500.)     # z start = sensor z
    assert lob[2, 1, 0] == pytest.approx(500.)     # z end = sensor z (no z displacement)


def test_draw_lob_3d_with_elevation():
    """3-D sensor + elevation: end point direction matches [cos(az)*cos(el), sin(az)*cos(el), sin(el)]."""
    x_sensor_3d = np.array([[0.], [0.], [0.]])     # sensor at origin
    az = 0.0                                        # pointing east
    el = np.pi / 4                                  # 45 degrees up
    lob = model.draw_lob(x_sensor_3d, psi=az, el=el)
    assert lob.shape == (3, 2, 1)
    end = lob[:, 1, 0]
    # Expected end (range=1, scale=1): [cos(0)*cos(pi/4), sin(0)*cos(pi/4), sin(pi/4)]
    expected = np.array([np.cos(el), 0., np.sin(el)])
    assert np.allclose(end, expected)


# ===========================================================================
# DirectionFinder.draw_lobs  (1D AOA)
# ===========================================================================

def test_draw_lobs_1d_output_shape_single_case():
    """1D AOA, single case -> (2, 2, n_sensors, 1)."""
    df = DirectionFinder(X_SENSOR, COV_1DEG, do_2d_aoa=False)
    zeta = model.measurement(X_SENSOR, X_SOURCE)   # shape (3,)
    lobs = df.draw_lobs(zeta)
    assert lobs.shape == (2, 2, 3, 1), f"Expected (2, 2, 3, 1), got {lobs.shape}"


def test_draw_lobs_1d_output_shape_multi_case():
    """1D AOA, two cases -> fourth dimension is 2."""
    df = DirectionFinder(X_SENSOR, COV_1DEG, do_2d_aoa=False)
    x_sources = np.column_stack([X_SOURCE, X_SOURCE + 200.0])
    zeta = model.measurement(X_SENSOR, x_sources)  # shape (3, 2)
    lobs = df.draw_lobs(zeta)
    assert lobs.shape == (2, 2, 3, 2), f"Expected (2, 2, 3, 2), got {lobs.shape}"


def test_draw_lobs_1d_starts_at_sensor():
    """Start points of each LOB must equal the corresponding sensor position."""
    df = DirectionFinder(X_SENSOR, COV_1DEG, do_2d_aoa=False)
    zeta = model.measurement(X_SENSOR, X_SOURCE)
    lobs = df.draw_lobs(zeta)
    assert equal_to_tolerance(lobs[:, 0, :, 0], X_SENSOR)


def test_draw_lobs_1d_direction_toward_source():
    """For noiseless measurements the LOB bearing must match the azimuth to source."""
    df = DirectionFinder(X_SENSOR, COV_1DEG, do_2d_aoa=False)
    zeta = model.measurement(X_SENSOR, X_SOURCE)
    lobs = df.draw_lobs(zeta)
    for i in range(3):
        vec = lobs[:, 1, i, 0] - lobs[:, 0, i, 0]
        bearing = np.arctan2(vec[1], vec[0])
        diff = np.mod(bearing - zeta[i] + np.pi, 2 * np.pi) - np.pi
        assert abs(diff) < 1e-6, \
            f"Sensor {i}: bearing {np.rad2deg(bearing):.4f}° != expected {np.rad2deg(zeta[i]):.4f}°"


# ===========================================================================
# DirectionFinder.draw_lobs  (2D AOA, 3D sensor positions)
# ===========================================================================

# 3D sensor positions for 2D AOA tests (az + el both meaningful)
_X_SENSOR_3D = np.array([[0.,    1000.,  500.],
                          [0.,       0.,  866.],
                          [500.,   500.,  500.]])   # shape (3, 3)
_X_SOURCE_3D = np.array([500., 300., 0.])           # shape (3,)
_sig_2d = np.deg2rad(1.0)
_COV_2D = CovarianceMatrix(_sig_2d ** 2 * np.eye(6))  # 2*n_sensors = 6


def test_draw_lobs_2d_output_shape():
    """2D AOA with 3D sensors, single case -> (3, 2, n_sensors, 1)."""
    df = DirectionFinder(_X_SENSOR_3D, _COV_2D, do_2d_aoa=True)
    zeta = model.measurement(_X_SENSOR_3D, _X_SOURCE_3D, do_2d_aoa=True)  # shape (6,)
    lobs = df.draw_lobs(zeta)
    assert lobs.shape == (3, 2, 3, 1), f"Expected (3, 2, 3, 1), got {lobs.shape}"


def test_draw_lobs_2d_aoa_uses_elevation():
    """2D AOA LOBs must have a nonzero z displacement when source is not at sensor altitude."""
    df = DirectionFinder(_X_SENSOR_3D, _COV_2D, do_2d_aoa=True)
    zeta = model.measurement(_X_SENSOR_3D, _X_SOURCE_3D, do_2d_aoa=True)
    lobs = df.draw_lobs(zeta)                       # shape (3, 2, 3, 1)
    # Sensors are at z=500; source is at z=0 → LOBs must point downward
    z_displacement = lobs[2, 1, :, 0] - lobs[2, 0, :, 0]
    assert np.all(z_displacement < 0.), "Expected negative z displacement (sensors above source)"
