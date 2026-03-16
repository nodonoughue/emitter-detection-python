import numpy as np

import ewgeo.triang as triang
import ewgeo.tdoa as tdoa
import ewgeo.fdoa as fdoa
from ewgeo.hybrid import model as hybrid_model, perf as hybrid_perf
from ewgeo.hybrid import HybridPassiveSurveillanceSystem
from ewgeo.triang import DirectionFinder
from ewgeo.tdoa import TDOAPassiveSurveillanceSystem
from ewgeo.fdoa import FDOAPassiveSurveillanceSystem
from ewgeo.utils.covariance import CovarianceMatrix
from ewgeo.utils import SearchSpace


def equal_to_tolerance(x, y, tol=1e-6):
    if np.size(x) != np.size(y):
        return False
    return np.all(np.fabs(np.array(x) - np.array(y)) < tol)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Common triangle geometry (same as other test files)
X_SENSOR = np.array([[0.0, 1000.0, 500.0],
                     [0.0,    0.0, 866.0]])   # shape (2, 3)

X_SOURCE = np.array([500.0, 300.0])

# FDOA sensor velocities
V_SENSOR = np.array([[100.0, -100.0,   0.0],
                     [  0.0,    0.0, 100.0]])  # shape (2, 3)

# Per-subsystem covariance matrices (per-sensor, resampled to pair space at runtime)
_sig_rad = np.deg2rad(1.0)
COV_AOA  = CovarianceMatrix(_sig_rad**2 * np.eye(3))   # 3 AOA measurements
COV_TDOA = CovarianceMatrix(100.0 * np.eye(3))          # 3 TDOA sensors (m²), resampled to 2 pairs
COV_FDOA = CovarianceMatrix(1.0   * np.eye(3))          # 3 FDOA sensors ((m/s)²), resampled to 2 pairs

# Block-diagonal covariance for the AOA+TDOA hybrid (6×6 per-sensor, resampled to 5×5)
COV_AOA_TDOA = CovarianceMatrix.block_diagonal(COV_AOA, COV_TDOA)  # shape (6, 6)

# Block-diagonal for the full AOA+TDOA+FDOA hybrid (9×9 per-sensor, resampled to 7×7)
COV_ALL = CovarianceMatrix.block_diagonal(COV_AOA, COV_TDOA, COV_FDOA)  # shape (9, 9)


def _make_aoa_pss():
    return DirectionFinder(x=X_SENSOR, cov=COV_AOA)


def _make_tdoa_pss():
    return TDOAPassiveSurveillanceSystem(x=X_SENSOR, cov=COV_TDOA, variance_is_toa=False, do_resample=True)


def _make_fdoa_pss():
    return FDOAPassiveSurveillanceSystem(x=X_SENSOR, vel=V_SENSOR, cov=COV_FDOA, do_resample=True)


def _make_hybrid_aoa_tdoa():
    return HybridPassiveSurveillanceSystem(aoa=_make_aoa_pss(), tdoa=_make_tdoa_pss())


def _make_hybrid_all():
    return HybridPassiveSurveillanceSystem(aoa=_make_aoa_pss(),
                                           tdoa=_make_tdoa_pss(),
                                           fdoa=_make_fdoa_pss())


# ===========================================================================
# hybrid.model.measurement
# ===========================================================================

def test_hybrid_measurement_aoa_only_matches_triang():
    """AOA-only hybrid measurement matches triang.model.measurement."""
    z_hybrid = hybrid_model.measurement(X_SOURCE, x_aoa=X_SENSOR)
    z_triang = triang.model.measurement(X_SENSOR, X_SOURCE)
    assert equal_to_tolerance(z_hybrid, z_triang)


def test_hybrid_measurement_tdoa_only_matches_tdoa():
    """TDOA-only hybrid measurement matches tdoa.model.measurement."""
    z_hybrid = hybrid_model.measurement(X_SOURCE, x_tdoa=X_SENSOR)
    z_tdoa   = tdoa.model.measurement(X_SENSOR, X_SOURCE)
    assert equal_to_tolerance(z_hybrid, z_tdoa)


def test_hybrid_measurement_fdoa_only_matches_fdoa():
    """FDOA-only hybrid measurement matches fdoa.model.measurement."""
    z_hybrid = hybrid_model.measurement(X_SOURCE, x_fdoa=X_SENSOR, v_fdoa=V_SENSOR)
    z_fdoa   = fdoa.model.measurement(X_SENSOR, X_SOURCE, v_sensor=V_SENSOR)
    assert equal_to_tolerance(z_hybrid, z_fdoa)


def test_hybrid_measurement_combined_is_concatenation():
    """Combined (AOA+TDOA) measurement equals concatenation of components."""
    z_aoa  = triang.model.measurement(X_SENSOR, X_SOURCE)
    z_tdoa = tdoa.model.measurement(X_SENSOR, X_SOURCE)
    z_expected = np.concatenate([z_aoa, z_tdoa])

    z_hybrid = hybrid_model.measurement(X_SOURCE, x_aoa=X_SENSOR, x_tdoa=X_SENSOR)
    assert equal_to_tolerance(z_hybrid, z_expected)


def test_hybrid_measurement_all_three_combined():
    """AOA+TDOA+FDOA measurement equals concatenation of all three components."""
    z_aoa  = triang.model.measurement(X_SENSOR, X_SOURCE)
    z_tdoa = tdoa.model.measurement(X_SENSOR, X_SOURCE)
    z_fdoa = fdoa.model.measurement(X_SENSOR, X_SOURCE, v_sensor=V_SENSOR)
    z_expected = np.concatenate([z_aoa, z_tdoa, z_fdoa])

    z_hybrid = hybrid_model.measurement(X_SOURCE, x_aoa=X_SENSOR,
                                        x_tdoa=X_SENSOR, x_fdoa=X_SENSOR,
                                        v_fdoa=V_SENSOR)
    assert equal_to_tolerance(z_hybrid, z_expected)


def test_hybrid_measurement_shape_aoa_tdoa():
    z = hybrid_model.measurement(X_SOURCE, x_aoa=X_SENSOR, x_tdoa=X_SENSOR)
    # 3 AOA + 2 TDOA = 5
    assert z.shape == (5,), f"Expected (5,), got {z.shape}"


# ===========================================================================
# hybrid.model.jacobian
# ===========================================================================

def test_hybrid_jacobian_shape_aoa_tdoa():
    """AOA+TDOA Jacobian should have n_dim rows and (n_aoa + n_tdoa_pair) columns."""
    J = hybrid_model.jacobian(X_SOURCE, x_aoa=X_SENSOR, x_tdoa=X_SENSOR)
    # n_dim=2, n_aoa=3, n_tdoa_pair=2 → (2, 5)
    assert J.shape == (2, 5), f"Expected (2, 5), got {J.shape}"


def test_hybrid_jacobian_aoa_block_matches_triang():
    """First n_aoa columns of hybrid Jacobian match triang Jacobian."""
    J_hybrid = hybrid_model.jacobian(X_SOURCE, x_aoa=X_SENSOR, x_tdoa=X_SENSOR)
    J_triang = triang.model.jacobian(X_SENSOR, X_SOURCE)
    assert equal_to_tolerance(J_hybrid[:, :3], J_triang)


def test_hybrid_jacobian_tdoa_block_matches_tdoa():
    """Last n_tdoa columns of hybrid Jacobian match TDOA Jacobian."""
    J_hybrid = hybrid_model.jacobian(X_SOURCE, x_aoa=X_SENSOR, x_tdoa=X_SENSOR)
    J_tdoa   = tdoa.model.jacobian(X_SENSOR, X_SOURCE)
    assert equal_to_tolerance(J_hybrid[:, 3:], J_tdoa)


# ===========================================================================
# hybrid_model.grad_source
# ===========================================================================

def test_hybrid_grad_source_shape_aoa_tdoa():
    """grad_source for AOA+TDOA should have n_dim rows and total measurement columns."""
    gs = hybrid_model.grad_source(X_SOURCE, x_aoa=X_SENSOR, x_tdoa=X_SENSOR)
    # n_dim=2, n_aoa=3, n_tdoa_pairs=2 → (2, 5)
    assert gs.shape == (2, 5), f"Expected (2, 5), got {gs.shape}"


def test_hybrid_grad_source_matches_jacobian_aoa_tdoa():
    """grad_source should equal hybrid jacobian exactly for AOA+TDOA (no FDOA)."""
    gs = hybrid_model.grad_source(X_SOURCE, x_aoa=X_SENSOR, x_tdoa=X_SENSOR)
    J  = hybrid_model.jacobian(X_SOURCE,  x_aoa=X_SENSOR, x_tdoa=X_SENSOR)
    assert np.allclose(gs, J), f"Max diff: {np.max(np.abs(gs - J)):.2e}"


def test_hybrid_grad_source_shape_all_three():
    """With FDOA present all rows are padded to 2*n_dim: shape (4, 7)."""
    gs = hybrid_model.grad_source(X_SOURCE, x_aoa=X_SENSOR, x_tdoa=X_SENSOR,
                                  x_fdoa=X_SENSOR, v_fdoa=V_SENSOR)
    assert gs.shape == (4, 7), f"Expected (4, 7), got {gs.shape}"


def test_hybrid_grad_source_matches_jacobian_all_three():
    """grad_source should equal hybrid jacobian exactly for all three sub-systems."""
    gs = hybrid_model.grad_source(X_SOURCE, x_aoa=X_SENSOR, x_tdoa=X_SENSOR,
                                  x_fdoa=X_SENSOR, v_fdoa=V_SENSOR)
    J  = hybrid_model.jacobian(X_SOURCE,  x_aoa=X_SENSOR, x_tdoa=X_SENSOR,
                               x_fdoa=X_SENSOR, v_fdoa=V_SENSOR)
    assert np.allclose(gs, J), f"Max diff: {np.max(np.abs(gs - J)):.2e}"


# ===========================================================================
# hybrid_model.grad_bias
# ===========================================================================

def test_hybrid_grad_bias_shape_aoa_tdoa():
    """grad_bias for AOA+TDOA: block-diag of (3,3) and (3,2) → (6,5)."""
    gb = hybrid_model.grad_bias(X_SOURCE, x_aoa=X_SENSOR, x_tdoa=X_SENSOR)
    assert gb.shape == (6, 5), f"Expected (6, 5), got {gb.shape}"


def test_hybrid_grad_bias_aoa_block_is_identity():
    """AOA bias gradient is I (no reference sensor; each sensor maps to one measurement)."""
    gb = hybrid_model.grad_bias(X_SOURCE, x_aoa=X_SENSOR, x_tdoa=X_SENSOR)
    assert np.allclose(gb[0:3, 0:3], np.eye(3)), "AOA block should be identity"


def test_hybrid_grad_bias_block_diagonal_structure():
    """Off-diagonal blocks between AOA and TDOA bias rows must be zero."""
    gb = hybrid_model.grad_bias(X_SOURCE, x_aoa=X_SENSOR, x_tdoa=X_SENSOR)
    assert np.all(gb[0:3, 3:5] == 0), "AOA rows must not couple to TDOA measurement columns"
    assert np.all(gb[3:6, 0:3] == 0), "TDOA rows must not couple to AOA measurement columns"


def test_hybrid_grad_bias_shape_all_three():
    """grad_bias for AOA+TDOA+FDOA: block-diag of (3,3), (3,2), (3,2) → (9,7)."""
    gb = hybrid_model.grad_bias(X_SOURCE, x_aoa=X_SENSOR, x_tdoa=X_SENSOR,
                                x_fdoa=X_SENSOR, v_fdoa=V_SENSOR)
    assert gb.shape == (9, 7), f"Expected (9, 7), got {gb.shape}"


# ===========================================================================
# hybrid_model.jacobian_uncertainty
# ===========================================================================

def test_hybrid_jacobian_uncertainty_no_flags_shape():
    """Without optional flags, result equals grad_source: shape (n_dim, n_meas) = (2, 5)."""
    j = hybrid_model.jacobian_uncertainty(X_SOURCE, x_aoa=X_SENSOR, x_tdoa=X_SENSOR)
    assert j.shape == (2, 5), f"Expected (2, 5), got {j.shape}"


def test_hybrid_jacobian_uncertainty_with_bias_shape():
    """do_bias=True appends grad_bias rows (6): shape (2+6, 5) = (8, 5)."""
    j = hybrid_model.jacobian_uncertainty(X_SOURCE, x_aoa=X_SENSOR, x_tdoa=X_SENSOR,
                                          do_bias=True)
    assert j.shape == (8, 5), f"Expected (8, 5), got {j.shape}"


def test_hybrid_jacobian_uncertainty_with_pos_error_shape():
    """do_pos_error=True appends grad_sensor_pos rows (24): shape (2+24, 5) = (26, 5).
    grad_sensor_pos for AOA+TDOA: 2 blocks of (2*n_dim*n_k, n_k_meas), total 24 rows."""
    j = hybrid_model.jacobian_uncertainty(X_SOURCE, x_aoa=X_SENSOR, x_tdoa=X_SENSOR,
                                          do_pos_error=True)
    assert j.shape == (26, 5), f"Expected (26, 5), got {j.shape}"


def test_hybrid_jacobian_uncertainty_with_all_flags_shape():
    """Both flags: shape (2+6+24, 5) = (32, 5)."""
    j = hybrid_model.jacobian_uncertainty(X_SOURCE, x_aoa=X_SENSOR, x_tdoa=X_SENSOR,
                                          do_bias=True, do_pos_error=True)
    assert j.shape == (32, 5), f"Expected (32, 5), got {j.shape}"


def test_hybrid_jacobian_uncertainty_no_flags_all_three_shape():
    """With FDOA, grad_source has 2*n_dim rows: shape (4, 7)."""
    j = hybrid_model.jacobian_uncertainty(X_SOURCE, x_aoa=X_SENSOR, x_tdoa=X_SENSOR,
                                          x_fdoa=X_SENSOR, v_fdoa=V_SENSOR)
    assert j.shape == (4, 7), f"Expected (4, 7), got {j.shape}"


def test_hybrid_jacobian_uncertainty_all_flags_all_three_shape():
    """All three sub-systems, both flags:
    grad_source (4,7) + grad_bias (9,7) + grad_sensor_pos (36,7) → (49, 7)."""
    j = hybrid_model.jacobian_uncertainty(X_SOURCE, x_aoa=X_SENSOR, x_tdoa=X_SENSOR,
                                          x_fdoa=X_SENSOR, v_fdoa=V_SENSOR,
                                          do_bias=True, do_pos_error=True)
    assert j.shape == (49, 7), f"Expected (49, 7), got {j.shape}"


# ===========================================================================
# hybrid_model.error
# ===========================================================================

def test_hybrid_error_grid_shape():
    """error() should return a (num_pts, num_pts) grid and matching x/y vectors."""
    num_pts = 11
    epsilon, x_vec, y_vec = hybrid_model.error(
        X_SOURCE, COV_AOA_TDOA,
        x_aoa=X_SENSOR, x_tdoa=X_SENSOR,
        x_max=300., num_pts=num_pts, do_resample=True)
    assert epsilon.shape == (num_pts, num_pts), \
        f"Expected ({num_pts},{num_pts}), got {epsilon.shape}"
    assert x_vec.shape == (num_pts,), f"x_vec shape wrong: {x_vec.shape}"
    assert y_vec.shape == (num_pts,), f"y_vec shape wrong: {y_vec.shape}"


def test_hybrid_error_minimum_near_source():
    """The error surface minimum should be close to the true source position."""
    num_pts = 21
    epsilon, x_vec, y_vec = hybrid_model.error(
        X_SOURCE, COV_AOA_TDOA,
        x_aoa=X_SENSOR, x_tdoa=X_SENSOR,
        x_max=300., num_pts=num_pts, do_resample=True)
    idx = np.unravel_index(np.argmin(epsilon), epsilon.shape)
    x_min = np.array([x_vec[idx[1]], y_vec[idx[0]]])
    grid_spacing = 600. / (num_pts - 1)
    assert np.linalg.norm(x_min - X_SOURCE) < 2 * grid_spacing, \
        f"Error minimum {x_min} too far from source {X_SOURCE}"


# ===========================================================================
# hybrid_model.grad_sensor_pos
# ===========================================================================

def test_hybrid_grad_sensor_pos_shape_all_three():
    """With n_dim=2 and 3 sensors per sub-system:
    rows = 2 * n_dim * (n_aoa + n_tdoa + n_fdoa) = 2*2*(3+3+3) = 36
    cols = n_aoa + (n_tdoa-1) + (n_fdoa-1)       = 3 + 2 + 2   = 7
    """
    gp = hybrid_model.grad_sensor_pos(
        x_source=X_SOURCE, x_aoa=X_SENSOR, x_tdoa=X_SENSOR,
        x_fdoa=X_SENSOR, v_fdoa=V_SENSOR)
    assert gp.shape == (36, 7), f"Expected (36, 7), got {gp.shape}"


def test_hybrid_grad_sensor_pos_block_diagonal_structure():
    """Off-diagonal blocks between AOA, TDOA, and FDOA sub-systems must be zero.

    Block layout (n_dim=2, 3 sensors each):
      rows [0:12]  = AOA  (pos + vel zeros),  cols [0:3]
      rows [12:24] = TDOA (pos + vel zeros),  cols [3:5]
      rows [24:36] = FDOA (pos + vel),        cols [5:7]
    """
    gp = hybrid_model.grad_sensor_pos(
        x_source=X_SOURCE, x_aoa=X_SENSOR, x_tdoa=X_SENSOR,
        x_fdoa=X_SENSOR, v_fdoa=V_SENSOR)
    assert np.all(gp[0:12, 3:7] == 0),  "AOA rows must not couple to TDOA/FDOA columns"
    assert np.all(gp[12:24, 0:3] == 0), "TDOA rows must not couple to AOA columns"
    assert np.all(gp[12:24, 5:7] == 0), "TDOA rows must not couple to FDOA columns"
    assert np.all(gp[24:36, 0:5] == 0), "FDOA rows must not couple to AOA/TDOA columns"


# ===========================================================================
# hybrid.perf.compute_crlb
# ===========================================================================

def test_hybrid_crlb_returns_covariance_matrix():
    crlb = hybrid_perf.compute_crlb(X_SOURCE, COV_AOA_TDOA,
                                    x_aoa=X_SENSOR, x_tdoa=X_SENSOR,
                                    do_resample=True)
    assert isinstance(crlb, CovarianceMatrix)


def test_hybrid_crlb_is_positive_definite():
    crlb = hybrid_perf.compute_crlb(X_SOURCE, COV_AOA_TDOA,
                                    x_aoa=X_SENSOR, x_tdoa=X_SENSOR,
                                    do_resample=True)
    eigenvalues = np.linalg.eigvalsh(crlb.cov)
    assert np.all(eigenvalues > 0), f"CRLB not positive definite: {eigenvalues}"


def test_hybrid_crlb_shape():
    crlb = hybrid_perf.compute_crlb(X_SOURCE, COV_AOA_TDOA,
                                    x_aoa=X_SENSOR, x_tdoa=X_SENSOR,
                                    do_resample=True)
    assert crlb.cov.shape == (2, 2)


def test_hybrid_crlb_smaller_than_aoa_alone():
    """Combining AOA+TDOA should yield a smaller position uncertainty than AOA alone."""
    crlb_aoa    = triang.perf.compute_crlb(X_SENSOR, X_SOURCE, COV_AOA)
    crlb_hybrid = hybrid_perf.compute_crlb(X_SOURCE, COV_AOA_TDOA,
                                           x_aoa=X_SENSOR, x_tdoa=X_SENSOR,
                                           do_resample=True)
    assert np.trace(crlb_hybrid.cov) < np.trace(crlb_aoa.cov)


def test_hybrid_crlb_smaller_than_tdoa_alone():
    """Combining AOA+TDOA should yield a smaller position uncertainty than TDOA alone."""
    crlb_tdoa   = tdoa.perf.compute_crlb(X_SENSOR, X_SOURCE, COV_TDOA,
                                         variance_is_toa=False, do_resample=True)
    crlb_hybrid = hybrid_perf.compute_crlb(X_SOURCE, COV_AOA_TDOA,
                                           x_aoa=X_SENSOR, x_tdoa=X_SENSOR,
                                           do_resample=True)
    assert np.trace(crlb_hybrid.cov) < np.trace(crlb_tdoa.cov)


# ===========================================================================
# HybridPassiveSurveillanceSystem
# ===========================================================================

def test_hybrid_pss_num_measurements_aoa_tdoa():
    """AOA (3) + TDOA (2) = 5 total measurements."""
    h = _make_hybrid_aoa_tdoa()
    assert h.num_measurements == 5


def test_hybrid_pss_num_measurements_all_three():
    """AOA (3) + TDOA (2) + FDOA (2) = 7 total measurements."""
    h = _make_hybrid_all()
    assert h.num_measurements == 7


def test_hybrid_pss_measurement_matches_model():
    """PSS measurement should equal the concatenation of component models."""
    h = _make_hybrid_aoa_tdoa()
    z_pss      = h.measurement(X_SOURCE)
    z_expected = hybrid_model.measurement(X_SOURCE, x_aoa=X_SENSOR, x_tdoa=X_SENSOR)
    assert equal_to_tolerance(z_pss, z_expected)


def test_hybrid_pss_measurement_shape():
    h = _make_hybrid_aoa_tdoa()
    z = h.measurement(X_SOURCE)
    assert z.shape == (5,), f"Expected (5,), got {z.shape}"


def test_hybrid_pss_log_likelihood_peaks_at_source():
    """Noiseless measurement should yield higher LL at the true source."""
    h = _make_hybrid_aoa_tdoa()
    zeta = h.measurement(X_SOURCE)
    ll_true  = h.log_likelihood(x_source=X_SOURCE,               zeta=zeta)
    ll_wrong = h.log_likelihood(x_source=X_SOURCE + np.array([500.0, 500.0]), zeta=zeta)
    assert ll_true > ll_wrong


def test_hybrid_pss_aoa_measurement_idx_correct():
    h = _make_hybrid_aoa_tdoa()
    assert h.num_aoa_measurements == 3
    assert h.num_tdoa_measurements == 2
    assert h.num_fdoa_measurements == 0


# ===========================================================================
# HybridPassiveSurveillanceSystem iterative and grid solvers
# ===========================================================================

def test_hybrid_pss_least_square_near_source():
    """LS solver should converge close to the source given noiseless measurements."""
    h = _make_hybrid_aoa_tdoa()
    zeta = h.measurement(X_SOURCE)
    x_est, _ = h.least_square(zeta, x_init=X_SOURCE + np.array([50., 50.]))
    assert np.linalg.norm(x_est - X_SOURCE) < 1.0, \
        f"LS estimate error too large: {np.linalg.norm(x_est - X_SOURCE):.2f} m"


def test_hybrid_pss_gradient_descent_near_source():
    """GD solver should converge close to the source given noiseless measurements."""
    h = _make_hybrid_aoa_tdoa()
    zeta = h.measurement(X_SOURCE)
    x_est, _ = h.gradient_descent(zeta, x_init=X_SOURCE + np.array([50., 50.]))
    assert np.linalg.norm(x_est - X_SOURCE) < 1.0, \
        f"GD estimate error too large: {np.linalg.norm(x_est - X_SOURCE):.2f} m"


def test_hybrid_pss_max_likelihood_near_source():
    """ML grid search should locate the source within one grid-diagonal of the truth."""
    h = _make_hybrid_aoa_tdoa()
    zeta = h.measurement(X_SOURCE)
    ss = SearchSpace(x_ctr=X_SOURCE, epsilon=10., max_offset=300.)
    x_est, _, _ = h.max_likelihood(zeta, search_space=ss)
    assert np.linalg.norm(x_est - X_SOURCE) < 15., \
        f"ML estimate error too large: {np.linalg.norm(x_est - X_SOURCE):.2f} m"
