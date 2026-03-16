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
