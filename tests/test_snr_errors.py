"""
Tests for SNR-based error covariance estimation across TDOA, FDOA, AOA, and Hybrid PSS classes.
"""

import numpy as np
import pytest

from ewgeo.utils.snr import compute_snr_per_sensor
from ewgeo.utils.constants import speed_of_light
from ewgeo.utils.covariance import CovarianceMatrix
from ewgeo.tdoa import model as tdoa_model
from ewgeo.tdoa import TDOAPassiveSurveillanceSystem
from ewgeo.fdoa import model as fdoa_model
from ewgeo.fdoa import FDOAPassiveSurveillanceSystem
from ewgeo.triang import model as triang_model
from ewgeo.triang import DirectionFinder
from ewgeo.hybrid import HybridPassiveSurveillanceSystem
from ewgeo.utils.unit_conversions import db_to_lin

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
X_SENSOR = np.array([[0., 1000., 500.], [0., 0., 866.]])   # 3 sensors, 2D
X_SOURCE = np.array([500., 300.])
ERP_DBW = 30.0
MDS_DBW = -120.0
FREQ_HZ = 1e9
BW_HZ = 1e6
PULSE_LEN_S = 1e-3
BW_RMS_HZ = BW_HZ / np.sqrt(12)
T_RMS_S = PULSE_LEN_S * np.sqrt(4 / 3)
APERTURE_M = 10.0


def equal_to_tolerance(x, y, tol=1e-6):
    return abs(x - y) / (abs(y) + 1e-99) < tol


# ===========================================================================
# SNR utility (utils/snr.py)
# ===========================================================================

def test_snr_known_range():
    """Hand-verify SNR at a known range."""
    r = 1000.0  # metres
    prop_gain_db = 20.0 * np.log10(speed_of_light / (4.0 * np.pi * r * FREQ_HZ))
    expected_snr = MDS_DBW + ERP_DBW + prop_gain_db

    # Single sensor at exactly r = 1000 m from source at origin
    x_s = np.array([[1000.], [0.]])  # (2, 1) sensor
    x_src = np.array([0., 0.])
    snr = compute_snr_per_sensor(x_s, x_src, ERP_DBW, MDS_DBW, FREQ_HZ)

    assert equal_to_tolerance(snr[0], expected_snr, tol=1e-9)


def test_snr_decreases_with_range():
    """SNR must decrease as range increases."""
    x_near = np.array([[500.], [0.]])
    x_far = np.array([[2000.], [0.]])
    x_src = np.array([0., 0.])

    snr_near = compute_snr_per_sensor(x_near, x_src, ERP_DBW, MDS_DBW, FREQ_HZ)
    snr_far = compute_snr_per_sensor(x_far, x_src, ERP_DBW, MDS_DBW, FREQ_HZ)
    assert snr_near[0] > snr_far[0]


def test_snr_shape():
    """Output shape must equal (n_sensor,)."""
    snr = compute_snr_per_sensor(X_SENSOR, X_SOURCE, ERP_DBW, MDS_DBW, FREQ_HZ)
    assert snr.shape == (3,)


# ===========================================================================
# TDOA error
# ===========================================================================

def test_tdoa_toa_error_known_snr():
    """toa_error_cross_corr at a fixed SNR matches the analytic formula."""
    snr_db = 10.0
    snr_lin = db_to_lin(snr_db)
    expected = 1.0 / (8.0 * np.pi * snr_lin * BW_HZ * PULSE_LEN_S * BW_RMS_HZ)
    result = tdoa_model.toa_error_cross_corr(snr_db, BW_HZ, PULSE_LEN_S, BW_RMS_HZ)
    assert equal_to_tolerance(result, expected, tol=1e-9)


def test_tdoa_cov_from_snr_shape():
    """tdoa_cov_from_snr returns a square N×N matrix."""
    cov = tdoa_model.tdoa_cov_from_snr(X_SENSOR, X_SOURCE, ERP_DBW, MDS_DBW, FREQ_HZ,
                                        BW_HZ, PULSE_LEN_S)
    assert cov.cov.shape == (3, 3)


def test_tdoa_cov_from_snr_diagonal():
    """Off-diagonal entries are zero."""
    cov = tdoa_model.tdoa_cov_from_snr(X_SENSOR, X_SOURCE, ERP_DBW, MDS_DBW, FREQ_HZ,
                                        BW_HZ, PULSE_LEN_S)
    m = cov.cov
    np.testing.assert_array_almost_equal(m - np.diag(np.diag(m)), 0)


def test_tdoa_cov_closer_sensor_lower_error():
    """A closer sensor has smaller TOA variance than a farther one."""
    # Sensor at (100, 0) is closer to source (500, 300) than sensor at (2000, 0)
    x_s = np.array([[100., 2000.], [0., 0.]])
    cov = tdoa_model.tdoa_cov_from_snr(x_s, X_SOURCE, ERP_DBW, MDS_DBW, FREQ_HZ,
                                        BW_HZ, PULSE_LEN_S)
    assert cov.cov[0, 0] < cov.cov[1, 1]


def test_tdoa_compute_cov():
    """pss.compute_cov(x_source) returns same as tdoa_cov_from_snr scaled to ROA."""
    pss = TDOAPassiveSurveillanceSystem(
        x=X_SENSOR, variance_is_toa=False,
        erp_dbw=ERP_DBW, mds_dbw=MDS_DBW, freq_hz=FREQ_HZ,
        bandwidth_hz=BW_HZ, pulse_len_s=PULSE_LEN_S,
    )
    result = pss.compute_cov(X_SOURCE)

    # Expected: tdoa_cov_from_snr → ROA → resample
    raw_toa = tdoa_model.tdoa_cov_from_snr(X_SENSOR, X_SOURCE, ERP_DBW, MDS_DBW, FREQ_HZ,
                                            BW_HZ, PULSE_LEN_S)
    raw_roa = raw_toa.multiply(speed_of_light ** 2, overwrite=False)
    expected = raw_roa.resample(ref_idx=pss.ref_idx)

    np.testing.assert_allclose(result.cov, expected.cov, rtol=1e-10)


# ===========================================================================
# FDOA error
# ===========================================================================

def test_fdoa_foa_error_known_snr():
    """foa_error_cross_corr at a fixed SNR matches the Stein (1981) formula."""
    snr_db = 10.0
    snr_lin = db_to_lin(snr_db)
    expected = 1.0 / (4.0 * np.pi ** 2 * T_RMS_S ** 2 * PULSE_LEN_S * BW_HZ * snr_lin)
    result = fdoa_model.foa_error_cross_corr(snr_db, BW_HZ, PULSE_LEN_S, T_RMS_S)
    assert equal_to_tolerance(result, expected, tol=1e-9)


def test_fdoa_foa_error_default_t_rms():
    """Omitting t_rms_s gives the same result as passing pulse_len_s * sqrt(4/3)."""
    snr_db = 10.0
    explicit = fdoa_model.foa_error_cross_corr(snr_db, BW_HZ, PULSE_LEN_S, T_RMS_S)
    default = fdoa_model.foa_error_cross_corr(snr_db, BW_HZ, PULSE_LEN_S)
    assert equal_to_tolerance(explicit, default, tol=1e-12)


def test_fdoa_cov_from_snr_shape():
    cov = fdoa_model.fdoa_cov_from_snr(X_SENSOR, X_SOURCE, ERP_DBW, MDS_DBW, FREQ_HZ,
                                        BW_HZ, PULSE_LEN_S)
    assert cov.cov.shape == (3, 3)


def test_fdoa_cov_from_snr_diagonal():
    cov = fdoa_model.fdoa_cov_from_snr(X_SENSOR, X_SOURCE, ERP_DBW, MDS_DBW, FREQ_HZ,
                                        BW_HZ, PULSE_LEN_S)
    m = cov.cov
    np.testing.assert_array_almost_equal(m - np.diag(np.diag(m)), 0)


def test_fdoa_cov_closer_sensor_lower_error():
    x_s = np.array([[100., 2000.], [0., 0.]])
    cov = fdoa_model.fdoa_cov_from_snr(x_s, X_SOURCE, ERP_DBW, MDS_DBW, FREQ_HZ,
                                        BW_HZ, PULSE_LEN_S)
    assert cov.cov[0, 0] < cov.cov[1, 1]


def test_fdoa_compute_cov():
    """pss.compute_cov(x_source) matches fdoa_cov_from_snr → resample."""
    v_sensor = np.zeros_like(X_SENSOR)
    pss = FDOAPassiveSurveillanceSystem(
        x=X_SENSOR, vel=v_sensor,
        erp_dbw=ERP_DBW, mds_dbw=MDS_DBW, freq_hz=FREQ_HZ,
        bandwidth_hz=BW_HZ, pulse_len_s=PULSE_LEN_S,
    )
    result = pss.compute_cov(X_SOURCE)

    expected_raw = fdoa_model.fdoa_cov_from_snr(X_SENSOR, X_SOURCE, ERP_DBW, MDS_DBW, FREQ_HZ,
                                                 BW_HZ, PULSE_LEN_S)
    expected = expected_raw.resample(ref_idx=pss.ref_idx)
    np.testing.assert_allclose(result.cov, expected.cov, rtol=1e-10)


# ===========================================================================
# AOA / Direction Finder error
# ===========================================================================

def test_aoa_error_known_snr():
    """At SNR=0 dB, sigma² = (c/(2pi*f*d))²/2."""
    snr_db = 0.0
    phase_to_angle = speed_of_light / (2.0 * np.pi * FREQ_HZ * APERTURE_M)
    expected = 0.5 * phase_to_angle ** 2
    result = triang_model.aoa_error_from_snr(snr_db, FREQ_HZ, APERTURE_M)
    assert equal_to_tolerance(result, expected, tol=1e-9)


def test_aoa_error_larger_aperture_lower_error():
    """Larger interferometer baseline → smaller AOA variance."""
    snr_db = 10.0
    var_small = triang_model.aoa_error_from_snr(snr_db, FREQ_HZ, aperture_m=5.0)
    var_large = triang_model.aoa_error_from_snr(snr_db, FREQ_HZ, aperture_m=50.0)
    assert var_large < var_small


def test_aoa_cov_from_snr_shape():
    cov = triang_model.aoa_cov_from_snr(X_SENSOR, X_SOURCE, ERP_DBW, MDS_DBW, FREQ_HZ, APERTURE_M)
    assert cov.cov.shape == (3, 3)


def test_aoa_cov_from_snr_diagonal():
    cov = triang_model.aoa_cov_from_snr(X_SENSOR, X_SOURCE, ERP_DBW, MDS_DBW, FREQ_HZ, APERTURE_M)
    m = cov.cov
    np.testing.assert_array_almost_equal(m - np.diag(np.diag(m)), 0)


def test_aoa_cov_closer_sensor_lower_error():
    x_s = np.array([[100., 2000.], [0., 0.]])
    cov = triang_model.aoa_cov_from_snr(x_s, X_SOURCE, ERP_DBW, MDS_DBW, FREQ_HZ, APERTURE_M)
    assert cov.cov[0, 0] < cov.cov[1, 1]


def test_df_compute_cov():
    """pss.compute_cov(x_source) matches aoa_cov_from_snr directly."""
    pss = DirectionFinder(
        x=X_SENSOR,
        erp_dbw=ERP_DBW, mds_dbw=MDS_DBW, freq_hz=FREQ_HZ, aperture_m=APERTURE_M,
    )
    result = pss.compute_cov(X_SOURCE)
    expected = triang_model.aoa_cov_from_snr(X_SENSOR, X_SOURCE, ERP_DBW, MDS_DBW, FREQ_HZ, APERTURE_M)
    np.testing.assert_allclose(result.cov, expected.cov, rtol=1e-10)


# ===========================================================================
# Hybrid
# ===========================================================================

def _make_hybrid_snr():
    """Build a Hybrid PSS with all-SNR components."""
    aoa = DirectionFinder(
        x=X_SENSOR,
        erp_dbw=ERP_DBW, mds_dbw=MDS_DBW, freq_hz=FREQ_HZ, aperture_m=APERTURE_M,
    )
    tdoa = TDOAPassiveSurveillanceSystem(
        x=X_SENSOR, variance_is_toa=False,
        erp_dbw=ERP_DBW, mds_dbw=MDS_DBW, freq_hz=FREQ_HZ,
        bandwidth_hz=BW_HZ, pulse_len_s=PULSE_LEN_S,
    )
    return HybridPassiveSurveillanceSystem(aoa=aoa, tdoa=tdoa)


def test_hybrid_compute_cov_shape():
    """Combined SNR covariance has the correct shape."""
    pss = _make_hybrid_snr()
    cov = pss.compute_cov(X_SOURCE)
    # 3 AOA + 2 TDOA pairs = 5 measurements
    assert cov.cov.shape == (5, 5)


def test_hybrid_compute_cov_block_diagonal():
    """Cross-type blocks are zero (AOA and TDOA are independent)."""
    pss = _make_hybrid_snr()
    cov = pss.compute_cov(X_SOURCE)
    m = cov.cov
    # AOA block: rows/cols 0:3, TDOA pair block: rows/cols 3:5
    aoa_tdoa_cross = m[0:3, 3:5]
    np.testing.assert_array_equal(aoa_tdoa_cross, 0)


def test_hybrid_compute_snr_shape():
    """compute_snr returns one value per sensor across all sub-PSSs (3 AOA + 3 TDOA = 6)."""
    pss = _make_hybrid_snr()
    snr = pss.compute_snr(X_SOURCE)
    assert snr.shape == (6,)


def test_hybrid_compute_snr_partial_nan():
    """Sub-PSSs without SNR params contribute NaN entries."""
    aoa = DirectionFinder(x=X_SENSOR, cov=np.eye(3),
                          erp_dbw=ERP_DBW, mds_dbw=MDS_DBW, freq_hz=FREQ_HZ, aperture_m=APERTURE_M)
    tdoa = TDOAPassiveSurveillanceSystem(x=X_SENSOR, cov=np.eye(3), variance_is_toa=False)
    pss = HybridPassiveSurveillanceSystem(aoa=aoa, tdoa=tdoa)
    snr = pss.compute_snr(X_SOURCE)
    assert snr.shape == (6,)
    assert np.all(np.isfinite(snr[:3]))   # AOA has SNR params
    assert np.all(np.isnan(snr[3:]))      # TDOA does not


def test_hybrid_compute_snr_no_params_raises():
    """ValueError when no sub-PSS has SNR parameters."""
    aoa = DirectionFinder(x=X_SENSOR, cov=np.eye(3))
    tdoa = TDOAPassiveSurveillanceSystem(x=X_SENSOR, cov=np.eye(3), variance_is_toa=False)
    pss = HybridPassiveSurveillanceSystem(aoa=aoa, tdoa=tdoa)
    with pytest.raises(ValueError):
        pss.compute_snr(X_SOURCE)


def test_hybrid_compute_cov_varies_with_position():
    """Covariance changes when the source position changes."""
    pss = _make_hybrid_snr()
    cov1 = pss.compute_cov(np.array([500., 300.]))
    cov2 = pss.compute_cov(np.array([5000., 3000.]))
    assert not np.allclose(cov1.cov, cov2.cov)


# ===========================================================================
# Propagation model selection (snr.py)
# ===========================================================================

def test_snr_2d_uses_free_space():
    """2-D positions with no coord_system must match get_free_space_path_loss directly."""
    from ewgeo.prop.model import get_free_space_path_loss

    x_s = np.array([[1000.], [0.]])
    x_src = np.array([0., 0.])
    snr = compute_snr_per_sensor(x_s, x_src, ERP_DBW, MDS_DBW, FREQ_HZ)

    r = 1000.0
    expected_loss = get_free_space_path_loss(r, FREQ_HZ, include_atm_loss=False)
    expected_snr = ERP_DBW - expected_loss + MDS_DBW
    assert equal_to_tolerance(snr[0], expected_snr, tol=1e-9)


def test_snr_3d_enu_no_ref_uses_up_as_height():
    """ENU with no ref: Up component is used as height, result differs from 2-D free-space."""
    x_s_3d = np.array([[1000.], [0.], [100.]])   # 100 m height
    x_src_3d = np.array([0., 0., 50.])            # 50 m height
    x_src_2d = np.array([0., 0.])

    snr_3d = compute_snr_per_sensor(x_s_3d, x_src_3d, ERP_DBW, MDS_DBW, FREQ_HZ,
                                    coord_system='enu')
    snr_2d = compute_snr_per_sensor(x_s_3d[:2], x_src_2d, ERP_DBW, MDS_DBW, FREQ_HZ)

    # get_path_loss (with atmospheric loss) vs get_free_space_path_loss (no atm) must differ
    assert abs(float(snr_3d[0]) - float(snr_2d[0])) > 1e-4


def test_snr_3d_enu_with_ref_differs_from_no_ref():
    """Providing enu_ref_lla shifts the altitudes via enu_to_lla and changes SNR."""
    x_s_3d = np.array([[1000.], [0.], [100.]])
    x_src_3d = np.array([0., 0., 50.])
    ref_lla = (40.0, -105.0, 1600.0)   # e.g. Denver area, 1600 m MSL

    snr_no_ref = compute_snr_per_sensor(x_s_3d, x_src_3d, ERP_DBW, MDS_DBW, FREQ_HZ,
                                        coord_system='enu')
    snr_with_ref = compute_snr_per_sensor(x_s_3d, x_src_3d, ERP_DBW, MDS_DBW, FREQ_HZ,
                                          coord_system='enu', enu_ref_lla=ref_lla)

    # With 1600 m MSL offset the effective altitudes differ → SNR must differ
    assert abs(float(snr_no_ref[0]) - float(snr_with_ref[0])) > 1e-4


def test_snr_3d_ecef():
    """ECEF coord_system must run without error and produce a finite SNR."""
    from ewgeo.utils.coordinates import enu_to_ecef

    # Build an ECEF point near Denver (lat=40°, lon=-105°, alt=1600 m)
    lat_ref, lon_ref, alt_ref = 40.0, -105.0, 1600.0
    xe_src, ye_src, ze_src = enu_to_ecef(0., 0., 50., lat_ref, lon_ref, alt_ref)
    xe_rx, ye_rx, ze_rx = enu_to_ecef(1000., 0., 100., lat_ref, lon_ref, alt_ref)

    x_sensor_ecef = np.array([[xe_rx], [ye_rx], [ze_rx]])
    x_source_ecef = np.array([xe_src, ye_src, ze_src])

    snr = compute_snr_per_sensor(x_sensor_ecef, x_source_ecef, ERP_DBW, MDS_DBW, FREQ_HZ,
                                 coord_system='ecef')
    assert np.isfinite(float(snr[0]))


def test_snr_pss_constructor_stores_coord_system():
    """PSS constructors must forward coord_system and enu_ref_lla into _snr_params."""
    pss = TDOAPassiveSurveillanceSystem(
        x=X_SENSOR, variance_is_toa=False,
        erp_dbw=ERP_DBW, mds_dbw=MDS_DBW, freq_hz=FREQ_HZ,
        bandwidth_hz=BW_HZ, pulse_len_s=PULSE_LEN_S,
        coord_system='enu', enu_ref_lla=(40.0, -105.0, 1600.0),
    )
    assert pss._snr_params['coord_system'] == 'enu'
    assert pss._snr_params['enu_ref_lla'] == (40.0, -105.0, 1600.0)
