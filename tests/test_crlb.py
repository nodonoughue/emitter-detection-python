import numpy as np

from ewgeo.fdoa import FDOAPassiveSurveillanceSystem
from ewgeo.hybrid import HybridPassiveSurveillanceSystem
from ewgeo.tdoa import TDOAPassiveSurveillanceSystem
from ewgeo.triang import DirectionFinder
from ewgeo.utils import SearchSpace
from ewgeo.utils.constants import speed_of_light
from ewgeo.utils.covariance import CovarianceMatrix
from ewgeo.utils.errors import compute_cep50, compute_rmse

_rad2deg = 180.0/np.pi
_deg2rad = np.pi/180.0

# Sensor and Source Positions
# These sensor positions are used for all three examples in Chapter 2.
x_source = np.array([3, 3]) * 1e3
x_aoa = np.array([[4], [0]]) * 1e3
x_tdoa = np.array([[1, 3], [0, 0.5]]) * 1e3
x_fdoa = np.array([[0, 0], [1, 2]]) * 1e3
v_fdoa = np.array([[1, 1], [-1, -1]]) * np.sqrt(0.5) * 300  # 300 m/s, at -45 deg heading

def _make_pss_systems(err_aoa=None, err_time=None, err_freq=None, f0=1.0, tdoa_ref_idx=None, fdoa_ref_idx=None) \
        -> HybridPassiveSurveillanceSystem:
    """

    :param err_aoa: Angle-of-Arrival measurement error (in deg)
    :param err_time: Time-of-Arrival measurement error (in seconds)
    :param err_freq: Frequency measurement error (in Hz)
    :param f0: Operating frequency (in Hz)
    :param tdoa_ref_idx: Index of TDOA reference sensor
    :param fdoa_ref_idx: Index of FDOA reference sensor
    :return pss: Hybrid PSS system object
    """

    # Count the number of sensors in each type
    num_dim, num_aoa = np.shape(x_aoa)
    _, num_tdoa = np.shape(x_tdoa)
    _, num_fdoa = np.shape(x_fdoa)

    # Define Error Covariance Matrix
    if err_aoa is not None:
        cov_psi = (err_aoa * _deg2rad) ** 2.0 * np.eye(num_aoa)  # rad ^ 2
        aoa_pss = DirectionFinder(x=x_aoa, cov=CovarianceMatrix(cov_psi), do_2d_aoa=num_dim>2)
    else:
        aoa_pss = None

    if err_time is not None:
        err_r = err_time * speed_of_light
        cov_r = (err_r ** 2.0) * np.eye(num_tdoa)  # m ^ 2
        tdoa_pss = TDOAPassiveSurveillanceSystem(x=x_tdoa, cov=CovarianceMatrix(cov_r), variance_is_toa=False,
                                                 ref_idx=tdoa_ref_idx)
    else:
        tdoa_pss = None

    if err_freq is not None:
        rr_err = err_freq * speed_of_light / f0  # (m/s)
        cov_rr = (rr_err ** 2) * np.eye(num_fdoa)  # (m/s)^2
        fdoa_pss = FDOAPassiveSurveillanceSystem(x=x_fdoa, vel=v_fdoa, cov=CovarianceMatrix(cov_rr),
                                                 ref_idx=fdoa_ref_idx)
    else:
        fdoa_pss = None

    pss = HybridPassiveSurveillanceSystem(aoa=aoa_pss, tdoa=tdoa_pss, fdoa=fdoa_pss)
    return pss

def test_example_2p3():
    # Generate Error Covariance
    err_aoa = 3  # deg
    err_time = 1e-7  # 100 ns timing error
    err_freq = 10  # Hz
    f0 = 1e9  # Hz
    tdoa_ref_idx = 1
    fdoa_ref_idx = 1
    pss = _make_pss_systems(err_aoa=err_aoa, err_time=err_time, err_freq=err_freq, f0=f0,
                            tdoa_ref_idx=tdoa_ref_idx, fdoa_ref_idx=fdoa_ref_idx)

    # ---- Estimate Error Bounds ----
    # CRLB
    crlb = pss.compute_crlb(x_source=x_source)
    assert equal_to_tolerance(crlb.cov, np.array([[.5124, .5082],[.5082, 4.0515]])*1e4, tol=1e0)

    # RMSE
    rmse_crlb = compute_rmse(crlb)
    assert equal_to_tolerance(rmse_crlb, 210, tol=1e1)

    # CEP50
    cep50_crlb = compute_cep50(crlb)
    assert equal_to_tolerance(cep50_crlb, 150, tol=1e1)

def equal_to_tolerance(x, y, tol=1e-6)->bool:
    """
    Shorthand to compare two lists and ensure each entry has an error less than the specified tolerance
    """
    if np.any(np.shape(x) != np.shape(y)): return False
    return all([abs(xx-yy)<tol for xx, yy in zip(np.ravel(x),np.ravel(y))])