"""
Tests for the ewgeo.aoa module — sensor-level direction-finding techniques.

Covers the three pure-function subsystems:
  - watson_watt  (Adcock + reference antenna)
  - interferometer (phase-difference DF)
  - directional   (amplitude-scan DF with arbitrary gain pattern)

The gain-function factory (make_gain_functions) is already covered in
test_aoa_gain_functions.py and is not duplicated here.
"""

import numpy as np

from ewgeo.aoa import make_gain_functions
from ewgeo.aoa import watson_watt, interferometer, directional


def equal_to_tolerance(x, y, tol=1e-6):
    if np.size(x) != np.size(y):
        return False
    return np.all(np.fabs(np.array(x) - np.array(y)) < tol)


# ===========================================================================
# watson_watt.crlb
# ===========================================================================

def test_watson_watt_crlb_is_positive():
    crlb_val = watson_watt.crlb(snr=10.0, num_samples=100)
    assert crlb_val > 0


def test_watson_watt_crlb_decreases_with_snr():
    """Higher SNR → tighter bound."""
    crlb_low  = watson_watt.crlb(snr=0.0,  num_samples=100)
    crlb_high = watson_watt.crlb(snr=20.0, num_samples=100)
    assert crlb_high < crlb_low


def test_watson_watt_crlb_decreases_with_samples():
    """More samples → tighter bound."""
    crlb_few  = watson_watt.crlb(snr=10.0, num_samples=10)
    crlb_many = watson_watt.crlb(snr=10.0, num_samples=1000)
    assert crlb_many < crlb_few


def test_watson_watt_crlb_known_value():
    """
    For SNR=0 dB (linear=1) and num_samples=1:
        CRLB = 1 / (1 * 1) = 1.0 rad²
    """
    crlb_val = watson_watt.crlb(snr=0.0, num_samples=1)
    assert equal_to_tolerance(crlb_val, 1.0)


# ===========================================================================
# watson_watt.compute_df
# ===========================================================================

def test_watson_watt_compute_df_known_angle():
    """
    With a noiseless reference and Adcock signals, compute_df should recover
    the true angle of arrival exactly.

    For psi_true, x = cos(psi_true), y = sin(psi_true), r = 1:
        vdot(r, x) = cos(psi_true)
        vdot(r, y) = sin(psi_true)
        arctan2(sin, cos) = psi_true
    """
    for psi_true in [0.0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2]:
        r = np.array([1.0 + 0j])
        x = np.array([np.cos(psi_true) + 0j])
        y = np.array([np.sin(psi_true) + 0j])
        psi_est = watson_watt.compute_df(r, x, y)
        assert equal_to_tolerance(psi_est, psi_true, tol=1e-12), \
            f"psi_true={psi_true:.4f}, psi_est={psi_est:.4f}"


def test_watson_watt_compute_df_returns_scalar():
    r = np.array([1.0 + 0j])
    x = np.array([1.0 + 0j])
    y = np.array([0.0 + 0j])
    psi_est = watson_watt.compute_df(r, x, y)
    assert np.isscalar(psi_est) or np.ndim(psi_est) == 0


def test_watson_watt_compute_df_multi_sample():
    """compute_df should handle arrays with multiple samples (vdot sums)."""
    psi_true = np.pi / 4
    n = 50
    r = np.ones(n, dtype=complex)
    x = np.full(n, np.cos(psi_true), dtype=complex)
    y = np.full(n, np.sin(psi_true), dtype=complex)
    psi_est = watson_watt.compute_df(r, x, y)
    assert equal_to_tolerance(psi_est, psi_true, tol=1e-12)


# ===========================================================================
# interferometer.crlb
# ===========================================================================

def test_interferometer_crlb_is_positive():
    crlb_val = interferometer.crlb(snr1=10.0, snr2=10.0, num_samples=100,
                                   d_lam=0.5, psi_true=0.0)
    assert crlb_val > 0


def test_interferometer_crlb_decreases_with_snr():
    crlb_low  = interferometer.crlb(snr1=0.0,  snr2=0.0,  num_samples=100,
                                    d_lam=0.5, psi_true=0.0)
    crlb_high = interferometer.crlb(snr1=20.0, snr2=20.0, num_samples=100,
                                    d_lam=0.5, psi_true=0.0)
    assert crlb_high < crlb_low


def test_interferometer_crlb_decreases_with_samples():
    crlb_few  = interferometer.crlb(snr1=10.0, snr2=10.0, num_samples=10,
                                    d_lam=0.5, psi_true=0.0)
    crlb_many = interferometer.crlb(snr1=10.0, snr2=10.0, num_samples=1000,
                                    d_lam=0.5, psi_true=0.0)
    assert crlb_many < crlb_few


def test_interferometer_crlb_increases_near_endfire():
    """
    Geometry degrades near end-fire (psi → ±π/2): cos(psi) → 0 so the
    CRLB should grow as psi approaches ±π/2.
    """
    crlb_broadside = interferometer.crlb(snr1=10.0, snr2=10.0, num_samples=100,
                                         d_lam=0.5, psi_true=0.0)
    crlb_endfire   = interferometer.crlb(snr1=10.0, snr2=10.0, num_samples=100,
                                         d_lam=0.5, psi_true=np.pi / 3)
    assert crlb_endfire > crlb_broadside


def test_interferometer_crlb_known_value():
    """
    With SNR1=SNR2=0 dB, M=1, d/λ=0.5, psi=0:
        snr_eff = 0.5
        CRLB = (1/(2*0.5)) * (1/(2*π*0.5))² = 1 * (1/π)²
    """
    expected = (1.0 / (2.0 * 0.5)) * (1.0 / (2.0 * np.pi * 0.5)) ** 2
    crlb_val = interferometer.crlb(snr1=0.0, snr2=0.0, num_samples=1,
                                   d_lam=0.5, psi_true=0.0)
    assert equal_to_tolerance(crlb_val, expected, tol=1e-10)


# ===========================================================================
# interferometer.compute_df
# ===========================================================================

def test_interferometer_compute_df_known_angle():
    """
    With noiseless signals related by the interferometer phase,
    compute_df should recover the true angle exactly.

    If phi = 2π * d_lam * sin(psi_true), then:
        x1 = [1+0j], x2 = [exp(j*phi)]
        y = vdot(x1, x2) = exp(j*phi)
        phi_est = angle(y) = phi
        psi_est = arcsin(phi / (2π*d_lam)) = psi_true
    """
    d_lam = 0.5
    for psi_true in [0.0, np.pi / 6, -np.pi / 4]:
        phi = 2.0 * np.pi * d_lam * np.sin(psi_true)
        x1 = np.array([1.0 + 0j])
        x2 = np.array([np.exp(1j * phi)])
        psi_est = interferometer.compute_df(x1, x2, d_lam)
        assert equal_to_tolerance(psi_est, psi_true, tol=1e-12), \
            f"psi_true={psi_true:.4f}, psi_est={psi_est:.4f}"


def test_interferometer_compute_df_multi_sample():
    """Result should be consistent with a multi-sample noiseless input."""
    d_lam = 0.5
    psi_true = np.pi / 6
    phi = 2.0 * np.pi * d_lam * np.sin(psi_true)
    n = 50
    x1 = np.ones(n, dtype=complex)
    x2 = np.exp(1j * phi) * np.ones(n, dtype=complex)
    psi_est = interferometer.compute_df(x1, x2, d_lam)
    assert equal_to_tolerance(psi_est, psi_true, tol=1e-12)


# ===========================================================================
# directional.crlb
# ===========================================================================

_g_adcock, _g_dot_adcock = make_gain_functions('adcock', d_lam=0.5, psi_0=0.0)
_PSI_SAMPLES = np.linspace(-np.pi, np.pi, 9, endpoint=False)
_PSI_TRUE    = np.pi / 4


def test_directional_crlb_is_positive():
    crlb_val = directional.crlb(snr=10.0, num_samples=10,
                                g=_g_adcock, g_dot=_g_dot_adcock,
                                psi_samples=_PSI_SAMPLES, psi_true=_PSI_TRUE)
    assert crlb_val > 0


def test_directional_crlb_decreases_with_snr():
    crlb_low  = directional.crlb(snr=0.0,  num_samples=10,
                                 g=_g_adcock, g_dot=_g_dot_adcock,
                                 psi_samples=_PSI_SAMPLES, psi_true=_PSI_TRUE)
    crlb_high = directional.crlb(snr=20.0, num_samples=10,
                                 g=_g_adcock, g_dot=_g_dot_adcock,
                                 psi_samples=_PSI_SAMPLES, psi_true=_PSI_TRUE)
    assert crlb_high < crlb_low


def test_directional_crlb_decreases_with_samples():
    crlb_few  = directional.crlb(snr=10.0, num_samples=1,
                                 g=_g_adcock, g_dot=_g_dot_adcock,
                                 psi_samples=_PSI_SAMPLES, psi_true=_PSI_TRUE)
    crlb_many = directional.crlb(snr=10.0, num_samples=100,
                                 g=_g_adcock, g_dot=_g_dot_adcock,
                                 psi_samples=_PSI_SAMPLES, psi_true=_PSI_TRUE)
    assert crlb_many < crlb_few


def test_directional_crlb_omni_smaller_than_adcock_few_samples():
    """
    With only one steering angle, the omni antenna has zero gradient →
    infinite CRLB, while Adcock has a finite bound (g_dot ≠ 0 away from
    the null).  The Adcock bound should therefore be finite.
    """
    g_omni, g_dot_omni = make_gain_functions('omni', d_lam=0.5, psi_0=0.0)
    # Adcock CRLB should be finite
    crlb_adcock = directional.crlb(snr=10.0, num_samples=10,
                                   g=_g_adcock, g_dot=_g_dot_adcock,
                                   psi_samples=_PSI_SAMPLES, psi_true=_PSI_TRUE)
    assert np.isfinite(crlb_adcock)


# ===========================================================================
# directional.compute_df
# ===========================================================================

_PSI_RES = 1e-4  # resolution for iterative search in compute_df tests


def _make_s(g, psi_samples, psi_true, n_temporal=1):
    """Noiseless gain measurements at each steering angle, shape (num_angles, n_temporal)."""
    return np.tile(g(psi_samples - psi_true)[:, np.newaxis], (1, n_temporal))


def test_directional_compute_df_returns_scalar():
    s = _make_s(_g_adcock, _PSI_SAMPLES, _PSI_TRUE)
    psi_est = directional.compute_df(s, _PSI_SAMPLES, _g_adcock, psi_res=_PSI_RES)
    assert np.size(psi_est) == 1


def test_directional_compute_df_at_45_degrees():
    """Noiseless adcock measurements at π/4 should be recovered to within psi_res."""
    s = _make_s(_g_adcock, _PSI_SAMPLES, _PSI_TRUE)
    psi_est = directional.compute_df(s, _PSI_SAMPLES, _g_adcock, psi_res=_PSI_RES)
    assert abs(psi_est - _PSI_TRUE) < 3 * _PSI_RES, \
        f"Estimate {np.rad2deg(float(psi_est)):.4f}°, expected {np.rad2deg(_PSI_TRUE):.4f}°"


def test_directional_compute_df_at_negative_angle():
    """Noiseless adcock measurements at -π/6 should be recovered accurately."""
    psi_true = -np.pi / 6
    s = _make_s(_g_adcock, _PSI_SAMPLES, psi_true)
    psi_est = directional.compute_df(s, _PSI_SAMPLES, _g_adcock, psi_res=_PSI_RES)
    assert abs(psi_est - psi_true) < 3 * _PSI_RES, \
        f"Estimate {np.rad2deg(float(psi_est)):.4f}°, expected {np.rad2deg(psi_true):.4f}°"


def test_directional_compute_df_rectangular_aperture():
    """Works with a rectangular-aperture gain function."""
    g_rect, _ = make_gain_functions('rectangular', d_lam=5.0, psi_0=0.0)
    psi_true = np.pi / 8
    s = _make_s(g_rect, _PSI_SAMPLES, psi_true)
    psi_est = directional.compute_df(s, _PSI_SAMPLES, g_rect, psi_res=_PSI_RES)
    assert abs(psi_est - psi_true) < 3 * _PSI_RES, \
        f"Estimate {np.rad2deg(float(psi_est)):.4f}°, expected {np.rad2deg(psi_true):.4f}°"


def test_directional_compute_df_multi_temporal_sample():
    """Noiseless repeated temporal samples should give the same result as one sample."""
    s = _make_s(_g_adcock, _PSI_SAMPLES, _PSI_TRUE, n_temporal=5)
    psi_est = directional.compute_df(s, _PSI_SAMPLES, _g_adcock, psi_res=_PSI_RES)
    assert abs(psi_est - _PSI_TRUE) < 3 * _PSI_RES, \
        f"Estimate {np.rad2deg(float(psi_est)):.4f}°, expected {np.rad2deg(_PSI_TRUE):.4f}°"
