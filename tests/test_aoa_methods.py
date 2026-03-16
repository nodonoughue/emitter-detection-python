import numpy as np

from ewgeo.aoa import watson_watt, interferometer
from ewgeo.utils.unit_conversions import db_to_lin


def equal_to_tolerance(x, y, tol=1e-9):
    if np.size(x) != np.size(y):
        return False
    return np.all(np.fabs(np.array(x) - np.array(y)) < tol)


# ===========================================================================
# watson_watt.crlb
# ===========================================================================

def test_ww_crlb_formula_1sample_0dB():
    """CRLB(0 dB, M=1) = 1 / (1 * 1) = 1.0 rad²."""
    assert equal_to_tolerance(watson_watt.crlb(0.0, 1), 1.0)


def test_ww_crlb_formula_10dB():
    """CRLB(10 dB, M=1) = 1 / (1 * 10) = 0.1 rad²."""
    assert equal_to_tolerance(watson_watt.crlb(10.0, 1), 0.1)


def test_ww_crlb_formula_more_samples():
    """CRLB(0 dB, M=10) = 1 / 10 = 0.1 rad²."""
    assert equal_to_tolerance(watson_watt.crlb(0.0, 10), 0.1)


def test_ww_crlb_decreases_with_snr():
    """Higher SNR → tighter bound."""
    assert watson_watt.crlb(10.0, 1) < watson_watt.crlb(0.0, 1)


def test_ww_crlb_decreases_with_samples():
    """More samples → tighter bound."""
    assert watson_watt.crlb(0.0, 10) < watson_watt.crlb(0.0, 1)


def test_ww_crlb_is_positive():
    assert watson_watt.crlb(5.0, 5) > 0


# ===========================================================================
# watson_watt.compute_df
# ===========================================================================

def test_ww_compute_df_zero_degrees():
    """Signal at 0°: x=1, y=0, r=1 → arctan2(0, 1) = 0."""
    r = np.array([1.0])
    x = np.array([1.0])
    y = np.array([0.0])
    assert equal_to_tolerance(watson_watt.compute_df(r, x, y), 0.0)


def test_ww_compute_df_ninety_degrees():
    """Signal at 90°: x=0, y=1, r=1 → arctan2(1, 0) = π/2."""
    r = np.array([1.0])
    x = np.array([0.0])
    y = np.array([1.0])
    assert equal_to_tolerance(watson_watt.compute_df(r, x, y), np.pi / 2)


def test_ww_compute_df_45_degrees():
    c = np.cos(np.pi / 4)
    r = np.array([1.0])
    x = np.array([c])
    y = np.array([c])
    assert equal_to_tolerance(watson_watt.compute_df(r, x, y), np.pi / 4, tol=1e-9)


def test_ww_compute_df_negative_angle():
    """Signal at -45°: x=cos(45), y=-sin(45) → -π/4."""
    c = np.cos(np.pi / 4)
    r = np.array([1.0])
    x = np.array([ c])
    y = np.array([-c])
    assert equal_to_tolerance(watson_watt.compute_df(r, x, y), -np.pi / 4, tol=1e-9)


def test_ww_compute_df_multi_sample():
    """Multi-sample signals (complex dot product) should still recover angle."""
    n = 32
    psi = np.pi / 6  # 30 degrees
    t = np.linspace(0, 1, n)
    r = np.cos(2 * np.pi * 1e6 * t)
    x = np.cos(psi) * r
    y = np.sin(psi) * r
    psi_est = watson_watt.compute_df(r, x, y)
    assert equal_to_tolerance(psi_est, psi, tol=1e-9)


# ===========================================================================
# interferometer.crlb
# ===========================================================================

def test_interf_crlb_equal_snr_formula():
    """With equal SNR, snr_eff = snr_lin/2; check closed-form value."""
    snr1 = snr2 = 10.0  # dB
    M = 1
    d_lam = 0.5
    psi = 0.0
    snr_lin = db_to_lin(snr1)
    snr_eff = snr_lin / 2.0  # 1/(1/s+1/s) = s/2
    expected = (1.0 / (2.0 * M * snr_eff)) * (1.0 / (2.0 * np.pi * d_lam * np.cos(psi))) ** 2
    assert equal_to_tolerance(interferometer.crlb(snr1, snr2, M, d_lam, psi), expected)


def test_interf_crlb_positive():
    assert interferometer.crlb(0.0, 0.0, 1, 0.5, 0.0) > 0


def test_interf_crlb_decreases_with_snr():
    c = interferometer.crlb(10.0, 10.0, 1, 0.5, 0.0)
    c_low = interferometer.crlb(0.0, 0.0, 1, 0.5, 0.0)
    assert c < c_low


def test_interf_crlb_decreases_with_samples():
    c1 = interferometer.crlb(10.0, 10.0, 1,  0.5, 0.0)
    c10 = interferometer.crlb(10.0, 10.0, 10, 0.5, 0.0)
    assert c10 < c1


def test_interf_crlb_decreases_with_larger_baseline():
    c_half = interferometer.crlb(10.0, 10.0, 1, 0.5, 0.0)
    c_one  = interferometer.crlb(10.0, 10.0, 1, 1.0, 0.0)
    assert c_one < c_half


# ===========================================================================
# interferometer.compute_df
# ===========================================================================

def test_interf_compute_df_zero_phase():
    """Signals with zero phase difference → 0° arrival."""
    n = 64
    sig = (np.random.default_rng(0).standard_normal(n) +
           1j * np.random.default_rng(1).standard_normal(n))
    x1 = sig
    x2 = sig  # zero phase shift
    psi_est = interferometer.compute_df(x1, x2, d_lam=0.5)
    assert equal_to_tolerance(psi_est, 0.0, tol=1e-6)


def test_interf_compute_df_known_angle():
    """Known phase shift → known angle of arrival."""
    d_lam = 0.5
    psi_true = np.pi / 6  # 30°
    phi = 2 * np.pi * d_lam * np.sin(psi_true)
    n = 1
    # Single complex sample to keep it deterministic
    x1 = np.array([1.0 + 0.0j])
    x2 = x1 * np.exp(1j * phi)
    psi_est = interferometer.compute_df(x1, x2, d_lam)
    assert equal_to_tolerance(psi_est, psi_true, tol=1e-9)
