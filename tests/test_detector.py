import numpy as np
import pytest

from ewgeo.detector import squareLaw, xcorr


# ---------------------------------------------------------------------------
# Shared parameters
# ---------------------------------------------------------------------------

NUM_SAMPLES  = 64     # number of complex samples per detection event
NOISE_VAR    = 1.0    # noise variance
PROB_FA      = 1e-3   # probability of false alarm
PROB_D       = 0.9    # desired probability of detection
SNR_HIGH_DB  = 20.0   # clearly detectable signal
SNR_LOW_DB   = -20.0  # clearly undetectable signal


# ===========================================================================
# squareLaw.det_test
# ===========================================================================

def test_squarelaw_det_test_pure_noise_shape():
    """Pure noise input should return a boolean result for each trial."""
    rng = np.random.default_rng(0)
    z = rng.standard_normal((NUM_SAMPLES, 100)) + 1j * rng.standard_normal((NUM_SAMPLES, 100))
    result = squareLaw.det_test(z, NOISE_VAR, PROB_FA)
    assert result.shape == (100,)


def test_squarelaw_det_test_returns_bool():
    rng = np.random.default_rng(1)
    # 1-D input (single detection event); previously failed due to putmask on scalar
    z = rng.standard_normal((NUM_SAMPLES,)) + 1j * rng.standard_normal((NUM_SAMPLES,))
    result = squareLaw.det_test(z, NOISE_VAR, PROB_FA)
    assert result.dtype == bool or np.issubdtype(result.dtype, np.bool_)


def test_squarelaw_det_test_high_snr_detects():
    """A strong signal well above the noise should almost always be detected."""
    rng = np.random.default_rng(2)
    snr_lin  = 10 ** (SNR_HIGH_DB / 10)
    sig_amp  = np.sqrt(snr_lin * NOISE_VAR)
    # M complex samples of a pure tone + noise
    num_trials = 200
    signal  = sig_amp * np.ones((NUM_SAMPLES, num_trials))
    noise   = (rng.standard_normal((NUM_SAMPLES, num_trials)) +
               1j * rng.standard_normal((NUM_SAMPLES, num_trials))) * np.sqrt(NOISE_VAR / 2)
    z = signal + noise
    det = squareLaw.det_test(z, NOISE_VAR, PROB_FA)
    pd_empirical = np.mean(det)
    assert pd_empirical > 0.8, f"High-SNR Pd too low: {pd_empirical:.3f}"


def test_squarelaw_det_test_low_snr_rarely_detects():
    """A weak signal far below the noise should rarely be detected."""
    rng = np.random.default_rng(3)
    snr_lin  = 10 ** (SNR_LOW_DB / 10)
    sig_amp  = np.sqrt(snr_lin * NOISE_VAR)
    num_trials = 500
    signal  = sig_amp * np.ones((NUM_SAMPLES, num_trials))
    noise   = (rng.standard_normal((NUM_SAMPLES, num_trials)) +
               1j * rng.standard_normal((NUM_SAMPLES, num_trials))) * np.sqrt(NOISE_VAR / 2)
    z = signal + noise
    det = squareLaw.det_test(z, NOISE_VAR, PROB_FA)
    # Should be close to PROB_FA (false-alarm limited); allow generous margin
    assert np.mean(det) < 0.05, f"Low-SNR Pd too high: {np.mean(det):.3f}"


# ===========================================================================
# squareLaw.min_sinr
# ===========================================================================

def test_squarelaw_min_sinr_returns_array():
    xi = squareLaw.min_sinr(PROB_FA, PROB_D, NUM_SAMPLES)
    assert xi is not None
    assert np.size(xi) >= 1


def test_squarelaw_min_sinr_positive():
    xi = squareLaw.min_sinr(PROB_FA, PROB_D, NUM_SAMPLES)
    assert np.all(xi > -30), "min SNR should be reasonable (> -30 dB)"


def test_squarelaw_min_sinr_higher_pd_needs_higher_snr():
    xi_low  = squareLaw.min_sinr(PROB_FA, 0.5, NUM_SAMPLES)
    xi_high = squareLaw.min_sinr(PROB_FA, 0.9, NUM_SAMPLES)
    assert np.all(xi_high > xi_low), \
        f"Higher Pd should need higher SNR: {xi_high} vs {xi_low}"


def test_squarelaw_min_sinr_more_samples_lower_snr():
    xi_few  = squareLaw.min_sinr(PROB_FA, PROB_D, 8)
    xi_many = squareLaw.min_sinr(PROB_FA, PROB_D, 128)
    assert np.all(xi_many < xi_few), \
        f"More samples should reduce required SNR: {xi_many} vs {xi_few}"


# ===========================================================================
# xcorr.det_test
# ===========================================================================

def _unit_power_reference():
    """Return a deterministic unit-power complex reference signal (M,1).

    The xcorr threshold formula sigma_0_sq = M * noise_var^2 / 2 assumes
    |y2_k|^2 = 1 for all k.  Using a random y2 violates this and inflates
    the empirical PFA.
    """
    t = np.arange(NUM_SAMPLES)
    return np.exp(1j * 2 * np.pi * 0.1 * t).reshape(-1, 1)


def test_xcorr_det_test_correlated_detects():
    """When y1 is a noisy version of the reference at high SNR, detection occurs."""
    rng = np.random.default_rng(10)
    snr_lin = 10 ** (SNR_HIGH_DB / 10)
    sig_amp = np.sqrt(snr_lin * NOISE_VAR)
    y2 = _unit_power_reference()           # (M, 1), unit power
    y1 = sig_amp * y2 + (rng.standard_normal((NUM_SAMPLES, 1)) +
                         1j * rng.standard_normal((NUM_SAMPLES, 1))) * np.sqrt(NOISE_VAR / 2)
    result = xcorr.det_test(y1, y2, NOISE_VAR, NUM_SAMPLES, PROB_FA)
    assert bool(np.all(result))


def test_xcorr_det_test_uncorrelated_no_detection():
    """Pure noise against the unit-power reference should give ~PROB_FA detections."""
    rng = np.random.default_rng(11)
    num_trials = 5000
    y2 = _unit_power_reference()           # (M, 1), unit power
    # y1 is pure noise: real and imaginary each N(0, noise_var/2)
    y1 = (rng.standard_normal((NUM_SAMPLES, num_trials)) +
          1j * rng.standard_normal((NUM_SAMPLES, num_trials))) * np.sqrt(NOISE_VAR / 2)
    results = xcorr.det_test(y1, y2, NOISE_VAR, NUM_SAMPLES, PROB_FA)
    pfa_empirical = np.mean(results)
    # Allow a generous 10× margin given finite-sample variance
    assert pfa_empirical < 10 * PROB_FA, \
        f"Uncorrelated Pfa too high: {pfa_empirical:.4f}"


def test_xcorr_det_test_returns_bool():
    y2 = _unit_power_reference()
    rng = np.random.default_rng(12)
    y1 = rng.standard_normal((NUM_SAMPLES, 1)) + 1j * rng.standard_normal((NUM_SAMPLES, 1))
    result = xcorr.det_test(y1, y2, NOISE_VAR, NUM_SAMPLES, PROB_FA)
    assert result.dtype == bool or np.issubdtype(result.dtype, np.bool_)


# ===========================================================================
# xcorr.min_sinr
# ===========================================================================

def test_xcorr_min_sinr_returns_value():
    xi = xcorr.min_sinr(PROB_FA, PROB_D,
                        corr_time=1e-3,
                        pulse_duration=1e-3,
                        bw_noise=1e6,
                        bw_signal=1e6)
    assert xi is not None
    assert np.size(xi) >= 1


def test_xcorr_min_sinr_higher_pd_needs_higher_snr():
    common = dict(corr_time=1e-3, pulse_duration=1e-3, bw_noise=1e6, bw_signal=1e6)
    xi_low  = xcorr.min_sinr(PROB_FA, 0.5, **common)
    xi_high = xcorr.min_sinr(PROB_FA, 0.9, **common)
    assert np.all(xi_high >= xi_low)
