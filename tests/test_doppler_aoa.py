import numpy as np

from ewgeo.aoa import doppler
from ewgeo.utils.constants import speed_of_light


# ---------------------------------------------------------------------------
# Shared parameters — match the run_example setup in doppler.py
# ---------------------------------------------------------------------------

F         = 1e9                     # carrier frequency [Hz]
C         = speed_of_light          # speed of light [m/s]
LAM       = C / F                   # wavelength [m]
RADIUS    = LAM / 2                 # Doppler rotation radius [m]
TS        = 1.0 / (5.0 * F)        # sampling period [s]
M         = 100                     # number of samples
FR        = 1.0 / (TS * M)         # rotation rate [rev/s] — one cycle per M samples
AMPLITUDE = 1.0
PSI_TRUE  = np.pi / 4              # 45° [rad]
PSI_RES   = 0.01                   # desired DF resolution [rad]
SNR_DB    = 10.0


def _make_signals(psi_true=PSI_TRUE):
    """Generate noiseless reference and Doppler signals at the given AoA."""
    phi0 = 0.0
    t_vec = TS * np.arange(M)
    r = AMPLITUDE * np.exp(1j * phi0) * np.exp(1j * 2 * np.pi * F * t_vec)
    x = AMPLITUDE * np.exp(1j * phi0) * np.exp(1j * 2 * np.pi * F * t_vec) \
        * np.exp(1j * 2 * np.pi * F * RADIUS / C * np.cos(2 * np.pi * FR * t_vec - psi_true))
    return r, x


# ===========================================================================
# crlb
# ===========================================================================

def test_crlb_returns_positive():
    c = doppler.crlb(SNR_DB, M, AMPLITUDE, TS, F, RADIUS, FR, PSI_TRUE)
    assert float(c) > 0


def test_crlb_decreases_with_snr():
    c_low  = doppler.crlb(0.0,  M, AMPLITUDE, TS, F, RADIUS, FR, PSI_TRUE)
    c_high = doppler.crlb(20.0, M, AMPLITUDE, TS, F, RADIUS, FR, PSI_TRUE)
    assert c_high < c_low, f"CRLB should decrease with SNR: {c_high} vs {c_low}"


def test_crlb_decreases_with_samples():
    c_few  = doppler.crlb(SNR_DB, 10,  AMPLITUDE, TS, F, RADIUS, 1.0 / (TS * 10),  PSI_TRUE)
    c_many = doppler.crlb(SNR_DB, 200, AMPLITUDE, TS, F, RADIUS, 1.0 / (TS * 200), PSI_TRUE)
    assert c_many < c_few, f"CRLB should decrease with more samples: {c_many} vs {c_few}"


def test_crlb_vectorized_snr():
    """Array SNR input should return a monotonically decreasing array."""
    snr_vec = np.array([0., 5., 10., 15., 20.])
    result = doppler.crlb(snr_vec, M, AMPLITUDE, TS, F, RADIUS, FR, PSI_TRUE)
    assert result.shape == (5,)
    assert np.all(np.diff(result) < 0), "CRLB should be monotonically decreasing in SNR"


def test_crlb_decreases_with_radius():
    """Larger rotation radius → more Doppler phase → lower CRLB."""
    c_small = doppler.crlb(SNR_DB, M, AMPLITUDE, TS, F, LAM * 0.1, FR, PSI_TRUE)
    c_large = doppler.crlb(SNR_DB, M, AMPLITUDE, TS, F, LAM * 1.0, FR, PSI_TRUE)
    assert c_large < c_small, f"Larger radius should lower CRLB: {c_large} vs {c_small}"


def test_crlb_depends_on_angle():
    """CRLB can vary with AoA; verify it's computable at multiple angles."""
    angles = [0.0, np.pi / 6, np.pi / 4, np.pi / 3]
    for psi in angles:
        c = doppler.crlb(SNR_DB, M, AMPLITUDE, TS, F, RADIUS, FR, psi)
        assert float(c) > 0, f"CRLB must be positive at psi={psi:.3f}"


# ===========================================================================
# compute_df
# ===========================================================================

def test_compute_df_returns_scalar():
    r, x = _make_signals()
    psi_est = doppler.compute_df(r, x, TS, F, RADIUS, FR, PSI_RES, -np.pi, np.pi)
    assert np.size(psi_est) == 1


def test_compute_df_at_zero():
    """Signal at 0° should be estimated near 0."""
    r, x = _make_signals(psi_true=0.0)
    psi_est = doppler.compute_df(r, x, TS, F, RADIUS, FR, PSI_RES, -np.pi, np.pi)
    assert abs(psi_est - 0.0) < 3 * PSI_RES, \
        f"Estimate at 0°: {np.rad2deg(float(psi_est)):.2f}°"


def test_compute_df_at_45_degrees():
    """Signal at 45° (π/4) should be estimated near π/4."""
    r, x = _make_signals(psi_true=PSI_TRUE)
    psi_est = doppler.compute_df(r, x, TS, F, RADIUS, FR, PSI_RES, -np.pi, np.pi)
    assert abs(psi_est - PSI_TRUE) < 3 * PSI_RES, \
        f"Estimate at 45°: {np.rad2deg(float(psi_est)):.2f}°"


def test_compute_df_at_negative_angle():
    """Signal at −30° (−π/6) should be estimated near −π/6."""
    psi_true = -np.pi / 6
    r, x = _make_signals(psi_true=psi_true)
    psi_est = doppler.compute_df(r, x, TS, F, RADIUS, FR, PSI_RES, -np.pi, np.pi)
    assert abs(psi_est - psi_true) < 3 * PSI_RES, \
        f"Estimate at -30°: {np.rad2deg(float(psi_est)):.2f}°"


def test_compute_df_coarse_resolution_still_converges():
    """With a coarser resolution the estimate should still be in the right ballpark."""
    r, x = _make_signals()
    psi_est = doppler.compute_df(r, x, TS, F, RADIUS, FR, 0.1, -np.pi, np.pi)
    assert abs(psi_est - PSI_TRUE) < 0.3, \
        f"Coarse estimate at 45°: {np.rad2deg(float(psi_est)):.2f}°"
