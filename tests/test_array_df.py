import numpy as np

from ewgeo.array_df.model import make_steering_vector, compute_array_factor, compute_array_factor_ula
from ewgeo.array_df.perf import crlb_det, crlb_stochastic
from ewgeo.array_df.solvers import beamscan, beamscan_mvdr, music


# ---------------------------------------------------------------------------
# Shared parameters
# ---------------------------------------------------------------------------

N_ELEMENTS    = 8          # ULA elements
D_LAM         = 0.5        # inter-element spacing (wavelengths)
PSI_TRUE      = np.pi / 4  # true source angle [rad]
SNR_DB        = 20.0       # signal-to-noise ratio [dB]
NUM_SNAPSHOTS = 100
NOISE_POWER   = 1.0
SIGNAL_POWER  = NOISE_POWER * 10 ** (SNR_DB / 10)


def _make_data(psi_true=PSI_TRUE, n=N_ELEMENTS, d_lam=D_LAM,
               snr_db=SNR_DB, m=NUM_SNAPSHOTS, noise_power=NOISE_POWER):
    """Return (x, v, v_dot) where x is (N, M) array data for a signal at psi_true."""
    signal_power = noise_power * 10 ** (snr_db / 10)
    v, v_dot = make_steering_vector(d_lam, n)
    a = v(np.array([psi_true]))   # (N, 1)
    rng = np.random.default_rng(0)
    s = (np.sqrt(signal_power / 2) *
         (rng.standard_normal((1, m)) + 1j * rng.standard_normal((1, m))))
    noise = (np.sqrt(noise_power / 2) *
             (rng.standard_normal((n, m)) + 1j * rng.standard_normal((n, m))))
    x = a @ s + noise
    return x, v, v_dot


# ===========================================================================
# make_steering_vector
# ===========================================================================

def test_make_steering_vector_returns_two_callables():
    v, v_dot = make_steering_vector(D_LAM, N_ELEMENTS)
    assert callable(v)
    assert callable(v_dot)


def test_steering_vector_shape():
    v, _ = make_steering_vector(D_LAM, N_ELEMENTS)
    psi = np.linspace(-np.pi / 2, np.pi / 2, 37)
    sv = v(psi)
    assert sv.shape == (N_ELEMENTS, len(psi))


def test_steering_vector_unit_amplitude():
    """Every element of the steering vector must have unit amplitude."""
    v, _ = make_steering_vector(D_LAM, N_ELEMENTS)
    sv = v(np.array([0.0, np.pi / 4, -np.pi / 3]))
    assert np.allclose(np.abs(sv), 1.0)


def test_steering_gradient_shape():
    _, v_dot = make_steering_vector(D_LAM, N_ELEMENTS)
    psi = np.linspace(-np.pi / 2, np.pi / 2, 37)
    dv = v_dot(psi)
    assert dv.shape == (N_ELEMENTS, len(psi))


def test_steering_gradient_numerical_check():
    """Finite-difference approximation of v_dot should match the analytic derivative."""
    v, v_dot = make_steering_vector(D_LAM, N_ELEMENTS)
    psi0 = np.array([np.pi / 6])
    dpsi = 1e-6
    fd = (v(psi0 + dpsi) - v(psi0 - dpsi)) / (2 * dpsi)
    analytic = v_dot(psi0)
    assert np.allclose(np.abs(analytic - fd), 0.0, atol=1e-5)


# ===========================================================================
# compute_array_factor
# ===========================================================================

def test_compute_array_factor_output_shape():
    """Output should be (len(psi), 1) for an array of angles."""
    v, _ = make_steering_vector(D_LAM, N_ELEMENTS)
    h = v(np.array([PSI_TRUE]))[:, 0]          # (N,) matched to steering angle
    psi = np.linspace(-np.pi / 2, np.pi / 2, 37)
    af = compute_array_factor(v, h, psi)
    assert af.shape == (len(psi), 1)


def test_compute_array_factor_peak_at_steering_angle():
    """Matched beamformer h=v(psi_0) must produce its maximum response at psi_0."""
    v, _ = make_steering_vector(D_LAM, N_ELEMENTS)
    psi_0 = PSI_TRUE
    h = v(np.array([psi_0]))[:, 0]
    psi = np.linspace(-np.pi / 2, np.pi / 2, 181)
    af = compute_array_factor(v, h, psi)
    peak_idx = np.argmax(np.abs(af.ravel()))
    assert abs(psi[peak_idx] - psi_0) < 0.05, \
        f"Peak at {np.rad2deg(psi[peak_idx]):.2f}°, expected near {np.rad2deg(psi_0):.2f}°"


def test_compute_array_factor_amplitude_invariant():
    """Scaling h should not change the result because h is normalized internally."""
    v, _ = make_steering_vector(D_LAM, N_ELEMENTS)
    h = v(np.array([PSI_TRUE]))[:, 0]
    psi = np.linspace(-np.pi / 2, np.pi / 2, 37)
    af1 = compute_array_factor(v, h,      psi)
    af2 = compute_array_factor(v, 5.0 * h, psi)
    assert np.allclose(af1, af2, atol=1e-12)


def test_compute_array_factor_complex_output():
    """Output should be complex-valued."""
    v, _ = make_steering_vector(D_LAM, N_ELEMENTS)
    h = v(np.array([PSI_TRUE]))[:, 0]
    psi = np.linspace(-np.pi / 2, np.pi / 2, 37)
    af = compute_array_factor(v, h, psi)
    assert np.iscomplexobj(af)


# ===========================================================================
# compute_array_factor_ula
# ===========================================================================

def test_array_factor_ula_peaks_at_steering_angle():
    """AF at the steering angle psi_0 must equal 1.0."""
    psi_0 = np.pi / 4
    af = compute_array_factor_ula(D_LAM, N_ELEMENTS, psi_0, psi_0=psi_0)
    assert np.isclose(float(af), 1.0), f"AF at steering angle: {af}"


def test_array_factor_ula_peaks_at_broadside_default():
    """Default psi_0 = pi/2; AF at pi/2 should be 1."""
    af = compute_array_factor_ula(D_LAM, N_ELEMENTS, np.pi / 2)
    assert np.isclose(float(af), 1.0), f"AF at broadside: {af}"


def test_array_factor_ula_range():
    """AF must lie in [0, 1] for all angles."""
    psi = np.linspace(-np.pi / 2, np.pi / 2, 181)
    af = compute_array_factor_ula(D_LAM, N_ELEMENTS, psi)
    assert np.all(af >= -1e-10)
    assert np.all(af <= 1.0 + 1e-10)


def test_array_factor_ula_peak_at_scan_angle():
    """The maximum over a dense angular grid should be at the steering angle."""
    psi_0 = np.pi / 6
    psi = np.linspace(-np.pi / 2, np.pi / 2, 1801)
    af = compute_array_factor_ula(D_LAM, N_ELEMENTS, psi, psi_0=psi_0)
    peak_psi = psi[np.argmax(af)]
    assert abs(peak_psi - psi_0) < 0.01, \
        f"AF peak at {peak_psi:.4f}, expected near {psi_0:.4f}"


# ===========================================================================
# crlb_det
# ===========================================================================

def test_crlb_det_returns_positive():
    v, v_dot = make_steering_vector(D_LAM, N_ELEMENTS)
    crlb = crlb_det(SIGNAL_POWER, NOISE_POWER, PSI_TRUE, NUM_SNAPSHOTS, v, v_dot)
    assert float(np.squeeze(crlb)) > 0


def test_crlb_det_decreases_with_snapshots():
    v, v_dot = make_steering_vector(D_LAM, N_ELEMENTS)
    c_few  = float(np.squeeze(crlb_det(SIGNAL_POWER, NOISE_POWER, PSI_TRUE, 10,  v, v_dot)))
    c_many = float(np.squeeze(crlb_det(SIGNAL_POWER, NOISE_POWER, PSI_TRUE, 100, v, v_dot)))
    assert c_many < c_few, f"CRLB should decrease with more snapshots: {c_many} vs {c_few}"


def test_crlb_det_decreases_with_snr():
    v, v_dot = make_steering_vector(D_LAM, N_ELEMENTS)
    c_low  = float(np.squeeze(crlb_det(1.0 * NOISE_POWER,   NOISE_POWER, PSI_TRUE, NUM_SNAPSHOTS, v, v_dot)))
    c_high = float(np.squeeze(crlb_det(100.0 * NOISE_POWER, NOISE_POWER, PSI_TRUE, NUM_SNAPSHOTS, v, v_dot)))
    assert c_high < c_low, f"CRLB should decrease with higher SNR: {c_high} vs {c_low}"


def test_crlb_det_increases_with_fewer_elements():
    """More array elements → lower CRLB (better resolution)."""
    psi = PSI_TRUE
    np_low = 4
    np_high = 16
    v4,  vd4  = make_steering_vector(D_LAM, np_low)
    v16, vd16 = make_steering_vector(D_LAM, np_high)
    c4  = float(np.squeeze(crlb_det(SIGNAL_POWER, NOISE_POWER, psi, NUM_SNAPSHOTS, v4,  vd4)))
    c16 = float(np.squeeze(crlb_det(SIGNAL_POWER, NOISE_POWER, psi, NUM_SNAPSHOTS, v16, vd16)))
    assert c16 < c4, f"CRLB should decrease with more elements: {c16} vs {c4}"


# ===========================================================================
# crlb_stochastic
# ===========================================================================

def test_crlb_stochastic_single_source_positive():
    """Single-source stochastic CRLB must be a positive scalar."""
    v, v_dot = make_steering_vector(D_LAM, N_ELEMENTS)
    c = crlb_stochastic(SIGNAL_POWER, NOISE_POWER, PSI_TRUE, NUM_SNAPSHOTS, v, v_dot)
    assert float(np.squeeze(c)) > 0


def test_crlb_stochastic_multi_source_positive_definite():
    """Multi-source path (the bug path): result must be a 2x2 positive-definite matrix."""
    v, v_dot = make_steering_vector(D_LAM, N_ELEMENTS)
    psi_two = np.array([np.pi / 6, -np.pi / 6])
    cov_two = SIGNAL_POWER * np.eye(2)
    c = crlb_stochastic(cov_two, NOISE_POWER, psi_two, NUM_SNAPSHOTS, v, v_dot)
    assert c.shape == (2, 2), f"Expected (2, 2), got {c.shape}"
    eigvals = np.linalg.eigvalsh(c)
    assert np.all(eigvals > 0), f"CRLB matrix not positive definite: eigenvalues={eigvals}"


def test_crlb_stochastic_decreases_with_snapshots():
    v, v_dot = make_steering_vector(D_LAM, N_ELEMENTS)
    c_few  = float(np.squeeze(crlb_stochastic(SIGNAL_POWER, NOISE_POWER, PSI_TRUE, 10,  v, v_dot)))
    c_many = float(np.squeeze(crlb_stochastic(SIGNAL_POWER, NOISE_POWER, PSI_TRUE, 100, v, v_dot)))
    assert c_many < c_few, f"Stochastic CRLB should decrease with more snapshots: {c_many} vs {c_few}"


def test_crlb_stochastic_decreases_with_snr():
    v, v_dot = make_steering_vector(D_LAM, N_ELEMENTS)
    c_low  = float(np.squeeze(crlb_stochastic(1.0 * NOISE_POWER,   NOISE_POWER, PSI_TRUE, NUM_SNAPSHOTS, v, v_dot)))
    c_high = float(np.squeeze(crlb_stochastic(100.0 * NOISE_POWER, NOISE_POWER, PSI_TRUE, NUM_SNAPSHOTS, v, v_dot)))
    assert c_high < c_low, f"Stochastic CRLB should decrease with higher SNR: {c_high} vs {c_low}"


def test_crlb_stochastic_geq_deterministic():
    """Stochastic CRLB >= deterministic CRLB (stochastic is a looser bound)."""
    v, v_dot = make_steering_vector(D_LAM, N_ELEMENTS)
    c_det  = float(np.squeeze(crlb_det(SIGNAL_POWER, NOISE_POWER, PSI_TRUE, NUM_SNAPSHOTS, v, v_dot)))
    c_stoc = float(np.squeeze(crlb_stochastic(SIGNAL_POWER, NOISE_POWER, PSI_TRUE, NUM_SNAPSHOTS, v, v_dot)))
    assert c_stoc >= c_det - 1e-12, \
        f"Stochastic CRLB ({c_stoc:.6g}) should be >= deterministic ({c_det:.6g})"


# ===========================================================================
# beamscan
# ===========================================================================

def test_beamscan_returns_two_outputs():
    x, v, _ = _make_data()
    p, psi_vec = beamscan(x, v)
    assert p is not None
    assert psi_vec is not None


def test_beamscan_output_shapes():
    x, v, _ = _make_data()
    num_pts = 51
    p, psi_vec = beamscan(x, v, num_points=num_pts)
    assert p.shape == (num_pts,)
    assert psi_vec.shape == (num_pts,)


def test_beamscan_peak_near_true_angle():
    x, v, _ = _make_data()
    p, psi_vec = beamscan(x, v, num_points=201)
    peak_psi = psi_vec[np.argmax(p)]
    assert abs(peak_psi - PSI_TRUE) < 0.1, \
        f"Beamscan peak {peak_psi:.3f} far from truth {PSI_TRUE:.3f}"


def test_beamscan_positive_power():
    x, v, _ = _make_data()
    p, _ = beamscan(x, v)
    assert np.all(p >= 0)


# ===========================================================================
# beamscan_mvdr
# ===========================================================================

def test_beamscan_mvdr_returns_two_outputs():
    x, v, _ = _make_data()
    p, psi_vec = beamscan_mvdr(x, v)
    assert p is not None
    assert psi_vec is not None


def test_beamscan_mvdr_output_shapes():
    x, v, _ = _make_data()
    num_pts = 51
    p, psi_vec = beamscan_mvdr(x, v, num_points=num_pts)
    assert p.shape == (num_pts,)
    assert psi_vec.shape == (num_pts,)


def test_beamscan_mvdr_peak_near_true_angle():
    x, v, _ = _make_data()
    p, psi_vec = beamscan_mvdr(x, v, num_points=201)
    peak_psi = psi_vec[np.argmax(p)]
    assert abs(peak_psi - PSI_TRUE) < 0.15, \
        f"MVDR peak {peak_psi:.3f} far from truth {PSI_TRUE:.3f}"


# ===========================================================================
# music
# ===========================================================================

def test_music_returns_two_outputs():
    x, v, _ = _make_data()
    p, psi_vec = music(x, v, num_sig_dims=1)
    assert p is not None
    assert psi_vec is not None


def test_music_output_shapes():
    x, v, _ = _make_data()
    num_pts = 51
    p, psi_vec = music(x, v, num_sig_dims=1, num_points=num_pts)
    assert p.shape == (num_pts,)
    assert psi_vec.shape == (num_pts,)


def test_music_peak_near_true_angle():
    x, v, _ = _make_data()
    p, psi_vec = music(x, v, num_sig_dims=1, num_points=201)
    peak_psi = psi_vec[np.argmax(p)]
    assert abs(peak_psi - PSI_TRUE) < 0.1, \
        f"MUSIC peak {peak_psi:.3f} far from truth {PSI_TRUE:.3f}"
