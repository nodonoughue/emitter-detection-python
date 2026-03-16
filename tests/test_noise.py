import numpy as np

from ewgeo.noise.model import (get_thermal_noise, get_atmospheric_noise_temp,
                                get_sun_noise_temp, get_moon_noise_temp,
                                get_cosmic_noise_temp, get_ground_noise_temp)
from ewgeo.utils.constants import boltzmann, ref_temp
from ewgeo.utils.unit_conversions import lin_to_db, db_to_lin


def equal_to_tolerance(x, y, tol=1e-6):
    if np.size(x) != np.size(y):
        return False
    return np.all(np.fabs(np.array(x) - np.array(y)) < tol)


# ===========================================================================
# get_thermal_noise
# ===========================================================================

def test_thermal_noise_formula():
    """N = lin_to_db(k * T_ref * BW) with NF=0, T_ext=0."""
    bw = 1e6  # Hz
    expected = lin_to_db(boltzmann * ref_temp * bw)
    result = get_thermal_noise(bw, noise_figure_db=0, temp_ext_k=0)
    assert equal_to_tolerance(result, expected, tol=1e-6)


def test_thermal_noise_value_magnitude():
    """At 1 MHz BW the thermal floor should be around -144 dBW."""
    result = get_thermal_noise(1e6, noise_figure_db=0, temp_ext_k=0)
    assert -150 < result < -135, f"Unexpected thermal noise value: {result:.2f} dBW"


def test_thermal_noise_nf_adds_linearly():
    """Noise figure adds directly to the dB result."""
    nf = 5.0  # dB
    n0 = get_thermal_noise(1e6, noise_figure_db=0.0, temp_ext_k=0)
    n1 = get_thermal_noise(1e6, noise_figure_db=nf,  temp_ext_k=0)
    assert equal_to_tolerance(n1 - n0, nf, tol=1e-9)


def test_thermal_noise_increases_with_bandwidth():
    """Doubling bandwidth adds 3 dB."""
    n1 = get_thermal_noise(1e6, noise_figure_db=0, temp_ext_k=0)
    n2 = get_thermal_noise(2e6, noise_figure_db=0, temp_ext_k=0)
    assert equal_to_tolerance(n2 - n1, 10 * np.log10(2), tol=1e-6)


def test_thermal_noise_external_temp_increases_noise():
    n0 = get_thermal_noise(1e6, noise_figure_db=0, temp_ext_k=0)
    n1 = get_thermal_noise(1e6, noise_figure_db=0, temp_ext_k=290)
    assert n1 > n0


def test_thermal_noise_external_temp_290k_adds_3dB():
    """Adding T_ext = T_ref = 290 K doubles the noise power (+3 dB)."""
    n0 = get_thermal_noise(1e6, noise_figure_db=0, temp_ext_k=0)
    n1 = get_thermal_noise(1e6, noise_figure_db=0, temp_ext_k=290)
    assert equal_to_tolerance(n1 - n0, 10 * np.log10(2), tol=1e-6)


def test_thermal_noise_vectorized_bandwidth():
    bw_vec = np.array([1e6, 2e6, 4e6])
    result = get_thermal_noise(bw_vec, noise_figure_db=0, temp_ext_k=0)
    assert result.shape == (3,)
    # Each doubling should add ~3 dB
    assert result[1] > result[0]
    assert result[2] > result[1]


# ===========================================================================
# get_atmospheric_noise_temp
# ===========================================================================

def test_atmospheric_noise_temp_positive():
    """Atmospheric noise temperature must be non-negative."""
    t = get_atmospheric_noise_temp(freq_hz=10e9)
    assert t >= 0, f"Expected non-negative, got {t}"


def test_atmospheric_noise_temp_increases_with_lower_elevation():
    """Lower elevation angle → longer atmospheric path → higher noise temp."""
    t_zenith  = get_atmospheric_noise_temp(freq_hz=10e9, el_angle_deg=90)
    t_low     = get_atmospheric_noise_temp(freq_hz=10e9, el_angle_deg=10)
    assert t_low > t_zenith, \
        f"Expected low-elevation temp ({t_low:.2f} K) > zenith temp ({t_zenith:.2f} K)"


def test_atmospheric_noise_temp_vectorized_freq():
    """Vectorized frequency input should return an array."""
    freqs = np.array([1e9, 10e9, 30e9])
    t = get_atmospheric_noise_temp(freq_hz=freqs)
    assert np.shape(t) == (3,), f"Expected shape (3,), got {np.shape(t)}"
    assert np.all(t >= 0), "All atmospheric noise temps should be non-negative"


# ===========================================================================
# get_sun_noise_temp
# ===========================================================================

def test_sun_noise_temp_positive_in_band():
    """Sun noise temp should be positive at 1 GHz (within the ITU reference range)."""
    t = get_sun_noise_temp(1e9)
    assert t > 0, f"Expected positive sun noise temp at 1 GHz, got {t}"


def test_sun_noise_temp_known_value_1ghz():
    """At 1 GHz the ITU reference value is ~2e5 K."""
    t = get_sun_noise_temp(1e9)
    assert 1e5 < t < 4e5, f"1 GHz sun noise temp out of expected range: {t:.0f} K"


def test_sun_noise_temp_decreases_with_frequency():
    """Sun noise temp decreases with increasing frequency in the GHz range."""
    t_low  = get_sun_noise_temp(1e9)
    t_high = get_sun_noise_temp(10e9)
    assert t_high < t_low, \
        f"Expected sun temp to decrease: {t_low:.0f} K at 1 GHz, {t_high:.0f} K at 10 GHz"


def test_sun_noise_temp_zero_out_of_band():
    """Returns 0 outside the defined frequency range (extrapolation = 0)."""
    assert get_sun_noise_temp(1e3) == 0,   "Below 50 MHz should return 0"
    assert get_sun_noise_temp(200e9) == 0, "Above 100 GHz should return 0"


def test_sun_noise_temp_vectorized():
    """Vectorized frequency input returns an array."""
    freqs = np.array([1e9, 5e9, 10e9])
    t = get_sun_noise_temp(freqs)
    assert np.shape(t) == (3,), f"Expected shape (3,), got {np.shape(t)}"


# ===========================================================================
# get_moon_noise_temp
# ===========================================================================

def test_moon_noise_temp_known_value():
    """Moon noise temp is the arithmetic mean of 140 K and 280 K = 210 K."""
    assert get_moon_noise_temp() == 210, \
        f"Expected 210 K, got {get_moon_noise_temp()}"


# ===========================================================================
# get_cosmic_noise_temp
# ===========================================================================

def test_cosmic_noise_temp_positive():
    """Cosmic noise temperature must be positive."""
    t = get_cosmic_noise_temp(freq_hz=1e9)
    assert t > 0, f"Expected positive cosmic noise temp, got {t}"


def test_cosmic_noise_temp_high_freq_approaches_cmb():
    """Above 2 GHz the galactic term is replaced by 2.7 K (CMB); result is near that floor."""
    t = get_cosmic_noise_temp(freq_hz=10e9)
    # Atmospheric loss reduces the 2.7 K slightly; should be less than 10 K
    assert 0 < t < 10, f"High-freq cosmic temp should be ~2.7 K, got {t:.2f} K"


def test_cosmic_noise_temp_decreases_with_frequency():
    """Below 2 GHz galactic background dominates and falls steeply with frequency."""
    t_low  = get_cosmic_noise_temp(freq_hz=100e6)
    t_high = get_cosmic_noise_temp(freq_hz=1e9)
    assert t_high < t_low, \
        f"Expected cosmic temp to decrease: {t_low:.0f} K at 100 MHz, {t_high:.0f} K at 1 GHz"


def test_cosmic_noise_temp_sun_contribution():
    """Pointing antenna at sun (high gain) should increase cosmic noise temp."""
    t_no_sun   = get_cosmic_noise_temp(freq_hz=1e9, gain_sun_dbi=-np.inf)
    t_with_sun = get_cosmic_noise_temp(freq_hz=1e9, gain_sun_dbi=30.0)
    assert t_with_sun > t_no_sun, "Sun contribution should raise cosmic noise temp"


def test_cosmic_noise_temp_vectorized():
    """Vectorized frequency input returns an array."""
    freqs = np.array([100e6, 1e9, 10e9])
    t = get_cosmic_noise_temp(freq_hz=freqs)
    assert np.shape(t) == (3,), f"Expected shape (3,), got {np.shape(t)}"


# ===========================================================================
# get_ground_noise_temp
# ===========================================================================

def test_ground_noise_temp_positive():
    """Ground noise temperature must be positive."""
    t = get_ground_noise_temp()
    assert t > 0, f"Expected positive ground noise temp, got {t}"


def test_ground_noise_temp_known_default_value():
    """Default params: pi * db_to_lin(-5) * 1 * 290 / (4*pi) = db_to_lin(-5) * 290/4."""
    expected = db_to_lin(-5) * ref_temp / 4
    t = get_ground_noise_temp()
    assert abs(t - expected) < 1e-6, f"Expected {expected:.4f} K, got {t:.4f} K"


def test_ground_noise_temp_increases_with_gain():
    """Higher antenna gain toward ground → higher ground noise temp."""
    t_low  = get_ground_noise_temp(ant_gain_ground_dbi=-10)
    t_high = get_ground_noise_temp(ant_gain_ground_dbi=0)
    assert t_high > t_low, "Higher ground gain should increase noise temp"


def test_ground_noise_temp_increases_with_emissivity():
    """Higher ground emissivity → higher ground noise temp."""
    t_low  = get_ground_noise_temp(ground_emissivity=0.5)
    t_high = get_ground_noise_temp(ground_emissivity=1.0)
    assert t_high > t_low, "Higher emissivity should increase noise temp"
