import numpy as np

from ewgeo.noise.model import get_thermal_noise
from ewgeo.utils.constants import boltzmann, ref_temp
from ewgeo.utils.unit_conversions import lin_to_db


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
