"""
Tests for ewgeo.atm — atmospheric propagation models.

Covers:
  reference.py  — get_standard_atmosphere, get_spectroscopic_table_oxygen/water,
                   get_spectral_lines
  model.py      — get_gas_loss_coeff, get_fog_loss_coeff, get_rain_loss_coeff,
                   calc_atm_loss, calc_zenith_loss
"""

import numpy as np
import pytest

from ewgeo.atm import model, reference


# ===========================================================================
# reference.get_standard_atmosphere
# ===========================================================================

def test_standard_atmosphere_sea_level_temp():
    """ISA standard day at 0 m: temperature must be 288.15 K."""
    atm = reference.get_standard_atmosphere(0)
    assert abs(atm.temp - 288.15) < 0.01


def test_standard_atmosphere_sea_level_pressure():
    """ISA standard day at 0 m: pressure must be 1013.25 hPa."""
    atm = reference.get_standard_atmosphere(0)
    assert abs(atm.press - 1013.25) < 0.1


def test_standard_atmosphere_temp_decreases_with_altitude():
    """Temperature should drop when moving from 0 m to 5000 m (troposphere)."""
    atm0 = reference.get_standard_atmosphere(0)
    atm5 = reference.get_standard_atmosphere(5000)
    assert atm5.temp < atm0.temp


def test_standard_atmosphere_pressure_decreases_with_altitude():
    """Pressure should drop with increasing altitude."""
    atm0 = reference.get_standard_atmosphere(0)
    atm5 = reference.get_standard_atmosphere(5000)
    assert atm5.press < atm0.press


def test_standard_atmosphere_water_vapor_positive():
    """Water vapor density and partial pressure should be positive at sea level."""
    atm = reference.get_standard_atmosphere(0)
    assert atm.water_vapor_dens > 0
    assert atm.water_vapor_press > 0


def test_standard_atmosphere_array_input_shapes():
    """Array altitude input should return arrays of matching length."""
    alts = np.array([0., 1000., 5000., 10000.])
    atm = reference.get_standard_atmosphere(alts)
    assert np.size(atm.temp) == len(alts)
    assert np.size(atm.press) == len(alts)


def test_standard_atmosphere_array_monotone():
    """Temperature and pressure must both decrease monotonically in the troposphere."""
    alts = np.array([0., 2000., 4000., 6000., 8000., 10000.])
    atm = reference.get_standard_atmosphere(alts)
    assert np.all(np.diff(atm.temp) < 0)
    assert np.all(np.diff(atm.press) < 0)


def test_standard_atmosphere_above_100km_raises():
    with pytest.raises(ValueError):
        reference.get_standard_atmosphere(110_000)


# ===========================================================================
# reference.get_spectroscopic_table_oxygen / water
# ===========================================================================

def test_spectroscopic_table_oxygen_shape():
    """Table must be (44, 7): 44 spectral lines, 7 columns (freq + 6 coeffs)."""
    tbl = reference.get_spectroscopic_table_oxygen()
    assert tbl.shape == (44, 7)


def test_spectroscopic_table_oxygen_frequencies_positive():
    tbl = reference.get_spectroscopic_table_oxygen()
    assert np.all(tbl[:, 0] > 0)


def test_spectroscopic_table_water_shape():
    """Table must be (35, 7): 35 spectral lines, 7 columns."""
    tbl = reference.get_spectroscopic_table_water()
    assert tbl.shape == (35, 7)


def test_spectroscopic_table_water_frequencies_positive():
    tbl = reference.get_spectroscopic_table_water()
    assert np.all(tbl[:, 0] > 0)


# ===========================================================================
# reference.get_spectral_lines
# ===========================================================================

def test_spectral_lines_no_filter_lengths():
    """Without filters the returned arrays should match the table row counts."""
    f_o, f_w = reference.get_spectral_lines()
    assert len(f_o) == 44
    assert len(f_w) == 35


def test_spectral_lines_in_hz():
    """Spectral line frequencies should be in Hz (all > 1e9)."""
    f_o, f_w = reference.get_spectral_lines()
    assert np.all(f_o > 1e9)
    assert np.all(f_w > 1e9)


def test_spectral_lines_min_filter():
    """min_freq_hz filter should remove lines below the threshold."""
    f_o_all, f_w_all = reference.get_spectral_lines()
    cutoff = 100e9
    f_o, f_w = reference.get_spectral_lines(min_freq_hz=cutoff)
    assert len(f_o) < len(f_o_all)
    assert np.all(f_o >= cutoff)
    assert np.all(f_w >= cutoff)


def test_spectral_lines_max_filter():
    """max_freq_hz filter should remove lines above the threshold."""
    f_o_all, _ = reference.get_spectral_lines()
    cutoff = 100e9
    f_o, f_w = reference.get_spectral_lines(max_freq_hz=cutoff)
    assert len(f_o) < len(f_o_all)
    assert np.all(f_o <= cutoff)


# ===========================================================================
# model.get_gas_loss_coeff
# ===========================================================================

def _sea_level_atm():
    return reference.get_standard_atmosphere(0)


def test_get_gas_loss_coeff_positive():
    """Both oxygen and water vapor coefficients must be positive at 10 GHz."""
    atm = _sea_level_atm()
    coeff_ox, coeff_water = model.get_gas_loss_coeff(10e9, atm.press,
                                                     atm.water_vapor_press, atm.temp)
    assert float(np.squeeze(coeff_ox))    > 0
    assert float(np.squeeze(coeff_water)) > 0


def test_get_gas_loss_coeff_water_resonance():
    """Water vapor coefficient should peak near the 22.235 GHz resonance line."""
    atm = _sea_level_atm()
    _, coeff_off = model.get_gas_loss_coeff(10e9,      atm.press, atm.water_vapor_press, atm.temp)
    _, coeff_res = model.get_gas_loss_coeff(22.235e9,  atm.press, atm.water_vapor_press, atm.temp)
    assert float(np.squeeze(coeff_res)) > float(np.squeeze(coeff_off))


def test_get_gas_loss_coeff_returns_two_values():
    atm = _sea_level_atm()
    result = model.get_gas_loss_coeff(10e9, atm.press, atm.water_vapor_press, atm.temp)
    assert len(result) == 2


def test_get_gas_loss_coeff_vectorized_freq():
    """Array frequency input should return arrays of the same length."""
    atm = _sea_level_atm()
    freqs = np.array([10e9, 22e9, 60e9])
    coeff_ox, coeff_water = model.get_gas_loss_coeff(freqs, atm.press,
                                                     atm.water_vapor_press, atm.temp)
    assert np.size(coeff_ox)    == len(freqs)
    assert np.size(coeff_water) == len(freqs)


# ===========================================================================
# model.get_fog_loss_coeff
# ===========================================================================

def test_get_fog_loss_coeff_positive():
    """Loss coefficient must be positive for any finite cloud density."""
    coeff = model.get_fog_loss_coeff(10e9, cloud_dens=0.1)
    assert float(np.squeeze(coeff)) > 0


def test_get_fog_loss_coeff_increases_with_density():
    """Denser cloud/fog should produce higher attenuation."""
    c_light = model.get_fog_loss_coeff(10e9, cloud_dens=0.05)
    c_dense = model.get_fog_loss_coeff(10e9, cloud_dens=0.5)
    assert float(np.squeeze(c_dense)) > float(np.squeeze(c_light))


def test_get_fog_loss_coeff_explicit_temp():
    """Explicit temperature argument should run without error and return positive."""
    coeff = model.get_fog_loss_coeff(10e9, cloud_dens=0.1, temp_k=280.0)
    assert float(np.squeeze(coeff)) > 0


# ===========================================================================
# model.get_rain_loss_coeff
# ===========================================================================

def test_get_rain_loss_coeff_positive():
    """Rain loss coefficient must be positive for nonzero rainfall."""
    coeff = model.get_rain_loss_coeff(10e9, pol_angle_rad=0, el_angle_rad=0,
                                      rainfall_rate=10)
    assert float(np.squeeze(coeff)) > 0


def test_get_rain_loss_coeff_increases_with_rate():
    """Heavier rainfall should produce higher attenuation."""
    c_light = model.get_rain_loss_coeff(10e9, pol_angle_rad=0, el_angle_rad=0,
                                         rainfall_rate=1)
    c_heavy = model.get_rain_loss_coeff(10e9, pol_angle_rad=0, el_angle_rad=0,
                                         rainfall_rate=50)
    assert float(np.squeeze(c_heavy)) > float(np.squeeze(c_light))


def test_get_rain_loss_coeff_increases_with_frequency():
    """At microwave frequencies, rain loss generally increases with frequency."""
    c_low  = model.get_rain_loss_coeff(5e9,  pol_angle_rad=0, el_angle_rad=0, rainfall_rate=10)
    c_high = model.get_rain_loss_coeff(30e9, pol_angle_rad=0, el_angle_rad=0, rainfall_rate=10)
    assert float(np.squeeze(c_high)) > float(np.squeeze(c_low))


# ===========================================================================
# model.calc_atm_loss
# ===========================================================================

def test_calc_atm_loss_zero_paths():
    """All-zero path lengths should produce zero loss."""
    loss = model.calc_atm_loss(10e9)
    assert float(np.squeeze(loss)) == 0.0


def test_calc_atm_loss_positive_gas_path():
    """A nonzero gas path should produce positive loss."""
    loss = model.calc_atm_loss(10e9, gas_path_len_m=1000)
    assert float(np.squeeze(loss)) > 0


def test_calc_atm_loss_increases_with_gas_path():
    """Longer gas path → more loss."""
    loss_short = model.calc_atm_loss(10e9, gas_path_len_m=1000)
    loss_long  = model.calc_atm_loss(10e9, gas_path_len_m=10000)
    assert float(np.squeeze(loss_long)) > float(np.squeeze(loss_short))


def test_calc_atm_loss_rain_path():
    """Nonzero rain path with nonzero rainfall should produce positive loss."""
    from ewgeo.atm.reference import Atmosphere
    rainy = Atmosphere(alt=0, temp=288.15, press=1013.25,
                       water_vapor_dens=7.5, water_vapor_press=15.0,
                       rainfall=10.0, cloud_dens=0.0)
    loss = model.calc_atm_loss(10e9, rain_path_len_m=5000, atmosphere=rainy)
    assert float(np.squeeze(loss)) > 0


def test_calc_atm_loss_cloud_path():
    """Nonzero cloud path with nonzero cloud density should produce positive loss."""
    from ewgeo.atm.reference import Atmosphere
    cloudy = Atmosphere(alt=0, temp=288.15, press=1013.25,
                        water_vapor_dens=7.5, water_vapor_press=15.0,
                        rainfall=0.0, cloud_dens=0.1)
    loss = model.calc_atm_loss(10e9, cloud_path_len_m=2000, atmosphere=cloudy)
    assert float(np.squeeze(loss)) > 0


# ===========================================================================
# model.calc_zenith_loss
# ===========================================================================

def test_calc_zenith_loss_returns_three_values():
    result = model.calc_zenith_loss(10e9)
    assert len(result) == 3


def test_calc_zenith_loss_non_negative():
    total, loss_o, loss_w = model.calc_zenith_loss(10e9)
    assert float(np.squeeze(total))  >= 0
    assert float(np.squeeze(loss_o)) >= 0
    assert float(np.squeeze(loss_w)) >= 0


def test_calc_zenith_loss_total_equals_o_plus_w():
    total, loss_o, loss_w = model.calc_zenith_loss(10e9)
    assert np.isclose(float(np.squeeze(total)),
                      float(np.squeeze(loss_o)) + float(np.squeeze(loss_w)), rtol=1e-10)


def test_calc_zenith_loss_increases_with_zenith_angle():
    """Larger zenith angle (off-nadir path) means longer slant path → more loss."""
    total_0,  _, _ = model.calc_zenith_loss(10e9, zenith_angle_deg=np.array([0.]))
    total_45, _, _ = model.calc_zenith_loss(10e9, zenith_angle_deg=np.array([45.]))
    assert float(np.squeeze(total_45)) > float(np.squeeze(total_0))


def test_calc_zenith_loss_vectorized_freq():
    """Array frequency input should produce array outputs of matching size."""
    freqs = np.array([10e9, 22e9, 60e9])
    total, loss_o, loss_w = model.calc_zenith_loss(freqs)
    assert np.size(total)  == len(freqs)
    assert np.size(loss_o) == len(freqs)
    assert np.size(loss_w) == len(freqs)
