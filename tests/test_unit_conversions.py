import math

import numpy as np
import pytest

from ewgeo.utils.unit_conversions import (
    lin_to_db,
    db_to_lin,
    convert,
    parse_units,
    kft_to_km,
    km_to_kft,
    kph_to_mps,
    mps_to_kph,
)


def equal_to_tolerance(x, y, tol=1e-9):
    if np.size(x) != np.size(y):
        return False
    return np.all(np.fabs(np.array(x) - np.array(y)) < tol)


# ===========================================================================
# lin_to_db
# ===========================================================================

def test_lin_to_db_zero_returns_neg_inf():
    assert lin_to_db(0.0) == -np.inf


def test_lin_to_db_one_returns_zero():
    assert equal_to_tolerance(lin_to_db(1.0), 0.0)


def test_lin_to_db_ten_returns_ten():
    assert equal_to_tolerance(lin_to_db(10.0), 10.0)


def test_lin_to_db_hundred_returns_twenty():
    assert equal_to_tolerance(lin_to_db(100.0), 20.0)


def test_lin_to_db_fraction():
    assert equal_to_tolerance(lin_to_db(0.1), -10.0)


def test_lin_to_db_vectorized():
    result = lin_to_db(np.array([1.0, 10.0, 100.0]))
    assert equal_to_tolerance(result, np.array([0.0, 10.0, 20.0]))


# ===========================================================================
# db_to_lin
# ===========================================================================

def test_db_to_lin_zero_returns_one():
    assert equal_to_tolerance(db_to_lin(0.0), 1.0)


def test_db_to_lin_ten_returns_ten():
    assert equal_to_tolerance(db_to_lin(10.0), 10.0)


def test_db_to_lin_twenty_returns_hundred():
    assert equal_to_tolerance(db_to_lin(20.0), 100.0)


def test_db_to_lin_negative_ten():
    assert equal_to_tolerance(db_to_lin(-10.0), 0.1)


def test_db_to_lin_overflow_returns_inf():
    assert db_to_lin(3001.0) == np.inf


def test_db_to_lin_vectorized():
    result = db_to_lin(np.array([0.0, 10.0, 20.0]))
    assert equal_to_tolerance(result, np.array([1.0, 10.0, 100.0]))


def test_db_lin_round_trip():
    values = np.array([0.01, 0.1, 1.0, 10.0, 1000.0])
    assert equal_to_tolerance(db_to_lin(lin_to_db(values)), values, tol=1e-9)


# ===========================================================================
# parse_units / convert — length
# ===========================================================================

def test_convert_km_to_m():
    assert equal_to_tolerance(convert(1.0, 'km', 'm'), 1000.0)


def test_convert_m_to_km():
    assert equal_to_tolerance(convert(1000.0, 'm', 'km'), 1.0)


def test_convert_km_to_kft():
    # 1 km = 1000 m; 1 kft = 304.8 m → 1000/304.8 kft
    expected = 1000.0 / 304.8
    assert equal_to_tolerance(convert(1.0, 'km', 'kft'), expected, tol=1e-6)


def test_convert_ft_to_m():
    assert equal_to_tolerance(convert(1.0, 'ft', 'm'), 0.3048)


# ===========================================================================
# parse_units / convert — angle
# ===========================================================================

def test_convert_rad_to_deg():
    assert equal_to_tolerance(convert(1.0, 'rad', 'deg'), 180.0 / math.pi, tol=1e-12)


def test_convert_deg_to_rad():
    assert equal_to_tolerance(convert(180.0, 'deg', 'rad'), math.pi, tol=1e-12)


def test_convert_deg_to_deg_identity():
    assert equal_to_tolerance(convert(45.0, 'deg', 'deg'), 45.0)


# ===========================================================================
# parse_units / convert — speed
# ===========================================================================

def test_convert_mps_to_kph():
    assert equal_to_tolerance(convert(1.0, 'm/s', 'kph'), 3.6)


def test_convert_kph_to_mps():
    assert equal_to_tolerance(convert(3.6, 'kph', 'm/s'), 1.0)


# ===========================================================================
# parse_units error cases
# ===========================================================================

def test_parse_units_unknown_from_unit_raises():
    with pytest.raises(ValueError):
        parse_units('lightyear', 'm')


def test_parse_units_cross_category_length_to_angle_raises():
    with pytest.raises(ValueError):
        parse_units('m', 'deg')


def test_parse_units_cross_category_speed_to_length_raises():
    with pytest.raises(ValueError):
        parse_units('m/s', 'm')


# ===========================================================================
# Convenience wrappers
# ===========================================================================

def test_kft_to_km_one_kft():
    # 1 kft = 304.8 m = 0.3048 km
    assert equal_to_tolerance(kft_to_km(1.0), 0.3048)


def test_km_to_kft_round_trip():
    assert equal_to_tolerance(km_to_kft(kft_to_km(1.0)), 1.0, tol=1e-9)


def test_kph_to_mps_36kph():
    assert equal_to_tolerance(kph_to_mps(3.6), 1.0)


def test_mps_to_kph_1mps():
    assert equal_to_tolerance(mps_to_kph(1.0), 3.6)


def test_mps_kph_round_trip():
    assert equal_to_tolerance(mps_to_kph(kph_to_mps(100.0)), 100.0, tol=1e-9)
