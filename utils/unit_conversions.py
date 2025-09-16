import numpy as np

_length_factors = {
    "m": 1,
    "km": 1000,
    "ft": 0.3048,
    "kft": 304.8,
    "yd": 0.9144,
    "mi": 1609.34,
    "nmi": 1852
}

_angle_factors = {
    "deg": 1,
    "rad": 180/np.pi
}

_speed_factors = {
    "m/s": 1,
    "kph": 1./3.6,
    "knot": 0.5144,
}


def lin_to_db(lin_value, eps=1e-99):
    """
    Convert inputs from linear units to dB, via the simple equation
         db = 10 * log10 (lin)

    Nicholas O'Donoughue
    7 May 2021

    :param lin_value: scalar or numpy array of linear values to convert
    :param eps: minimum precision, any inputs < eps will be capped, to prevent a divide by zero runtime error
    :return: scalar or numpy array of db values
    """

    # Use the optional eps argument as a minimum allowable precision, to prevent divide by zero errors if we take the
    # logarithm of 0.
    return 10 * np.log10(lin_value, out=-np.inf*np.ones_like(lin_value), where=lin_value>= eps)


def db_to_lin(db_value, inf_val=3000):
    """
    Convert input from db units to linear, via the simple equation
        lin = 10^(db/10)

    Nicholas O'Donoughue
    7 May 2021

    :param db_value: scalar or numpy array of db values to convert
    :param inf_val: any dB values above this point will be converted to np.inf to avoid overflow
    :return: scalar or numpy array of linear values
    """

    # Use the optional inf_val argument as a maximum dB value, above which we convert the output to np.inf to prevent
    # overflow errors
    return np.power(10., db_value/10., out=np.inf*np.ones_like(db_value), where=db_value <= inf_val)


def convert(value, from_unit, to_unit):
    """
    Convert a speed, length, or angle from one unit to another.

    :param value: float or numpy array
    :param from_unit: str denoting the units of value
    :param to_unit: str denoting the desired units of the return value
    :return: output after conversion to desired units
    """
    factor = parse_units(from_unit, to_unit)

    return value * factor


def parse_units(from_unit, to_unit):
    """
    Attempt to determine what type of conversion this is, and use the appropriate factor
    dictionary.
    """

    # Make sure the units are lower-case, our dictionaries are case-insensitive
    from_unit = from_unit.lower()
    to_unit = to_unit.lower()

    # Determine which type of unit conversion is being asked for, and lookup the factor
    if from_unit in _length_factors:
        #  Length factor
        if to_unit not in _length_factors:
            # Not a valid length conversion
            raise ValueError("Invalid unit; length conversion detected. Please use on of: "
                             + ", ".join(_length_factors.keys()))

        factor = _length_factors[from_unit] / _length_factors[to_unit]

    elif from_unit in _angle_factors:
        if to_unit not in _angle_factors:
            # Not a valid angle conversion
            raise ValueError("Invalid unit; angle conversion detected. Please use on of: "
                             + ", ".join(_angle_factors.keys()))

        factor = _angle_factors[from_unit] / _angle_factors[to_unit]

    elif from_unit in _speed_factors:
        if to_unit not in _speed_factors:
            # Not a valid speed conversion
            raise ValueError("Invalid unit; speed conversion detected. Please use on of: "
                             + ", ".join(_speed_factors.keys()))

        factor = _speed_factors[from_unit] / _speed_factors[to_unit]

    else:
        raise ValueError("Invalid unit; unable to determine desired conversion type.")

    return factor


def kft_to_km(kft_value):
    """
    Convert altitude from kft (thousands of feet) to km.

    :param kft_value:
    :return:
    """
    # return kft_value * _ft2m
    return convert(kft_value, "kft", "km")


def km_to_kft(km_value):
    """
    Convert altitude from km to kft (thousands of feet)

    :param km_value:
    :return:
    """
    # return km_value * _m2ft
    return convert(km_value, "km", "kft")


def kph_to_mps(kph_value):
    """
    Convert speed from kph to m/s

    :param kph_value:
    :return:
    """
    # return kph_value * 1e3 / 3600
    return convert(kph_value, "kph", "m/s")


def mps_to_kph(mps_value):
    """
    Convert speed from m/s to kph

    :param mps_value:
    :return:
    """
    # return mps_value * 3.6
    return convert(mps_value, "m/s", "kph")
