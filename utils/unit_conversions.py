import numpy as np


_ft2m = 0.3048
_m2ft = 3.2808


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
    lin_value = np.where(lin_value >= eps, lin_value, eps)

    return 10. * np.log10(lin_value)


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
    db_value = np.where(db_value <= inf_val, db_value, np.inf)

    return np.power(10., db_value/10.)


def kft_to_km(kft_value):
    """
    Convert altitude from kft (thousands of feet) to km.

    :param kft_value:
    :return:
    """
    return kft_value * _ft2m


def km_to_kft(km_value):
    """
    Convert altitude from km to kft (thousands of feet)

    :param km_value:
    :return:
    """

    return km_value * _m2ft
