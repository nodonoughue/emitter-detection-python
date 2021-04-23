import numpy as np


_ft2m = 0.3048
_m2ft = 3.2808


def lin_to_db(lin_value):
    return 10. * np.log10(lin_value)


def db_to_lin(db_value):
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
