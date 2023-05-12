import numpy as np
from ..utils import sinc_derivative


def make_gain_functions(type, d_lam, psi_0):
    """
    Generate function handles for the gain pattern (g) and gradient (g_dot),
    given the specified aperture type, aperture size, and mechanical steering
    angle.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    9 January 2021

    :param type: String indicating the type of aperture requested.  Supports 'omni', 'adcock', and 'rectangular'
    :param d_lam: Aperture length, in units of wavelength (d/lambda)
    :param psi_0: Mechanical steering angle (in radians) of the array [default=0]
    :return g: Function handle to the antenna pattern g(psi), for psi in radians
    :return g_dot: Function handle to the gradient of the antenna pattern, g_dot(psi), for psi in radians.
    """

    #  type = 'Adcock' or 'Rectangular'
    # params
    #   d_lam = baseline (in wavelengths)
    #   psi_0 = central angle

    # Define all the possible functions
    def g_omni(_):
        return 1.

    def g_dot_omni(_):
        return 0.

    def g_adcock(psi):
        return 2*np.sin(np.pi * d_lam * np.cos(psi-psi_0))

    def g_dot_adcock(psi):
        return -2*np.pi*d_lam*np.sin(psi-psi_0)*np.cos(np.pi*d_lam*np.cos(psi-psi_0))

    def g_rect(psi):
        return np.abs(np.sinc((psi-psi_0)*d_lam/np.pi))  # sinc includes implicit pi

    def g_dot_rect(psi):
        return sinc_derivative((psi - psi_0) * d_lam) * d_lam

    switcher = {'omni': (g_omni, g_dot_omni),
                'adcock': (g_adcock, g_dot_adcock),
                'rectangular': (g_rect, g_dot_rect)}

    result = switcher.get(type.lower())

    return result[0], result[1]
