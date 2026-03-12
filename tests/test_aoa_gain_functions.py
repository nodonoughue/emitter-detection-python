import numpy as np

from ewgeo.aoa import make_gain_functions
from ewgeo import utils

def test_gain_vectorized():
    """
    Generate gain functions for 'omni', 'adcock', and 'rectangular' apertures.
    Test that each functions with vectorized inputs.
    """

    d_lam = .5
    psi_0 = 0
    g_omni, g_omni_dot = make_gain_functions('omni', d_lam, psi_0)
    g_adcock, g_adcock_dot = make_gain_functions('adcock', d_lam, psi_0)
    g_rect, g_rect_dot = make_gain_functions('rectangular', d_lam, psi_0)

    # This should run
    psi = np.linspace(0, 2*np.pi, 100)
    g_omni_test = g_omni(psi)
    g_omni_dot_test = g_omni_dot(psi)
    g_adcock_test = g_adcock(psi)
    g_adcock_dot_test = g_adcock_dot(psi)
    g_rect_test = g_rect(psi)
    g_rect_dot_test = g_rect_dot(psi)

    # Test values
    assert equal_to_tolerance(g_omni_test, np.ones_like(g_omni_test))
    assert equal_to_tolerance(g_omni_dot_test, np.zeros_like(g_omni_dot_test))
    g_adcock_ref = 2 * np.sin(np.pi*np.cos(psi)*d_lam)
    assert equal_to_tolerance(g_adcock_test, g_adcock_ref)
    g_adcock_dot_ref = -2*np.pi*d_lam*np.sin(psi)*np.cos(np.pi*d_lam*np.cos(psi))
    assert equal_to_tolerance(g_adcock_dot_test, g_adcock_dot_ref)
    g_rect_ref = np.abs(np.sinc(psi*d_lam/np.pi))
    assert equal_to_tolerance(g_rect_test, g_rect_ref)
    g_rect_dot_ref = utils.sinc_derivative(psi * d_lam) * d_lam
    assert equal_to_tolerance(g_rect_dot_test, g_rect_dot_ref)

def equal_to_tolerance(x, y, tol=1e-6)->bool:
    """
    Shorthand to compare two lists and ensure each entry has an error less than the specified tolerance
    """
    if len(x) != len(y): return False
    return all([abs(xx-yy)<tol for xx, yy in zip(x,y)])