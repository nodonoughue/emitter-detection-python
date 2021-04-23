import numpy as np
import matplotlib.pyplot as plt
from . import reference, model


def plot_itu_ref_figs():
    """
    Generates figures to compare with ITU-R P.676-12

    Can be used to ensure atmospheric loss tables and calculations are reasonably accurate.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    21 March 2021

    :return figs: array of figure handle objects
    """

    # Figure 1 - Specific attenuation
    f_ghz = np.arange(1001)
    f_ctr_o, f_ctr_w = reference.get_spectral_lines(f_ghz[0]*1e9, f_ghz[-1]*1e9)
    f = np.sort(np.concatenate((f_ghz * 1e9, f_ctr_o, f_ctr_w)))
    f_ghz = f/1e9  # recompute f_ghz to include spectral lines
    atmosphere = reference.get_standard_atmosphere(0)
    
    gamma_ox, gamma_h2o = model.get_gas_loss_coeff(f, atmosphere.press, atmosphere.water_vapor_press, atmosphere.temp)
    
    fig1 = plt.figure()
    plt.semilogy(f_ghz, gamma_ox, linestyle='b-', label='Dry')
    plt.semilogy(f_ghz, gamma_ox+gamma_h2o, linestyle='r-', label='Standard')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Specific Attenuation (dB/km)')
    plt.grid()
    plt.legend(loc='NorthWest')
    plt.title('Replication of ITU-R P.676-12, Figure 1')
    plt.xlim((0, 1e3))
    
    # Figure 2
    f_ghz = np.arange(start=50, stop=70.1, step=0.1)
    f_ctr_o, f_ctr_w = reference.get_spectral_lines(f_ghz[0] * 1e9, f_ghz[-1] * 1e9)
    f = np.sort(np.concatenate((f_ghz * 1e9, f_ctr_o, f_ctr_w)))
    f_ghz = f / 1e9  # recompute f_ghz to include spectral lines
    
    alts = np.arange(start=0, stop=25, step=5)*1e3
    atmosphere = reference.get_standard_atmosphere(alts)
    
    gamma_ox, gamma_h2o = model.get_gas_loss_coeff(f, atmosphere.press, atmosphere.water_vapor_press, atmosphere.temp)
    gamma = gamma_ox + gamma_h2o
    
    fig2 = plt.figure()
    for idx, alt in enumerate(alts):
        plt.semilogy(f_ghz, gamma[:, idx], linestyle='-', label='{} km'.format(alt/1e3))

    plt.legend(loc='NorthWest')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Specific Attenuation (dB/km)')
    plt.grid()
    plt.title('Replication of ITU-R P.676-12, Figure 2')
    
    # Figure 4
    f_ghz = np.arange(start=1, stop=1001, step=1)
    f_ctr_o, f_ctr_w = reference.get_spectral_lines(f_ghz[0] * 1e9, f_ghz[-1] * 1e9)
    f = np.sort(np.concatenate((f_ghz * 1e9, f_ctr_o, f_ctr_w)))
    f_ghz = f / 1e9  # recompute f_ghz to include spectral lines
    
    l, lo, _ = model.calc_zenith_loss(f, alt_start_m=0.0, zenith_angle_deg=0.0)
    
    fig4 = plt.figure()
    plt.semilogy(f_ghz, lo, linestyle='b-', label='Dry')
    plt.semilogy(f_ghz, l, linestyle='r-', label='Standard')
    plt.xlim((0, 1e3))
    plt.xlabel('Frequency (Ghz)')
    plt.ylabel('Zenith Attenuation (dB)')
    plt.grid()
    plt.legend(loc='NorthWest')
    plt.title('Replication of ITU-R P.676-12, Figure 4')
    
    # Figure 10
    f_ghz = np.arange(start=1, stop=351, step=1)
    f_ctr_o, f_ctr_w = reference.get_spectral_lines(f_ghz[0] * 1e9, f_ghz[-1] * 1e9)
    f = np.sort(np.concatenate((f_ghz * 1e9, f_ctr_o, f_ctr_w)))
    f_ghz = f / 1e9  # recompute f_ghz to include spectral lines
    
    atmosphere = reference.get_standard_atmosphere(0)
    
    gamma_ox, gamma_h2o = model.get_gas_loss_coeff(f, atmosphere.press, atmosphere.water_vapor_press, atmosphere.temp)
    
    fig10 = plt.figure()
    plt.loglog(f_ghz, gamma_ox, linestyle='b-', label='Dry')
    plt.loglog(f_ghz, gamma_h2o, linestyle='r-', label='Water Vapour')
    plt.loglog(f_ghz, gamma_ox+gamma_h2o, linestyle='k-', label='Total')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Specific Attenuation (dB/km)')
    plt.grid()
    plt.legend(loc='NorthWest')
    plt.title('Replication of ITU-R P.676-12, Figure 10')
    plt.xlim((0, 350))
    
    # Figure 11
    f_ghz = np.arange(start=1, stop=351, step=1)
    f_ctr_o, f_ctr_w = reference.get_spectral_lines(f_ghz[0] * 1e9, f_ghz[-1] * 1e9)
    f = np.sort(np.concatenate((f_ghz * 1e9, f_ctr_o, f_ctr_w)))
    f_ghz = f / 1e9  # recompute f_ghz to include spectral lines

    l, lo, lw = model.calc_zenith_loss(f, alt_start_m=0.0, zenith_angle_deg=0.0)
    
    fig11 = plt.figure()
    plt.loglog(f_ghz, lo, linestyle='b-', label='Dry')
    plt.loglog(f_ghz, lw, linestyle='r-', label='Water vapour')
    plt.semilogy(f_ghz, l, linestyle='k-', label='Total')
    plt.xlim((0, 350))
    plt.xlabel('Frequency (Ghz)')
    plt.ylabel('Zenith Attenuation (dB)')
    plt.grid()
    plt.legend(loc='NorthWest')
    plt.title('Replication of ITU-R P.676-12, Figure 11')
    
    # Figure 12
    f_ghz = np.arange(start=50, stop=70.01, step=.01)
    f_ctr_o, f_ctr_w = reference.get_spectral_lines(f_ghz[0] * 1e9, f_ghz[-1] * 1e9)
    f = np.sort(np.concatenate((f_ghz * 1e9, f_ctr_o, f_ctr_w)))
    f_ghz = f / 1e9  # recompute f_ghz to include spectral lines

    alts = np.arange(start=0, stop=25, step=5)*1e3
    
    l, _, _ = model.calc_zenith_loss(f, alts, zenith_angle_deg=0.0)
    
    fig12 = plt.figure()
    for idx, alt in enumerate(alts):
        plt.loglog(f_ghz, l[:, idx], linestyle='-', label='{} km'.format(alt/1e3))

    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Zenith Attenuation (dB)')
    plt.grid()
    plt.legend(loc='NorthWest')
    plt.title('Replication of ITU-R P.676-12, Figure 12')

    return fig1, fig2, fig4, fig10, fig11, fig12
