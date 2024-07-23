"""
Draw Figures - Appendix D

This script generates all the figures that appear in Appendix D of the textbook.

Ported from MATLAB Code

Nicholas O'Donoughue
8 December 2022
"""

import utils
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import atm
import noise


def make_all_figures(close_figs=False):
    """
    Call all the figure generators for this chapter

    :param close_figs: Boolean flag.  If true, will close all figures after generating them; for batch scripting.
                 Default=False
    :return: List of figure handles
    """

    # Find the output directory
    prefix = utils.init_output_dir('appendixD')

    # Random Number Generator
    # rng = np.random.default_rng(0)

    # Activate seaborn for prettier plots
    sns.set()

    # Colormap
    # colors = plt.get_cmap("tab10")

    # Generate all figures
    fig1 = make_figure_1(prefix)
    plt.show()
    fig2 = make_figure_2(prefix)
    plt.show()
    fig3 = make_figure_3(prefix)
    plt.show()
    fig4 = make_figure_4(prefix)
    plt.show()

    figs = [fig1, fig2, fig3, fig4]
    if close_figs:
        for fig in figs:
            plt.close(fig)

        return None
    else:
        plt.show()

        return figs


def make_figure_1(prefix=None):
    """
    Figure 1 - Noise vs. Noise Temp

    Ported from MATLAB Code

    Nicholas O'Donoughue
    8 December 2022

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    t_ext = np.arange(300.)
    # t_total = utils.constants.ref_temp + t_ext

    bw = 1e3
    noise_total = noise.model.get_thermal_noise(bandwidth_hz=bw, temp_ext_k=t_ext)
    noise_ref = noise.model.get_thermal_noise(bandwidth_hz=bw)

    # Open figure
    fig1 = plt.figure()
    plt.semilogx(t_ext, noise_total-noise_ref)
    plt.xlabel('Combined Sky Noise Temperature [K]')
    plt.ylabel('Increase in Noise Level [dB]')
    plt.grid('on')

    if prefix is not None:
        plt.savefig(prefix + 'fig1.png')
        plt.savefig(prefix + 'fig1.svg')

    return fig1


def make_figure_2(prefix=None):
    """
    Figure 2 - Cosmic Noise

    Ported from MATLAB Code

    Nicholas O'Donoughue
    8 December 2022

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    # Plot cosmic noise [dB] as a function of frequency for a fixed bandwidth
    # (a) without solar/lunar gain
    # (b) with solar gain = 0 dBi
    # (c) with solar gain = 30 dBi
    # (d) with lunar gain = 0 dBi
    # (e) with lunar gain = 30 dBi

    freq = np.arange(start=100e6, step=100e6, stop=1e10)
    freq_ghz = freq/1e9
    noise_temp = noise.model.get_cosmic_noise_temp(freq_hz=freq, rx_alt_m=0, alpha_c=.95)
    noise_temp_sun = noise.model.get_cosmic_noise_temp(freq_hz=freq, rx_alt_m=0, alpha_c=.95, gain_sun_dbi=30)
    noise_temp_moon = noise.model.get_cosmic_noise_temp(freq_hz=freq, rx_alt_m=0, alpha_c=.95, gain_moon_dbi=30)

    ref_temp = utils.constants.ref_temp

    fig2, ax = plt.subplots()
    plt.loglog(freq_ghz, noise_temp, linestyle='-', linewidth=1, label='Cosmic Noise')
    plt.loglog(freq_ghz, noise_temp_sun, linewidth=1, label=None)
    plt.loglog(freq_ghz, noise_temp_moon, linewidth=1, label=None)
    plt.loglog(freq_ghz, ref_temp*np.ones_like(freq), linestyle=':', linewidth=1, label='Thermal Noise')
    plt.loglog(freq_ghz, ref_temp + noise_temp, linestyle='-.', linewidth=1, label='Thermal + Cosmic Noise')
    plt.loglog(freq_ghz, ref_temp + noise_temp_sun, linestyle='-.', linewidth=1, label=None)
    plt.loglog(freq_ghz, ref_temp + noise_temp_moon, linestyle='-.', linewidth=1, label=None)

    plt.text(.7, 200, 'Impact of cosmic noise', fontsize=9)
    ax.annotate("", xy=(1, 1e3), xytext=(1, ref_temp), arrowprops=dict(arrowstyle="->", color="k"))
    plt.text(.45, 3, 'Sidelobe Cosmic Noise', fontsize=9)
    plt.text(1.5, 9, 'Mainbeam pointed at Moon', fontsize=9)
    plt.text(1, 75, 'Mainbeam pointed at Sun', fontsize=9)
    plt.legend(loc='upper left')
    plt.xlabel('Frequency [GHz]')
    plt.ylabel('Noise Temperature [K]')

    if prefix is not None:
        fig2.savefig(prefix + 'fig2.png')
        fig2.savefig(prefix + 'fig2.svg')

    return fig2


def make_figure_3(prefix=None):
    """
    Figure 3 - Atmospheric Noise

    Ported from MATLAB Code

    Nicholas O'Donoughue
    8 December 2022

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    zenith_angle_deg = np.array([0, 10, 30, 60])

    # Set up frequencies
    fo, fw = atm.reference.get_spectral_lines()
    freq_vec = np.sort(np.concatenate((fo, fw, fo + 50e6, fw + 50e6, fo - 100e6, fw - 100e6,
                                       np.arange(start=1e9, step=1e9, stop=350e9)),
                                      axis=0))
    freq_ghz = freq_vec/1e9
    ref_temp = utils.constants.ref_temp

    # Open the figure
    fig3, ax = plt.subplots()
    plt.semilogx(freq_ghz, ref_temp*np.ones_like(freq_vec), linestyle=':', label='Thermal Noise')
    degree_sign = u'\N{DEGREE SIGN}'
    thermal_plus_noise_label = 'Thermal + Atmospheric Noise'

    # Iterate over zenith angles
    for this_zenith in zenith_angle_deg:
        ta = noise.model.get_atmospheric_noise_temp(freq_hz=freq_vec, alt_start_m=0, el_angle_deg=90-this_zenith)

        handle = plt.semilogx(freq_ghz, ta, label='{}{} from Zenith'.format(this_zenith, degree_sign))
        plt.semilogx(freq_ghz, ref_temp + ta, linestyle='-.', color=handle[0].get_color(),
                     label=thermal_plus_noise_label)
        thermal_plus_noise_label = None  # clear the label, so we only get one entry in the legend

    plt.xlabel('Freq [GHz]')
    plt.ylabel('Noise Temperature[K]')
    plt.xlim([1, 350])

    plt.text(60, ref_temp+10, 'Impact of Atmospheric Noise', fontsize=9)
    ax.annotate("", xy=(60, 525), xytext=(60, ref_temp), arrowprops=dict(arrowstyle="->", color="k"))

    if prefix is not None:
        fig3.savefig(prefix + 'fig3.png')
        fig3.savefig(prefix + 'fig3.svg')

    return fig3


def make_figure_4(prefix=None):
    """
    Figure 4 - Ground Noise

    Ported from MATLAB Code

    Nicholas O'Donoughue
    8 December 2022

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    ground_ant_gain_dbi = np.arange(start=-30, stop=0)

    ref_temp = utils.constants.ref_temp
    ground_noise_temp = noise.model.get_ground_noise_temp(ant_gain_ground_dbi=ground_ant_gain_dbi)

    fig4, ax = plt.subplots()
    plt.plot(ground_ant_gain_dbi, ground_noise_temp, label='Ground Noise')
    plt.plot(ground_ant_gain_dbi, ref_temp * np.ones_like(ground_ant_gain_dbi), linestyle=':', label='Thermal Noise')
    plt.plot(ground_ant_gain_dbi, ground_noise_temp+ref_temp, linestyle='-.', label='Thermal + Ground Noise')
    plt.xlabel('Average Ground Antenna Gain [dBi]')
    plt.ylabel('Noise Temperature [K]')
    plt.legend(loc='upper left')

    # Annotation
    plt.text(-8, 270, 'Impact of Ground Noise', fontsize=9)
    ax.annotate("", xy=(-3, 330), xytext=(-3, ref_temp), arrowprops=dict(arrowstyle="->", color="k"))

    if prefix is not None:
        fig4.savefig(prefix + 'fig4.png')
        fig4.savefig(prefix + 'fig4.svg')

    return fig4
