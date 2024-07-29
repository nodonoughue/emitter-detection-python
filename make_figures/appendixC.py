"""
Draw Figures - Appendix C

This script generates all the figures that appear in Appendix C of the textbook.

Ported from MATLAB Code

Nicholas O'Donoughue
8 December 2022
"""

import utils
import matplotlib.pyplot as plt
import numpy as np
import atm


def make_all_figures(close_figs=False):
    """
    Call all the figure generators for this chapter

    :param close_figs: Boolean flag.  If true, will close all figures after generating them; for batch scripting.
                 Default=False
    :return: List of figure handles
    """

    # Find the output directory
    prefix = utils.init_output_dir('appendixC')
    utils.init_plot_style()

    # Random Number Generator
    # rng = np.random.default_rng(0)

    # Colormap
    # colors = plt.get_cmap("tab10")

    # Generate all figures
    fig2 = make_figure_2(prefix)
    fig3 = make_figure_3(prefix)
    fig4 = make_figure_4(prefix)
    fig5 = make_figure_5(prefix)

    figs = [fig2, fig3, fig4, fig5]
    if close_figs:
        for fig in figs:
            plt.close(fig)

        return None
    else:
        plt.show()

        return figs


def make_figure_2(prefix=None):
    """
    Figure 2 - Dry Air and Water Vapor

    Ported from MATLAB Code

    Nicholas O'Donoughue
    8 December 2022

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    print('Generating Figure C.2...')

    # Open the Figure and Initialize Labels
    fig2 = plt.figure()
    ao_label = 'Dry Air Only'
    aw_label = 'Water Vapor'
    a_tot_label = 'Total'

    # Set up frequencies
    fo, fw = atm.reference.get_spectral_lines()
    freq_vec = np.sort(np.concatenate((fo, fw, fo+50e6, fw+50e6, fo-100e6, fw-100e6,
                                       np.arange(start=1e9, step=1e9, stop=350e9)),
                                      axis=0))

    # Iterate over altitude bands
    for alt_m in np.array([0, 10, 20])*1e3:
        atmosphere = atm.reference.get_standard_atmosphere(alt_m)

        # Compute Loss Coefficients
        ao, aw = atm.model.get_gas_loss_coeff(freq_hz=freq_vec, press=atmosphere.press,
                                              water_vapor_press=atmosphere.water_vapor_press, temp=atmosphere.temp)

        # Plot
        handle = plt.loglog(freq_vec/1e9, np.squeeze(ao), linestyle=':', label=ao_label)
        plt.loglog(freq_vec/1e9, np.squeeze(aw), linestyle='--', color=handle[0].get_color(), label=aw_label)
        plt.loglog(freq_vec/1e9, np.squeeze(ao + aw), linestyle='-', color=handle[0].get_color(), label=a_tot_label)

        # Clear the labels -- keeps the legend clean (don't print labels for subsequent altitude bands)
        ao_label = None
        aw_label = None
        a_tot_label = None

    # Adjust Plot Display
    plt.xlim([1, 350])
    plt.ylim([1e-5, 1e2])
    plt.xlabel('Frequency [GHz]')
    plt.ylabel(r'Gas Loss Coefficient $\gamma_g$ [dB/km]')
    plt.legend(loc='upper left')

    # Text Annotation
    plt.text(2, 1.5e-2, '0 km')
    plt.text(2, 1.5e-3, '10 km')
    plt.text(2, 8e-5, '20 km')

    if prefix is not None:
        fig2.savefig(prefix + 'fig2.png')
        fig2.savefig(prefix + 'fig2.svg')

    return fig2


def make_figure_3(prefix=None, colors=None):
    """
    Figure 3 - Rain Loss Coefficient

    Ported from MATLAB Code

    Nicholas O'Donoughue
    8 December 2022

    :param prefix: output directory to place generated figure
    :param colors: list of colors to use for plotting (if not specified, will use the matplotlib default color order)
    :return: figure handle
    """

    print('Generating Figure C.3...')

    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Initialize Parameters
    rain_rate_set = [1., 4., 16., 100.]
    colors = colors[:len(rain_rate_set)]  # Keep as many colors as there are rainfall rate settings

    pol_ang_vec = [0, np.pi / 2]
    pol_set = ('Horizontal', 'Vertical')
    line_style_set = ('-', '-.')

    freq_hz = np.arange(start=1, stop=100, step=.5) * 1e9
    el_ang_rad = 0*np.pi/180

    # Open Figure
    fig3 = plt.figure()

    # Iterate over rainfall rate conditions and polarity conditions
    for rain_rate, this_color in zip(rain_rate_set, colors):
        for pol_ang_rad, pol_label, this_line_style in zip(pol_ang_vec, pol_set, line_style_set):
            # Compute rain loss coefficients
            gamma = atm.model.get_rain_loss_coeff(freq_hz=freq_hz, pol_angle_rad=pol_ang_rad,
                                                  el_angle_rad=el_ang_rad, rainfall_rate=rain_rate)

            plt.loglog(freq_hz/1e9, gamma, linestyle=this_line_style, color=this_color, label=pol_label)

        # Clear the polarization labels; so we only get one set of legend entries
        pol_set = (None, None)

    plt.grid(True)
    plt.ylim([.01, 50])
    plt.xlim([freq_hz[0]/1e9, freq_hz[-1]/1e9])

    # Add rainfall condition labels
    plt.text(10, 6, 'Very Heavy', rotation=25)  # set(ht,'rotation',25)
    plt.text(10, .6, 'Heavy', rotation=40)      # set(ht,'rotation',40)
    plt.text(10, .12, 'Moderate', rotation=40)  # set(ht,'rotation',40)
    plt.text(10, .023, 'Light', rotation=40)    # set(ht,'rotation',40)

    plt.xlabel('Frequency [GHz]')
    plt.ylabel(r'Rain Loss Coefficient $\gamma_r$ [dB/km]')
    plt.legend(loc='upper left')

    if prefix is not None:
        fig3.savefig(prefix + 'fig3.png')
        fig3.savefig(prefix + 'fig3.svg')

    return fig3


def make_figure_4(prefix=None):
    """
    Figure 4 - Cloud/Fog Loss

    Ported from MATLAB Code

    Nicholas O'Donoughue
    8 December 2022

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    print('Generating Figure C.4...')

    # Initialize Parameters
    fog_set = [.032, .32, 2.3]
    fog_names = ['600 m Visibility', '120 m Visibility', '30 m Visibility']

    freq_hz = np.arange(start=1, stop=100, step=.5)*1e9

    # Open the figure
    fig4 = plt.figure()

    # Iterate over fog conditions
    for this_fog, this_fog_label in zip(fog_set, fog_names):
        gamma = atm.model.get_fog_loss_coeff(f=freq_hz, cloud_dens=this_fog, temp_k=None)

        plt.loglog(freq_hz/1e9, gamma, label=this_fog_label)

    plt.grid(True)

    # ht=text(10,.2,'30 m Visibility');set(ht,'rotation',30);
    # ht=text(10,.025,'120 m Visibility');set(ht,'rotation',30);
    # ht=text(32,.025,'600 m Visibility');set(ht,'rotation',30);

    plt.ylim([.01, 10])
    plt.xlabel('Frequency [GHz]')
    plt.ylabel(r'Cloud Loss Coefficient $\gamma_c$ [dB/km]')
    plt.legend(loc='upper left')

    if prefix is not None:
        fig4.savefig(prefix + 'fig4.png')
        fig4.savefig(prefix + 'fig4.svg')

    return fig4


def make_figure_5(prefix=None):
    """
    Figure 5 - Zenith Loss

    Ported from MATLAB Code

    Nicholas O'Donoughue
    8 December 2022

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    print('Generating Figure C.5...')

    # Set up frequencies
    fo, fw = atm.reference.get_spectral_lines()
    freq_vec = np.sort(np.concatenate((fo, fw, fo + 50e6, fw + 50e6, fo - 100e6, fw - 100e6,
                                       np.arange(start=1e9, step=1e9, stop=350e9)),
                                      axis=0))

    # Set of nadir angles to calculate
    nadir_deg_set = [0, 10, 30, 60]
    degree_sign = u'\N{DEGREE SIGN}'
    nadir_labels = ['{}{} from Zenith'.format(this_nadir, degree_sign) for this_nadir in nadir_deg_set]
    nadir_labels[0] = 'Zenith'

    # Open figure
    fig5 = plt.figure()

    # Iterate over nadir angles
    for this_nadir, this_label in zip(nadir_deg_set, nadir_labels):
        # Order of outputs is (total_loss, loss_from_oxygen, loss_from_water_vapor)
        loss, _, _ = atm.model.calc_zenith_loss(freq_hz=freq_vec, alt_start_m=0, zenith_angle_deg=this_nadir)

        plt.loglog(freq_vec/1e9, loss, label=this_label)

    plt.xlabel('Frequency [GHz]')
    plt.ylabel('Zenith Attenuation')
    plt.legend(loc='upper left')

    plt.grid(True)
    plt.xlim([1, 350])
    plt.ylim([1e-2, 1e3])

    if prefix is not None:
        fig5.savefig(prefix + 'fig5.png')
        fig5.savefig(prefix + 'fig5.svg')

    return fig5
