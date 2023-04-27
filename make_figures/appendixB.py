"""
Draw Figures - Appendix B

This script generates all the figures that appear in Appendix B of the textbook.

Ported from MATLAB Code

Nicholas O'Donoughue
8 December 2022
"""

import utils
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import prop


def make_all_figures(close_figs=False):
    """
    Call all the figure generators for this chapter

    :param close_figs: Boolean flag.  If true, will close all figures after generating them; for batch scripting.
                 Default=False
    :return: List of figure handles
    """

    # Find the output directory
    prefix = utils.init_output_dir('appendixB')

    # Random Number Generator
    # rng = np.random.default_rng(0)

    # Activate seaborn for prettier plots
    sns.set()

    # Colormap
    # colors = plt.get_cmap("tab10")

    # Generate all figures
    fig4 = make_figure_4(prefix)
    plt.show()

    figs = [fig4]
    if close_figs:
        for fig in figs:
            plt.close(fig)

        return None
    else:
        plt.show()

        return figs


def make_figure_4(prefix=None):
    """
    Figure 4 - Fresnel Zone Illustration

    Ported from MATLAB Code

    Nicholas O'Donoughue
    8 December 2022

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    # Define range axis
    range_vec = np.concatenate((np.arange(start=1e3, stop=100e3, step=1e3),
                                np.arange(start=200e3, stop=1000e3, step=100e3),
                                np.arange(start=2000e3, stop=10000e3, step=1000e3)), axis=0)

    # Open the Plot
    fig4 = plt.figure()

    # Three Situations
    # 1 - L Band, ht=hr=10 m
    # 2 - L Band, ht=hr=100 m
    # 3 - X Band, ht=hr=100 m
    freq_vec = np.array([1e9, 1e10])
    ht_vec = np.array([10., 100.])

    fresnel_zone_range_vec = []
    fresnel_zone_loss_vec = []

    for freq_hz, ht_m in zip(freq_vec, ht_vec):

        # Compute Path Loss two ways
        fspl = prop.model.get_free_space_path_loss(range_m=range_vec, freq_hz=freq_hz, height_tx_m=ht_m,
                                                   include_atm_loss=False)
        two_ray = prop.model.get_two_ray_path_loss(range_m=range_vec, freq_hz=freq_hz, height_tx_m=ht_m,
                                                   include_atm_loss=False)

        handle = plt.plot(range_vec/1e3, fspl, label='Free-Space Path Loss, f={} GHz'.format(freq_hz/1e9))
        plt.plot(range_vec/1e3, two_ray, linestyle='-.', color=handle[0].get_color(),
                 label='Two-Ray Path Loss, f={} GHz, h={} m'.format(freq_hz/1e9, ht_m))

        # Overlay the Fresnel Zone Range
        r_fz = prop.model.get_fresnel_zone(f0=freq_hz, ht=ht_m, hr=ht_m)

        y_fz = prop.model.get_free_space_path_loss(range_m=r_fz, freq_hz=freq_hz, height_tx_m=ht_m,
                                                   include_atm_loss=False)

        fresnel_zone_range_vec.append(r_fz/1e3)
        fresnel_zone_loss_vec.append(y_fz)

    plt.scatter(fresnel_zone_range_vec, fresnel_zone_loss_vec,
                marker='^', color='k', zorder=3, label='Fresnel Zone')

    plt.xscale('log')
    plt.legend(loc='upper left')

    plt.xlabel('Path Length [km]')
    plt.ylabel('Loss')

    if prefix is not None:
        fig4.savefig(prefix + 'fig4.png')
        fig4.savefig(prefix + 'fig4.svg')

    return fig4
