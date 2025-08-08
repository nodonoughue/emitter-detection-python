"""
Draw Figures - Chapter 3

This script generates all the figures that appear in Chapter 3 of the textbook.

Ported from MATLAB Code

Nicholas O'Donoughue
17 January 2025
"""

import utils
import matplotlib.pyplot as plt
import numpy as np

import tdoa
from examples.practical_geo import chapter3


def make_all_figures(close_figs=False, force_recalc=False):
    """
    Call all the figure generators for this chapter

    :param close_figs: Boolean flag.  If true, will close all figures after generating them; for batch scripting.
                       Default=False
    :param force_recalc: optional flag (default=True), if False then the example does not run
    :return: List of figure handles
    """

    # Reset the random number generator, to ensure reproducibility
    # rng = np.random.default_rng()

    # Find the output directory
    prefix = utils.init_output_dir('practical_geo/chapter3')
    utils.init_plot_style()

    # Generate all figures
    fig1 = make_figure_1(prefix)
    fig2 = make_figure_2(prefix)
    figs_4_5 = make_figures_4_5(prefix)
    figs_6_7 = make_figures_6_7(prefix)
    figs_9_10 = make_figures_9_10(prefix)
    figs_11_12 = make_figures_11_12(prefix, force_recalc)

    figs = [fig1, fig2] + list(figs_4_5) + list(figs_6_7) + list(figs_9_10) + list(figs_11_12)
    if close_figs:
        for fig in figs:
            plt.close(fig)

        return None
    else:
        plt.show()

    return figs


def make_figure_1(prefix=None):
    """
    Figure 1

    :param prefix: output directory to place generated figure
    :return: handle
    """

    print('Generating Figure 3.1...')

    # Define source and sensor positions
    x_source = np.array([2, 3])
    x_sensor = np.array([[0, -1, 0, 1], [0, -.5, 1, -.5]])

    # Plot sensor / source positions
    fig = plt.figure()
    plt.scatter(x_source[0], x_source[1], marker='^', label='Source', clip_on=False, zorder=3)
    plt.scatter(x_sensor[0, :], x_sensor[1, :], marker='o', label='Sensor', clip_on=False, zorder=3)

    # Generate Isochrones
    isochrone_label = 'Isochrones'
    ref_idx = 0
    x_ref = x_sensor[:, ref_idx]

    for test_idx in np.arange(start=ref_idx+1, stop=4):
        x_test = x_sensor[:, test_idx]

        # TODO: Make sure test/ref indices are used consistently. Should be test-ref for TDOA and FDOA

        rdiff = utils.geo.calc_range_diff(x_source, x_ref, x_test)
        xy_iso = tdoa.model.draw_isochrone(x_test, x_ref, rdiff, 10000, 5)

        plt.plot(xy_iso[0], xy_iso[1], '--', label=isochrone_label)
        isochrone_label = None  # set label to none after first use, so only one shows up in the plot legend

    plt.xlim([-1, 3])
    plt.ylim([-1, 3])
    plt.legend(loc='upper left')

    if prefix is not None:
        fig.savefig(prefix + 'fig1.svg')
        fig.savefig(prefix + 'fig1.png')

    return fig


def make_figure_2(prefix=None):
    """
    Figure 2

    :param prefix: output directory to place generated figure
    :return: handle
    """

    print('Generating Figure 3.2...')

    # Define source and sensor positions
    x_source = np.array([2, 3])
    x_sensor = np.array([[0, -1, 0, 1], [0, -.5, 1, -.5]])

    # Plot sensor / source positions
    fig = plt.figure()
    plt.scatter(x_source[0], x_source[1], marker='^', label='Source', clip_on=False, zorder=3)
    plt.scatter(x_sensor[0, :], x_sensor[1, :], marker='o', label='Sensor', clip_on=False, zorder=3)

    # Generate Isochrones
    isochrone_label = 'Isochrones'
    for ref_idx in np.arange(3):
        x_ref = x_sensor[:, ref_idx]

        for test_idx in np.arange(start=ref_idx+1, stop=4):
            x_test = x_sensor[:, test_idx]

            rdiff = utils.geo.calc_range_diff(x_source, x_ref, x_test)
            xy_iso = tdoa.model.draw_isochrone(x_test, x_ref, rdiff, 10000, 5)

            plt.plot(xy_iso[0], xy_iso[1], '--', label=isochrone_label)
            isochrone_label = None  # set label to none after first use, so only one shows up in the plot legend

    plt.xlim([-1, 3])
    plt.ylim([-1, 3])
    plt.legend(loc='upper left')

    if prefix is not None:
        fig.savefig(prefix + 'fig2.svg')
        fig.savefig(prefix + 'fig2.png')

    return fig


def make_figures_4_5(prefix=None):
    """
    Figure 3.4 and 3.5

    :param prefix: output directory to place generated figure
    :return: handle
    """

    print('Generating Figures 3.4a, 3.4b, 3.5...')

    figs = chapter3.example1()

    # Display the plot
    plt.draw()

    # Output to file
    if prefix is not None:
        labels = ['fig4a', 'fig4b', 'fig5']
        if len(labels) != len(figs):
            print('**Error saving figures 3.4 and 3.5; unexpected number of figures returned from Example 3.1.')
        else:
            for fig, label in zip(figs, labels):
                fig.savefig(prefix + label + '.svg')
                fig.savefig(prefix + label + '.png')

    return figs


def make_figures_6_7(prefix=None):
    """
    Figure 3.6 & 3.7

    :param prefix: output directory to place generated figure
    :return: handle
    """

    print('Generating Figures 3.6, 3.7a, 3.7b, 3.7c, 3.7d, 3.7e (Example 3.2)...')

    figs = chapter3.example2()

    # Display the plot
    plt.draw()

    # Output to file
    if prefix is not None:
        labels = ['fig6', 'fig7a', 'fig7b', 'fig7c', 'fig7d', 'fig7e']
        if len(labels) != len(figs):
            print('**Error saving figures 3.6 and 3.7; unexpected number of figures returned from Example 3.2.')
        else:
            for fig, label in zip(figs, labels):
                fig.savefig(prefix + label + '.svg')
                fig.savefig(prefix + label + '.png')

    return figs


def make_figures_9_10(prefix=None):
    """
    Figures 3.9 & 3.10

    :param prefix: output directory to place generated figure
    :return: handle
    """

    print('Generating Figures 3.9, 3.10a, 3.10b, 3.10c (Example 3.3)...')

    figs = chapter3.example3()

    # Display the plot
    plt.draw()

    # Output to file
    if prefix is not None:
        labels = ['fig9', 'fig10a', 'fig10b', 'fig10c']
        if len(labels) != len(figs):
            print('**Error saving figures 3.9 and 3.10; unexpected number of figures returned from Example 3.3.')
        else:
            for fig, label in zip(figs, labels):
                fig.savefig(prefix + label + '.svg')
                fig.savefig(prefix + label + '.png')

    return figs


def make_figures_11_12(prefix=None, force_recalc=False):
    """
    Figures 3.11 and 3.12

    :param prefix: output directory to place generated figure
    :param force_recalc: boolean flag to force recalculation of figure 8
    :return: figure handle
    """

    if not force_recalc:
        print('Skipping Figures 3.11, and 3.12 (re-run with force_recalc=True to generate)...')
        return None, None

    print('Generating Figure 3.11, 3.12 (Example 3.4)...')

    figs = chapter3.example4()

    # Display the plot
    plt.draw()

    # Output to file
    if prefix is not None:
        labels = ['fig11', 'fig12']
        if len(labels) != len(figs):
            print('**Error saving figures 3.11 and 3.12; unexpected number of figures returned from Example 3.4.')
        else:
            for fig, label in zip(figs, labels):
                fig.savefig(prefix + label + '.svg')
                fig.savefig(prefix + label + '.png')

    return figs


if __name__ == "__main__":
    make_all_figures(close_figs=False, force_recalc=True)
