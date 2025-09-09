"""
Draw Figures - Chapter 2

This script generates all the figures that appear in Chapter 2 of the textbook.

Ported from MATLAB Code

Nicholas O'Donoughue
9 January 2025
"""

import utils
import matplotlib.pyplot as plt
# import numpy as np

from make_figures import chapter10
from make_figures import chapter11
from make_figures import chapter12
from make_figures import chapter13

from examples.practical_geo import chapter2


def make_all_figures(close_figs=False, mc_params=None):
    """
    Call all the figure generators for this chapter

    :param close_figs: Boolean flag.  If true, will close all figures after generating them; for batch scripting.
                 Default=False
    :param mc_params: Optional struct to control Monte Carlo trial size
    :return: List of figure handles
    """

    # Reset the random number generator, to ensure reproducibility
    # rng = np.random.default_rng()

    # Find the output directory
    prefix = utils.init_output_dir('practical_geo/chapter2')
    utils.init_plot_style()

    # Generate all figures
    fig1 = make_figure_1(prefix)
    fig2 = make_figure_2(prefix)
    fig3 = make_figure_3(prefix)
    fig4 = make_figure_4(prefix)
    fig5, fig6a, fig6b, fig6c, fig6d = make_figures_5_6(prefix, mc_params)
    fig8a, fig8b = make_figure_8(prefix, mc_params)
    fig10a, fig10b = make_figure_10(prefix, mc_params)

    figs = [fig1, fig2, fig3, fig4, fig5, fig6a, fig6b, fig6c, fig6d, fig8a, fig8b, fig10a, fig10b]
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

    :param prefix: output directory
    :return: figure handle
    """

    print('Generating Figure 2.1...')
    fig1 = chapter10.make_figure_2(prefix=None)

    # Display the plot
    plt.draw()

    # Output to file
    if prefix is not None:
        fig1.savefig(prefix + 'fig1.svg')
        fig1.savefig(prefix + 'fig1.png')

    return fig1


def make_figure_2(prefix=None):
    """
    Figure 2, TDOA Example. A reprint of Figure 11.1b from the 2019 text.

    :param prefix: output directory
    :return: figure handle
    """

    print('Generating Figure 2.2...')

    _, fig2 = chapter11.make_figure_1(prefix=None)

    # Display the plot
    plt.draw()

    # Output to file
    if prefix is not None:
        fig2.savefig(prefix + 'fig2.svg')
        fig2.savefig(prefix + 'fig2.png')

    return fig2


def make_figure_3(prefix=None):
    """
    Figure 3, FDOA Example. Recreation of Figure 12.1 from 2019 text.

    :param prefix: output directory
    :return: figure handle
    """

    print('Generating Figure 2.3...')
    fig3 = chapter12.make_figure_1(prefix=None)

    # Display the plot
    plt.draw()

    # Output to file
    if prefix is not None:
        fig3.savefig(prefix + 'fig3.svg')
        fig3.savefig(prefix + 'fig3.png')

    return fig3


def make_figure_4(prefix=None):
    """
    Figure 4

    :param prefix: output directory
    :return: figure handle
    """

    print('Generating Figure 2.4...')
    fig4 = chapter13.make_figure_1(prefix=None)

    # Display the plot
    plt.draw()

    # Output to file
    if prefix is not None:
        fig4.savefig(prefix + 'fig4.svg')
        fig4.savefig(prefix + 'fig4.png')

    return fig4


def make_figures_5_6(prefix=None, mc_params=None):
    """
    Figures 5 and 6(a, b, c, d)

    :param prefix: output directory to place generated figure
    :param mc_params: Optional struct to control Monte Carlo trial size
    :return: figure handle
    """

    if mc_params is not None and 'force_recalc' in mc_params and not mc_params['force_recalc']:
        print('Skipping Figures 2.5, 2.6a, 2.6b, 2.6c, and 2.6d (re-run with mc_params[\'force_recalc\']=True to generate)...')
        return None, None, None, None, None

    print('Generating Figures 2.5, 2.6a, 2.6b, 2.6c, and 2.6d (using Example 2.1)...')

    figs = chapter2.example1()

    # Display the plot
    plt.draw()

    # Output to file
    if prefix is not None:
        labels = ['fig5', 'fig6a', 'fig6b', 'fig6c', 'fig6d']
        if len(labels) != len(figs):
            print('**Error saving figures 2.5 and 2.6; unexpected number of figures returned from Example 2.1.')
        else:
            for fig, label in zip(figs, labels):
                fig.savefig(prefix + label + '.svg')
                fig.savefig(prefix + label + '.png')

    return figs


def make_figure_8(prefix=None, mc_params=None):
    """
    Figure 8

    :param prefix: output directory to place generated figure
    :param mc_params: Optional struct to control Monte Carlo trial size
    :return: figure handle
    """

    if mc_params is not None and 'force_recalc' in mc_params and not mc_params['force_recalc']:
        print('Skipping Figures 2.8a and 2.8b (re-run with mc_params[\'force_recalc\']=True to generate)...')
        return None, None

    print('Generating Figure 2.8a and 2.8b...')

    fig8a, fig8b = chapter2.example2()

    # Display the plot
    plt.draw()

    # Output to file
    if prefix is not None:
        fig8a.savefig(prefix + 'fig8a.svg')
        fig8a.savefig(prefix + 'fig8a.png')

        fig8b.savefig(prefix + 'fig8b.svg')
        fig8b.savefig(prefix + 'fig8b.png')

    return fig8a, fig8b


def make_figure_10(prefix=None, mc_params=None):
    """
    Figure 10

    :param prefix: output directory to place generated figure
    :param mc_params: Optional struct to control Monte Carlo trial size
    :return: figure handle
    """

    if mc_params is not None and 'force_recalc' in mc_params and not mc_params['force_recalc']:
        print('Skipping Figures 2.10a and 2.10b (re-run with mc_params[\'force_recalc\']=True to generate)...')
        return None, None

    print('Generating Figures 2.10a and 2.10b (using Example 2.3)...')

    fig10a, fig10b = chapter2.example3()

    # Display the plot
    plt.draw()

    # Output to file
    if prefix is not None:
        fig10a.savefig(prefix + 'fig10a.svg')
        fig10a.savefig(prefix + 'fig10a.png')

        fig10b.savefig(prefix + 'fig10b.svg')
        fig10b.savefig(prefix + 'fig10b.png')

    return fig10a, fig10b


if __name__ == "__main__":
    make_all_figures(close_figs=False, mc_params={'force_recalc': True})
