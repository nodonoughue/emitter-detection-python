"""
Draw Figures - Chapter 1

This script generates all the figures that appear in Chapter 1 of the textbook.

Ported from MATLAB Code

Nicholas O'Donoughue
21 March 2021
"""

import utils
import matplotlib.pyplot as plt

from make_figures import chapter10
from make_figures import chapter11
from make_figures import chapter12


def make_all_figures(close_figs=False):
    """
    Call all the figure generators for this chapter

    :param close_figs: Boolean flag.  If true, will close all figures after generating them; for batch scripting.
                 Default=False
    :return: List of figure handles
    """

    # Find the output directory
    prefix = utils.init_output_dir('practical_geo/chapter1')
    utils.init_plot_style()

    # Generate all figures
    fig2a = make_figure_2a(prefix)
    fig2b = make_figure_2b(prefix)
    fig2c = make_figure_2c(prefix)

    figs = [fig2a, fig2b, fig2c]
    if close_figs:
        for fig in figs:
            plt.close(fig)

        return None
    else:
        plt.show()

        return figs


def make_figure_2a(prefix=None):
    """
    Figure 2a, Triangulation Example. A reprint of Figure 10.1 from the 2019 text.

    :param prefix: output directory
    :return: figure handle
    """

    print('Generating Figure 1.2a...')

    fig2a = chapter10.make_figure_1(prefix=None)  # use prefix=None to suppress the figure export command

    # Display the plot
    plt.draw()

    # Output to file
    if prefix is not None:
        fig2a.savefig(prefix + 'fig2a.svg')
        fig2a.savefig(prefix + 'fig2a.png')

    return fig2a


def make_figure_2b(prefix=None):
    """
    Figure 2b, TDOA Example. A reprint of Figure 11.1b from the 2019 text.

    :param prefix: output directory
    :return: figure handle
    """

    print('Generating Figure 1.2b...')

    _, fig2b = chapter11.make_figure_1(prefix=None, do_uncertainty=True)

    # Display the plot
    plt.draw()

    # Output to file
    if prefix is not None:
        fig2b.savefig(prefix + 'fig2b.svg')
        fig2b.savefig(prefix + 'fig2b.png')

    return fig2b


def make_figure_2c(prefix=None):
    """
    Figure 2c, FDOA Example

    :param prefix: output directory
    :return: figure handle
    """

    print('Generating Figure 1.2c...')

    fig2c = chapter12.make_figure_1(prefix=None, do_uncertainty=True)

    # Display the plot
    plt.draw()

    # Output to file
    if prefix is not None:
        fig2c.savefig(prefix + 'fig2c.svg')
        fig2c.savefig(prefix + 'fig2c.png')

    return fig2c
