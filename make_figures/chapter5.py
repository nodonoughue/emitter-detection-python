"""
Draw Figures - Chapter 5

This script generates all the figures that appear in Chapter 5 of the textbook.

Ported from MATLAB Code

Nicholas O'Donoughue
25 March 2021
"""

import matplotlib.pyplot as plt

from ewgeo.utils import init_output_dir, init_plot_style
from examples import chapter5


def make_all_figures(close_figs=False):
    """
    Call all the figure generators for this chapter

    :close_figs: Boolean flag.  If true, will close all figures after generating them; for batch scripting.
                 Default=False
    :return: List of figure handles
    """

    # Find the output directory
    prefix = init_output_dir('chapter5')
    init_plot_style()

    # Generate all figures
    fig4 = make_figure_4(prefix)
    fig6 = make_figure_6(prefix)
    fig7 = make_figure_7(prefix)

    figs = [fig4, fig6, fig7]

    if close_figs:
        for fig in figs:
            plt.close(fig)

        return None
    else:
        plt.show()

        return figs


def make_figure_4(prefix=None):
    """
    Figure 4 - Example 5.1 - Super-heterodyne Performance

    Ported from MATLAB Code

    Nicholas O'Donoughue
    25 March 2021

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    print('Generating Figure 5.4 (using Example 5.1)...')

    fig4 = chapter5.example1()

    # Save figure
    if prefix is not None:
        fig4.savefig(prefix + 'fig4.svg')
        fig4.savefig(prefix + 'fig4.png')

    return fig4


def make_figure_6(prefix=None):
    """
    Figure 6 - Example 5.2 - FMCW Radar

    Ported from MATLAB Code

    Nicholas O'Donoughue
    25 March 2021

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    print('Generating Figure 5.6 (using Example 5.2)...')

    fig6 = chapter5.example2()

    # Save figure
    if prefix is not None:
        fig6.savefig(prefix + 'fig6.svg')
        fig6.savefig(prefix + 'fig6.png')

    return fig6


def make_figure_7(prefix=None):
    """
    Figure 7 - Example 5.3 - Pulsed Radar

    Ported from MATLAB Code

    Nicholas O'Donoughue
    25 March 2021

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    print('Generating Figure 5.7 (using Example 5.3)...')

    fig7 = chapter5.example3()

    # Save figure
    if prefix is not None:
        fig7.savefig(prefix + 'fig7.svg')
        fig7.savefig(prefix + 'fig7.png')

    return fig7


if __name__ == "__main__":
    make_all_figures(close_figs=False)
