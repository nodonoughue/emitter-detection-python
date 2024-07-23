"""
Draw Figures - Chapter 5

This script generates all the figures that appear in Chapter 5 of the textbook.

Ported from MATLAB Code

Nicholas O'Donoughue
25 March 2021
"""

import utils
import matplotlib.pyplot as plt
import seaborn as sns
from examples import chapter5


def make_all_figures(close_figs=False):
    """
    Call all the figure generators for this chapter

    :close_figs: Boolean flag.  If true, will close all figures after generating them; for batch scripting.
                 Default=False
    :return: List of figure handles
    """

    # Find the output directory
    prefix = utils.init_output_dir('chapter5')

    # Activate seaborn for prettier plots
    sns.set()

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

    fig7 = chapter5.example3()

    # Save figure
    if prefix is not None:
        fig7.savefig(prefix + 'fig7.svg')
        fig7.savefig(prefix + 'fig7.png')

    return fig7
