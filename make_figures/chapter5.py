"""
Draw Figures - Chapter 5

This script generates all of the figures that appear in Chapter 5 of the textbook.

Ported from MATLAB Code

Nicholas O'Donoughue
25 March 2021
"""

import utils
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from examples import chapter5


def make_all_figures(close_figs=False):
    """
    Call all the figure generators for this chapter

    :close_figs: Boolean flag.  If true, will close all figures after generating them; for batch scripting.
                 Default=False
    :return: List of figure handles
    """

    # Initializes colorSet - Mx3 RGB vector for successive plot lines
    colors = plt.get_cmap("tab10")

    # Reset the random number generator, to ensure reproducability
    rng = np.random.default_rng(0)

    # Find the output directory
    prefix = utils.init_output_dir('chapter5')

    # Activate seaborn for prettier plots
    sns.set()

    # Generate all figures
    fig4 = make_figure_4(prefix, rng, colors)
    fig6 = make_figure_6(prefix, rng, colors)
    fig7 = make_figure_7(prefix, rng, colors)

    figs = [fig4, fig6, fig7]

    if close_figs:
        for fig in figs:
            plt.close(fig)

        return None
    else:
        plt.show()

        return figs


def make_figure_4(prefix=None, rng=None, colors=None):
    """
    Figure 4 - Example 5.1 - Superhet Performance

    Ported from MATLAB Code

    Nicholas O'Donoughue
    25 March 2021

    :param prefix: output directory to place generated figure
    :param rng: random number generator
    :param colors: colormap for plots
    :return: figure handle
    """

    if rng is None:
        rng = np.random.default_rng(0)
    
    if colors is None:
        colors = plt.get_cmap('tab10')

    fig4 = chapter5.example1()

    # Save figure
    if prefix is not None:
        plt.savefig(prefix + 'fig4.svg')
        plt.savefig(prefix + 'fig4.png')

    return fig4


def make_figure_6(prefix=None, rng=None, colors=None):
    """
    Figure 6 - Example 5.2 - FMCW Radar

    Ported from MATLAB Code

    Nicholas O'Donoughue
    25 March 2021

    :param prefix: output directory to place generated figure
    :param rng: random number generator
    :param colors: colormap for plots
    :return: figure handle
    """

    if rng is None:
        rng = np.random.default_rng(0)

    if colors is None:
        colors = plt.get_cmap('tab10')

    fig6 = chapter5.example2()

    # Save figure
    if prefix is not None:
        plt.savefig(prefix + 'fig6.svg')
        plt.savefig(prefix + 'fig6.png')

    return fig6


def make_figure_7(prefix=None, rng=None, colors=None):
    """
    Figure 7 - Example 5.3 - Pulsed Radar

    Ported from MATLAB Code

    Nicholas O'Donoughue
    25 March 2021

    :param prefix: output directory to place generated figure
    :param rng: random number generator
    :param colors: colormap for plots
    :return: figure handle
    """

    if rng is None:
        rng = np.random.default_rng(0)

    if colors is None:
        colors = plt.get_cmap('tab10')

    fig7 = chapter5.example3()

    # Save figure
    if prefix is not None:
        plt.savefig(prefix + 'fig7.svg')
        plt.savefig(prefix + 'fig7.png')

    return fig7
