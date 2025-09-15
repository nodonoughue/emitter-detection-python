"""
Draw Figures - Chapter 4

This script generates all the figures that appear in Chapter 4 of the textbook.

Ported from MATLAB Code

Nicholas O'Donoughue
7 February 2025
"""

import utils
import matplotlib.pyplot as plt

from examples.practical_geo import chapter4


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
    prefix = utils.init_output_dir('practical_geo/chapter4')
    utils.init_plot_style()

    # Generate all figures
    figs_10_11 = make_figures_10_11(prefix, mc_params)
    fig12 = make_figure_12(prefix, mc_params)

    figs = list(figs_10_11) + list(fig12)
    if close_figs:
        for fig in figs:
            plt.close(fig)

        return None
    else:
        plt.show()

    return figs


def make_figures_10_11(prefix=None, mc_params=None):
    """
    Figure 4.10 and 4.11 from Example 4.1

    :param prefix: output directory to place generated figure
    :param mc_params: Optional struct to control Monte Carlo trial size
    :return: handle
    """

    if mc_params is not None and 'force_recalc' in mc_params and not mc_params['force_recalc']:
        print('Skipping Figures 4.10, and 4.11 (re-run with mc_params[\'force_recalc\']=True to generate)...')
        return None, None

    print('Generating Figures 4.10, 4.11 (from Example 4.1)...')

    figs = chapter4.example1(mc_params=mc_params)

    # Display the plot
    plt.draw()

    # Output to file
    if prefix is not None:
        labels = ['fig10', 'fig11']
        if len(labels) != len(figs):
            print('**Error saving figures 4.10 and 4.11; unexpected number of figures returned from Example 4.1.')
        else:
            for fig, label in zip(figs, labels):
                fig.savefig(prefix + label + '.svg')
                fig.savefig(prefix + label + '.png')

    return figs


def make_figure_12(prefix=None, mc_params=None):
    """
    Figure 4.12 from Example 4.2

    :param prefix: output directory to place generated figure
    :param mc_params: Optional struct to control Monte Carlo trial size
    :return: handle
    """

    if mc_params is not None and 'force_recalc' in mc_params and not mc_params['force_recalc']:
        print('Skipping Figure 4.12 (re-run with mc_params[\'force_recalc\']=True to generate)...')
        return None, None

    print('Generating Figure 4.12 (from Example 4.2)...')

    figs = chapter4.example2()

    # Display the plot
    plt.draw()

    # Output to file
    if prefix is not None:
        labels = ['fig12']
        if len(labels) != len(figs):
            print('**Error saving figure 4.12; unexpected number of figures returned from Example 4.2.')
        else:
            for fig, label in zip(figs, labels):
                fig.savefig(prefix + label + '.svg')
                fig.savefig(prefix + label + '.png')

    return figs


if __name__ == "__main__":
    make_all_figures(close_figs=False, mc_params={'force_recalc': True, 'monte_carlo_decimation': 1, 'min_num_monte_carlo': 1})
