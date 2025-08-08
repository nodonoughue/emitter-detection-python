"""
Draw Figures - Chapter 7

This script generates all the figures that appear in Chapter 7 of the textbook.

Ported from MATLAB Code

Nicholas O'Donoughue
28 June 2025
"""

import utils
import matplotlib.pyplot as plt

from examples.practical_geo import chapter8


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
    prefix = utils.init_output_dir('practical_geo/chapter8')
    utils.init_plot_style()

    # Generate all figures
    figs3_4_5 = make_figures_3_4_5(prefix, force_recalc)
    figs6_7 = make_figures_6_7(prefix, force_recalc)

    figs = list(figs3_4_5) +  list(figs6_7)
    if close_figs:
        [plt.close(fig) for fig in figs]
        return None
    else:
        # Display the plots
        plt.show()

    return figs


def make_figures_3_4_5(prefix=None, force_recalc=False):
    """
    Figures 8.3, 8.4, and 8.5 from Example 8.1

    :param prefix: output directory to place generated figure
    :param force_recalc: optional flag (default=True), if False then the example does not run
    :return: handle
    """

    if not force_recalc:
        print('Skipping Figures 8.3, 8.4, and 8.5 (re-run with force_recalc=True to generate)...')
        return None,

    print('Generating Figures 8.3, 8.4, and 8.5 (Example 8.1)...')

    figs = chapter8.example1()

    # Output to file
    if prefix is not None:
        labels = ['fig3', 'fig4', 'fig5']
        if len(labels) != len(figs):
            print('**Error saving figure 8.3, 8.4, and 8.5; unexpected number of figures returned from Example 8.1.')
        else:
            for fig, label in zip(figs, labels):
                fig.savefig(prefix + label + '.svg')
                fig.savefig(prefix + label + '.png')

    return figs


def make_figures_6_7(prefix=None, force_recalc=False):
    """
    Figures 8.6 and 8.7 from Example 7.2

    :param prefix: output directory to place generated figure
    :param force_recalc: optional flag (default=True), if False then the example does not run
    :return: handle
    """

    if not force_recalc:
        print('Skipping Figures 8.6 and 8.7 (re-run with force_recalc=True to generate)...')
        return None,

    print('Generating Figures 8.6 and 8.7 (Example 8.2)...')

    figs = chapter8.example2()

    # Output to file
    if prefix is not None:
        labels = ['fig6a', 'fig6b', 'fig7']
        if len(labels) != len(figs):
            print('**Error saving figure 8.6 and 8.7; unexpected number of figures returned from Example 8.2.')
        else:
            for fig, label in zip(figs, labels):
                fig.savefig(prefix + label + '.svg')
                fig.savefig(prefix + label + '.png')

    return figs


if __name__ == "__main__":
    make_all_figures(close_figs=False, force_recalc=True)
