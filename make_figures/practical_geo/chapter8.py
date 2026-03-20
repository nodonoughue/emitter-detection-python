"""
Draw Figures - Chapter 8

This script generates all the figures that appear in Chapter 8 of the textbook.

Ported from MATLAB Code

Nicholas O'Donoughue
28 June 2025
"""

import matplotlib.pyplot as plt

from ewgeo.utils import init_output_dir, init_plot_style

from examples.practical_geo import chapter8


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
    prefix = init_output_dir('practical_geo/chapter8')
    init_plot_style()

    # Generate all figures
    figs = []
    figs.extend(make_figures_3_4_5(prefix, mc_params))
    figs.extend(make_figures_7_8(prefix, mc_params))
    figs.extend(make_figures_10(prefix))
    figs.extend(make_figures_11_12(prefix))

    if close_figs:
        [plt.close(fig) for fig in figs]
        return None
    else:
        # Display the plots
        plt.show()

    return figs


def make_figures_3_4_5(prefix=None, mc_params=None):
    """
    Figures 8.3, 8.4, and 8.5 from Example 8.1

    :param prefix: output directory to place generated figure
    :param mc_params: Optional struct to control Monte Carlo trial size
    :return: handle
    """

    if mc_params is not None and 'force_recalc' in mc_params and not mc_params['force_recalc']:
        print('Skipping Figures 8.3, 8.4, and 8.5 (re-run with mc_params[\'force_recalc\']=True to generate)...')
        return None,

    print('Generating Figures 8.3, 8.4, and 8.5 (Example 8.1)...')

    figs = chapter8.example1(mc_params=mc_params)

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


def make_figures_7_8(prefix=None, mc_params=None):
    """
    Figures 8.7 and 8.8 from Example 7.2

    :param prefix: output directory to place generated figure
    :param mc_params: Optional struct to control Monte Carlo trial size
    :return: handle
    """

    if mc_params is not None and 'force_recalc' in mc_params and not mc_params['force_recalc']:
        print('Skipping Figures 8.7 and 8.8 (re-run with mc_params[\'force_recalc\']=True to generate)...')
        return None,

    print('Generating Figures 8.7 and 8.8 (Example 8.2)...')

    figs = chapter8.example2()

    # Output to file
    if prefix is not None:
        labels = ['fig7a', 'fig7b', 'fig8']
        if len(labels) != len(figs):
            print('**Error saving figure 8.7 and 8.8; unexpected number of figures returned from Example 8.2.')
        else:
            for fig, label in zip(figs, labels):
                fig.savefig(prefix + label + '.svg')
                fig.savefig(prefix + label + '.png')

    return figs


def make_figures_10(prefix=None):
    """
    Figures 8.10a and 8.10b from Example 8.3 (Ballistic Trajectory Tracking)

    :param prefix: output directory to place generated figures
    :return: list of figure handles
    """

    print('Generating Figures 8.10a and 8.10b (Example 8.3)...')

    figs = chapter8.example3()

    if prefix is not None:
        labels = ['fig10a', 'fig10b']
        if len(labels) != len(figs):
            print('**Error saving figures 8.10a and 8.10b; unexpected number of figures returned from Example 8.3.')
        else:
            for fig, label in zip(figs, labels):
                fig.savefig(prefix + label + '.svg')
                fig.savefig(prefix + label + '.png')

    return figs


def make_figures_11_12(prefix=None):
    """
    Figures 8.11a, 8.11b, 8.12a, and 8.12b from Example 8.4 (Constant-Turn Tracking)

    :param prefix: output directory to place generated figures
    :return: list of figure handles
    """

    print('Generating Figures 8.11a, 8.11b, 8.12a, and 8.12b (Example 8.4)...')

    figs = chapter8.example4()

    if prefix is not None:
        labels = ['fig11a', 'fig11b', 'fig12a', 'fig12b']
        if len(labels) != len(figs):
            print('**Error saving figures 8.11a–8.12b; unexpected number of figures returned from Example 8.4.')
        else:
            for fig, label in zip(figs, labels):
                fig.savefig(prefix + label + '.svg')
                fig.savefig(prefix + label + '.png')

    return figs


if __name__ == "__main__":
    make_all_figures(close_figs=False, mc_params={'force_recalc': True, 'monte_carlo_decimation': 1, 'min_num_monte_carlo': 1})
