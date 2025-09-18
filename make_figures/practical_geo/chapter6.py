"""
Draw Figures - Chapter 6

This script generates all the figures that appear in Chapter 6 of the textbook.

Ported from MATLAB Code

Nicholas O'Donoughue
7 April 2025
"""

import matplotlib.pyplot as plt
import numpy as np

import ewgeo.triang as triang
from ewgeo.utils import init_output_dir, init_plot_style, safe_2d_shape
from ewgeo.utils.geo import calc_range, find_intersect

from examples.practical_geo import chapter6


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
    prefix = init_output_dir('practical_geo/chapter6')
    init_plot_style()

    # Generate all figures
    fig1 = make_figure_1(prefix, mc_params)
    fig2 = make_figure_2(prefix, mc_params)
    fig6 = make_figure_6(prefix)
    fig8_9 = make_figures_8_9(prefix, mc_params)
    fig12 = make_figure_12(prefix, mc_params)

    figs = list(fig1) + list(fig2) + list(fig6) + list(fig8_9) + list(fig12)
    if close_figs:
        [plt.close(fig) for fig in figs]
        return None
    else:
        # Display the plots
        plt.show()

    return figs


def make_figure_1(prefix=None, mc_params=None):
    """
    Figure 6.1 from Example 6.1

    :param prefix: output directory to place generated figure
    :param mc_params: Optional struct to control Monte Carlo trial size
    :return: handle
    """

    if mc_params is not None and 'force_recalc' in mc_params and not mc_params['force_recalc']:
        print('Skipping Figure 6.1 from Example 6.1 (re-run with mc_params[\'force_recalc\']=True to generate)...')
        return None, None

    print('Generating Figure 6.1 (Example 6.1)...')

    figs = chapter6.example1()

    # Output to file
    if prefix is not None:
        labels = ['fig1']
        if len(labels) != len(figs):
            print('**Error saving figure 6.1; unexpected number of figures returned from Example 6.1.')
        else:
            for fig, label in zip(figs, labels):
                fig.savefig(prefix + label + '.svg')
                fig.savefig(prefix + label + '.png')

    return figs


def make_figure_2(prefix=None, mc_params=None):
    """
    Figure 6.2 from Example 6.2 and figure 6.5 from Example 6.3

    :param prefix: output directory to place generated figure
    :param mc_params: Optional struct to control Monte Carlo trial size
    :return: handle
    """

    if mc_params is not None and 'force_recalc' in mc_params and not mc_params['force_recalc']:
        print('Skipping Figures 6.2 and 6.5 from Examples 6.2 and 6.3 (re-run with mc_params[\'force_recalc\']=True to generate)...')
        return None,

    print('Generating Figures 6.2 and 6.5 from Examples 6.2 and 6.3...')

    figs = chapter6.example2()
    figs = figs + list(chapter6.example3())

    # Output to file
    if prefix is not None:
        labels = ['fig2', 'fig5']
        if len(labels) != len(figs):
            print('**Error saving figures 6.2 and 6.5; unexpected number of figures returned from Examples 6.2 and 6.3.')
        else:
            for fig, label in zip(figs, labels):
                fig.savefig(prefix + label + '.svg')
                fig.savefig(prefix + label + '.png')

    return figs


def make_figure_6(prefix=None):
    """
    Figure 6.6, Impact of Sensor Position Errors

    :param prefix: output directory to place generated figure
    :return: handle
    """

    print('Generating Figures 6.6a and 6.6b...')

    # Define Positions
    x_aoa = np.array([[-1., 1.], [0., 0.]])
    x_target = np.array([0., 5.])
    x_err = np.array([1., 0.])
    _, num_sensor = safe_2d_shape(x_aoa)

    x_aoa_un = x_aoa + x_err[:, np.newaxis]
    x_aoa_nonun = x_aoa + x_err[:, np.newaxis] @ np.array([[-1, 1]])

    # Solve True LOBs and AOAs
    # lob = x_target[:, np.newaxis] - x_aoa
    psi = triang.model.measurement(x_sensor=x_aoa, x_source=x_target)
    r = calc_range(x1=x_aoa, x2=x_target)

    # Solve Erroneous AOAs
    x_tgt_un = find_intersect(x0=x_aoa_un[:, 0], psi0=psi[0], x1=x_aoa_un[:, 1], psi1=psi[1])
    x_tgt_nonun = find_intersect(x0=x_aoa_nonun[:, 0], psi0=psi[0], x1=x_aoa_nonun[:, 1], psi1=psi[1])

    # Generate the lobs, dimensions are num_dim x num_tx x 2 (datapoints)
    # num_dim x num_sensor x 2
    r_vec = np.concatenate((np.zeros((num_sensor, 1)), 3*r[:, np.newaxis]), axis=1)  # num_sensor x 2
    lob_zero = np.array([np.cos(psi[:, np.newaxis]), np.sin(psi[:, np.newaxis])]) * r_vec[np.newaxis, :]

    lob_true = x_aoa[:, :, np.newaxis] + lob_zero
    lob_un = x_aoa_un[:, :, np.newaxis] + lob_zero
    lob_nonun = x_aoa_nonun[:, :, np.newaxis] + lob_zero

    # Draw Uniform Offset case
    figs = [plt.figure()]
    hdl_true = plt.scatter(x_aoa[0], x_aoa[1],
                           marker='^', label='True Sensor Position', clip_on=False)
    hdl_err = plt.scatter(x_aoa_un[0], x_aoa_un[1],
                          marker='v', label='Est. Sensor Position', clip_on=False)
    plt.scatter(x_target[0], x_target[1],
                marker='o', color=hdl_true.get_facecolor(), label='Target', clip_on=False)
    plt.scatter(x_tgt_un[0], x_tgt_un[1],
                marker='o', color=hdl_err.get_facecolor(), label='Est. Target', clip_on=False)
    plt.plot(lob_true[0].T, lob_true[1].T, color=hdl_true.get_facecolor(), label=('True LOB', None))
    plt.plot(lob_un[0].T, lob_un[1].T, color=hdl_err.get_facecolor(), label=('Perceived LOB', None))
    plt.xlim([-6, 6])
    plt.ylim([0, 11])
    plt.legend(loc='upper left')
    plt.title('Uniform Sensor Position Errors')

    # Draw Non-Uniform Offset case
    figs.append(plt.figure())
    hdl_true = plt.scatter(x_aoa[0], x_aoa[1],
                           marker='^', label='True Sensor Position', clip_on=False)
    hdl_err = plt.scatter(x_aoa_nonun[0], x_aoa_nonun[1],
                          marker='v', label='Est. Sensor Position', clip_on=False)
    plt.scatter(x_target[0], x_target[1],
                marker='o', color=hdl_true.get_facecolor(), label='Target', clip_on=False)
    plt.scatter(x_tgt_nonun[0], x_tgt_nonun[1],
                marker='o', color=hdl_err.get_facecolor(), label='Est. Target', clip_on=False)
    plt.plot(lob_true[0].T, lob_true[1].T, color=hdl_true.get_facecolor(), label=('True LOB', None))
    plt.plot(lob_nonun[0].T, lob_nonun[1].T, color=hdl_err.get_facecolor(), label=('Perceived LOB', None))
    plt.xlim([-6, 6])
    plt.ylim([0, 11])
    plt.legend(loc='upper left')
    plt.title('Non-Uniform Sensor Position Errors')

    # Output to file
    if prefix is not None:
        labels = ['fig6a', 'fig6b']
        if len(labels) != len(figs):
            print('**Error saving figure 6.6; unexpected number of figures generated.')
        else:
            for fig, label in zip(figs, labels):
                fig.savefig(prefix + label + '.svg')
                fig.savefig(prefix + label + '.png')

    return figs


def make_figures_8_9(prefix=None, mc_params=None):
    """
    Figures 6.8 and 6.9, from Example 6.4

    :param prefix: output directory to place generated figure
    :param mc_params: Optional struct to control Monte Carlo trial size
    :return: handle
    """

    if mc_params is not None and 'force_recalc' in mc_params and not mc_params['force_recalc']:
        print('Skipping Figures 6.8 and 6.9 from Example 6.4 (re-run with mc_params[\'force_recalc\']=True to generate)...')
        return None, None

    print('Generating Figures 6.8 and 6.9, from Example 6.4...')

    figs = chapter6.example4()

    # Output to file
    if prefix is not None:
        labels = ['fig8a', 'fig8b', 'fig9a', 'fig9b']
        if len(labels) != len(figs):
            print('**Error saving figures 6.8 and 6.8; unexpected number of figures returned from Example 6.4.')
        else:
            for fig, label in zip(figs, labels):
                fig.savefig(prefix + label + '.svg')
                fig.savefig(prefix + label + '.png')

    return figs


def make_figure_12(prefix=None, mc_params=None):
    """
    Figures 6.12, from Example 6.5

    :param prefix: output directory to place generated figure
    :param mc_params: Optional struct to control Monte Carlo trial size
    :return: handle
    """

    if mc_params is not None and 'force_recalc' in mc_params and not mc_params['force_recalc']:
        print('Skipping Figure 6.12 from Example 6.5 (re-run with mc_params[\'force_recalc\']=True to generate)...')
        return None, None

    print('Generating Figure 6.12, from Example 6.5...')

    figs = chapter6.example5()

    # Output to file
    if prefix is not None:
        labels = ['fig12']
        if len(labels) != len(figs):
            print('**Error saving figures 6.12; unexpected number of figures returned from Example 6.5.')
        else:
            for fig, label in zip(figs, labels):
                fig.savefig(prefix + label + '.svg')
                fig.savefig(prefix + label + '.png')

    return figs


if __name__ == "__main__":
    make_all_figures(close_figs=False, mc_params={'force_recalc': True, 'monte_carlo_decimation': 1, 'min_num_monte_carlo': 1})
