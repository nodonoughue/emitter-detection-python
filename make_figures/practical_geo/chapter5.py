"""
Draw Figures - Chapter 5

This script generates all the figures that appear in Chapter 5 of the textbook.

Ported from MATLAB Code

Nicholas O'Donoughue
1 April 2025
"""

import utils
import matplotlib.pyplot as plt
import numpy as np
import scipy

from examples.practical_geo import chapter5


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
    prefix = utils.init_output_dir('practical_geo/chapter5')
    utils.init_plot_style()

    # Generate all figures
    figs_5_6 = make_figures_5_6(prefix, force_recalc=force_recalc)
    fig_7 = make_figure_7(prefix, force_recalc=force_recalc)
    fig_8 = make_figure_8(prefix, force_recalc=force_recalc)
    fig_11 = make_figure_11(prefix, force_recalc=force_recalc)
    fig_12 = make_figure_12(prefix)
    fig_13 = make_figure_13(prefix)
    fig_14 = make_figure_14(prefix, force_recalc=force_recalc)

    figs = list(figs_5_6) + list(fig_7) + list(fig_8) + list(fig_11) + list(fig_12) + list(fig_13) + list(fig_14)
    if close_figs:
        for fig in figs:
            plt.close(fig)

        return None
    else:
        plt.show()

    return figs


def make_figures_5_6(prefix=None, force_recalc=False):
    """
    Figure 5.5 and 5.6 from Example 5.1

    :param prefix: output directory to place generated figure
    :param force_recalc: optional flag (default=True), if False then the example does not run
    :return: handle
    """

    if not force_recalc:
        print('Skipping Figures 5.5, and 5.6 (re-run with force_recalc=True to generate)...')
        return None, None

    print('Generating Figures 5.5, 5.6...')

    figs = chapter5.example1()

    # Display the plot
    plt.draw()

    # Output to file
    if prefix is not None:
        labels = ['fig5', 'fig6a', 'fig6b']
        if len(labels) != len(figs):
            print('**Error saving figures 5.5 and 5.6; unexpected number of figures returned from Example 5.1.')
        else:
            for fig, label in zip(figs, labels):
                fig.savefig(prefix + label + '.svg')
                fig.savefig(prefix + label + '.png')

    return figs


def make_figure_7(prefix=None, force_recalc=False):
    """
    Figure 5.7 from Example 5.2

    :param prefix: output directory to place generated figure
    :param force_recalc: optional flag (default=True), if False then the example does not run
    :return: handle
    """

    if not force_recalc:
        print('Skipping Figure 5.7 (re-run with force_recalc=True to generate)...')
        return None, None

    print('Generating Figure 5.7 (from Example 5.2)...')

    figs = chapter5.example2()

    # Display the plot
    plt.draw()

    # Output to file
    if prefix is not None:
        labels = ['fig7a','fig7b','fig7_video5_2']
        if len(labels) != len(figs):
            print('**Error saving figure 5.7; unexpected number of figures returned from Example 5.2.')
        else:
            for fig, label in zip(figs, labels):
                fig.savefig(prefix + label + '.svg')
                fig.savefig(prefix + label + '.png')

    return figs


def make_figure_8(prefix=None, force_recalc=False):
    """
    Figure 5.8 from Example 5.3

    :param prefix: output directory to place generated figure
    :param force_recalc: optional flag (default=True), if False then the example does not run
    :return: handle
    """

    if not force_recalc:
        print('Skipping Figure 5.8 (re-run with force_recalc=True to generate)...')
        return None, None

    print('Generating Figure 5.8 (from Example 5.3)...')

    figs = chapter5.example3()

    # Display the plot
    plt.draw()

    # Output to file
    if prefix is not None:
        labels = ['fig8']
        if len(labels) != len(figs):
            print('**Error saving figure 5.8; unexpected number of figures returned from Example 5.3.')
        else:
            for fig, label in zip(figs, labels):
                fig.savefig(prefix + label + '.svg')
                fig.savefig(prefix + label + '.png')

    return figs


def make_figure_11(prefix=None, force_recalc=False):
    """
    Figure 5.11 from Example 5.4

    :param prefix: output directory to place generated figure
    :param force_recalc: optional flag (default=True), if False then the example does not run
    :return: handle
    """

    if not force_recalc:
        print('Skipping Figure 5.11 (re-run with force_recalc=True to generate)...')
        return None, None

    print('Generating Figure 5.11 (from Example 5.4)...')

    figs = chapter5.example4()

    # Display the plot
    plt.draw()

    # Output to file
    if prefix is not None:
        labels = ['fig11']
        if len(labels) != len(figs):
            print('**Error saving figure 5.11; unexpected number of figures returned from Example 5.4.')
        else:
            for fig, label in zip(figs, labels):
                fig.savefig(prefix + label + '.svg')
                fig.savefig(prefix + label + '.png')

    return figs


def make_figure_12(prefix=None):
    """
    Figure 5.12

    :param prefix: output directory to place generated figure
    :return: handle
    """

    print('Generating Figure 5.12...')

    # Drawing of different barrier functions for inequality bounds

    # Ideal barrier
    u_vec = np.linspace(start=-3., stop=1., num=1001)
    ideal = np.zeros_like(u_vec)
    ideal[u_vec >= 0] = np.inf
    ideal[np.fabs(u_vec) <=.001] = 10

    # Logarithmic Barrier
    t_vec = np.array([.5, 1, 2])  # different log parameters
    log_kernel = np.log(np.fabs(u_vec))
    log_kernel[u_vec >=0] = np.inf
    logarithmic = [-(1/this_t) * log_kernel for this_t in t_vec]

    # Plot
    fig = plt.figure()
    plt.plot(u_vec,ideal,'k--',label='Ideal')
    for this_t, this_l in zip(t_vec, logarithmic):
        plt.plot(u_vec, np.real(this_l), label='t={:.1f}'.format(this_t))

    plt.xlim(-3, 1)
    plt.ylim(-5, 5)
    plt.xlabel('u')
    plt.ylabel('Cost')

    # Output to file
    if prefix is not None:
        label = 'fig12'
        fig.savefig(prefix + label + '.svg')
        fig.savefig(prefix + label + '.png')

    return [fig]


def make_figure_13(prefix=None):
    """
    Figure 5.13

    :param prefix: output directory to place generated figure
    :return: handle
    """

    print('Generating Figure 5.13...')

    # Axes
    y_axis = np.linspace(start=0, stop=10, num=101)
    x_axis = np.linspace(start=-10, stop=10, num=1001)
    # Define the corners of the grid, for imshow. Cast as float types to avoid type warning
    extent = (float(x_axis[0]), float(x_axis[-1]), float(y_axis[0]), float(y_axis[-1]))

    xx, yy = np.meshgrid(x_axis, y_axis)

    # Data Likelihood
    y_ctr = 5.
    x_ctr = 0.
    mean_vec = x_ctr - ((np.fabs(y_axis-y_ctr)**2)/4)
    mean_val = x_ctr - (np.fabs(yy-y_ctr)**2)/4
    std_dev = 3.

    ell = scipy.stats.norm.pdf(x=xx, loc=mean_val, scale=std_dev)

    # Subfigure generator
    def _make_sub_fig(data, vec=None, title=''):
        this_fig = plt.figure()
        plt.imshow(data, origin='lower', extent=extent, cmap='viridis')
        if vec is not None:
            plt.plot(vec, y_axis, 'k--')
        plt.grid(True, color='w')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(title)
        return this_fig

    # First Subplot -- Raw Likelihood
    figs = [_make_sub_fig(ell, vec=mean_vec, title='Data Likelihood')]

    # A priori
    pos = np.dstack((xx, yy))
    rv = scipy.stats.multivariate_normal(mean=np.array([x_ctr, y_ctr]), cov=np.diag([20, 2]))
    prior = np.reshape(rv.pdf(pos), xx.shape)

    figs.append(_make_sub_fig(prior, title='Prior Distribution on Target Location'))

    # Posterior
    figs.append(_make_sub_fig(prior*ell, title='Posterior Distribution on Target Location'))

    # Output to file
    if prefix is not None:
        labels = ['fig13a','fig13b','fig13c']
        if len(labels) != len(figs):
            print('**Error saving figure 5.13; unexpected number of figures generated.')
        else:
            for fig, label in zip(figs, labels):
                fig.savefig(prefix + label + '.svg')
                fig.savefig(prefix + label + '.png')

    return figs


def make_figure_14(prefix=None, force_recalc=False):
    """
    Figure 5.14 from Example 5.5

    :param prefix: output directory to place generated figure
    :param force_recalc: optional flag (default=True), if False then the example does not run
    :return: handle
    """

    if not force_recalc:
        print('Skipping Figure 5.14 (re-run with force_recalc=True to generate)...')
        return None, None

    print('Generating Figure 5.14 (from Example 5.5)...')

    figs = chapter5.example5()

    # Display the plot
    plt.draw()

    # Output to file
    if prefix is not None:
        labels = ['fig14a','fig14b']
        if len(labels) != len(figs):
            print('**Error saving figure 5.14; unexpected number of figures returned from Example 5.5.')
        else:
            for fig, label in zip(figs, labels):
                fig.savefig(prefix + label + '.svg')
                fig.savefig(prefix + label + '.png')

    return figs
