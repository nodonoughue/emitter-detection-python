"""
Draw Figures - Chapter 6

This script generates all of the figures that appear in Chapter 6 of the textbook.

Ported from MATLAB Code

Nicholas O'Donoughue
25 March 2021
"""

import utils
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def make_all_figures(close_figs=False):
    """
    Call all the figure generators for this chapter

    :close_figs: Boolean flag.  If true, will close all figures after generating them; for batch scripting.
                 Default=False
    :return: List of figure handles
    """

    # Find the output directory
    prefix = utils.init_output_dir('chapter6')

    # Activate seaborn for prettier plots
    sns.set()

    # Generate all figures
    fig1 = make_figure_1(prefix)
    fig2 = make_figure_2(prefix)
    fig3 = make_figure_3(prefix)
    fig4 = make_figure_4(prefix)

    figs = [fig1, fig2, fig3, fig4]

    if close_figs:
        for fig in figs:
            plt.close(fig)

        return None
    else:
        plt.show()

        return figs


def make_figure_1(prefix=None):
    """
    Figure 1, Bayesian Example

    Ported from MATLAB Code

    Nicholas O'Donoughue
    25 March 2021

    :param prefix: output directory to place generated figure
    :return: figure handle
    """
    
    d_lam_narrow_beam = 4
    num_array_elements = 10
    psi_0 = 135*np.pi/180
    # psi_0 = 95*np.pi/180
    
    # Narrow Beam (Marginal Distribution)
    def element_pattern(psi):
        return np.absolute(np.cos(psi-np.pi/2))**1.2
    
    def array_function_narrow(psi):
        numerator = np.sin(np.pi*d_lam_narrow_beam*num_array_elements*(np.cos(psi)-np.cos(psi_0)))
        denominator = np.sin(np.pi*d_lam_narrow_beam*(np.cos(psi)-np.cos(psi_0)))

        # Throw in a little error handling; division by zero throws a runtime warning
        limit_mask = denominator == 0
        valid_mask = np.logical_not(limit_mask)
        valid_result = np.absolute(numerator[valid_mask] / denominator[valid_mask])/num_array_elements

        return np.piecewise(x=psi, condlist=[limit_mask],
                            funclist=[1, valid_result])

    # Wide Beam (Prior Distribution)
    d_lam_wide_beam = .5

    def array_function_wide(psi):
        numerator = np.sin(np.pi*d_lam_wide_beam*num_array_elements*(np.cos(psi)-np.cos(psi_0)))
        denominator = np.sin(np.pi*d_lam_wide_beam*(np.cos(psi)-np.cos(psi_0)))

        # Throw in a little error handling; division by zero throws a runtime warning
        limit_mask = denominator == 0
        valid_mask = np.logical_not(limit_mask)
        valid_result = np.absolute(numerator[valid_mask] / denominator[valid_mask])/num_array_elements

        return np.piecewise(x=psi, condlist=[limit_mask],
                            funclist=[1, valid_result])
    
    fig1 = plt.figure()
    psi_vec = np.arange(start=0, step=np.pi/1000, stop=np.pi)
    plt.plot(psi_vec, element_pattern(psi_vec)*array_function_narrow(psi_vec), label='Narrow Beam (marginal)')
    plt.plot(psi_vec, array_function_wide(psi_vec), label='Wide Beam (prior)')
    plt.xlabel(r'$\psi$')
    plt.legend(loc='upper left')

    # Save figure
    if prefix is not None:
        fig1.savefig(prefix + 'fig1.svg')
        fig1.savefig(prefix + 'fig1.png')

    return fig1


def make_figure_2(prefix=None):
    """
    Figure 2, Convex Optimization Example

    Ported from MATLAB Code

    Nicholas O'Donoughue
    25 March 2021

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    # True position
    x0 = np.array([1, .5])

    # Grid
    xx = np.expand_dims(np.arange(start=-5, step=.01, stop=5), axis=1)
    yy = np.expand_dims(np.arange(start=-3, step=.01, stop=3), axis=0)

    # Broadcast xx and yy
    out_shp = np.broadcast(xx, yy)
    xx_full = np.broadcast_to(xx, out_shp.shape)
    yy_full = np.broadcast_to(yy, out_shp.shape)

    # Crude cost function; the different weights force an elliptic shape
    f = 1.*(xx-x0[0])**2 + 5.*(yy-x0[1])**2

    # Iterative estimates
    x_est = np.array([[-3, 2],
                      [2, -1.5],
                      [1, 2.2],
                      [1,  .6]])

    # Plot results
    fig2 = plt.figure()
    plt.contour(xx_full, yy_full, f)
    plt.scatter(x0[0], x0[1], marker='^', label='True Minimum')
    plt.plot(x_est[:, 0], x_est[:, 1], linestyle='--', marker='+', label='Estimate')

    plt.text(-3, 2.1, 'Initial Estimate', fontsize=10)
    plt.legend(loc='lower left')

    if prefix is not None:
        fig2.savefig(prefix + 'fig2.svg')
        fig2.savefig(prefix + 'fig2.png')

    return fig2


def make_figure_3(prefix=None):
    """
    Figure 3, Tracker Example

    Ported from MATLAB Code

    Nicholas O'Donoughue
    25 March 2021

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    # Measurements
    y = np.array([1, 1.1,  1.3, 1.4,  1.35,  1.3,  .7, .75])
    # Estimates
    x = np.array([1, 1.05, 1.2, 1.35, 1.45, 1.35, 1.2, .8])
    # Confidence Intervals
    s2 = np.array([.8, .5, .4, .3, .3, .2, .2, .6])

    num_updates = y.size

    # Plot result
    fig3 = plt.figure()
    for i in np.arange(start=1, stop=s2.size):
        plt.fill(i + np.array([.2, .2, -.2, -.2, .2]),
                 x[i] + s2[i-1]*np.array([-1, 1, 1, -1, -1]),
                 color=(.8, .8, .8), label=None)

    x_vec = np.arange(num_updates)
    plt.scatter(x_vec, y, marker='x', label='Measurement', zorder=10)
    plt.plot(x_vec, x, linestyle='-.', marker='o', label='Estimate')
    plt.legend(loc='upper left')
    plt.xlabel('Time')
    plt.ylabel(r'Parameter ($\theta$)')

    if prefix is not None:
        fig3.savefig(prefix + 'fig3.svg')
        fig3.savefig(prefix + 'fig3.png')

    return fig3


def make_figure_4(prefix=None):
    """
    Figure 4, Angle Error Variance

    Ported from MATLAB Code

    Nicholas O'Donoughue
    25 March 2021

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    # Sensor Coordinates
    x0 = np.array([0, 0])
    # xs = np.array([2, 1.4])

    # Bearing and confidence interval
    aoa = 40
    aoa_rad = np.deg2rad(aoa)
    std_dev = 8
    length = 5
    aoa_lwr_rad = np.deg2rad(aoa - std_dev)
    aoa_up_rad = np.deg2rad(aoa + std_dev)

    # Plot Result
    fig4 = plt.figure()
    plt.scatter(x0[0], x0[1], marker='o')
    plt.fill(x0[0] + np.array([0, length*np.cos(aoa_lwr_rad), length*np.cos(aoa_rad), length*np.cos(aoa_up_rad), 0]),
             x0[1] + np.array([0, length*np.sin(aoa_lwr_rad), length*np.sin(aoa_rad), length*np.sin(aoa_up_rad), 0]),
             color=[.8, .8, .8])

    plt.plot(x0[0] + np.array([0, length*np.cos(aoa_rad)]), x0[1] + np.array([0, length*np.sin(aoa_rad)]),
             linestyle='-.', color='k')
    plt.plot(x0[0] + np.array([0, length*np.cos(aoa_lwr_rad)]), x0[1] + np.array([0, length*np.sin(aoa_lwr_rad)]),
             color='k')
    plt.plot(x0[0] + np.array([0, length*np.cos(aoa_up_rad)]), x0[1] + np.array([0, length*np.sin(aoa_up_rad)]),
             color='k')

    plt.text(-.3, .1, 'Sensor', fontsize=10)
    plt.text(.95, .85, 'Estimated Bearing', fontsize=10, rotation=45)
    plt.text(1.6, 1.6, 'Confidence Interval', fontsize=10)

    plt.plot(length/5*np.cos(np.linspace(start=aoa_rad, stop=aoa_up_rad, num=10)),
             length/5*np.sin(np.linspace(start=aoa_rad, stop=aoa_up_rad, num=10)), color='k', linewidth=.5)
    plt.text(length/5*np.cos(aoa_up_rad)-.2, length/5*np.sin(aoa_up_rad)+.1, 'RMSE', fontsize=10)

    plt.xlim([-.5, 2.5])
    plt.ylim([-.2, 1.8])

    if prefix is not None:
        fig4.savefig(prefix + 'fig4.svg')
        fig4.savefig(prefix + 'fig4.png')

    return fig4
