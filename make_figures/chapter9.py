"""
Draw Figures - Chapter 9

This script generates all of the figures that appear in Chapter 8 of the textbook.

Ported from MATLAB Code

Nicholas O'Donoughue
17 May 2021
"""

import utils
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from examples import chapter9


def make_all_figures(close_figs=False):
    """
    Call all the figure generators for this chapter

    :close_figs: Boolean flag.  If true, will close all figures after generating them; for batch scripting.
                 Default=False
    :force_recalc: If set to False, will skip any figures that are time consuming to generate.
    :return: List of figure handles
    """

    # Find the output directory
    prefix = utils.init_output_dir('chapter9')

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
    Figure 1, Plot of Error Ellipse

    Ported from MATLAB Code

    Nicholas O'Donoughue
    17 May 2021

    :param prefix: output directory to place generated figure
    :return: figure handle
    """    

    # Define Positions
    x0 = np.array([0, 1])
    x2 = np.array([.2, .2])
    
    # Define Covariance Matrix
    sx2 = 5
    sy2 = 3
    rho = .8
    sxy = rho*np.sqrt(sx2*sy2)  # cross-covariance
    cov_mtx = np.array([[sx2, sxy], [sxy, sy2]])
    
    # Compute Elippses
    x_ell1, y_ell1 = utils.errors.draw_error_ellipse(x2, cov_mtx, num_pts=361, conf_interval=1)    # 1 sigma
    x_ell2, y_ell2 = utils.errors.draw_error_ellipse(x2, cov_mtx, num_pts=361, conf_interval=95)   # 95% conf interval
    
    # Draw figure
    fig1 = plt.figure()
    
    # Draw True/Estimate Positions
    plt.scatter(x0[0], x0[1], marker='^', label='True')
    plt.scatter(x2[0], x2[1], marker='+', label='Estimated')
    
    # Draw error ellipses
    plt.plot(x_ell1, y_ell1, linestyle='-', label=r'1$\sigma$ Ellipse')
    plt.plot(x_ell2, y_ell2, linestyle='--', label='95% Ellipse')
    
    # Adjust Figure Display
    # plt.xlim([-1, 1])
    plt.legend(loc='upper left')

    if prefix is not None:
        plt.savefig(prefix + 'fig1.png')
        plt.savefig(prefix + 'fig1.svg')

    return fig1


def make_figure_2(prefix=None):
    """
    Figure 2 Plot of Error Ellipse Example

    Ported from MATLAB Code

    Nicholas O'Donoughue
    17 May 2021

    :param prefix: output directory to place generated figure
    :return: figure handle
    """    

    fig2 = chapter9.example1()

    if prefix is not None:
        plt.savefig(prefix + 'fig2.png')
        plt.savefig(prefix + 'fig2.svg')

    return fig2


def make_figure_3(prefix=None):
    """
    Figure 3, Plot of CEP50

    Ported from MATLAB Code

    Nicholas O'Donoughue
    17 May 2021

    :param prefix: output directory to place generated figure
    :return: figure handle
    """    

    # Initialize Emitter Location and Estimate
    x0 = np.array([0, 0])
    x2 = np.array([.5, -.2])
    
    # Initialize Covariance Matrix
    sx2 = 5
    sy2 = 3
    rho = .8
    sxy = rho*np.sqrt(sx2*sy2)
    cov_mtx = np.array([[sx2, sxy], [sxy, sy2]])
    
    # Compute Error Ellipses
    x_ellipse, y_ellipse = utils.errors.draw_error_ellipse(x2, cov_mtx, num_pts=361, conf_interval=50)
    x_cep, y_cep = utils.errors.draw_cep50(x2, cov_mtx, num_pts=361)
    
    # Draw Figure
    fig3 = plt.figure()
    
    # Draw Ellipses
    plt.plot(x_cep, y_cep, linestyle='-', label=r'$CEP_{50}$')
    plt.plot(x_ellipse, y_ellipse, linestyle='--', label='50% Error Ellipse')
    plt.scatter(x0[0], x0[1], marker='^', label='True')
    plt.scatter(x2[0], x2[1], marker='+', label='Estimated')
    plt.xlim(1.1*np.amax(x_ellipse)*np.array([-1, 1]))
    
    # Adjust Display
    plt.legend(loc='upper left')

    if prefix is not None:
        plt.savefig(prefix + 'fig3.png')
        plt.savefig(prefix + 'fig3.svg')

    return fig3


def make_figure_4(prefix=None):
    """
    Figure 4 Plot of CEP50 and Error Ellipse Example

    Ported from MATLAB Code

    Nicholas O'Donoughue
    17 May 2021

    :param prefix: output directory to place generated figure
    :return: figure handle
    """    

    fig4 = chapter9.example2()

    if prefix is not None:
        plt.savefig(prefix + 'fig4.png')
        plt.savefig(prefix + 'fig4.svg')

    return fig4
