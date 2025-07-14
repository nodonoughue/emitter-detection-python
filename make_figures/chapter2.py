"""
Draw Figures - Chapter 2

This script generates all the figures that appear in
Chapter 2 of the textbook.

Ported from MATLAB Code

Nicholas O'Donoughue
22 March 2021
"""

import utils
from utils.unit_conversions import lin_to_db, db_to_lin
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf, erfinv
from scipy.linalg import toeplitz
from examples import chapter2


def make_all_figures(close_figs=False):
    """
    Call all the figure generators for this chapter

    :param close_figs: Boolean flag.  If true, will close all figures after generating them; for batch scripting.
                       Default=False
    :return: List of figure handles
    """

    # Initializes colorSet - Mx3 RGB vector for successive plot lines
    colors = plt.get_cmap("tab10")

    # Reset the random number generator, to ensure reproducibility
    rng = np.random.default_rng()

    # Find the output directory
    prefix = utils.init_output_dir('chapter2')
    utils.init_plot_style()

    # Generate all figures
    fig1 = make_figure_1(prefix)
    fig2 = make_figure_2(prefix, colors)
    fig3 = make_figure_3(prefix)
    fig4 = make_figure_4(prefix)
    fig5 = make_figure_5(prefix, rng)

    figs = [fig1, fig2, fig3, fig4, fig5]

    if close_figs:
        for fig in figs:
            plt.close(fig)

        return None
    else:
        plt.show()

        return figs


def make_figure_1(prefix=None):
    """
    Figure 1, Likelihood Function

    Ported from MATLAB Code

    Nicholas O'Donoughue
    22 March 2021

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    print('Generating Figure 2.1...')

    # Number of x-axis points
    num_points = 512
    
    # Centroids
    n1 = .3
    n2 = .45
    n3 = .6
    s2 = .02  # variance of .5
    
    # Generate PDFs
    x = np.linspace(start=0, stop=1, num=num_points)
    
    # Use an exponential (unscaled) and a random factor to make them look a little less clean
    f1 = np.exp(-(x-n1)**2/(2*s2))  # + .05*rng.standard_normal(size=(1, N))
    f2 = np.exp(-(x-n2)**2/(2*s2))  # + .05*rng.standard_normal(size=(1, N))
    f3 = np.exp(-(x-n3)**2/(2*s2))  # + .05*rng.standard_normal(size=(1, N))
    
    # Scale PDFs
    f1 = f1/np.linalg.norm(f1)
    f2 = f2/np.linalg.norm(f2)
    f3 = f3/np.linalg.norm(f3)
    
    # Plot
    fig1 = plt.figure()
    plt.plot(x, f1, label=r'$\theta=\theta_1$')
    plt.plot(x, f2, label=r'$\theta=\theta_2$')
    plt.plot(x, f3, label=r'$\theta=\theta_3$')
    
    plt.xlabel('$z$')
    plt.ylabel(r'$f_\theta(z)$')

    # Add legend
    plt.legend(loc='upper right')
    
    # Display the plot
    plt.draw()

    # Output to file
    if prefix is not None:
        fig1.savefig(prefix + 'fig1.svg')
        fig1.savefig(prefix + 'fig1.png')

    return fig1


def make_figure_2(prefix=None, cmap=None):
    """
    Figure 2 - Likelihood Ratio Test

    Ported from MATLAB Code

    Nicholas O'Donoughue
    22 March 2021

    :param prefix: output directory to place generated figure
    :param cmap: colormap
    :return: figure handle
    """

    print('Generating Figure 2.2...')

    # Get default colormap
    if cmap is None:
        cmap = plt.get_cmap('tab10')

    # Set up axes
    ell = np.arange(start=0, step=.01, stop=10.01)
    
    # Set variance to 2 under both hypotheses, mean to 4 and 6 under H0 and H1, respectively
    s2 = 2
    m0 = 2
    m1 = 6
    
    # PDF of ell(z) under H0 and H1, respectively
    f0 = (2*np.pi*s2)**(-.5)*np.exp(-.5*(ell-m0)**2/s2)
    f1 = (2*np.pi*s2)**(-.5)*np.exp(-.5*(ell-m1)**2/s2)
    
    # Detection threshold
    eta = 4.5
    mask = ell >= eta  # Region above the threshold
    fa = f0[mask]
    missed_det = f1[np.logical_not(mask)]

    # Plot the likelihood functions
    fig2 = plt.figure()
    
    plt.plot(ell, f0, label='H_0', color=cmap(3))
    plt.plot(ell, f1, label='H_1', color=cmap(2))
    
    # Add the threshold
    plt.plot(eta*np.array([1, 1]), np.array([0, .4]), linestyle='-.', linewidth=.5, label=r'$\eta$', color='k')
    # text(eta+.1,.38,'$\eta$');
    
    # Add the PFA/PD regions
    plt.fill_between(ell[np.logical_not(mask)], missed_det, label='Missed Detection', facecolor=cmap(2), alpha=.6)
    plt.fill_between(ell[mask], fa, label='False Alarm', facecolor=cmap(3), alpha=.6)
    
    # Add text overlay
    plt.annotate(text='', xy=(eta-1.5, 0.325), xytext=(eta+1.5, 0.325), 
                 arrowprops=dict(arrowstyle='<->', color='k', lw=1))
    plt.text(eta+.5, .35, 'Reduce $P_{FA}$')
    plt.text(eta-2.1, .35, 'Reduce $P_{MD}$')
    
    # Axis Labels
    plt.ylabel(r'$f_\theta(\ell)$')
    plt.xlabel(r'$\ell(z)$')
    
    # Turn on the legend
    plt.legend(loc='lower right')

    # Draw the figure
    plt.draw()

    # Save figure
    if prefix is not None:
        fig2.savefig(prefix + 'fig2.svg')
        fig2.savefig(prefix + 'fig2.png')

    return fig2


def make_figure_3(prefix=None):
    """
    Run example 2.2, and store the figure

    Ported from MATLAB Code

    Nicholas O'Donoughue
    23 March 2021

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    print('Generating Figure 2.3 (using Example 2.2)...')

    fig3 = chapter2.example2()

    # Draw the figure
    plt.draw()

    # Save figure
    if prefix is not None:
        fig3.savefig(prefix + 'fig3.svg')
        fig3.savefig(prefix + 'fig3.png')

    return fig3


def make_figure_4(prefix=None):
    """
    Figure 4, PD S-Curves

    Ported from MATLAB Code

    Nicholas O'Donoughue
    23 March 2021

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    print('Generating Figure 2.4...')

    # Set up PFA and SNR vectors
    prob_fa = np.expand_dims(np.array([1e-6, 1e-4, 1e-2]), axis=0)
    xi = np.expand_dims(np.arange(start=0, step=0.1, stop=10.1), axis=1)  # dB Scale
    xi_lin = db_to_lin(xi)  # Convert SNR to linear

    # Compute the PD according to the simplified erf equation and its inverse
    prob_det = .5*(1-erf(erfinv(1-2*prob_fa)-xi_lin/np.sqrt(2)))

    # Plot the ROC curve
    fig4 = plt.figure()
    for idx, this_pfa in enumerate(prob_fa[0, :]):
        plt.plot(xi, prob_det[:, idx], label='$P_{{FA}} = 10^{{{:.0f}}}$'.format(np.log10(this_pfa)))

    # Axes Labels
    plt.ylabel('$P_D$')
    plt.xlabel('SNR [dB]')

    # Legend
    plt.legend(loc='upper left')

    # Draw the figure
    plt.draw()

    # Save figure
    if prefix is not None:
        fig4.savefig(prefix + 'fig4.svg')
        fig4.savefig(prefix + 'fig4.png')

    return fig4


def make_figure_5(prefix=None, rng=np.random.default_rng()):
    """
    Figure 5, CFAR

    Ported from MATLAB Code

    Nicholas O'Donoughue
    23 March 2021

    :param prefix: output directory to place generated figure
    :param rng: random number generator
    :return: figure handle
    """

    print('Generating Figure 2.5...')

    # Noise Level
    s2 = 1
    s2high = 10
    
    # Generate Noise
    num_samples = 1024                 # number of samples
    idx_high_start = int(np.fix(num_samples/2))    # start of high sample
    idx_high_end = int(np.fix(3*num_samples/4))  # end of high sample
    
    noise_vec = np.sqrt(s2/2)*(rng.standard_normal(size=(num_samples, 1))+1j*rng.standard_normal(size=(num_samples, 1)))
    noise_vec[idx_high_start:idx_high_end] \
        = np.sqrt(s2high/2)*(rng.standard_normal(size=(idx_high_end-idx_high_start, 1))
                             + 1j*rng.standard_normal(size=(idx_high_end-idx_high_start, 1)))
    
    # Variance Estimate
    num_window = 64  # window size - one-sided
    num_guard = 4  # Number of guard cells - one-sided
    mask = np.concatenate((np.ones(shape=(num_window, 1)), np.zeros(shape=(2*num_guard+1, 1)),
                           np.ones(shape=(num_window, 1))), axis=0)  # Initial CA-CFAR mask
    mask = mask/(2*num_window)
    s2est = np.convolve(np.squeeze(np.absolute(noise_vec)**2), np.squeeze(mask), 'same')
    
    # Threshold Equation --- need to replace
    eta = np.sum(toeplitz(s2est[0:100], s2est), axis=0)/10

    # Plot
    fig5 = plt.figure()
    x_vec = np.arange(num_samples)
    plt.plot(x_vec, 2*lin_to_db(np.absolute(noise_vec)), label='Noise Level')
    plt.plot(x_vec, lin_to_db(eta), label='Threshold')
    
    plt.xlabel('Time/Frequency/Range')
    plt.ylabel('Power [dB]')
    
    # Legend
    plt.legend(loc='upper left')

    # Draw the figure
    plt.draw()

    # Save figure
    if prefix is not None:
        fig5.savefig(prefix + 'fig5.svg')
        fig5.savefig(prefix + 'fig5.png')

    return fig5


if __name__ == "__main__":
    make_all_figures(close_figs=False)
