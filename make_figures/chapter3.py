"""
Draw Figures - Chapter 3

This script generates all the figures that appear in Chapter 3 of the textbook.

Ported from MATLAB Code

Nicholas O'Donoughue
23 March 2021
"""

import utils
from utils.unit_conversions import lin_to_db, db_to_lin, kft_to_km
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
import atm
import prop
import detector
from examples import chapter3


def make_all_figures(close_figs=False):
    """
    Call all the figure generators for this chapter

    :close_figs: Boolean flag.  If true, will close all figures after generating them; for batch scripting.
                 Default=False
    :return: List of figure handles
    """

    # Initializes colorSet - Mx3 RGB vector for successive plot lines
    colors = plt.get_cmap("tab10")

    # Reset the random number generator, to ensure reproducibility
    rng = np.random.default_rng(0)

    # Find the output directory
    prefix = utils.init_output_dir('chapter3')

    # Activate seaborn for prettier plots
    sns.set()

    # Generate all figures
    fig1 = make_figure_1(prefix)
    fig2 = make_figure_2(prefix)
    fig3 = make_figure_3(prefix)
    fig4 = make_figure_4(prefix)
    fig7 = make_figure_7(prefix)
    fig8 = make_figure_8(prefix)
    fig9 = make_figure_9(prefix, rng, colors)
    fig10 = make_figure_10(prefix, rng, colors)

    figs = [fig1, fig2, fig3, fig4, fig7, fig8, fig9, fig10]

    if close_figs:
        for fig in figs:
            plt.close(fig)

        return None
    else:
        plt.show()

        return figs


def make_figure_1(prefix=None):
    """
    Figure 1, Spectral Content

    Ported from MATLAB Code

    Nicholas O'Donoughue
    23 March 2021

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    print('Generating Figure 3.1...')

    # Frequency of Signal
    freq0 = 1
    bandwidth = .4

    # Amplitudes
    noise_pwr = 1
    signal_amplitude = 5

    # Generate Frequency Content
    num_freq_bins = 201
    freq_vec = 2*freq0*np.linspace(start=-1, stop=1, num=num_freq_bins)
    noise_vec = noise_pwr*np.ones(shape=(num_freq_bins, ))
    signal_vec = np.fmax(0, signal_amplitude*(1-2*np.absolute(np.absolute(freq_vec)-freq0)/bandwidth))

    # Plot
    fig1 = plt.figure()
    plt.plot(freq_vec, noise_vec, label='Noise')
    plt.plot(freq_vec, signal_vec, label='Signal')

    plt.xlabel('$f$')
    plt.ylabel('$P(f)$')
    plt.legend(loc='upper right')

    # Annotate the bandwidth
    plt.annotate(text='', xy=(-1.2, 5.1), xytext=(-0.8, 5.1), arrowprops=dict(arrowstyle='<->', color='k'))
    plt.annotate(text='', xy=(1.2, 5.1), xytext=(0.8, 5.1), arrowprops=dict(arrowstyle='<->', color='k'))
    plt.text(-1.1, 5.2, r'$B_F$')
    plt.text(.9, 5.2, r'$B_F$')

    # Change the x/y ticks
    plt.xticks([-1, 0, 1], [r'$-f_0$', 0, r'$f_0$'])
    plt.yticks([noise_pwr, signal_amplitude], [r'$N_0/2$', r'$S/2$'])
    plt.xlim([freq_vec[0], freq_vec[-1]])

    # Save figure
    if prefix is not None:
        fig1.savefig(prefix + 'fig1.svg')
        fig1.savefig(prefix + 'fig1.png')

    return fig1


def make_figure_2(prefix=None):
    """
    Figure 2 - Plot with Spectrum

    Ported from MATLAB Code

    Nicholas O'Donoughue
    23 March 2021

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    print('Generating Figure 3.2...')

    # Frequency of Signal
    freq0 = 1
    bandwidth = .4

    # Amplitudes
    noise_pwr = 1
    signal_amplitude = 5

    # Generate Frequency Content
    num_freq_bins = 201
    freq_vec = 2 * freq0 * np.linspace(start=-1, stop=1, num=num_freq_bins)
    noise_vec = noise_pwr * np.ones(shape=(num_freq_bins, ))
    signal_vec = np.fmax(0, signal_amplitude * (1 - 2 * np.absolute(np.absolute(freq_vec) - freq0) / bandwidth))

    # Filtered
    bandwidth_filtered = bandwidth
    filter_mask = np.absolute(np.absolute(freq_vec) - freq0) <= bandwidth_filtered/2
    filter_vec = np.zeros_like(freq_vec)   
    filter_vec[filter_mask] = 1.2*signal_amplitude  # Mark the pass-band slightly higher than the signal amplitude
    noise_filtered = np.copy(noise_vec)     # Copy and filter the noise
    noise_filtered[np.logical_not(filter_mask)] = 0

    # Plot
    fig2 = plt.figure()
    plt.plot(freq_vec, noise_vec, label='Noise')
    plt.plot(freq_vec, noise_filtered, label='Noise (filtered)')
    plt.plot(freq_vec, signal_vec, label='Signal')
    plt.plot(freq_vec, filter_vec, '--', label='Filter')

    plt.xlabel('$f$')
    plt.ylabel('$P(f)$')
    plt.legend(loc='lower right')

    # Annotate the bandwidth
    plt.annotate(text='', xy=(-1.2, 5.1), xytext=(-0.8, 5.1), arrowprops=dict(arrowstyle='<->', color='k'))
    plt.annotate(text='', xy=(1.2, 5.1), xytext=(0.8, 5.1), arrowprops=dict(arrowstyle='<->', color='k'))
    plt.text(-1.1, 5.2, r'$B_F$')
    plt.text(.9, 5.2, r'$B_F$')

    # Change the x/y ticks
    plt.xticks([-1, 0, 1], [r'$-f_0$', 0, r'$f_0$'])
    plt.yticks([noise_pwr, signal_amplitude], [r'$N_0/2$', r'$S/2$'])
    plt.xlim([freq_vec[0], freq_vec[-1]])

    # Save figure
    if prefix is not None:
        fig2.savefig(prefix + 'fig2.svg')
        fig2.savefig(prefix + 'fig2.png')

    return fig2


def make_figure_3(prefix=None):
    """
    Figure 3 - CW Detection PFA vs. Threshold

    Ported from MATLAB Code

    Nicholas O'Donoughue
    23 March 2021

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    print('Generating Figure 3.3...')

    num_samples = np.array([1, 10, 100])
    eta_db = np.arange(start=-10, step=.1, stop=30.1)
    eta_lin = db_to_lin(eta_db)

    # The complementary cdf (1 - CDF) is called the 'survival function'
    prob_fa = stats.chi2.sf(x=np.expand_dims(eta_lin, axis=1), df=2*np.expand_dims(num_samples, axis=0))

    # Plot
    fig3 = plt.figure()
    for idx, this_m in enumerate(num_samples):
        plt.semilogy(eta_db, prob_fa[:, idx], label='M = {}'.format(this_m))

    plt.legend(loc='lower left')
    plt.xlabel(r'$\eta [dB]$')
    plt.ylabel('$P_{FA}$')
    plt.ylim([1e-6, 1.1])
    plt.xlim([eta_db[0], eta_db[-1]])

    # Save figure
    if prefix is not None:
        fig3.savefig(prefix + 'fig3.svg')
        fig3.savefig(prefix + 'fig3.png')

    return fig3


def make_figure_4(prefix=None):
    """
    Figure 4, PD vs. SNR for CW Detection

    Ported from MATLAB Code

    Nicholas O'Donoughue
    23 March 2021

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    print('Generating Figure 3.4...')

    prob_fa = 1e-6
    num_samples = np.array([1, 10, 100, 1000])
    xi_db = np.arange(start=-20, step=.1, stop=20.1)
    xi_lin = db_to_lin(xi_db)

    # Compute threshold
    eta = stats.chi2.ppf(q=1-prob_fa, df=2*num_samples)

    # Compute Probability of Detection
    chi_lambda = 2*xi_lin  # Non-centrality parameter, lambda, or chi-squared RV
    prob_det = 1 - stats.ncx2.cdf(x=eta[np.newaxis, :], df=2*num_samples[np.newaxis, :],
                                  nc=num_samples[np.newaxis, :]*chi_lambda[:, np.newaxis])

    # Plot
    fig4 = plt.figure()
    for idx, this_m in enumerate(num_samples):
        plt.plot(xi_db, prob_det[:, idx], label='M = {}'.format(this_m))

    plt.legend(loc='upper left')
    plt.xlabel(r'$\xi$ [dB]')
    plt.ylabel('$P_D$')

    # Save figure
    if prefix is not None:
        fig4.savefig(prefix + 'fig4.svg')
        fig4.savefig(prefix + 'fig4.png')

    return fig4


def make_figure_7(prefix=None):
    """
    Figure 7, Atmospheric Loss Table

    Ported from MATLAB Code

    Nicholas O'Donoughue
    23 March 2021

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    print('Generating Figure 3.7...')

    range_m = 1.0e3   # set ref distance to 1 km
    freq_vec = np.arange(start=1.e9, step=50.e6, stop=100.e9+50.e6)

    # Reference Atmosphere
    #  -- Sea Level, 10 kft, 20 kft, 30 kft, 40 kft
    alt_kft = np.array([0., 10., 20., 30., 40.])
    # T = [15, -4.8, -24.6, -44.4, -56.6];
    # P = [101325, 69680, 46560, 30090,18750];
    # g = [7.5,2.01,0.34,.05,.01];

    loss_atm = np.zeros(shape=(np.size(freq_vec), np.size(alt_kft)))

    for idx_alt, this_alt in enumerate(alt_kft):
        # Create atmosphere for this altitude band
        this_alt_m = kft_to_km(this_alt) * 1.0e3
        atmosphere = atm.reference.get_standard_atmosphere(this_alt_m)

        loss_atm[:, idx_alt] = atm.model.calc_atm_loss(freq_vec, gas_path_len_m=range_m, atmosphere=atmosphere)

    # Generate plot
    fig7 = plt.figure()
    for idx_alt, this_alt in enumerate(alt_kft):
        plt.semilogy(freq_vec/1e9, loss_atm[:, idx_alt], label='Alt = {} kft'.format(this_alt))

    plt.legend(loc='upper left')
    plt.xlabel('Frequency [GHz]')
    plt.ylabel('Specific Attenuation [dB/km]')

    # Save figure
    if prefix is not None:
        fig7.savefig(prefix + 'fig7.svg')
        fig7.savefig(prefix + 'fig7.png')

    return fig7


def make_figure_8(prefix=None):
    """
    Figures 8, FM Reception Power vs. Range

    Ported from MATLAB Code

    Nicholas O'Donoughue
    23 March 2021

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    print('Generating Figure 3.8...')

    # Set up RF environment
    ht = 100
    hr = 2
    range_vec = np.arange(start=10.0e3, step=10.0e3, stop=510.0e3)
    f0 = 100e6

    # Compute Losses and Fresnel Zone
    # loss_free_space = prop.model.get_free_space_path_loss(R=range_vec, f0=f0, include_atm_loss=False)
    # loss_two_ray = prop.model.get_two_ray_path_loss(R=range_vec, f0=f0, ht=ht, hr=hr, includeAtmLoss=False)
    loss_prop = prop.model.get_path_loss(range_m=range_vec, freq_hz=f0, tx_ht_m=ht, rx_ht_m=hr, include_atm_loss=False)

    # Noise Power
    bandwidth = 2e6  # channel bandwidth [Hz]
    noise_figure = 5  # noise figure [dB]
    noise_pwr = lin_to_db(utils.constants.kT*bandwidth)+noise_figure

    # Signal Power
    eirp = 47    # dBW
    rx_gain = 0  # Receive antenna gain
    rx_loss = 0

    # Received Power and SNR
    signal_pwr = eirp-loss_prop+rx_gain-rx_loss
    snr_min = 3.65
    signal_pwr_min = noise_pwr+snr_min

    snr0 = eirp+rx_gain-rx_loss-noise_pwr  # snr with no propagation loss
    range_max = detector.squareLaw.max_range(prob_fa=1e-6, prob_d=.5, num_samples=10, f0=f0, ht=ht, hr=hr,
                                             snr0=snr0, include_atm_loss=False)
    print('Max Range: {:.3f} km'.format(range_max[0]/1e3))

    fig8 = plt.figure()
    plt.plot(range_vec/1e3, signal_pwr, label='$P_R$')
    plt.plot(range_vec/1e3, signal_pwr_min*np.ones_like(range_vec), linestyle=':', label='MDS')
    plt.legend(loc='upper right')
    plt.xlabel('Range [km]')
    plt.ylabel('Received Power [dBW]')

    # Save figure
    if prefix is not None:
        fig8.savefig(prefix + 'fig8.svg')
        fig8.savefig(prefix + 'fig8.png')

    return fig8


def make_figure_9(prefix=None, rng=None, colors=None):
    """
    Figures 9, Example 3.1 Monte Carlo Results

    Ported from MATLAB Code

    Nicholas O'Donoughue
    23 March 2021

    :param prefix: output directory to place generated figure
    :param rng: random number generator
    :param colors: set of colors for plotting
    :return: figure handle
    """

    print('Generating Figure 3.9 (using Example 3.1)...')

    fig9 = chapter3.example1(rng, colors)

    # Save figure
    if prefix is not None:
        fig9.savefig(prefix + 'fig9.svg')
        fig9.savefig(prefix + 'fig9.png')

    return fig9


def make_figure_10(prefix=None, rng=None, colors=None):
    """
    Figures 10, Example 3.2 Monte Carlo results

    Ported from MATLAB Code

    Nicholas O'Donoughue
    23 March 2021

    :param prefix: output directory to place generated figure
    :param rng: random number generator
    :param colors: colormap for plotting
    :return: figure handle
    """

    print('Generating Figure 3.10 (using Example 3.2)...')

    fig10 = chapter3.example2(rng, colors)

    # Save figure
    if prefix is not None:
        fig10.savefig(prefix + 'fig10.svg')
        fig10.savefig(prefix + 'fig10.png')

    return fig10
