"""
Draw Figures - Chapter 4

This script generates all the figures that appear in Chapter 4 of the textbook.

Ported from MATLAB Code

Nicholas O'Donoughue
24 March 2021
"""

from .. import utils
from ..utils.unit_conversions import lin_to_db, db_to_lin
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from scipy.fft import fft, fftshift
from scipy import stats
import seaborn as sns
from .. import detector
from ..examples import chapter4

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
    prefix = utils.init_output_dir('chapter4')

    # Activate seaborn for prettier plots
    sns.set()

    # Generate all figures
    fig1a = make_figure_1a(prefix)
    fig1b = make_figure_1b(prefix, rng)
    fig2a = make_figure_2a(prefix, rng)
    fig2b = make_figure_2b(prefix, rng)
    fig3 = make_figure_3(prefix)
    fig5 = make_figure_5(prefix, colors)
    fig6 = make_figure_6(prefix, rng, colors)
    fig7 = make_figure_7(prefix)
    fig8 = make_figure_8(prefix, colors)

    figs = [fig1a, fig1b, fig2a, fig2b, fig3, fig5, fig6, fig7, fig8]

    if close_figs:
        for fig in figs:
            plt.close(fig)

        return None
    else:
        plt.show()

        return figs


def make_figure_1a(prefix=None):
    """
    Figure 1a - Alternating Sine Waves

    Ported from MATLAB Code

    Nicholas O'Donoughue
    24 March 2021

    :param prefix: output directory to place generated figure
    :return: figure handle
    """
    
    # Sine wave
    num_points = 1024  # Sample points
    y_chip = np.exp(1j*(np.pi/2+2*np.pi*np.arange(num_points)/num_points))

    # String together multiple periods
    code = np.array([0, 1, 1, 0, 1])
    symbol = np.exp(1j*np.pi*code)
    y_full = np.ravel(np.expand_dims(y_chip, axis=0)*np.expand_dims(symbol, axis=1))

    # x axis
    t_vec = np.arange(np.size(y_full))

    fig1a = plt.figure()
    plt.plot(t_vec, np.real(y_full), color='k', linewidth=0.5)
    plt.plot(t_vec, np.zeros_like(t_vec), color='k', linewidth=0.5)

    for idx, bit in enumerate(code):
        plt.text(num_points/2 + num_points*idx-1, 1.5, '{}'.format(bit))
        plt.plot(num_points*idx*np.array([1, 1]), np.array([-1, 2]), color='k', linestyle=':')

    # Annotation
    plt.annotate(text='', xy=(2*num_points, 1.1), xytext=(3*num_points, 1.1), arrowprops=dict(arrowstyle='<->'))
    plt.text(2.35*num_points, 1.25, r'$T_{chip}$')

    # Turn off the axes
    ax = plt.gca()
    ax.axis('off')

    # Save figure
    if prefix is not None:
        fig1a.savefig(prefix + 'fig1a.svg')
        fig1a.savefig(prefix + 'fig1a.png')

    return fig1a


def make_figure_1b(prefix=None, rng=None):
    """
    Figure 1b - Figure 1b, Bandwidth

    Ported from MATLAB Code

    Nicholas O'Donoughue
    24 March 2021

    :param prefix: output directory to place generated figure
    :param rng: random number generator
    :return: figure handle
    """

    if rng is None:
        rng = np.random.default_rng()

    num_samples = 16  # Number of samples per cycle
    num_code_bits = 128  # Length of transmit code in bits
    y_chip = np.exp(1j*(np.pi/2+2*np.pi*np.arange(num_samples)/num_samples))
    
    num_code_samples = num_code_bits*num_samples
    lo = np.exp(1j*2*np.pi*np.arange(num_code_samples)*4/num_samples)  # 4 samples/cycle

    num_monte_carlo = 100
    spectral_average = np.zeros_like(lo)
    for ii in range(num_monte_carlo):
        # Generate a random code
        code = rng.integers(low=0, high=2, size=(num_code_bits, 1))  # with random integers, the interval is [low, high)
        symbol = np.exp(1j*np.pi*code)

        # Random starting phase
        starting_phase = np.exp(1j*rng.uniform(low=0, high=2*np.pi))

        # Generate full transmit signal at the intermediate frequency (IF) of y_chip
        signal_if = np.ravel(starting_phase*symbol*y_chip)

        # Mix with the local oscillator (lo) to get the radio frequency (RF) sample
        signal_rf = signal_if*lo
    
        # Take the fourier transform
        spectral_average += np.absolute(fft((np.real(signal_rf))))

    # Normalize, and use an fftshift to put the center frequency in the middle of the vector
    spectral_average = fftshift(spectral_average)/np.max(np.absolute(spectral_average))
    
    fig1b = plt.figure()
    plt.plot(np.linspace(start=-1, stop=1, num=num_code_samples),
             2*lin_to_db(np.absolute(spectral_average)))

    # Plot top and -3 dB lines
    plt.plot([-1, 1], [0, 0], color='k', linestyle=':')
    plt.plot([-1, 1], [-3, -3], color='k', linestyle=':')
    plt.plot([0, 0], [-20, 0], color='k', linestyle='-')
    plt.ylim([-40, 3])
    
    # Create textbox
    plt.text(-.4, -1.5, '3 dB')
    plt.text(.3, 2, r'$f_0$')
    
    # Create double arrows
    f0 = 0.625  # (normalized freq is 1/16 (y_chip) + 4/16 (lo))
    bw = 0.125
    plt.annotate(text='', xy=(0, 1), xytext=(0.64, 1), arrowprops=dict(arrowstyle='<->', color='k'))
    plt.annotate(text='', xy=(-.5, 0), xytext=(-.5, -3), arrowprops=dict(arrowstyle='<->', color='k'))
    plt.annotate(text='', xy=(f0-bw/2, -3), xytext=(f0+bw/2, -3), arrowprops=dict(arrowstyle='<->', color='k'))
    plt.annotate(text=r'$B_s=1/T_{\mathrm{chip}}$', xy=(f0, -3), xytext=(.1, -6), arrowprops=dict(arrowstyle='-',
        color='k'))
    # Turn off the axes
    ax = plt.gca()
    ax.axis('off')

    # Save figure
    if prefix is not None:
        fig1b.savefig(prefix + 'fig1b.svg')
        fig1b.savefig(prefix + 'fig1b.png')

    return fig1b


def make_figure_2a(prefix=None, rng=None):
    """
    Figure 2a - Chip Rate

    Ported from MATLAB Code

    Nicholas O'Donoughue
    24 March 2021

    :param prefix: output directory to place generated figure
    :param rng: random number generator
    :return: figure handle
    """

    if rng is None:
        rng = np.random.default_rng()

    # Generate the digital signals
    rate_data = 4   # bits/sec
    rate_chip = 16  # bits/sec
    num_code_bits = int(np.fix(rate_chip/rate_data))
    num_data_bits = 4
    num_full_bits = num_code_bits*num_data_bits

    data_bits = rng.integers(low=0, high=2, size=(1, num_data_bits))
    code_bits = rng.integers(low=0, high=2, size=(num_code_bits, 1))

    code_bits_full, data_bits_full = np.meshgrid(code_bits, data_bits)
    out_bits_full = np.logical_xor(data_bits_full, code_bits_full)
    out_bits_full = out_bits_full.astype(int)

    # Convert from bits to symbols
    data_symbols = np.reshape(np.exp(1j*np.pi*data_bits_full), newshape=(num_full_bits, 1))
    code_symbols = np.reshape(np.exp(1j*np.pi*code_bits_full), newshape=(num_full_bits, 1))
    out_symbols = np.reshape(np.exp(1j*np.pi*out_bits_full), newshape=(num_full_bits, 1))

    # Generate the signals
    osf = 16  # Samples per cycle
    y = np.expand_dims(np.exp(1j*(np.pi/2+2*np.pi*np.arange(osf)/osf)), axis=0)

    # Construct the code signals
    y_data = np.ravel(y*data_symbols)
    y_code = np.ravel(y*code_symbols)
    y_dsss = np.ravel(y*out_symbols)

    fig2a = plt.figure()
    # Start with the Signals at the origin
    plt.plot(np.arange(num_full_bits*osf), np.real(y_data)+6, label='Data Signal')
    plt.plot(np.arange(num_code_bits*osf), np.real(y_code[0:num_code_bits*osf])+3, label='Spreading Code')
    plt.plot(np.arange(num_full_bits*osf), np.real(y_dsss), label='Encoded Signal')

    # Add the code and vertical lines
    for idx, bit in enumerate(np.ravel(out_bits_full)):
        plt.text(osf*idx+osf/2, 1.5, '{}'.format(bit))
        plt.plot(osf*idx*np.array([1, 1]), [-1, 2], color='w', linestyle='-', linewidth=.5, label=None)

    for idx, bit in enumerate(np.ravel(code_bits)):
        plt.text(osf*idx+osf/2, 4.5, '{}'.format(bit))
        plt.plot(osf*idx*np.array([1, 1]), [2, 5], color='w', linestyle='-', linewidth=.5, label=None)

    for idx, bit in enumerate(np.ravel(data_bits)):
        plt.text(osf*num_code_bits*idx+osf*num_code_bits/2, 7.5, '{}'.format(bit))
        plt.plot(osf*num_code_bits*idx*np.array([1, 1]), [2, 8], color='w', linestyle='-', linewidth=.5, label=None)

    plt.grid('off')
    ax = plt.gca()
    ax.axis('off')

    plt.legend(loc='right')

    # Save figure
    if prefix is not None:
        fig2a.savefig(prefix + 'fig2a.svg')
        fig2a.savefig(prefix + 'fig2a.png')

    return fig2a


def make_figure_2b(prefix=None, rng=None):
    """
    Figure 2b - Spectrum

    Ported from MATLAB Code

    Nicholas O'Donoughue
    24 March 2021

    :param prefix: output directory to place generated figure
    :param rng: random number generator
    :return: figure handle
    """

    if rng is None:
        rng = np.random.default_rng(0)

    num_monte_carlo = 1000

    num_bits_code1 = 16
    num_bits_code2 = 64
    samples_per_bit = 16
    num_samples_full = samples_per_bit*max(num_bits_code1, num_bits_code2)  # at least 16 samples per period
    chip_len_code1 = int(np.fix(num_samples_full/num_bits_code1))
    chip_len_code2 = int(np.fix(num_samples_full/num_bits_code2))

    # Generate the chips
    chip1 = np.exp(1j*(np.pi/2+2*np.pi*np.arange(chip_len_code1)/chip_len_code1))
    chip2 = np.exp(1j*(np.pi/2+2*np.pi*np.arange(chip_len_code2)/chip_len_code2))

    spectral_average1 = np.zeros(shape=(num_samples_full, ))
    spectral_average2 = np.zeros(shape=(num_samples_full, ))

    for jj in range(num_monte_carlo):
        # Generate new codes
        code1 = rng.integers(low=0, high=2, size=(num_bits_code1, ))
        code2 = rng.integers(low=0, high=2, size=(num_bits_code2, ))

        starting_phase1 = np.exp(1j*rng.uniform(low=0, high=2*np.pi))
        starting_phase2 = np.exp(1j*rng.uniform(low=0, high=2*np.pi))

        # Construct the full signals
        signal1 = np.zeros_like(spectral_average1, dtype=complex)
        for idx, bit in enumerate(code1):
            signal1[np.arange(chip_len_code1)+chip_len_code1*idx] = chip1*np.exp(1j*np.pi*bit)*starting_phase1

        signal2 = np.zeros_like(spectral_average2, dtype=complex)
        for idx, bit in enumerate(code2):
            signal2[np.arange(chip_len_code2)+chip_len_code2*idx] = chip2*np.exp(1j*np.pi*bit)*starting_phase2

        # Take the fourier transform
        spectral_average1 += np.absolute(fft(signal1))
        spectral_average2 += np.absolute(fft(signal2))

    # Normalize the spectra and use an fftshift to move the center frequency to the middle
    spectral_average1 = fftshift(spectral_average1)/np.amax(spectral_average1, axis=None)
    spectral_average2 = fftshift(spectral_average2)/np.amax(spectral_average2, axis=None)

    # Shift to account for central frequency of chip waveform
    spectral_average1 = np.roll(spectral_average1, -num_bits_code1)
    spectral_average2 = np.roll(spectral_average2, -num_bits_code2)

    # Plot
    fig2b = plt.figure()
    t_vec = np.linspace(start=0, stop=1, num=num_samples_full)
    plt.plot(t_vec, lin_to_db(spectral_average1), label='Data Signal')
    plt.plot(t_vec, lin_to_db(spectral_average2), label='Encoded Signal')
    plt.ylim([-20, 2])
    plt.xlim([.25, .75])

    plt.legend(loc='upper right')

    # Plot top and -3 dB lines
    plt.plot([0, 1], [-3, -3], color='k', linestyle=':', label=None)
    plt.text(.3, -2.5, '-3 dB')
    plt.text(.55, -2.25, r'$B_s = 1/T_{chip}$')
    plt.text(.55, -4.5, r'$B_d = 1/T_{sym}$')

    # Annotation
    plt.annotate(text='', xy=(.5-.7/num_bits_code1, -2.5), xytext=(.5+.7/num_bits_code1, -2.5),
                 arrowprops=dict(arrowstyle='<->', color='k'))
    plt.annotate(text='', xy=(.5-.7/num_bits_code2, -3.5), xytext=(.5+.7/num_bits_code2, -3.5),
                 arrowprops=dict(arrowstyle='<->', color='k'))

    # Save figure
    if prefix is not None:
        fig2b.savefig(prefix + 'fig2b.svg')
        fig2b.savefig(prefix + 'fig2b.png')

    return fig2b


def make_figure_3(prefix=None):
    """
    Figure 3, Spreading of SNR

    Ported from MATLAB Code

    Nicholas O'Donoughue
    24 March 2021

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    snr_ref_db = 0
    bw_ref = 1e6

    bw = np.logspace(start=3, stop=9, num=10)

    loss_spreading = lin_to_db(bw / bw_ref)

    snr_o_db = snr_ref_db - loss_spreading

    fig3 = plt.figure()
    plt.semilogx(bw, snr_o_db)
    plt.xlabel(r'$B_s$ [Hz]')
    plt.ylabel(r'$\xi$ [dB]')

    # Save figure
    if prefix is not None:
        fig3.savefig(prefix + 'fig3.svg')
        fig3.savefig(prefix + 'fig3.png')

    return fig3


def make_figure_5(prefix=None, colors=None):
    """
    Figures 5, Cross-Correlator SNR

    Ported from MATLAB Code

    Nicholas O'Donoughue
    24 March 2021

    :param prefix: output directory to place generated figure
    :param colors: colormap for plotting
    :return: figure handle
    """

    if colors is None:
        colors = plt.get_cmap('tab10')

    # SNR axis
    snr_i_db = np.arange(start=-30, stop=30.1, step=.1)
    snr_i_lin = np.expand_dims(db_to_lin(snr_i_db), axis=1)

    pulse_duration = 1e-6
    bandwidth = np.logspace(start=6, stop=9, num=4)
    tbwp_lin = pulse_duration*bandwidth
    tbwp_db = lin_to_db(tbwp_lin)
    snr_o_lin = np.expand_dims(tbwp_lin, axis=0)*snr_i_lin**2/(1+2*snr_i_lin)
    snr_o_db = lin_to_db(snr_o_lin)

    snr_o_ideal = np.expand_dims(snr_i_db, axis=1) + np.expand_dims(tbwp_db, axis=0)

    fig5 = plt.figure()
    for idx, tbwp in enumerate(tbwp_lin):
        plt.plot(snr_i_db, snr_o_db[:, idx], color=colors(idx), label='TB={:.0f}'.format(tbwp))
        plt.plot(snr_i_db, snr_o_ideal[:, idx], color=colors(idx), linestyle='-.', label=None)

    plt.legend(loc='upper left')
    plt.xlabel(r'$\xi_i$ [dB]')
    plt.ylabel(r'$\xi_o$ [dB]')

    # Save figure
    if prefix is not None:
        fig5.savefig(prefix + 'fig5.svg')
        fig5.savefig(prefix + 'fig5.png')

    return fig5


def make_figure_6(prefix=None, rng=None, colors=None):
    """
    Figures 6, Comparison of Performance

    Ported from MATLAB Code

    Nicholas O'Donoughue
    24 March 2021

    :param prefix: output directory to place generated figure
    :param rng: random number generator
    :param colors: colormap for plotting
    :return: figure handle
    """

    # Vary Time-Bandwidth Product
    tbwp_vec_db = np.arange(start=10., stop=31., step=10., dtype=int)
    tbwp_vec_lin = np.expand_dims(db_to_lin(tbwp_vec_db), axis=0).astype(int)

    input_snr_vec_db = np.arange(start=-20, stop=10.1, step=0.1)
    input_snr_vec_lin = np.expand_dims(db_to_lin(input_snr_vec_db), axis=1)

    output_snr_vec_lin = tbwp_vec_lin*input_snr_vec_lin**2/(1+2*input_snr_vec_lin)
    # output_snr_vec_db = lin_to_db(output_snr_vec_lin)

    # Energy Detector Performance
    prob_fa = 1e-6

    threshold_ed = stats.chi2.ppf(q=1-prob_fa, df=2*tbwp_vec_lin)
    prob_det_ed = stats.ncx2.sf(x=threshold_ed, df=2*tbwp_vec_lin, nc=2*tbwp_vec_lin*input_snr_vec_lin)

    # Cross-Correlator Performance
    threshold_xc = stats.chi2.ppf(q=1-prob_fa, df=2)
    prob_det_xc = stats.ncx2.sf(x=threshold_xc/(1+2*input_snr_vec_lin), df=2, nc=2*output_snr_vec_lin)
    
    # Monte Carlo Trials
    input_snr_vec_coarse_db = input_snr_vec_db[::10]
    input_snr_vec_coarse_lin = db_to_lin(input_snr_vec_coarse_db)

    num_monte_carlo = int(1e4)
    num_tbwp = int(tbwp_vec_lin.size)
    num_snr = int(input_snr_vec_coarse_lin.size)

    # Generate noise vectors
    noise_pwr = 1  # Unit Variance

    prob_det_ed_mc = np.zeros(shape=(num_snr, num_tbwp))
    prob_det_xc_mc = np.zeros(shape=(num_snr, num_tbwp))
    
    for idx_tbwp, tbwp in enumerate(np.ravel(tbwp_vec_lin)):
        # Generate the noise vectors
        noise1 = np.sqrt(noise_pwr/2)*(rng.standard_normal(size=(tbwp, num_monte_carlo))
                                       + 1j*rng.standard_normal(size=(tbwp, num_monte_carlo)))
        noise2 = np.sqrt(noise_pwr/2)*(rng.standard_normal(size=(tbwp, num_monte_carlo))
                                       + 1j*rng.standard_normal(size=(tbwp, num_monte_carlo)))

        # Generate a signal vector
        signal = np.sqrt(1/2)*(rng.standard_normal(size=(tbwp, num_monte_carlo))
                               + 1j*rng.standard_normal(size=(tbwp, num_monte_carlo)))
        phase_difference = np.exp(1j*rng.uniform(low=0, high=2*np.pi, size=(1, num_monte_carlo)))

        for idx_snr, snr in enumerate(input_snr_vec_coarse_lin):
            # Scale the signal power to match SNR
            this_signal = signal * np.sqrt(snr)

            y1 = this_signal+noise1
            y2 = this_signal*phase_difference+noise2

            det_result_ed = detector.squareLaw.det_test(z=y1, noise_var=noise_pwr/2, prob_fa=prob_fa)
            prob_det_ed_mc[idx_snr, idx_tbwp] = np.sum(det_result_ed, axis=None)/num_monte_carlo

            det_result_xc = detector.xcorr.det_test(y1=y1, y2=y2, noise_var=noise_pwr, num_samples=tbwp,
                                                    prob_fa=prob_fa)
            prob_det_xc_mc[idx_snr, idx_tbwp] = np.sum(det_result_xc, axis=None)/num_monte_carlo

    fig6 = plt.figure()
    for idx, tbwp in enumerate(tbwp_vec_lin[0, :]):
        if idx == 0:
            ed_label = 'ED'
            xc_label = 'XC'
            ed_mc_label = 'ED (Monte Carlo)'
            xc_mc_label = 'XC (Monte Carlo)'
        else:
            ed_label = None
            xc_label = None
            ed_mc_label = None
            xc_mc_label = None

        plt.plot(input_snr_vec_db, prob_det_ed[:, idx], color=colors(idx), linestyle='-', label=ed_label)
        plt.plot(input_snr_vec_db, prob_det_xc[:, idx], color=colors(idx), linestyle='--', label=xc_label)
        plt.scatter(input_snr_vec_coarse_db, prob_det_ed_mc[:, idx], color=colors(idx), marker='^', label=ed_mc_label)
        plt.scatter(input_snr_vec_coarse_db, prob_det_xc_mc[:, idx], color=colors(idx), marker='x', label=xc_mc_label)

    plt.legend(loc='lower right')

    # Create ellipses
    ax = plt.gca()
    ell = Ellipse(xy=(2, .4), width=5, height=.05)
    ell.set(fill=False, edgecolor=colors(0))
    ax.add_artist(ell)
    plt.annotate(text='TB=10', xy=(-.5, .4), xytext=(-16, .3), arrowprops=dict(arrowstyle='-', color=colors(0)))

    ell = Ellipse(xy=(-3.5, .5), width=3, height=.05)
    ell.set(fill=False, edgecolor=colors(1))
    ax.add_artist(ell)
    plt.annotate(text='TB=100', xy=(-5, .5), xytext=(-16, .5), arrowprops=dict(arrowstyle='-', color=colors(1)))

    ell = Ellipse(xy=(-8.5, .6), width=3, height=.05)
    ell.set(fill=False, edgecolor=colors(2))
    ax.add_artist(ell)
    plt.annotate(text='TB=1,000', xy=(-10, .6), xytext=(-16, .7), arrowprops=dict(arrowstyle='-', color=colors(2)))

    # Save figure
    if prefix is not None:
        fig6.savefig(prefix + 'fig6.svg')
        fig6.savefig(prefix + 'fig6.png')

    return fig6


def make_figure_7(prefix=None):
    """
    Figure 7, Example 4.1 Monte Carlo results

    Ported from MATLAB Code

    Nicholas O'Donoughue
    24 March 2021

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    
    fig7 = chapter4.example1()

    # Save figure
    if prefix is not None:
        fig7.savefig(prefix + 'fig7.svg')
        fig7.savefig(prefix + 'fig7.png')

    return fig7


def make_figure_8(prefix=None, colors=None):
    """
    Figure 8, Example 4.2 Monte Carlo results

    Ported from MATLAB Code

    Nicholas O'Donoughue
    24 March 2021

    :param prefix: output directory to place generated figure
    :param colors: colormap for plotting
    :return: figure handle
    """

    fig8 = chapter4.example2(colors)

    # Save figure
    if prefix is not None:
        fig8.savefig(prefix + 'fig8.svg')
        fig8.savefig(prefix + 'fig8.png')

    return fig8
