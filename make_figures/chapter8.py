"""
Draw Figures - Chapter 8

This script generates all the figures that appear in Chapter 8 of the textbook.

Ported from MATLAB Code

Nicholas O'Donoughue
6 May 2021
"""

import utils
from utils.unit_conversions import lin_to_db, db_to_lin
import matplotlib.pyplot as plt
import numpy as np
import array_df
import time
from examples import chapter8


def make_all_figures(close_figs=False, force_recalc=True):
    """
    Call all the figure generators for this chapter.

    :close_figs: Boolean flag.  If true, will close all figures after generating them; for batch scripting.
                 Default=False
    :force_recalc: If set to False, will skip any figures that are time-consuming to generate.
    :return: List of figure handles
    """

    # Find the output directory
    prefix = utils.init_output_dir('chapter8')
    utils.init_plot_style()

    # Random Number Generator
    rng = np.random.default_rng(0)

    # Generate all figures
    fig3 = make_figure_3(prefix)
    fig4 = make_figure_4(prefix)
    fig5 = make_figure_5(prefix)
    fig6 = make_figure_6(prefix)
    fig7a = make_figure_7a(prefix)
    fig7b = make_figure_7b(prefix)
    fig9 = make_figure_9(prefix, rng)
    fig10 = make_figure_10(prefix)
    fig12 = make_figure_12(prefix, rng, force_recalc)
    fig13 = make_figure_13(prefix, rng, force_recalc)

    figs = [fig3, fig4, fig5, fig6, fig7a, fig7b, fig9, fig10, fig12, fig13]

    if close_figs:
        for fig in figs:
            plt.close(fig)

        return None
    else:
        plt.show()

        return figs


def make_figure_3(prefix=None):
    """
    Figure 3 -- Array Factor

    Ported from MATLAB Code

    Nicholas O'Donoughue
    6 May 2021

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    print('Generating Figure 8.3...')

    # Array parameters
    d_lam = 1/2
    th_0 = -30
    th = np.linspace(start=-90, stop=90, num=1001)

    psi_0 = th_0*np.pi/180
    psi = th*np.pi/180

    u = np.sin(psi)

    fig3 = plt.figure()

    num_elements_vec = [10, 25, 100]
    for num_elements in num_elements_vec:
        af = array_df.model.compute_array_factor_ula(d_lam, num_elements, psi, psi_0)
        plt.plot(u, af, label='N={:d}'.format(num_elements))

    plt.legend(loc='upper right')
    plt.xlabel(r'$u=sin(\theta)$')
    plt.ylabel('Array Factor [linear]')

    if prefix is not None:
        fig3.savefig(prefix + 'fig3.png')
        fig3.savefig(prefix + 'fig3.svg')

    return fig3


def make_figure_4(prefix=None):
    """
    Figure 4 -- Array Factor w/Grating Lobes

    Ported from MATLAB Code

    Nicholas O'Donoughue
    6 May 2021

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    print('Generating Figure 8.4...')

    num_elements = 11
    th_0 = -30
    th = np.linspace(start=-90, stop=90, num=1001)

    psi_0 = th_0*np.pi/180
    psi = th*np.pi/180

    u = np.sin(psi)

    fig4 = plt.figure()

    d_vec = [.5, 1, 2]
    for d in d_vec:
        af = array_df.model.compute_array_factor_ula(d, num_elements, psi, psi_0)
        plt.plot(u, af, label=r'd/$\lambda$' + '={:.1f}'.format(d))

    plt.legend(loc='upper right')
    plt.xlabel(r'$u=sin(\theta)$')
    plt.ylabel('Array Factor [linear]')

    # Annotation
    plt.text(-.4, .8, 'Mainlobe', fontsize=10)
    plt.text(.1, .65, 'Grating Lobes', fontsize=10)
    plt.plot([.05, .1], [.5, .6], color='black', label=None)
    plt.plot([.425, .1], [.5, .6], color='black', label=None)

    if prefix is not None:
        fig4.savefig(prefix + 'fig4.png')
        fig4.savefig(prefix + 'fig4.svg')

    return fig4


def make_figure_5(prefix=None):
    """
    Figure 5 - Grating Lobes w/Cosine Pattern

    Ported from MATLAB Code

    Nicholas O'Donoughue
    6 May 2021

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    print('Generating Figure 8.5...')

    num_elements = 11
    th_0 = -30
    th = np.linspace(start=-90, stop=90, num=1001)

    psi_0 = th_0*np.pi/180
    psi = th*np.pi/180

    u = np.sin(psi)

    def el_pat(angle_rad):
        return np.cos(angle_rad) ** 1.2

    fig5 = plt.figure()

    d_vec = [.5, 1, 2]
    for idx_d, this_d in enumerate(d_vec):
        af = array_df.model.compute_array_factor_ula(this_d, num_elements, psi, psi_0, el_pat)
        plt.plot(u, af, label=r'$d/\lambda$' + '={:.1f}'.format(this_d))

    plt.xlabel(r'$u=sin(\theta)$')
    plt.ylabel('Array Factor [linear]')
    plt.legend(loc='upper right')

    # Annotation
    plt.text(-.4, .8, 'Mainlobe', fontsize=10)
    plt.text(.1, .65, 'Grating Lobes', fontsize=10)
    plt.plot([.05, .1], [.5, .6], color='black', label=None)
    plt.plot([.425, .1], [.5, .6], color='black', label=None)

    if prefix is not None:
        fig5.savefig(prefix + 'fig5.png')
        fig5.savefig(prefix + 'fig5.svg')

    return fig5


def make_figure_6(prefix=None):
    """
    Figure 6 -- Beamwidth Plot

    Ported from MATLAB Code

    Nicholas O'Donoughue
    6 May 2021

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    print('Generating Figure 8.6...')

    # Set up array
    num_elements = 25
    d_lam = .5
    d_psi = .89/((num_elements-1)*d_lam)
    du = np.sin(d_psi)
    v, _ = array_df.model.make_steering_vector(d_lam, num_elements)

    # Set up source signals
    spacing = 1
    u1 = du*spacing/2
    u2 = -u1

    psi1 = np.arcsin(u1)
    psi2 = np.arcsin(u2)

    v1 = v(psi1)  # num_elements x 1
    v2 = v(psi2)  # num_elements x 1

    # Compute Array Factor
    u_scan = np.linspace(start=-.5, stop=.5, num=1001)
    psi_scan = np.arcsin(u_scan)

    array_factor1 = array_df.model.compute_array_factor(v, v1, psi_scan)
    array_factor2 = array_df.model.compute_array_factor(v, v2, psi_scan)

    fig6 = plt.figure()
    plt.plot(u_scan, 2*lin_to_db(np.abs(array_factor1)/np.max(np.abs(array_factor1), axis=0)))
    plt.plot(u_scan, 2*lin_to_db(np.abs(array_factor2)/np.max(np.abs(array_factor2), axis=0)))
    plt.plot(u1*np.array([1, 1]), [-20, 0], linestyle='--', color='black')
    plt.plot(u2*np.array([1, 1]), [-20, 0], linestyle='--', color='black')

    # Annotate the beamwidth
    plt.annotate(text='', xy=(u1, -16.5), xytext=(u2, -16.5), arrowprops=dict(arrowstyle='<->', color='k'))
    plt.text(-.02, -18, r'$\delta_u$', fontsize=11)

    plt.ylabel('Array Factor [dB]')  # Printed figure in text shows [linear], but units are in fact [dB]
    plt.xlabel(r'$u=sin(\theta)$')
    plt.ylim([-20, 0])
    plt.xlim([-.3, .3])

    if prefix is not None:
        fig6.savefig(prefix + 'fig6.png')
        fig6.savefig(prefix + 'fig6.svg')

    return fig6


def make_figure_7a(prefix=None):
    """
    Figure 7a - Array Tapers

    Ported from MATLAB Code

    Nicholas O'Donoughue
    6 May 2021

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    print('Generating Figure 8.7a...')

    # Initialize tapers
    num_elements = 11
    element_idx_vec = np.arange(num_elements)
    taper_types = ['uniform', 'cosine', 'hann', 'hamming']

    fig7a = plt.figure()

    for taper in taper_types:
        w, _ = utils.make_taper(num_elements, taper)
        plt.plot(element_idx_vec, w, label=taper)

    plt.xlabel('array element index')
    plt.ylabel('taper weight')
    plt.ylim([0, 1.1])
    plt.xlim([-0.5, 10.5])
    plt.legend(loc='upper right')

    if prefix is not None:
        fig7a.savefig(prefix + 'fig7a.png')
        fig7a.savefig(prefix + 'fig7a.svg')

    return fig7a


def make_figure_7b(prefix=None):
    """
    Figure 7b - Array Tapers

    Ported from MATLAB Code

    Nicholas O'Donoughue
    6 May 2021

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    print('Generating Figure 8.7b...')

    # Taper Parameters
    num_elements = 11
    taper_types = ['uniform', 'cosine', 'hann', 'hamming']

    fig7b = plt.figure()

    # Beampattern Parameters
    # Over-sampling factor; for smoothness
    osf = 100
    # Sampling index
    spatial_freq_vec = np.linspace(start=-1, stop=1, num=osf*num_elements)

    for taper in taper_types:
        # Make taper
        w, _ = utils.make_taper(num_elements, taper)

        # Take fourier transform, and normalize peak response
        window_spatial_freq = np.fft.fftshift(np.fft.fft(w, n=osf*num_elements))
        window_spatial_freq = window_spatial_freq / np.max(np.abs(window_spatial_freq))
    
        plt.plot(spatial_freq_vec, 2*lin_to_db(np.abs(window_spatial_freq)), label=taper)

    plt.xlim([0, 1])
    plt.ylim([-80, 0])
    plt.xlabel(r'$u = \cos(\theta)$')
    plt.ylabel(r'$|G(\theta)|^2$ [dB]')
    plt.legend(loc='upper right')

    if prefix is not None:
        fig7b.savefig(prefix + 'fig7b.png')
        fig7b.savefig(prefix + 'fig7b.svg')

    return fig7b


def make_figure_9(prefix=None, rng=None):
    """
    Figure 9/11 - Beamscan Images

    Ported from MATLAB Code

    Nicholas O'Donoughue
    6 May 2021

    :param prefix: output directory to place generated figure
    :param rng: random number generator
    :return: figure handle
    """

    print('Generating Figure 8.9...')

    if rng is None:
        rng = np.random.default_rng(0)

    # Set up array
    num_elements = 25
    d_lam = .5
    d_psi = .89/((num_elements-1)*d_lam)
    du = np.sin(d_psi)
    v, _ = array_df.model.make_steering_vector(d_lam, num_elements)
    
    # Set up source signals
    spacing = 1
    u1 = du*spacing/2
    u2 = -u1
    
    psi1 = np.arcsin(u1)
    psi2 = np.arcsin(u2)
    
    v1 = v(psi1)
    v2 = v(psi2)
    
    # Generate snapshots
    num_samples = 30
    s1 = np.sqrt(.5) * (rng.standard_normal(size=(1, num_samples)) + 1j * rng.standard_normal(size=(1, num_samples)))
    s2 = np.sqrt(.5) * (rng.standard_normal(size=(1, num_samples)) + 1j * rng.standard_normal(size=(1, num_samples)))
    x1 = v1 * s1
    x2 = v2 * s2
    
    # Add noise
    snr_db = 10
    snr = db_to_lin(snr_db)
    n = np.sqrt(1/(snr*2))*(rng.standard_normal(size=np.shape(x1)) + 1j * rng.standard_normal(size=np.shape(x1)))
    x = x1 + x2 + n
    
    # Generate Beamscan images
    pwr_vec, psi_vec = array_df.solvers.beamscan(x, v, np.pi/2, 1001)
    pwr_vec_mvdr, _ = array_df.solvers.beamscan_mvdr(x, v, np.pi/2, 1001)
    
    # Scale outputs
    pwr_vec = pwr_vec/np.max(pwr_vec)
    pwr_vec_mvdr = pwr_vec_mvdr/np.max(pwr_vec_mvdr)
    
    fig9 = plt.figure()
    plt.plot(np.sin(psi_vec), lin_to_db(pwr_vec), linewidth=1.5, label='Beamscan')
    plt.plot(np.sin(psi_vec), lin_to_db(pwr_vec_mvdr), linewidth=1.25, label='MVDR')
    plt.ylim([-30, 0])
    plt.xlabel('u')
    plt.ylabel('P [dB]')
    plt.legend(loc='upper right')
    
    if prefix is not None:
        fig9.savefig(prefix + 'fig9.png')
        fig9.savefig(prefix + 'fig9.svg')
    
    pwr_vec_music, _ = array_df.solvers.music(x, v, 2, np.pi/2, 1001)
    pwr_vec_music = pwr_vec_music/np.amax(np.abs(pwr_vec_music))
    plt.plot(np.sin(psi_vec), lin_to_db(np.abs(pwr_vec_music)), label='MUSIC')
    plt.legend()

    if prefix is not None:
        fig9.savefig(prefix + 'fig11.png')
        fig9.savefig(prefix + 'fig11.svg')

    return fig9
    

def make_figure_10(prefix=None):
    """
    Figure 10 - Beamscan Example Images

    Ported from MATLAB Code

    Nicholas O'Donoughue
    6 May 2021

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    print('Generating Figure 8.10 (using Example 8.1)...')

    fig10 = chapter8.example1()

    if prefix is not None:
        fig10.savefig(prefix + 'fig10.png')
        fig10.savefig(prefix + 'fig10.svg')

    return fig10


def make_figure_12(prefix=None, rng=None, force_recalc=True):
    """
    Figure 12 - CRLB Plot

    Ported from MATLAB Code

    Nicholas O'Donoughue
    6 May 2021

    :param prefix: output directory to place generated figure
    :param rng: random number generator
    :param force_recalc: if False, this routine will return an empty figure, to avoid time-consuming recalculation
    :return: figure handle
    """

    if not force_recalc:
        print('Skipping Figure 8.12... (re-run with force_recalc=True to generate)')
        return None

    print('Generating Figure 8.12...')

    if rng is None:
        rng = np.random.default_rng(0)
                    
    snr_db = np.arange(start=-20, step=.5, stop=0)
    snr_lin = db_to_lin(snr_db)
    psi = 5*np.pi/180
    num_samples = 100
    num_elements = 11
    v, v_dot = array_df.model.make_steering_vector(.5, num_elements)

    # Compute CRLB
    crlb_psi = np.zeros(shape=(np.size(snr_db), ))
    crlb_psi_stoch = np.zeros_like(crlb_psi)

    for idx_xi, this_xi in enumerate(snr_lin):
        crlb_psi[idx_xi] = array_df.perf.crlb_det(this_xi, 1, psi, num_samples, v, v_dot)
        crlb_psi_stoch[idx_xi] = array_df.perf.crlb_stochastic(this_xi, 1, psi, num_samples, v, v_dot)

    crlb_rmse_deg = np.sqrt(crlb_psi)*180/np.pi
    crlb_rmse_deg_stoch = np.sqrt(crlb_psi_stoch)*180/np.pi

    num_monte_carlo = 1000
    s = np.exp(1j*rng.uniform(low=0, high=2*np.pi, size=(1, num_samples, num_monte_carlo)))
    x0 = np.expand_dims(v(psi), axis=2) * s
    n = np.sqrt(1/2)*(rng.standard_normal(size=(num_elements, num_samples, num_monte_carlo))
                      + 1j * rng.standard_normal(size=(num_elements, num_samples, num_monte_carlo)))
    rmse_deg_beamscan = np.zeros(shape=(np.size(snr_db), ))
    rmse_deg_mvdr = np.zeros(shape=(np.size(snr_db), ))
    rmse_deg_music = np.zeros(shape=(np.size(snr_db), ))
    
    print('Executing array DF monte carlo trial...')
    iterations_per_marker = 10
    markers_per_row = 40
    iterations_per_row = markers_per_row * iterations_per_marker
    total_iterations = num_monte_carlo * len(snr_lin)
    t_start = time.perf_counter()
    for idx_xi, this_xi in enumerate(snr_lin):
        sn2 = 1/this_xi
        x = x0 + np.sqrt(sn2)*n

        this_err_beamscan = np.zeros(shape=(num_monte_carlo, ))
        this_err_mvdr = np.zeros(shape=(num_monte_carlo, ))
        this_err_music = np.zeros(shape=(num_monte_carlo, ))

        for idx_mc in np.arange(num_monte_carlo):
            curr_idx = idx_mc + idx_xi * num_monte_carlo
            utils.print_progress(total_iterations, curr_idx, iterations_per_marker, iterations_per_row, t_start)

            # Compute beamscan image
            pwr_vec, psi_vec = array_df.solvers.beamscan(x[:, :, idx_mc], v, np.pi/2, 2001)
            idx_pk = np.argmax(np.abs(pwr_vec))
            this_err_beamscan[idx_mc] = np.abs(psi_vec[idx_pk]-psi)

            # Compute MVDR beamscan image
            pwr_vec_mvdr, psi_vec = array_df.solvers.beamscan_mvdr(x[:, :, idx_mc], v, np.pi/2, 2001)
            idx_pk_mvdr = np.argmax(np.abs(pwr_vec_mvdr))
            this_err_mvdr[idx_mc] = np.abs(psi_vec[idx_pk_mvdr]-psi)

            # Compute MUSIC image
            pwr_vec_music, psi_vec = array_df.solvers.music(x[:, :, idx_mc], v, 1, np.pi/2, 2001)
            idx_pk_music = np.argmax(np.abs(pwr_vec_music))
            this_err_music[idx_mc] = np.abs(psi_vec[idx_pk_music]-psi)
        
        # Average Results
        rmse_deg_beamscan[idx_xi] = (180/np.pi)*np.sqrt(sum(this_err_beamscan**2)/num_monte_carlo)
        rmse_deg_mvdr[idx_xi] = (180/np.pi)*np.sqrt(sum(this_err_mvdr**2)/num_monte_carlo)
        rmse_deg_music[idx_xi] = (180/np.pi)*np.sqrt(sum(this_err_music**2)/num_monte_carlo)

    print('done.')
    t_elapsed = time.perf_counter() - t_start
    utils.print_elapsed(t_elapsed)

    fig12 = plt.figure()
    plt.semilogy(snr_db, crlb_rmse_deg, linestyle='--', color='black', label='Det. CRLB')
    plt.plot(snr_db, crlb_rmse_deg_stoch, linestyle='-.', color='black', label='Stochastic CRLB')

    plt.plot(snr_db, rmse_deg_beamscan, label='Beamscan', linewidth=2)
    plt.plot(snr_db, rmse_deg_mvdr, label='MVDR', linewidth=1.5)
    plt.plot(snr_db, rmse_deg_music, label='MUSIC', linewidth=1)
    
    plt.xlabel('SNR [dB]')
    plt.ylabel('RMSE [deg]')
    plt.legend(loc='upper right')

    if prefix is not None:
        fig12.savefig(prefix + 'fig12.png')
        fig12.savefig(prefix + 'fig12.svg')

    return fig12


def make_figure_13(prefix=None, rng=None, force_recalc=True):
    """
    Figure 13 - Example 8.2

    Ported from MATLAB Code

    Nicholas O'Donoughue
    6 May 2021

    :param prefix: output directory to place generated figure
    :param rng: random number generator
    :param force_recalc: if False, this routine will return an empty figure, to avoid time-consuming recalculation
    :return: figure handle
    """

    if not force_recalc:
        print('Skipping Figure 8.13 (re-run with force_recalc=True to generate)')
        return None

    print('Generating Figure 8.13...')

    if rng is None:
        rng = np.random.default_rng(0)

    fig13 = chapter8.example2(rng)

    if prefix is not None:
        fig13.savefig(prefix + 'fig13.png')
        fig13.savefig(prefix + 'fig13.svg')

    return fig13
