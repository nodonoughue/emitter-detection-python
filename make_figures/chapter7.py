"""
Draw Figures - Chapter 7

This script generates all the figures that appear in Chapter 7 of the textbook.

Ported from MATLAB Code

Nicholas O'Donoughue
26 March 2021
"""
import time

import utils
from utils.unit_conversions import lin_to_db, db_to_lin
import matplotlib.pyplot as plt
import numpy as np
import aoa
from examples import chapter7


def make_all_figures(close_figs=False, force_recalc=True):
    """
    Call all the figure generators for this chapter

    :close_figs: Boolean flag.  If true, will close all figures after generating them; for batch scripting.
                 Default=False
    :force_recalc: If set to False, will skip any figures that are time-consuming to generate.
    :return: List of figure handles
    """

    # Find the output directory
    prefix = utils.init_output_dir('chapter7')
    utils.init_plot_style()

    # Random Number Generator
    rng = np.random.default_rng(0)

    # Colormap
    colors = plt.get_cmap('tab10')

    # Generate all figures
    fig1 = make_figure_1(prefix)
    fig3 = make_figure_3(prefix, rng, colors, force_recalc)
    fig5 = make_figure_5(prefix, rng, colors, force_recalc)
    fig6b = make_figure_6b(prefix)
    fig7 = make_figure_7(prefix, rng, colors, force_recalc)
    fig8 = make_figure_8(prefix)
    fig10 = make_figure_10(prefix, rng, colors, force_recalc)
    fig12 = make_figure_12(prefix)
    fig14 = make_figure_14(prefix)
    fig15b = make_figure_15b(prefix)
    fig16 = make_figure_16(prefix)

    figs = [fig1, fig3, fig5, fig6b, fig7, fig8, fig10, fig12, fig14, fig15b, fig16]

    if close_figs:
        for fig in figs:
            plt.close(fig)

        return None
    else:
        plt.show()

        return figs


def make_figure_1(prefix=None):
    """
    Figure 1, Lobing Interpolation

    Ported from MATLAB Code

    Nicholas O'Donoughue
    26 March 2021

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    print('Generating Figure 7.1...')

    num_lobes = 3
    num_samples = 501
    x = np.linspace(start=-num_lobes, stop=num_lobes, num=num_samples)
    
    x0 = .3
    
    y = np.sinc(x-x0)
    
    # Sample points
    xs = np.arange(start=-2, step=.5, stop=2)
    ys = np.sinc(xs-x0)
    
    th_mult = 45/num_lobes  # Set the end points to +/- 45 degrees
    
    fig1 = plt.figure()
    plt.scatter(xs*th_mult, np.absolute(ys), marker='o', label='Sample Points')
    plt.plot(x*th_mult, np.absolute(y), linestyle='--', label='Interpolated Beampattern')
    plt.plot(np.array([1, 1])*x0*th_mult, [0, 1], linestyle=':', label='True Emitter Bearing')
    plt.xlabel(r'$\phi$ [degrees]')
    plt.ylabel('Antenna Gain')
    plt.legend(loc='upper left', prop={'size': 6})  # Manually reduce font size so legend is less obtrusive
    
    if prefix is not None:
        fig1.savefig(prefix + 'fig1.png')
        fig1.savefig(prefix + 'fig1.svg')
    
    return fig1


def make_figure_3(prefix=None, rng=np.random.default_rng(), colors=None, force_recalc=True):
    """
    Figure 3, Adcock CRLB

    Ported from MATLAB Code

    Nicholas O'Donoughue
    26 March 2021

    :param prefix: output directory to place generated figure
    :param rng: random number generator
    :param colors: colormap
    :param force_recalc: if False, this routine will return an empty figure, to avoid time-consuming recalculation
    :return: figure handle
    """

    if not force_recalc:
        print('Skipping Figure 7.3... (re-run with force_recalc=True to generate)')
        return None

    print('Generating Figure 7.3...')

    if colors is None:
        colors = plt.get_cmap('tab10')

    # Create the antenna pattern generating function
    d_lam = .25
    g, g_dot = aoa.make_gain_functions(aperture_type='adcock', d_lam=d_lam, psi_0=0)
    # --- NOTE --- g,g_dot take radian inputs (psi, not theta)
        
    # Generate the angular samples and true gain values
    th_true = 5
    psi_true = th_true*np.pi/180
    psi_res = .001  # desired resolution from multi-stage search; aoa/directional_df for details.

    num_angular_samples = 10
    th = np.linspace(start=-180, stop=180-360/num_angular_samples, num=num_angular_samples)
    psi = th*np.pi/180
    min_psi = np.min(psi, axis=None)
    max_psi = np.max(psi, axis=None)
    true_signal = np.expand_dims(g((psi-psi_true)), axis=[1, 2])  # Actual gain values
    
    # Set up the parameter sweep
    num_samples_vec = np.array([1, 10, 100])  # Number of temporal samples at each antenna test point
    snr_db_vec = np.arange(start=-20, step=2, stop=20)  # signal-to-noise ratio
    num_mc = 10000  # number of monte carlo trials at each parameter setting
    
    # Set up output scripts
    out_shp = (np.size(num_samples_vec), np.size(snr_db_vec))
    rmse_psi = np.zeros(shape=out_shp, dtype=float)
    crlb_psi = np.zeros(shape=out_shp, dtype=float)
    
    # Loop over parameters
    print('Executing Adcock Monte Carlo sweep...')
    iterations_per_marker = 1000
    markers_per_row = 40
    iterations_per_row = markers_per_row * iterations_per_marker
    total_iterations = num_mc*len(snr_db_vec)*len(num_samples_vec)
    t_start = time.perf_counter()
    for idx_samples, num_samples in enumerate(num_samples_vec):
        # Generate Monte Carlo Noise with unit power
        # -- for simplicity, we only generate the real component, since this
        #    receiver is only working on the real portion of the received
        #    signal
        noise_base = (1/np.sqrt(2))*rng.standard_normal(size=(num_angular_samples, num_samples, num_mc))

        for idx_snr, snr_db in enumerate(snr_db_vec.tolist()):
            # Compute noise power, scale base noise
            noise_pwr = db_to_lin(-snr_db)
            
            # Generate noisy measurement
            noise = np.sqrt(noise_pwr)*noise_base
            rx_signal = true_signal+noise
            
            # Estimate
            psi_est = np.zeros(shape=(num_mc, 1), dtype=float)

            for idx_mc in range(num_mc):
                curr_idx = idx_mc + idx_snr * num_mc + idx_samples * len(snr_db_vec) * num_mc
                utils.print_progress(total_iterations, curr_idx, iterations_per_marker, iterations_per_row, t_start)

                psi_est[idx_mc] = aoa.directional.compute_df(rx_signal[:, :, idx_mc], psi, g, psi_res,
                                                             min_psi, max_psi)
            
            rmse_psi[idx_samples, idx_snr] = np.sqrt(np.sum(np.absolute((psi_est-psi_true))**2, axis=None)/num_mc)
            
            # CRLB
            crlb_psi[idx_samples, idx_snr] = np.absolute(aoa.directional.crlb(snr_db, num_samples, g, g_dot, psi,
                                                                              psi_true))
        
    print('done')
    t_elapsed = time.perf_counter() - t_start
    utils.print_elapsed(t_elapsed)
    
    # Generate plot
    fig3 = plt.figure()
    crlb_label = 'CRLB'
    mc_label = 'Simulation Result'
    for idx_samples, num_samples in enumerate(num_samples_vec):
        plt.semilogy(snr_db_vec, np.sqrt(crlb_psi[idx_samples, :])*180/np.pi, color=colors(idx_samples),
                     label=crlb_label)
        plt.semilogy(snr_db_vec, rmse_psi[idx_samples, :]*180/np.pi, linestyle='--', color=colors(idx_samples),
                     label=mc_label)
        
        # Clear the labels, so only the first loop is printed in the legend
        crlb_label = None
        mc_label = None

    plt.xlabel(r'$\xi$ [dB]')
    plt.ylabel('RMSE [deg]')
    plt.title('Adcock DF Performance')
    plt.legend(loc='lower left')
    
    plt.text(2, 17, 'M=1', fontsize=10)
    plt.text(2, 4, 'M=10', fontsize=10)
    plt.text(2, 1.4, 'M=100', fontsize=10)
    
    if prefix is not None:
        fig3.savefig(prefix + 'fig3.png')
        fig3.savefig(prefix + 'fig3.svg')

    return fig3


def make_figure_5(prefix=None, rng=np.random.default_rng(), colors=None, force_recalc=True):
    """
    Figure 5, Rectangular Aperture CRLB

    Ported from MATLAB Code

    Nicholas O'Donoughue
    26 March 2021

    :param prefix: output directory to place generated figure
    :param rng: random number generator
    :param colors: colormap
    :param force_recalc: if False, this routine will return an empty figure, to avoid time-consuming recalculation
    :return: figure handle
    """

    if not force_recalc:
        print('Skipping Figure 7.5... (re-run with force_recalc=True to generate)')
        return None

    print('Generating Figure 7.5...')

    if colors is None:
        colors = plt.get_cmap('tab10')

    # Create the antenna pattern generating function
    aperture_wavelengths = 5
    g, g_dot = aoa.make_gain_functions(aperture_type='rectangular', d_lam=aperture_wavelengths, psi_0=0)
    # --- NOTE --- g,g_dot take radian inputs (psi, not theta)

    # Generate the angular samples and true gain values
    th_true = 5
    psi_true = th_true*np.pi/180
    num_angular_samples = 36  # number of samples
    th = np.linspace(start=-180, stop=180-360/num_angular_samples, num=num_angular_samples)
    psi = th*np.pi/180
    psi_res = .001  # desired resolution from multi-stage search; see aoa/directional_df for details.
    signal_true = np.expand_dims(g((psi-psi_true)), axis=[1, 2])  # Actual gain values

    # Set up the parameter sweep
    num_samples_vec = np.array([1, 10, 100], dtype=int)  # Number of temporal samples at each antenna test point
    snr_db_vec = np.arange(start=-10, stop=20.1, step=1)  # signal-to-noise ratio
    num_mc = 10000  # number of monte carlo trials at each parameter setting

    # Set up output scripts
    out_shp = (np.size(num_samples_vec), np.size(snr_db_vec))
    rmse_psi = np.zeros(shape=out_shp)
    crlb_psi = np.zeros(shape=out_shp)

    # Loop over parameters
    print('Executing Rectangular Aperture Monte Carlo sweep...')
    iterations_per_marker = 1000
    markers_per_row = 40
    iterations_per_row = markers_per_row * iterations_per_marker
    total_iterations = num_mc * len(snr_db_vec) * len(num_samples_vec)
    t_start = time.perf_counter()
    for idx_num_samples, num_samples in enumerate(num_samples_vec):
        # Generate Monte Carlo Noise with unit power
        noise_base = 1/np.sqrt(2)*(rng.standard_normal(size=(num_angular_samples, num_samples, num_mc))
                                   + 1j*rng.standard_normal(size=(num_angular_samples, num_samples, num_mc)))

        for idx_snr, snr_db in enumerate(snr_db_vec.tolist()):
            # Compute noise power, scale base noise
            noise_pwr = db_to_lin(-snr_db)

            # Generate noisy measurement
            noise = np.sqrt(noise_pwr)*noise_base
            signal_rx = signal_true+noise

            # Estimate
            psi_est = np.zeros(shape=(num_mc, ))

            for idx_mc in range(num_mc):
                curr_idx = idx_mc + idx_snr * num_mc + idx_num_samples * len(snr_db_vec) * num_mc
                utils.print_progress(total_iterations, curr_idx, iterations_per_marker, iterations_per_row, t_start)

                psi_est[idx_mc] = aoa.directional.compute_df(signal_rx[:, :, idx_mc], psi, g, psi_res,
                                                             -np.pi, np.pi)

            rmse_psi[idx_num_samples, idx_snr] = np.sqrt(np.sum(np.absolute((psi_est-psi_true))**2)/num_mc)

            # CRLB
            crlb_psi[idx_num_samples, idx_snr] = np.absolute(aoa.directional.crlb(snr_db, num_samples, g, g_dot, psi,
                                                                                  psi_true))

    print('done')
    t_elapsed = time.perf_counter() - t_start
    utils.print_elapsed(t_elapsed)

    # Generate figure
    fig5 = plt.figure()
    crlb_label = 'CRLB'
    mc_label = 'Simulation Result'

    for idx_num_samples in range(np.size(num_samples_vec)):
        plt.semilogy(snr_db_vec, np.sqrt(crlb_psi[idx_num_samples, :])*180/np.pi,
                     color=colors(idx_num_samples), label=crlb_label)
        plt.semilogy(snr_db_vec, rmse_psi[idx_num_samples, :]*180/np.pi, linestyle='--',
                     color=colors(idx_num_samples), label=mc_label)

        crlb_label = None
        mc_label = None

    plt.xlabel(r'$\xi$ [dB]')
    plt.ylabel('RMSE [deg]')
    plt.title('Rectangular Aperture DF Performance')
    plt.legend(loc='lower left')

    plt.text(7, 5, 'M=1', fontsize=10)
    plt.text(7, 1.7, 'M=10', fontsize=10)
    plt.text(7, .5, 'M=100', fontsize=10)

    if prefix is not None:
        fig5.savefig(prefix + 'fig5.svg')
        fig5.savefig(prefix + 'fig5.png')

    return fig5


def make_figure_6b(prefix=None):
    """
    Figure 6b, Watson-Watt Patterns

    Ported from MATLAB Code

    Nicholas O'Donoughue
    26 March 2021

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    print('Generating Figure 7.6b...')

    phi = np.arange(start=-np.pi, step=np.pi/101, stop=np.pi)

    gain_pattern_one = 2*np.cos(phi)
    gain_pattern_two = 2*np.cos(phi-np.pi/2)
    gain_omni = np.ones_like(phi)

    fig6b = plt.figure()
    plt.polar(phi, np.absolute(gain_pattern_one))
    plt.polar(phi, np.absolute(gain_pattern_two))
    plt.polar(phi, gain_omni, linestyle='--')

    plt.axis('off')

    # Text coordinates are in (th, r) instead of (x, y) because we're using plt.polar
    plt.text(.25, 1.1, 'Reference', fontsize=10)
    plt.text(0, 2.1, 'Horizontal Adcock', fontsize=10)
    plt.text(np.pi/2, 2.1, 'Vertical Adcock', fontsize=10)

    if prefix is not None:
        fig6b.savefig(prefix + 'fig6b.png')
        fig6b.savefig(prefix + 'fig6b.svg')

    return fig6b


def make_figure_7(prefix=None, rng=np.random.default_rng(), colors=None, force_recalc=True):
    """
    Figure 7, Watson-Watt Performance

    Ported from MATLAB Code

    Nicholas O'Donoughue
    26 March 2021

    :param prefix: output directory to place generated figure
    :param rng: random number generator
    :param colors: colormap
    :param force_recalc: if False, this routine will return an empty figure, to avoid time-consuming recalculation
    :return: figure handle
    """

    if not force_recalc:
        print('Skipping Figure 7.7... (re-run with force_recalc=True to generate)')
        return None

    print('Generating Figure 7.7...')

    if colors is None:
        colors = plt.get_cmap('tab10')

    # Generate the Signals
    th_true = 45
    psi_true = th_true*np.pi/180
    f = 1e9
    t_samp = 1/(3*f)  # ensure the Nyquist criteria is satisfied (1/2 would be at the Nyquist rate, 1/3 is oversampled)

    # Set up the parameter sweep
    num_samples_vec = np.array([1, 10, 100], dtype=int)  # Number of temporal samples at each antenna test point
    snr_db_vec = np.arange(start=-10, step=.2, stop=20)  # signal-to-noise ratio
    num_mc = 10000  # number of monte carlo trials at each parameter setting

    # Set up output scripts
    out_shp = np.broadcast(np.expand_dims(num_samples_vec, axis=0), np.expand_dims(snr_db_vec, axis=1))
    rmse_psi = np.zeros(shape=out_shp.shape)
    crlb_psi = np.zeros(shape=out_shp.shape)

    # Loop over parameters
    print('Executing Watson-Watt Monte Carlo sweep...')
    iterations_per_marker = 1000
    markers_per_row = 40
    iterations_per_row = markers_per_row * iterations_per_marker
    total_iterations = num_mc * len(snr_db_vec) * len(num_samples_vec)
    t_start = time.perf_counter()
    for idx_num_samples, this_num_samples in enumerate(num_samples_vec.tolist()):
        # Generate signal vectors
        t_vec = np.expand_dims(np.arange(this_num_samples)*t_samp, axis=1)
        r0 = np.cos(2*np.pi*f*t_vec)
        y0 = np.sin(psi_true)*r0
        x0 = np.cos(psi_true)*r0

        # Generate Monte Carlo Noise with unit power
        sample_power = np.linalg.norm(r0)/np.sqrt(this_num_samples)  # average sample power
        noise_base_r = sample_power*rng.standard_normal(size=(this_num_samples, num_mc))
        noise_base_x = sample_power*rng.standard_normal(size=(this_num_samples, num_mc))
        noise_base_y = sample_power*rng.standard_normal(size=(this_num_samples, num_mc))

        # Loop over SNR vector
        for idx_snr, snr_db in enumerate(snr_db_vec):
            # Compute noise power, scale base noise
            snr_lin = db_to_lin(snr_db)

            # Generate noisy measurement -- divide noise power (which, by default has equal power to signal) by SNR
            # to get scaled noise power for this SNR setting
            r = r0 + noise_base_r/np.sqrt(snr_lin)
            y = y0 + noise_base_y/np.sqrt(snr_lin)
            x = x0 + noise_base_x/np.sqrt(snr_lin)

            # Compute the estimate
            psi_est = np.zeros(shape=(num_mc, ))

            for idx_mc in np.arange(num_mc):
                curr_idx = idx_mc + idx_snr * num_mc + idx_num_samples * len(snr_db_vec) * num_mc
                utils.print_progress(total_iterations, curr_idx, iterations_per_marker, iterations_per_row, t_start)

                psi_est[idx_mc] = aoa.watson_watt.compute_df(r[:, idx_mc], x[:, idx_mc], y[:, idx_mc])

            rmse_psi[idx_snr, idx_num_samples] = np.sqrt(((psi_est-psi_true)**2).mean())

            # CRLB
            crlb_psi[idx_snr, idx_num_samples] = np.absolute(aoa.watson_watt.crlb(snr_db, this_num_samples))

    print('done')
    t_elapsed = time.perf_counter() - t_start
    utils.print_elapsed(t_elapsed)

    fig7 = plt.figure()
    crlb_label = 'CRLB'
    mc_label = 'Simulation Result'
    for idx_num_samples, _ in enumerate(num_samples_vec):
        plt.semilogy(snr_db_vec, np.sqrt(crlb_psi[:, idx_num_samples])*180/np.pi,
                     color=colors(idx_num_samples), label=crlb_label)
        plt.semilogy(snr_db_vec, rmse_psi[:, idx_num_samples]*180/np.pi, linestyle='--',
                     color=colors(idx_num_samples), label=mc_label)

        crlb_label = None
        mc_label = None

    plt.xlabel(r'$\xi$ [dB]')
    plt.ylabel('RMSE [deg]')
    plt.title('Watson-Watt DF Performance')
    plt.legend(loc='lower left')

    plt.text(10, 25, 'M=1', fontsize=10)
    plt.text(10, 7, 'M=10', fontsize=10)
    plt.text(10, 2.5, 'M=100', fontsize=10)

    if prefix is not None:
        fig7.savefig(prefix + 'fig7.png')
        fig7.savefig(prefix + 'fig7.svg')

    return fig7


def make_figure_8(prefix=None):
    """
    Figure 8b, Doppler

    Ported from MATLAB Code

    Nicholas O'Donoughue
    26 March 2021

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    print('Generating Figure 7.8b...')

    num_samples = 128
    m_vec = np.arange(num_samples)
    phi0 = 5*np.pi/3
    fd = np.cos(m_vec/num_samples*2*np.pi+phi0)
    f0 = 2.5

    fig8b = plt.figure()
    plt.plot(m_vec, f0*np.ones_like(fd), label='f')
    plt.plot(m_vec, f0+fd, label='f_d')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.text(.8*num_samples, f0+.1, r'$f$', fontsize=12)
    plt.text(.35*num_samples, f0 + fd[int(np.fix(.3*num_samples))] - .15, r'$f+f_d(t)$', fontsize=12)

    fd_desc = np.copy(fd)
    np.place(fd_desc[:-1], fd[1:] > fd[:-1], np.inf)
    idx = np.argmin(np.absolute(fd_desc), axis=None)
    plt.text(idx+1, f0-.8*np.max(fd, axis=None), r'$\tau$', fontsize=12)
    plt.vlines(idx, 0, f0, linestyles='dashed')

    plt.ylim([f0+np.min(fd)-.2, f0+np.max(fd)+.2])

    if prefix is not None:
        fig8b.savefig(prefix + 'fig8b.png')
        fig8b.savefig(prefix + 'fig8b.svg')

    return fig8b


def make_figure_10(prefix=None, rng=np.random.default_rng(), colors=None, force_recalc=True):
    """
    Figure 10, Doppler CRLB

    Ported from MATLAB Code

    Nicholas O'Donoughue
    26 March 2021

    :param prefix: output directory to place generated figure
    :param rng: random number generator
    :param colors: colormap
    :param force_recalc: if False, this routine will return an empty figure, to avoid time-consuming recalculation
    :return: figure handle
    """

    if not force_recalc:
        print('Skipping Figure 7.10... (re-run with force_recalc=True to generate)')
        return None

    print('Generating Figure 7.10...')

    if colors is None:
        colors = plt.get_cmap('tab10')

    # Generate the Signals
    th_true = 45
    psi_true = th_true*np.pi/180
    signal_amp = 1
    phi0 = rng.uniform(low=0, high=2*np.pi)
    f = 1e9
    ts = 1/(5*f)

    # Doppler antenna parameters
    c = utils.constants.speed_of_light
    lam = c/f
    ant_radius = lam/2
    psi_res = .0001  # desired doppler resolution

    # Set up the parameter sweep
    num_samples_vec = np.array([10, 100, 1000], dtype=int)  # Number of temporal samples at each antenna test point
    snr_db_vec = np.arange(start=-10, step=2, stop=30.1)  # signal-to-noise ratio
    num_mc = 10000  # number of monte carlo trials at each parameter setting

    # Set up output scripts
    out_shp = np.broadcast(np.expand_dims(num_samples_vec, axis=1), np.expand_dims(snr_db_vec, axis=0))
    rmse_psi = np.zeros(shape=out_shp.shape)
    crlb_psi = np.zeros(shape=out_shp.shape)

    # Loop over parameters
    print('Executing Doppler Monte Carlo sweep...')
    iterations_per_marker = 1000
    markers_per_row = 40
    iterations_per_row = markers_per_row * iterations_per_marker
    total_iterations = num_mc * len(snr_db_vec) * len(num_samples_vec)
    t_start = time.perf_counter()
    for idx_num_samples, num_samples in enumerate(num_samples_vec.tolist()):
        # Reference signal
        t_vec = ts*np.arange(num_samples)
        r0 = signal_amp*np.exp(1j*phi0)*np.exp(1j*2*np.pi*f*t_vec)

        # Doppler signal
        fr = 1/(ts*num_samples)  # Ensure a single cycle during num_samples
        x0 = signal_amp*np.exp(1j*phi0)*np.exp(1j*2*np.pi*f*t_vec)*np.exp(1j*2*np.pi*f*ant_radius/c *
                                                                          np.cos(2*np.pi*fr*t_vec-psi_true))

        # Generate noise signal
        sample_power = np.sqrt(signal_amp/2)
        noise_shp = (num_samples, num_mc)
        noise_base_r = sample_power*(rng.standard_normal(size=noise_shp) + 1j*rng.standard_normal(size=noise_shp))
        noise_base_x = sample_power*(rng.standard_normal(size=noise_shp) + 1j*rng.standard_normal(size=noise_shp))

        for idx_snr, snr_db in enumerate(snr_db_vec.tolist()):
            # Compute noise power, scale base noise
            noise_pwr = db_to_lin(-snr_db)

            # Generate noisy signals
            r = np.expand_dims(r0, axis=1) + noise_base_r*np.sqrt(noise_pwr)
            x = np.expand_dims(x0, axis=1) + noise_base_x*np.sqrt(noise_pwr)

            # Compute the estimate
            psi_est = np.zeros(shape=(num_mc, ))
            for idx_mc in range(num_mc):
                curr_idx = idx_mc + idx_snr * num_mc + idx_num_samples * len(snr_db_vec) * num_mc
                utils.print_progress(total_iterations, curr_idx, iterations_per_marker, iterations_per_row, t_start)

                psi_est[idx_mc] = aoa.doppler.compute_df(r[:, idx_mc], x[:, idx_mc], ts, f, ant_radius, fr, psi_res,
                                                         -np.pi, np.pi)
            
            rmse_psi[idx_num_samples, idx_snr] = np.sqrt(np.sum(np.absolute((psi_est-psi_true))**2, axis=None)
                                                         / num_mc)

            # CRLB
            crlb_psi[idx_num_samples, idx_snr] = aoa.doppler.crlb(snr_db, num_samples, signal_amp, ts, f, ant_radius,
                                                                  fr, psi_true)
        
    print('done')
    t_elapsed = time.perf_counter() - t_start
    utils.print_elapsed(t_elapsed)

    fig10 = plt.figure()
    crlb_label = 'CRLB'
    mc_label = 'Simulation Result'
    for idx_num_samples, _ in enumerate(num_samples_vec):
        plt.semilogy(snr_db_vec, np.sqrt(crlb_psi[idx_num_samples, :]) * 180 / np.pi,
                     color=colors(idx_num_samples), label=crlb_label)
        plt.semilogy(snr_db_vec, rmse_psi[idx_num_samples, :] * 180 / np.pi, linestyle='--',
                     color=colors(idx_num_samples), label=mc_label)

        crlb_label = None
        mc_label = None

    plt.xlabel(r'$\xi$ [dB]')
    plt.ylabel('RMSE [deg]')
    plt.title('Doppler DF Performance')
    plt.legend(loc='lower left')

    plt.text(5, 5, 'M=10', fontsize=10)
    plt.text(5, 1.7, 'M=100', fontsize=10)
    plt.text(5, .5, 'M=1,000', fontsize=10)

    if prefix is not None:
        fig10.savefig(prefix + 'fig10.svg')
        fig10.savefig(prefix + 'fig10.png')

    return fig10


def make_figure_12(prefix=None):
    """
    Figure 12, Interferometer

    Ported from MATLAB Code

    Nicholas O'Donoughue
    26 March 2021

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    print('Generating Figure 7.12...')

    d_lam = np.array([.5, 1, 2, 4])

    psi_s = np.linspace(start=-60, stop=60, num=201)*np.pi/180
    psi_0 = -10*np.pi/180

    g_true = (1+np.exp(1j*2*np.pi*np.expand_dims(d_lam, axis=1)*np.expand_dims(np.sin(psi_s)-np.sin(psi_0), axis=0)))/2

    fig12 = plt.figure()
    for idx_plot, this_d_lam in enumerate(d_lam):
        this_width = 2 - idx_plot*.25
        plt.plot(180*psi_s/np.pi, lin_to_db(np.absolute(g_true[idx_plot, :])), linewidth=this_width,
                 label=r'd/$\lambda$' + '={:.1f}'.format(this_d_lam))

    plt.ylim([-10, 0])
    plt.xlabel(r'$\theta$')
    plt.ylabel('Normalized Response [dB]')
    plt.plot(180*psi_0/np.pi*np.array([1, 1]), np.array([-10, 0]), color='k', linestyle='--', label='AOA')
    plt.legend(loc='lower left')

    if prefix is not None:
        fig12.savefig(prefix + 'fig12.svg')
        fig12.savefig(prefix + 'fig12.png')

    return fig12


def make_figure_14(prefix=None):
    """
    Figure 14, Interferometer Example

    Ported from MATLAB Code

    Nicholas O'Donoughue
    26 March 2021

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    print('Generating Figure 7.14 (using Example 7.1)...')

    fig14 = chapter7.example1(None)
    chapter7.example2(fig14)
    chapter7.example3(fig14)
    chapter7.example4(fig14)

    if prefix is not None:
        fig14.savefig(prefix + 'fig14.svg')
        fig14.savefig(prefix + 'fig14.png')

    return fig14


def make_figure_15b(prefix=None):
    """
    Figure 15b, Monopulse

    Ported from MATLAB Code

    Nicholas O'Donoughue
    26 March 2021

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    print('Generating Figure 7.15b...')

    # Make monopulse beampatterns
    num_angle_samples = 10001
    x = np.linspace(start=-3, stop=3, num=num_angle_samples)

    # Illumination Pattern
    a = np.sinc(x)

    # Individual Signals
    s = a*np.cos(np.pi*x)
    d = a*np.sin(np.pi*x)
    k = d/s
    k[np.absolute(x) > .5] = np.inf

    fig15b = plt.figure()
    plt.plot(x, s, label='Sum')
    plt.plot(x, d, label='Difference')
    plt.plot(x, k, linestyle='--', label='Monopulse Ratio')
    plt.xlabel('Angle (beamwidths)')
    plt.ylabel('Normalized Antenna Pattern')
    plt.legend(loc='lower left')
    plt.ylim([-1, 1])

    if prefix is not None:
        fig15b.savefig(prefix + 'fig15b.svg')
        fig15b.savefig(prefix + 'fig15b.png')

    return fig15b


def make_figure_16(prefix=None):
    """
    Figure 16, Monopulse Error

    Ported from MATLAB Code

    Nicholas O'Donoughue
    26 March 2021

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    print('Generating Figure 7.16...')

    th_bw = 4
    th = np.arange(start=-th_bw/2, step=.1, stop=th_bw/2+.1)
    km = 1.606

    snr_db = np.array([10, 20])
    snr_lin = db_to_lin(snr_db)

    rmse_th = np.sqrt(1/(km**2*np.expand_dims(snr_lin, axis=1))*(1+(km*np.expand_dims(th, axis=0)/th_bw)**2))

    fig16 = plt.figure()
    for idx, this_snr_db in enumerate(snr_db):
        plt.plot(th/th_bw, rmse_th[idx, :], label=r'$\xi$' + ' = {:d} dB'.format(this_snr_db))

    plt.legend(loc='upper left')
    plt.xlabel(r'$\overline{\theta}$')
    plt.ylabel(r'$\sigma_{\overline{\theta}}$ [deg]')
    plt.ylim([0, .5])

    if prefix is not None:
        fig16.savefig(prefix + 'fig16.svg')
        fig16.savefig(prefix + 'fig16.png')

    return fig16
