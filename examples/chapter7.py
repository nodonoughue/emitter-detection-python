import matplotlib.pyplot as plt
import numpy as np

from ewgeo import aoa
from ewgeo.prop.model import get_path_loss
from ewgeo.utils.constants import kT, speed_of_light
from ewgeo.utils.unit_conversions import db_to_lin, lin_to_db


def run_all_examples():
    """
    Run all chapter 7 examples and return a list of figure handles

    :return figs: list of figure handles
    """

    fig = example1()
    example2(fig)
    example3(fig)
    example4(fig)

    return [fig]


def initialize_parameters():
    """
    Initialize the common set of parameters for examples 7.1-7.4, to ensure they all run the same set of inputs.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    26 March 2021

    :return: dict storing scenario parameters
    """

    # Parameter Definition
    parameters = {'tx_pwr': 35e-3, 'tx_gain': 34, 'bw_signal': 31.3e6, 'freq_hz': 35e9, 'tx_loss': 0, 'tx_ht_m': 500,
                  'rx_loss': 2, 'noise_figure': 4, 'bw_noise': 40e6, 'signal_dur': 10e-6, 'rx_ht_m': 500,
                  'aoa_true_deg': 10, 'range_m': np.logspace(start=3, stop=5, num=20)}

    # Compute dependent parameters
    parameters['wavelength'] = speed_of_light / parameters['freq_hz']  # m
    parameters['aoa_true_rad'] = np.deg2rad(parameters['aoa_true_deg'])

    path_loss = get_path_loss(range_m=parameters['range_m'], freq_hz=parameters['freq_hz'],
                              tx_ht_m=parameters['tx_ht_m'], rx_ht_m=parameters['rx_ht_m'],
                              include_atm_loss=False)

    parameters['eirp'] = lin_to_db(parameters['tx_pwr']) + parameters['tx_gain'] - parameters['tx_loss']
    parameters['sig_pwr'] = parameters['eirp'] - path_loss - parameters['rx_loss']
    parameters['noise_pwr'] = lin_to_db(kT * parameters['bw_noise']) + parameters['noise_figure']
    parameters['snr_db'] = parameters['sig_pwr'] - parameters['noise_pwr']

    # Compute the number of time samples available
    parameters['num_samples'] = 1 + np.floor(parameters['signal_dur'] * parameters['bw_noise'])

    return parameters


def example1(fig=None):
    """
    Executes Example 7.1; if fig is provided then it adds the curve to the
    # existing plot, otherwise a new figure is generated.

    Ported from MATLAB Code
    
    Nicholas O'Donoughue
    26 March 2021

    :param fig: Existing Figure object on which to add curve
    :return: figure handle to generated graphic
    """

    # Parameter Definition
    params = initialize_parameters()

    # Generate the gain functions
    g, g_dot = aoa.make_gain_functions(aperture_type='adcock', d_lam=.25, psi_0=0)
    th_steer = np.array([-15, 0, 15])
    psi_steer = th_steer * np.pi / 180
    th_true = 10
    psi_true = th_true * np.pi / 180

    # Compute the CRLB
    crlb_psi = aoa.directional.crlb(params['snr_db'], params['num_samples'], g, g_dot, psi_steer, psi_true)
    rmse_th = (180 / np.pi) * np.sqrt(crlb_psi)

    # Re-do for directional
    d_lam_rectangular = 5
    g, g_dot = aoa.make_gain_functions(aperture_type='Rectangular', d_lam=d_lam_rectangular, psi_0=0)
    crlb_psi_rect = aoa.directional.crlb(params['snr_db'], params['num_samples'], g, g_dot, psi_steer,
                                         psi_true)
    rmse_th_rect = (180 / np.pi) * np.sqrt(crlb_psi_rect)

    # Plot
    if fig is None:
        fig = plt.figure()
    else:
        plt.figure(fig.number())

    plt.loglog(params['range_m'] / 1e3, rmse_th, linestyle='-', marker='+', label='Adcock')
    plt.loglog(params['range_m'] / 1e3, rmse_th_rect, linestyle='-', marker='^', label='Rectangular Aperture')
    plt.xlabel('Range [km]')
    plt.ylabel('RMSE [deg]')
    plt.title('Collision Avoidance Radar DF Example')
    plt.ylim([.1, 100])

    return fig


def example2(fig=None):
    """
    Executes Example 7.2; if fig is provided then it adds the curve to the
    # existing plot, otherwise a new figure is generated.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    26 March 2021

    :param fig: Existing Figure object on which to add curve
    :return: figure handle to generated graphic
    """

    params = initialize_parameters()

    # Watson Watt
    crlb_psi_watson = aoa.watson_watt.crlb(params['snr_db'], params['num_samples'])
    rmse_th_watson = (180 / np.pi) * np.sqrt(crlb_psi_watson)

    if fig is None:
        fig = plt.figure()
    else:
        plt.figure(fig.number)

    # Add to plot
    plt.loglog(params['range_m'] / 1e3, rmse_th_watson, linestyle='-', marker='s', label='Watson-Watt')
    plt.xlabel('Range [km]')
    plt.ylabel('RMSE [deg]')
    plt.title('Collision Avoidance Radar DF Example')
    plt.legend(loc='upper left')
    plt.ylim([.1, 100])

    return fig


def example3(fig=None):
    """
    Executes Example 7.3; if fig is provided then it adds the curve to the
    # existing plot, otherwise a new figure is generated.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    26 March 2021

    :param fig: Existing Figure object on which to add curve
    :return: figure handle to generated graphic
    """

    params = initialize_parameters()

    # Doppler Model
    ts = 1 / (2 * params['freq_hz'])
    radius_doppler = params['wavelength'] / 2  # Radius of the Doppler sensor (either ring or moving sensor)
    fr = 1 / params['signal_dur']
    crlb_psi_dop = aoa.doppler.crlb(params['snr_db'], params['num_samples'], 1, ts, params['freq_hz'],
                                    radius_doppler, fr, params['aoa_true_rad'])
    rmse_th_dop = (180 / np.pi) * np.sqrt(crlb_psi_dop)

    # Add to plot
    if fig is None:
        fig = plt.figure()
    else:
        plt.figure(fig.number)

    plt.loglog(params['range_m'] / 1e3, rmse_th_dop, linestyle='-', marker='v', label='Doppler')

    return fig


def example4(fig=None):
    """
    Executes Example 7.4; if fig is provided then it adds the curve to the
    # existing plot, otherwise a new figure is generated.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    26 March 2021

    :param fig: Existing Figure object on which to add curve
    :return: figure handle to generated graphic
    """

    params = initialize_parameters()

    # First -- we analytically compute the max range for an RMSE of 1 degree based on SNR
    rmse_max = 1  # Desired max angle of arrival RMSE (in degrees)
    crlb_max = (rmse_max * np.pi / 180) ** 2  # Max angle of arrival CRLB (in rad^2)

    # Loss as a function of range
    range_m_vec = np.arange(start=1e3, step=10, stop=100e3)  # 10 m spacing from 1 km to 100 km
    path_loss_vec = get_path_loss(range_m=range_m_vec, freq_hz=params['freq_hz'], tx_ht_m=params['tx_ht_m'],
                                  rx_ht_m=params['rx_ht_m'], include_atm_loss=False)
    noise_pwr = lin_to_db(kT * params['bw_noise']) + params['noise_figure']
    snr0 = lin_to_db(params['tx_pwr']) + params['tx_gain'] - params['tx_loss'] - params['rx_loss'] - noise_pwr
    snr_vec = snr0 - path_loss_vec

    # === Narrow Baseline Interferometer
    d_lam_narrow = .5  # Separation, in wavelengths

    # Find the minimum SNR (per sensor) to achieve the desired CRLB
    snr_min_eff = (1 / (2 * params['num_samples'] * crlb_max)) * (
            1 / (2 * np.pi * d_lam_narrow * np.cos(params['aoa_true_rad']))) ** 2
    snr_min_lin = 2 * snr_min_eff

    # Find the max range at which the minimum SNR is achieved
    idx_max_range = np.argmin(np.absolute(snr_vec - lin_to_db(snr_min_lin)), axis=None)
    max_range_narrow_m = range_m_vec[idx_max_range]
    print('Narrow Baseline Interferometer, max range: {:.1f} km'.format(max_range_narrow_m / 1e3))

    # === Wide Baseline Interferometer
    d_lam_wide = 2

    # Find the minimum SNR (per sensor) to achieve the desired CRLB
    snr_min_eff = (1 / (2 * params['num_samples'] * crlb_max)) * (
            1 / (2 * np.pi * d_lam_wide * np.cos(params['aoa_true_rad']))) ** 2
    snr_min_lin = 2 * snr_min_eff

    # Find the minimum SNR (per sensor) to achieve the desired CRLB
    idx_max_range = np.argmin(np.absolute(snr_vec - lin_to_db(snr_min_lin)), axis=None)
    max_range_wide_m = range_m_vec[idx_max_range]
    print('Narrow Baseline Interferometer, max range: {:.1f} km'.format(max_range_wide_m / 1e3))

    # Compute RMSE
    d_lam_narrow = .5
    crlb_psi_interf = aoa.interferometer.crlb(params['snr_db'], params['snr_db'], params['num_samples'],
                                              d_lam_narrow, params['aoa_true_rad'])
    rmse_th_interf = (180 / np.pi) * np.sqrt(crlb_psi_interf)

    d_lam_narrow = 2
    crlb_psi_interf = aoa.interferometer.crlb(params['snr_db'], params['snr_db'], params['num_samples'],
                                              d_lam_narrow, params['aoa_true_rad'])
    rmse_th_interf2 = (180 / np.pi) * np.sqrt(crlb_psi_interf)

    # Add to plot
    if fig is None:
        fig = plt.figure()
    else:
        plt.figure(fig.number)

    plt.loglog(params['range_m'] / 1e3, rmse_th_interf, linestyle='-', marker='o',
               label=r'Interferometer ($d=\lambda/2$)')
    plt.loglog(params['range_m'] / 1e3, rmse_th_interf2, linestyle='-', marker='x',
               label=r'Interferometer ($d=2\lambda$)')
    plt.legend(loc='upper left')

    return fig


def directional_df_monte_carlo(mc_params=None):
    """
    Monte Carlo simulation of Adcock and rectangular-aperture DF receivers:
    compares empirical RMSE against the CRLB across a range of SNR values
    and sample counts.

    Ported from MATLAB code (aoa/directional.py run_example).

    Nicholas O'Donoughue
    14 January 2021

    :param mc_params: Optional dict with keys 'monte_carlo_decimation' and 'min_num_monte_carlo'
                      to reduce the number of trials (useful for quick testing).
    :return figs: list of two figure handles [fig_adcock, fig_rectangular]
    """

    num_samples_vec = np.array([1, 10, 100])
    snr_db_vec      = np.arange(start=-20, step=2, stop=22)
    num_monte_carlo = 1000
    if mc_params is not None:
        num_monte_carlo = max(int(num_monte_carlo / mc_params['monte_carlo_decimation']),
                              mc_params['min_num_monte_carlo'])

    out_shp = (np.size(num_samples_vec), np.size(snr_db_vec))

    figs = []

    # ======================== Adcock ========================
    g_adcock, g_dot_adcock = aoa.make_gain_functions(aperture_type='adcock', d_lam=0.25, psi_0=0.0)

    th_true   = 5.0
    psi_true  = np.deg2rad(th_true)
    psi_res   = 0.001
    num_angles = 10
    psi       = np.deg2rad(np.linspace(-180.0, 180.0, num_angles, endpoint=False))
    x_adcock  = g_adcock(psi - psi_true)

    rmse_psi = np.zeros(out_shp)
    crlb_psi = np.zeros(out_shp)

    print('Executing Adcock Monte Carlo sweep...')
    for idx_m, num_samples in enumerate(num_samples_vec.tolist()):
        this_num_mc = int(num_monte_carlo // num_samples)
        print('\tM={}'.format(num_samples))
        noise_base = [np.random.normal(size=(num_angles, num_samples)) for _ in range(this_num_mc)]
        for idx_snr, snr_db in enumerate(snr_db_vec.tolist()):
            noise_amp = db_to_lin(-snr_db / 2)
            y = [x_adcock[:, np.newaxis] + noise_amp * n for n in noise_base]
            psi_est = np.array([aoa.directional.compute_df(this_y, psi, g_adcock, psi_res,
                                                           psi.min(), psi.max()) for this_y in y])
            rmse_psi[idx_m, idx_snr] = np.sqrt(np.mean((psi_est - psi_true) ** 2))
            crlb_psi[idx_m, idx_snr] = aoa.directional.crlb(snr_db, num_samples,
                                                             g_adcock, g_dot_adcock, psi, psi_true)
    print('done.')

    fig_adcock = plt.figure()
    for idx_m, this_num_samples in enumerate(num_samples_vec.tolist()):
        handle1 = plt.semilogy(snr_db_vec, np.rad2deg(np.sqrt(crlb_psi[idx_m, :])),
                               label='CRLB, M={}'.format(this_num_samples))
        plt.semilogy(snr_db_vec, np.rad2deg(rmse_psi[idx_m, :]),
                     color=handle1[0].get_color(), linestyle='--',
                     label='Simulation Result, M={}'.format(this_num_samples))
    plt.xlabel(r'$\xi$ [dB]')
    plt.ylabel('RMSE [deg]')
    plt.title('Adcock DF Performance')
    plt.legend(loc='lower left')
    figs.append(fig_adcock)

    # ======================== Rectangular Aperture ========================
    g_rect, g_dot_rect = aoa.make_gain_functions(aperture_type='rectangular',
                                                  d_lam=5, psi_0=0.0)

    th_true    = 5.0
    psi_true   = np.deg2rad(th_true)
    psi_res    = 0.001
    num_angles = 36
    psi        = np.deg2rad(np.linspace(-180.0, 180.0, num_angles, endpoint=False))
    x_rect     = g_rect(psi - psi_true)

    rmse_psi = np.zeros(out_shp)
    crlb_psi = np.zeros(out_shp)

    print('Executing Rectangular Aperture Monte Carlo sweep...')
    for idx_m, num_samples in enumerate(num_samples_vec.tolist()):
        this_num_mc = int(num_monte_carlo // num_samples)
        print('\tM={}'.format(num_samples))
        noise_base = [np.random.normal(size=(num_angles, num_samples)) for _ in range(this_num_mc)]
        for idx_snr, snr_db in enumerate(snr_db_vec.tolist()):
            noise_amp = db_to_lin(-snr_db / 2)
            y = [x_rect[:, np.newaxis] + noise_amp * n for n in noise_base]
            psi_est = np.array([aoa.directional.compute_df(this_y, psi, g_rect, psi_res,
                                                           psi.min(), psi.max()) for this_y in y])
            rmse_psi[idx_m, idx_snr] = np.sqrt(np.mean((psi_est - psi_true) ** 2))
            crlb_psi[idx_m, idx_snr] = aoa.directional.crlb(snr_db, num_samples,
                                                             g_rect, g_dot_rect, psi, psi_true)
    print('done.')

    fig_rect = plt.figure()
    for idx_m, this_num_samples in enumerate(num_samples_vec.tolist()):
        handle1 = plt.semilogy(snr_db_vec, np.rad2deg(np.sqrt(crlb_psi[idx_m, :])),
                               label='CRLB, M={}'.format(this_num_samples))
        plt.semilogy(snr_db_vec, np.rad2deg(rmse_psi[idx_m, :]),
                     color=handle1[0].get_color(), linestyle='--',
                     label='Simulation Result, M={}'.format(this_num_samples))
    plt.xlabel(r'$\xi$ [dB]')
    plt.ylabel('RMSE [deg]')
    plt.title('Rectangular Aperture DF Performance')
    plt.legend(loc='lower left')
    figs.append(fig_rect)

    return figs


def doppler_df_monte_carlo(mc_params=None):
    """
    Monte Carlo simulation of a Doppler DF receiver: compares empirical RMSE against the
    CRLB across a range of SNR values and sample counts.

    Ported from MATLAB code.

    Nicholas O'Donoughue
    14 January 2021

    :param mc_params: Optional dict with keys 'monte_carlo_decimation' and 'min_num_monte_carlo'
                      to reduce the number of trials (useful for quick testing).
    :return fig: figure handle to the generated RMSE vs SNR plot
    """

    # Signal parameters
    th_true   = 45.0
    psi_true  = np.deg2rad(th_true)
    amplitude = 1.0
    phi0      = 2 * np.pi * np.random.uniform()   # random starting phase
    f         = 1.0e9                              # carrier frequency [Hz]
    ts        = 1.0 / (5.0 * f)                   # sampling period [s]

    # Doppler antenna parameters
    c      = speed_of_light
    lam    = c / f
    radius = lam / 2.0      # half-wavelength rotation radius [m]
    psi_res = 1e-4           # desired DF resolution [rad]

    # Parameter sweep
    num_samples_vec  = np.asarray([10, 100, 1000])
    snr_db_vec       = np.arange(start=-10, stop=22, step=2)
    num_monte_carlo  = int(1e6)
    if mc_params is not None:
        num_monte_carlo = max(int(num_monte_carlo / mc_params['monte_carlo_decimation']),
                              mc_params['min_num_monte_carlo'])

    # Output arrays
    out_shp  = (np.size(num_samples_vec), np.size(snr_db_vec))
    rmse_psi = np.zeros(shape=out_shp)
    crlb_psi = np.zeros(shape=out_shp)

    print('Executing Doppler Monte Carlo sweep...')
    for idx_num_samples, this_num_samples in enumerate(num_samples_vec.tolist()):
        this_num_monte_carlo = int(num_monte_carlo / this_num_samples)
        print('\t M={}'.format(this_num_samples))

        # Reference signal (noiseless)
        t_vec = ts * np.arange(this_num_samples)
        r0 = amplitude * np.exp(1j * phi0) * np.exp(1j * 2 * np.pi * f * t_vec)

        # Doppler signal (noiseless) — one rotation per M samples
        fr = 1.0 / (ts * this_num_samples)
        x0 = amplitude * np.exp(1j * phi0) * np.exp(1j * 2 * np.pi * f * t_vec) \
             * np.exp(1j * 2 * np.pi * f * radius / c * np.cos(2 * np.pi * fr * t_vec - psi_true))

        # Pre-generate unit-amplitude complex noise bases
        noise_base_r = [np.random.normal(size=(this_num_samples, 2)).view(np.complex128).ravel()
                        for _ in range(this_num_monte_carlo)]
        noise_base_x = [np.random.normal(size=(this_num_samples, 2)).view(np.complex128).ravel()
                        for _ in range(this_num_monte_carlo)]

        # Loop over SNR levels
        for idx_snr, snr_db in enumerate(snr_db_vec):
            if np.mod(idx_snr, 10) == 0:
                print('.', end='')

            # Scale noise to achieve desired SNR  (sigma = amplitude / sqrt(SNR_lin))
            noise_amp = db_to_lin(-snr_db / 2)

            # Generate noisy signals
            r = [r0 + n * noise_amp for n in noise_base_r]
            x = [x0 + n * noise_amp for n in noise_base_x]

            # Compute the DF estimate for each trial
            psi_est = np.asarray([aoa.doppler.compute_df(this_r, this_x, ts, f, radius, fr,
                                                         psi_res, -np.pi, np.pi)
                                  for this_r, this_x in zip(r, x)])

            # Empirical RMSE and CRLB
            rmse_psi[idx_num_samples, idx_snr] = np.sqrt(np.mean((psi_est - psi_true) ** 2))
            crlb_psi[idx_num_samples, idx_snr] = aoa.doppler.crlb(snr_db, this_num_samples,
                                                                   amplitude, ts, f, radius, fr,
                                                                   psi_true)

    print('done.')

    fig = plt.figure()
    for idx_num_samples, this_num_samples in enumerate(num_samples_vec.tolist()):
        crlb_label = 'CRLB, M={}'.format(this_num_samples)
        mc_label   = 'Simulation Result, M={}'.format(this_num_samples)

        handle1 = plt.semilogy(snr_db_vec,
                               np.rad2deg(np.sqrt(crlb_psi[idx_num_samples, :])),
                               label=crlb_label)
        plt.semilogy(snr_db_vec,
                     np.rad2deg(rmse_psi[idx_num_samples, :]),
                     color=handle1[0].get_color(), linestyle='--', label=mc_label)

    plt.xlabel(r'$\xi$ [dB]')
    plt.ylabel('RMSE [deg]')
    plt.title('Doppler DF Performance')
    plt.legend(loc='lower left')

    return fig


if __name__ == '__main__':
    run_all_examples()
    plt.show()
