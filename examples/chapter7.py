import numpy as np
import matplotlib.pyplot as plt
from utils import constants
from utils.unit_conversions import lin_to_db
import prop
import aoa


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
    parameters['wavelength'] = constants.speed_of_light / parameters['freq_hz']  # m
    parameters['aoa_true_rad'] = np.deg2rad(parameters['aoa_true_deg'])

    path_loss = prop.model.get_path_loss(range_m=parameters['range_m'], freq_hz=parameters['freq_hz'],
                                         tx_ht_m=parameters['tx_ht_m'], rx_ht_m=parameters['rx_ht_m'],
                                         include_atm_loss=False)

    parameters['eirp'] = lin_to_db(parameters['tx_pwr']) + parameters['tx_gain'] - parameters['tx_loss']
    parameters['sig_pwr'] = parameters['eirp'] - path_loss - parameters['rx_loss']
    parameters['noise_pwr'] = lin_to_db(constants.kT * parameters['bw_noise']) + parameters['noise_figure']
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
    path_loss_vec = prop.model.get_path_loss(range_m=range_m_vec, freq_hz=params['freq_hz'], tx_ht_m=params['tx_ht_m'],
                                             rx_ht_m=params['rx_ht_m'], include_atm_loss=False)
    noise_pwr = lin_to_db(constants.kT * params['bw_noise']) + params['noise_figure']
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
