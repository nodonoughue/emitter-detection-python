"""

"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.unit_conversions import db_to_lin


def crlb(snr, num_samples):
    """
    Compute the lower bound on unbiased estimation error for a Watson - Watt based angle of arrival receiver.

    Ported from MATLAB code.

    Nicholas O'Donoughue
    9 January 2021

    :param snr: signal-to-noise ratio [dB]
    :param num_samples: number of samples taken
    :return crlb: Cramer-Rao Lower Bound on error variance [psi]
    """

    snr_lin = db_to_lin(snr)
    return 1. / (num_samples * snr_lin)


def compute_df(r, x, y):
    """
    Compute the angle of arrival given a centrally located reference signal, and a pair of Adcock antennas oriented
    orthogonally.

    Ported from MATLAB code.

    Nicholas O'Donoughue
    9 January 2021

    :param r: reference signal from centrally located antenna
    :param x: test signal from the primary Adcock receiver, oriented in the +x direction (0 degrees)
    :param y: test signal from the secondary Adcock receiver, oriented in the +y direction (90 degrees CCW)
    :return psi: estimated angle of arrival (radians)
    """

    # Remove reference signal from test data, achieved via a conjugate inner transpose
    xx = np.inner(np.conjugate(r), x)
    yy = np.inner(np.conjugate(r), y)

    # Results should be V * cos(th) and V * sin(th), use atan2 to solve for th
    return np.arctan2(np.real(yy), np.real(xx))  # output in radians


def run_example():
    """
    Test script that demonstrates how to analyze a Watson-Watt DF receiver.

    Ported from MATLAB code.

    Nicholas O'Donoughue
    9 January 2021

    :return:
    """

    # Generate the Signals
    th_true = 45.
    psi_true = np.deg2rad(th_true)
    f = 1.0e9
    t_samp = 1 / (3 * f)  # ensure the Nyquist criteria is satisfied

    # Set up the parameter sweep
    num_samples_vec = np.asarray([1., 10., 100.])  # Number of temporal samples at each antenna test point
    snr_db_vec = np.arange(start=-10., step=0.2, stop=20.2)  # signal to noise ratio
    num_mc = 10000  # number of monte carlo trials at each parameter setting

    # Set up output variables
    out_shp = (np.size(num_samples_vec), np.size(snr_db_vec))
    rmse_psi = np.zeros(shape=out_shp)
    crlb_psi = np.zeros(shape=out_shp)

    # Loop over parameters
    print('Executing Watson Watt Monte Carlo sweep...')
    for idx_num_samples, this_num_samples in enumerate(num_samples_vec.tolist()):
        this_num_mc = num_mc / this_num_samples
        print('\t {} samples per estimate...'.format(this_num_samples))

        # Generate signal vectors
        t_vec = np.arange(this_num_samples) * t_samp   # Time vector
        r0 = np.cos(2 * np.pi * f * t_vec)             # Reference signal
        y0 = np.sin(psi_true) * r0
        x0 = np.cos(psi_true) * r0

        # Generate Monte Carlo Noise with unit power -- generate one for each MC trial
        ref_pwr = np.sqrt(np.mean(r0**2))  # root-mean-square of reference signal
        noise_base_r = [np.random.normal(loc=0., scale=ref_pwr, size=(this_num_samples, 1))
                        for _ in np.arange(this_num_mc)]
        noise_base_x = [np.random.normal(loc=0., scale=ref_pwr, size=(this_num_samples, 1))
                        for _ in np.arange(this_num_mc)]
        noise_base_y = [np.random.normal(loc=0., scale=ref_pwr, size=(this_num_samples, 1))
                        for _ in np.arange(this_num_mc)]

        # Loop over SNR levels
        for idx_snr, this_snr_db in enumerate(snr_db_vec.tolist()):
            if np.mod(idx_snr/10.) == 0:
                print('.', end='', flush=True)

            # Compute noise power, scale base noise
            noise_amp = np.sqrt(db_to_lin(-this_snr_db))

            # Generate noisy measurements
            r = [r0 + r * noise_amp for r in noise_base_r]
            y = [y0 + y * noise_amp for y in noise_base_y]
            x = [x0 + x * noise_amp for x in noise_base_x]

            # Compute the estimate for each Monte Carlo trial
            psi_est = np.asarray([compute_df(this_r, this_x, this_y) for this_r, this_x, this_y in zip(r, x, y)])

            # Compute RMS Error
            rmse_psi[idx_num_samples, idx_snr] = np.sqrt(np.mean((psi_est - psi_true)**2))

            # Compute CRLB for RMS Error
            crlb_psi[idx_num_samples, idx_snr] = np.abs(crlb(this_snr_db, this_num_samples))

        print('done.')

    # Generate the plot
    sns.set()

    _, _ = plt.subplots()

    crlb_label = 'CRLB'
    mc_label = 'Simulation Result'

    for idx_num_samples, this_num_samples in enumerate(num_samples_vec):
        if idx_num_samples == 0:
            crlb_label = 'CRLB, M={}'.format(this_num_samples)
            mc_label = 'Simulation Result, M={}'.format(this_num_samples)

        handle1 = plt.semilogy(snr_db_vec, np.rad2deg(np.sqrt(crlb_psi[idx_num_samples, :])), label=crlb_label)
        plt.semilogy(snr_db_vec, np.rad2deg(rmse_psi[idx_num_samples, :]), color=handle1[0].get_color(),
                     style='--', label=mc_label)

    plt.xlabel(r'$\xi$ [dB]')
    plt.ylabel('RMSE [deg]')
    plt.title('Watson Watt DF Performance')
    plt.legend(loc='lower left')
