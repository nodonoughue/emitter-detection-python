
import numpy as np
import matplotlib.pyplot as plt
from ..utils.unit_conversions import lin_to_db, db_to_lin


def crlb(snr1, snr2, num_samples, d_lam, psi_true):
    """
    Computes the lower bound on unbiased estimator error for an
    interferometer based direction of arrival receiver with multiple
    amplitude samples taken (M samples)

    Ported from MATLAB code.

    Nicholas O'Donoughue
    10 January 2021

    :param snr1:         Signal-to-Noise ratio [dB] at receiver 1
    :param snr2:         Signal-to-Noise ratio [dB] at receiver 2
    :param num_samples:  Number of samples
    :param d_lam:        Distance between receivers, divided by the signal wavelength
    :param psi_true:     Angle of arrival [radians]
    :return crlb:        Lower bound on the Mean Squared Error of an unbiased estimation of psi (radians)
    """

    # Compute the effective SNR
    snr_lin1 = db_to_lin(snr1)
    snr_lin2 = db_to_lin(snr2)
    snr_eff = 1./(1./snr_lin1 + 1./snr_lin2)

    return (1. / (2. * num_samples * snr_eff)) * (1. / (2. * np.pi * d_lam * np.cos(psi_true)))**2  # output in radians


def compute_df(x1, x2, d_lam):
    """
    Compute the estimated angle of arrival for an interferometer, given the
    complex signal at each of two receivers, and the distance between them.

    Ported from MATLAB code.

    Nicholas O'Donoughue
    10 January 2021

    :param x1:      Signal vector from antenna 1
    :param x2:      Signal vector from antenna 2
    :param d_lam:   Antenna spacing, divided by the signal wavelength
    :return:        Estimated angle of arrival [radians]
    """

    # The inner product of the two signals is a sufficient statistic for the
    # phase between them, in the presence of a single signal and Gaussian noise
    y = np.inner(np.conjugate(x1), x2)

    # Use atan2 to solve for the complex phase angle
    phi_est = np.arctan2(np.imag(y), np.real(y))

    # Convert from phase angle to angle of arrival
    return np.arcsin(phi_est/(2.*np.pi*d_lam))


def run_example():
    """
    Example approach to analyze an interferometer

    Ported from MATLAB code.

    Nicholas O'Donoughue
    10 January 2021

    :return: None
    """

    # Generate the Signals
    th_true = 45                                # angle (degrees)
    d_lam = .5
    psi_true = np.deg2rad(th_true)              # angle (radians)
    phi = 2*np.pi*d_lam*np.sin(psi_true)        # interferometer phase
    alpha = 1                                   # power scale
    
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
    for this_num_samples, idx_num_samples in enumerate(num_samples_vec):
        this_num_mc = num_mc / this_num_samples
        print('\t {} samples per estimate...'.format(this_num_samples))

        # Generate Signals
        iq_amp = np.sqrt(2)/2
        s1 = [np.random.normal(loc=0.0, scale=iq_amp, size=(this_num_samples, 2)).view(np.complex128)
              for _ in np.arange(this_num_mc)]
        s2 = [alpha*this_s1*np.exp(1j*phi) for this_s1 in s1]
    
        # Generate Noise
        noise_base1 = [np.random.normal(loc=0.0, scale=iq_amp, size=(this_num_samples, 2)).view(np.complex128)
                       for _ in np.arange(this_num_mc)]
        noise_base2 = [np.random.normal(loc=0.0, scale=iq_amp, size=(this_num_samples, 2)).view(np.complex128)
                       for _ in np.arange(this_num_mc)]
    
        # Loop over SNR levels
        for this_snr_db, idx_snr in enumerate(snr_db_vec):
            if np.mod(idx_snr, 10) == 0:
                print('.', end='', flush=True)
    
            # Compute noise power, scale base noise
            noise_pwr = db_to_lin(-this_snr_db)
    
            # Generate noisy signals
            x1 = [this_s1 + np.sqrt(noise_pwr)*this_noise for (this_s1, this_noise) in zip(s1, noise_base1)]
            x2 = [this_s2 + np.sqrt(noise_pwr)*this_noise for (this_s2, this_noise) in zip(s2, noise_base2)]
    
            # Compute the estimate for each Monte Carlo trial
            psi_est = np.asarray([compute_df(this_x1, this_x2, d_lam) for (this_x1, this_x2) in zip(x1, x2)])

            # Compute RMS Error
            rmse_psi[idx_num_samples, idx_snr] = np.sqrt(np.mean((psi_est-psi_true)**2))
    
            # Compute CRLB for RMS Error
            crlb_psi[idx_num_samples, idx_snr] = crlb(this_snr_db, this_snr_db + lin_to_db(alpha ** 2),
                                                      this_num_samples, d_lam, psi_true)

        print('done.')

    _, _ = plt.subplots()

    for this_num_samples, idx_num_samples in enumerate(num_samples_vec):
        crlb_label = 'CRLB, M={}'.format(this_num_samples)
        mc_label = 'Simulation Result, M={}'.format(this_num_samples)

        # Plot the MC and CRLB results for this number of samples
        handle1 = plt.semilogy(snr_db_vec, np.rad2deg(np.sqrt(crlb_psi[idx_num_samples, :])), label=crlb_label)
        plt.semilogy(snr_db_vec, np.rad2deg(rmse_psi[idx_num_samples, :]), color=handle1[0].get_color(),
                     style='--', label=mc_label)

    plt.xlabel(r'$\xi$ [dB]')
    plt.ylabel('RMSE [deg]')
    plt.title('Interferometer DF Performance')
    plt.legend(loc='lower left')
