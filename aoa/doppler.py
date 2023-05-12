import numpy as np
import matplotlib.pyplot as plt
from ..utils.unit_conversions import db_to_lin
from ..utils import constants


def crlb(snr, num_samples, amplitude, ts, f, radius, fr, psi):
    """
    Compute the lower bound on unbiased estimation error for a Doppler-based direction finding receiver.
    
    Ported from MATLAB code.
    
    Nicholas O'Donoughue
    10 January 2021
    
    :param snr:  Signal to Noise ratio [dB]
    :param num_samples: Number of samples
    :param amplitude: Received signal amplitude
    :param ts: Sampling period [s]
    :param f: Carrier frequency [Hz]
    :param radius: Doppler rotation radius [m]
    :param fr: Doppler rotation rate [rotations/sec]
    :param psi: True angle of arrival [radians]
    :return crlb: Lower bound on the Mean Squared Error of an unbiased estimation of psi (radians)
    """

    # Recursion to handle SNR array inputs
    if np.size(snr) > 1:
        return np.asarray([crlb(this_snr, num_samples, amplitude, ts, f, radius, fr, psi) for this_snr in snr])

    # Build Jacobian of reference signal
    snr_lin = db_to_lin(snr)
    jr = 2. * num_samples * snr_lin \
            * np.array([[1. / amplitude ** 2, 0., 0., 0.],
                        [0., 1., ts * (num_samples - 1) / 2., 0.],
                        [0, ts * (num_samples - 1.) / 2., ts ** 2 * (num_samples - 1.) * (2. * num_samples - 1.) / 6.,
                         0.],
                        [0., 0., 0., 0.]])  # Eq 7.67
    
    # Build signal vectors and gradients
    c = constants.speed_of_light
    sample_vec = np.arange(num_samples)
    sx_dot = (ts * sample_vec + (radius / c) * np.cos(2 * np.pi * fr * ts * sample_vec - psi))   # Eq 7.72
    sx_ddot = 2 * np.pi * f * (radius / c) * np.sin(2 * np.pi * fr * ts * sample_vec - psi)          # Eq 7.72
    
    # Build Jacobian of test (doppler) signal
    es_dot = np.sum(sx_dot)                     # Eq 7.75
    es_ddot = np.sum(sx_ddot)                   # Eq 7.76
    es_dot_dot = np.sum(sx_dot**2)              # Eq 7.77
    es_ddot_ddot = np.sum(sx_ddot**2)           # Eq 7.78
    es_dot_ddot = np.sum(sx_dot*sx_ddot)        # Eq 7.79
    
    jx = 2 * num_samples * snr_lin \
           * np.array([[1. / amplitude ** 2, 0., 0., 0.],
                       [0., 1., es_dot / num_samples, es_ddot / num_samples],
                       [0., es_dot / num_samples, es_dot_dot / num_samples, es_dot_ddot / num_samples],
                       [0., es_ddot / num_samples, es_dot_ddot / num_samples, es_ddot_ddot / num_samples]])  # Eq 7.80
    
    # Construct full jacobian and invert
    j_full = jr + jx
    cov_full = np.linalg.pinv(j_full)
    return cov_full[-1, -1]  # output in radians -- angle is in the final dimension, based on how the jacobian was built

    
def compute_df(r, x, ts, f, radius, fr, psi_res, min_psi, max_psi):
    """
    Compute the estimate of a signal's angle of arrival (in radians), given a
    reference signal, and Doppler signal (from an antenna rotating about the
    reference antenna).

    Ported from MATLAB code.

    Nicholas O'Donoughue
    10 January 2021

    :param r: Reference signal, length M
    :param x: Doppler test signal, length M
    :param ts: Sampling period [s]
    :param f: Carrier frequency [Hz]
    :param radius: Doppler rotation radius [m]
    :param fr: Doppler rotation rate [rotations/sec]
    :param psi_res: Desired output AoA resolution [radians]
    :param min_psi: Minimum bound on valid region for psi [default = -np.pi]
    :param max_psi: Maximum bound on valid region for psi [default = np.pi]
    :return psi: Estimated angle of arrival [radians]
    """

    # Speed of light
    c = constants.speed_of_light

    # Filter the reference signal out of the test signal
    y = x * np.conjugate(r)

    # Compute (and unwrap) the complex phase angle
    phi = np.unwrap(np.arctan2(np.imag(y), np.real(y)))

    # Generate test phase signal
    num_samples = np.size(phi)
    sample_vec = np.arange(num_samples)
    t_vec = np.expand_dims(ts*sample_vec, axis=1)

    def phi_0(psi_local):
        # Converts from angle of arrival (psi_local) to complex phase (phi_0) over sample interval (t_vec)
        return 2. * np.pi * f * radius / c * np.cos(2 * np.pi * fr * t_vec - psi_local.T)

    # Initialize DF search loop
    this_psi_res = 1  # Start at 1 radian resolution
    psi_vec = np.expand_dims(np.arange(start=min_psi, stop=max_psi+this_psi_res, step=this_psi_res), axis=1)
    psi = 0.          # Initialize output

    # Make sure the loop happens at least once
    if psi_res > this_psi_res:
        psi_res = this_psi_res

    # Loop until desired DF resolution achieved
    while this_psi_res >= psi_res:
        # Compute error at each test point in search vector
        err = np.sum((np.expand_dims(phi, axis=1)-phi_0(psi_vec))**2, axis=0)

        # Find point with minimal error, use as center point for next
        # iteration of search
        idx_opt = np.argmin(err)
        psi = psi_vec[idx_opt]

        # Increase resolution (decrease search vector step size)
        # and interval create search vector for next iteration
        this_psi_res = this_psi_res/10
        min_psi = psi_vec[np.maximum(0, idx_opt-2)]
        max_psi = psi_vec[np.minimum(np.size(psi_vec)-1, idx_opt+2)]
        psi_vec = np.arange(start=min_psi, stop=max_psi+this_psi_res, step=this_psi_res)

    return psi

    
def run_example():
    """
    Example script to demonstrate analysis of a Doppler DF receiver.

    Ported from MATLAB code.

    Nicholas O'Donoughue
    14 January 2021

    :return: None
    """

    # Generate the Signals
    th_true = 45.
    psi_true = np.deg2rad(th_true)
    amplitude = 1.
    phi0 = 2*np.pi*np.random.uniform()   # Random starting phase
    f = 1.e9          # Carrier Frequency [Hz]
    ts = 1./(5.*f)    # Sampling rate

    # Doppler antenna parameters
    c = constants.speed_of_light      # speed of light [m/s]
    lam = c/f                         # wavelength [m]
    radius = lam/2.                   # Doppler radius (chosen to be half-wavelength) [m]
    psi_res = .0001                   # Desired DF resolution [rad]

    # Set up the parameter sweep
    num_samples_vec = np.asarray([10, 100, 1000])   # Number of temporal samples at each antenna test point
    snr_db_vec = np.arange(start=-10, step=2, stop=22)   # signal to noise ratio
    num_mc = 1.e6                       # number of monte carlo trials at each parameter setting

    # Set up output variables
    out_shp = (np.size(num_samples_vec), np.size(snr_db_vec))
    rmse_psi = np.zeros(shape=out_shp)
    crlb_psi = np.zeros(shape=out_shp)

    # Loop over parameters
    print('Executing Doppler Monte Carlo sweep...')
    for this_num_samples, idx_num_samples in enumerate(num_samples_vec):
        this_num_mc = num_mc / this_num_samples
        print('\t M={}'.format(this_num_samples))

        # Reference signal
        sample_vec = np.arange(this_num_samples)
        t_vec = ts*sample_vec
        r0 = amplitude*np.exp(1j*phi0)*np.exp(1j*2*np.pi*f*t_vec)

        # Doppler signal
        fr = 1/(ts*this_num_samples)            # Ensure a np.single cycle during M
        x0 = amplitude * np.exp(1j*phi0)*np.exp(1j*2*np.pi*f*t_vec)\
                       * np.exp(1j*2*np.pi*f*radius/c*np.cos(2*np.pi*fr*t_vec-psi_true))

        # Generate noise signal
        noise_amp = np.sqrt(np.sum(abs(r0)**2)/(this_num_samples*2))
        noise_base_r = [np.random.normal(loc=0.0, scale=noise_amp, size=(this_num_samples, 2)).view(np.complex128)
                        for _ in np.arange(this_num_mc)]
        noise_base_x = [np.random.normal(loc=0.0, scale=noise_amp, size=(this_num_samples, 2)).view(np.complex128)
                        for _ in np.arange(this_num_mc)]

        # Loop over SNR levels
        for snr_db, idx_snr in snr_db_vec:
            if np.mod(idx_snr, 10) == 0:
                print('.', end='')

            # Compute noise power, scale base noise
            noise_amp = db_to_lin(-snr_db/2)

            # Generate noisy signals
            r = [r0 + this_noise_r*noise_amp for this_noise_r in noise_base_r]
            x = [x0 + this_noise_x*noise_amp for this_noise_x in noise_base_x]

            # Compute the estimate
            psi_est = np.asarray([compute_df(this_r, this_x, ts, f, radius, fr, psi_res, -np.pi, np.pi)
                                  for (this_r, this_x) in zip(r, x)])

            # Compute RMS Error
            rmse_psi[idx_num_samples, idx_snr] = np.sqrt(np.mean((psi_est - psi_true) ** 2))

            # Compute CRLB for RMS Error
            crlb_psi[idx_num_samples, idx_snr] = crlb(snr_db, this_num_samples, amplitude,
                                                      ts, f, radius, fr, psi_true)

    print('done.')

    for this_num_samples, idx_num_samples in enumerate(num_samples_vec):
        crlb_label = 'CRLB, M={}'.format(this_num_samples)
        mc_label = 'Simulation Result, M={}'.format(this_num_samples)

        # Plot the MC and CRLB results for this number of samples
        handle1 = plt.semilogy(snr_db_vec, np.rad2deg(np.sqrt(crlb_psi[idx_num_samples, :])), label=crlb_label)
        plt.semilogy(snr_db_vec, np.rad2deg(rmse_psi[idx_num_samples, :]), color=handle1[0].get_color(),
                     style='--', label=mc_label)

    plt.xlabel(r'$\xi$ [dB]')
    plt.ylabel('RMSE [deg]')
    plt.title('Doppler DF Performance')
    plt.legend(loc='lower left')
