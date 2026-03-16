import numpy as np

from ewgeo.utils.constants import speed_of_light
from ewgeo.utils.unit_conversions import db_to_lin


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
    c = speed_of_light
    sample_vec = np.arange(num_samples)
    sx_dot = (ts * sample_vec + (radius / c) * np.cos(2 * np.pi * fr * ts * sample_vec - psi))   # Eq 7.72
    sx_ddot = 2 * np.pi * f * (radius / c) * np.sin(2 * np.pi * fr * ts * sample_vec - psi)          # Eq 7.72
    
    # Build Jacobian of test (doppler) signal
    es_dot = np.sum(sx_dot).item()               # Eq 7.75
    es_ddot = np.sum(sx_ddot).item()             # Eq 7.76
    es_dot_dot = np.sum(sx_dot**2).item()        # Eq 7.77
    es_ddot_ddot = np.sum(sx_ddot**2).item()     # Eq 7.78
    es_dot_ddot = np.sum(sx_dot*sx_ddot).item()  # Eq 7.79
    
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

    # Constants and preprocessing
    c = speed_of_light
    y = x * np.conjugate(r)  # filter reference data to find signal
    phi = np.unwrap(np.angle(y))

    # Generate test phase signal
    num_samples = np.size(phi)
    sample_vec = np.arange(num_samples)
    t_vec = ts*sample_vec

    def phi_0(psi_local):
        # psi_local: shape (N,)
        # returns: shape (num_samples, N)

        # Converts from angle of arrival (psi_local) to complex phase (phi_0) over sample interval (t_vec)
        return 2. * np.pi * f * radius / c * np.cos(2 * np.pi * fr * t_vec[:, np.newaxis] - psi_local[np.newaxis, :])

    # Initial search resolution
    this_psi_res = 1.0
    psi = 0.0

    # Ensure at least one loop
    psi_res = min(psi_res, this_psi_res)

    # Loop until desired DF resolution achieved
    while this_psi_res >= psi_res:
        # search vector
        psi_vec = np.arange(min_psi, max_psi + this_psi_res, this_psi_res)
        phi0_mat = phi_0(psi_vec)  # shape: (num_samples, len(psi_vec))

        # Compute error at each test point in search vector and find minimum
        err = np.sum(np.absolute(phi[:, np.newaxis] - phi0_mat)**2, axis=0)
        idx_opt = np.argmin(err)
        psi = psi_vec[idx_opt]

        # Refine search bounds for next iteration
        this_psi_res /= 10
        idx_min = max(0, idx_opt - 4)
        idx_max = min(len(psi_vec) - 1, idx_opt + 4)
        min_psi = psi_vec[idx_min]
        max_psi = psi_vec[idx_max]

    return psi

    
