import numpy as np
import utils
from utils.unit_conversions import db_to_lin
from . import model
from utils.covariance import CovarianceMatrix


def compute_crlb(x_sensor, v_sensor, x_source, cov: CovarianceMatrix, v_source=None, ref_idx=None, do_resample=True,
                 print_progress=False):
    """
    Computes the CRLB on position accuracy for source at location xs and
    sensors at locations in x_fdoa (Ndim x N) with velocity v_fdoa.
    C is a Nx1 vector of FOA variances at each of the N sensors, and ref_idx
    defines the reference sensor(s) used for FDOA.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    21 February 2021

    :param x_sensor: (Ndim x N) array of FDOA sensor positions
    :param v_sensor: (Ndim x N) array of FDOA sensor velocities
    :param x_source: (Ndim x M) array of source positions over which to calculate CRLB
    :param v_source: n_dim x n_source vector of source velocities
    :param cov: CovarianceMatrix object for range rate estimates
    :param ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings
    :param do_resample: Boolean flag; if true the covariance matrix will be resampled, using ref_idx
    :param print_progress: Boolean flag, if true then progress updates and elapsed/remaining time will be printed to
                           the console. [default=False]
    :return crlb: Lower bound on the error covariance matrix for an unbiased FDOA estimator (Ndim x Ndim)
    """

    # Parse inputs
    _, n_source = utils.safe_2d_shape(x_source)

    # Make sure that xs is 2D
    if n_source == 1:
        x_source = x_source[:, np.newaxis]

    if do_resample:
        cov = cov.resample(ref_idx)

    # Define a wrapper for the jacobian matrix that accepts only the position 'x'
    def jacobian(x):
        return model.jacobian(x_sensor=x_sensor, v_sensor=v_sensor,
                              x_source=x, v_source=v_source, ref_idx=ref_idx)

    crlb = utils.perf.compute_crlb_gaussian(x_source=x_source, jacobian=jacobian, cov=cov,
                                            print_progress=print_progress)

    return crlb


def freq_crlb(sample_time, num_samples, snr_db):
    """
    Compute the CRLB for the frequency difference estimate from a pair of
    sensors, given the time duration of the sampled signals, receiver
    bandwidth, and average SNR.

    Ported from MATLAB code.

    Nicholas O'Donoughue
    21 February 2021

    :param sample_time: Received signal duration [s]
    :param num_samples: Number of receiver samples [Hz]
    :param snr_db: SNR [dB]
    :return: Frequency difference estimate error standard deviation [Hz]
    """

    # Convert SNR to linear units
    snr_lin = db_to_lin(snr_db)

    # Compute the CRLB of the center frequency variance
    sigma = np.sqrt(3 / (np.pi**2 * sample_time**2 * num_samples * (num_samples**2 - 1) * snr_lin))

    return sigma


def freq_diff_crlb(time_s, bw_hz, snr_db):
    """
    Compute the CRLB for the frequency difference estimate from a pair of
    sensors, given the time duration of the sampled signals, receiver
    bandwidth, and average SNR.

    Ported from MATLAB code

    Nicholas O'Donoughue
    21 February 2021

    :param time_s: Received signal duration [s]
    :param bw_hz: Received signal bandwidth [Hz]
    :param snr_db: Average SNR [dB]
    :return sigma: Frequency difference estimate error standard deviation [Hz]
    """

    # Convert SNR to linear units
    snr_lin = db_to_lin(snr_db)

    # Apply the CRLB equations
    sigma = np.sqrt(3 / (4 * np.pi**2 * time_s**3 * bw_hz * snr_lin))

    return sigma
