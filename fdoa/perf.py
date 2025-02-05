import numpy as np
import utils
from utils.unit_conversions import db_to_lin
from . import model
from scipy.linalg import solve_triangular, pinvh


def compute_crlb(x_sensor, v_sensor, x_source, cov, ref_idx=None, do_resample=True, cov_is_inverted=False):
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
    :param cov: Covariance matrix for range rate estimates at the N FDOA sensors [(m/s)^2]
    :param ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings
    :param do_resample: Boolean flag; if true the covariance matrix will be resampled, using ref_idx
    :param cov_is_inverted: Boolean flag, if false then cov is the covariance matrix. If true, then it is the
                            inverse of the covariance matrix.
    :return crlb: Lower bound on the error covariance matrix for an unbiased FDOA estimator (Ndim x Ndim)
    """

    # Parse inputs
    n_dim, n_sensor = np.shape(x_sensor)
    _, n_source = utils.safe_2d_shape(x_source)

    # Make sure that xs is 2D
    if n_source == 1:
        x_source = x_source[:, np.newaxis]

    # Pre-process the covariance matrix
    if cov_is_inverted:
        # The covariance matrix was pre-inverted, use it directly
        cov_inv = cov
        cov_lower = None  # pre-define to avoid a 'use before defined' error
    else:
        # The covariance matrix was not pre-inverted, resample if necessary and then use
        # cholesky decomposition to improve stability and speed for repeated calculation of
        # the Fisher Information Matrix
        if do_resample:
            # Resample the covariance matrix
            cov = utils.resample_covariance_matrix(cov, ref_idx)
            cov = utils.ensure_invertible(cov)

        # Pre-compute the matrix inverse, to speed up repeated calls
        cov_lower = np.linalg.cholesky(cov)
        cov_inv = None  # pre-define to avoid a 'use before defined' error

    # Initialize output variable
    crlb = np.zeros((n_dim, n_dim, n_source))

    # Repeat CRLB for each of the n_source test positions
    for idx in np.arange(n_source):
        this_x = x_source[:, idx]

        # Evaluate the Jacobian - n_dim x n_dim x n_source
        this_jacobian = model.jacobian(x_sensor=x_sensor, v_sensor=v_sensor,
                                       x_source=this_x, v_source=None,
                                       ref_idx=ref_idx)

        # Compute the Fisher Information Matrix
        if cov_is_inverted:
            fisher_matrix = this_jacobian.dot(cov_inv.dot(np.conjugate(this_jacobian.T)))
        else:
            # Use cholesky decomposition
            cov_jacob = solve_triangular(cov_lower, np.conj(np.transpose(this_jacobian)), lower=True)
            fisher_matrix = cov_jacob.T @ cov_jacob

        if np.any(np.isnan(fisher_matrix)) or np.any(np.isinf(fisher_matrix)):
            # Problem is ill-defined, Fisher Information Matrix cannot be
            # inverted
            crlb[:, :, idx] = np.nan
        else:
            # crlb[:, :, idx] = np.linalg.pinv(fisher_matrix)
            crlb[:, :, idx] = np.real(pinvh(fisher_matrix))

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
