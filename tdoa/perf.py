import numpy as np
import scipy
import utils
from . import model


def compute_crlb(x_sensor, x_source, cov, ref_idx=None, do_resample=True, variance_is_toa=True):
    """
    Computes the CRLB on position accuracy for source at location xs and
    sensors at locations in x_tdoa (Ndim x N).

    Ported from MATLAB Code

    Nicholas O'Donoughue
    21 February 2021

    :param x_sensor: (Ndim x N) array of TDOA sensor positions
    :param x_source: (Ndim x M) array of source positions over which to calculate CRLB
    :param cov: Covariance matrix for range rate estimates at the N FDOA sensors [(m/s)^2]
    :param ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings
    :param do_resample: Boolean flag; if true the covariance matrix will be resampled, using ref_idx
    :param variance_is_toa: Boolean flag; if true then the input covariance matrix is in units of s^2; if false, then
    it is in m^2
    :return crlb: Lower bound on the error covariance matrix for an unbiased FDOA estimator (Ndim x Ndim)
    """

    # Parse inputs
    n_dim, n_sensor = np.shape(x_sensor)
    _, n_source = utils.safe_2d_shape(x_source)

    # Make sure that xs is 2D
    if n_source == 1:
        x_source = x_source[:, np.newaxis]

    # Correct units for the covariance matrix
    if variance_is_toa:
        # Use the speed of light (squared) to covert from TOA or TDOA to ROA or RDOA
        cov = cov * utils.constants.speed_of_light**2

    # Resample the covariance matrix
    if do_resample:
        # Resample the covariance matrix (convert from ROA to RDOA)
        test_idx_vec, ref_idx_vec = utils.parse_reference_sensor(ref_idx, n_sensor)
        cov = utils.resample_covariance_matrix(cov, test_idx_vec, ref_idx_vec)

    # Pre-compute the matrix inverse, to speed up repeated calls
    # TODO: Use Cholesky decomposition to speed this up!
    # cov_inv = np.linalg.inv(cov)
    cov = utils.ensure_invertible(cov)
    cov_lower = np.linalg.cholesky(cov, upper=False)

    # Initialize output variable
    crlb = np.zeros((n_dim, n_dim, n_source))

    # Repeat CRLB for each of the n_source test positions
    for idx in np.arange(n_source):
        this_x = x_source[:, idx]

        # Evaluate the Jacobian
        this_jacobian = model.jacobian(x_sensor, this_x, ref_idx)

        # Compute the Fisher Information Matrix
        A = scipy.linalg.solve(cov_lower, np.conj(np.transpose(this_jacobian)))
        fisher_matrix = np.conj(np.transpose(A)) @ A

        if np.any(np.isnan(fisher_matrix)) or np.any(np.isinf(fisher_matrix)):
            # Problem is ill-defined, Fisher Information Matrix cannot be
            # inverted
            crlb[:, :, idx] = np.nan
        else:
            crlb[:, :, idx] = np.linalg.pinv(fisher_matrix)

    return crlb
