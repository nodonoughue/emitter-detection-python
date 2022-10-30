import numpy as np
import utils
from . import model


def compute_crlb(x_tdoa, xs, cov, ref_idx=None, do_resample=True):
    """
    Computes the CRLB on position accuracy for source at location xs and
    sensors at locations in x_tdoa (Ndim x N).

    Ported from MATLAB Code

    Nicholas O'Donoughue
    21 February 2021

    :param x_tdoa: (Ndim x N) array of TDOA sensor positions
    :param xs: (Ndim x M) array of source positions over which to calculate CRLB
    :param cov: Covariance matrix for range rate estimates at the N FDOA sensors [(m/s)^2]
    :param ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings
    :param do_resample: Boolean flag; if true the covariance matrix will be resampled, using ref_idx
    :return crlb: Lower bound on the error covariance matrix for an unbiased FDOA estimator (Ndim x Ndim)
    """

    # Parse inputs
    n_dim, n_sensor = np.shape(x_tdoa)
    _, n_source = utils.safe_2d_shape(xs)

    # Make sure that xs is 2D
    if n_source == 1:
        xs = xs[:, np.newaxis]

    # Resample the covariance matrix
    if do_resample:
        # Resample the covariance matrix
        test_idx_vec, ref_idx_vec = utils.parse_reference_sensor(ref_idx, n_sensor)
        cov_resample = utils.resample_covariance_matrix(cov, test_idx_vec, ref_idx_vec)
        cov_inv = np.linalg.inv(cov_resample)
    else:
        cov_inv = np.linalg.inv(cov)

    # Initialize output variable
    crlb = np.zeros((n_dim, n_dim, n_source))

    # Repeat CRLB for each of the n_source test positions
    for idx in np.arange(n_source):
        this_x = xs[:, idx]

        # Evaluate the Jacobian
        this_jacobian = model.jacobian(x_tdoa, this_x, ref_idx)

        # Compute the Fisher Information Matrix
        fisher_matrix = this_jacobian.dot(cov_inv.dot(np.conj(np.transpose(this_jacobian))))

        if np.any(np.isnan(fisher_matrix)) or np.any(np.isinf(fisher_matrix)):
            # Problem is ill-defined, Fisher Information Matrix cannot be
            # inverted
            crlb[:, :, idx] = np.NaN
        else:
            crlb[:, :, idx] = np.linalg.inv(fisher_matrix)

    return crlb
