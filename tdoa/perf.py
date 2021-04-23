import numpy as np
import utils
from . import model


def compute_crlb(x_tdoa, xs, cov, ref_idx=None):
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
    :return crlb: Lower bound on the error covariance matrix for an unbiased FDOA estimator (Ndim x Ndim)
    """

    # Parse inputs
    n_dim, n_sensor = np.size(x_tdoa)
    _, n_source = np.size(xs)

    # Resample the covariance matrix
    test_idx_vec, ref_idx_vec = utils.parse_reference_sensor(ref_idx, n_sensor)

    # Resample the covariance matrix
    cov_resample = utils.resample_covariance_matrix(cov, test_idx_vec, ref_idx_vec)
    cov_inv = np.linalg.pinv(cov_resample)

    # Initialize output variable
    crlb = np.zeros((n_dim, n_dim, n_source))

    # Repeat CRLB for each of the n_source test positions
    for idx in np.arange(n_source):
        this_x = xs[:, idx]

        # Evaluate the Jacobian
        this_jacobian = model.jacobian(x_tdoa, this_x, ref_idx)

        # Compute the Fisher Information Matrix
        fisher_matrix = this_jacobian.dot(cov_inv.dot(this_jacobian.H))

        if np.any(np.isnan(fisher_matrix)) or np.any(np.isinf(fisher_matrix)):
            # Problem is ill defined, Fisher Information Matrix cannot be
            # inverted
            crlb[:, :, idx] = np.NaN
        else:
            crlb[:, :, idx] = np.linalg.pinv(fisher_matrix)

    return crlb
