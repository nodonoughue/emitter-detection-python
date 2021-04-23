import numpy as np
import utils
from . import model
import scipy


def compute_crlb(x_aoa, x_tdoa, x_fdoa, v_fdoa, x_source, cov_aoa, cov_tdoa, cov_fdoa, tdoa_ref_idx=None,
                 fdoa_ref_idx=None):
    """
    Computes the CRLB on position accuracy for source at location xs and
    a combined set of AOA, TDOA, and FDOA measurements.  The covariance
    matrix C dictates the combined variances across the three measurement
    types.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    10 March 2021

    :param x_aoa: nDim x nAOA array of sensor positions
    :param x_tdoa: nDim x nTDOA array of TDOA sensor positions
    :param x_fdoa: nDim x nFDOA array of FDOA sensor positions
    :param v_fdoa: nDim x nFDOA array of FDOA sensor velocities
    :param x_source: Candidate source positions
    :param cov_aoa: AOA measurement error covariance matrix
    :param cov_tdoa: TDOA measurement error covariance matrix
    :param cov_fdoa: FDOA measurement error covariance matrix
    :param tdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for TDOA
    :param fdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for FDOA
    :return crlb: Lower bound on the error covariance matrix for an unbiased AOA/TDOA/FDOA estimator (Ndim x Ndim)
    """

    n_dim, n_source = np.shape(x_source)

    # Parse the TDOA and FDOA sensor pairs
    _, n_tdoa = np.shape(x_tdoa)
    _, n_fdoa = np.shape(x_fdoa)
    tdoa_test_idx_vec, tdoa_ref_idx_vec = utils.parse_reference_sensor(tdoa_ref_idx, n_tdoa)
    fdoa_test_idx_vec, fdoa_ref_idx_vec = utils.parse_reference_sensor(fdoa_ref_idx, n_fdoa)

    # Resample the covariance matrices
    cov_tdoa_resample = utils.resample_covariance_matrix(cov_tdoa, tdoa_test_idx_vec, tdoa_ref_idx_vec)
    cov_fdoa_resample = utils.resample_covariance_matrix(cov_fdoa, fdoa_test_idx_vec, fdoa_ref_idx_vec)

    # Pre-compute covariance matrix inverses
    cov_aoa_inv = np.linalg.pinv(cov_aoa)
    cov_tdoa_inv = np.linalg.pinv(cov_tdoa_resample)
    cov_fdoa_inv = np.linalg.pinv(cov_fdoa_resample)

    # Assemble into a single covariance matrix matching the measurement vector zeta
    cov_inv = scipy.linalg.blkdiag(cov_aoa_inv, cov_tdoa_inv, cov_fdoa_inv)

    # Initialize output variable
    crlb = np.zeros((n_dim, n_dim, n_source))

    # Repeat CRLB for each of the n_source test positions
    for idx in np.arange(n_source):
        this_x = x_source[:, idx]

        # Evaluate the Jacobian
        this_jacobian = model.jacobian(x_aoa, x_tdoa, x_fdoa, v_fdoa, this_x, tdoa_ref_idx, fdoa_ref_idx)

        # Compute the Fisher Information Matrix
        fisher_matrix = this_jacobian.dot(cov_inv.dot(this_jacobian.H))

        if np.any(np.isnan(fisher_matrix)) or np.any(np.isinf(fisher_matrix)):
            # Problem is ill defined, Fisher Information Matrix cannot be
            # inverted
            crlb[:, :, idx] = np.NaN
        else:
            crlb[:, :, idx] = np.linalg.pinv(fisher_matrix)

    return crlb
