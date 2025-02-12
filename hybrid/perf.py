import numpy as np
import hybrid.model
import utils
from . import model
from scipy.linalg import solve_triangular, pinvh


def compute_crlb(x_aoa, x_tdoa, x_fdoa, v_fdoa, x_source, cov, tdoa_ref_idx=None,
                 fdoa_ref_idx=None, do_resample=False, cov_is_inverted=False):
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
    :param cov: Measurement error covariance matrix
    :param tdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for TDOA
    :param fdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for FDOA
    :param do_resample: Boolean flag; if true the covariance matrix will be resampled, using ref_idx
    :param cov_is_inverted: Boolean flag, if false then cov is the covariance matrix. If true, then it is the
                            inverse of the covariance matrix.
    :return crlb: Lower bound on the error covariance matrix for an unbiased AOA/TDOA/FDOA estimator (Ndim x Ndim)
    """

    n_dim, n_source = utils.safe_2d_shape(x_source)

    if n_source == 1:
        # Make sure it's got a second dimension, so that it doesn't fail when we iterate over source positions
        x_source = x_source[:, np.newaxis]

    # Pre-compute covariance matrix inverses
    if cov_is_inverted:
        # You can't resample a covariance matrix after inversion, so if it's already inverted, we assume it was
        # resampled, regardless of what the 'do_resample' flag says
        cov_inv = cov
        cov_lower = None  # pre-define to avoid a 'use before defined' error
    else:
        # Resample the covariance matrix, if necessary
        if do_resample:
            # Use the hybrid-specific covariance matrix resampler, which handles the assumed structure.
            cov = hybrid.model.resample_hybrid_covariance_matrix(cov=cov, x_aoa=x_aoa, x_tdoa=x_tdoa, x_fdoa=x_fdoa,
                                                                 tdoa_ref_idx=tdoa_ref_idx, fdoa_ref_idx=fdoa_ref_idx)
            cov = utils.ensure_invertible(cov)

        if np.isscalar(cov):
            # The covariance matrix is a scalar, this is easy, go ahead and invert it
            cov_inv = 1. / cov
            cov_lower = None
            cov_is_inverted = True
        else:
            # Use the Cholesky decomposition to speed things up
            cov_lower = np.linalg.cholesky(cov)
            cov_inv = None  # pre-define to avoid a 'use before defined' error

    # Initialize output variable
    crlb = np.zeros((n_dim, n_dim, n_source))

    # Repeat CRLB for each of the n_source test positions
    for idx in np.arange(n_source):
        this_x = x_source[:, idx]

        # Evaluate the Jacobian
        this_jacobian = model.jacobian(x_aoa=x_aoa, x_tdoa=x_tdoa,
                                       x_fdoa=x_fdoa, v_fdoa=v_fdoa,
                                       x_source=this_x,
                                       tdoa_ref_idx=tdoa_ref_idx, fdoa_ref_idx=fdoa_ref_idx)

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
    if n_source == 1:
        # There's only one source, trim the third dimension
        crlb = crlb[:, :, 0]

    return crlb
