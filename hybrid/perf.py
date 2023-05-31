import numpy as np

from .. import utils
from . import model


def compute_crlb(x_aoa, x_tdoa, x_fdoa, v_fdoa, x_source, cov, tdoa_ref_idx=None,
                 fdoa_ref_idx=None, do_resample=True, cov_is_inverted=False):
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
    else:
        # Resample the covariance matrix, if necessary
        if do_resample:
            # TODO: Test matrix resampling
            num_aoa_sensors = num_tdoa_sensors = num_fdoa_sensors = 0

            # First, we generate the test and reference index vectors
            test_vec_to_concat = []
            ref_vec_to_concat = []
            if x_aoa is not None:
                _, num_aoa_sensors = utils.safe_2d_shape(x_aoa)
                test_idx_vec_aoa = np.arange(num_aoa_sensors)
                ref_idx_vec_aoa = np.nan * np.ones((num_aoa_sensors,))
                test_vec_to_concat.append(
                    test_idx_vec_aoa)
                ref_vec_to_concat.append(
                    ref_idx_vec_aoa)
            if x_tdoa is not None:
                _, num_tdoa_sensors = utils.safe_2d_shape(x_tdoa)
                test_idx_vec_tdoa, ref_idx_vec_tdoa = utils.parse_reference_sensor(tdoa_ref_idx, num_tdoa_sensors)
                test_vec_to_concat.append(
                    num_aoa_sensors + test_idx_vec_tdoa)
                ref_vec_to_concat.append(
                    num_aoa_sensors + ref_idx_vec_tdoa)
            if x_fdoa is not None:
                _, num_fdoa_sensors = utils.safe_2d_shape(x_fdoa)
                test_idx_vec_fdoa, ref_idx_vec_fdoa = utils.parse_reference_sensor(fdoa_ref_idx, num_fdoa_sensors)
                test_vec_to_concat.append(
                    num_aoa_sensors + num_fdoa_sensors + test_idx_vec_fdoa)
                ref_vec_to_concat.append(
                    num_aoa_sensors + num_fdoa_sensors + ref_idx_vec_fdoa)
            # Second, we assemble them into a single vector
            test_idx_vec = np.concatenate(
                test_vec_to_concat, 
                axis=0)
            ref_idx_vec = np.concatenate(
                ref_vec_to_concat,
                axis=0)
            # Finally, we resample the full covariance matrix using the assembled indices
            cov = utils.resample_covariance_matrix(cov, test_idx_vec, ref_idx_vec)

        # Invert the covariance matrix
        cov_inv = np.linalg.pinv(cov)

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
        fisher_matrix = this_jacobian.dot(cov_inv.dot(np.conjugate(this_jacobian.T)))

        if np.any(np.isnan(fisher_matrix)) or np.any(np.isinf(fisher_matrix)):
            # Problem is ill-defined, Fisher Information Matrix cannot be
            # inverted
            crlb[:, :, idx] = np.NaN
        else:
            crlb[:, :, idx] = np.linalg.pinv(fisher_matrix)

    return crlb
