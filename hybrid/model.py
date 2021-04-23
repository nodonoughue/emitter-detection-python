import numpy as np
import triang
import tdoa
import fdoa
import utils
import scipy


def measurement(x_aoa, x_tdoa, x_fdoa, v_fdoa, x_source, tdoa_ref_idx=None, fdoa_ref_idx=None):
    """
    Computes hybrid measurements, for AOA, TDOA, and FDOA sensors.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    10 March 2021

    :param x_aoa: nDim x nAOA array of sensor positions
    :param x_tdoa: nDim x nTDOA array of TDOA sensor positions
    :param x_fdoa: nDim x nFDOA array of FDOA sensor positions
    :param v_fdoa: nDim x nFDOA array of FDOA sensor velocities
    :param x_source: nDim x n_source array of source positions
    :param tdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for TDOA
    :param fdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for FDOA
    :return zeta: nAoa + nTDOA + nFDOA - 2 x nSource array of measurements
    """

    # Construct component measurements
    z_a = triang.model.measurement(x_aoa, x_source)
    z_t = tdoa.model.measurement(x_tdoa, x_source,tdoa_ref_idx)
    z_f = fdoa.model.measurement(x_fdoa, v_fdoa, x_source,fdoa_ref_idx)

    # Combine into a single data vector
    z = np.cat(z_a, z_t, z_f, axis=0)

    return z


def jacobian(x_aoa, x_tdoa, x_fdoa, v_fdoa, x_source, tdoa_ref_idx=None, fdoa_ref_idx=None):
    """
    # Returns the Jacobian matrix for hybrid set of AOA, TDOA, and FDOA
    # measurements.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    10 March 2021

    :param x_aoa: nDim x nAOA array of sensor positions
    :param x_tdoa: nDim x nTDOA array of TDOA sensor positions
    :param x_fdoa: nDim x nFDOA array of FDOA sensor positions
    :param v_fdoa: nDim x nFDOA array of FDOA sensor velocities
    :param x_source: nDim x n_source array of source positions
    :param tdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for TDOA
    :param fdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for FDOA
    :return: n_dim x nMeasurement x n_source matrix of Jacobians, one for each candidate source position
    """

    # Compute Jacobian for AOA measurements
    if x_aoa:
        j_aoa = triang.jacobian(x_aoa, x_source)
    else:
        j_aoa = None


    # Compute Jacobian for TDOA measurements
    if x_tdoa:
        j_tdoa= tdoa.jacobian(x_tdoa, x_source, tdoa_ref_idx)
    else:
        j_tdoa = None

    # Compute Jacobian for FDOA measurements
    if x_fdoa and v_fdoa:
        j_fdoa= fdoa.jacobian(x_fdoa, v_fdoa, x_source, fdoa_ref_idx)
    else:
        j_fdoa = None

    # Combine component Jacobians
    return np.horzcat(j_aoa, j_tdoa, j_fdoa)


def log_likelihood(x_aoa, x_tdoa, x_fdoa, v_fdoa, zeta, cov_aoa, cov_tdoa, cov_fdoa, x_source, tdoa_ref_idx=None,
                   fdoa_ref_idx=None):
    """
    Computes the Log Likelihood for Hybrid sensor measurement (AOA, TDOA, and
    FDOA), given the received measurement vector zeta, covariance matrix C,
    and set of candidate source positions x_source.

    Ported from MATLAB Code.

    Nicholas O'Donoughue
    10 March 2021

    :param x_aoa: nDim x nAOA array of sensor positions
    :param x_tdoa: nDim x nTDOA array of TDOA sensor positions
    :param x_fdoa: nDim x nFDOA array of FDOA sensor positions
    :param v_fdoa: nDim x nFDOA array of FDOA sensor velocities
    :param zeta: Combined AOA/TDOA/FDOA measurement vector
    :param cov_aoa: AOA measurement error covariance matrix
    :param cov_tdoa: TDOA measurement error covariance matrix
    :param cov_fdoa: FDOA measurement error covariance matrix
    :param x_source: Candidate source positions
    :param tdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for TDOA
    :param fdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for FDOA
    :return ell: Log-likelihood evaluated at each position x_source.
    """

    n_dim, n_source_pos = np.shape(x_source)
    ell = np.zeros((n_source_pos,1))

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

    for idx_source in np.arange(n_source_pos):
        x_i = x_source[:, idx_source]

        # Generate the ideal measurement matrix for this position
        zeta_dot = measurement(x_aoa, x_tdoa, x_fdoa, v_fdoa, x_i, tdoa_ref_idx, fdoa_ref_idx)

        # Evaluate the measurement error
        err = (zeta_dot - zeta)

        # Compute the scaled log likelihood
        ell[idx_source] = -err.dot(cov_inv).dot(err)
