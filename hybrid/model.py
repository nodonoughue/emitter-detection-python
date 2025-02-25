import numpy as np
import triang
import tdoa
import fdoa
import utils
from scipy.linalg import solve_triangular


def measurement(x_source, x_aoa=None, x_tdoa=None, x_fdoa=None, v_fdoa=None, v_source=None,
                do_2d_aoa=False, tdoa_ref_idx=None, fdoa_ref_idx=None):
    """
    Computes hybrid measurements, for AOA, TDOA, and FDOA sensors.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    10 March 2021

    :param x_source: nDim x n_source array of source positions
    :param x_aoa: nDim x nAOA array of sensor positions
    :param x_tdoa: nDim x nTDOA array of TDOA sensor positions
    :param x_fdoa: nDim x nFDOA array of FDOA sensor positions
    :param v_fdoa: nDim x nFDOA array of FDOA sensor velocities
    :param v_source: nDim x 1 source velocity; assumed stationary if left blank
    :param do_2d_aoa: Optional boolean parameter specifying whether 1D (az-only) or 2D (az/el) AOA is being performed
    :param tdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for TDOA
    :param fdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for FDOA
    :return zeta: nAoa + nTDOA + nFDOA - 2 x nSource array of measurements
    """

    # Construct component measurements
    to_concat = []
    if x_aoa is not None:
        z_a = triang.model.measurement(x_sensor=x_aoa, x_source=x_source, do_2d_aoa=do_2d_aoa)
        to_concat.append(z_a)
    if x_tdoa is not None:
        z_t = tdoa.model.measurement(x_sensor=x_tdoa, x_source=x_source, ref_idx=tdoa_ref_idx)
        to_concat.append(z_t)
    if x_fdoa is not None:
        z_f = fdoa.model.measurement(x_sensor=x_fdoa, v_sensor=v_fdoa, 
                                     x_source=x_source, v_source=v_source,
                                     ref_idx=fdoa_ref_idx)
        to_concat.append(z_f)

    # Combine into a single data vector
    z = np.concatenate(to_concat, axis=0)

    return z


def jacobian(x_source, x_aoa=None, x_tdoa=None, x_fdoa=None, v_fdoa=None, v_source=None,
             do_2d_aoa=False, tdoa_ref_idx=None, fdoa_ref_idx=None):
    """
    # Returns the Jacobian matrix for hybrid set of AOA, TDOA, and FDOA
    # measurements.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    10 March 2021

    :param x_source: nDim x n_source array of source positions
    :param x_aoa: nDim x nAOA array of sensor positions
    :param x_tdoa: nDim x nTDOA array of TDOA sensor positions
    :param x_fdoa: nDim x nFDOA array of FDOA sensor positions
    :param v_fdoa: nDim x nFDOA array of FDOA sensor velocities
    :param v_source: nDim x 1 source velocity; assumed stationary if left blank
    :param do_2d_aoa: Optional boolean parameter specifying whether 1D (az-only) or 2D (az/el) AOA is being performed
    :param tdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for TDOA
    :param fdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for FDOA
    :return: n_dim x nMeasurement x n_source Jacobian matrices, one for each candidate source position
    """

    # Find out how many source points are requested
    n_dim, n_source = utils.safe_2d_shape(x_source)
    if n_source > 1:
        empty_dims = (n_dim, 0, n_source)
    else:
        empty_dims = (n_dim, 0)

    # Compute Jacobian for AOA measurements
    if x_aoa is not None:
        j_aoa = triang.model.jacobian(x_aoa, x_source, do_2d_aoa=do_2d_aoa)
    else:
        j_aoa = np.zeros(shape=empty_dims)

    # Compute Jacobian for TDOA measurements
    if x_tdoa is not None:
        j_tdoa = tdoa.model.jacobian(x_tdoa, x_source, tdoa_ref_idx)
    else:
        j_tdoa = np.zeros(shape=empty_dims)

    # Compute Jacobian for FDOA measurements
    if x_fdoa is not None and v_fdoa is not None:
        j_fdoa = fdoa.model.jacobian(x_sensor=x_fdoa, v_sensor=v_fdoa, x_source=x_source, v_source=v_source,
                                     ref_idx=fdoa_ref_idx)
    else:
        j_fdoa = np.zeros(shape=empty_dims)

    # Combine component Jacobian matrices
    return np.concatenate((j_aoa, j_tdoa, j_fdoa), axis=1)


def log_likelihood(x_source, zeta, cov, x_aoa=None, x_tdoa=None, x_fdoa=None, v_fdoa=None, v_source=None,
                   do_2d_aoa=False, tdoa_ref_idx=None, fdoa_ref_idx=None, do_resample=False, cov_is_inverted=False):
    """
    Computes the Log Likelihood for Hybrid sensor measurement (AOA, TDOA, and
    FDOA), given the received measurement vector zeta, covariance matrix C,
    and set of candidate source positions x_source.

    Ported from MATLAB Code.

    Nicholas O'Donoughue
    10 March 2021

    :param x_source: Candidate source positions
    :param zeta: Combined AOA/TDOA/FDOA measurement vector
    :param cov: Measurement covariance matrix (or its precomputed inverse)
    :param x_aoa: nDim x nAOA array of sensor positions
    :param x_tdoa: nDim x nTDOA array of TDOA sensor positions
    :param x_fdoa: nDim x nFDOA array of FDOA sensor positions
    :param v_fdoa: nDim x nFDOA array of FDOA sensor velocities
    :param v_source: nDim x 1 source velocity; assumed stationary if left blank
    :param do_2d_aoa: Optional boolean parameter specifying whether 1D (az-only) or 2D (az/el) AOA is being performed
    :param tdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for TDOA
    :param fdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for FDOA
    :param do_resample: Boolean flag; if true the covariance matrix will be resampled, using ref_idx
    :param cov_is_inverted: Boolean flag, if false then cov is the covariance matrix. If true, then it is the
                            inverse of the covariance matrix.
    :return ell: Log-likelihood evaluated at each position x_source.
    """

    n_dim, n_source_pos = utils.safe_2d_shape(x_source)
    _, n_source_pos2 = utils.safe_2d_shape(v_source)

    # Make the source position and velocity 2D, if they're not already
    if n_source_pos == 1:
        x_source = x_source[:, np.newaxis]
    if v_source is not None and n_source_pos2 == 1:
        v_source = v_source[:, np.newaxis]

    # Initialize the output variable
    ell = np.zeros((n_source_pos, ))

    # Pre-compute covariance matrix inverses
    if cov_is_inverted:
        cov_inv = cov
        cov_lower = None  # pre-define to avoid a 'use before defined' error
    else:
        if do_resample:
            cov = resample_hybrid_covariance_matrix(cov, x_aoa, x_tdoa, x_fdoa, do_2d_aoa, tdoa_ref_idx, fdoa_ref_idx)
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

    # Loop across source positions
    for idx_source in np.arange(n_source_pos):
        x_i = x_source[:, idx_source]
        if v_source is None:
            v_i = None
        else:
            v_i = v_source[:, idx_source]

        # Generate the ideal measurement matrix for this position
        zeta_dot = measurement(x_aoa=x_aoa, x_tdoa=x_tdoa, x_fdoa=x_fdoa, v_fdoa=v_fdoa, x_source=x_i,
                               v_source=v_i, do_2d_aoa=do_2d_aoa, tdoa_ref_idx=tdoa_ref_idx, fdoa_ref_idx=fdoa_ref_idx)

        # Evaluate the measurement error
        err = (zeta_dot - zeta)

        # Compute the scaled log likelihood
        if cov_is_inverted:
            if np.isscalar(cov_inv):
                ell[idx_source] = - cov_inv * (err ** 2)
            else:
                ell[idx_source] = - err.T @ cov_inv @ err
        else:
            # Use Cholesky decomposition
            cov_err = solve_triangular(cov_lower, err, lower=True)
            ell[idx_source] = - np.sum(cov_err**2)

    return ell


def error(x_source, cov, x_aoa=None, x_tdoa=None, x_fdoa=None, v_fdoa=None, v_source=None, x_max=1, num_pts=11,
          do_2d_aoa=False, tdoa_ref_idx=None, fdoa_ref_idx=None, do_resample=False, cov_is_inverted=False):
    """
    Construct a 2-D field from -x_max to +x_max, using numPts in each
    dimension.  For each point, compute the hybrid solution for each sensor
    against the reference (the first sensor), and compare to the FDOA
    solution from the true emitter position.

    Ported from MATLAB Code.

    Nicholas O'Donoughue
    21 January 2021

    :param x_source: nDim x 1 matrix of true emitter position
    :param cov: N x N covariance matrix
    :param x_aoa: nDim x nAOA array of sensor positions
    :param x_tdoa: nDim x nTDOA array of TDOA sensor positions
    :param x_fdoa: nDim x nFDOA array of FDOA sensor positions
    :param v_fdoa: nDim x N matrix of sensor velocities
    :param v_source: nDim x 1 matrix of true emitter velocity
    :param x_max: nDim x 1 (or scalar) vector of maximum offset from origin for plotting
    :param num_pts: Number of test points along each dimension
    :param do_2d_aoa: Optional boolean parameter specifying whether 1D (az-only) or 2D (az/el) AOA is being performed
    :param tdoa_ref_idx: Scalar index of reference sensor, or n_dim x n_pair matrix of sensor pairings for TDOA
    :param fdoa_ref_idx: Scalar index of reference sensor, or n_dim x n_pair matrix of sensor pairings for FDOA
    :param do_resample: Boolean flag; if true the covariance matrix will be resampled, using ref_idx
    :param cov_is_inverted: Boolean flag, if false then cov is the covariance matrix. If true, then it is the
                            inverse of the covariance matrix.
    :return epsilon: 2-D plot of FDOA error
    :return x_vec:
    :return y_vec:
    """

    # Compute the true measurement
    zeta = measurement(x_aoa=x_aoa, x_tdoa=x_tdoa, x_fdoa=x_fdoa, v_fdoa=v_fdoa, x_source=x_source, v_source=v_source,
                       do_2d_aoa=do_2d_aoa, tdoa_ref_idx=tdoa_ref_idx, fdoa_ref_idx=fdoa_ref_idx)

    # ToDo: Make a util function that pre-processes covariance matrices for fast inversion; put in a cov matrix object?
    # Resample the covariance matrix
    if cov_is_inverted:
        cov_inv = cov
        cov_lower = None  # pre-define to avoid a 'use before defined' error
    else:
        # The covariance matrix was not pre-inverted, resample if necessary and then use
        # cholesky decomposition to improve stability and speed for repeated calculation of
        # the Fisher Information Matrix
        if do_resample:
            # Resample the covariance matrix
            cov = resample_hybrid_covariance_matrix(cov, x_aoa, x_tdoa, x_fdoa, do_2d_aoa, tdoa_ref_idx, fdoa_ref_idx)
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

    # Set up test points
    grid_res = 2*x_max / (num_pts-1)
    x_set, x_grid, grid_shape = utils.make_nd_grid(x_ctr=x_source, max_offset=x_max, grid_spacing=grid_res)
    x_vec = x_grid[0][0, :]
    y_vec = x_grid[1][:, 0]

    zeta_list = measurement(x_aoa=x_aoa, x_tdoa=x_tdoa, x_fdoa=x_fdoa, v_fdoa=v_fdoa,
                            x_source=x_set, v_source=v_source, tdoa_ref_idx=tdoa_ref_idx, fdoa_ref_idx=fdoa_ref_idx)

    err = zeta[:, np.newaxis] - zeta_list

    # Angle measurements require a modulo 2*pi operation to avoid edge effects
    if x_aoa is not None:
        for idx_aoa in np.arange(utils.safe_2d_shape(x_aoa)[1]):
            err[idx_aoa, :] = utils.modulo2pi(err[idx_aoa, :])

    if cov_is_inverted:
        epsilon_list = [np.conjugate(this_err).T @  cov_inv @ this_err for this_err in err.T]
    else:
        def compute_epsilon(this_err):
            cov_err = solve_triangular(cov_lower, this_err, lower=True)
            return np.sum(cov_err**2)

        epsilon_list = [compute_epsilon(this_err) for this_err in err.T]

    return np.reshape(epsilon_list, grid_shape), x_vec, y_vec


def resample_hybrid_covariance_matrix(cov, x_aoa=None, x_tdoa=None, x_fdoa=None,
                                      do_2d_aoa=False, tdoa_ref_idx=None, fdoa_ref_idx=None):
    """
    Resample a block-diagonal covariance matrix representing AOA, TDOA, and FDOA measurements errors. Original matrix
    size is square with (num_aoa*aoa_dim + num_tdoa + num_fdoa) rows/columns. Output matrix size will be square with
    (num_aoa*aoa_dim + num_tdoa_sensor_pairs + num_fdoa_sensor_pairs) rows/columns.

    The covariance matrix is assumed to have AOA measurements first, then TDOA, then FDOA.

    Ported from MATLAB Code.

    Nicholas O'Donoughue
    21 January 2021

    :param cov: N x N covariance matrix
    :param x_aoa: nDim x nAOA array of sensor positions
    :param x_tdoa: nDim x nTDOA array of TDOA sensor positions
    :param x_fdoa: nDim x nFDOA array of FDOA sensor positions
    :param do_2d_aoa: Optional boolean parameter specifying whether 1D (az-only) or 2D (az/el) AOA is being performed
    :param tdoa_ref_idx: Scalar index of reference sensor, or n_dim x n_pair matrix of sensor pairings for TDOA
    :param fdoa_ref_idx: Scalar index of reference sensor, or n_dim x n_pair matrix of sensor pairings for FDOA
    :return cov_out:
    """

    # Calculate the Number of Sensors in each Data Type
    n_dim1, num_aoa_sensors = utils.safe_2d_shape(x_aoa)
    if do_2d_aoa:
        num_aoa_sensors = 2 * num_aoa_sensors  # double it to account for 2D measurements

    n_dim2, num_tdoa_sensors = utils.safe_2d_shape(x_tdoa)
    n_dim3, num_fdoa_sensors = utils.safe_2d_shape(x_fdoa)

    # First, we generate the test and reference index vectors
    test_idx_vec_aoa = np.arange(num_aoa_sensors)
    ref_idx_vec_aoa = np.nan * np.ones((num_aoa_sensors,))
    test_idx_vec_tdoa, ref_idx_vec_tdoa = utils.parse_reference_sensor(tdoa_ref_idx, num_tdoa_sensors)
    test_idx_vec_fdoa, ref_idx_vec_fdoa = utils.parse_reference_sensor(fdoa_ref_idx, num_fdoa_sensors)

    # Second, we assemble them into a single vector
    test_idx_vec = np.concatenate((test_idx_vec_aoa, num_aoa_sensors + test_idx_vec_tdoa,
                                   num_aoa_sensors + num_tdoa_sensors + test_idx_vec_fdoa), axis=0)
    ref_idx_vec = np.concatenate((ref_idx_vec_aoa, num_aoa_sensors + ref_idx_vec_tdoa,
                                  num_aoa_sensors + num_tdoa_sensors + ref_idx_vec_fdoa), axis=0)

    # Finally, we resample the full covariance matrix using the assembled indices
    return utils.resample_covariance_matrix(cov, test_idx_vec, ref_idx_vec)
