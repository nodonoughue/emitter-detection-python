import numpy as np
from scipy.linalg import block_diag
import triang
import tdoa
import fdoa
import utils
from utils.covariance import CovarianceMatrix


def measurement(x_source, x_aoa=None, x_tdoa=None, x_fdoa=None, v_fdoa=None, v_source=None,
                do_2d_aoa=False, tdoa_ref_idx=None, fdoa_ref_idx=None, angle_bias=None,
                range_bias=None, range_rate_bias=None):
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
    :param angle_bias: (Optional) nAOA x 1 (or nAOA x 2 if do_2d_aoa=True) vector of angle bias terms [default=None]
    :param range_bias: (Optional) nTDOA x 1 vector of range bias terms (time bias * c) [default=None]
    :param range_rate_bias: (Optional) nFDOA x 1 vector of range-rate bias terms (freq bias * c/f0) [default=None]
    :return zeta: nAoa + nTDOA + nFDOA - 2 x nSource array of measurements
    """

    # Construct component measurements
    to_concat = []
    if x_aoa is not None:
        z_a = triang.model.measurement(x_sensor=x_aoa, x_source=x_source, do_2d_aoa=do_2d_aoa, bias=angle_bias)
        to_concat.append(z_a)
    if x_tdoa is not None:
        z_t = tdoa.model.measurement(x_sensor=x_tdoa, x_source=x_source, ref_idx=tdoa_ref_idx, bias=range_bias)
        to_concat.append(z_t)
    if x_fdoa is not None:
        z_f = fdoa.model.measurement(x_sensor=x_fdoa, v_sensor=v_fdoa, 
                                     x_source=x_source, v_source=v_source,
                                     ref_idx=fdoa_ref_idx, bias=range_rate_bias)
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


def log_likelihood(x_source, zeta, cov: CovarianceMatrix, x_aoa=None, x_tdoa=None, x_fdoa=None,
                   v_fdoa=None, v_source=None, do_2d_aoa=False, tdoa_ref_idx=None, fdoa_ref_idx=None,
                   do_resample=False):
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
    if do_resample:
        # Use the hybrid-specific covariance matrix resampler, which handles the assumed structure.
        cov = cov.resample_hybrid(x_aoa=x_aoa, x_tdoa=x_tdoa, x_fdoa=x_fdoa, do_2d_aoa=do_2d_aoa,
                                  tdoa_ref_idx=tdoa_ref_idx, fdoa_ref_idx=fdoa_ref_idx)

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
        ell[idx_source] = - cov.solve_aca(err)

    return ell


def log_likelihood_uncertainty(theta, zeta, cov: CovarianceMatrix, cov_pos: CovarianceMatrix,
                               x_aoa=None, x_tdoa=None, x_fdoa=None, v_fdoa=None,
                               do_2d_aoa=False, tdoa_ref_idx=None, fdoa_ref_idx=None, do_resample=False,
                               do_aoa_bias=False, do_tdoa_bias=False, do_fdoa_bias=False):
    """
    Computes the Log Likelihood for Hybrid sensor measurement (AOA, TDOA, and
    FDOA), given the received measurement vector zeta, covariance matrix C,
    and set of candidate source positions x_source.

    theta = [x_source, v_source, bias, x_sensor.ravel(), v_sensor.ravel()]

    Source velocity is only part of the theta vector if FDOA is in use.
    Sensor position and velocity are only part of the theta vector if cov_pos is defined.
    Sensor bias is only part of the theta vector is the flag do_sensor_bias=True (for a given sensor modality)

    Nicholas O'Donoughue
    29 April 2025

    :param theta: Candidate uncertainty parameter vectors
    :param zeta: Combined AOA/TDOA/FDOA measurement vector
    :param cov: Measurement covariance matrix (or its precomputed inverse)
    :param cov_pos: sensor position error covariance matrix (if set to None, then sensor position error is ignored)
    :param x_aoa: nDim x nAOA array of sensor positions
    :param x_tdoa: nDim x nTDOA array of TDOA sensor positions
    :param x_fdoa: nDim x nFDOA array of FDOA sensor positions
    :param v_fdoa: nDim x nFDOA array of FDOA sensor velocities
    :param do_2d_aoa: Optional boolean parameter specifying whether 1D (az-only) or 2D (az/el) AOA is being performed
    :param tdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for TDOA
    :param fdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for FDOA
    :param do_resample: Boolean flag; if true the covariance matrix will be resampled, using reference indices
    :param do_aoa_bias: Boolean flag; if true, then AOA sensor bias terms will be included in search
    :param do_tdoa_bias: Boolean flag; if true, then TDOA sensor bias terms will be included in search
    :param do_fdoa_bias: Boolean flag; if true, then FDOA sensor bias terms will be included in search
    :return ell: Log-likelihood evaluated at each position x_source.
    """
    # TODO: Test

    # Parse Inputs
    num_dim = 0
    num_aoa = 0
    num_tdoa = 0
    num_fdoa = 0
    if x_aoa is not None:
        num_dim, num_aoa = utils.safe_2d_shape(x_aoa)
    if x_tdoa is not None:
        num_dim, num_tdoa = utils.safe_2d_shape(x_tdoa)
    if x_fdoa is not None:
        num_dim, num_fdoa = utils.safe_2d_shape(x_fdoa)
    if num_dim == 0:
        raise ValueError('No sensor positions provided. At least one of x_aoa, x_tdoa, x_fdoa must be defined.')
    _, n_source_pos = utils.safe_2d_shape(theta)

    # Make the source position is 2D, if it's not already
    if n_source_pos == 1:
        theta = theta[:, np.newaxis]
    if v_fdoa is None:
        v_fdoa = np.zeros_like(x_fdoa)

    # Initialize the output variable
    ell = np.zeros((n_source_pos, ))

    # Parse the Covariance Matrix, if needed
    if do_resample:
        # Use the hybrid-specific covariance matrix resampler, which handles the assumed structure.
        cov = cov.resample_hybrid(x_aoa=x_aoa, x_tdoa=x_tdoa, x_fdoa=x_fdoa, do_2d_aoa=do_2d_aoa,
                                  tdoa_ref_idx=tdoa_ref_idx, fdoa_ref_idx=fdoa_ref_idx)

    # Generate the indices for source position, measurement bias, and sensor position errors in the expanded
    # parameter vector represented by theta. Instead of nDim x nSourcePos, the matrix is assumed to have
    # size (nDim + nAOA + nDim*nAOA) x nSourcePos, where the first nDim rows represent the unknown target
    # position, the next nAOA rows are measurement biases, and the remainder are sensor positions.
    do_pos_error = cov_pos is not None
    parameter_indices = utils.make_uncertainty_indices(num_dim=num_dim,
                                                       num_aoa=num_aoa,
                                                       num_tdoa=num_tdoa,
                                                       num_fdoa=num_fdoa,
                                                       do_aoa_bias=do_aoa_bias,
                                                       do_tdoa_bias=do_tdoa_bias,
                                                       do_fdoa_bias=do_fdoa_bias,
                                                       do_aoa_pos_error=do_pos_error and num_aoa > 0,
                                                       do_tdoa_pos_error=do_pos_error and num_tdoa > 0,
                                                       do_fdoa_pos_error=do_pos_error and num_fdoa > 0)
    # Assume the sensor positions and velocities provided are truth.
    beta = np.concatenate((x_aoa, x_tdoa, x_fdoa, v_fdoa), axis=None)

    # Initialize Measurement Arguments
    msmt_args = {'x_aoa':x_aoa,
                 'x_tdoa':x_tdoa,
                 'x_fdoa':x_fdoa,
                 'v_fdoa':v_fdoa,
                 'do_2d_aoa':do_2d_aoa,
                 'tdoa_ref_idx':tdoa_ref_idx,
                 'fdoa_ref_idx':fdoa_ref_idx,
                 'angle_bias': None,
                 'range_bias': None,
                 'range_rate_bias': None}

    # Loop across source positions
    for idx_source, th_i in enumerate(theta.T):
        # Parse the parameter vector to grab the assumed target position, sensor measurement
        # biases, and sensor positions
        x_i = th_i[parameter_indices['source_pos']]
        if num_fdoa > 0:
            v_i = th_i[parameter_indices['source_vel']]
        else:
            v_i = None

        if do_aoa_bias:
            msmt_args['angle_bias'] = th_i[parameter_indices['aoa_bias']]
        if do_tdoa_bias:
            msmt_args['range_bias'] = th_i[parameter_indices['tdoa_bias']]
        if do_fdoa_bias:
            msmt_args['range_rate_bias'] = th_i[parameter_indices['fdoa_bias']]

        if do_pos_error:
            beta_i = th_i[parameter_indices['sensor']]
            msmt_args['x_aoa'] = np.reshape(th_i[parameter_indices['aoa_sensor_pos']], (num_dim, num_aoa))
            msmt_args['x_tdoa'] = np.reshape(th_i[parameter_indices['tdoa_sensor_vel']], (num_dim, num_tdoa))
            msmt_args['x_fdoa'] = np.reshape(th_i[parameter_indices['fdoa_sensor_pos']], (num_dim, num_fdoa))
            msmt_args['v_fdoa'] = np.reshape(th_i[parameter_indices['fdoa_sensor_vel']], (num_dim, num_fdoa))
        else:
            beta_i = beta

        # Generate the ideal measurement matrix for this position
        zeta_dot = measurement(x_source=x_i, v_source=v_i, **msmt_args)

        # Evaluate the measurement error
        err = (zeta_dot - zeta)
        err_pos = beta - beta_i

        # Compute the scaled log likelihood
        this_ell = - cov.solve_aca(err)
        if do_pos_error:
            this_ell -= cov_pos.solve_aca(err_pos)
        ell[idx_source] = this_ell

    return ell


def error(x_source, cov: CovarianceMatrix, x_aoa=None, x_tdoa=None, x_fdoa=None, v_fdoa=None, v_source=None,
          x_max=1, num_pts=11, do_2d_aoa=False, tdoa_ref_idx=None, fdoa_ref_idx=None, do_resample=False):
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
    :return epsilon: 2-D plot of FDOA error
    :return x_vec:
    :return y_vec:
    """

    # Compute the true measurement
    zeta = measurement(x_aoa=x_aoa, x_tdoa=x_tdoa, x_fdoa=x_fdoa, v_fdoa=v_fdoa, x_source=x_source, v_source=v_source,
                       do_2d_aoa=do_2d_aoa, tdoa_ref_idx=tdoa_ref_idx, fdoa_ref_idx=fdoa_ref_idx)

    # Pre-process the covariance matrix
    if do_resample:
        cov = cov.resample_hybrid(x_aoa=x_aoa, x_tdoa=x_tdoa, x_fdoa=x_fdoa, do_2d_aoa=do_2d_aoa,
                                  tdoa_ref_idx=tdoa_ref_idx, fdoa_ref_idx=fdoa_ref_idx)

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

    epsilon_list = [cov.solve_aca(this_err) for this_err in err.T]

    return np.reshape(epsilon_list, grid_shape), x_vec, y_vec
