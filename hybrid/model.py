import time
import numpy as np
from scipy.linalg import block_diag
import triang
import tdoa
import fdoa
import utils
from utils.covariance import CovarianceMatrix
from utils import SearchSpace


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
    :param angle_bias: (Optional) nAOA x 1 (or nAOA x 2, if do_2d_aoa=True) vector of angle bias terms [default=None]
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
        j_aoa = np.concatenate((j_aoa, np.zeros_like(j_aoa)), axis=0)
        j_tdoa = np.concatenate((j_tdoa, np.zeros_like(j_tdoa)), axis=0)
        j_fdoa = fdoa.model.jacobian(x_sensor=x_fdoa, v_sensor=v_fdoa, x_source=x_source, v_source=v_source,
                                     ref_idx=fdoa_ref_idx)
    else:
        j_fdoa = np.zeros(shape=empty_dims)

    # Combine component Jacobian matrices
    return np.concatenate((j_aoa, j_tdoa, j_fdoa), axis=1)


def jacobian_uncertainty(x_source, x_aoa=None, x_tdoa=None, x_fdoa=None, v_fdoa=None, v_source=None,
                         do_2d_aoa=False, tdoa_ref_idx=None, fdoa_ref_idx=None,
                         do_bias=False, do_pos_error=False):
    """
    Returns the Jacobian matrix for a set of TDOA measurements in the presence of sensor
    uncertainty, in the form of measurement bias and/or sensor position errors.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    30 April 2025

    :param x_source: nDim x n_source array of source positions
    :param x_aoa: nDim x nAOA array of sensor positions
    :param x_tdoa: nDim x nTDOA array of TDOA sensor positions
    :param x_fdoa: nDim x nFDOA array of FDOA sensor positions
    :param v_fdoa: nDim x nFDOA array of FDOA sensor velocities
    :param v_source: nDim x 1 source velocity; assumed stationary if left blank
    :param do_2d_aoa: Optional boolean parameter specifying whether 1D (az-only) or 2D (az/el) AOA is being performed
    :param tdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for TDOA
    :param fdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for FDOA
    :param do_bias: if True, jacobian includes gradient w.r.t. measurement biases
    :param do_pos_error: if True, jacobian includes gradient w.r.t. sensor pos/vel errors
    :return: n_dim x nMeasurement x n_source matrix of Jacobians, one for each candidate source position
    """

    # Make a dict with the args for all three gradient function calls
    jacob_args = {'x_source': x_source,
                  'v_source': v_source,
                  'x_aoa': x_aoa,
                  'x_tdoa': x_tdoa,
                  'x_fdoa': x_fdoa,
                  'v_fdoa': v_fdoa,
                  'do_2d_aoa': do_2d_aoa,
                  'tdoa_ref_idx': tdoa_ref_idx,
                  'fdoa_ref_idx': fdoa_ref_idx}

    # Gradient w.r.t source position
    j_source = grad_x(**jacob_args)
    j_list = [j_source]

    # Gradient w.r.t measurement biases
    if do_bias:
        j_bias = grad_bias(**jacob_args)
        j_list.append(j_bias)

    # Gradient w.r.t sensor position
    if do_pos_error:
        j_sensor_pos = grad_sensor_pos(**jacob_args)
        j_list.append(j_sensor_pos)

    # Combine component Jacobians
    j = np.concatenate(j_list, axis=0)

    return j


def log_likelihood(x_source, zeta, cov: CovarianceMatrix, x_aoa=None, x_tdoa=None, x_fdoa=None,
                   v_fdoa=None, v_source=None, do_2d_aoa=False, tdoa_ref_idx=None, fdoa_ref_idx=None,
                   do_resample=False, angle_bias=None, range_bias=None, range_rate_bias=None,
                   print_progress=False):
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
    :param angle_bias: AOA measurement bias
    :param range_bias: TDOA measurement bias
    :param range_rate_bias: FDOA measurement bias
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

    if print_progress:
        t_start = time.perf_counter()
        max_num_rows = 20
        desired_iter_per_row = np.ceil(n_source_pos / max_num_rows).astype(int)
        markers_per_row = 40
        desired_iter_per_marker = np.ceil(desired_iter_per_row / markers_per_row).astype(int)

        # Make sure we don't exceed the min/max iter per marker
        min_iter_per_marker = 10
        max_iter_per_marker = 1e6
        iter_per_marker = np.maximum(min_iter_per_marker, np.minimum(max_iter_per_marker, desired_iter_per_marker))
        iter_per_row = iter_per_marker * markers_per_row

        print('Computing Log Likelihood...')

    # Loop across source positions
    for idx_source in np.arange(n_source_pos):
        if print_progress:
            utils.print_progress(num_total=n_source_pos, curr_idx=idx_source,
                                 iterations_per_marker=iter_per_marker,
                                 iterations_per_row=iter_per_row,
                                 t_start=t_start)

        x_i = x_source[:, idx_source]
        if v_source is None:
            v_i = None
        else:
            v_i = v_source[:, idx_source]

        # Generate the ideal measurement matrix for this position
        zeta_dot = measurement(x_aoa=x_aoa, x_tdoa=x_tdoa, x_fdoa=x_fdoa, v_fdoa=v_fdoa, x_source=x_i,
                               v_source=v_i, do_2d_aoa=do_2d_aoa, tdoa_ref_idx=tdoa_ref_idx, fdoa_ref_idx=fdoa_ref_idx,
                                angle_bias=angle_bias, range_bias=range_bias, range_rate_bias=range_rate_bias)

        # Evaluate the measurement error
        err = (zeta_dot - zeta)

        # Compute the scaled log likelihood
        ell[idx_source] = - cov.solve_aca(err)

    if print_progress:
        print('done')
        t_elapsed = time.perf_counter() - t_start
        utils.print_elapsed(t_elapsed)

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
    search_space = SearchSpace(x_ctr=x_source,
                               max_offset=x_max,
                               epsilon=grid_res)
    x_set, x_grid, grid_shape = utils.make_nd_grid(search_space)
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


def grad_x(x_source, x_aoa=None, x_tdoa=None, x_fdoa=None, v_fdoa=None, v_source=None,
           do_2d_aoa=False, tdoa_ref_idx=None, fdoa_ref_idx=None):
    """
    Return the gradient of Hybrid measurements, with sensor uncertainties, with respect to target position, x.
    Equation 6.43. This function calls grad_x for each sensor type in succession, then concatenates the results.

    Ported from MATLAB code.

    Nicholas O'Donoughue
    14 April 2025

    :param x_source:
    :param x_aoa: nDim x nAOA array of sensor positions
    :param x_tdoa: nDim x nTDOA array of TDOA sensor positions
    :param x_fdoa: nDim x nFDOA array of FDOA sensor positions
    :param v_fdoa: nDim x nFDOA array of FDOA sensor velocities
    :param v_source: nDim x 1 source velocity; assumed stationary if left blank
    :param do_2d_aoa: Optional boolean parameter specifying whether 1D (az-only) or 2D (az/el) AOA is being performed
    :param tdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for TDOA
    :param fdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for FDOA
    :return jacobian:   Jacobian matrix representing the desired gradient
    """
    # TODO: Debug

    gradients = []
    if x_aoa is not None:
        gradients.append(triang.model.grad_x(x_source=x_source, x_sensor=x_aoa, do_2d_aoa=do_2d_aoa))

    if x_tdoa is not None:
        gradients.append(tdoa.model.grad_x(x_source=x_source, x_sensor=x_tdoa, ref_idx=tdoa_ref_idx))

    if x_fdoa is not None:
        gradients.append(fdoa.model.grad_x(x_source=x_source, x_sensor=x_fdoa, v_sensor=v_fdoa, v_source=v_source,
                                           ref_idx=fdoa_ref_idx))

    grad = np.concatenate(gradients, axis=1)

    return grad


def grad_bias(x_source, x_aoa=None, x_tdoa=None, x_fdoa=None, v_fdoa=None, v_source=None,
              do_2d_aoa=False, tdoa_ref_idx=None, fdoa_ref_idx=None):
    """
    Return the gradient of FDOA measurements, with sensor uncertainties, with respect to the unknown measurement bias
    terms.

    Ported from MATLAB code.

    Nicholas O'Donoughue
    14 April 2025

    :param x_source:
    :param x_aoa: nDim x nAOA array of sensor positions
    :param x_tdoa: nDim x nTDOA array of TDOA sensor positions
    :param x_fdoa: nDim x nFDOA array of FDOA sensor positions
    :param v_fdoa: nDim x nFDOA array of FDOA sensor velocities
    :param v_source: nDim x 1 source velocity; assumed stationary if left blank
    :param do_2d_aoa: Optional boolean parameter specifying whether 1D (az-only) or 2D (az/el) AOA is being performed
    :param tdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for TDOA
    :param fdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for FDOA
    :return jacobian:   Jacobian matrix representing the desired gradient
    """
    # TODO: Debug

    gradients = []
    if x_aoa is not None:
        gradients.append(triang.model.grad_bias(x_source=x_source, x_sensor=x_aoa, do_2d_aoa=do_2d_aoa))

    if x_tdoa is not None:
        gradients.append(tdoa.model.grad_bias(x_source=x_source, x_sensor=x_tdoa, ref_idx=tdoa_ref_idx))

    if x_fdoa is not None:
        gradients.append(fdoa.model.grad_bias(x_source=x_source, x_sensor=x_fdoa,
                                              ref_idx=fdoa_ref_idx))

    _, n_source = utils.safe_2d_shape(x_source)
    if n_source <= 1:
        # There is only one source, combine the gradients with a block diagonal across axes 0 and 1
        grad = block_diag(gradients)
    else:
        # The individual gradients are 3D, but block_diag only works on 2D, let's do some reshaping.
        # We need to move the third axis to the front
        gradients_reshape = [np.moveaxis(x, -1, 0) for x in gradients]

        # Now we can use list comprehension to call block_diag on each in turn
        res = [block_diag(*arrs) for arrs in zip(*gradients_reshape)]

        # This is now a list of length n_source, where each entry is a block-diagonal jacobian matrix at that source
        # position. Convert back to an ndarray and rearrange the axes
        grad = np.moveaxis(np.asarray(res), 0, -1)  # Move the first axis (n_source) back to the end.

    return grad


def grad_sensor_pos(x_source, x_aoa=None, x_tdoa=None, x_fdoa=None, v_fdoa=None, v_source=None,
                    do_2d_aoa=False, tdoa_ref_idx=None, fdoa_ref_idx=None):
    """
    Compute the gradient of hybrid measurements, with sensor uncertainties, with respect to sensor position and
    velocity, according to equation 6.43.

    Ported from MATLAB code.

    Nicholas O'Donoughue
    14 April 2025

    :param x_source:
    :param x_aoa: nDim x nAOA array of sensor positions
    :param x_tdoa: nDim x nTDOA array of TDOA sensor positions
    :param x_fdoa: nDim x nFDOA array of FDOA sensor positions
    :param v_fdoa: nDim x nFDOA array of FDOA sensor velocities
    :param v_source: nDim x 1 source velocity; assumed stationary if left blank
    :param do_2d_aoa: Optional boolean parameter specifying whether 1D (az-only) or 2D (az/el) AOA is being performed
    :param tdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for TDOA
    :param fdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for FDOA
    :return jacobian:   Jacobian matrix representing the desired gradient
    """
    # TODO: Debug

    gradients = []
    if x_aoa is not None:
        grad_a = triang.model.grad_sensor_pos(x_source=x_source, x_sensor=x_aoa, do_2d_aoa=do_2d_aoa)

        # Append zeros to reflect lack of dependence on sensor velocity
        grad_a = np.concatenate((grad_a, np.zeros_like(grad_a)), axis=0)

        # Add to running list of gradients
        gradients.append(grad_a)

    if x_tdoa is not None:
        grad_t = tdoa.model.grad_sensor_pos(x_source=x_source, x_sensor=x_tdoa, ref_idx=tdoa_ref_idx)

        # Append zeros to reflect lack of dependence on sensor velocity
        grad_t = np.concatenate((grad_t, np.zeros_like(grad_t)), axis=0)

        # Add to running list of gradients
        gradients.append(grad_t)

    if x_fdoa is not None:
        # No need to append zeros, the FDOA gradient assumes both velocity and position are considered.
        gradients.append(fdoa.model.grad_sensor_pos(x_source=x_source, x_sensor=x_fdoa, v_sensor=v_fdoa,
                                                    v_source=v_source, ref_idx=fdoa_ref_idx))

    # Concatenate the available gradients along the second dimension (axis=1)
    grad = np.concatenate(gradients, axis=1)

    return grad
