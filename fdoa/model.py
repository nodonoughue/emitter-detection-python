import time
import numpy as np
import utils
from utils.covariance import CovarianceMatrix
import matplotlib.pyplot as plt
from utils import SearchSpace


def measurement(x_sensor, x_source, v_sensor=None, v_source=None, ref_idx=None, bias=None):
    """
    # Computed range rate difference measurements, using the
    # final sensor as a common reference for all FDOA measurements.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    21 January 2021

    :param x_sensor: nDim x n_sensor array of sensor positions
    :param x_source: nDim x n_source array of source positions
    :param v_sensor: nDim x nSensor array of sensor velocities
    :param v_source: nDim x n_source array of source velocities
    :param ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings
    :param bias: nSensor x 1 array of range-rate bias terms
    :return rrdoa: nSensor -1 x n_source array of RRDOA measurements
    """

    # Parse inputs
    n_dim, n_source, n_sensor, v_source, v_sensor = _check_inputs(x_source, v_source, x_sensor, v_sensor)
    test_idx_vec, ref_idx_vec = utils.parse_reference_sensor(ref_idx, n_sensor)

    # Make sure that x_source is 2D
    # We could use np.atleast_2d, but that will add the new dimension at the start, not the end
    if len(x_source.shape) == 1:
        x_source = x_source[:, np.newaxis]
    if len(v_source.shape) == 1:
        v_source = v_source[:, np.newaxis]

    # Compute distance from each source position to each sensor
    dx = x_source[:, np.newaxis, :] - x_sensor[:, :, np.newaxis]  # (n_dim, n_sensor, n_source)
    r = np.linalg.norm(dx, axis=0)  # (n_sensor, n_source)

    # Compute range rate from range and velocity
    dv = v_sensor[:, :, np.newaxis] - v_source[:, np.newaxis, :] # (n_dim, n_sensor, n_source)
    rr = np.divide(np.sum(dv*dx, axis=0), r, out=np.zeros((n_sensor, n_source)), where=r!=0)  # (n_sensor, n_source)

    # Add bias, if provided
    if bias is not None:
        if len(rr.shape)>1:
            rr = rr + bias[:, np.newaxis]  # (n_sensor, n_source)
        else:
            rr = rr + bias

    # Apply reference sensors to compute range rate difference for each sensor
    # pair
    rrdoa = rr[test_idx_vec, :] - rr[ref_idx_vec, :]  # (nPair, n_source)

    return np.atleast_1d(rrdoa.squeeze() if n_source == 1 else rrdoa)


def jacobian(x_sensor, x_source, v_sensor=None, v_source=None, ref_idx=None):
    """
    # Returns the Jacobian matrix for FDOA of a source at x_source
    # (n_dim x n_source) from sensors at x_sensor (n_dim x n_sensor) with velocity
    # v_sensor.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    21 January 2021

    :param x_sensor: n_dim x n_sensor vector of sensor positions
    :param x_source: n_dim x n_source vector of source positions
    :param v_sensor: n_dim x n_sensor vector of sensor velocities
    :param v_source: n_dim x n_source vector of source velocities
    :param ref_idx: Scalar index of reference sensor, or n_dim x nPair matrix of sensor pairings
    :return j: n_dim x nMeasurement x n_source matrix of Jacobians, one for each candidate source position
    """
    # ToDo: Think about refactoring as a two-element response (j_pos, j_vel) for easier parsing at the output

    # Parse inputs
    n_dim, n_source, n_sensor, v_source, v_sensor = _check_inputs(x_source, v_source, x_sensor, v_sensor)
    test_idx_vec, ref_idx_vec = utils.parse_reference_sensor(ref_idx, n_sensor)

    # Compute the Offset Vectors
    dx = x_sensor[:, :, np.newaxis] - np.reshape(x_source, (n_dim, 1, n_source))  # n_dim x n_sensor x n_source
    rn = np.reshape(np.sqrt(np.sum(dx**2, axis=0)), (1, n_sensor, n_source))  # Euclidean norm for each offset vector
    dx_norm = np.divide(dx, rn, out=np.zeros((n_dim, n_sensor, n_source)), where=rn!=0)  # n_dim x n_sensor x n_source
    px = np.reshape(dx_norm, (n_dim, 1, n_sensor, n_source)) * np.reshape(dx_norm, (1, n_dim, n_sensor, n_source))
    # n_dim x n_dim x n_sensor x n_source
    
    # Compute the gradient of R_n
    dv = (np.reshape(v_sensor, (n_dim, n_sensor, 1))
          - np.reshape(v_source, (n_dim, 1, n_source)))  # n_dim x n_sensor x n_source
    dv_norm = np.divide(dv, rn, out=np.zeros((n_dim, n_sensor, n_source)), where=rn!=0)  # n_dim x n_sensor x n_source
    # Iterate the matmul over the number of sensors and sources
    nabla_rn = np.asarray([[np.dot(np.eye(n_dim) - px[:, :, idx_sen, idx_src],
                                   dv_norm[:, idx_sen, idx_src])
                            for idx_src in np.arange(n_source)] for idx_sen in np.arange(n_sensor)])

    # Rearrange the axes to match expectation (n_dim x n_sensor x n_source)
    nabla_rn = np.moveaxis(nabla_rn, source=2, destination=0)

    # Compute test/reference differences and reshape output
    num_measurements = test_idx_vec.size
    if n_source > 1:
        out_dims = (n_dim, num_measurements, n_source)
    else:
        out_dims = (n_dim, num_measurements)

    result_pos = np.reshape(nabla_rn[:, test_idx_vec, :] - nabla_rn[:, ref_idx_vec, :], shape=out_dims)
    result_vel = np.reshape(dx_norm[:, test_idx_vec, :] - dx_norm[:, ref_idx_vec, :], shape=out_dims)

    return np.concatenate((result_pos, result_vel), axis=0)  # 2*n_dim x nPair x n_source


def jacobian_uncertainty(x_sensor, x_source, v_sensor=None, v_source=None, ref_idx=None, do_bias=False, do_pos_error=False):
    """
    Returns the Jacobian matrix for a set of FDOA measurements in the presence of sensor
    uncertainty, in the form of measurement bias and/or sensor position errors..

    Ported from MATLAB Code

    Nicholas O'Donoughue
    30 April 2025

    :param x_sensor: nDim x nTDOA array of TDOA sensor positions
    :param x_source: nDim x n_source array of source positions
    :param v_sensor: n_dim x n_sensor vector of sensor velocities
    :param v_source: n_dim x n_source vector of source velocities
    :param ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for TDOA
    :param do_bias: if True, jacobian includes gradient w.r.t. measurement biases
    :param do_pos_error: if True, jacobian includes gradient w.r.t. sensor pos/vel errors
    :return: n_dim x nMeasurement x n_source matrix of Jacobians, one for each candidate source position
    """

    # Parse inputs
    n_dim1, n_sensor = utils.safe_2d_shape(x_sensor)
    n_dim2, n_source = utils.safe_2d_shape(x_source)

    if n_dim1 != n_dim2:
        raise TypeError('Input variables must match along first dimension.')

    # Make a dict with the args for all three gradient function calls
    jacob_args = {'x_sensor': x_sensor,
                  'x_source': x_source,
                  'v_sensor': v_sensor,
                  'v_source': v_source,
                  'ref_idx': ref_idx}

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


def log_likelihood(x_sensor, rho_dot, cov: CovarianceMatrix, x_source,
                   v_sensor=None, v_source=None, ref_idx=None, do_resample=False, bias=None, print_progress=False):
    """
    # Computes the Log Likelihood for FDOA sensor measurement, given the
    # received measurement vector rho_dot, covariance matrix C,
    # and set of candidate source positions x_source.

    Ported from MATLAB Code.

    Nicholas O'Donoughue
    21 January 2021

    :param x_sensor: Sensor positions [m]
    :param rho_dot: FDOA measurement vector
    :param cov: CovarianceMatrix object
    :param x_source: Candidate source positions
    :param v_sensor: Sensor velocities [m/s]
    :param v_source: n_dim x n_source vector of source velocities
    :param ref_idx: Scalar index of reference sensor, or n_dim x n_pair matrix of sensor pairings
    :param do_resample: Boolean flag; if true the covariance matrix will be resampled, using ref_idx
    :return ell: Log-likelihood evaluated at each position x_source.
    """

    # Parse inputs
    n_dim, n_source, n_sensor, v_source, v_sensor = _check_inputs(x_source=x_source, v_source=v_source,
                                                                  x_sensor=x_sensor, v_sensor=v_sensor)

    # x_source and v_source might be vectors; 2D indexing below will fail.
    # Let's add a new axis
    if len(np.shape(x_source))==1:
        x_source = x_source[:, np.newaxis]
    if len(np.shape(v_source))==1:
        v_source = v_source[:, np.newaxis]

    # Make sure v_source and x_source have the same shape
    v_source = np.broadcast_to(array=v_source, shape=x_source.shape)

    if do_resample:
        cov = cov.resample(ref_idx=ref_idx)

    # Initialize output
    ell = np.zeros((n_source, ))

    if print_progress:
        t_start = time.perf_counter()
        max_num_rows = 20
        desired_iter_per_row = np.ceil(n_source / max_num_rows).astype(int)
        markers_per_row = 40
        desired_iter_per_marker = np.ceil(desired_iter_per_row / markers_per_row).astype(int)

        # Make sure we don't exceed the min/max iter per marker
        min_iter_per_marker = 10
        max_iter_per_marker = 1e6
        iter_per_marker = np.maximum(min_iter_per_marker, np.minimum(max_iter_per_marker, desired_iter_per_marker))
        iter_per_row = iter_per_marker * markers_per_row

        print('Computing Log Likelihood...')

    for idx_source in np.arange(n_source):
        if print_progress:
            utils.print_progress(num_total=n_source, curr_idx=idx_source,
                                 iterations_per_marker=iter_per_marker,
                                 iterations_per_row=iter_per_row,
                                 t_start=t_start)

        x_i = x_source[:, idx_source]
        v_i = v_source[:, idx_source]

        # Generate the ideal measurement matrix for this position
        r_dot = measurement(x_sensor=x_sensor, x_source=x_i,
                            v_sensor=v_sensor, v_source=v_i,
                            ref_idx=ref_idx, bias=bias)
        
        # Evaluate the measurement error
        err = (rho_dot - r_dot)

        # Compute the scaled log likelihood
        ell[idx_source] = - cov.solve_aca(err)

    if print_progress:
        print('done')
        t_elapsed = time.perf_counter() - t_start
        utils.print_elapsed(t_elapsed)

    return ell


def error(x_sensor, cov: CovarianceMatrix, x_source, x_max, num_pts,
          v_sensor=None, v_source=None, ref_idx=None, do_resample=False):
    """
    Construct a 2-D field from -x_max to +x_max, using numPts in each
    dimension.  For each point, compute the FDOA solution for each sensor
    against the reference (the first sensor), and compare to the FDOA
    solution from the true emitter position.

    Ported from MATLAB Code.

    Nicholas O'Donoughue
    21 January 2021

    :param x_sensor: nDim x N matrix of sensor positions
    :param cov: Covariance Matrix object
    :param x_source: nDim x 1 matrix of true emitter position
    :param x_max: nDim x 1 (or scalar) vector of maximum offset from origin for plotting
    :param num_pts: Number of test points along each dimension
    :param v_sensor: nDim x N matrix of sensor velocities
    :param v_source: nDim x 1 matrix of true emitter velocity
    :param ref_idx: Scalar index of reference sensor, or n_dim x n_pair matrix of sensor pairings
    :param do_resample: Boolean flag; if true the covariance matrix will be resampled, using ref_idx
    :return epsilon: 2-D plot of FDOA error
    :return x_vec:
    :return y_vec:
    """

    # Compute the True FDOA measurement
    # Compute true range rate difference measurements default condition is to
    # use the final sensor as the reference for all difference measurements.
    rr = measurement(x_sensor=x_sensor, x_source=x_source,
                     v_sensor=v_sensor, v_source=v_source,
                     ref_idx=ref_idx)  # shape: (n_pair, 1)

    if do_resample:
        cov = cov.resample(ref_idx=ref_idx)

    # Set up test points
    grid_res = 2*x_max / (num_pts-1)
    search_space = SearchSpace(x_ctr=np.array([0., 0.]),
                               max_offset=x_max,
                               epsilon=grid_res)
    x_set, x_grid, grid_shape = utils.make_nd_grid(search_space)
    x_vec = x_grid[0][0, :]
    y_vec = x_grid[1][:, 0]

    rr_list = measurement(x_sensor=x_sensor, x_source=x_set,
                          v_sensor=v_sensor, v_source=v_source,
                          ref_idx=ref_idx)

    err = rr[:, np.newaxis] - rr_list

    epsilon_list = [cov.solve_aca(this_err) for this_err in err.T]

    return np.reshape(epsilon_list, grid_shape), x_vec, y_vec


def draw_isodoppler(x_ref, v_ref, x_test, v_test, vdiff, num_pts, max_ortho, v_source=None):
    """
    # Finds the isochrone with the stated range rate difference from points x1
    # and x2.  Generates an arc with 2*numPts-1 points, that spans up to
    # maxOrtho distance from the intersection line of x1 and x2

    Ported from MATLAB Code.

    Nicholas O'Donoughue
    21 January 2021

    :param x_ref: Position of first sensor (Ndim x 1) [m]
    :param v_ref: Velocity vector of first sensor (Ndim x 1) [m/s]
    :param x_test: Position of second sensor (Ndim x 1) [m]
    :param v_test: Velocity vector of second sensor (Ndim x 1) [m/s]
    :param vdiff: Desired velocity difference [m/s]
    :param num_pts: Number of points to compute
    :param max_ortho: Maximum offset from line of sight between x1 and x2 [m]
    :return x_iso: First dimension of iso doppler curve [m]
    :return y_iso: Second dimension of iso doppler curve [m]
    """

    # Set frequency to 3e8, so that c/f_0 is unity, and output of utils.dopDiff
    # is velocity difference [m/s]
    f_0 = utils.constants.speed_of_light

    # Set up test points
    grid_spacing = 2 * max_ortho / (num_pts - 1)  # Compute grid density
    search_space = SearchSpace(x_ctr=np.array([0., 0.]),
                               max_offset=max_ortho,
                               epsilon=grid_spacing)
    x_set, x_grid, grid_shape = utils.make_nd_grid(search_space)

    if v_source is None:
        v_source = np.zeros_like(x_set)
    df_plot = utils.geo.calc_doppler_diff(x_source=x_set, v_source=v_source,
                                          x_ref=x_ref, v_ref=v_ref, x_test=x_test, v_test=v_test, f=f_0)

    # Generate Levels
    if np.size(vdiff) > 1:
        # Multiple velocity differences were specified, they have to be in ascending order to be used in contour command
        sort_idx = np.argsort(vdiff)
        unsort_idx = np.argsort(sort_idx)   # Indices to unsort
        sort_idx = np.unravel_index(sort_idx, vdiff.shape)
        level_set = vdiff[sort_idx]
        multiple_outputs = True
    elif np.isscalar(vdiff):
        # We only care about the contour level at vdiff; but pyplot.contour won't let us
        # draw just one.
        unsort_idx = int(1)  # The only level we want to output
        level_set = np.array([np.amin(df_plot), vdiff, np.amax(df_plot)])
        multiple_outputs = False
    else:
        # If vdiff is an array, with only one element, we get a deprecation warning if we treat it like a scalar.
        # First, let's reshape it to make sure it's a vector, then index the first element when constructing our
        # level set
        vdiff = np.reshape(vdiff, (1,))
        unsort_idx = int(1)  # The only level we want to output
        level_set = np.array([np.amin(df_plot), vdiff[0], np.amax(df_plot)])
        multiple_outputs = False

    # Compute contour
    fig00 = plt.figure()
    contour_set = plt.contour(x_grid[0], x_grid[1], np.reshape(df_plot, grid_shape), levels=level_set)

    # Close the figure generated
    plt.close(fig00)

    # Extract the desired coordinates
    x_iso = []
    y_iso = []
    for this_contour in contour_set.allsegs:
        # Each contour level is a list of contour regions
        this_x_iso = np.zeros(shape=(0, ))
        this_y_iso = []

        for this_region in this_contour:
            # Grab the x/y coordinates for this region
            this_x = this_region[:, 0]
            this_y = this_region[:, 1]

            # Remove any that are out of bounds
            out_of_bounds = np.fmax(np.abs(this_x), np.abs(this_y)) > max_ortho

            this_x_iso = np.append(this_x_iso, np.append(this_x[~out_of_bounds], np.nan))
            this_y_iso = np.append(this_y_iso, np.append(this_y[~out_of_bounds], np.nan))

        # Add to the list of isodoppler contours; remove the leading nan (for simplicity)
        x_iso.append(this_x_iso)
        y_iso.append(this_y_iso)

    # Unsort the levels
    if multiple_outputs:
        # The output is a list of x_iso and y_iso entries
        x_iso = [x_iso[idx] for idx in unsort_idx]
        y_iso = [y_iso[idx] for idx in unsort_idx]
    else:
        # idx is a scalar, use it directly, no need for list comprehension
        x_iso = x_iso[unsort_idx]
        y_iso = y_iso[unsort_idx]

    return x_iso, y_iso


def grad_x(x_sensor, x_source, v_sensor=None, v_source=None, ref_idx=None):
    """
    Return the gradient of FDOA measurements, with sensor uncertainties, with respect to target position, x.
    Equation 6.31. The sensor uncertainties don't impact the gradient for FDOA, so this reduces to the previously
    defined Jacobian. This function is merely a wrapper for calls to fdoa.model.jacobian, with the optional argument
    'bias' ignored.

    Ported from MATLAB code.

    Nicholas O'Donoughue
    14 April 2025

    :param x_sensor:    FDOA sensor positions
    :param x_source:    Source positions
    :param v_sensor:    Optional FDOA sensor velocities (0 if not defined)
    :param v_source:    Optional FDOA source velocities (0 if not defined)
    :param ref_idx:     Reference index (optional)
    :return jacobian:   Jacobian matrix representing the desired gradient
    """
    # TODO: Debug

    # Sensor uncertainties don't impact the gradient with respect to target position; this is the same as the previously
    # defined function fdoa.model.jacobian.
    return jacobian(x_sensor=x_sensor, x_source=x_source, v_sensor=v_sensor, v_source=v_source, ref_idx=ref_idx)


def grad_bias(x_sensor, x_source, ref_idx=None):
    """
    Return the gradient of FDOA measurements, with sensor uncertainties, with respect to the unknown measurement bias
    terms.

    Ported from MATLAB code.

    Nicholas O'Donoughue
    14 April 2025

    :param x_sensor:    FDOA sensor positions
    :param x_source:    Source positions
    :param ref_idx:     Reference index (optional)
    :return jacobian:   Jacobian matrix representing the desired gradient
    """
    # TODO: Debug

    # Parse the reference index
    _, num_sensors = utils.safe_2d_shape(x_sensor)
    test_idx_vec, ref_idx_vec = utils.parse_reference_sensor(ref_idx, num_sensors)

    # According to eq 6.42, the m-th row is 1 for every column in which the m-th sensor is a test index, and -1 for
    # every column in which the m-th sensor is a reference index.
    num_measurements = np.size(test_idx_vec)
    grad = np.zeros((num_sensors, num_measurements))
    for i, (test, ref) in enumerate(zip(test_idx_vec, ref_idx_vec)):
        grad[i, test] = 1
        grad[i, ref] = -1

    # Repeat for each source position
    _, num_sources = utils.safe_2d_shape(x_source)
    if num_sources > 1:
        grad = np.repeat(grad, num_sources, axis=2)

    return grad


def grad_sensor_pos(x_sensor, x_source, v_sensor=None, v_source=None, ref_idx=None):
    """
    Compute the gradient of FDOA measurements, with sensor uncertainties, with respect to sensor position and velocity.

    Ported from MATLAB code.

    Nicholas O'Donoughue
    14 April 2025

    :param x_sensor:    FDOA sensor positions
    :param x_source:    Source positions
    :param v_sensor:    Optional FDOA sensor velocities (0 if not defined)
    :param v_source:    Optional FDOA source velocities (0 if not defined)
    :param ref_idx:     Reference index (optional)
    :return jacobian:   Jacobian matrix representing the desired gradient
    """
    # TODO: Debug

    # Parse inputs
    n_dim, n_source, n_sensor, v_source, v_sensor = _check_inputs(x_source, v_source, x_sensor, v_sensor)

    # Compute pointing vectors and projection matrix
    dx = x_sensor - np.reshape(x_source, shape=(n_dim, 1, n_source))
    dv = v_sensor - np.reshape(v_source, shape=(n_dim, 1, n_source))
    rn = np.sqrt(np.sum(np.fabs(dx)**2, axis=0))  # (1, n_sensor, n_source)
    dx_norm = dx / rn
    dv_norm = dv / rn

    proj_x = (np.reshape(dx_norm, shape=(n_dim, 1, n_sensor, n_source)) *
              np.reshape(np.conjugate(dx_norm), shape=(1, n_dim, n_sensor, n_source)))

    # Compute the gradient of R_n
    nabla_rn = np.squeeze(np.sum((np.eye(n_dim) - proj_x) *
                                 np.reshape(dv_norm, shape=(1, n_dim, n_sensor, n_source)), axis=1))
    # (n_dim, n_sensor, n_source)

    # Parse the reference index
    test_idx_vec, ref_idx_vec = utils.parse_reference_sensor(ref_idx, n_sensor)

    # Build the Gradient
    n_measurement = np.size(test_idx_vec)
    grad_pos = np.zeros((n_dim * n_sensor, n_measurement, n_source))
    grad_vel = np.zeros((n_dim * n_sensor, n_measurement, n_source))
    for i, (test, ref) in enumerate(zip(test_idx_vec, ref_idx_vec)):
        # Gradient w.r.t. sensor pos, eq 6.38
        start_test = n_dim * test
        end_test = start_test + n_dim  # add +1 because of the way python indexing works
        grad_pos[start_test:end_test, i, :] = nabla_rn[:, test, :]

        start_ref = n_dim * ref
        end_ref = start_ref + n_dim
        grad_pos[start_ref:end_ref, i, :] = -nabla_rn[:, ref, :]

        # Gradient w.r.t. sensor vel, eq 6.40
        grad_vel[start_test:end_test, i, :] = dx_norm[:, test, :]
        grad_vel[start_ref:end_ref, i, :] = -dx_norm[:, ref, :]

    # Combine the gradient w.r.t. sensor pos and sensor vel
    # eq 6.36
    grad = np.concatenate((grad_pos, grad_vel), axis=1)

    return grad


def _check_inputs(x_source, v_source, x_sensor, v_sensor):
    if v_sensor is None and v_source is None:
        raise ValueError('At least one of either v_sensor or v_source must be defined to use FDOA.')
    elif v_sensor is None:
        v_sensor = np.zeros_like(x_sensor)
    elif v_source is None:
        v_source = np.zeros_like(x_source)

    n_dim1, n_sensor1 = utils.safe_2d_shape(x_sensor)
    n_dim2, n_sensor2 = utils.safe_2d_shape(v_sensor)
    n_dim3, n_source1 = utils.safe_2d_shape(x_source)
    n_dim4, n_source2 = utils.safe_2d_shape(v_source)

    if n_dim1 != n_dim2 or n_dim1 != n_dim3 or n_dim1 != n_dim4:
        raise TypeError('First dimension of all inputs must match')

    if not utils.is_broadcastable(x_sensor, v_sensor):
        raise TypeError('Sensor position and velocity inputs must have matching shapes.')

    if not utils.is_broadcastable(x_source, v_source):
        raise TypeError('Sensor position and velocity inputs must have matching shapes.')

    n_sensor = np.amax((n_sensor1, n_sensor2))
    n_source = np.amax((n_source1, n_source2))

    return n_dim1, n_source, n_sensor, v_source, v_sensor
