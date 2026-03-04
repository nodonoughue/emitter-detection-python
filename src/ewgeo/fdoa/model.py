import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from ewgeo.utils import parse_reference_sensor, SearchSpace, broadcast_backwards
from ewgeo.utils.constants import speed_of_light
from ewgeo.utils.covariance import CovarianceMatrix
from ewgeo.utils.geo import calc_doppler_diff


def measurement(x_sensor: npt.ArrayLike,
                x_source: npt.ArrayLike,
                v_sensor: npt.ArrayLike | None=None,
                v_source: npt.ArrayLike | None=None,
                ref_idx=None,
                bias: npt.ArrayLike | None=None):
    """
    # Computed range rate difference measurements, using the
    # final sensor as a common reference for all FDOA measurements.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    21 January 2021

    :param x_sensor: (nDim, n_sensor, *out_shp) array of sensor positions.
    :param x_source: nDim x n_source array of source positions.
    :param v_sensor: nDim x nSensor array of sensor velocities.
    :param v_source: nDim x n_source array of source velocities.
    :param ref_idx: Scalar index of reference sensor or nDim x nPair matrix of sensor pairings.
    :param bias: nSensor x 1 array of range-rate bias terms.
    :return rrdoa: nSensor -1 x n_source array of RRDOA measurements.
    """

    # Parse inputs
    n_dim, n_source, n_sensor, out_shp, x_source, v_source, x_sensor, v_sensor = _check_inputs(x_source, v_source, x_sensor, v_sensor)
    test_idx_vec, ref_idx_vec = parse_reference_sensor(ref_idx, n_sensor)

    #
    # # Manually extend the dimensions
    # x1 = x1[:, :, np.newaxis]
    # while len(x1.shape) < len(out_shp) + 1: x1 = np.expand_dims(x1, axis=-1)
    # x2 = x2[:, np.newaxis, :]
    # while len(x2.shape) < len(out_shp) + 1: x2 = np.expand_dims(x2, axis=-1)

    # Compute distance from each source position to each sensor
    dx = x_source[:, np.newaxis, :] - x_sensor[:, :, np.newaxis]  # (n_dim, n_sensor, n_source)
    r = np.linalg.norm(dx, axis=0)  # (n_sensor, n_source)

    # Compute range rate from range and velocity
    dv = v_sensor[:, :, np.newaxis] - v_source[:, np.newaxis, :] # (n_dim, n_sensor, n_source)
    rr = np.divide(np.sum(dv*dx, axis=0), r, out=np.zeros(out_shp), where=r!=0)  # (n_sensor, n_source)

    # Add bias, if provided
    if bias is not None:
        if bias.shape != r.shape:
            while bias.ndim < r.ndim:
                bias = np.expand_dims(bias, axis=-1)
        rr = rr + bias

    # Apply reference sensors to compute range-rate difference for each sensor
    # pair
    rrdoa = rr[test_idx_vec, :] - rr[ref_idx_vec, :]  # (nPair, n_source)

    return np.atleast_1d(rrdoa.squeeze() if n_source == 1 else rrdoa)


def jacobian(x_sensor: npt.ArrayLike,
             x_source: npt.ArrayLike,
             v_sensor: npt.ArrayLike | None=None,
             v_source: npt.ArrayLike | None=None,
             ref_idx=None):
    """
    Returns the Jacobian matrix for FDOA of a source at x_source (n_dim x n_source) from sensors at x_sensor
    (n_dim x n_sensor) with velocity v_sensor.

    Ported from MATLAB Code.

    Nicholas O'Donoughue
    21 January 2021

    :param x_sensor: n_dim x n_sensor vector of sensor positions.
    :param x_source: n_dim x n_source vector of source positions.
    :param v_sensor: n_dim x n_sensor vector of sensor velocities.
    :param v_source: n_dim x n_source vector of source velocities.
    :param ref_idx: Scalar index of reference sensor, or n_dim x nPair matrix of sensor pairings.
    :return j: n_dim x nMeasurement x n_source matrix of Jacobians, one for each candidate source position.
    """
    # ToDo: Think about refactoring as a two-element response (j_pos, j_vel) for easier parsing at the output

    # Parse inputs
    n_dim, n_source, n_sensor, out_shp, x_source, v_source, x_sensor, v_sensor = _check_inputs(x_source, v_source, x_sensor, v_sensor)
    test_idx_vec, ref_idx_vec = parse_reference_sensor(ref_idx, n_sensor)

    # Compute the Offset Vectors
    dx = x_sensor[:, :, np.newaxis] - x_source[:, np.newaxis, :]  # shape: (1, *out_shp)
    rn = np.sqrt(np.sum(dx**2, axis=0))  # Euclidean norm for each offset vector, shape: out_shp
    dx_norm = np.divide(dx, rn, out=np.zeros((n_dim, *out_shp)), where=rn!=0)  # shape: (n_dim, *out_shp)
    px = dx_norm[:, np.newaxis, :] * dx_norm[np.newaxis, :]  # shape: (n_dim, n_dim, *out_shp)
    
    # Compute the gradient of R_n
    dv = v_sensor[:, :, np.newaxis] - v_source[:, np.newaxis, :]  # shape: (n_dim, *out_shp)
    dv_norm = np.divide(dv, rn, out=np.zeros((n_dim, *out_shp)), where=rn!=0)  # shape: (n_dim, *out_shp)
    # Iterate the matmul over the number of sensors and sources
    this_eye = np.eye(n_dim)
    arrs, _ = broadcast_backwards([this_eye, px], start_dim=0, do_broadcast=True)
    this_eye, px = arrs
    nabla_rn = np.sum((this_eye - px) * dv_norm[np.newaxis, :, :], axis=1)  # shape: (n_dim, *out_shp)

    # Compute test/reference differences and reshape output
    num_measurements = test_idx_vec.size
    out_dims = [n_dim, num_measurements]
    if len(out_shp) > 1: out_dims.extend(out_shp[1:])

    result_pos = np.reshape(nabla_rn[:, test_idx_vec, :] - nabla_rn[:, ref_idx_vec, :], shape=out_dims)
    result_vel = np.reshape(dx_norm[:, test_idx_vec, :] - dx_norm[:, ref_idx_vec, :], shape=out_dims)

    result = np.concatenate((result_pos, result_vel), axis=0)  # 2*n_dim x nPair x n_source

    # strip any singleton dimensions off the end
    return np.squeeze(result, axis=tuple(range(2,np.ndim(result))))


def jacobian_uncertainty(x_sensor: npt.ArrayLike,
                         x_source: npt.ArrayLike,
                         v_sensor: npt.ArrayLike | None=None,
                         v_source: npt.ArrayLike | None=None,
                         ref_idx=None,
                         do_bias: bool=False,
                         do_pos_error: bool=False):
    """
    Returns the Jacobian matrix for a set of FDOA measurements in the presence of sensor
    uncertainty, in the form of measurement bias and/or sensor position errors..

    Ported from MATLAB Code

    Nicholas O'Donoughue
    30 April 2025

    :param x_sensor: nDim x nTDOA array of TDOA sensor positions.
    :param x_source: nDim x n_source array of source positions.
    :param v_sensor: n_dim x n_sensor vector of sensor velocities.
    :param v_source: n_dim x n_source vector of source velocities.
    :param ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for TDOA.
    :param do_bias: if True, jacobian includes gradient w.r.t. measurement biases.
    :param do_pos_error: if True, jacobian includes gradient w.r.t. sensor pos/vel errors.
    :return: n_dim x nMeasurement x n_source matrix of Jacobians, one for each candidate source position.
    """

    # Parse inputs
    n_dim1 = np.shape(x_sensor)[0] if np.ndim(x_sensor) > 0 else 1
    n_dim2 = np.shape(x_source)[0] if np.ndim(x_source) > 0 else 1

    if n_dim1 != n_dim2:
        raise TypeError('Input variables must match along first dimension.')

    # Make a dict with the args for all three gradient function calls
    jacob_args = {'x_sensor': x_sensor,
                  'x_source': x_source,
                  'v_sensor': v_sensor,
                  'v_source': v_source,
                  'ref_idx': ref_idx}

    # Gradient w.r.t source position
    j_source = grad_source(**jacob_args)
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


def log_likelihood(x_sensor: npt.ArrayLike,
                   rho_dot: npt.ArrayLike,
                   cov: CovarianceMatrix,
                   x_source: npt.ArrayLike,
                   v_sensor: npt.ArrayLike | None=None,
                   v_source: npt.ArrayLike | None=None,
                   ref_idx=None,
                   do_resample: bool=False,
                   bias: npt.ArrayLike | None=None):
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
    :param bias: sensor measurement biases
    :return ell: Log-likelihood evaluated at each position x_source.
    """

    if do_resample:
        cov = cov.resample(ref_idx=ref_idx)

    # Generate the ideal measurement matrix for this position
    r_dot = measurement(x_sensor=x_sensor, x_source=x_source,
                        v_sensor=v_sensor, v_source=v_source,
                        ref_idx=ref_idx, bias=bias)

    while r_dot.ndim < rho_dot.ndim: r_dot = np.expand_dims(r_dot, -1)
    while rho_dot.ndim < r_dot.ndim: rho_dot = np.expand_dims(rho_dot, -1)

    # Evaluate the measurement error
    err = (rho_dot - r_dot)

    # Compute the scaled log likelihood
    ell = - cov.solve_aca(np.moveaxis(err, source=0, destination=-1))

    return ell


def error(x_sensor: npt.ArrayLike,
          cov: CovarianceMatrix,
          x_source: npt.ArrayLike,
          x_max: npt.ArrayLike,
          num_pts: int,
          v_sensor: npt.ArrayLike | None=None,
          v_source: npt.ArrayLike | None=None,
          ref_idx=None,
          do_resample: bool=False):
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
    x_set, x_grid = search_space.x_set, search_space.x_grid
    x_vec = x_grid[0][0, :]
    y_vec = x_grid[1][:, 0]

    rr_list = measurement(x_sensor=x_sensor, x_source=x_set,
                          v_sensor=v_sensor, v_source=v_source,
                          ref_idx=ref_idx)

    err = rr[:, np.newaxis] - rr_list

    # epsilon_list = [cov.solve_aca(this_err) for this_err in err.T]
    epsilon_list = cov.solve_aca(err.T)  # transpose it; solve_aca operates over the last dimension
    return np.reshape(epsilon_list, search_space.grid_shape), x_vec, y_vec


def draw_isodoppler(x_ref: npt.NDArray[np.float64],
                    v_ref: npt.NDArray[np.float64],
                    x_test: npt.NDArray[np.float64],
                    v_test: npt.NDArray[np.float64],
                    vdiff: npt.NDArray[np.float64],
                    num_pts: int,
                    max_ortho: float,
                    v_source: npt.NDArray[np.float64] | None=None)\
        -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    # Finds the isochrone with the stated range rate difference from points x1
    # and x2.  Generates an arc with 2*numPts-1 points, that spans up to
    # maxOrtho distance from the intersection line of x1 and x2

    Ported from MATLAB Code.

    Nicholas O'Donoughue
    21 January 2021

    :param x_ref: Position of first sensor (Ndim x 1) [m].
    :param v_ref: Velocity vector of first sensor (Ndim x 1) [m/s].
    :param x_test: Position of second sensor (Ndim x 1) [m].
    :param v_test: Velocity vector of second sensor (Ndim x 1) [m/s].
    :param vdiff: Desired velocity difference [m/s].
    :param num_pts: Number of points to compute.
    :param max_ortho: Maximum offset from line of sight between x1 and x2 [m].
    :param v_source: Optional velocity vector of source at each position
                     (0 if not defined) [m/s].
    :return x_iso: First dimension of iso doppler curve [m].
    :return y_iso: Second dimension of iso doppler curve [m].
    """

    # Set frequency to 3e8, so that c/f_0 is unity, and output of dopDiff
    # is velocity difference [m/s]
    f_0 = speed_of_light

    # Set up test points
    grid_spacing = 2 * max_ortho / (num_pts - 1)  # Compute grid density
    search_space = SearchSpace(x_ctr=np.array([0., 0.]),
                               max_offset=max_ortho,
                               epsilon=grid_spacing)
    x_set, x_grid = search_space.x_set, search_space.x_grid

    if v_source is None:
        v_source = np.zeros_like(x_set)
    df_plot = calc_doppler_diff(x_source=x_set, v_source=v_source,
                                x_ref=x_ref, v_ref=v_ref,
                                x_test=x_test, v_test=v_test, f=f_0)

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
    contour_set = plt.contour(x_grid[0], x_grid[1], np.reshape(df_plot, search_space.grid_shape), levels=level_set)

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


def grad_source(x_sensor: npt.ArrayLike,
           x_source: npt.ArrayLike,
           v_sensor: npt.ArrayLike | None=None,
           v_source: npt.ArrayLike | None=None,
           ref_idx=None):
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
    :param v_source: Optional FDOA source velocities (0 if not defined)
    :param ref_idx:     Reference index (optional)
    :return jacobian:   Jacobian matrix representing the desired gradient
    """

    # Sensor uncertainties don't impact the gradient with respect to target position; this is the same as the previously
    # defined function fdoa.model.jacobian.
    return jacobian(x_sensor=x_sensor, x_source=x_source, v_sensor=v_sensor, v_source=v_source, ref_idx=ref_idx)


def grad_bias(x_sensor: npt.ArrayLike, x_source: npt.ArrayLike, ref_idx=None):
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
    shp = np.shape(x_sensor)
    num_sensors = shp[1] if len(shp) > 1 else 1
    test_idx_vec, ref_idx_vec = parse_reference_sensor(ref_idx, num_sensors)

    # According to eq 6.42, the m-th row is 1 for every column in which the m-th sensor is a test index, and -1 for
    # every column in which the m-th sensor is a reference index.
    num_measurements = np.size(test_idx_vec)
    grad = np.zeros((num_sensors, num_measurements))
    for i, (test, ref) in enumerate(zip(test_idx_vec, ref_idx_vec)):
        grad[test, i] = 1
        grad[ref, i] = -1

    # Repeat for each source position
    shp = np.shape(x_source)
    num_sources = shp[1] if len(shp) > 1 else 1
    if num_sources > 1:
        grad = np.repeat(grad[:, :, np.newaxis], num_sources, axis=2)

    return grad


def grad_sensor_pos(x_sensor: npt.ArrayLike,
                    x_source: npt.ArrayLike,
                    v_sensor: npt.ArrayLike | None=None,
                    v_source: npt.ArrayLike | None=None,
                    ref_idx=None):
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

    # Parse inputs
    n_dim, n_source, n_sensor, out_shp, x_source, v_source, x_sensor, v_sensor = _check_inputs(x_source, v_source, x_sensor, v_sensor)

    # Compute pointing vectors and projection matrix
    dx = x_sensor[:, :, np.newaxis] - x_source[:, np.newaxis, :]  # shape: (n_dim, *out_shp)
    dv = v_sensor[:, :, np.newaxis] - v_source[:, np.newaxis, :]  # shape: (n_dim, *out_shp)
    rn = np.sqrt(np.sum(np.fabs(dx)**2, axis=0))  # (1, *out_shp)
    dx_norm = dx / rn  # shape: (n_dim, *out_shp)
    dv_norm = dv / rn  # shape: (n_dim, *out_shp)

    proj_x = dx_norm[:, np.newaxis, :] * np.conjugate(dx_norm[np.newaxis, :])  # shape: (n_dim, n_dim, *out_shp)

    # Compute the gradient of R_n
    this_eye = np.eye(n_dim)  # shape: (n_dim, n_dim)
    arrs, _ = broadcast_backwards([this_eye, proj_x], start_dim=0, do_broadcast=True)
    this_eye, proj_x = arrs
    nabla_rn = np.sum((this_eye - proj_x) * dv_norm[np.newaxis, :], axis=1)  # shape: (n_dim, *out_shp)

    # Parse the reference index
    test_idx_vec, ref_idx_vec = parse_reference_sensor(ref_idx, n_sensor)

    # Build the Gradient
    n_measurement = np.size(test_idx_vec)
    grad_pos = np.zeros((n_dim * n_sensor, n_measurement, *out_shp[1:]))
    for i, (test, ref) in enumerate(zip(test_idx_vec, ref_idx_vec)):
        # Gradient w.r.t. sensor pos, eq 6.38
        start_test = n_dim * test
        end_test = start_test + n_dim  # add +1 because of the way python indexing works
        grad_pos[start_test:end_test, i, :] = nabla_rn[:, test, :]

        start_ref = n_dim * ref
        end_ref = start_ref + n_dim
        grad_pos[start_ref:end_ref, i, :] = -nabla_rn[:, ref, :]

    return grad_pos


def grad_sensor_vel(x_sensor: npt.ArrayLike,
                    x_source: npt.ArrayLike,
                    v_sensor: npt.ArrayLike | None=None,
                    v_source: npt.ArrayLike | None=None,
                    ref_idx=None):
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

    # Parse inputs
    n_dim, n_source, n_sensor, out_shp, x_source, v_source, x_sensor, v_sensor = _check_inputs(x_source, v_source, x_sensor, v_sensor)

    # Compute pointing vectors and projection matrix
    dx = x_sensor[:, :, np.newaxis] - x_source[:, np.newaxis, :]  # shape: (1, *out_shp)
    rn = np.sqrt(np.sum(np.fabs(dx)**2, axis=0))  # shape: out_shp
    dx_norm = dx / rn  # shape: (1, *out_shp)

    # Parse the reference index
    test_idx_vec, ref_idx_vec = parse_reference_sensor(ref_idx, n_sensor)

    # Build the Gradient
    n_measurement = np.size(test_idx_vec)
    grad_vel = np.zeros((n_dim * n_sensor, n_measurement, *out_shp[1:]))
    for i, (test, ref) in enumerate(zip(test_idx_vec, ref_idx_vec)):
        # Gradient w.r.t. sensor pos, eq 6.38
        start_test = n_dim * test
        end_test = start_test + n_dim  # add +1 because of the way python indexing works

        start_ref = n_dim * ref
        end_ref = start_ref + n_dim

        # Gradient w.r.t. sensor vel, eq 6.40
        grad_vel[start_test:end_test, i, :] = -dx_norm[:, test, :]
        grad_vel[start_ref:end_ref, i, :] = dx_norm[:, ref, :]

    return grad_vel

def _check_inputs(x_source: npt.NDArray[np.float64], v_source: npt.NDArray[np.float64] | None,
                  x_sensor: npt.NDArray[np.float64], v_sensor: npt.NDArray[np.float64] | None)\
                  -> tuple[int, int, int, tuple[int, ...], npt.NDArray[np.float64],
                  npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Check the position and velocity inputs for source and sensors, enforce consistency in the number of sources/sensors
    and spatial dimensions, extend array dimensions to ensure proper broadcasting.

    :param x_source:    Source positions (n_dim, n_source, *out_shp)
    :param v_source:    Source velocities (n_dim, n_source, *out_shp)
    :param x_sensor:    Sensor positions (n_dim, n_sensor, *out_shp)
    :param v_sensor:    Sensor velocities (n_dim, n_sensor, *out_shp)
    :return n_dim:       Number of spatial dimensions
    :return n_source:    Number of sources
    :return n_sensor:    Number of sensors
    :return x_source:    Extended source positions (n_dim, n_source, *out_shp)
    :return v_source:    Extended source velocities (n_dim, n_source, *out_shp)
    :return x_sensor:    Extended sensor positions (n_dim, n_sensor, *out_shp)
    :return v_sensor:    Extended sensor velocities (n_dim, n_sensor, *out_shp)
    """

    if v_sensor is None and v_source is None:
        raise ValueError('At least one of either v_sensor or v_source must be defined to use FDOA.')
    elif v_sensor is None:
        v_sensor = np.zeros_like(x_sensor)
    elif v_source is None:
        v_source = np.zeros_like(x_source)

    # Broadcast -- first the source/sensor position and velocity starting at axis 0
    arrs, _ = broadcast_backwards([x_source, v_source], start_dim=0)
    x_source, v_source = arrs

    arrs, _ = broadcast_backwards([x_sensor, v_sensor], start_dim=0)
    x_sensor, v_sensor = arrs

    # Broadcast between them for any axes after the second
    arrs, in_shp = broadcast_backwards([x_source, v_source, x_sensor, v_sensor], start_dim=2)
    x_source, v_source, x_sensor, v_sensor = arrs

    # Find the dimensions -- we already know it's at least 2D, thanks to broadcast_backwards
    num_dims, num_sources, *out_shp_1 = np.shape(x_source)
    _, num_sensors, *_ = np.shape(x_sensor)

    # Find the output shape
    # in_shp = np.broadcast_shapes(out_shp_1, out_shp_2, out_shp_3, out_shp_4)
    out_shp = [num_sensors, num_sources]
    out_shp.extend(in_shp)

    return num_dims, num_sources, num_sensors, tuple(out_shp), x_source, v_source, x_sensor, v_sensor
