import numpy as np
import utils
import matplotlib.pyplot as plt


def measurement(x_sensor, x_source, v_sensor=None, v_source=None, ref_idx=None):
    """
    # Computed range rate difference measurements, using the
    # final sensor as a common reference for all FDOA measurements.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    21 January 2021

    :param x_sensor: nDim x nSensor array of sensor positions
    :param x_source: nDim x n_source array of source positions
    :param v_sensor: nDim x nSensor array of sensor velocities
    :param v_source: nDim x n_source array of source velocities
    :param ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings
    :return rrdoa: nSensor -1 x n_source array of RRDOA measurements
    """

    # Parse inputs
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
    test_idx_vec, ref_idx_vec = utils.parse_reference_sensor(ref_idx, n_sensor2)
    
    # Compute distance from each source position to each sensor
    dx = np.reshape(x_source, (n_dim1, 1, n_source1)) - np.reshape(x_sensor, (n_dim1, n_sensor1, 1))
    r = np.sqrt(np.sum(dx**2, axis=0))  # 1 x nSensor1 x n_source1
    
    # Compute range rate from range and velocity
    dv = np.reshape(v_sensor, (n_dim1, n_sensor2, 1)) - np.reshape(v_source, (n_dim1, 1, n_source2))
    rr = np.reshape(np.sum(dv*dx/r, axis=0), (n_sensor, n_source))  # nSensor x n_source
    
    # Apply reference sensors to compute range rate difference for each sensor
    # pair
    if n_source1 > 1 or n_source2 > 1:
        # There are multiple sources; they must traverse the second dimension
        out_dims = (np.size(test_idx_vec), np.amax((n_source1, n_source2)))
    else:
        # Single source, make it an array
        out_dims = (np.size(test_idx_vec), )

    rrdoa = np.reshape(rr[test_idx_vec, :] - rr[ref_idx_vec, :], newshape=out_dims)

    return rrdoa


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

    # Parse inputs
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

    n_dim = n_dim1
    n_source = np.amax((n_source1, n_source2))
    n_sensor = np.amax((n_sensor1, n_sensor2))

    test_idx_vec, ref_idx_vec = utils.parse_reference_sensor(ref_idx, n_sensor1)

    # Compute the Offset Vectors
    dx = x_sensor[:, :, np.newaxis] - np.reshape(x_source, (n_dim, 1, n_source1))  # n_dim x n_sensor x n_source
    rn = np.reshape(np.sqrt(np.sum(dx**2, axis=0)), (1, n_sensor1, n_source1))  # Euclidean norm for each offset vector
    dx_norm = dx / rn  # n_dim x n_sensor x n_source
    px = np.reshape(dx_norm, (n_dim, 1, n_sensor1, n_source1)) * np.reshape(dx_norm, (1, n_dim, n_sensor1, n_source1))
    # n_dim x n_dim x n_sensor x n_source
    
    # Compute the gradient of R_n
    dv = np.reshape(v_sensor, (n_dim, n_sensor2, 1)) - np.reshape(v_source, (n_dim, 1, n_source2))  # n_dim x n_sensor x n_source
    dv_norm = dv/rn  # n_dim x n_sensor x n_source
    # Iterate the matmul over the number of sensors and sources
    nabla_rn = np.asarray([[np.dot(np.eye(n_dim) - px[:, :, idx_sen, idx_src],
                                   dv_norm[:, idx_sen, idx_src])
                            for idx_src in np.arange(n_source)] for idx_sen in np.arange(n_sensor)])

    # Rearrange the axes to match expectation (n_dim x n_sensor x n_source)
    nabla_rn = np.moveaxis(nabla_rn, source=2, destination=0)

    # Compute test/reference differences and reshape output
    n_msmt = test_idx_vec.size
    if n_source > 1:
        out_dims = (n_dim, n_msmt, n_source)
    else:
        out_dims = (n_dim, n_msmt)

    result = np.reshape(nabla_rn[:, test_idx_vec, :] - nabla_rn[:, ref_idx_vec, :], newshape=out_dims)
    #result = np.delete(result, np.unique(ref_idx_vec), axis=1) # rm ref_sensor column b/c all zeros # not needed when parse_ref_sensors is fixed
    return result  # n_dim x nPair x n_source


def log_likelihood(x_sensor, rho_dot, cov, x_source, v_sensor=None, v_source=None, ref_idx=None, do_resample=False):
    """
    # Computes the Log Likelihood for FDOA sensor measurement, given the
    # received measurement vector rho_dot, covariance matrix C,
    # and set of candidate source positions x_source.

    Ported from MATLAB Code.

    Nicholas O'Donoughue
    21 January 2021

    :param x_sensor: Sensor positions [m]
    :param rho_dot: FDOA measurement vector
    :param cov: FDOA measurement error covariance matrix
    :param x_source: Candidate source positions
    :param v_sensor: Sensor velocities [m/s]
    :param v_source: n_dim x n_source vector of source velocities
    :param ref_idx: Scalar index of reference sensor, or n_dim x n_pair matrix of sensor pairings
    :param do_resample: If true, cov is a sensor-level covariance matrix and must be resampled
    :return ell: Log-likelihood evaluated at each position x_source.
    """

    # Parse inputs
    n_dim, n_sensors = utils.safe_2d_shape(x_sensor)
    n_dim2, n_source_pos = utils.safe_2d_shape(x_source)
    assert n_dim == n_dim2, 'Input dimension mismatch.'

    # Handle vector input
    if n_source_pos == 1 and len(x_source.shape) == 1:
        # x_source is a vector; 2D indexing below will fail.
        # Let's add a newaxis
        x_source = x_source[:, np.newaxis]

    # Resample the covariance matrix
    if do_resample:
        cov = utils.resample_covariance_matrix(cov, ref_idx)

    if np.isscalar(cov):
        # cov_lower = 1/cov
        cov = np.array([[cov]])
    # Pre-invert the covariance matrix; to avoid repeatedly redoing the same work in the
    # for loop over source positions
    cov_inv = np.linalg.pinv(cov)

    # Initialize output
    ell = np.zeros((n_source_pos, ))

    for idx_source in np.arange(n_source_pos):
        x_i = x_source[:, idx_source]
        
        # Generate the ideal measurement matrix for this position
        r_dot = measurement(x_sensor=x_sensor, x_source=x_i,
                            v_sensor=v_sensor, v_source=v_source,
                            ref_idx=ref_idx)
        
        # Evaluate the measurement error
        err = (rho_dot - r_dot)
        
        # Compute the scaled log likelihood
        ell[idx_source] = -np.squeeze(err.T @ cov_inv @ err)

    return ell


def error(x_sensor, cov, x_source, x_max, num_pts, v_sensor=None, v_source=None, ref_idx=None, do_resample=False):
    """
    Construct a 2-D field from -x_max to +x_max, using numPts in each
    dimension.  For each point, compute the FDOA solution for each sensor
    against the reference (the first sensor), and compare to the FDOA
    solution from the true emitter position.

    Ported from MATLAB Code.

    Nicholas O'Donoughue
    21 January 2021

    :param x_sensor: nDim x N matrix of sensor positions
    :param cov: N x N covariance matrix
    :param x_source: nDim x 1 matrix of true emitter position
    :param x_max: nDim x 1 (or scalar) vector of maximum offset from origin for plotting
    :param num_pts: Number of test points along each dimension
    :param v_sensor: nDim x N matrix of sensor velocities
    :param v_source: nDim x 1 matrix of true emitter velocity
    :param ref_idx: Scalar index of reference sensor, or n_dim x n_pair matrix of sensor pairings
    :param do_resample: If true, cov is a sensor-level covariance matrix and must be resampled
    :return epsilon: 2-D plot of FDOA error
    :return x_vec:
    :return y_vec:
    """

    # Compute the True FDOA measurement
    # Compute true range rate difference measurements default condition is to
    # use the final sensor as the reference for all difference measurements.
    rr = measurement(x_sensor=x_sensor, x_source=x_source,
                     v_sensor=v_sensor, v_source=v_source,
                     ref_idx=ref_idx)

    # Resample the covariance matrix
    if do_resample:
        cov = utils.resample_covariance_matrix(cov, ref_idx)

    # Pre-invert the covariance matrix, to avoid repeatedly doing the same calculation
    cov_inv = np.linalg.pinv(cov)

    # Set up test points
    grid_res = 2*x_max / (num_pts-1)
    x_set, x_grid, grid_shape = utils.make_nd_grid(x_ctr=(0., 0.), max_offset=x_max, grid_spacing=grid_res)
    x_vec = x_grid[0][0, :]
    y_vec = x_grid[1][:, 0]

    rr_list = measurement(x_sensor=x_sensor, x_source=x_set.T,
                          v_sensor=v_sensor, v_source=v_source,
                          ref_idx=ref_idx)

    err = rr[:, np.newaxis] - rr_list

    epsilon_list = [np.conjugate(this_err).T @  cov_inv @ this_err for this_err in err.T]

    return np.reshape(epsilon_list, grid_shape), x_vec, y_vec


def draw_isodop(x1, v1, x2, v2, vdiff, num_pts, max_ortho):
    """
    # Finds the isochrone with the stated range rate difference from points x1
    # and x2.  Generates an arc with 2*numPts-1 points, that spans up to
    # maxOrtho distance from the intersection line of x1 and x2

    Ported from MATLAB Code.

    Nicholas O'Donoughue
    21 January 2021

    :param x1: Position of first sensor (Ndim x 1) [m]
    :param v1: Velocity vector of first sensor (Ndim x 1) [m/s]
    :param x2: Position of second sensor (Ndim x 1) [m]
    :param v2: Velocity vector of second sensor (Ndim x 1) [m/s]
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
    x_set, x_grid, grid_shape = utils.make_nd_grid(x_ctr=(0., 0.),
                                                   max_offset=max_ortho,
                                                   grid_spacing=grid_spacing)

    df_plot = utils.geo.calc_doppler_diff(x_set.T, np.zeros_like(x_set).T, x1, v1, x2, v2, f_0)

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
