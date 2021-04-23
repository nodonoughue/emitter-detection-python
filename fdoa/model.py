import numpy as np
import utils
import matplotlib.pyplot as plt


def measurement(x_sensor, v_sensor, x_source, ref_idx=None):
    """
    # Computed range rate difference measurements, using the
    # final sensor as a common reference for all FDOA measurements.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    21 January 2021

    :param x_sensor: nDim x nSensor array of sensor positions
    :param v_sensor: nDim x nSensor array of sensor velocities
    :param x_source: nDim x n_source array of source positions
    :param ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings
    :return rrdoa: nSensor -1 x n_source array of RRDOA measurements
    """

    # Parse inputs
    n_dim1, n_sensor1 = np.shape(x_sensor)
    n_dim2, n_sensor2 = np.shape(v_sensor)
    n_dim3, n_source = np.shape(x_source)

    if n_dim1 != n_dim2 or n_sensor1 != n_sensor2:
        raise TypeError('Sensor position and velocity inputs must have matching shapes.')
    
    if n_dim1 != n_dim3:
        raise TypeError('First dimension of all inputs must match')
    
    test_idx_vec, ref_idx_vec = utils.parse_reference_sensor(ref_idx, n_sensor1)
    
    # Compute distance from each source position to each sensor
    dx = np.reshape(x_source, (n_dim1, 1, n_source)) - np.reshape(x_sensor, (n_dim1, n_sensor1, 1))
    r = np.sqrt(np.sum(dx**2, axis=0))  # 1 x nSensor x n_source
    
    # Compute range rate from range and velocity
    rr = np.reshape(np.sum(v_sensor*dx/r, axis=0), (n_sensor1, n_source))  # nSensor x n_source
    
    # Apply reference sensors to compute range rate difference for each sensor
    # pair
    rrdoa = rr[test_idx_vec, :] - rr[ref_idx_vec, :]

    return rrdoa


def jacobian(x_sensor, v_sensor, x_source, ref_idx=None):
    """
    # Returns the Jacobian matrix for FDOA of a source at x_source
    # (n_dim x n_source) from sensors at x_sensor (n_dim x n_sensor) with velocity
    # v_sensor.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    21 January 2021

    :param x_sensor: n_dim x n_sensor vector of sensor positions
    :param v_sensor: n_dim x n_sensor vector of sensor velocities
    :param x_source: n_dim x n_source vector of source positions
    :param ref_idx: Scalar index of reference sensor, or n_dim x nPair matrix of sensor pairings
    :return j: n_dim x nMeasurement x n_source matrix of Jacobians, one for each candidate source position
    """

    # Parse inputs
    n_dim1, n_sensor = np.shape(x_sensor)
    n_dim2, n_source = np.shape(x_source)
    
    if n_dim1 != n_dim2:
        raise TypeError('Input variables must match along first dimension.')

    n_dim = n_dim1

    test_idx_vec, ref_idx_vec = utils.parse_reference_sensor(ref_idx, n_sensor)

    # Compute the Offset Vectors
    dx = x_sensor - np.reshape(x_source, (n_dim, 1, n_source))  # n_dim x n_sensor x n_source
    rn = np.sqrt(np.sum(dx**2, axis=0))  # Euclidean norm for each offset vector
    dx_norm = dx / np.sqrt(np.sum(dx**2, axis=0))
    px = np.reshape(dx_norm, (n_dim, 1, n_sensor, n_source)) * np.reshape(dx_norm, (1, n_dim, n_sensor, n_source))
    # n_dim x n_dim x n_sensor
    
    # Compute the gradient of R_n
    nabla_rn = np.squeeze(np.sum((np.eye(n_dim) - px) * np.reshape(v_sensor/rn, (1, n_dim, n_sensor)), axis=1))
    # n_dim x n_sensor x n_source
    
    # Take the reference of each w.r.t. to the N-th
    return nabla_rn[:, test_idx_vec, :] - nabla_rn[:, ref_idx_vec, :]  # n_dim x nPair x n_source


def log_likelihood(x_fdoa, v_fdoa, rho_dot, cov, x_source, ref_idx=None):
    """
    # Computes the Log Likelihood for FDOA sensor measurement, given the
    # received measurement vector rho_dot, covariance matrix C,
    # and set of candidate source positions x_source.

    Ported from MATLAB Code.

    Nicholas O'Donoughue
    21 January 2021

    :param x_fdoa: Sensor positions [m]
    :param v_fdoa: Sensor velocities [m/s]
    :param rho_dot: FDOA measurement vector
    :param cov: FDOA measurement error covariance matrix
    :param x_source: Candidate source positions
    :param ref_idx: Scalar index of reference sensor, or n_dim x n_pair matrix of sensor pairings
    :return ell: Log-likelihood evaluated at each position x_source.
    """

    # Parse inputs
    n_dim, n_source_pos = np.shape(x_fdoa)
    ell = np.zeros(n_source_pos, 1)

    _, num_sensors = np.shape(x_fdoa)
    test_idx_vec, ref_idx_vec = utils.parse_reference_sensor(ref_idx, num_sensors)

    # Resample the covariance matrix
    cov_resample = utils.resample_covariance_matrix(cov, test_idx_vec, ref_idx_vec)
    cov_inv = np.linalg.pinv(cov_resample)
    
    for idx_source in np.arange(n_source_pos):
        x_i = x_source[:, idx_source]
        
        # Generate the ideal measurement matrix for this position
        r_dot = measurement(x_fdoa, v_fdoa, x_i, ref_idx)
        
        # Evaluate the measurement error
        err = (rho_dot - r_dot)
        
        # Compute the scaled log likelihood
        ell[idx_source] = -err.dot(cov_inv).dot(err)

    return ell


def error(x_sensor, v_sensor, cov, x_source, x_max, num_pts, ref_idx=None):
    """
    Construct a 2-D field from -x_max to +x_max, using numPts in each
    dimension.  For each point, compute the FDOA solution for each sensor
    against the reference (the first sensor), and compare to the FDOA
    solution from the true emitter position.

    Ported from MATLAB Code.

    Nicholas O'Donoughue
    21 January 2021

    :param x_sensor: nDim x N matrix of sensor positions
    :param v_sensor: nDim x N matrix of sensor velocities
    :param cov: N x N covariance matrix
    :param x_source: nDim x 1 matrix of true emitter position
    :param x_max: nDim x 1 (or scalar) vector of maximum offset from origin for plotting
    :param num_pts: Number of test points along each dimension
    :param ref_idx: Scalar index of reference sensor, or n_dim x n_pair matrix of sensor pairings
    :return epsilon: 2-D plot of FDOA error
    :return x_vec:
    :return y_vec:
    """

    # Compute the True FDOA measurement
    # Compute true range rate difference measurements default condition is to
    # use the final sensor as the reference for all difference measurements.
    rr = measurement(x_sensor, v_sensor, x_source, ref_idx)

    _, num_sensors = np.shape(x_sensor)
    test_idx_vec, ref_idx_vec = utils.parse_reference_sensor(ref_idx, num_sensors)

    # Resample the covariance matrix
    cov_resample = utils.resample_covariance_matrix(cov, test_idx_vec, ref_idx_vec)
    cov_inv = np.linalg.pinv(cov_resample)

    # Set up test points
    xx_vec = x_max.flatten() * np.reshape(np.linspace(start=-1, stop=1, num=num_pts), (1, num_pts))
    x_vec = xx_vec[0, :]
    y_vec = xx_vec[1, :]
    xx, yy = np.mgrid(x_vec, y_vec)
    x_plot = np.vstack(xx.flatten(), yy.flatten()).T  # 2 x numPts^2
    
    epsilon = np.zeros_like(xx)
    for idx_pt in np.arange(np.size(xx)):
        x_i = x_plot[:, idx_pt]
        
        # Evaluate the measurement at x_i
        rr_i = measurement(x_sensor, v_sensor, x_i, ref_idx)
        
        # Compute the measurement error
        err = rr-rr_i
        
        # Evaluate the scaled log likelihood
        epsilon[idx_pt] = err.H.dot(cov_inv).dot(err)

    return epsilon


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
    f_0 = 3.0e8
    
    # Set up test points
    xx_vec = max_ortho.flatten() * np.reshape(np.linspace(start=-1, stop=1, num=num_pts), (1, num_pts))
    x_vec = xx_vec[0, :]
    y_vec = xx_vec[1, :]
    xx, yy = np.mgrid(x_vec, y_vec)
    x_plot = np.vstack(xx.flatten(), yy.flatten()).T  # 2 x numPts^2
    
    df_plot = utils.geo.calc_doppler_diff(x_plot, np.array([0, 0]).T, x1, v1, x2, v2, f_0)  # numPts^2 x (N-1)

    # Compute contour
    fig00 = plt.figure()
    contour_set = plt.contour(x_vec, y_vec, np.reshape(df_plot, (num_pts, num_pts)), [vdiff, vdiff])

    # Close the figure generated
    plt.close(fig00)

    # Extract the desired coordinates
    contour_data = contour_set.allSegs[0]

    x_iso = contour_data[:, 0]
    y_iso = contour_data[:, 1]
    
    # Filter points out of bounds
    out_of_bounds = abs(x_iso) > max_ortho | abs(y_iso) > max_ortho
    x_iso = x_iso(~out_of_bounds)
    y_iso = y_iso(~out_of_bounds)

    return x_iso, y_iso
