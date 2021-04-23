import numpy as np


def measurement(x_sensor, x_source):
    """
    Computes angle of arrival measurements.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    22 February 2021

    :param x_sensor: nDim x nSensor array of sensor positions
    :param x_source: nDim x n_source array of source positions
    :return psi: nSensor -1 x n_source array of AOA measurements
    """

    # Parse inputs
    n_dim1, n_sensor = np.shape(x_sensor)
    n_dim2, n_source = np.shape(x_source)

    if n_dim1 != n_dim2:
        raise TypeError('First dimension of all inputs must match')

    # Compute cartesian offset from each source position to each sensor
    dx = np.reshape(x_source, (n_dim1, 1, n_source)) - np.reshape(x_sensor, (n_dim1, n_sensor, 1))

    # Compute angle in radians
    psi = np.reshape(np.arctan2(dx[1, :, :], dx[0, :, :]), (n_sensor, n_source))  # nSensor x n_source

    return psi


def jacobian(x_sensor, x_source):
    """
    Returns the Jacobian matrix for triangulation of a source at x_source (nDim x nSource) from sensors at x_sensor
    (nDim x nSensor)

    Ported from MATLAB Code

    Nicholas O'Donoughue
    22 February 2021

    :param x_sensor: n_dim x n_sensor vector of sensor positions
    :param x_source: n_dim x n_source vector of source positions
    :return j: n_dim x nMeasurement x n_source matrix of Jacobians, one for each candidate source position
    """

    # Parse inputs
    n_dim1, n_sensor = np.shape(x_sensor)
    n_dim2, n_source = np.shape(x_source)

    if n_dim1 != n_dim2:
        raise TypeError('Input variables must match along first dimension.')

    n_dim = n_dim1

    # Compute the Offset Vectors
    dx = np.reshape(x_source, (n_dim, 1, n_source)) - x_sensor  # n_dim x n_sensor x n_source
    r_sq = np.sum(dx ** 2, axis=0)  # Euclidean norm-squared for each offset vector

    # Compute the Jacobian
    return np.concatenate(np.ones_like(dx[1, :, :]), -dx[1, :, :], dx[0, :, :]) / r_sq


def log_likelihood(x_aoa, psi, cov, x_source):
    """
    Computes the Log Likelihood for AOA sensor measurement, given the
    received measurement vector psi, covariance matrix cov,
    and set of candidate source positions x_source.

    Ported from MATLAB Code.

    Nicholas O'Donoughue
    22 February 2021

    :param x_aoa: Sensor positions [m]
    :param psi: AOA measurement vector
    :param cov: FDOA measurement error covariance matrix
    :param x_source: Candidate source positions
    :return ell: Log-likelihood evaluated at each position x_source.
    """

    # Parse inputs
    n_dim1, n_sensor = np.shape(x_aoa)
    n_dim2, n_source_pos = np.shape(x_source)

    if n_dim1 != n_dim2:
        raise TypeError('Input variables must match along first dimension.')

    # Pre-invert covariance matrix for speedup
    cov_inv = np.linalg.pinv(cov)

    # Initialize Output
    ell = np.zeros(n_source_pos, 1)

    for idx_source in np.arange(n_source_pos):
        x_i = x_source[:, idx_source]

        # Generate the ideal measurement matrix for this position
        this_psi = measurement(x_aoa, x_i)

        # Evaluate the measurement error
        err = (psi - this_psi)

        # Compute the scaled log likelihood
        ell[idx_source] = -err.H.dot(cov_inv).dot(err)

    return ell


def error(x_sensor, cov, x_source, x_max, num_pts):
    """
    Construct a 2-D field from -x_max to +x_max, using numPts in each
    dimension.  For each point, compute the AOA solution for each sensor, and
    compare to the AOA solution from the true emitter position.

    Ported from MATLAB Code.

    Nicholas O'Donoughue
    22 February 2021

    :param x_sensor: nDim x N matrix of sensor positions
    :param cov: N x N covariance matrix
    :param x_source: nDim x 1 matrix of true emitter position
    :param x_max: nDim x 1 (or scalar) vector of maximum offset from origin for plotting
    :param num_pts: Number of test points along each dimension
    :return epsilon: 2-D plot of AOA error
    :return x_vec:
    :return y_vec:
    """

    n_dim1, n_sensors = np.shape(x_sensor)
    n_dim2, n_sourse = np.shape(x_source)

    if n_dim1 != n_dim2:
        raise TypeError('Input variables must match along first dimension.')

    # Compute the True AOA measurement
    psi = measurement(x_sensor, x_source)

    # Pre-invert the covariance matrix for speedup
    cov_inv = np.linalg.pinv(cov)

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
        psi_i = measurement(x_sensor, x_i)

        # Compute the measurement error
        err = psi - psi_i

        # Evaluate the scaled log likelihood
        epsilon[idx_pt] = err.H.dot(cov_inv).dot(err)

    return epsilon
