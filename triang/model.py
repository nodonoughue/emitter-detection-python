import numpy as np
from scipy.linalg import solve_triangular
import utils


def measurement(x_sensor, x_source, do_2d_aoa=False):
    """
    Computes angle of arrival measurements.

    Modified from the original version to include a flag for 2D AOA calculations (where both az/el are computed). If
    specified, the assumption is that the measurements are arranged with the n_sensor azimuth measurements first, and
    the n_sensor elevation measurements second.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    5 September 2021

    :param x_sensor: nDim x nSensor array of sensor positions
    :param x_source: nDim x n_source array of source positions
    :param do_2d_aoa: Optional boolean parameter specifying whether 1D (az-only) or 2D (az/el) AOA is being performed
    :return psi: nSensor -1 x n_source array of AOA measurements
    """

    # Parse inputs
    n_dim1, n_sensor = utils.safe_2d_shape(x_sensor)
    n_dim2, n_source = utils.safe_2d_shape(x_source)
    if n_source > 1:
        out_dims = (n_sensor, n_source)
    else:
        out_dims = (n_sensor, )

    if n_dim1 != n_dim2:
        raise TypeError('First dimension of all inputs must match')

    # Compute cartesian offset from each source position to each sensor
    dx = np.reshape(x_source, (n_dim1, 1, n_source)) - np.reshape(x_sensor, (n_dim1, n_sensor, 1))

    # Compute angle in radians
    az = np.reshape(np.arctan2(dx[1, :, :], dx[0, :, :]), newshape=out_dims)

    # Elevation angle, if desired
    if do_2d_aoa and n_dim1 == 3:
        ground_rng = np.expand_dims(np.sqrt(np.sum(dx[0:1, :, :]**2, axis=0)), axis=0)
        el = np.reshape(np.arctan2(dx[2, :, :], ground_rng), newshape=out_dims)

        # Stack az/el along the first dimension
        psi = np.concatenate((az, el), axis=0)
    else:
        # Just return az
        psi = az

    return psi


def jacobian(x_sensor, x_source, do_2d_aoa=False):
    """
    Returns the Jacobian matrix for triangulation of a source at x_source (nDim x nSource) from sensors at x_sensor
    (nDim x nSensor)

    Modified from the original version to include a flag for 2D AOA calculations (where both az/el are computed). If
    specified, the assumption is that the measurements are arranged with the n_sensor azimuth measurements first, and
    the n_sensor elevation measurements second.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    5 September 2021

    :param x_sensor: n_dim x n_sensor vector of sensor positions
    :param x_source: n_dim x n_source vector of source positions
    :param do_2d_aoa: Optional boolean parameter specifying whether 1D (az-only) or 2D (az/el) AOA is being performed
    :return j: n_dim x nMeasurement x n_source collection of Jacobian matrices, one for each candidate source position
    """

    # Parse inputs
    n_dim1, n_sensor = utils.safe_2d_shape(x_sensor)
    n_dim2, n_source = utils.safe_2d_shape(x_source)

    if n_dim1 != n_dim2:
        raise TypeError('Input variables must match along first dimension.')

    n_dim = n_dim1
    if n_source > 1:
        out_dims = (2, n_sensor, n_source)
    else:
        out_dims = (2, n_sensor)

    # Turn off divide by zero warning
    np.seterr(all='ignore')

    # Compute the Offset Vectors
    dx = (np.reshape(x_source, (n_dim, 1, n_source))
          - np.reshape(x_sensor, (n_dim, n_sensor, 1)))  # n_dim x n_sensor x n_source
    slant_range_sq = np.sum(dx ** 2, axis=0)  # Euclidean norm-squared for each offset vector, n_sensor x n_source
    ground_range_sq = np.sum(dx[0:2, :, :]**2, axis=0)  # Ground grange squared, n_sensor x n_source

    # Grab the x and y components of dx
    dxx = dx[0, :, :]  # 1 x n_sensor x n_source
    dxy = dx[1, :, :]  # 1 x n_sensor x n_source

    # Build Jacobian for azimuth measurements ( 2 x n_sensor x n_source)
    j = np.concatenate((-dxy[np.newaxis, :], dxx[np.newaxis, :]), axis=0)/ground_range_sq[np.newaxis, :]

    # Repeat for elevation angle, if necessary
    if do_2d_aoa and n_dim == 3:
        if n_source > 1:
            out_dims = (n_dim, 2*n_sensor, n_source)
        else:
            out_dims = (n_dim, 2*n_sensor)

        # Add a z dimension to the azimuth Jacobian
        j = np.concatenate((j, np.zeros(shape=(1, n_sensor, n_source))), axis=0)

        # Get the z component of dx
        dxz = dx[2, :, :]  # 1 x n_sensor x n_source

        # Jacobian for elevation angle
        # del_x(phi_n(x)) = -(x-xn)(z-zn)/ ground_range * slant_range^2
        # del_y(phi_n(x)) = -(y-yn)(z-zn)/ ground_range * slant_range^2
        # del_z(phi_n(x)) = ground_range / slant_range^2
        j_el = np.concatenate((-dxx*dxz/np.sqrt(ground_range_sq),
                               -dxy*dxz/np.sqrt(ground_range_sq),
                               np.sqrt(ground_range_sq)), axis=0)/slant_range_sq

        # The elevation measurements are concatenated after the azimuth
        # measurements, so the Jacobian is concatenated in the second (nSensor)
        # dimension
        j = np.concatenate((j, j_el), axis=1)

    # Reactive warning
    np.seterr(all='warn')

    j = np.reshape(j, out_dims)

    return j


def log_likelihood(x_aoa, psi, cov, x_source, do_2d_aoa=False):
    """
    Computes the Log Likelihood for AOA sensor measurement, given the
    received measurement vector psi, covariance matrix cov,
    and set of candidate source positions x_source.

    Modified from the original version to include a flag for 2D AOA calculations (where both az/el are computed). If
    specified, the assumption is that the measurements are arranged with the n_sensor azimuth measurements first, and
    the n_sensor elevation measurements second.

    Ported from MATLAB Code.

    Nicholas O'Donoughue
    5 September 2021

    :param x_aoa: Sensor positions [m]
    :param psi: AOA measurement vector
    :param cov: FDOA measurement error covariance matrix
    :param x_source: Candidate source positions
    :param do_2d_aoa: Optional boolean parameter specifying whether 1D (az-only) or 2D (az/el) AOA is being performed
    :return ell: Log-likelihood evaluated at each position x_source.
    """

    # Parse inputs
    n_dim1, n_sensor = utils.safe_2d_shape(x_aoa)
    n_dim2, n_source_pos = utils.safe_2d_shape(x_source)

    if n_dim1 != n_dim2:
        raise TypeError('Input variables must match along first dimension.')

    if n_source_pos == 1:
        x_source = np.expand_dims(x_source, axis=1)

    # Pre-invert covariance matrix for speedup
    if np.isscalar(cov):
        # cov_lower = 1/cov
        cov = np.array([[cov]])
    
    cov_lower = np.linalg.cholesky(cov)

    # Initialize Output
    ell = np.zeros(shape=(n_source_pos, ))

    for idx_source in np.arange(n_source_pos):
        x_i = x_source[:, idx_source]

        # Generate the ideal measurement matrix for this position
        this_psi = measurement(x_aoa, x_i, do_2d_aoa)

        # Evaluate the measurement error
        err = utils.modulo2pi(psi - this_psi)

        # Compute the scaled log likelihood
        a = solve_triangular(cov_lower, err)
        ell[idx_source] = - np.conj(a.T) @ a

    return ell


def error(x_sensor, cov, x_source, x_max, num_pts, do_2d_aoa):
    """
    Construct a 2-D field from -x_max to +x_max, using numPts in each
    dimension.  For each point, compute the AOA solution for each sensor, and
    compare to the AOA solution from the true emitter position.

    Modified from the original version to include a flag for 2D AOA calculations (where both az/el are computed). If
    specified, the assumption is that the measurements are arranged with the n_sensor azimuth measurements first, and
    the n_sensor elevation measurements second.

    Ported from MATLAB Code.

    Nicholas O'Donoughue
    5 September 2021

    :param x_sensor: nDim x N matrix of sensor positions
    :param cov: N x N covariance matrix
    :param x_source: nDim x 1 matrix of true emitter position
    :param x_max: nDim x 1 (or scalar) vector of maximum offset from origin for plotting
    :param num_pts: Number of test points along each dimension
    :param do_2d_aoa: Optional boolean parameter specifying whether 1D (az-only) or 2D (az/el) AOA is being performed
    :return epsilon: 2-D plot of AOA error
    :return x_vec:
    :return y_vec:
    """

    n_dim1, n_sensors = utils.safe_2d_shape(x_sensor)
    n_dim2, n_source = utils.safe_2d_shape(x_source)

    if n_dim1 != n_dim2:
        raise TypeError('Input variables must match along first dimension.')

    # Compute the True AOA measurement
    psi = measurement(x_sensor, x_source, do_2d_aoa)

    # Decompose the covariance matrix for speedup
    cov_lower = np.linalg.cholesky(cov)

    # Set up test points
    xx_vec = np.expand_dims(x_max, axis=1) * np.reshape(np.linspace(start=-1, stop=1, num=num_pts), (1, num_pts))
    x_vec = xx_vec[0, :]
    y_vec = xx_vec[1, :]
    xx, yy = np.meshgrid(x_vec, y_vec)
    x_plot = np.stack((xx.flatten(), yy.flatten()), axis=1).T  # 2 x numPts^2

    epsilon = np.zeros_like(xx)
    for idx_pt in np.arange(np.size(xx)):
        x_i = x_plot[:, idx_pt]

        # Evaluate the measurement at x_i
        psi_i = measurement(x_sensor, x_i, do_2d_aoa)

        # Compute the measurement error
        err = utils.modulo2pi(psi - psi_i)

        # Evaluate the scaled log likelihood
        a = solve_triangular(cov_lower, err)
        epsilon[idx_pt] = np.conj(a.T) @ a

    return epsilon


def draw_lob(x_sensor, psi, x_source=None, scale=1):
    num_sensors = x_sensor.shape[1]
    num_measurements = len(psi)

    if num_sensors != num_measurements:
        print('The number of sensor positions and measurements must match.\n')
        num_measurements = np.min([num_sensors, num_measurements])
        x_sensor = x_sensor[:num_measurements]
        psi = psi[:num_measurements]
    else:
        num_measurements = num_sensors

    if x_source is None:
        range_to_source = 1
    else:
        # range = utils.rng(x_sensor, x_source)
        range_to_source = utils.geo.calc_range(x_sensor, x_source)

    x_end = np.cos(psi) * range_to_source * scale
    y_end = np.sin(psi) * range_to_source * scale

    xy_end = np.vstack([np.reshape(x_end, [1, 1, num_measurements]),
                        np.reshape(y_end, [1, 1, num_measurements])])
    xy_start = np.zeros([2, 1, num_measurements])
    xy_lob_centered = np.hstack([xy_start,
                                 xy_end
                                 ])
    xy_lob = np.reshape(x_sensor, [2, 1, num_measurements]) + xy_lob_centered

    return xy_lob
