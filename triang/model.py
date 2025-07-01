import numpy as np
import utils
from utils.covariance import CovarianceMatrix


def measurement(x_sensor, x_source, do_2d_aoa=False, bias=None):
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
    :param bias:  Optional nSensor x 1 vector of AOA biases (nSensor x 2 if do2DAoA is true).  [default = None]
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

    # Check angle bias dimensions and parse
    angle_bias_az = 0.
    angle_bias_el = 0.
    if bias is not None:
        n_dim_bias, n_bias = utils.safe_2d_shape(bias)
        # Check for vector input
        if len(np.shape(bias)):
            n_bias = n_dim_bias
            n_dim_bias = 1
        if (do_2d_aoa and n_dim_bias != 2) or (not do_2d_aoa and n_dim_bias != 1) or n_bias != n_sensor:
            raise TypeError('Angle bias dimensions must match number of sensor measurements to make.')

        if do_2d_aoa:
            angle_bias_az = np.reshape(bias[0], (1, n_sensor, n_source))
            angle_bias_el = np.reshape(bias[1], (1, n_sensor, n_source))
        else:
            angle_bias_az = np.reshape(bias, (1, n_sensor, n_source))

    # Compute cartesian offset from each source position to each sensor
    dx = np.reshape(x_source, (n_dim1, 1, n_source)) - np.reshape(x_sensor, (n_dim1, n_sensor, 1))

    # Compute angle in radians
    az = np.reshape(np.arctan2(dx[1, :, :], dx[0, :, :]) + angle_bias_az, newshape=out_dims)

    # Elevation angle, if desired
    if do_2d_aoa and n_dim1 == 3:
        ground_rng = np.expand_dims(np.sqrt(np.sum(dx[0:2, :, :]**2, axis=0)), axis=0)
        el = np.reshape(np.arctan2(dx[2, :, :], ground_rng) + angle_bias_el, newshape=out_dims)

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
        j_el = np.concatenate((-dxx[np.newaxis, :]*dxz[np.newaxis, :]/np.sqrt(ground_range_sq[np.newaxis, :]),
                               -dxy*dxz/np.sqrt(ground_range_sq[np.newaxis, :]),
                               np.sqrt(ground_range_sq[np.newaxis, :])), axis=0)/slant_range_sq[np.newaxis, :]
        # 1 x n_sensor x n_source

        # The elevation measurements are concatenated after the azimuth
        # measurements, so the Jacobian is concatenated in the second (nSensor)
        # dimension
        j_el = np.reshape(j_el, (n_dim, n_sensor, n_source))
        j = np.concatenate((j, j_el), axis=1)

    # Reactive warning
    np.seterr(all='warn')

    j = np.reshape(j, out_dims)

    return j

def jacobian_uncertainty(x_sensor, x_source, do_2d_aoa=False, do_bias=False, do_pos_error=False):
    """
    Returns the Jacobian matrix for a set of AOA measurements in the presence of sensor
    uncertainty, in the form of measurement bias and/or sensor position errors.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    30 April 2025

    :param x_sensor: nDim x n_sensor array of sensor positions
    :param x_source: nDim x n_source array of source positions
    :param do_bias: if True, jacobian includes gradient w.r.t. measurement biases
    :param do_pos_error: if True, jacobian includes gradient w.r.t. sensor pos/vel errors
    :return: n_dim x nMeasurement x n_source matrix of Jacobians, one for each candidate source position
    """

    # Parse inputs
    n_dim1, _ = utils.safe_2d_shape(x_sensor)
    n_dim2, n_source = utils.safe_2d_shape(x_source)

    if n_dim1 != n_dim2:
        raise TypeError('Input variables must match along first dimension.')

    # Gradient w.r.t source position
    j_source = grad_x(x_sensor, x_source, do_2d_aoa)
    j_list = [j_source]

    # Gradient w.r.t measurement biases
    if do_bias:
        j_bias = grad_bias(x_sensor, x_source, do_2d_aoa)
        j_list.append(j_bias)

    # Gradient w.r.t sensor position
    if do_pos_error:
        j_sensor_pos = grad_sensor_pos(x_sensor, x_source, do_2d_aoa)
        j_list.append(j_sensor_pos)

    # Combine component Jacobians
    j = np.concatenate(j_list, axis=0)

    return j

def log_likelihood(x_sensor, zeta, cov: CovarianceMatrix, x_source, do_2d_aoa=False, bias=None):
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

    :param x_sensor: Sensor positions [m]
    :param zeta: AOA measurement vector
    :param cov: AOA measurement error covariance matrix; object of the CovarianceMatrix class
    :param x_source: Candidate source positions
    :param do_2d_aoa: Optional boolean parameter specifying whether 1D (az-only) or 2D (az/el) AOA is being performed
    :return ell: Log-likelihood evaluated at each position x_source.
    """

    # Parse inputs
    n_dim1, n_sensor = utils.safe_2d_shape(x_sensor)
    n_dim2, n_source_pos = utils.safe_2d_shape(x_source)

    if n_dim1 != n_dim2:
        raise TypeError('Input variables must match along first dimension.')

    if n_source_pos == 1:
        x_source = np.expand_dims(x_source, axis=1)

    # Initialize Output
    ell = np.zeros(shape=(n_source_pos, ))

    for idx_source in np.arange(n_source_pos):
        x_i = x_source[:, idx_source]

        # Generate the ideal measurement matrix for this position
        this_psi = measurement(x_sensor, x_i, do_2d_aoa, bias=bias)

        # Evaluate the measurement error
        err = utils.modulo2pi(zeta - this_psi)

        # Compute the scaled log likelihood
        ell[idx_source] = - cov.solve_aca(err)

    return ell


def error(x_sensor, cov: CovarianceMatrix, x_source, x_max, num_pts, do_2d_aoa=False):
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
    :param cov: AOA measurement error covariance matrix; object of the CovarianceMatrix class
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
        # a = solve_triangular(cov_lower, err, lower=True)
        # epsilon[idx_pt] = np.sum(a**2)
        epsilon[idx_pt] = cov.solve_aca(err)

    return epsilon


def draw_lob(x_sensor, psi, x_source=None, scale=1):
    # TODO: Expand for 3D LOBs
    _, num_sensors = utils.safe_2d_shape(x_sensor)
    num_measurements = np.size(psi)

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


def grad_x(x_sensor, x_source, do_2d_aoa=False):
    """
    Return the gradient of AOA measurements, with sensor uncertainties, with respect to target position, x.
    Equation 6.16. The sensor uncertainties don't impact the gradient for AOA, so this reduces to the previously
    defined Jacobian. This function is merely a wrapper for calls to triang.model.jacobian, with the optional argument
    'bias' ignored.

    Ported from MATLAB code.

    Nicholas O'Donoughue
    14 April 2025

    :param x_sensor:    AOA sensor positions
    :param x_source:    Source positions
    :param do_2d_aoa: Optional boolean parameter specifying whether 1D (az-only) or 2D (az/el) AOA is being performed
    :return jacobian:   Jacobian matrix representing the desired gradient
    """
    # TODO: Debug

    # Sensor uncertainties don't impact the gradient with respect to target position; this is the same as the previously
    # defined function triang.model.jacobian.
    return jacobian(x_sensor=x_sensor, x_source=x_source, do_2d_aoa=do_2d_aoa)


def grad_bias(x_sensor, x_source, do_2d_aoa=False):
    """
    Return the gradient of AOA measurements, with sensor uncertainties, with respect to the unknown measurement bias
    terms, from equation 6.24.

    Ported from MATLAB code.

    Nicholas O'Donoughue
    14 April 2025

    :param x_sensor:    TDOA sensor positions
    :param x_source:    Source positions
    :param do_2d_aoa: Optional boolean parameter specifying whether 1D (az-only) or 2D (az/el) AOA is being performed
    :return jacobian:   Jacobian matrix representing the desired gradient
    """
    # TODO: Debug

    # Parse the reference index
    num_dim, num_sensors = utils.safe_2d_shape(x_sensor)
    _, num_sources = utils.safe_2d_shape(x_source)

    # According to eq 6.32, the m-th row is 1 for every column in which the m-th sensor is a test index, and -1 for
    # every column in which the m-th sensor is a reference index.
    num_measurements = num_sensors * (1 + do_2d_aoa)
    grad = np.eye(num_measurements)

    # Repeat for each source position
    _, num_sources = utils.safe_2d_shape(x_source)
    if num_sources > 1:
        grad = np.repeat(grad, num_sources, axis=2)

    return grad


def grad_sensor_pos(x_sensor, x_source, do_2d_aoa=False):
    """
    Compute the gradient of TDOA measurements, with sensor uncertainties, with respect to sensor position,
    equation 6.21.

    Ported from MATLAB code.

    Nicholas O'Donoughue
    14 April 2025

    :param x_sensor:    TDOA sensor positions
    :param x_source:    Source positions
    :param do_2d_aoa: Optional boolean parameter specifying whether 1D (az-only) or 2D (az/el) AOA is being performed
    :return jacobian:   Jacobian matrix representing the desired gradient
    """
    # TODO: Debug

    # Parse inputs
    n_dim, n_sensor = utils.safe_2d_shape(x_sensor)
    _, n_source = utils.safe_2d_shape(x_source)

    # Compute the Jacobian for Azimuth measurements
    # Equation 6.22 and 6.23 show that the gradient with respect to sensor position is the negative of the gradient
    # with respect to target position for both azimuth and elevation angle measurements, but resampled to be on a block
    # diagonal.
    _grad_x = grad_x(x_sensor, x_source, do_2d_aoa)

    grad = np.zeros((n_dim*n_sensor, n_sensor, n_source))
    for i in np.arange(n_sensor):
        start = n_dim * i
        end = start + n_dim
        grad[start:end, i, :] = _grad_x[:, i, :]  # The first n_sensor columns are J_az

    if do_2d_aoa:
        grad_el = np.zeros_like(grad)
        for i in np.arange(n_sensor):
            start = n_dim * i
            end = start + n_dim
            grad_el[start:end, i, :] = _grad_x[:, n_sensor + i, :]  # The second n_sensor columns are J_el

        grad = np.concatenate((grad, grad_el), axis=1)

    return grad
