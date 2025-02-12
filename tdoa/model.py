import numpy as np
# from utils import unit_conversions, geo
import utils
from scipy.linalg import solve_triangular
unit_conversions = utils.unit_conversions
geo = utils.geo


def measurement(x_sensor, x_source, ref_idx=None):
    """
    Computes TDOA measurements and converts to range difference of arrival (by compensating for the speed of light).

    Ported from MATLAB Code

    Nicholas O'Donoughue
    11 March 2021

    :param x_sensor: nDim x nTDOA array of TDOA sensor positions
    :param x_source: nDim x n_source array of source positions
    :param ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for TDOA
    :return rdoa: nAoa + nTDOA + nFDOA - 2 x n_source array of range difference of arrival measurements
    """

    # Construct component measurements
    n_dim1, n_sensor = utils.safe_2d_shape(x_sensor)
    n_dim2, n_source = utils.safe_2d_shape(x_source)

    if n_dim1 != n_dim2:
        raise TypeError('First dimension of all inputs must match')

    # Parse sensor pairs
    test_idx_vec, ref_idx_vec = utils.parse_reference_sensor(ref_idx, n_sensor)

    # Compute range from each source to each sensor
    # dx = np.reshape(x_source, (n_dim1, 1, n_source)) - x_sensor
    # r = np.reshape(np.sqrt(np.sum(np.fabs(dx)**2, axis=0)), (n_sensor, n_source))  # n_sensor x n_source
    r = utils.geo.calc_range(x_sensor, x_source)

    # Compute range difference for each pair of sensors
    rdoa = r[test_idx_vec] - r[ref_idx_vec]
    return rdoa


def jacobian(x_sensor, x_source, ref_idx=None):
    """
    # Returns the Jacobian matrix for a set of TDOA measurements.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    11 March 2021

    :param x_sensor: nDim x nTDOA array of TDOA sensor positions
    :param x_source: nDim x n_source array of source positions
    :param ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for TDOA
    :return: n_dim x nMeasurement x n_source matrix of Jacobians, one for each candidate source position
    """

    # Parse inputs
    n_dim1, n_sensor = utils.safe_2d_shape(x_sensor)
    n_dim2, n_source = utils.safe_2d_shape(x_source)

    if n_dim1 != n_dim2:
        raise TypeError('Input variables must match along first dimension.')

    n_dim = n_dim1

    test_idx_vec, ref_idx_vec = utils.parse_reference_sensor(ref_idx, n_sensor)

    # Compute the Offset Vectors
    dx = np.reshape(x_source, (n_dim, 1, n_source)) - np.reshape(x_sensor, (n_dim, n_sensor, 1))
    r = np.reshape(np.sqrt(np.sum(np.abs(dx) ** 2, axis=0)), (1, n_sensor, n_source))  # n_sensor x n_source

    # Remove any zero-range samples; replace with epsilon (a very-small value), to avoid a divide
    # by zero error
    mask = np.abs(r < 1e-10)
    r[mask] = 1e-10

    # Compute the Jacobians
    j = (dx[:, test_idx_vec, :] / r[:, test_idx_vec, :]) - (dx[:, ref_idx_vec, :] / r[:, ref_idx_vec, :])

    # Reshape
    num_measurements = test_idx_vec.size
    if n_source > 1:
        out_dims = (n_dim, num_measurements, n_source)
    else:
        out_dims = (n_dim, num_measurements)

    j = np.reshape(j, newshape=out_dims)
    # j = np.delete(j, np.unique(ref_idx_vec), axis=1) # remove reference id b/c it is all zeros
    # not needed when parse_ref_sensors is fixed
    return j


def log_likelihood(x_sensor, rho, cov, x_source, ref_idx=None, do_resample=False, cov_is_inverted=False):
    """
    Computes the Log Likelihood for TDOA sensor measurement, given the received range difference measurement vector
    rho, covariance matrix cov, and set of candidate source positions x_source.

    Ported from MATLAB Code.

    Nicholas O'Donoughue
    11 March 2021

    :param x_sensor: nDim x nTDOA array of TDOA sensor positions
    :param rho: Combined TDOA measurement vector (range difference)
    :param cov: measurement error covariance matrix
    :param x_source: Candidate source positions
    :param ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for TDOA
    :param do_resample: Boolean flag; if true the covariance matrix will be resampled, using ref_idx
    :param cov_is_inverted: Boolean flag, if false then cov is the covariance matrix. If true, then it is the
                            inverse of the covariance matrix.
    :return ell: Log-likelihood evaluated at each position x_source.
    """

    n_dim, n_source_pos = utils.safe_2d_shape(x_source)
    ell = np.zeros((n_source_pos, 1))

    # Make sure the source pos is a matrix, rather than simply a vector
    if n_source_pos == 1:
        x_source = x_source[:, np.newaxis]

    # Parse the TDOA sensor pairs
    _, n_tdoa = utils.safe_2d_shape(x_sensor)

    # Pre-process the covariance matrix
    if cov_is_inverted:
        cov_inv = cov
        cov_lower = None  # pre-define to avoid a 'use before defined' error
    else:
        if do_resample:
            cov = utils.resample_covariance_matrix(cov, ref_idx)
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

    for idx_source in np.arange(n_source_pos):
        x_i = x_source[:, idx_source]

        # Generate the ideal measurement matrix for this position
        rho_dot = measurement(x_sensor, x_i, ref_idx)

        # Evaluate the measurement error
        err = (rho_dot - rho)

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


def error(x_sensor, cov, x_source, x_max, num_pts, ref_idx=None, do_resample=False, cov_is_inverted=False):
    """
    Construct a 2-D field from -x_max to +x_max, using numPts in each
    dimension.  For each point, compute the TDOA solution for each sensor
    against the reference (the first sensor), and compare to the TDOA
    solution from the true emitter position.

    Ported from MATLAB Code.

    Nicholas O'Donoughue
    21 January 2021

    :param x_sensor: nDim x N matrix of sensor positions
    :param cov: N x N covariance matrix
    :param x_source: nDim x 1 matrix of true emitter position
    :param x_max: nDim x 1 (or scalar) vector of maximum offset from origin for plotting
    :param num_pts: Number of test points along each dimension
    :param ref_idx: Scalar index of reference sensor, or n_dim x n_pair matrix of sensor pairings
    :param do_resample: Boolean flag; if true the covariance matrix will be resampled, using ref_idx
    :param cov_is_inverted: Boolean flag, if false then cov is the covariance matrix. If true, then it is the
                            inverse of the covariance matrix.
    :return epsilon: 2-D plot of FDOA error
    :return x_vec:
    :return y_vec:
    """

    # Compute the True TDOA measurement
    # Compute true range rate difference measurements default condition is to
    # use the final sensor as the reference for all difference measurements.
    r = measurement(x_sensor, x_source, ref_idx)

    _, num_sensors = np.shape(x_sensor)

    # Pre-process the covariance matrix
    if cov_is_inverted:
        cov_inv = cov
        cov_lower = None  # pre-define to avoid a 'use before defined' error
    else:
        if do_resample:
            cov = utils.resample_covariance_matrix(cov, ref_idx)
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
    xx_vec = x_max.flatten() * np.reshape(np.linspace(start=-1, stop=1, num=num_pts), (1, num_pts))
    x_vec = xx_vec[0, :]
    y_vec = xx_vec[1, :]
    xx, yy = np.meshgrid(x_vec, y_vec)
    x_plot = np.vstack((xx.flatten(), yy.flatten())).T  # 2 x numPts^2

    epsilon = np.zeros_like(xx)
    for idx_pt in np.arange(np.size(xx)):
        x_i = x_plot[:, idx_pt]

        # Evaluate the measurement at x_i
        rr_i = measurement(x_sensor, x_i, ref_idx)

        # Compute the measurement error
        err = r - rr_i

        # Evaluate the scaled log likelihood
        if cov_is_inverted:
            epsilon[idx_pt] = err.H.dot(cov_inv).dot(err)
        else:
            cov_err = solve_triangular(cov_lower, err, lower=True)
            epsilon[idx_pt] = np.sum(cov_err**2)

    return epsilon


def toa_error_peak_detection(snr):
    """
    Computes the error in time of arrival estimation for a peak detection 
    algorithm, based on input SNR.
    
    Ported from MATLAB Code
    
    Nicholas O'Donoughue
    11 March 2021
    
    :param snr: Signal-to-Noise Ratio [dB]
    :return: expected time of arrival error variance [s^2]
    """

    # Convert SNR to linear units
    snr_lin = utils.unit_conversions.db_to_lin(snr)

    # Compute Error
    return 1/(2*snr_lin)


def toa_error_cross_corr(snr, bandwidth, pulse_len, bandwidth_rms=None):
    """
    Computes the timing error for a Cross-Correlation time of arrival
    estimator, given the input signal's bandwidth, pulse length, and RMS
    bandwidth.

    Ported from MATLAB Code
    
    Nicholas O'Donoughue
    11 March 2021
    
    :param snr: Signal-to-Noise ratio [dB]
    :param bandwidth: Input signal bandwidth [Hz]
    :param pulse_len: Length of the input signal [s]
    :param bandwidth_rms: RMS bandwidth of input signal [Hz]
    :return: Expected time of arrival error variance [s^2]
    """    

    # Convert input SNR to linear units
    snr_lin = unit_conversions.db_to_lin(snr)

    # Compute the product of SNR, bandwidth, pulse length, and RMS bandwidth
    a = snr_lin * bandwidth * pulse_len * bandwidth_rms

    # Invert and apply 8*pi scale factor
    return 1/(8*np.pi*a)


def draw_isochrone(x1, x2, range_diff, num_pts, max_ortho):
    """
    Finds the isochrone with the stated range difference from points x1
    and x2.  Generates an arc with 2*numPts-1 points, that spans up to
    maxOrtho distance from the intersection line of x1 and x2

    Ported from MATLAB Code.

    Nicholas O'Donoughue
    11 March 2021

    :param x1: Position of first sensor (Ndim x 1) [m]
    :param x2: Position of second sensor (Ndim x 1) [m]
    :param range_diff: Desired range difference [m]
    :param num_pts: Number of points to compute
    :param max_ortho: Maximum offset from line of sight between x1 and x2 [m]
    :return x_iso: First dimension of isochrone [m]
    :return y_iso: Second dimension of isochrone [m]
    """

    # Generate pointing vectors u and v in rotated coordinate space
    #  u = unit vector from x1 to x2
    #  v = unit vector orthogonal to u
    rot_mat = np.array(((0, 1), (-1, 0)))
    r = geo.calc_range(x1, x2)
    u = (x2-x1)/r
    v = rot_mat.dot(u)
    x_proj = np.array([u, v])

    # Position of reference points in uv-space
    x1uv = np.zeros(shape=(2, 1))
    x2uv = np.array((r, 0))
    
    # Initialize isochrone positions in uv-space
    vv = np.linspace(0, max_ortho, num_pts)
    uu = np.zeros(np.shape(vv))
    uu[0] = (r - range_diff) / 2
    xuv = np.array([uu, vv])

    # Integrate over points, search for isochrone position
    max_iter = 10000
    max_err = r/1000  # isochrone position error should be .1% or less

    for i in np.arange(num_pts):
        if i == 0:
            # Ignore the first index; it's done
            continue

        # Initialize the search for the u position
        xuv[0, i] = xuv[0, i-1]  # Initialize the u-dimension with the estimate for the previous step
        offset = r
        num_iter = 1
        
        while abs(offset) > max_err and num_iter <= max_iter:
            num_iter += 1

            # Compute the current range difference
            this_rng_diff = geo.calc_range_diff(xuv[:, i], x2uv, x1uv)

            # Offset is the difference between the current and desired
            # range difference
            offset = range_diff - this_rng_diff

            # Apply the offset directly to the u-dimension and repeat
            xuv[0, i] = xuv[0, i] - offset/2

    # Isochrone is symmetric about u axis
    # Flip the u axis, flip and negate v
    xuv_mirror = [np.flipud(xuv[0, 1:]), -np.flipud(xuv[1, 1:])]
    xuv = np.concatenate((xuv_mirror, xuv), axis=1)

    # Convert to x/y space and re-center at origin
    iso = x_proj.dot(xuv) + x1[:, np.newaxis]
    x_iso = iso[0, :]
    y_iso = iso[1, :]

    return x_iso, y_iso
