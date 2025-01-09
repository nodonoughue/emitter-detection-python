import numpy as np
from scipy import stats
import warnings


def compute_cep50(covariance):
    """
    Computes the radius for a CEP_50 circle from a given error covariance
    matrix C.  The CEP_50 circle is a circle that contains half of the random
    samples defined by the error covariance matrix.

    Calculation is extremely complex, and requires numerical integration, so
    the equation used herein is an approximation, depending on the ratio of
    the dominant to secondary eigenvalues.  If the ratio is less than 2,
    meaning that both eigenvectors contribute roughly the same amount of
    error, then we apply the approximation:
       cep = .59*(sqrt(lamMin)+sqrt(lamMax));
    otherwise, the dominant eigenvector is responsible for the majority of
    the error, and we apply the approximation:
       cep = sqrt(lamMax)*(.67+.8*lamMin/lamMax);

    Ported from MATLAB code.

    Nicholas O'Donoughue
    16 January 2021

    :param covariance: 2x2 error covariance matrix (additional dimensions are assumed to correspond to independent
                       cases, and are computed in turn)
    :return cep: Radius of the corresponding CEP_50 circle
    """

    # print('Computing CEP50...')

    # Parse input dimensions
    in_shape = np.shape(covariance)
    if np.size(in_shape) < 2:
        raise TypeError('Covariance matrix must have at least two dimensions.')
    elif in_shape[0] != 2 or in_shape[1] != 2:
        raise TypeError('First two dimensions of input covariance matrix must have size 2.')
    elif np.size(in_shape) > 2:
        # Multiple 2x2 matrices, keep track of the input size, and then reshape for easier looping
        out_shape = in_shape[2:]
        num_matrices = np.prod(out_shape)
        covariance = np.reshape(covariance, newshape=(in_shape[0], in_shape[1], num_matrices))
    else:
        # Only one input matrix
        out_shape = (1,)
        num_matrices = 1

        # Add a third dimension to covariance, so that the for loop has a (singleton) dimension to loop across
        covariance = np.expand_dims(covariance, axis=2)

    cep = np.zeros(shape=out_shape)
    for idx_matrix in np.arange(num_matrices):
        this_covariance = covariance[:, :, idx_matrix]

        # print('cov: {}'.format(this_covariance))

        # If any of the CEP elements are NaN or INF, then mark the CEP as INF
        if not np.all(np.isfinite(this_covariance)):
            cep[idx_matrix] = np.inf
            print('\tpoorly formed (NaN or INF values encountered), skipping...')
            continue

        # Eigenvector analysis to identify independent components of error
        lam, _ = np.linalg.eigh(this_covariance)
        # print('\tEigenvalues: {}'.format(lam))
        lam_min = np.min(lam)
        lam_max = np.max(lam)

        # Check the eigenvalues; they should not be complex or negative
        assert not (np.iscomplex(lam_max) or np.iscomplex(lam_min)), 'Complex eigenvalue encountered; check for errors.'

        # Ratio of dominant to secondary eigenvalues
        ratio = np.sqrt(lam_min/lam_max)

        # Depending on the eigenvalue ratio, use the appropriate approximation
        if ratio > .5:
            cep[idx_matrix] = .59*(np.sqrt(lam_min)+np.sqrt(lam_max))
        else:
            cep[idx_matrix] = np.sqrt(lam_max)*(.67+.8*lam_min/lam_max)

    return np.reshape(cep, newshape=out_shape)


def compute_rmse_scaling(conf_interval):
    """
    Computes the RMSE scaling factor for the specified confidence
    interval (between 0 and 1).  Defined as the integral limits (-gamma to
    gamma) that contain the desired percentage of random samples from a
    standard normal distribution (mean = 0, standard deviation = 1).

    It is computed simply with:
       gamma = norminv(.5 + confInterval/2);

    and returns a value gamma such that
       normcdf(gamma) - normcdf(-gamma) = confInterval

    Ported from MATLAB Code.

    Nicholas O'Donoughue
    16 January 2021

    :param conf_interval: Confidence interval, between 0 and 1
    :return gamma: Scale factor to apply for RMSE scaling
    """

    # Test for input validity
    if conf_interval <= 0 or conf_interval >= 1:
        raise TypeError('Input out of bounds.  Confidence Interval must be between 0 and 1')

    # Compute Gamma
    return stats.norm.ppf(.5 + conf_interval / 2)


def compute_rmse_confidence_interval(gamma):
    """
    Determines the confidence interval for a given scale factor gamma, which
    is defined as the percentage of a standard normal distribution that falls
    within the bounds -gamma to gamma.

    Computed simply with:
       confInterval = normcdf(gamma) - normcdf(-gamma)

    Ported from MATLAB Code.

    Nicholas O'Donoughue
    16 January 2021

    :param gamma: Scale factor
    :return: Confidence interval on scale (0-1)
    """

    return stats.norm.cdf(gamma) - stats.norm.cdf(-gamma)


def draw_cep50(x, covariance, num_pts=100):
    """
    Return the (x,y) coordinates of a circle with radius given by the CEP50
    of the covariance matrix C, centered on the point x, with numPts points
    around the circle

    Ported from MATLAB Code

    Nicholas O'Donoughue
    16 January 2021

    :param x: Center position for CEP circle (2-dimensional)
    :param covariance: Covariance matrix (2x2)
    :param num_pts: Number of points to use to draw circle [Default=100]
    :return x: array of x-coordinates for CEP drawing
    :return y: array of y-coordinates for CEP drawing
    """

    # Find the CEP; the radius of the circle to draw
    cep = compute_cep50(covariance)

    # Initialize the angular indices of the circle
    th = np.linspace(start=0, stop=2*np.pi, num=num_pts)

    # Convert from r,th to cartesian coordinates
    xx = cep*np.cos(th)
    yy = cep*np.sin(th)

    # Offset the cartesian coordinates
    return xx + x[0], yy + x[1]


def draw_error_ellipse(x, covariance, num_pts=100, conf_interval=50):
    """
    # Compute and return the error ellipse coordinates with numPts
    # samples around the ellipse, for an error centered at the location x and
    # with covariance matrix C.  The confidence interval specifies what
    # percentage of random errors will fall within the error ellipse.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    16 January 2021

    :param x: Center position of ellipse (2-dimensional coordinates)
    :param covariance: Error covariance matrix (2x2)
    :param num_pts: Number of points to use to draw the ellipse [default = 100]
    :param conf_interval: Desired confidence interval, must be one of:
                         1  -- 1 standard deviation (67#)
                         50 -- 50 percent
                         90 -- 90 percent
                         95 -- 95 percent
    :return x: x-coordinate defining error ellipse
    :return y: y-coordinate defining error ellipse
    """

    # Eigenvector analysis to identify major/minor axes rotation and length
    lam, v = np.linalg.eigh(covariance)

    # Sort the eigenvalues
    idx_sort = np.argsort(lam)  # Sorted in ascending order by default

    # Major Axis
    v_max = v[:, idx_sort[1]]
    lam_max = lam[idx_sort[1]]

    # Minor Axis
    lam_min = lam[idx_sort[0]]

    # Compute the rotation angle
    rot_angle = np.arctan2(v_max[1], v_max[0])
    rot_angle = np.mod(rot_angle+np.pi, 2*np.pi) - np.pi  # ensure angle is on interval [-pi,pi]

    # Lookup scale factor from confidence interval
    if conf_interval == 1:
        gamma = 1  # 1 sigma
    elif conf_interval == 50:
        gamma = 1.386
    elif conf_interval == 90:
        gamma = 4.601
    elif conf_interval == 95:
        gamma = 5.991  # 95# CI
    else:
        gamma = 1
        warnings.warn('Confidence Interval not recognized, using 1 standard deviation...')

    # Define Error Ellipse in rotated coordinate frame
    th = np.linspace(start=0, stop=2*np.pi, num=num_pts)  # Angle from center point, in rotated frame
    a = np.sqrt(gamma * lam_max)       # Major axis length
    b = np.sqrt(gamma * lam_min)       # Minor axis length

    ellipse_x = a * np.cos(th)         # Major axis coordinates
    ellipse_y = b * np.sin(th)         # Minor axis coordinates

    # Rotate the ellipse to standard reference frame
    rot_matrix = np.array([[np.cos(rot_angle), -np.sin(rot_angle)], [np.sin(rot_angle), np.cos(rot_angle)]])
    xx = np.matmul(rot_matrix, np.vstack((ellipse_x, ellipse_y)))  # Store as a 2 x N matrix of positions

    # Apply bias; so the ellipse is centered on the input x
    return np.expand_dims(x, axis=1) + xx
