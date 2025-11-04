import numpy as np
from numpy import typing as npt
from scipy import stats
import warnings

from ewgeo.utils.covariance import CovarianceMatrix


def compute_cep50(covariance: CovarianceMatrix | list[CovarianceMatrix],
                  print_warnings: bool=True) -> np.float64 | npt.NDArray[np.float64]:
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

    For 3D problems, where covariance is 3x3xN, the smallest eigenvalue is
    ignored, and the calculations are applied to the two largest eigenvalues.

    Ported from MATLAB code.

    Nicholas O'Donoughue
    16 January 2021

    :param covariance: CovarianceMatrix object or list of CovarianceMatrix objects
    :param print_warnings: Optional bool (default=True). If true, warnings will be printed when NaN or INF values
                           are encountered.
    :return cep: Radius of the corresponding CEP_50 circle or list of radii
    """

    # Check if it's a list or tuple, and use the batch mode
    if isinstance(covariance, (list, tuple)):
        return compute_cep50_fast(covariance, print_warnings)
    else:
        # Eigenvector analysis to identify independent components of error
        # lam[0] will be the smallest, and lam[-1] will be the largest
        lam = np.sort(covariance.eigenvalues, axis=None)

        # print('\tEigenvalues: {}'.format(lam))
        lam_max = lam[-1]  # eigenvalues are returned in ascending order; max is the last entry
        lam_min = lam[-2]  # use the second-largest as lam_min (ignores smallest eigenvalue in 3D problems)

        # Check the eigenvalues; they should not be complex or negative
        assert not (np.iscomplex(lam_max) or np.iscomplex(lam_min)), 'Complex eigenvalue encountered; check for errors.'

        # Depending on the eigenvalue ratio, use the appropriate approximation
        if lam_min > .25*lam_max:
            cep = .59*(np.sqrt(lam_min)+np.sqrt(lam_max))
        else:
            # ToDo: suppress runtime warning (invalid value encountered in scalar divide....inf? nan?) while generating Fig 13.8a
            cep = np.sqrt(lam_max)*(.67+.8*lam_min/lam_max)

    return cep

def compute_cep50_fast(covariance: list[CovarianceMatrix], print_warnings: bool=True)-> npt.NDArray[np.float64]:
    """
    Fast version of compute_cep50 that operates quickly over a list of CovarianceMatrices.

    Uses a single call to numpy that leverages numpy.linalg.eigh's batch processing mode to offload most of the work
    to efficient LAPACK code.

    :param covariance: list of CovarianceMatrix objects
    :param print_warnings: Optional bool (default=True). If true, warnings will be printed when NaN or INF values
                           are encountered.
    :return cep: numpy array of CEP50 calculations for each input covariance matrix
    """

    # Error check
    if len(set([c.size for c in covariance])) != 1:
        # Not all covariance matrices have the same size
        raise ValueError("Unable to compute CEP50 in parallel for covariance matrices of different sizes.")

    if covariance[0].size > 3:
        raise ValueError("Unable to compute CEP50 in parallel for covariance matrices of dimension > 3.")
    if covariance[0].size < 2:
        raise ValueError("Unable to compute CEP50 in parallel for covariance matrices of dimension < 2.")

    # Batch mode; stack all the raw covariances
    covs = np.stack([c.cov for c in covariance], axis=0)  # shape: (N, M, M) where M is the number of spatial dimensions

    # Compute eigenvalues
    lam = np.linalg.eigh(covs)[0]  # shape (N, M)

    # Pull the two largest eigenvalues; this ignores the smallest eigenvalue in 3D scenarios
    lam_max = lam[:, -1]
    lam_min = lam[:, -2]

    # Compute the CEP50 using one of two methods
    cep = np.empty_like(lam_max) # shape (N, )
    mask = lam_min > .25 * lam_max or np.isnan(lam_min) or np.isnan(lam_max)  # use the circular approximation when the
                                                                              # eigenvalues are approximately equal, or
                                                                              # when one or both are NaN (avoid dividing
                                                                              # by nan)
    cep[mask] = .59 * np.sqrt(lam_min[mask]) + np.sqrt(lam_max[mask])
    cep[~mask] = np.sqrt(lam_max[~mask]) * (0.67 + 0.8 * lam_min[~mask] / lam_max[~mask])

    # Replace invalid with inf if desired
    bad_mask = ~np.isfinite(covs).all(axis=(1, 2))
    if np.any(bad_mask) and print_warnings:
        warnings.warn("Poorly formed (NaN or INF values encountered) in CEP50 calculation.")
    cep[bad_mask] = np.inf

    return cep

def compute_rmse_scaling(conf_interval: float)-> float:
    """
    Computes the RMSE scaling factor for the specified confidence
    interval (between 0 and 1).  Defined as the integral limits (-gamma to
    gamma) that contain the desired percentage of random samples from a
    standard normal distribution (mean = 0, standard deviation = 1).

    It is computed simply with:
       gamma = norm_inv(.5 + conf_interval/2);

    and returns a value gamma such that
       norm_cdf(gamma) - norm_cdf(-gamma) = conf_interval

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


def compute_rmse_confidence_interval(gamma: float)-> float:
    """
    Determines the confidence interval for a given scale factor gamma, which
    is defined as the percentage of a standard normal distribution that falls
    within the bounds -gamma to gamma.

    Computed simply with:
       confInterval = norm_cdf(gamma) - norm_cdf(-gamma)

    Ported from MATLAB Code.

    Nicholas O'Donoughue
    16 January 2021

    :param gamma: Scale factor
    :return: Confidence interval on scale (0-1)
    """

    return stats.norm.cdf(gamma) - stats.norm.cdf(-gamma)


def compute_rmse(covariance: CovarianceMatrix | list[CovarianceMatrix])-> np.float64 | npt.NDArray[np.float64]:
    """
    Compute the RMSE for a covariance matrix or a list of covariance matrices; defined as the square root of the
    trace of each covariance matrix.
    """
    if isinstance(covariance, list):
        return np.array([compute_rmse(c) for c in covariance])

    return np.sqrt(np.trace(covariance.cov))

def draw_cep50(x: npt.ArrayLike,
               covariance: CovarianceMatrix | list[CovarianceMatrix],
               num_pts: int=100)-> tuple[npt.NDArray, npt.NDArray] | list[tuple[npt.NDArray, npt.NDArray]]:
    """
    Return the (x,y) coordinates of a circle with radius given by the CEP50
    of the covariance matrix C, centered on the point x, with numPts points
    around the circle

    Ported from MATLAB Code

    Nicholas O'Donoughue
    16 January 2021

    :param x: Center position for CEP circle (2-dimensional)
    :param covariance: CovarianceMatrix object or list of CovarianceMatrix objects
    :param num_pts: Number of points to use to draw circle [Default=100]
    :return x: array of x-coordinates for CEP drawing
    :return y: array of y-coordinates for CEP drawing
    """

    if isinstance(covariance, list):
        return [draw_cep50(x, cov, num_pts) for cov in covariance]
    else:
        # Find the CEP; the radius of the circle to draw
        cep = compute_cep50(covariance)

        # Initialize the angular indices of the circle
        th = np.linspace(start=0, stop=2*np.pi, num=num_pts)

        # Convert from r,th to cartesian coordinates
        xx = cep*np.cos(th)
        yy = cep*np.sin(th)

        # Offset the cartesian coordinates
        return xx + x[0], yy + x[1]


def draw_error_ellipse(x: npt.ArrayLike,
                       covariance: CovarianceMatrix,
                       num_pts: int=100,
                       conf_interval: float=50)-> tuple[npt.NDArray, npt.NDArray]:
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
                         99 -- 99 percent
                         xx -- any floating point value between 0 and 1
    :return x: x-coordinate defining error ellipse
    :return y: y-coordinate defining error ellipse
    """

    # Eigenvector analysis to identify major/minor axes rotation and length
    lam = covariance.eigenvalues
    v = covariance.eigenvectors

    # Sort the eigenvalues
    idx_sort = np.argsort(lam)  # Sorted in ascending order by default

    # Major Axis
    v_max = v[:, idx_sort[-1]]
    lam_max = lam[idx_sort[-1]]

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
        gamma = 5.991
    else:
        assert 0 < conf_interval < 1, (
            'Attempted to parse confidence interval as number between 0 and 1, but found {}'.format(conf_interval))
        gamma = -2 * np.log(1-conf_interval)

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
