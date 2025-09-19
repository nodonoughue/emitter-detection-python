from itertools import combinations
import math
import numpy as np
import numpy.typing as npt

from . import model
from ewgeo.utils import make_pdfs, safe_2d_shape, SearchSpace
from ewgeo.utils.covariance import CovarianceMatrix
from ewgeo.utils.geo import find_intersect
from ewgeo.utils.solvers import ml_solver, gd_solver, ls_solver, bestfix_solver


def max_likelihood(x_sensor, zeta, cov: CovarianceMatrix, search_space:SearchSpace, do_2d_aoa=False, bias=None,
                   **kwargs):
    """
    Construct the ML Estimate by systematically evaluating the log
    likelihood function at a series of coordinates, and returning the index
    of the maximum.  Optionally returns the full set of evaluated
    coordinates, as well.

    Ported from MATLAB Code
    
    Nicholas O'Donoughue
    22 February 2021
    
    :param x_sensor: Sensor positions [m]
    :param zeta: AOA measurement vector [rad]
    :param cov: Measurement error covariance matrix
    :param do_2d_aoa: Optional boolean parameter specifying whether 1D (az-only) or 2D (az/el) AOA is being performed
    :return x_est: Estimated source position [m]
    :return likelihood: Likelihood computed across the entire set of candidate source positions
    :return x_grid: Candidate source positions
    """

    # Set up function handle
    def ell(x, **ell_kwargs):
        return model.log_likelihood(x_sensor, zeta, cov, x, do_2d_aoa=do_2d_aoa, bias=bias, **ell_kwargs)

    # Call the util function
    x_est, likelihood, x_grid = ml_solver(ell, search_space=search_space, **kwargs)

    return x_est, likelihood, x_grid


def gradient_descent(x_sensor, zeta, cov: CovarianceMatrix, x_init, do_2d_aoa=False, bias=None, **kwargs):
    """
    Computes the gradient descent solution for FDOA processing.

    Ported from MATLAB code.

    Nicholas O'Donoughue
    22 February 2021

    :param x_sensor: Sensor positions [m]
    :param zeta: AOA Measurement vector [rad]
    :param cov: Measurement error covariance matrix
    :param x_init: Initial estimate of source position [m]
    :param do_2d_aoa: Optional boolean parameter specifying whether 1D (az-only) or 2D (az/el) AOA is being performed
    :return x: Estimated source position
    :return x_full: Iteration-by-iteration estimated source positions
    """

    # Initialize measurement error and jacobian functions
    def y(this_x):
        return zeta - model.measurement(x_sensor, this_x, do_2d_aoa=do_2d_aoa, bias=bias)

    def jacobian(this_x):
        return model.jacobian(x_sensor, this_x, do_2d_aoa=do_2d_aoa)

    # Call generic Gradient Descent solver
    x, x_full = gd_solver(y=y, jacobian=jacobian, cov=cov, x_init=x_init, **kwargs)

    return x, x_full


def least_square(x_sensor, zeta, cov: CovarianceMatrix, x_init, do_2d_aoa=False, bias=None, **kwargs):
    """
    Computes the least square solution for FDOA processing.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    22 February 2021

    :param x_sensor: Sensor positions [m]
    :param zeta: AOA Measurements [rad]
    :param cov: Measurement Error Covariance Matrix [(m/s)^2]
    :param x_init: Initial estimate of source position [m]
    :param do_2d_aoa: Optional boolean parameter specifying whether 1D (az-only) or 2D (az/el) AOA is being performed
    :return x: Estimated source position
    :return x_full: Iteration-by-iteration estimated source positions
    """

    # Initialize measurement error and Jacobian function handles
    def y(this_x):
        return zeta - model.measurement(x_sensor, this_x, do_2d_aoa=do_2d_aoa, bias=bias)

    def jacobian(this_x):
        return model.jacobian(x_sensor, this_x, do_2d_aoa=do_2d_aoa)

    # Call the generic Least Square solver
    x, x_full = ls_solver(zeta=y, jacobian=jacobian, cov=cov, x_init=x_init, **kwargs)

    return x, x_full


def bestfix(x_sensor, zeta, cov: CovarianceMatrix, search_space: SearchSpace, pdf_type=None, do_2d_aoa=False):
    """
    Construct the BestFix estimate by systematically evaluating the PDF at
    a series of coordinates, and returning the index of the maximum.
    Optionally returns the full set of evaluated coordinates, as well.

    Assumes a multi-variate Gaussian distribution with covariance matrix C,
    and unbiased estimates at each sensor.  Note that the BestFix algorithm
    implicitly assumes each measurement is independent, so any cross-terms in
    the covariance matrix C are ignored.

    Ref:
     Eric Hodson, "Method and arrangement for probabilistic determination of
     a target location," U.S. Patent US5045860A, 1990, https://patents.google.com/patent/US5045860A

    Ported from MATLAB Code

    Nicholas O'Donoughue
    21 February 2021

    :param x_sensor: Sensor positions [m]
    :param zeta: Measurement vector [rad]
    :param cov: Measurement error covariance matrix
    :param pdf_type: String indicating the type of distribution to use. See +utils/makePDFs.m for options.
    :param do_2d_aoa: Optional boolean parameter specifying whether 1D (az-only) or 2D (az/el) AOA is being performed
    :return x_est: Estimated source position [m]
    :return likelihood: Likelihood computed across the entire set of candidate source positions
    :return x_grid: Candidate source positions
    """

    # Generate the PDF
    def measurement(x):
        return model.measurement(x_sensor, x, do_2d_aoa=do_2d_aoa)

    pdfs = make_pdfs(measurement, zeta, pdf_type, cov.cov)

    # Call the util function
    x_est, likelihood, x_grid = bestfix_solver(pdfs, search_space)

    return x_est, likelihood, x_grid


def angle_bisector(x_sensor, zeta):
    """
    Compute the center via intersection of angle bisectors for  
    3 or more LOBs given by sensor positions xi and angle 
    of arrival measurements zeta.

    If more than 3 measurements are provided, this method repeats on each set
    of 3 measurements, and then takes the average of the solutions.
    
    Ported from MATLAB code.
    
    Nicholas O'Donoughue
    22 February 2021
    
    :param x_sensor: N x 2 matrix of sensor positions
    :param zeta: N x 1 vector of AOA measurements (radians)
    :return x_est: 2 x 1 vector of estimated source position
    """

    # Parse Inputs
    sensor_sets, num_sets = _parse_sensor_triplets(x_sensor)

    # Loop over sets -- adding each estimate to x_est
    x_est = np.zeros((2, ))

    for sensor_set in sensor_sets:
        # Find vertices
        v0, v1, v2 = _find_vertices(x_sensor[:, np.asarray(sensor_set)], zeta[np.asarray(sensor_set)])

        # Find angle bisectors
        th_fwd0 = np.arctan2(v1[1]-v0[1], v1[0]-v0[0])
        th_fwd1 = np.arctan2(v2[1]-v1[1], v2[0]-v1[0])

        th_back0 = np.arctan2(v2[1]-v0[1], v2[0]-v0[0])
        th_back1 = np.arctan2(v0[1]-v1[1], v0[0]-v1[0])

        th_bi0 = .5*(th_fwd0 + th_back0)
        th_bi1 = .5*(th_fwd1 + th_back1)

        # Add pi if the angle is flipped (pointing out)
        if th_bi0 < th_fwd0:
            th_bi0 += np.pi

        # Add pi if the angle is flipped (pointing out)
        if th_bi1 < th_fwd1:
            th_bi1 += np.pi

        # Find the intersection
        this_cntr = find_intersect(v0, th_bi0, v1, th_bi1)

        # Accumulate the estimates, we'll divide by num_sets later to make it an average.
        x_est += this_cntr

    return x_est/num_sets


def centroid(x_sensor, zeta):
    """
    Compute the centroid of the intersection of 3 or more LOBs given by
    sensor positions x_source and angle of arrival measurements zeta.

    If more than 3 measurements are provided, this method repeats on each set
    of 3 measurements, and then takes the average of the solutions.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    22 February 2021

    :param x_sensor: 2 x N matrix of sensor positions
    :param zeta: N x 1 vector of AOA measurements (radians)
    :return x_est: 2 x 1 vector of estimated source position
    """

    # Parse Inputs
    sensor_sets, num_sets = _parse_sensor_triplets(x_sensor)

    # Loop over sets -- adding each estimate to x_est
    x_est = np.zeros(shape=(2, ))

    for sensor_set in sensor_sets:
        # Find vertices
        v0, v1, v2 = _find_vertices(x_sensor[:, np.asarray(sensor_set)], zeta[np.asarray(sensor_set)])

        # Find Centroid by averaging the three vertices
        this_cntr = (v0 + v1 + v2) / 3

        # Accumulate the estimates, we'll divide by num_sets later to make it an average.
        x_est += this_cntr

    return x_est/num_sets


def _parse_sensor_triplets(x_sensor):

    # Parse Inputs
    num_dim, num_sensors = safe_2d_shape(x_sensor)

    if num_dim != 2:
        raise TypeError("Angle Bisector Solution only works in x/y space.")

    # Set up all possible permutations of 3 elements
    if num_sensors <= 15:
        # Get all possible sets of 3 sensors
        sensor_sets = combinations(np.arange(num_sensors), 3)
        num_sets = math.factorial(num_sensors) / (math.factorial(3) * math.factorial(num_sensors - 3))
    else:
        # If there are more than ~15 rows, combinations returns an extremely large set.
        # Instead, just make sequential groups of three
        sensor_sets = [(i, i + 1, i + 2) for i in np.arange(num_sensors - 2)]
        num_sets = len(sensor_sets)

    return sensor_sets, num_sets


def _find_vertices(x: npt.ArrayLike, zeta: npt.ArrayLike) -> (npt.ArrayLike, npt.ArrayLike, npt.ArrayLike):
    # Find vertices
    v0 = find_intersect(x[:, 0], zeta[0], x[:, 1], zeta[1])
    v1 = find_intersect(x[:, 1], zeta[1], x[:, 2], zeta[2])
    v2 = find_intersect(x[:, 2], zeta[2], x[:, 0], zeta[0])

    return v0, v1, v2
