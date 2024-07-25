import utils
from utils import solvers
from . import model
import numpy as np
import math
from itertools import combinations


def max_likelihood(x_sensor, psi, cov, x_ctr, search_size, epsilon=None):
    """
    Construct the ML Estimate by systematically evaluating the log
    likelihood function at a series of coordinates, and returning the index
    of the maximum.  Optionally returns the full set of evaluated
    coordinates, as well.

    Ported from MATLAB Code
    
    Nicholas O'Donoughue
    22 February 2021
    
    :param x_sensor: Sensor positions [m]
    :param psi: AOA measurement vector [rad]
    :param cov: Measurement error covariance matrix
    :param x_ctr: Center of search grid [m]
    :param search_size: 2-D vector of search grid sizes [m]
    :param epsilon: Desired resolution of search grid [m]
    :return x_est: Estimated source position [m]
    :return likelihood: Likelihood computed across the entire set of candidate source positions
    :return x_grid: Candidate source positions
    """

    # Set up function handle
    def ell(x):
        return model.log_likelihood(x_sensor, psi, cov, x)

    # Call the util function
    x_est, likelihood, x_grid = solvers.ml_solver(ell, x_ctr, search_size, epsilon)

    return x_est, likelihood, x_grid


def gradient_descent(x_sensor, psi, cov, x_init, alpha=0.3, beta=0.8, epsilon=10, max_num_iterations=1e3,
                     force_full_calc=False, plot_progress=False):
    """
    Computes the gradient descent solution for FDOA processing.

    Ported from MATLAB code.

    Nicholas O'Donoughue
    22 February 2021

    :param x_sensor: Sensor positions [m]
    :param psi: AOA Measurement vector [rad]
    :param cov: Measurement error covariance matrix
    :param x_init: Initial estimate of source position [m]
    :param alpha: Backtracking line search parameter
    :param beta: Backtracking line search parameter
    :param epsilon: Desired position error tolerance (stopping condition)
    :param max_num_iterations: Maximum number of iterations to perform
    :param force_full_calc: Boolean flag to force all iterations (up to max_num_iterations) to be computed, regardless
                            of convergence (DEFAULT = False)
    :param plot_progress: Boolean flag that dictates whether to plot intermediate solutions as they are derived
                          (DEFAULT = False).
    :return x: Estimated source position
    :return x_full: Iteration-by-iteration estimated source positions
    """

    # Initialize measurement error and jacobian functions
    def y(this_x):
        return psi - model.measurement(x_sensor, this_x)

    def jacobian(this_x):
        return model.jacobian(x_sensor, this_x)

    # Call generic Gradient Descent solver
    x, x_full = solvers.gd_solver(y, jacobian, cov, x_init, alpha, beta, epsilon, max_num_iterations, force_full_calc,
                                  plot_progress)

    return x, x_full


def least_square(x_sensor, psi, cov, x_init, epsilon=10, max_num_iterations=1e3, force_full_calc=False,
                 plot_progress=False):
    """
    Computes the least square solution for FDOA processing.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    22 February 2021

    :param x_sensor: Sensor positions [m]
    :param psi: AOA Measurements [rad]
    :param cov: Measurement Error Covariance Matrix [(m/s)^2]
    :param x_init: Initial estimate of source position [m]
    :param epsilon: Desired estimate resolution [m]
    :param max_num_iterations: Maximum number of iterations to perform
    :param force_full_calc: Boolean flag to force all iterations (up to max_num_iterations) to be computed, regardless
                            of convergence (DEFAULT = False)
    :param plot_progress: Boolean flag that dictates whether to plot intermediate solutions as they are derived
                          (DEFAULT = False).
    :return x: Estimated source position
    :return x_full: Iteration-by-iteration estimated source positions
    """

    # Initialize measurement error and Jacobian function handles
    def y(this_x):
        return psi - model.measurement(x_sensor, this_x)

    def jacobian(this_x):
        return model.jacobian(x_sensor, this_x)

    # Call the generic Least Square solver
    x, x_full = solvers.ls_solver(y, jacobian, cov, x_init, epsilon, max_num_iterations, force_full_calc, plot_progress)

    return x, x_full


def bestfix(x_sensor, psi, cov, x_ctr, search_size, epsilon, pdf_type=None):
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
    :param psi: Measurement vector [rad]
    :param cov: Measurement error covariance matrix
    :param x_ctr: Center of search grid [m]
    :param search_size: 2-D vector of search grid sizes [m]
    :param epsilon: Desired resolution of search grid [m]
    :param pdf_type: String indicating the type of distribution to use. See +utils/makePDFs.m for options.
    :return x_est: Estimated source position [m]
    :return likelihood: Likelihood computed across the entire set of candidate source positions
    :return x_grid: Candidate source positions
    """

    # Generate the PDF
    def measurement(x):
        return model.measurement(x_sensor, x)

    pdfs = utils.make_pdfs(measurement, psi, pdf_type, cov)

    # Call the util function
    x_est, likelihood, x_grid = solvers.bestfix(pdfs, x_ctr, search_size, epsilon)

    return x_est, likelihood, x_grid


def angle_bisector(x_sensor, psi):
    """
    Compute the center via intersection of angle bisectors for  
    3 or more LOBs given by sensor positions xi and angle 
    of arrival measurements psi.

    If more than 3 measurements are provided, this method repeats on each set
    of 3 measurements, and then takes the average of the solutions.
    
    Ported from MATLAB code.
    
    Nicholas O'Donoughue
    22 February 2021
    
    :param x_sensor: N x 2 matrix of sensor positions
    :param psi: N x 1 vector of AOA measurements (radians)
    :return x_est: 2 x 1 vector of estimated source position
    """

    # Parse Inputs
    num_dim, num_sensors = utils.safe_2d_shape(x_sensor)

    if num_dim != 2:
        raise TypeError("Angle Bisector Solution only works in x/y space.")

    # Set up all possible permutations of 3 elements
    if num_sensors <= 15:
        # Get all possible sets of 3 sensors
        sensor_sets = combinations(np.arange(num_sensors), 3)
        num_sets = math.factorial(num_sensors) / (math.factorial(3)*math.factorial(num_sensors-3))
    else:
        # If there are more than ~15 rows, nchoosek returns an extremely large
        # set.
        #
        # Instead, just make sequential groups of three
        sensor_sets = [(i, i+1, i+2) for i in np.arange(num_sensors-2)]
        num_sets = len(sensor_sets)

    # Loop over sets -- adding each estimate to x_est
    x_est = np.zeros((2, ))

    for sensor_set in sensor_sets:
        this_x = x_sensor[:, sensor_set]  # 2 x 3 array
        this_psi = psi[sensor_set, ]  # 3 x 1 array

        # Find vertices
        v0 = utils.geo.find_intersect(this_x[:, 0], this_psi[0], this_x[:, 1], this_psi[1])
        v1 = utils.geo.find_intersect(this_x[:, 1], this_psi[1], this_x[:, 2], this_psi[2])
        v2 = utils.geo.find_intersect(this_x[:, 2], this_psi[2], this_x[:, 0], this_psi[0])

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
        this_cntr = utils.geo.find_intersect(v0, th_bi0, v1, th_bi1)

        # Accumulate the estimates, we'll divide by num_sets later to make it an average.
        x_est += this_cntr

    return x_est/num_sets


def centroid(x_sensor, psi):
    """
    Compute the centroid of the intersection of 3 or more LOBs given by
    sensor positions x_source and angle of arrival measurements psi.

    If more than 3 measurements are provided, this method repeats on each set
    of 3 measurements, and then takes the average of the solutions.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    22 February 2021

    :param x_sensor: 2 x N matrix of sensor positions
    :param psi: N x 1 vector of AOA measurements (radians)
    :return x_est: 2 x 1 vector of estimated source position
    """

    # Parse Inputs
    num_dim, num_sensors = utils.safe_2d_shape(x_sensor)

    if num_dim != 2:
        raise TypeError("Angle Bisector Solution only works in x/y space.")

    # Set up all possible permutations of 3 elements
    if num_sensors <= 15:
        # Get all possible sets of 3 sensors
        sensor_sets = combinations(np.arange(num_sensors), 3)
        num_sets = math.factorial(num_sensors) / (math.factorial(3)*math.factorial(num_sensors-3))
    else:
        # If there are more than ~15 rows, nchoosek returns an extremely large
        # set.
        #
        # Instead, just make sequential groups of three
        sensor_sets = [(i, i + 1, i + 2) for i in np.arange(num_sensors - 2)]
        num_sets = len(sensor_sets)

    # Loop over sets -- adding each estimate to x_est
    x_est = np.zeros(shape=(2, ))

    for sensor_set in sensor_sets:
        this_x = x_sensor[:, sensor_set]  # 2 x 3 array
        this_psi = psi[sensor_set, ]  # 3 x 1 array

        # Find vertices
        v0 = utils.geo.find_intersect(this_x[:, 0], this_psi[0], this_x[:, 1], this_psi[1])
        v1 = utils.geo.find_intersect(this_x[:, 1], this_psi[1], this_x[:, 2], this_psi[2])
        v2 = utils.geo.find_intersect(this_x[:, 2], this_psi[2], this_x[:, 0], this_psi[0])

        # Find Centroid by averaging the three vertices
        this_cntr = (v0 + v1 + v2) / 3

        # Accumulate the estimates, we'll divide by num_sets later to make it an average.
        x_est += this_cntr

    return x_est/num_sets
