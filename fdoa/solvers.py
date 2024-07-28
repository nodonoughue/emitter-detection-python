import utils
from utils import solvers
from . import model
import numpy as np


def max_likelihood(x_sensor, v_sensor, rho, cov, x_ctr, search_size, epsilon=None, ref_idx=None, do_resample=False):
    """
    Construct the ML Estimate by systematically evaluating the log
    likelihood function at a series of coordinates, and returning the index
    of the maximum.  Optionally returns the full set of evaluated
    coordinates, as well.

    :param x_sensor: Sensor positions [m]
    :param v_sensor: Sensor velocities [m/s]
    :param rho: Measurement vector [Hz]
    :param cov: Measurement error covariance matrix
    :param x_ctr: Center of search grid [m]
    :param search_size: 2-D vector of search grid sizes [m]
    :param epsilon: Desired resolution of search grid [m]
    :param ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings
    :param do_resample: Boolean flag; if true the covariance matrix will be resampled, using ref_idx
    :return x_est: Estimated source position [m]
    :return likelihood: Likelihood computed across the entire set of candidate source positions
    :return x_grid: Candidate source positions
    """

    # Resample the covariance matrix
    if do_resample:
        # Do it here, instead of passing on to model.log_likelihood, in order to avoid repeatedly resampling
        # the same covariance matrix for every test point.
        cov = utils.resample_covariance_matrix(cov, ref_idx)

    # Set up function handle
    def ell(x):
        return model.log_likelihood(x_sensor=x_sensor, v_sensor=v_sensor, rho_dot=rho, cov=cov,
                                    x_source=x, v_source=None, ref_idx=ref_idx, do_resample=False)

    # Call the util function
    x_est, likelihood, x_grid = solvers.ml_solver(ell, x_ctr, search_size, epsilon)

    return x_est, likelihood, x_grid


def gradient_descent(x_sensor, v_sensor, rho, cov, x_init, v_source=None, alpha=None, beta=None, epsilon=None,
                     max_num_iterations=None, force_full_calc=False, plot_progress=False, ref_idx=None,
                     do_resample=False):
    """
    Computes the gradient descent solution for FDOA processing.

    Ported from MATLAB code.

    Nicholas O'Donoughue
    21 February 2021

    :param x_sensor: FDOA sensor positions [m]
    :param v_sensor: FDOA sensor velocities [m/s]
    :param rho: Measurement vector
    :param cov: FDOA error covariance matrix
    :param x_init: Initial estimate of source position [m]
    :param v_source: Source velocity (assumed to be true) [m/s]
    :param alpha: Backtracking line search parameter
    :param beta: Backtracking line search parameter
    :param epsilon: Desired position error tolerance (stopping condition)
    :param max_num_iterations: Maximum number of iterations to perform
    :param force_full_calc: Boolean flag to force all iterations (up to max_num_iterations) to be computed, regardless
                            of convergence (DEFAULT = False)
    :param plot_progress: Boolean flag dictacting whether to plot intermediate solutions as they are derived
                          (DEFAULT = False).
    :param ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings
    :param do_resample: Boolean flag; if true the covariance matrix will be resampled, using ref_idx
    :return x: Estimated source position
    :return x_full: Iteration-by-iteration estimated source positions
    """

    # Initialize measurement error and jacobian functions
    def y(this_x):
        return rho - model.measurement(x_sensor=x_sensor, v_sensor=v_sensor,
                                       x_source=this_x, v_source=v_source, ref_idx=ref_idx)

    def jacobian(this_x):
        return model.jacobian(x_sensor=x_sensor, v_sensor=v_sensor,
                              x_source=this_x, v_source=v_source,
                              ref_idx=ref_idx)

    # Resample the covariance matrix
    if do_resample:
        cov = utils.resample_covariance_matrix(cov, ref_idx)

    # Call generic Gradient Descent solver
    x, x_full = solvers.gd_solver(y, jacobian, cov, x_init, alpha, beta, epsilon, max_num_iterations, force_full_calc,
                                  plot_progress)

    return x, x_full


def least_square(x_sensor, v_sensor, rho, cov, x_init, epsilon=None, max_num_iterations=None, force_full_calc=False,
                 plot_progress=False, ref_idx=None, do_resample=False):
    """
    Computes the least square solution for FDOA processing.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    21 February 2021

    :param x_sensor: Sensor positions [m]
    :param v_sensor: Sensor velocities [m/s]
    :param rho: Range Rate-Difference Measurements [m/s]
    :param cov: Measurement Error Covariance Matrix [(m/s)^2]
    :param x_init: Initial estimate of source position [m]
    :param epsilon: Desired estimate resolution [m]
    :param max_num_iterations: Maximum number of iterations to perform
    :param force_full_calc: Boolean flag to force all iterations (up to max_num_iterations) to be computed, regardless
                            of convergence (DEFAULT = False)
    :param plot_progress: Boolean flag dictacting whether to plot intermediate solutions as they are derived
                          (DEFAULT = False).
    :param ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings
    :param do_resample: Boolean flag; if true the covariance matrix will be resampled, using ref_idx
    :return x: Estimated source position
    :return x_full: Iteration-by-iteration estimated source positions
    """

    # Initialize measurement error and Jacobian function handles
    def y(this_x):
        return rho - model.measurement(x_sensor=x_sensor, v_sensor=v_sensor,
                                       x_source=this_x, v_source=None,
                                       ref_idx=ref_idx)

    def jacobian(this_x):
        return model.jacobian(x_sensor=x_sensor, v_sensor=v_sensor,
                              x_source=this_x, v_source=None,
                              ref_idx=ref_idx)

    # Resample the covariance matrix
    if do_resample:
        cov = utils.resample_covariance_matrix(cov, ref_idx)

    # Call the generic Least Square solver
    x, x_full = solvers.ls_solver(y, jacobian, cov, x_init, epsilon, max_num_iterations, force_full_calc, plot_progress)

    return x, x_full


def bestfix(x_sensor, v_sensor, rho, cov, x_ctr, search_size, epsilon, ref_idx=None, pdf_type=None, do_resample=False):
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
    :param v_sensor: Sensor velocities [m/s]
    :param rho: Measurement vector [Hz]
    :param cov: Measurement error covariance matrix
    :param x_ctr: Center of search grid [m]
    :param search_size: 2-D vector of search grid sizes [m]
    :param epsilon: Desired resolution of search grid [m]
    :param ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings
    :param pdf_type: String indicating the type of distribution to use. See +utils/makePDFs.m for options.
    :param do_resample: Boolean flag; if true the covariance matrix will be resampled, using ref_idx
    :return x_est: Estimated source position [m]
    :return likelihood: Likelihood computed across the entire set of candidate source positions
    :return x_grid: Candidate source positions
    """

    # Resample the covariance matrix
    if do_resample:
        cov = utils.resample_covariance_matrix(cov, ref_idx)

    # Make sure that rho is a vector -- the pdf functions choke if the mean value
    # is an Nx1 matrix
    rho = np.squeeze(rho)

    # Generate the PDF
    def msmt(x):
        # We have to squeeze rho, so let's also squeeze msmt
        return np.squeeze(model.measurement(x_sensor=x_sensor, v_sensor=v_sensor,
                                            x_source=x, v_source=None, ref_idx=ref_idx))

    pdfs = utils.make_pdfs(msmt, rho, pdf_type, cov)

    # Call the util function
    x_est, likelihood, x_grid = solvers.bestfix(pdfs, x_ctr, search_size, epsilon)

    return x_est, likelihood, x_grid
