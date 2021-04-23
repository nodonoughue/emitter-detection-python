import utils
from utils import solvers
from . import model


def max_likelihood(x_fdoa, v_fdoa, rho_dot, cov, x_ctr, search_size, epsilon=None, ref_idx=None):
    """
    Construct the ML Estimate by systematically evaluating the log
    likelihood function at a series of coordinates, and returning the index
    of the maximum.  Optionally returns the full set of evaluated
    coordinates, as well.

    :param x_fdoa: Sensor positions [m]
    :param v_fdoa: Sensor velocities [m/s]
    :param rho_dot: Measurement vector [Hz]
    :param cov: Measurement error covariance matrix
    :param x_ctr: Center of search grid [m]
    :param search_size: 2-D vector of search grid sizes [m]
    :param epsilon: Desired resolution of search grid [m]
    :param ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings
    :return x_est: Estimated source position [m]
    :return likelihood: Likelihood computed across the entire set of candidate source positions
    :return x_grid: Candidate source positions
    """

    # Set up function handle
    def ell(x):
        return model.log_likelihood(x_fdoa, v_fdoa, rho_dot, cov, x, ref_idx)

    # Call the util function
    x_est, likelihood, x_grid = solvers.ml_soln(ell, x_ctr, search_size, epsilon)

    return x_est, likelihood, x_grid


def gradient_descent(x_fdoa, v_fdoa, rho_dot, cov, x_init, alpha=None, beta=None, epsilon=None, max_num_iterations=None,
                     force_full_calc=False, plot_progress=False, ref_idx=None):
    """
    Computes the gradient descent solution for FDOA processing.

    Ported from MATLAB code.

    Nicholas O'Donoughue
    21 February 2021

    :param x_fdoa: FDOA sensor positions [m]
    :param v_fdoa: FDOA sensor velocities [m/s]
    :param rho_dot: Measurement vector
    :param cov: FDOA error covariance matrix
    :param x_init: Initial estimate of source position [m]
    :param alpha: Backtracking line search parameter
    :param beta: Backtracking line search parameter
    :param epsilon: Desired position error tolerance (stopping condition)
    :param max_num_iterations: Maximum number of iterations to perform
    :param force_full_calc: Boolean flag to force all iterations (up to max_num_iterations) to be computed, regardless
                            of convergence (DEFAULT = False)
    :param plot_progress: Boolean flag dictacting whether to plot intermediate solutions as they are derived
                          (DEFAULT = False).
    :param ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings
    :return x: Estimated source position
    :return x_full: Iteration-by-iteration estimated source positions
    """

    # Initialize measurement error and jacobian functions
    def y(this_x):
        return rho_dot - model.measurement(x_fdoa, v_fdoa, this_x, ref_idx=ref_idx)

    def jacobian(this_x):
        return model.jacobian(x_fdoa, v_fdoa, this_x, ref_idx=ref_idx)

    # Call generic Gradient Descent solver
    x, x_full = solvers.gd_soln(y, jacobian, cov, x_init, alpha, beta, epsilon, max_num_iterations, force_full_calc,
                                plot_progress)

    return x, x_full


def least_square(x0, v0, rho_dot, cov, x_init, epsilon=None, max_num_iterations=None, force_full_calc=False,
                 plot_progress=False, ref_idx=None):
    """
    Computes the least square solution for FDOA processing.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    21 February 2021

    :param x0: Sensor positions [m]
    :param v0: Sensor velocities [m/s]
    :param rho_dot: Range Rate-Difference Measurements [m/s]
    :param cov: Measurement Error Covariance Matrix [(m/s)^2]
    :param x_init: Initial estimate of source position [m]
    :param epsilon: Desired estimate resolution [m]
    :param max_num_iterations: Maximum number of iterations to perform
    :param force_full_calc: Boolean flag to force all iterations (up to max_num_iterations) to be computed, regardless
                            of convergence (DEFAULT = False)
    :param plot_progress: Boolean flag dictacting whether to plot intermediate solutions as they are derived
                          (DEFAULT = False).
    :param ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings
    :return x: Estimated source position
    :return x_full: Iteration-by-iteration estimated source positions
    """

    # Initialize measurement error and Jacobian function handles
    def y(this_x):
        return rho_dot - model.measurement(x0, v0, this_x, ref_idx=ref_idx)

    def jacobian(this_x):
        return model.jacobian(x0, v0, this_x, ref_idx=ref_idx)

    # Call the generic Least Square solver
    x, x_full = solvers.ls_soln(y, jacobian, cov, x_init, epsilon, max_num_iterations, force_full_calc, plot_progress)

    return x, x_full


def bestfix(x_fdoa, v_fdoa, rho_dot, cov, x_ctr, search_size, epsilon, ref_idx=None, pdftype=None):
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

    :param x_fdoa: Sensor positions [m]
    :param v_fdoa: Sensor velocities [m/s]
    :param rho_dot: Measurement vector [Hz]
    :param cov: Measurement error covariance matrix
    :param x_ctr: Center of search grid [m]
    :param search_size: 2-D vector of search grid sizes [m]
    :param epsilon: Desired resolution of search grid [m]
    :param ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings
    :param pdftype: String indicating the type of distribution to use. See +utils/makePDFs.m for options.
    :return x_est: Estimated source position [m]
    :return likelihood: Likelihood computed across the entire set of candidate source positions
    :return x_grid: Candidate source positions
    """

    # Generate the PDF
    def msmt(x):
        return model.measurement(x_fdoa, v_fdoa, x, ref_idx)

    pdfs = utils.make_pdfs(msmt, rho_dot, pdftype, cov)

    # Call the util function
    x_est, likelihood, x_grid = solvers.bestfix(pdfs, x_ctr, search_size, epsilon)

    return x_est, likelihood, x_grid
