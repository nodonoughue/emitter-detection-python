import utils
from utils import solvers
from . import model


def max_likelihood(x_aoa, x_tdoa, x_fdoa, v_fdoa, zeta, cov, x_ctr, search_size, epsilon=None, tdoa_ref_idx=None, fdoa_ref_idx=None):
    """
    Construct the ML Estimate by systematically evaluating the log
    likelihood function at a series of coordinates, and returning the index
    of the maximum.  Optionally returns the full set of evaluated
    coordinates, as well.

    :param x_aoa: AOA sensor positions [m]
    :param x_tdoa: TDOA sensor positions [m]
    :param x_fdoa: FDOA sensor positions [m]
    :param v_fdoa: FDOA sensor velocities [m/s]
    :param zeta: Combined measurement vector
    :param cov: Measurement error covariance matrix
    :param x_ctr: Center of search grid [m]
    :param search_size: 2-D vector of search grid sizes [m]
    :param epsilon: Desired resolution of search grid [m]
    :param tdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for TDOA
    :param fdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for FDOA
    :return x_est: Estimated source position [m]
    :return likelihood: Likelihood computed across the entire set of candidate source positions
    :return x_grid: Candidate source positions
    """

    # Set up function handle
    def ell(x):
        return model.log_likelihood(x_aoa, x_tdoa, x_fdoa, v_fdoa, zeta, cov, x, tdoa_ref_idx, fdoa_ref_idx)

    # Call the util function
    x_est, likelihood, x_grid = solvers.ml_solver(ell, x_ctr, search_size, epsilon)

    return x_est, likelihood, x_grid


def gradient_descent(x_aoa, x_tdoa, x_fdoa, v_fdoa, zeta, cov, x_init, alpha, beta, epsilon=None,
                     max_num_iterations=None, force_full_calc=False, plot_progress=False, tdoa_ref_idx=None,
                     fdoa_ref_idx=None):
    """
    Computes the gradient descent solution for FDOA processing.

    Ported from MATLAB code.

    Nicholas O'Donoughue
    21 February 2021

    :param x_aoa: AOA sensor positions [m]
    :param x_tdoa: TDOA sensor positions [m]
    :param x_fdoa: FDOA sensor positions [m]
    :param v_fdoa: FDOA sensor velocities [m/s]
    :param zeta: Combined measurement vector
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
    :param tdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings, for TDOA
    :param fdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings, for FDOA
    :return x: Estimated source position
    :return x_full: Iteration-by-iteration estimated source positions
    """

    # Initialize measurement error and jacobian functions
    def y(this_x):
        return zeta - model.measurement(x_aoa, x_tdoa, x_fdoa, v_fdoa, this_x, tdoa_ref_idx, fdoa_ref_idx)

    def jacobian(this_x):
        return model.jacobian(x_aoa, x_tdoa, x_fdoa, v_fdoa, this_x, tdoa_ref_idx, fdoa_ref_idx)

    # Call generic Gradient Descent solver
    x, x_full = solvers.gd_solver(y, jacobian, cov, x_init, alpha, beta, epsilon, max_num_iterations, force_full_calc,
                                  plot_progress)

    return x, x_full


def least_square(x_aoa, x_tdoa, x_fdoa, v_fdoa, zeta, cov, x_init, epsilon=None, max_num_iterations=None,
                 force_full_calc=False, plot_progress=False, tdoa_ref_idx=None, fdoa_ref_idx=None):
    """
    Computes the least square solution for FDOA processing.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    21 February 2021

    :param x_aoa: AOA sensor positions [m]
    :param x_tdoa: TDOA sensor positions [m]
    :param x_fdoa: FDOA sensor positions [m]
    :param v_fdoa: FDOA sensor velocities [m/s]
    :param zeta: Combined measurement vector
    :param cov: Measurement Error Covariance Matrix [(m/s)^2]
    :param x_init: Initial estimate of source position [m]
    :param epsilon: Desired estimate resolution [m]
    :param max_num_iterations: Maximum number of iterations to perform
    :param force_full_calc: Boolean flag to force all iterations (up to max_num_iterations) to be computed, regardless
                            of convergence (DEFAULT = False)
    :param plot_progress: Boolean flag dictating whether to plot intermediate solutions as they are derived
                          (DEFAULT = False).
    :param tdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for TDOA
    :param fdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for FDOA
    :return x: Estimated source position
    :return x_full: Iteration-by-iteration estimated source positions
    """

    # Initialize measurement error and Jacobian function handles
    def y(this_x):
        return zeta - model.measurement(x_aoa, x_tdoa, x_fdoa, v_fdoa, this_x, tdoa_ref_idx, fdoa_ref_idx)

    def jacobian(this_x):
        return model.jacobian(x_aoa, x_tdoa, x_fdoa, v_fdoa, this_x, tdoa_ref_idx, fdoa_ref_idx)

    # Call the generic Least Square solver
    x, x_full = solvers.ls_solver(y, jacobian, cov, x_init, epsilon, max_num_iterations, force_full_calc, plot_progress)

    return x, x_full


def bestfix(x_aoa, x_tdoa, x_fdoa, v_fdoa, zeta, cov, x_ctr, search_size, epsilon, tdoa_ref_idx=None, fdoa_ref_idx=None,
            pdftype=None):
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

    :param x_aoa: AOA sensor positions [m]
    :param x_tdoa: TDOA sensor positions [m]
    :param x_fdoa: FDOA sensor positions [m]
    :param v_fdoa: FDOA sensor velocities [m/s]
    :param zeta: Combined measurement vector
    :param cov: Measurement error covariance matrix
    :param x_ctr: Center of search grid [m]
    :param search_size: 2-D vector of search grid sizes [m]
    :param epsilon: Desired resolution of search grid [m]
    :param tdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for TDOA
    :param fdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for FDOA
    :param pdftype: String indicating the type of distribution to use. See +utils/makePDFs.m for options.
    :return x_est: Estimated source position [m]
    :return likelihood: Likelihood computed across the entire set of candidate source positions
    :return x_grid: Candidate source positions
    """

    # Generate the PDF
    def msmt(this_x):
        return model.measurement(x_aoa, x_tdoa, x_fdoa, v_fdoa, this_x, tdoa_ref_idx, fdoa_ref_idx)

    pdfs = utils.make_pdfs(msmt, zeta, pdftype, cov)

    # Call the util function
    x_est, likelihood, x_grid = solvers.bestfix(pdfs, x_ctr, search_size, epsilon)

    return x_est, likelihood, x_grid
