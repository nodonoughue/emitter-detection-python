import utils
from utils import solvers
from . import model
import numpy as np


def max_likelihood(x_tdoa, rho_dot, cov, x_ctr, search_size, epsilon=None, ref_idx=None):
    """
    Construct the ML Estimate by systematically evaluating the log
    likelihood function at a series of coordinates, and returning the index
    of the maximum.  Optionally returns the full set of evaluated
    coordinates, as well.

    :param x_tdoa: Sensor positions [m]
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
        return model.log_likelihood(x_tdoa, rho_dot, cov, x, ref_idx)

    # Call the util function
    x_est, likelihood, x_grid = solvers.ml_solver(ell, x_ctr, search_size, epsilon)

    return x_est, likelihood, x_grid


def gradient_descent(x_tdoa, rho_dot, cov, x_init, alpha=None, beta=None, epsilon=None, max_num_iterations=None,
                     force_full_calc=False, plot_progress=False, ref_idx=None):
    """
    Computes the gradient descent solution for tdoa processing.

    Ported from MATLAB code.

    Nicholas O'Donoughue
    21 February 2021

    :param x_tdoa: tdoa sensor positions [m]
    :param rho_dot: Measurement vector
    :param cov: tdoa error covariance matrix
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
        return rho_dot - model.measurement(x_tdoa, this_x, ref_idx=ref_idx)

    def jacobian(this_x):
        return model.jacobian(x_tdoa, this_x, ref_idx=ref_idx)

    # Call generic Gradient Descent solver
    x, x_full = solvers.gd_solver(y, jacobian, cov, x_init, alpha, beta, epsilon, max_num_iterations, force_full_calc,
                                  plot_progress)

    return x, x_full


def least_square(x0, rho_dot, cov, x_init, epsilon=None, max_num_iterations=None, force_full_calc=False,
                 plot_progress=False, ref_idx=None):
    """
    Computes the least square solution for tdoa processing.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    21 February 2021

    :param x0: Sensor positions [m]
    :param rho_dot: Range Difference Measurements [m/s]
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
        return rho_dot - model.measurement(x0, this_x, ref_idx=ref_idx)

    def jacobian(this_x):
        return model.jacobian(x0, this_x, ref_idx=ref_idx)

    # Call the generic Least Square solver
    x, x_full = solvers.ls_solver(y, jacobian, cov, x_init, epsilon, max_num_iterations, force_full_calc, plot_progress)

    return x, x_full


def bestfix(x_tdoa, rho_dot, cov, x_ctr, search_size, epsilon, ref_idx=None, pdftype=None):
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

    :param x_tdoa: Sensor positions [m]
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
        return model.measurement(x_tdoa, x, ref_idx)

    pdfs = utils.make_pdfs(msmt, rho_dot, pdftype, cov)

    # Call the util function
    x_est, likelihood, x_grid = solvers.bestfix(pdfs, x_ctr, search_size, epsilon)

    return x_est, likelihood, x_grid


def chan_ho(x0, rho, cov):
    """
    Computes the Chan-Ho solution for TDOA processing.

    Ref:  Y.-T. Chan and K. Ho, “A simple and efficient estimator for 
          hyperbolic location,” IEEE Transactions on signal processing, 
          vol. 42, no. 8, pp. 1905–1915, 1994.

    Ported from MATLAB Code
    
    Nicholas O'Donoughue
    11 March 2021
    
    :param x0: sensor positions [m]
    :param rho: range-difference measurements [m]
    :param cov: error covariance matrix for range-difference [m]
    :return: estimated source position [m]
    """
    
    # Stage 1: Initial Position Estimate
    # Compute system matrix overline(A) according to 13.23

    # Compute shifted measurement vector overline(y) according to 13.24
    r = np.sqrt(np.sum(np.fabs(x0)**2, axis=0))
    n_dims, n_sensor = np.shape(x0)

    y1 = (rho**2-r[0:-2]**2+r[-1]**2)
    g1 = -2*np.horzcat(x0[:, 0:-2] - x0[:, -1], rho)

    # Compute initial position estimate overline(theta) according to 13.25
    b = np.eye(n_sensor-1)
    w1 = b.dot(cov).dot(b.H)
    w1_inv = np.linalg.pinv(w1)
    g1w = g1.H.dot(w1_inv)
    g1wg = g1w.dot(g1)
    g1wy = g1w.dot(y1)
    th1 = np.linalg.lstsq(g1wg, g1wy)

    # Refine sensor estimate
    for _ in np.arange(3):
        ri_hat = np.sqrt(np.sum((x0-th1[0:-2])**2, axis=0))

        # Re-compute initial position estimate overline(theta) according to 13.25
        b = 2*np.diag(ri_hat[0:-2])
        w1 = b.dot(cov).dot(b.H)
        w1_inv = np.linalg.pinv(w1)
        g1w = g1.H.dot(w1_inv)
        g1wg = g1w.dot(g1)
        g1wy = g1w.dot(y1)
        th1 = np.linalg.lstsq(g1wg, g1wy)[0]

    th1p = np.subtract(th1, np.vertcat(x0[:, -1], 0))

    # Stage 2: Account for Parameter Dependence
    y2 = th1**2
    g2 = np.vertcat(np.eye(n_dims), np.ones(shape=(1, n_dims)))

    b2 = 2*np.diag(th1p)

    # Compute final parameter estimate overline(theta)' according to 13.32
    g1wg1 = g1.H.dot(w1).dot(g1)
    w2 = np.linalg.pinv(b2.H).dot(g1wg1).dot(np.linalg.pinv(b2))
    g2w = g2.H.dot(w2)
    g2wg2 = g2w.dot(g2)
    g2wy = g2w.dot(y2)
    th2 = np.linalg.lstsq(g2wg2, g2wy)[0]

    # Compute position estimate overline(x)' according to 13.33
    # x_prime1 = x0(:,end)+sqrt(th_prime);
    # x_prime2 = x0(:,end)-sqrt(th_prime);
    #
    # offset1 = norm(x_prime1-th(1:end-1));
    # offset2 = norm(x_prime2-th(1:end-1));
    #
    # if offset1 <= offset2
    #     x = x_prime1;
    # else
    #     x = x_prime2;
    # end
    x = np.sign(np.diag(th1[0:-2]))*np.sqrt(th2)+x0[:, -1]

    return x
