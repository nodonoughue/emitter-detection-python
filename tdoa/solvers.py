import utils
from utils import solvers
from . import model
import numpy as np


def max_likelihood(x_sensor, rho, cov, x_ctr, search_size, epsilon=None, ref_idx=None, do_resample=False):
    """
    Construct the ML Estimate by systematically evaluating the log
    likelihood function at a series of coordinates, and returning the index
    of the maximum.  Optionally returns the full set of evaluated
    coordinates, as well.

    :param x_sensor: Sensor positions [m]
    :param rho: Measurement vector [m]
    :param cov: Measurement error covariance matrix
    :param x_ctr: Center of search grid [m]
    :param search_size: 2-D vector of search grid sizes [m]
    :param epsilon: Desired resolution of search grid [m]
    :param ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings
    :param do_resample: If true, cov is a sensor-level covariance matrix and must be resampled
    :return x_est: Estimated source position [m]
    :return likelihood: Likelihood computed across the entire set of candidate source positions
    :return x_grid: Candidate source positions
    """

    # Set up function handle
    def ell(x):
        return model.log_likelihood(x_sensor, rho, cov, x, ref_idx, do_resample)

    # Call the util function
    x_est, likelihood, x_grid = solvers.ml_solver(ell, x_ctr, search_size, epsilon)

    return x_est, likelihood, x_grid


def gradient_descent(x_sensor, rho, cov, x_init, alpha=None, beta=None, epsilon=None, max_num_iterations=None,
                     force_full_calc=False, plot_progress=False, ref_idx=None, do_resample=False):
    """
    Computes the gradient descent solution for tdoa processing.

    Ported from MATLAB code.

    Nicholas O'Donoughue
    21 February 2021

    :param x_sensor: tdoa sensor positions [m]
    :param rho: Measurement vector
    :param cov: tdoa error covariance matrix
    :param x_init: Initial estimate of source position [m]
    :param alpha: Backtracking line search parameter
    :param beta: Backtracking line search parameter
    :param epsilon: Desired position error tolerance (stopping condition)
    :param max_num_iterations: Maximum number of iterations to perform
    :param force_full_calc: Boolean flag to force all iterations (up to max_num_iterations) to be computed, regardless
                            of convergence (DEFAULT = False)
    :param plot_progress: Boolean flag dictating whether to plot intermediate solutions as they are derived
                          (DEFAULT = False).
    :param ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings
    :param do_resample: If true, cov is a sensor-level covariance matrix and must be resampled
    :return x: Estimated source position
    :return x_full: Iteration-by-iteration estimated source positions
    """

    # Initialize measurement error and jacobian functions
    def y(this_x):
        return rho - model.measurement(x_sensor, this_x, ref_idx=ref_idx)

    def jacobian(this_x):
        return model.jacobian(x_sensor, this_x, ref_idx=ref_idx)

    if do_resample:
        _, num_sensors = utils.safe_2d_shape(x_sensor)
        test_idx_vec, ref_idx_vec = utils.parse_reference_sensor(ref_idx, num_sensors)
        cov = utils.resample_covariance_matrix(cov, test_idx_vec, ref_idx_vec)

    # Call generic Gradient Descent solver
    x, x_full = solvers.gd_solver(y, jacobian, cov, x_init, alpha, beta, epsilon, max_num_iterations, force_full_calc,
                                  plot_progress)

    return x, x_full


def least_square(x_sensor, rho, cov, x_init, epsilon=None, max_num_iterations=None, force_full_calc=False,
                 plot_progress=False, ref_idx=None, do_resample=False):
    """
    Computes the least square solution for tdoa processing.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    21 February 2021

    :param x_sensor: Sensor positions [m]
    :param rho: Range Difference Measurements [m/s]
    :param cov: Measurement Error Covariance Matrix [(m/s)^2]
    :param x_init: Initial estimate of source position [m]
    :param epsilon: Desired estimate resolution [m]
    :param max_num_iterations: Maximum number of iterations to perform
    :param force_full_calc: Boolean flag to force all iterations (up to max_num_iterations) to be computed, regardless
                            of convergence (DEFAULT = False)
    :param plot_progress: Boolean flag dictating whether to plot intermediate solutions as they are derived
                          (DEFAULT = False).
    :param ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings
    :param do_resample: If true, cov is a sensor-level covariance matrix and must be resampled
    :return x: Estimated source position
    :return x_full: Iteration-by-iteration estimated source positions
    """

    # Initialize measurement error and Jacobian function handles
    def y(this_x):
        return rho - model.measurement(x_sensor, this_x, ref_idx=ref_idx)

    def jacobian(this_x):
        return model.jacobian(x_sensor, this_x, ref_idx=ref_idx)

    if do_resample:
        _, num_sensors = utils.safe_2d_shape(x_sensor)
        test_idx_vec, ref_idx_vec = utils.parse_reference_sensor(ref_idx, num_sensors)
        cov = utils.resample_covariance_matrix(cov, test_idx_vec, ref_idx_vec)

    # Call the generic Least Square solver
    x, x_full = solvers.ls_solver(y, jacobian, cov, x_init, epsilon, max_num_iterations, force_full_calc, plot_progress)

    return x, x_full


def bestfix(x_sensor, rho, cov, x_ctr, search_size, epsilon, ref_idx=None, pdf_type=None, do_resample=False):
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
    :param rho: Measurement vector [Hz]
    :param cov: Measurement error covariance matrix
    :param x_ctr: Center of search grid [m]
    :param search_size: 2-D vector of search grid sizes [m]
    :param epsilon: Desired resolution of search grid [m]
    :param ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings
    :param pdf_type: String indicating the type of distribution to use. See +utils/makePDFs.m for options.
    :param do_resample: If true, cov is a sensor-level covariance matrix and must be resampled
    :return x_est: Estimated source position [m]
    :return likelihood: Likelihood computed across the entire set of candidate source positions
    :return x_grid: Candidate source positions
    """

    # Generate the PDF
    def measurement(x):
        return model.measurement(x_sensor, x, ref_idx)

    # Resample the covariance matrix
    if do_resample:
        _, num_sensors = utils.safe_2d_shape(x_sensor)
        test_idx_vec, ref_idx_vec = utils.parse_reference_sensor(ref_idx, num_sensors)

        cov = utils.resample_covariance_matrix(cov, test_idx_vec, ref_idx_vec)

    pdfs = utils.make_pdfs(measurement, rho, pdf_type, cov)

    # Call the util function
    x_est, likelihood, x_grid = solvers.bestfix(pdfs, x_ctr, search_size, epsilon)

    return x_est, likelihood, x_grid


def chan_ho(x_sensor, rho, cov, ref_idx=None, do_resample=False):
    """
    Computes the Chan-Ho solution for TDOA processing.

    Ref:  Y.-T. Chan and K. Ho, “A simple and efficient estimator for 
          hyperbolic location,” IEEE Transactions on signal processing, 
          vol. 42, no. 8, pp. 1905–1915, 1994.

    Ported from MATLAB Code
    
    Nicholas O'Donoughue
    11 March 2021
    
    :param x_sensor: sensor positions [m]
    :param rho: range-difference measurements [m]
    :param cov: error covariance matrix for range-difference [m]
    :param ref_idx: index of the reference sensor for TDOA processing. Default is the last sensor. Must be scalar.
    :param do_resample: If true, cov is a sensor-level covariance matrix and must be resampled
    :return: estimated source position [m]
    """

    # TODO: Debug

    # Accept an arbitrary reference position
    n_dims, n_sensor = np.shape(x_sensor)
    if ref_idx is not None and ref_idx != n_sensor-1:
        # Throw an error if there are multiple reference sensors
        # TODO: Test assertion
        assert np.size(ref_idx) == 1, 'The Chan-Ho solver currently requires a single reference sensor.'

        # Re-arrange the sensors
        sort_idx = [i for i in np.arange(n_sensor) if i != ref_idx]
        sort_idx = sort_idx.append(ref_idx)
        x_sensor = x_sensor[sort_idx, :]

        # Note: We don't need to rearrange rho, since it is the range
        # difference measurements. We don't need to rearrange cov, since
        # the following code block will handle its resampling to account
        # for the test and reference indices.

    # Resample the covariance matrix
    if do_resample:
        test_idx_vec, ref_idx_vec = utils.parse_reference_sensor(ref_idx, n_sensor)
        cov = utils.resample_covariance_matrix(cov, test_idx_vec, ref_idx_vec)

    # Stage 1: Initial Position Estimate
    # Compute system matrix overline(A) according to 11.23

    # Compute shifted measurement vector overline(y) according to 11.24
    # NOTE: In python, indexing with [-1] is the final entry, while [:-1] is all
    # entries *except* the last entry in a list or array.
    r = np.sqrt(np.sum(np.fabs(x_sensor) ** 2, axis=0))
    y1 = (rho**2-r[:-1]**2+r[-1]**2)
    last_sensor = x_sensor[:, -1]
    dx = x_sensor[:, :-1] - last_sensor[:, np.newaxis]
    g1 = -2*np.concatenate((np.transpose(dx), rho[:, np.newaxis]), axis=1)

    # Compute initial position estimate overline(theta) according to 11.25
    b = np.eye(n_sensor-1)
    th1, cov_mod = _chan_ho_theta(cov, b, g1, y1)

    # w1 = b.dot(cov).dot(np.transpose(np.conjugate(b)))
    # w1_inv = np.linalg.pinv(w1)
    # g1w = np.transpose(np.conjugate(g1)).dot(w1_inv)
    # g1wg = g1w.dot(g1)
    # g1wy = g1w.dot(y1)
    # th1 = np.linalg.lstsq(g1wg, g1wy, rcond=None)[0]

    # Refine sensor estimate
    for _ in np.arange(3):
        ri_hat = np.sqrt(np.sum((x_sensor - th1[:-1, np.newaxis]) ** 2, axis=0))

        # Re-compute initial position estimate overline(theta) according to 11.25
        b = 2*np.diag(ri_hat[0:-1])
        th1, cov_mod = _chan_ho_theta(cov, b, g1, y1)
        # w1 = b.dot(cov).dot(np.transpose(np.conjugate(b)))
        # w1_inv = np.linalg.pinv(w1)
        # g1w = np.transpose(np.conjugate(g1)).dot(w1_inv)
        # g1wg = g1w.dot(g1)
        # g1wy = g1w.dot(y1)
        # th1 = np.linalg.lstsq(g1wg, g1wy, rcond=None)[0]

    th1p = np.subtract(th1, np.concatenate((x_sensor[:, -1], [0]), axis=0))

    # Stage 2: Account for Parameter Dependence
    y2 = th1**2
    g2 = np.concatenate((np.eye(n_dims), np.ones(shape=(1, n_dims))), axis=0)
    b2 = 2*np.diag(th1p)

    # Compute final parameter estimate overline(theta)' according to 13.32
    th2 = _chan_ho_theta_hat(cov_mod, b2, g1, g2, y2)

    # g1wg1 = np.transpose(np.conjugate(g1)).dot(cov_mod).dot(g1)
    # w2 = np.linalg.pinv(np.conjugate(np.transpose(b2))).dot(g1wg1).dot(np.linalg.pinv(b2))
    # g2w = np.transpose(np.conjugate(g2)).dot(w2)
    # g2wg2 = g2w.dot(g2)
    # g2wy = g2w.dot(y2)
    # th2 = np.linalg.lstsq(g2wg2, g2wy, rcond=None)[0]

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

    x = np.sign(np.diag(th1[:-1])).dot(np.sqrt(th2)) + x_sensor[:, -1]

    return x


def _chan_ho_theta(cov, b, g, y):
    """
    Compute position estimate overline(theta) according to 11.25.
    This is an internal function called by chan_ho.

    :param cov: measurement-level covariance matrix
    :param b:
    :param g: system matrix
    :param y: shifted measurement vector
    :return theta: parameter vector (eq 11.26)
    :return cov_mod: modified covariance matrix (eq 11.28)
    """

    cov_mod = b.dot(cov).dot(np.transpose(np.conjugate(b)))  # BCB', eq 11.28
    cov_mod_inv = np.linalg.inv(cov_mod)

    # Assemble matrix products g'*w_inv*g, and g'*w_inv*y
    gw = np.transpose(np.conjugate(g)).dot(cov_mod_inv)
    gwg = gw.dot(g)
    gwy = gw.dot(y)

    theta = np.linalg.lstsq(gwg, gwy, rcond=None)[0]

    return theta, cov_mod


def _chan_ho_theta_hat(cov_mod, b2, g1, g2, y2):
    """
    Compute position estimate overline(theta) according to 11.25.
    This is an internal function called by chan_ho.

    :param cov_mod: measurement-level covariance matrix
    :param b2:
    :param g1: system matrix
    :param g2: system matrix
    :param y2: shifted measurement vector
    :return theta: parameter vector (eq 11.26)
    :return cov_mod: modified covariance matrix (eq 11.28)
    """

    g1wg1 = np.transpose(np.conjugate(g1)).dot(cov_mod).dot(g1)
    w2 = np.linalg.pinv(np.conjugate(np.transpose(b2))).dot(g1wg1).dot(np.linalg.pinv(b2))
    g2w = np.transpose(np.conjugate(g2)).dot(w2)
    g2wg2 = g2w.dot(g2)
    g2wy = g2w.dot(y2)

    theta = np.linalg.lstsq(g2wg2, g2wy, rcond=None)[0]

    return theta
