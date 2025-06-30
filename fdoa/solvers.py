import utils
from utils import solvers
from . import model
import numpy as np
from utils.covariance import CovarianceMatrix


def max_likelihood(x_sensor, v_sensor, zeta, cov: CovarianceMatrix, x_ctr, search_size, epsilon=None, ref_idx=None,
                   do_resample=False, bias=None, **kwargs):
    """
    Construct the ML Estimate by systematically evaluating the log
    likelihood function at a series of coordinates, and returning the index
    of the maximum.  Optionally returns the full set of evaluated
    coordinates, as well.

    :param x_sensor: Sensor positions [m]
    :param v_sensor: Sensor velocities [m/s]
    :param zeta: Measurement vector [Hz]
    :param cov: Measurement error covariance matrix
    :param x_ctr: Center of search grid [m]
    :param search_size: 2-D vector of search grid sizes [m]
    :param epsilon: Desired resolution of search grid [m]
    :param ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings
    :param do_resample: Boolean flag; if true the covariance matrix will be resampled, using ref_idx
    :param bias: measurement bias (optional)
    :return x_est: Estimated source position [m]
    :return likelihood: Likelihood computed across the entire set of candidate source positions
    :return x_grid: Candidate source positions
    """

    # Resample the covariance matrix
    if do_resample:
        cov = cov.resample(ref_idx=ref_idx)

    # Set up function handle
    def ell(x):
        return model.log_likelihood(x_sensor=x_sensor, v_sensor=v_sensor, rho_dot=zeta, cov=cov,
                                    x_source=x, v_source=None, ref_idx=ref_idx, do_resample=False, bias=bias)

    # Call the util function
    x_est, likelihood, x_grid = solvers.ml_solver(ell=ell, x_ctr=x_ctr, search_size=search_size, epsilon=epsilon,
                                                  **kwargs)

    return x_est, likelihood, x_grid


def max_likelihood_uncertainty(x_sensor, zeta, cov: CovarianceMatrix, cov_pos: CovarianceMatrix,
                               x_ctr, search_size, epsilon=None, v_sensor=None, ref_idx=None,
                               do_resample=False, do_sensor_bias=False, **kwargs):
    """
    Construct the ML Estimate by systematically evaluating the log
    likelihood function at a series of coordinates, and returning the index
    of the maximum.  Optionally returns the full set of evaluated
    coordinates, as well.

    :param x_sensor: Sensor positions [m]
    :param zeta: Measurement vector [Hz]
    :param cov: Measurement error covariance matrix
    :param cov_pos: Sensor position error covariance matrix
    :param x_ctr: Center of search grid [m]
    :param search_size: 2-D vector of search grid sizes [m]
    :param epsilon: Desired resolution of search grid [m]
    :param v_sensor: Sensor velocities [m/s]
    :param ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings
    :param do_resample: Boolean flag; if true the covariance matrix will be resampled, using ref_idx
    :param do_sensor_bias: Boolean flag; if true, then sensor bias terms will be included in search
    :return x_est: Estimated source position [m]
    :return bias_est: Estimated sensor bias [m/s]
    :return sensor_pos_est: Estimated sensor positions [m]
    :return sensor_vel_est: Estimated sensor velocities [m/s]
    :return likelihood: Likelihood computed across the entire set of candidate source positions
    :return x_grid: Candidate source positions
    """
    num_dim, num_sensors = utils.safe_2d_shape(x_sensor)

    # Resample the covariance matrix
    if do_resample:
        cov = cov.resample(ref_idx=ref_idx)

    # Make sure the search space is properly defined, and parse the parameter indices
    search_params = {'th_center': x_ctr,
                     'search_size': search_size,
                     'search_resolution': epsilon,
                     'do_fdoa_bias': do_sensor_bias,
                     'x_fdoa': x_sensor,
                     'v_fdoa': v_sensor}
    search_center, search_size, search_resolution, param_indices = utils.make_uncertainty_search_space(**search_params)

    # Set up function handle
    # We must take care to ensure that it can handle an n_th x N matrix of
    # inputs; for compatibility with how utils.solvers.ml_solver will call it.
    def ell(theta):
        return model.log_likelihood_uncertainty(x_sensor=x_sensor, rho_dot=zeta, cov=cov,
                                                cov_pos=cov_pos, theta=theta, ref_idx=ref_idx,
                                                do_resample=False,
                                                do_sensor_bias=do_sensor_bias)

    # Call the util function
    th_est, likelihood, x_grid = solvers.ml_solver(ell=ell, x_ctr=search_center, search_size=search_size,
                                                   epsilon=search_resolution, **kwargs)

    x_est = th_est[param_indices['source_pos']]
    bias_est = th_est[param_indices['bias']] if do_sensor_bias else None
    sensor_pos_est = np.reshape(th_est[param_indices['fdoa_pos']], (num_dim, num_sensors)) if cov_pos is not None else None
    sensor_vel_est = np.reshape(th_est[param_indices['fdoa_vel']], (num_dim, num_sensors)) if cov_pos is not None else None

    return x_est, bias_est, sensor_pos_est, sensor_vel_est, likelihood, x_grid


def gradient_descent(x_sensor, v_sensor, zeta, cov: CovarianceMatrix, x_init, v_source=None, ref_idx=None,
                     do_resample=False, bias=None, **kwargs):
    """
    Computes the gradient descent solution for FDOA processing.

    Ported from MATLAB code.

    Nicholas O'Donoughue
    21 February 2021

    :param x_sensor: FDOA sensor positions [m]
    :param v_sensor: FDOA sensor velocities [m/s]
    :param zeta: Measurement vector
    :param cov: FDOA error covariance matrix
    :param x_init: Initial estimate of source position [m]
    :param v_source: Source velocity (assumed to be true) [m/s]
    :param ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings
    :param do_resample: Boolean flag; if true the covariance matrix will be resampled, using ref_idx
    :return x: Estimated source position
    :return x_full: Iteration-by-iteration estimated source positions
    """

    # Initialize measurement error and jacobian functions
    def y(this_x):
        return zeta - model.measurement(x_sensor=x_sensor, v_sensor=v_sensor,
                                        x_source=this_x, v_source=v_source, ref_idx=ref_idx, bias=bias)

    def jacobian(this_x):
        return model.jacobian(x_sensor=x_sensor, v_sensor=v_sensor,
                              x_source=this_x, v_source=v_source,
                              ref_idx=ref_idx)

    # Resample the covariance matrix
    if do_resample:
        cov = cov.resample(ref_idx=ref_idx)

    # Call generic Gradient Descent solver
    x, x_full = solvers.gd_solver(y, jacobian, cov, x_init, **kwargs)

    return x, x_full


def least_square(x_sensor, v_sensor, zeta, cov: CovarianceMatrix, x_init, ref_idx=None, do_resample=False,
                 bias=None, **kwargs):
    """
    Computes the least square solution for FDOA processing.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    21 February 2021

    :param x_sensor: Sensor positions [m]
    :param v_sensor: Sensor velocities [m/s]
    :param zeta: Range Rate-Difference Measurements [m/s]
    :param cov: Measurement Error Covariance Matrix [(m/s)^2]
    :param x_init: Initial estimate of source position [m]
    :param ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings
    :param do_resample: Boolean flag; if true the covariance matrix will be resampled, using ref_idx
    :return x: Estimated source position
    :return x_full: Iteration-by-iteration estimated source positions
    """

    # Initialize measurement error and Jacobian function handles
    def y(this_x):
        return zeta - model.measurement(x_sensor=x_sensor, v_sensor=v_sensor,
                                        x_source=this_x, v_source=None,
                                        ref_idx=ref_idx, bias=bias)

    def jacobian(this_x):
        return model.jacobian(x_sensor=x_sensor, v_sensor=v_sensor,
                              x_source=this_x, v_source=None,
                              ref_idx=ref_idx)

    # Resample the covariance matrix
    if do_resample:
        cov = cov.resample(ref_idx=ref_idx)

    # Call the generic Least Square solver
    x, x_full = solvers.ls_solver(y, jacobian, cov, x_init, **kwargs)

    return x, x_full


def bestfix(x_sensor, v_sensor, zeta, cov: CovarianceMatrix, x_ctr, search_size, epsilon, ref_idx=None, pdf_type=None,
            do_resample=False):
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
    :param zeta: Measurement vector [Hz]
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
        cov = cov.resample(ref_idx=ref_idx)

    # Make sure that rho is a vector -- the pdf functions choke if the mean value
    # is a Nx1 matrix
    zeta = np.squeeze(zeta)

    # Generate the PDF
    def msmt(x):
        # We have to squeeze rho, so let's also squeeze msmt
        return np.squeeze(model.measurement(x_sensor=x_sensor, v_sensor=v_sensor,
                                            x_source=x, v_source=None, ref_idx=ref_idx))

    pdfs = utils.make_pdfs(msmt, zeta, pdf_type, cov.cov)

    # Call the util function
    x_est, likelihood, x_grid = solvers.bestfix(pdfs, x_ctr, search_size, epsilon)

    return x_est, likelihood, x_grid

