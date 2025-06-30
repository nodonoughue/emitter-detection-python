import utils
from utils import solvers
from utils.covariance import CovarianceMatrix
from . import model
import numpy as np


def max_likelihood(zeta, cov: CovarianceMatrix, x_aoa=None, x_tdoa=None, x_fdoa=None, v_fdoa=None, v_source=None,
                   x_ctr=0., search_size=1., epsilon=None, do_2d_aoa=False, tdoa_ref_idx=None, fdoa_ref_idx=None,
                   do_resample=False, angle_bias=None, range_bias=None, range_rate_bias=None, **kwargs):
    """
    Construct the ML Estimate by systematically evaluating the log
    likelihood function at a series of coordinates, and returning the index
    of the maximum.  Optionally returns the full set of evaluated
    coordinates, as well.

    :param x_aoa: AOA sensor positions [m]
    :param x_tdoa: TDOA sensor positions [m]
    :param x_fdoa: FDOA sensor positions [m]
    :param v_fdoa: FDOA sensor velocities [m/s]
    :param v_source: Source velocity [m/s] [assumed zero if not provided]
    :param zeta: Combined measurement vector
    :param cov: Measurement error covariance matrix
    :param x_ctr: Center of search grid [m]
    :param search_size: 2-D vector of search grid sizes [m]
    :param epsilon: Desired resolution of search grid [m]
    :param do_2d_aoa: Optional boolean parameter specifying whether 1D (az-only) or 2D (az/el) AOA is being performed
    :param tdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for TDOA
    :param fdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for FDOA
    :param do_resample: Boolean flag; if true the covariance matrix will be resampled, using ref_idx
    :return x_est: Estimated source position [m]
    :return likelihood: Likelihood computed across the entire set of candidate source positions
    :return x_grid: Candidate source positions
    """

    if do_resample:
        cov = cov.resample_hybrid(x_aoa=x_aoa, x_tdoa=x_tdoa, x_fdoa=x_fdoa, do_2d_aoa=do_2d_aoa,
                                  tdoa_ref_idx=tdoa_ref_idx, fdoa_ref_idx=fdoa_ref_idx)

    # Set up function handle
    def ell(x):
        return model.log_likelihood(x_aoa=x_aoa, x_tdoa=x_tdoa,
                                    x_fdoa=x_fdoa, v_fdoa=v_fdoa,
                                    zeta=zeta, x_source=x, v_source=v_source,
                                    cov=cov, do_2d_aoa=do_2d_aoa,
                                    tdoa_ref_idx=tdoa_ref_idx, fdoa_ref_idx=fdoa_ref_idx, do_resample=False,
                                    angle_bias=angle_bias, range_bias=range_bias, range_rate_bias=range_rate_bias)

    # Call the util function
    x_est, likelihood, x_grid = solvers.ml_solver(ell=ell, x_ctr=x_ctr, search_size=search_size, epsilon=epsilon,
                                                  **kwargs)

    return x_est, likelihood, x_grid


def max_likelihood_uncertainty(zeta, cov: CovarianceMatrix, cov_pos: CovarianceMatrix,
                               x_aoa=None, x_tdoa=None, x_fdoa=None, v_fdoa=None, v_source=None,
                               x_ctr=0., search_size=1., epsilon=None,
                               do_2d_aoa=False, tdoa_ref_idx=None, fdoa_ref_idx=None, do_resample=False,
                               do_aoa_bias=False, do_tdoa_bias=False, do_fdoa_bias=False, **kwargs):
    """
    Construct the ML Estimate by systematically evaluating the log
    likelihood function at a series of coordinates, and returning the index
    of the maximum.  Optionally returns the full set of evaluated
    coordinates, as well.
    # TODO: Test

    :param x_aoa: AOA sensor positions [m]
    :param x_tdoa: TDOA sensor positions [m]
    :param x_fdoa: FDOA sensor positions [m]
    :param v_fdoa: FDOA sensor velocities [m/s]
    :param v_source: Source velocity [m/s] [assumed zero if not provided]
    :param zeta: Combined measurement vector
    :param cov: Measurement error covariance matrix
    :param cov_pos: Sensor position error covariance matrix
    :param x_ctr: Center of search grid [m]
    :param search_size: 2-D vector of search grid sizes [m]
    :param epsilon: Desired resolution of search grid [m]
    :param do_2d_aoa: Optional boolean parameter specifying whether 1D (az-only) or 2D (az/el) AOA is being performed
    :param tdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for TDOA
    :param fdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for FDOA
    :param do_resample: Boolean flag; if true the covariance matrix will be resampled, using ref_idx
    :param do_aoa_bias: Flag for AOA measurement bias
    :param do_tdoa_bias: Flag for TDOA measurement bias
    :param do_fdoa_bias: Flag for FDOA measurement bias
    :return x_est: Estimated source position [m]
    :return bias_est: Estimated sensor bias (dict with fields 'aoa', 'tdoa', and 'fdoa')
    :return sensor_pos_est: Estimated sensor positions (dict with fields 'aoa', 'tdoa', and 'fdoa')
    :return sensor_vel_est: Estimated FDOA sensor velocities [m/s]
    :return likelihood: Likelihood computed across the entire set of candidate source positions
    :return x_grid: Candidate source positions
    """

    # Parse Inputs
    num_dim = 0
    num_aoa = 0
    num_tdoa = 0
    num_fdoa = 0
    if x_aoa is not None:
        num_dim, num_aoa = utils.safe_2d_shape(x_aoa)
    if x_tdoa is not None:
        num_dim, num_tdoa = utils.safe_2d_shape(x_tdoa)
    if x_fdoa is not None:
        num_dim, num_fdoa = utils.safe_2d_shape(x_fdoa)
    if num_dim == 0:
        raise ValueError('No sensor positions provided. At least one of x_aoa, x_tdoa, x_fdoa must be defined.')

    # Resample the covariance matrix, if needed
    if do_resample:
        cov = cov.resample_hybrid(x_aoa=x_aoa, x_tdoa=x_tdoa, x_fdoa=x_fdoa, do_2d_aoa=do_2d_aoa,
                                  tdoa_ref_idx=tdoa_ref_idx, fdoa_ref_idx=fdoa_ref_idx)

    # Make sure the search space is properly defined, and parse the parameter indices
    search_params = {'th_center': np.concatenate((x_ctr,v_source), axis=None) if v_source is not None else x_ctr,
                     'search_size': search_size,
                     'search_resolution': epsilon,
                     'do_aoa_bias': do_aoa_bias,
                     'do_tdoa_bias': do_tdoa_bias,
                     'do_fdoa_bias': do_fdoa_bias,
                     'x_aoa': x_aoa,
                     'x_tdoa': x_tdoa,
                     'x_fdoa': x_fdoa,
                     'v_fdoa': v_fdoa}
    search_center, search_size, search_resolution, param_indices = utils.make_uncertainty_search_space(**search_params)

    # Set up function handle
    # We must take care to ensure that it can handle an n_th x N matrix of
    # inputs; for compatibility with how utils.solvers.ml_solver will call it.

    # Set up function handle
    def ell(theta):
        return model.log_likelihood_uncertainty(x_aoa=x_aoa, x_tdoa=x_tdoa, x_fdoa=x_fdoa, v_fdoa=v_fdoa,
                                                zeta=zeta, theta=theta, cov=cov, cov_pos=cov_pos,
                                                do_2d_aoa=do_2d_aoa,
                                                tdoa_ref_idx=tdoa_ref_idx, fdoa_ref_idx=fdoa_ref_idx, do_resample=False)

    # Call the util function
    th_est, likelihood, x_grid = solvers.ml_solver(ell=ell, x_ctr=x_ctr, search_size=search_size, epsilon=epsilon,
                                                   **kwargs)

    x_est = th_est[param_indices['source_pos']]
    bias_est = {'aoa': th_est[param_indices['aoa_bias']] if do_aoa_bias else None,
                'tdoa': th_est[param_indices['tdoa_bias']] if do_tdoa_bias else None,
                'fdoa': th_est[param_indices['fdoa_bias']] if do_fdoa_bias else None}
    if cov_pos is None:
        sensor_pos_est = None
        sensor_vel_est = None
    else:
        sensor_pos_est = {'aoa': np.reshape(th_est[param_indices['aoa_pos']], (num_dim, num_aoa)),
                          'tdoa': np.reshape(th_est[param_indices['tdoa_pos']], (num_dim, num_tdoa)),
                          'fdoa': np.reshape(th_est[param_indices['fdoa_pos']], (num_dim, num_fdoa))}
        sensor_vel_est = np.reshape(th_est[param_indices['fdoa_vel']], (num_dim, num_fdoa)) \
            if num_fdoa > 0 else None

    return x_est, bias_est, sensor_pos_est, sensor_vel_est, likelihood, x_grid


def gradient_descent(zeta, cov: CovarianceMatrix, x_init, x_aoa=None, x_tdoa=None, x_fdoa=None, v_fdoa=None,
                     v_source=None, do_2d_aoa=False, tdoa_ref_idx=None, fdoa_ref_idx=None, do_resample=False,
                     angle_bias=None, range_bias=None, range_rate_bias=None, **kwargs):
    """
    Computes the gradient descent solution for FDOA processing.

    Ported from MATLAB code.

    Nicholas O'Donoughue
    21 February 2021

    :param zeta: Combined measurement vector
    :param cov: FDOA error covariance matrix
    :param x_aoa: AOA sensor positions [m]
    :param x_tdoa: TDOA sensor positions [m]
    :param x_fdoa: FDOA sensor positions [m]
    :param v_fdoa: FDOA sensor velocities [m/s]
    :param v_source: Source velocity [m/s] [assumed zero if not provided]
    :param x_init: Initial estimate of source position [m]
    :param do_2d_aoa: Optional boolean parameter specifying whether 1D (az-only) or 2D (az/el) AOA is being performed
    :param tdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings, for TDOA
    :param fdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings, for FDOA
    :param do_resample: Boolean flag; if true the covariance matrix will be resampled, using ref_idx
    :return x: Estimated source position
    :return x_full: Iteration-by-iteration estimated source positions
    """

    # Initialize measurement error and jacobian functions
    def y(this_x):
        return zeta - model.measurement(x_aoa=x_aoa, x_tdoa=x_tdoa, x_fdoa=x_fdoa, v_fdoa=v_fdoa,
                                        x_source=this_x, v_source=v_source, do_2d_aoa=do_2d_aoa,
                                        tdoa_ref_idx=tdoa_ref_idx, fdoa_ref_idx=fdoa_ref_idx,
                                        angle_bias=angle_bias, range_bias=range_bias, range_rate_bias=range_rate_bias)

    def jacobian(this_x):
        return model.jacobian(x_aoa=x_aoa, x_tdoa=x_tdoa, x_fdoa=x_fdoa, v_fdoa=v_fdoa,
                              x_source=this_x, v_source=v_source,
                              do_2d_aoa=do_2d_aoa, tdoa_ref_idx=tdoa_ref_idx, fdoa_ref_idx=fdoa_ref_idx)

    # Re-sample the covariance matrix, if needed
    if do_resample:
        cov = cov.resample_hybrid(x_aoa=x_aoa, x_tdoa=x_tdoa, x_fdoa=x_fdoa,
                                  do_2d_aoa=do_2d_aoa, tdoa_ref_idx=tdoa_ref_idx, fdoa_ref_idx=fdoa_ref_idx)

    # Call generic Gradient Descent solver
    x, x_full = solvers.gd_solver(y=y, jacobian=jacobian, cov=cov, x_init=x_init, **kwargs)

    return x, x_full


def least_square(zeta, cov: CovarianceMatrix, x_init, x_aoa=None, x_tdoa=None, x_fdoa=None, v_fdoa=None, v_source=None,
                 do_2d_aoa=False, tdoa_ref_idx=None, fdoa_ref_idx=None, do_resample=False,
                 angle_bias=None, range_bias=None, range_rate_bias=None, **kwargs):
    """
    Computes the least square solution for FDOA processing.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    21 February 2021

    :param zeta: Combined measurement vector
    :param cov: Measurement Error Covariance Matrix [(m/s)^2]
    :param x_init: Initial estimate of source position [m]
    :param x_aoa: AOA sensor positions [m]
    :param x_tdoa: TDOA sensor positions [m]
    :param x_fdoa: FDOA sensor positions [m]
    :param v_fdoa: FDOA sensor velocities [m/s]
    :param v_source: Source velocity [m/s] [assumed zero if not provided]
    :param do_2d_aoa: Optional boolean parameter specifying whether 1D (az-only) or 2D (az/el) AOA is being performed
    :param tdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for TDOA
    :param fdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for FDOA
    :param do_resample: Boolean flag; if true the covariance matrix will be resampled, using ref_idx
    :return x: Estimated source position
    :return x_full: Iteration-by-iteration estimated source positions
    """

    # Initialize measurement error and Jacobian function handles
    def y(this_x):
        return zeta - model.measurement(x_aoa=x_aoa, x_tdoa=x_tdoa, x_fdoa=x_fdoa, v_fdoa=v_fdoa,
                                        x_source=this_x, v_source=v_source, do_2d_aoa=do_2d_aoa,
                                        tdoa_ref_idx=tdoa_ref_idx, fdoa_ref_idx=fdoa_ref_idx,
                                        angle_bias=angle_bias, range_bias=range_bias, range_rate_bias=range_rate_bias)

    def jacobian(this_x):
        return model.jacobian(x_aoa=x_aoa, x_tdoa=x_tdoa, x_fdoa=x_fdoa, v_fdoa=v_fdoa,
                              x_source=this_x, v_source=v_source, do_2d_aoa=do_2d_aoa,
                              tdoa_ref_idx=tdoa_ref_idx, fdoa_ref_idx=fdoa_ref_idx)

    # Re-sample the covariance matrix, if needed
    if do_resample:
        cov = cov.resample_hybrid(x_aoa=x_aoa, x_tdoa=x_tdoa, x_fdoa=x_fdoa, do_2d_aoa=do_2d_aoa,
                                  tdoa_ref_idx=tdoa_ref_idx, fdoa_ref_idx=fdoa_ref_idx)

    # Call the generic Least Square solver
    x, x_full = solvers.ls_solver(zeta=y, jacobian=jacobian, cov=cov, x_init=x_init, **kwargs)

    return x, x_full


def bestfix(zeta, cov: CovarianceMatrix, x_aoa=None, x_tdoa=None, x_fdoa=None, v_fdoa=None, v_source=None, x_ctr=0.,
            search_size=1., epsilon=None, do_2d_aoa=False, tdoa_ref_idx=None, fdoa_ref_idx=None, do_resample=False,
            pdf_type=None):
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

    :param zeta: Combined measurement vector
    :param cov: Measurement error covariance matrix
    :param x_aoa: AOA sensor positions [m]
    :param x_tdoa: TDOA sensor positions [m]
    :param x_fdoa: FDOA sensor positions [m]
    :param v_fdoa: FDOA sensor velocities [m/s]
    :param v_source: Source velocity [m/s] [assumed zero if not provided]
    :param x_ctr: Center of search grid [m]
    :param search_size: 2-D vector of search grid sizes [m]
    :param epsilon: Desired resolution of search grid [m]
    :param do_2d_aoa: Optional boolean parameter specifying whether 1D (az-only) or 2D (az/el) AOA is being performed
    :param tdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for TDOA
    :param fdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for FDOA
    :param do_resample: Boolean flag; if true the covariance matrix will be resampled, using ref_idx
    :param pdf_type: String indicating the type of distribution to use. See +utils/makePDFs.m for options.
    :return x_est: Estimated source position [m]
    :return likelihood: Likelihood computed across the entire set of candidate source positions
    :return x_grid: Candidate source positions
    """

    # Generate the PDF
    def measurement(this_x):
        return model.measurement(x_aoa=x_aoa, x_tdoa=x_tdoa, x_fdoa=x_fdoa, v_fdoa=v_fdoa,
                                 x_source=this_x, v_source=v_source, do_2d_aoa=do_2d_aoa,
                                 tdoa_ref_idx=tdoa_ref_idx, fdoa_ref_idx=fdoa_ref_idx)

    # Re-sample the covariance matrix, if needed
    if do_resample:
        cov = cov.resample_hybrid(x_aoa=x_aoa, x_tdoa=x_tdoa, x_fdoa=x_fdoa, do_2d_aoa=do_2d_aoa,
                                  tdoa_ref_idx=tdoa_ref_idx, fdoa_ref_idx=fdoa_ref_idx)

    pdfs = utils.make_pdfs(measurement, zeta, pdf_type, cov.cov)

    # Call the util function
    x_est, likelihood, x_grid = solvers.bestfix(pdfs, x_ctr, search_size, epsilon)

    return x_est, likelihood, x_grid