import utils
from utils import solvers
from . import model


def max_likelihood(zeta, cov, x_aoa=None, x_tdoa=None, x_fdoa=None, v_fdoa=None, v_source=None,
                   x_ctr=0., search_size=1., epsilon=None, do_2d_aoa=False, tdoa_ref_idx=None, fdoa_ref_idx=None,
                   do_resample=False, cov_is_inverted=False):
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
    :param cov_is_inverted: Boolean flag, if false then cov is the covariance matrix. If true, then it is the
                            inverse of the covariance matrix.
    :return x_est: Estimated source position [m]
    :return likelihood: Likelihood computed across the entire set of candidate source positions
    :return x_grid: Candidate source positions
    """

    # Set up function handle
    def ell(x):
        return model.log_likelihood(x_aoa=x_aoa, x_tdoa=x_tdoa,
                                    x_fdoa=x_fdoa, v_fdoa=v_fdoa,
                                    zeta=zeta, x_source=x, v_source=v_source,
                                    cov=cov, do_2d_aoa=do_2d_aoa,
                                    tdoa_ref_idx=tdoa_ref_idx, fdoa_ref_idx=fdoa_ref_idx, do_resample=do_resample,
                                    cov_is_inverted=cov_is_inverted)

    # Call the util function
    x_est, likelihood, x_grid = solvers.ml_solver(ell=ell, x_ctr=x_ctr, search_size=search_size, epsilon=epsilon)

    return x_est, likelihood, x_grid


def gradient_descent(zeta, cov, x_init, x_aoa=None, x_tdoa=None, x_fdoa=None, v_fdoa=None, v_source=None,
                     do_2d_aoa=False, tdoa_ref_idx=None, fdoa_ref_idx=None, do_resample=False, **kwargs):
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
                                        tdoa_ref_idx=tdoa_ref_idx, fdoa_ref_idx=fdoa_ref_idx)

    def jacobian(this_x):
        return model.jacobian(x_aoa=x_aoa, x_tdoa=x_tdoa, x_fdoa=x_fdoa, v_fdoa=v_fdoa,
                              x_source=this_x, v_source=v_source,
                              do_2d_aoa=do_2d_aoa, tdoa_ref_idx=tdoa_ref_idx, fdoa_ref_idx=fdoa_ref_idx)

    # Re-sample the covariance matrix, if needed
    if do_resample:
        cov = model.resample_hybrid_covariance_matrix(cov, x_aoa, x_tdoa, x_fdoa, do_2d_aoa, tdoa_ref_idx, fdoa_ref_idx)

    # Call generic Gradient Descent solver
    x, x_full = solvers.gd_solver(y=y, jacobian=jacobian, covariance=cov, x_init=x_init, **kwargs)

    return x, x_full


def least_square(zeta, cov, x_init, x_aoa=None, x_tdoa=None, x_fdoa=None, v_fdoa=None, v_source=None,
                 do_2d_aoa=False, tdoa_ref_idx=None, fdoa_ref_idx=None, do_resample=False, **kwargs):
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
                                        tdoa_ref_idx=tdoa_ref_idx, fdoa_ref_idx=fdoa_ref_idx)

    def jacobian(this_x):
        return model.jacobian(x_aoa=x_aoa, x_tdoa=x_tdoa, x_fdoa=x_fdoa, v_fdoa=v_fdoa,
                              x_source=this_x, v_source=v_source, do_2d_aoa=do_2d_aoa,
                              tdoa_ref_idx=tdoa_ref_idx, fdoa_ref_idx=fdoa_ref_idx)

    # Re-sample the covariance matrix, if needed
    if do_resample:
        cov = model.resample_hybrid_covariance_matrix(cov, x_aoa, x_tdoa, x_fdoa, do_2d_aoa, tdoa_ref_idx, fdoa_ref_idx)

    # Call the generic Least Square solver
    x, x_full = solvers.ls_solver(zeta=y, jacobian=jacobian, covariance=cov, x_init=x_init, **kwargs)

    return x, x_full


def bestfix(zeta, cov, x_aoa=None, x_tdoa=None, x_fdoa=None, v_fdoa=None, v_source=None, x_ctr=0., search_size=1.,
            epsilon=None, do_2d_aoa=False, tdoa_ref_idx=None, fdoa_ref_idx=None, do_resample=False, pdf_type=None):
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
        cov = model.resample_hybrid_covariance_matrix(cov, x_aoa, x_tdoa, x_fdoa, do_2d_aoa, tdoa_ref_idx, fdoa_ref_idx)

    pdfs = utils.make_pdfs(measurement, zeta, pdf_type, cov)

    # Call the util function
    x_est, likelihood, x_grid = solvers.bestfix(pdfs, x_ctr, search_size, epsilon)

    return x_est, likelihood, x_grid
