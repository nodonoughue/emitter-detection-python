import numpy as np
import utils
from . import model
from utils.covariance import CovarianceMatrix


def compute_crlb(x_sensor, x_source, cov: CovarianceMatrix, ref_idx=None, do_resample=True, variance_is_toa=True,
                 print_progress=False):
    """
    Computes the CRLB on position accuracy for source at location xs and
    sensors at locations in x_tdoa (Ndim x N).

    Ported from MATLAB Code

    Nicholas O'Donoughue
    21 February 2021

    :param x_sensor: (Ndim x N) array of TDOA sensor positions
    :param x_source: (Ndim x M) array of source positions over which to calculate CRLB
    :param cov: TDOA measurement error covariance matrix; object of the CovarianceMatrix class
    :param ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings
    :param do_resample: Boolean flag; if true the covariance matrix will be resampled, using ref_idx
    :param variance_is_toa: Boolean flag; if true then the input covariance matrix is in units of s^2; if false, then
    it is in m^2
    :param print_progress: Boolean flag, if true then progress updates and elapsed/remaining time will be printed to
                           the console. [default=False]
    :return crlb: Lower bound on the error covariance matrix for an unbiased FDOA estimator (Ndim x Ndim)
    """

    # Parse inputs
    _, n_source = utils.safe_2d_shape(x_source)

    # Make sure that xs is 2D
    if n_source == 1:
        x_source = x_source[:, np.newaxis]

    if variance_is_toa:
        # Multiply by the speed of light squared, unless it is inverted (then divide)
        cov = cov.multiply(utils.constants.speed_of_light ** 2, overwrite=False)

    if do_resample:
        cov = cov.resample(ref_idx=ref_idx)

    # Define a wrapper for the jacobian matrix that accepts only the position 'x'
    def jacobian(x):
        return model.jacobian(x_sensor=x_sensor, x_source=x, ref_idx=ref_idx)

    crlb = utils.perf.compute_crlb_gaussian(x_source=x_source, jacobian=jacobian, cov=cov,
                                            print_progress=print_progress)

    return crlb
