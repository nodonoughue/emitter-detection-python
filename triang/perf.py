import numpy as np
import utils
from . import model
from utils.covariance import CovarianceMatrix


def compute_crlb(x_aoa, x_source, cov: CovarianceMatrix, do_2d_aoa=False, print_progress=False, **kwargs):
    """
    Computes the CRLB on position accuracy for source at location xs and
    sensors at locations in x_aoa (Ndim x N).  C is an NxN matrix of TOA
    covariances at each of the N sensors.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    22 February 2021

    :param x_aoa: (Ndim x N) array of AOA sensor positions
    :param x_source: (Ndim x M) array of source positions over which to calculate CRLB
    :param cov: AOA measurement error covariance matrix; object of the CovarianceMatrix class
    :param do_2d_aoa: Optional boolean parameter specifying whether 1D (az-only) or 2D (az/el) AOA is being performed
    :param print_progress: Boolean flag, if true then progress updates and elapsed/remaining time will be printed to
                           the console. [default=False]
    :return crlb: Lower bound on the error covariance matrix for an unbiased FDOA estimator (Ndim x Ndim)
    """

    # Parse inputs
    num_dimension, num_sensors = utils.safe_2d_shape(x_aoa)
    num_dimension2, num_sources = utils.safe_2d_shape(x_source)

    assert num_dimension == num_dimension2, "Sensor and Target positions must have the same number of dimensions"

    # Make sure that xs is a 2d array
    if len(np.shape(x_source)) == 1:
        x_source = np.expand_dims(x_source, axis=1)

    # Define a wrapper for the jacobian matrix that accepts only the position 'x'
    def jacobian(x):
        return model.jacobian(x_sensor=x_aoa, x_source=x, do_2d_aoa=do_2d_aoa)

    crlb = utils.perf.compute_crlb_gaussian(x_source=x_source, jacobian=jacobian, cov=cov,
                                            print_progress=print_progress, **kwargs)

    return crlb
