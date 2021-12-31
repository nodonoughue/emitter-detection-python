import numpy as np
import scipy
import warnings

import utils
from . import model


def compute_crlb(x_aoa, xs, cov):
    """
    Computes the CRLB on position accuracy for source at location xs and
    sensors at locations in x_aoa (Ndim x N).  C is an NxN matrix of TOA
    covariances at each of the N sensors.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    22 February 2021

    :param x_aoa: (Ndim x N) array of AOA sensor positions
    :param xs: (Ndim x M) array of source positions over which to calculate CRLB
    :param cov: Covariance matrix for range rate estimates at the N FDOA sensors [(m/s)^2]
    :return crlb: Lower bound on the error covariance matrix for an unbiased FDOA estimator (Ndim x Ndim)
    """

    # Parse inputs
    num_dimension, num_sensors = utils.safe_2d_shape(x_aoa)
    num_dimension2, num_sources = utils.safe_2d_shape(xs)

    assert num_dimension == num_dimension2, "Sensor and Target positions must have the same number of dimensions"

    # Make sure that xs is a 2d array
    if len(np.shape(xs)) == 1:
        xs = np.expand_dims(xs, axis=1)

    # Pre-compute the covariance matrix inverse for speed
    cov_lower = np.linalg.cholesky(cov)

    # Initialize output variable
    crlb = np.zeros((num_dimension, num_dimension, num_sources))

    # Repeat CRLB for each of the n_source test positions
    for idx in np.arange(num_sources):
        this_x = xs[:, idx]

        # Evaluate the Jacobian
        this_jacobian = np.squeeze(model.jacobian(x_aoa, this_x))  # Remove the third dimension (n_source=1)

        if np.any(np.isnan(this_jacobian)):
            # This occurs when the ground range is zero for one of the sensors, which causes a divide by zero error.
            # In other words, if this_x overlaps perfectly with one of the sensors.
            crlb[:, :, idx] = np.NaN
            continue

        # Compute the Fisher Information Matrix
        a = scipy.linalg.solve_triangular(cov_lower, this_jacobian.T)
        fisher_matrix = a.T @ a

        if np.any(np.isnan(fisher_matrix)) or np.any(np.isinf(fisher_matrix)):
            # Problem is ill-defined, Fisher Information Matrix cannot be
            # inverted
            crlb[:, :, idx] = np.NaN
        else:
            crlb[:, :, idx] = np.real(scipy.linalg.pinvh(fisher_matrix))

    return crlb
