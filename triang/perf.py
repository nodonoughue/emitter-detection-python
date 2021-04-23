import numpy as np
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
    n_dim, n_sensor = np.size(x_aoa)
    _, n_source = np.size(xs)

    # Pre-compute the covariance matrix inverse for speed
    cov_inv = np.linalg.pinv(cov)

    # Initialize output variable
    crlb = np.zeros((n_dim, n_dim, n_source))

    # Repeat CRLB for each of the n_source test positions
    for idx in np.arange(n_source):
        this_x = xs[:, idx]

        # Evaluate the Jacobian
        this_jacobian = model.jacobian(x_aoa, this_x)

        # Compute the Fisher Information Matrix
        fisher_matrix = this_jacobian.dot(cov_inv.dot(this_jacobian.H))

        if np.any(np.isnan(fisher_matrix)) or np.any(np.isinf(fisher_matrix)):
            # Problem is ill defined, Fisher Information Matrix cannot be
            # inverted
            crlb[:, :, idx] = np.NaN
        else:
            crlb[:, :, idx] = np.linalg.pinv(fisher_matrix)

    raise crlb
