import utils
from utils.covariance import CovarianceMatrix
import numpy as np
import time
from scipy.linalg import pinvh


def compute_crlb_gaussian(x_source, jacobian, cov: CovarianceMatrix, print_progress=False):
    """
    Computes the CRLB for a Gaussian problem at one or more source positions. The CRLB for Gaussian problems takes the
    general form:

     C >= F^{-1} = [J C_z^{-1} J^T]^{-1}

     where C is the covariance matrix for the estimate of x, J is the Jacobian evaluated at x, and C_z is the
     measurement (z) covariance matrix.

     Nicholas O'Donoughue
     25 February 2025

     :param x_source: n_dim x n_source ndarray of source positions
     :param jacobian: function that accepts a single source position and returns the n_dim x n_measurement Jacobian
     :param cov: Covariance Matrix object
     :param print_progress: Boolean flag; if true then elapsed/remaining time estimates will be printed to the console
     :return crlb: n_dim x n_dim x n_source lower bound on the estimate covariance matrix
    """

    # Parse inputs
    n_dim, n_source = utils.safe_2d_shape(x_source)

    # Initialize output variable
    crlb = np.zeros((n_dim, n_dim, n_source))

    # Print CRLB calculation progress, if desired
    markers_per_row = 40
    desired_num_rows = 10
    min_iter_per_marker = 10
    max_iter_per_marker = 10000
    iter_per_marker = int(np.floor(n_source / markers_per_row / desired_num_rows))
    iter_per_marker = np.fmin(max_iter_per_marker, np.fmax(min_iter_per_marker, iter_per_marker))
    iter_per_row = markers_per_row * iter_per_marker

    # at least 1 iteration per marker, no more than 100 iterations per marker
    t_start = time.perf_counter()

    if print_progress:
        print('Computing CRLB solution for {} source positions...'.format(n_source))

    # Repeat CRLB for each of the n_source test positions
    for idx in np.arange(n_source):
        if print_progress:
            utils.print_progress(num_total=n_source, curr_idx=idx, iterations_per_marker=iter_per_marker,
                                 iterations_per_row=iter_per_row, t_start=t_start)

        this_x = x_source[:, idx]

        # Evaluate the Jacobian
        this_jacobian = jacobian(this_x)

        # Compute the Fisher Information Matrix
        fisher_matrix = cov.solve_aca(this_jacobian)

        if np.any(np.isnan(fisher_matrix)) or np.any(np.isinf(fisher_matrix)):
            # Problem is ill-defined, Fisher Information Matrix cannot be
            # inverted
            crlb[:, :, idx] = np.nan
        else:
            crlb[:, :, idx] = np.real(pinvh(fisher_matrix))

    if print_progress:
        print('done')
        t_elapsed = time.perf_counter() - t_start
        utils.print_elapsed(t_elapsed)

    if n_source == 1:
        # There's only one source, trim the third dimension
        crlb = crlb[:, :, 0]

    return crlb
