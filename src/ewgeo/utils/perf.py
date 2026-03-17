import numpy as np
from scipy.linalg import pinvh, cholesky, solve_triangular
import time

from . import utils
from .covariance import CovarianceMatrix


def compute_crlb_gaussian(x_source, jacobian, cov: CovarianceMatrix, print_progress=False,
                          eq_constraints_grad:list = None, unobservable_dim_err: float = 1e99,
                          precision: float = 1e-30) \
        -> CovarianceMatrix | list[CovarianceMatrix]:
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
     :param eq_constraints_grad: list of constraint gradients for equality constraints to be applied
     :param unobservable_dim_err: error value to assume for any unobservable dimensions
     :param precision: any numbers less than this amount will be rounded to zero.
     :return crlb: n_dim x n_dim x n_source lower bound on the estimate covariance matrix
    """

    # Parse inputs
    shp = np.shape(x_source)
    n_dim = shp[0] if len(shp) > 0 else 1
    n_source = shp[1] if len(shp) > 1 else 1
    if n_source==1 and len(x_source.shape) == 1:
        x_source = x_source[:, np.newaxis]

    do_eq_constraints = eq_constraints_grad is not None
    if do_eq_constraints:
        # Compute the gradient for all positions and store the result in an array of dimension
        #   num_constraints x n_dim x n_source
        constraint_grad = np.asarray([eq(x_source) for eq in eq_constraints_grad])

    # Initialize output variable
    crlb = []

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
    for idx in range(n_source):
        if print_progress:
            utils.print_progress(num_total=n_source, curr_idx=idx, iterations_per_marker=iter_per_marker,
                                 iterations_per_row=iter_per_row, t_start=t_start)

        this_x = x_source[:, idx]

        # Evaluate the Jacobian
        this_jacobian = jacobian(this_x)

        # Compute the Fisher Information Matrix
        fisher_matrix = cov.solve_aca(this_jacobian, do_2d=True)

        # Compute Constraint Gradients, if any
        if do_eq_constraints:
            # Grab the constraint gradient for the current source position
            # Transpose it so the dimensions are n_dim x num_constraints
            # noinspection PyUnboundLocalVariable
            this_gradient = constraint_grad[:, :, idx].T

        if np.any(np.isnan(fisher_matrix)) or np.any(np.isinf(fisher_matrix)):
            # Problem is ill-defined, Fisher Information Matrix cannot be
            # inverted
            # We'll use a matrix with NaN on the diagonal to represent this case.
            dummy = np.zeros((n_dim, n_dim))
            np.fill_diagonal(dummy, np.nan)
            crlb.append(CovarianceMatrix(dummy))
        else:
            fisher_inv = np.real(pinvh(fisher_matrix))
            if do_eq_constraints:
                # Apply the impact of equality constraints
                # noinspection PyUnboundLocalVariable
                fg = fisher_inv @ this_gradient
                gfg = this_gradient.T @ fg
                lower = cholesky(gfg, lower=True)

                res = solve_triangular(lower, fg.T, lower=True)

                fisher_const_inv = res.T @ res

                # Subtract the Fisher inverse for the constraint from the unconstrained Fisher inverse to yield the
                # constrained Fisher inverse
                fisher_inv = fisher_inv - fisher_const_inv

            # Apply precision
            fisher_inv[np.fabs(fisher_inv) < precision] = 0.0

            # Check for unobservable dimensions, represented by a zero along the diagonal
            diag_mask = np.diag(fisher_inv) < precision
            if np.any(diag_mask):
                fisher_inv[diag_mask, diag_mask] = unobservable_dim_err

            crlb.append(CovarianceMatrix(fisher_inv))

    if print_progress:
        print('done')
        t_elapsed = time.perf_counter() - t_start
        utils.print_elapsed(t_elapsed)

    if n_source == 1:
        # There's only one source, pull the CovarianceMatrix object out of the list it's in
        crlb = crlb[0]

    return crlb
