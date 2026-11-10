import warnings

import numpy as np
from scipy.linalg import eigh, null_space
import time

from . import utils
from .covariance import CovarianceMatrix


def compute_crlb_gaussian(x_source, jacobian, cov: CovarianceMatrix, print_progress=False,
                          eq_constraints_grad:list = None, unobservable_dim_err: float = 1e99,
                          precision: float = 1e-30, rcond: float = 1e-12) \
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
     :param unobservable_dim_err: variance to assign along any unobservable direction (a large finite value is used
                                  instead of inf so that downstream covariance algebra does not produce NaNs). Keep
                                  this moderate (e.g. 1e6) if unobservable directions may not be axis-aligned; a
                                  value like the 1e99 default swamps the observable part of the matrix in floating
                                  point when the direction is rotated.
     :param precision: any numbers less than this amount will be rounded to zero.
     :param rcond: relative tolerance for observability; eigenvalues of the (constrained) Fisher Information Matrix
                   smaller than rcond times the largest eigenvalue are treated as zero
     :return crlb: n_dim x n_dim x n_source lower bound on the estimate covariance matrix

    Equality constraints are applied by reduced parameterization (Stoica and Ng, 1998). With constraint gradients G
    (n_dim x n_constraint) and U an orthonormal basis for the null space of G', the constrained bound is

     C = U [U' F U]^{-1} U'

    Dimensions that are fixed exactly by a constraint therefore have zero variance. Directions along which the
    constrained Fisher Information Matrix is singular are unobservable; they are assigned variance
    unobservable_dim_err (along the direction itself, which need not be aligned with a coordinate axis), and a
    RuntimeWarning is issued.
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
    num_unobservable_sources = 0
    first_unobservable_idx = None

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

        # Resolve covariance matrix (may be position-dependent callable)
        this_cov = cov(this_x) if callable(cov) else cov

        # Compute the Fisher Information Matrix
        fisher_matrix = this_cov.solve_aca(this_jacobian, do_2d=True)

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
            fisher = np.real(fisher_matrix)
            fisher = 0.5 * (fisher + fisher.T)

            # Reduce to the subspace that satisfies the (linearized) equality constraints
            if do_eq_constraints:
                # noinspection PyUnboundLocalVariable
                basis = null_space(this_gradient.T)
            else:
                basis = np.eye(n_dim)
            fisher_red = basis.T @ fisher @ basis

            # Separate observable from unobservable directions, using a tolerance relative to the largest eigenvalue
            fisher_inv = np.zeros((n_dim, n_dim))
            if fisher_red.size > 0:
                eig_val, eig_vec = eigh(fisher_red)
                observable = eig_val > rcond * np.fmax(eig_val[-1], 0.0)
                if np.any(observable):
                    vec_obs = eig_vec[:, observable]
                    red_inv = (vec_obs / eig_val[observable]) @ vec_obs.T
                    red_inv[np.fabs(red_inv) < precision] = 0.0
                    fisher_inv = basis @ red_inv @ basis.T
                if not np.all(observable):
                    # Assign the unobservable error along each unobservable direction
                    vec_unobs = basis @ eig_vec[:, ~observable]
                    fisher_inv = fisher_inv + unobservable_dim_err * (vec_unobs @ vec_unobs.T)
                    num_unobservable_sources += 1
                    if first_unobservable_idx is None:
                        first_unobservable_idx = idx

            crlb.append(CovarianceMatrix(fisher_inv))

    if print_progress:
        print('done')
        t_elapsed = time.perf_counter() - t_start
        utils.print_elapsed(t_elapsed)

    if num_unobservable_sources > 0:
        warnings.warn(f'CRLB has unobservable direction(s) at {num_unobservable_sources} of {n_source} source '
                      f'position(s) (first at index {first_unobservable_idx}); variance along those directions is '
                      f'set to unobservable_dim_err={unobservable_dim_err:g}. Equality constraints may restore '
                      f'observability.', RuntimeWarning, stacklevel=2)

    if n_source == 1:
        # There's only one source, pull the CovarianceMatrix object out of the list it's in
        crlb = crlb[0]

    return crlb
