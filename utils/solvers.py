import numpy as np
import matplotlib.pyplot as plt
import utils
from utils import SearchSpace
from utils.covariance import CovarianceMatrix
from numpy import typing as npt


def ls_solver(zeta,
              jacobian,
              cov: CovarianceMatrix,
              x_init: npt.ArrayLike,
              epsilon:float=1e-6,
              max_num_iterations:int=int(10e3),
              force_full_calc:bool=False,
              plot_progress:bool=False,
              eq_constraints:list=None,
              ineq_constraints:list=None,
              constraint_tolerance:float=1e-6):
    """
    Computes the least square solution for geolocation processing.
    
    Ported from MATLAB Code.
    
    Nicholas O'Donoughue
    14 January 2021
    
    :param zeta: Measurement vector function handle (accepts n_dim vector of source position estimate, responds with
                 error between received and modeled data vector)
    :param jacobian: Jacobian matrix function handle (accepts n_dim vector of source position estimate, and responds 
                     with n_dim x n_sensor Jacobian matrix)
    :param cov: Measurement error covariance matrix
    :param x_init: Initial estimate of source position
    :param epsilon: Desired position error tolerance (stopping condition)
    :param max_num_iterations: Maximum number of LS iterations to perform
    :param force_full_calc: Forces all max_num_iterations to be calculated
    :param plot_progress: Binary flag indicating whether to plot error/pos est over time
    :param eq_constraints: List of equality constraint functions (see utils.constraints)
    :param ineq_constraints: List of inequality constraint functions (see utils.constraints)
    :param constraint_tolerance: Tolerance to apply to equality constraints (default = 1e-6); any deviations with a
                                 Euclidean norm less than this tolerance are considered to satisfy the constraint.
    :return x: Estimated source position.
    :return x_full: Iteration-by-iteration estimated source positions
    """

    # Parse inputs
    n_dims = np.size(x_init)

    # Make certain that eq_constraints and ineq_constraints are iterable
    if ineq_constraints is not None:
        utils.ensure_iterable(ineq_constraints, flatten=True)
    if eq_constraints is not None:
        utils.ensure_iterable(eq_constraints, flatten=True)

    # Initialize loop
    current_iteration = 0
    error = np.inf
    x_full = np.zeros(shape=(n_dims, max_num_iterations))
    x_prev = x_init
    x_full[:, current_iteration] = x_prev

    # Initialize Plotting
    if plot_progress:
        _, ax = plt.subplots()
        plt.xlabel('Iteration Number')
        plt.ylabel('Change in Position Estimate')
        plt.yscale('log')

    # Divergence Detection
    num_expanding_iterations = 0
    max_num_expanding_iterations = 10
    prev_error = np.inf

    # Loop until either the desired tolerance is achieved or the maximum number of iterations have occurred
    while current_iteration < (max_num_iterations-1) and (force_full_calc or error >= epsilon):
        current_iteration += 1

        # Evaluate Residual and Jacobian Matrix
        y_i = zeta(x_prev)
        jacobian_i = np.squeeze(jacobian(x_prev))  # Use the squeeze command to drop the third dim (n_source = 1)

        # Compute delta_x^(i), according to 10.20
        delta_x = cov.solve_lstsq(y_i, jacobian_i)

        # Update predicted location
        x_update = x_prev + np.squeeze(delta_x)

        # Apply Equality Constraints
        if eq_constraints is not None:
            x_update = utils.constraints.snap_to_equality_constraints(x_update, eq_constraints=eq_constraints,
                                                                      tol=constraint_tolerance)

        # Apply Inequality Constraints
        if ineq_constraints is not None:
            x_update = utils.constraints.snap_to_inequality_constraints(x_update, ineq_constraints=ineq_constraints)

        # TODO: What to do if both ineq and eq constraints are in use?

        # Update variables
        x_full[:, current_iteration] = x_update
        x_prev = x_update
        error = np.linalg.norm(delta_x)

        if plot_progress:
            plt.plot(current_iteration, error, '.')

        # Check for divergence
        if error <= prev_error:
            num_expanding_iterations = 0
        else:
            num_expanding_iterations += 1
            if num_expanding_iterations >= max_num_expanding_iterations:
                # Divergence detected
                x_full[:, current_iteration:] = np.nan
                break
                
        prev_error = error

    x = x_prev

    # Bookkeeping
    if not force_full_calc:
        x_full[:, current_iteration+1:] = x[:, np.newaxis]

    return x, x_full


def gd_solver(y,
              jacobian,
              cov: CovarianceMatrix,
              x_init:npt.ArrayLike,
              alpha:float=0.3,
              beta:float=0.8,
              epsilon:float=1.e-6,
              max_num_iterations:int=int(10e3),
              force_full_calc:bool=False,
              plot_progress:bool=False,
              eq_constraints:list=None,
              ineq_constraints:list=None,
              constraint_tolerance:float=1e-6):
    """
    Computes the gradient descent solution for localization given the provided measurement and Jacobian function 
    handles, and measurement error covariance.

    Ported from MATLAB code.
    
    Nicholas O'Donoughue
    29 April 2025
        
    :param y: Measurement vector function handle (accepts n_dim vector of source position estimate, responds with error 
              between received and modeled data vector)
    :param jacobian: Jacobian matrix function handle (accepts n_dim vector of source position estimate, and responds 
                     with n_dim x n_sensor Jacobian matrix)
    :param cov: Measurement error covariance matrix
    :param x_init: Initial estimate of source position
    :param alpha: Backtracking line search parameter
    :param beta: Backtracking line search parameter
    :param epsilon: Desired position error tolerance (stopping condition)
    :param max_num_iterations: Maximum number of LS iterations to perform
    :param force_full_calc: Forces all max_num_iterations to be executed
    :param plot_progress: Binary flag indicating whether to plot error/pos est over time
    :param eq_constraints: List of equality constraint functions (see utils.constraints)
    :param ineq_constraints: List of inequality constraint functions (see utils.constraints)
    :param constraint_tolerance: Tolerance to apply to equality constraints (default = 1e-6); any deviations with a
                                 Euclidean norm less than this tolerance are considered to satisfy the constraint.
    :return x: Estimated source position
    :return x_full: Iteration-by-iteration estimated source positions
    """
    # Parse inputs
    n_dims = np.size(x_init)

    # Make certain that eq_constraints and ineq_constraints are iterable
    if ineq_constraints is not None:
        utils.ensure_iterable(ineq_constraints, flatten=True)
    if eq_constraints is not None:
        utils.ensure_iterable(eq_constraints, flatten=True)

    # Initialize loop
    current_iteration = 0
    error = np.inf
    x_full = np.zeros(shape=(n_dims, max_num_iterations))
    x_prev = x_init
    x_full[:, current_iteration] = x_prev
    
    # Cost Function for Gradient Descent
    def cost_fxn(z):
        this_y = y(z)
        return cov.solve_aca(this_y.T)

    # Initialize Plotting
    if plot_progress:
        fig_plot, (ax1, ax2) = plt.subplots(ncols=1, nrows=2)
        ax1.set_xlabel('Iteration Number')
        ax1.set_ylabel('Change in Position Estimate')
        ax1.set_yscale('log')
        
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
    else:
        ax1 = ax2 = None  # This removes a 'variable possibly unset' warning

    # Divergence Detection
    num_expanding_iterations = 0
    max_num_expanding_iterations = 5
    prev_error = np.inf

    # Loop until either the desired tolerance is achieved or the maximum
    # number of iterations have occurred
    while current_iteration < (max_num_iterations-1) and (force_full_calc or error >= epsilon):
        current_iteration += 1
    
        # Evaluate Residual and Jacobian Matrix
        y_i = y(x_prev)
        jacobian_i = np.squeeze(jacobian(x_prev))  # Use squeeze to remove third dimension (n_source=1)
        
        # Compute Gradient and Cost function
        grad = -2 * cov.solve_acb(jacobian_i, y_i)
        # if cov_is_inverted:
        #     grad = -2 * jacobian_i @ covariance_inverse @ y_i
        # else:
        #     a = scipy.linalg.solve_triangular(covariance_lower, jacobian_i.T, lower=True)
        #     b = scipy.linalg.solve_triangular(covariance_lower, y_i, lower=True)
        #     grad = -2 * a.T @ b
        
        # Descent direction is the negative of the gradient
        del_x = -np.squeeze(grad/np.linalg.norm(grad))
        
        # Compute the step size
        t = backtracking_line_search(cost_fxn, x_prev, grad, del_x, alpha, beta)
        
        # Update x position
        x_update = x_prev + t*del_x

        # Apply Equality Constraints
        if eq_constraints is not None:
            x_update = utils.constraints.snap_to_equality_constraints(x_update, eq_constraints=eq_constraints,
                                                                      tol=constraint_tolerance)

        # Apply Inequality Constraints
        if ineq_constraints is not None:
            x_update = utils.constraints.snap_to_inequality_constraints(x_update, ineq_constraints=ineq_constraints)

        # TODO: What to do if both ineq and eq constraints are in use?

        # Update variables
        x_full[:, current_iteration] = x_update
        x_prev = x_update
        error = t
        
        if plot_progress:
            ax1.plot(current_iteration, error, '.')
            ax2.plot(x_full[0, np.arange(current_iteration)], x_full[1, np.arange(current_iteration)], '-+')
        
        # Check for divergence
        if error <= prev_error:
            num_expanding_iterations = 0
        else:
            num_expanding_iterations += 1
            if num_expanding_iterations >= max_num_expanding_iterations:
                # Divergence detected
                x_full[:, current_iteration:] = np.nan
                break

        prev_error = error

    x = x_prev

    # Bookkeeping
    if not force_full_calc:
        x_full[:, current_iteration + 1:] = x[:, np.newaxis]

    return x, x_full
        

def backtracking_line_search(f, x, grad, del_x, alpha=0.3, beta=0.8):
    """
    # Performs backtracking line search according to algorithm 9.2 of
    # Stephen Boyd's, Convex Optimization

    Ported from MATLAB Code

    Nicholas O'Donoughue
    14 January 2021

    :param f: Function handle to minimize
    :param x: Current estimate of x
    :param grad: Gradient of f at x
    :param del_x: Descent direction
    :param alpha: Constant between 0 and 0.5 (default=0.3)
    :param beta: Constant between 0 and 1 (default=0.8)
    :return t: Optimal step size for the current estimate x.
    """

    # Initialize the search parameters and direction
    t = 1
    starting_val = np.squeeze(f(x))
    slope = np.squeeze(np.conjugate(grad.T) @ del_x)

    # Make sure that x, del_x are arrays (not matrices)
    del_x = np.squeeze(del_x)
    x = np.squeeze(x)
    # Make sure the starting value is large enough
    while f(x+t*del_x) < starting_val+alpha*t*slope:
        t = 2*t
    
    # Conduct the backtracking line search
    while f(x+t*del_x) > starting_val+alpha*t*slope:
        t = beta*t

    return t


def ml_solver(ell, search_space: SearchSpace, eq_constraints=None, ineq_constraints=None, constraint_tolerance=None,
              prior=None, prior_wt: float = 0.):
    """
    Execute ML estimation through brute force computational methods.

    Ported from MATLAB code.

    Nicholas O'Donoughue
    14 January 2021

    :param ell: Function handle for the likelihood of a given position must accept x_ctr (and similar sized vectors)
                as the sole input.
    :param search_space: SearchSpace object defining the space over which to search
    :param eq_constraints: List of equality constraint functions (see utils.constraints)
    :param ineq_constraints: List of inequality constraint functions (see utils.constraints)
    :param constraint_tolerance: Tolerance to apply to equality constraints (default = 1e-6); any deviations with a
                                 Euclidean norm less than this tolerance are considered to satisfy the constraint.
    :param prior: Function handle that accepts one or more positions and returns the probability of that position being
                  the true source location, according to some prior distribution. Will be multiplied by log10 when
                  combined with the likelihood distribution (which is assumed to be a log likelihood).
    :param prior_wt: Weight to apply to the prior distribution; (1-prior_wt) will be applied to the likelihood function.
    :return x_est: Estimated minimum
    :return A: Likelihood computed at each x position in the search space
    :return x_grid: Set of x positions for the entire search space (M x N) for N=1, 2, or 3.
    """

    # Set up the search space
    x_set, x_grid, out_shape = utils.make_nd_grid(search_space)

    # Constrain the likelihood, if needed
    if ineq_constraints is not None or eq_constraints is not None:
        ell = utils.constraints.constrain_likelihood(ell=ell, eq_constraints=eq_constraints,
                                                     ineq_constraints=ineq_constraints, tol=constraint_tolerance)

    # Evaluate the likelihood function at each coordinate in the search space
    likelihood = ell(x_set)

    if prior is not None and prior_wt > 0:
        pdf_prior = np.reshape(prior(x_set), shape=likelihood.shape)

        likelihood = (1 - prior_wt) * likelihood + prior_wt * np.log10(pdf_prior)

    # Find the peak
    idx_pk = likelihood.argmax()
    x_est = x_set[:, idx_pk]

    return x_est, likelihood, x_grid


def bestfix(pdfs, search_space: SearchSpace):
    """
    Based on the BESTFIX algorithm, invented in 1990 by Eric Hodson (R&D Associates, now with Naval Postgraduate
    School).  Patent is believed to be in the public domain.

    Ref:
     Eric Hodson, "Method and arrangement for probabilistic determination of a target location,"
     U.S. Patent US5045860A, 1990, https://patents.google.com/patent/US5045860A

    Ported from MATLAB Code.

    Nicholas O'Donoughue
    14 January 2021

    :param pdfs: Lx1 cell list of PDF functions, each of which represents the probability that an input position
                 (Ndim x 3 array) is the true source position for one of the measurements.
    :return x_est: Estimated position
    :return result: Likelihood computed at each x position in the search space
    :return x_grid: Set of x positions for the entire search space (M x N) for N=1, 2, or 3.
    """

    # Set up the search space
    x_set, x_grid, out_shape = utils.make_nd_grid(search_space)

    # Apply each PDF to all input coordinates and then multiply across PDFs
    result = np.asarray([np.prod(np.asarray([this_pdf(this_x) for this_pdf in pdfs])) for this_x in x_set.T])

    # Reshape
    result = np.reshape(result, out_shape)

    # Find the highest scoring position
    idx_pk = result.argmax()
    x_est = x_set[:, idx_pk]

    return x_est, result, x_grid

def sensor_calibration(ell,
                       pos_search: SearchSpace,
                       vel_search: SearchSpace,
                       bias_search: SearchSpace,
                       num_iterations=1):
    """
    This function attempts to calibrate sensor uncertainties given a series of measurements (AOA, TDOA, and/or FDOA)
    against a set of calibration emitters. Sensor uncertainties can take the form of unknown measurement bias or
    position/velocity errors.

    This follows the logic in Figure 6.11 of the 2022 text, loosely summarized:
    1. Assume that the sensor positions and velocities are accurate, and estimate measurement biases.
    2. Use the estimated measurement biases to update the sensor positions.
    3. Use the estimated measurement biases and sensor positions to update sensor velocities.
    4. (Optionally) Repeat Steps 1-3 num_iterations times.

    Figure 6.11 shows this as a linear operation, but if the estimates are not accurate enough, repeated iterations
    of the process may be desired. This can be achieved by calling the sensor_calibration function again with the
    updated position and measurement bias estimates.

    :param ell: function handle that accepts two inputs: (bias, x_sensor); bias being an array of measurement biases,
                and x_sensor a 2D array of sensor positions.
    :param pos_search: dictionary with parameters for the ML search for sensor positions
    :param vel_search: dictionary with parameters for the ML search for sensor velocities
    :param bias_search: dictionary with parameters for the ML search for measurement bias
    :param num_iterations: number of times to repeat the calibration search
    :return x_sensor_est: Estimated sensor positions
    :return bias_est: Estimated measurement biases
    """

    # Initialize Outputs
    bias_est = bias_search.x_ctr if bias_search is not None else None
    x_sensor_est = pos_search.x_ctr if pos_search is not None else None
    v_sensor_est = vel_search.x_ctr if vel_search is not None else None

    for _ in np.arange(num_iterations):

        # ================= Estimate Measurement Bias ========================
        if bias_search is not None and np.any(bias_search.points_per_dim > 1):
            x_sensor_vec = pos_search.x_ctr
            v_sensor_vec = vel_search.x_ctr
            def ell_bias(bias):
                return ell(bias, x_sensor_vec, v_sensor_vec)

            result = utils.solvers.ml_solver(ell=ell_bias, search_space=bias_search)
            bias_est = result[0]
            bias_search.x_ctr = bias_est # store result as center for next iteration

        # =================== Estimate Sensor Position =========================
        if pos_search is not None and np.any(pos_search.points_per_dim > 1):
            v_sensor_vec = vel_search.x_ctr
            def ell_pos(x):
                return ell(bias_est, x, v_sensor_vec)

            result = utils.solvers.ml_solver(ell=ell_pos, search_space=pos_search)
            x_sensor_est = result[0]
            pos_search.x_ctr = x_sensor_est # store result as center for next iteration

        # =================== Estimate Sensor Velocity =========================
        if vel_search is not None and np.any(vel_search.points_per_dim > 1):
            def ell_vel(v):
                return ell(bias_est, x_sensor_est, v)

            result = utils.solvers.ml_solver(ell=ell_vel, search_space=vel_search)
            v_sensor_est = result[0]
            vel_search.x_ctr = v_sensor_est # store result as center for next iteration

    return x_sensor_est, v_sensor_est, bias_est
