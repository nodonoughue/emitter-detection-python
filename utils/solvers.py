import numpy as np
import matplotlib.pyplot as plt


def ls_soln(msmt, jacobian, covariance, x_init, epsilon=1e-6, max_num_iterations=10e3, force_full_calc=False, 
            plot_progress=False):
    """
    Computes the least square solution for TDOA processing.
    
    Ported from MATLAB Code.
    
    Nicholas O'Donoughue
    14 January 2021
    
    :param msmt: Measurement vector function handle (accepts n_dim vector of source position estimate, responds with 
                 error between received and modeled data vector)
    :param jacobian: Jacobian matrix function handle (accepts n_dim vector of source position estimate, and responds 
                     with n_dim x n_sensor Jacobian matrix
    :param covariance: Measurement error covariance matrix
    :param x_init: Initial estimate of source position
    :param epsilon: Desired position error tolerance (stopping condition)
    :param max_num_iterations: Maximum number of LS iterations to perform
    :param force_full_calc: Forces all max_num_iterations to be calculated
    :param plot_progress: Binary flag indicating whether to plot error/pos est over time
    :return x: Estimated source position.
    :return x_full: Iteration-by-iteration estimated source positions
    """

    # Parse inputs
    n_dims = np.size(x_init)

    # Initialize loop
    current_iteration = 1
    error = np.inf
    x_full = np.zeros(shape=(n_dims, max_num_iterations))
    x_prev = x_init
    x_full[:, 0] = x_prev

    # Compute the pseudoinverse
    covariance_pinv = np.linalg.pinv(covariance)

    # Initialize Plotting
    if plot_progress:
        _, ax = plt.subplots()
        plt.xlabel('Iteration Number')
        plt.ylabel('Change in Position Estimate')
        plt.yscale('log')

    # Divergence Detection
    num_expanding_iters = 0
    max_num_expanding_iters = 10
    prev_error = np.inf

    # Loop until either the desired tolerance is achieved or the maximum number of iterations have occurred
    while current_iteration < max_num_iterations and (force_full_calc or error >= epsilon):
        current_iteration += 1

        # Evaluate Residual and Jacobian Matrix
        y_i = msmt(x_prev)
        jacobian_i = jacobian(x_prev)

        # Compute delta_x^(i), according to 13.18
        j_c = np.inner(jacobian_i, covariance_pinv)
        j_c_j = np.inner(j_c, jacobian_i.H)
        delta_x = np.inner(np.linalg.lstsq(j_c_j, j_c), y_i)

        # Update predicted location
        x_full[:, current_iteration] = x_prev + delta_x

        # Update variables
        x_prev = x_full[:, current_iteration]
        error = np.linalg.norm(delta_x)

        if plot_progress:
            plt.plot(current_iteration, error, '.')

        # Check for divergence
        if error <= prev_error:
            num_expanding_iters = 0
        else:
            num_expanding_iters += 1
            if num_expanding_iters >= max_num_expanding_iters:
                # Divergence detected
                x_full[:, current_iteration:] = np.nan
                break
                
        prev_error = error

    # Bookkeeping
    if not force_full_calc:
        x_full = x_full[:, :current_iteration]

    x = x_full[:, -1]

    return x, x_full


def gd_soln(y, jacobian, covariance, x_init, alpha=0.3, beta=0.8, epsilon=1.e-6, max_num_iterations=10e3, 
            force_full_calc=False, plot_progress=False):
    """
    Computes the gradient descent solution for localization given the provided measurement and Jacobian function 
    handles, and measurement error covariance.

    Ported from MATLAB code.
    
    Nicholas O'Donoughue
    14 January 2021
        
    :param y: Measurement vector function handle (accepts n_dim vector of source position estimate, responds with error 
              between received and modeled data vector)
    :param jacobian: Jacobian matrix function handle (accepts n_dim vector of source position estimate, and responds 
                     with n_dim x n_sensor Jacobian matrix
    :param covariance: Measurement error covariance matrix
    :param x_init: Initial estimate of source position
    :param alpha: Backtracking line search parameter
    :param beta: Backtracking line search parameter
    :param epsilon: Desired position error tolerance (stopping condition)
    :param max_num_iterations: Maximum number of LS iterations to perform
    :param force_full_calc: Forces all max_num_iterations to be executed
    :param plot_progress: Binary flag indicating whether to plot error/pos est over time
    :return x: Estimated source position
    :return x_full: Iteration-by-iteration estimated source positions
    """
    # Parse inputs
    n_dims = np.size(x_init)
        
    # Initialize loop
    current_iteration = 1
    error = np.inf
    x_full = np.zeros(shape=(n_dims, max_num_iterations))
    x_prev = x_init
    x_full[:, 0] = x_prev
    
    # Pre-compute covariance matrix inverses
    covariance_pinv = np.linalg.pinv(covariance)
    
    # Cost Function for Gradient Descent
    def cost_fxn(z):
        this_y = y(z)
        return np.inner(np.inner(this_y.H, covariance_pinv*this_y))
    
    # Initialize Plotting
    if plot_progress:
        fig_plot, ax_set = plt.subplots(ncols=1, nrows=2)
        plt.xlabel(ax_set[0], 'Iteration Number')
        plt.ylabel(ax_set[0], 'Change in Position Estimate')
        plt.yscale(ax_set[0], 'log')
        
        plt.xlabel(ax_set[1], 'x')
        plt.ylabel(ax_set[1], 'y')
    
    # Divergence Detection
    num_expanding_iters = 0
    max_num_expanding_iters = 5
    prev_error = np.inf
    
    # Loop until either the desired tolerance is achieved or the maximum
    # number of iterations have occurred
    while current_iteration < max_num_iterations and (force_full_calc or error >= epsilon):
        current_iteration += 1
    
        # Evaluate Residual and Jacobian Matrix
        y_i = y(x_prev)
        jacobian_i = jacobian(x_prev)
        
        # Compute Gradient and Cost function
        grad = -2*np.inner(np.inner(jacobian_i, covariance_pinv), y_i)
        
        # Descent direction is the negative of the gradient
        del_x = -grad/np.linalg.norm(grad)
        
        # Compute the step size
        t = backtracking_line_search(cost_fxn, x_prev, grad, del_x, alpha, beta)
        
        # Update x position
        x_full[:, current_iteration] = x_prev + t*del_x
        
        # Update variables
        x_prev = x_full[:, current_iteration]
        error = t
        
        if plot_progress:
            plt.plot(ax_set[0], current_iteration, error, '.')
            plt.plot(ax_set[1], x_full[0, np.arange(current_iteration)], x_full[1, np.arange(current_iteration)], '-+')
        
        # Check for divergence
        if error <= prev_error:
            num_expanding_iters = 0
        else:
            num_expanding_iters += 1
            if num_expanding_iters >= max_num_expanding_iters:
                # Divergence detected
                x_full[:, current_iteration:] = np.NaN
                break

        prev_error = error

    # Bookkeeping
    if not force_full_calc:
        x_full = x_full[:, :current_iteration]

    x = x_full[:, -1]

    return x, x_full
        

def backtracking_line_search(f, x, grad, del_x, alpha, beta):
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
    :param alpha: Constant between 0 and 0.5
    :param beta: Constant between 0 and 1
    :return t: Optimal step size for the current estimate x.
    """

    # Initialize the search parameters and direction
    t = 1
    starting_val = f(x)
    slope = np.inner(grad.H, del_x)
    
    # Make sure the starting value is large enough
    while f(x+t*del_x) < starting_val+alpha*t*slope:
        t = 2*t
    
    # Conduct the backtracking line search
    while f(x+t*del_x) > starting_val+alpha*t*slope:
        t = beta*t

    return t


def ml_soln(ell, x_ctr, search_size, epsilon):
    """
    Execute ML estimation through brute force computational methods.

    Ported from MATLAB code.

    Nicholas O'Donoughue
    14 January 2021

    :param ell: Function handle for the likelihood of a given position must accept x_ctr (and similar sized vectors)
                as the sole input.
    :param x_ctr: Center position for search space (x, x/y, or z/y/z).
    :param search_size: Search space size (same units as x_ctr)
    :param epsilon: Search space resolution (same units as x_ctr)
    :return x_est: Estimated minimum
    :return A: Likelihood computed at each x position in the search space
    :return x_grid: Set of x positions for the entire search space (M x N) for N=1, 2, or 3.
    """

    n_dim = np.size(x_ctr)

    if n_dim < 1 or n_dim > 3:
        raise AttributeError('Number of spatial dimensions must be between 1 and 3')

    if np.size(search_size) == 1:
        search_size = search_size*np.ones_like(x_ctr)

    # Initialize search space
    xx = x_ctr[0] + np.arange(start=-search_size[0], step=epsilon, stop=search_size[0]+epsilon)
    if n_dim > 1:
        yy = x_ctr[1] + np.arange(start=-search_size[1], step=epsilon, stop=search_size[1]+epsilon)
        if n_dim > 2:
            zz = x_ctr[2] + np.arange(start=-search_size[2], step=epsilon, stop=search_size[2]+epsilon)

            xx_full, yy_full, zz_full = np.mgrid(xx, yy, zz)
            x_set = np.concatenate((xx_full.flatten(), yy_full.flatten(), zz_full.flatten()), axis=1).T
            x_grid = (xx, yy, zz)
        else:
            xx_full, yy_full = np.mgrid(xx, yy)
            x_set = np.concatenate((xx_full.flatten(), yy_full.flatten()), axis=1).T
            x_grid = (xx, yy)
    else:
        x_set = xx.flatten().T
        x_grid = xx

    # Evaluate the likelihood function at each coordinate in the search space
    likelihood = ell(x_set)

    # Find the peak
    idx_pk = np.argmax(likelihood.flatten())
    x_est = x_set[:, idx_pk]

    return x_est, likelihood, x_grid


def bestfix(pdfs, x_ctr, search_size, epsilon):
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
    :param x_ctr: Center position for search space (x, x/y, or z/y/z).
    :param search_size: Search space size (same units as x_ctr)
    :param epsilon: Search space resolution (same units as x_ctr)
    :return x_est: Estimated position
    :return result: Likelihood computed at each x position in the search space
    :return x_grid: Set of x positions for the entire search space (M x N) for N=1, 2, or 3.
    """

    n_dim = np.size(x_ctr)

    if n_dim < 1 or n_dim > 3:
        raise AttributeError('Number of spatial dimensions must be between 1 and 3')

    if np.size(search_size) == 1:
        search_size = search_size * np.ones_like(x_ctr)

    # Initialize search space
    xx = x_ctr[0] + np.arange(start=-search_size[0], step=epsilon, stop=search_size[0] + epsilon)
    if n_dim > 1:
        yy = x_ctr[1] + np.arange(start=-search_size[1], step=epsilon, stop=search_size[1] + epsilon)
        if n_dim > 2:
            zz = x_ctr[2] + np.arange(start=-search_size[2], step=epsilon, stop=search_size[2] + epsilon)

            xx_full, yy_full, zz_full = np.mgrid(xx, yy, zz)
            x_set = np.concatenate((xx_full.flatten(), yy_full.flatten(), zz_full.flatten()), axis=1).T
            x_grid = (xx, yy, zz)
        else:
            xx_full, yy_full = np.mgrid(xx, yy)
            x_set = np.concatenate((xx_full.flatten(), yy_full.flatten()), axis=1).T
            x_grid = (xx, yy)
    else:
        x_set = xx.flatten().T
        x_grid = xx

    # Apply each PDF to all input coordinates and then mutliply across PDFs
    result = np.prod(np.vstack([this_pdf(x_set).flatten() for this_pdf in pdfs]), axis=1)

    # Find the highest scoring position
    idx_pk = np.argmax(result.flatten())
    x_est = x_set[:, idx_pk]

    return x_est, result, x_grid