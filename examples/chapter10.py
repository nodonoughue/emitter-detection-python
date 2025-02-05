import numpy as np
import matplotlib.pyplot as plt

import utils
import triang
import time


def run_all_examples():
    """
    Run all chapter 10 examples and return a list of figure handles

    :return figs: list of figure handles
    """

    # Random Number Generator
    rng = np.random.default_rng(0)

    # Colormap
    cmap = plt.get_cmap("tab10")

    fig1 = example1(rng, cmap)
    fig2 = example2()
    fig3 = example3()

    return [fig1, fig2, fig3]


def example1(rng=None, cmap=None):
    """
    Executes Example 10.1 and generates two figures

    Ported from MATLAB Code

    Added multiprocessing support to use all available CPU cores

    Nicholas O'Donoughue
    18 May 2021

    :param rng: random number generator
    :param cmap: colormap
    :return fig_geo: figure handle to generated graphic with geographic layout
    :return fig_err: figure handle to generated graphic with error as a function of iteration
    """

    # Random Number Generator
    if rng is None:
        rng = np.random.default_rng(0)

    if cmap is None:
        cmap = plt.get_cmap("tab10")

    # Clear the numpy warnings about underflow; we don't care
    # Underflow warnings can indicate a loss of precision; in our case, these are likely occurring
    # from positions where our sensors are poorly aligned to determined the target's location. We
    # can ignore the loss of precision there.
    np.seterr(under='ignore')


    # Define sensor positions
    x_sensor = 30.0*np.array([[-1., 0., 1.], [0.,  0., 0.]])
    num_dims, num_sensors = utils.safe_2d_shape(x_sensor)
    
    # Define source position
    x_source = np.array([15, 45])
    
    # Grab a noisy measurement
    psi_act = triang.model.measurement(x_sensor, x_source)
    covar_psi = (2*np.pi/180)**2 * np.eye(num_sensors)

    # Compute Ranges
    range_act = utils.geo.calc_range(x_sensor, x_source)

    # Error Values
    angle_error = 3*np.sqrt(np.diag(covar_psi))

    # Start first figure; geometry
    fig_geo = plt.figure()

    # Geometry
    for idx_sensor, this_psi in enumerate(psi_act):
        this_x = np.expand_dims(x_sensor[:, idx_sensor], axis=1)
        this_range = range_act[idx_sensor]
        this_err = angle_error[idx_sensor]

        this_color = cmap(idx_sensor)

        # Vector from sensor to source
        # dx = x_source - this_x

        # Find AOA
        # lob = this_x + np.array([[0.0, np.cos(this_psi)], [0.0, np.sin(this_psi)]]) * 5 * this_range
        lob_err1 = np.array([[0, np.cos(this_psi + this_err)], [0, np.sin(this_psi + this_err)]]) * 5 * this_range
        lob_err0 = np.array([[0, np.cos(this_psi - this_err)], [0, np.sin(this_psi - this_err)]]) * 5 * this_range
        lob_fill1 = this_x + np.concatenate((lob_err1, np.fliplr(lob_err0), np.expand_dims(lob_err1[:, 0], axis=1)),
                                            axis=1)

        # Plot the Uncertainty Interval
        plt.fill(lob_fill1[0, :], lob_fill1[1, :], linestyle='--', alpha=.1, edgecolor='k', facecolor=this_color,
                 label=None)

    # Position Markers
    plt.scatter(x_sensor[0, :], x_sensor[1, :], marker='o', label='Sensors')

    # Position Labels
    plt.text(float(x_sensor[0, 0] + 2), float(x_sensor[1, 0] - .1), r'$S_0$')
    plt.text(float(x_sensor[0, 1] + 2), float(x_sensor[1, 1] - .1), r'$S_1$')
    plt.text(float(x_sensor[0, 2] + 2), float(x_sensor[1, 2] - .1), r'$S_2$')

    # Plot the points and lobs
    plt.scatter(x_source[0], x_source[1], marker='^', label='Transmitter')

    # Iterative Methods
    epsilon = .5  # km
    num_mc_trials = 1000
    num_iterations = 50

    # Decompose the covariance matrix, using Cholesky Decomposition, into a lower triangular matrix, for generating
    # correlated random variables
    covar_lower = np.linalg.cholesky(covar_psi)

    out_shp = (2, num_mc_trials)
    out_iterative_shp = (2, num_iterations, num_mc_trials)
    x_ml = np.zeros(shape=out_shp)
    x_bf = np.zeros(shape=out_shp)
    x_centroid = np.zeros(shape=out_shp)
    x_incenter = np.zeros(shape=out_shp)
    x_ls_full = np.zeros(shape=out_iterative_shp)
    x_grad_full = np.zeros(shape=out_iterative_shp)

    print('Conducting MC trial for triangulation error...')
    t_start = time.perf_counter()

    x_init = np.array([5, 5])
    x_extent = np.array([50, 50])

    args = {'psi_act': psi_act,
            'num_sensors': num_sensors,
            'x_sensor': x_sensor,
            'x_init': x_init,
            'x_extent': x_extent,
            'covar_psi': covar_psi,
            'covar_lower': covar_lower,
            'epsilon': epsilon,
            'num_iterations': num_iterations,
            'rng': rng}

    iterations_per_marker = 1
    markers_per_row = 40
    iterations_per_row = markers_per_row * iterations_per_marker
    for idx in np.arange(num_mc_trials):
        utils.print_progress(num_mc_trials, idx, iterations_per_marker, iterations_per_row, t_start)

        result = _mc_iteration(args)
        x_ml[:, idx] = result['ml']
        x_bf[:, idx] = result['bf']
        x_centroid[:, idx] = result['centroid']
        x_incenter[:, idx] = result['incenter']
        x_ls_full[:, :, idx] = result['ls']
        x_grad_full[:, :, idx] = result['gd']

    print('done')
    t_elapsed = time.perf_counter() - t_start
    utils.print_elapsed(t_elapsed)

    # Plot Closed-Form Solutions
    plt.scatter(x_centroid[0, 0], x_centroid[1, 0], marker='*', label='Centroid')
    plt.scatter(x_incenter[0, 0], x_incenter[1, 0], marker='+', label='Incenter')
    plt.scatter(x_ml[0, 0], x_ml[1, 0], marker='v', label='Maximum Likelihood')
    plt.scatter(x_bf[0, 0], x_bf[1, 0], marker='o', label='BestFix')
    
    # Plot Iterative Solutions
    plt.scatter(5, 5, marker='x', label='Initial Estimate')
    plt.plot(x_ls_full[0, :, 0], x_ls_full[1, :, 0], linestyle=':', label='Least Squares')
    plt.plot(x_grad_full[0, :, 0], x_grad_full[1, :, 0], linestyle='--', label='Grad Descent')
    # plt.text(15, 10, 'Least Squares', fontsize=10) -- commented out to clean up graphic; rely on legend
    # plt.text(-16, 10, 'Grad Descent', fontsize=10) -- commented out to clean up graphic; rely on legend
    plt.xlabel('[km]')
    plt.ylabel('[km]')

    # Compute and Plot CRLB and Error Ellipse Expectations
    err_crlb = np.squeeze(triang.perf.compute_crlb(x_sensor, x_source, cov=covar_psi))
    crlb_cep50 = utils.errors.compute_cep50(err_crlb)  # [km]
    crlb_ellipse = utils.errors.draw_error_ellipse(x=x_source, covariance=err_crlb, num_pts=100, conf_interval=90)
    plt.plot(crlb_ellipse[0, :], crlb_ellipse[1, :], linewidth=.5, label='90% Error Ellipse')
    # plt.text(-20, 45, '90% Error Ellipse', fontsize=10) -- commented out to clean up graphic; rely on legend
    # plt.plot([1, 11], [45, 45], linestyle='-', linewidth=.5, label=None)

    plt.xlim([-50, 50])
    plt.ylim([-10, 70])
    plt.legend(loc='upper left')
    plt.show()

    # Compute Error Statistics
    err_ml = x_source[:, np.newaxis] - x_ml
    err_bf = x_source[:, np.newaxis] - x_bf
    err_cnt = x_source[:, np.newaxis] - x_centroid
    err_inc = x_source[:, np.newaxis] - x_incenter
    err_ls = x_source[:, np.newaxis, np.newaxis] - x_ls_full
    err_grad = x_source[:, np.newaxis, np.newaxis] - x_grad_full

    bias_ml = np.mean(err_ml, axis=1)
    bias_bf = np.mean(err_bf, axis=1)
    bias_cnt = np.mean(err_cnt, axis=1)
    bias_inc = np.mean(err_inc, axis=1)
    cov_ml = np.cov(err_ml) + bias_ml.dot(bias_ml.T)
    cov_bf = np.cov(err_bf) + bias_bf.dot(bias_bf.T)
    cov_cnt = np.cov(err_cnt) + bias_cnt.dot(bias_cnt.T)
    cov_inc = np.cov(err_inc) + bias_inc.dot(bias_inc.T)
    cep50_ml = utils.errors.compute_cep50(cov_ml)
    cep50_bf = utils.errors.compute_cep50(cov_bf)
    cep50_cnt = utils.errors.compute_cep50(cov_cnt)
    cep50_inc = utils.errors.compute_cep50(cov_inc)

    out_shp = (2, num_iterations)
    out_cov_shp = (2, 2, num_iterations)
    bias_ls = np.zeros(shape=out_shp)
    bias_grad = np.zeros(shape=out_shp)
    cov_ls = np.zeros(shape=out_cov_shp)
    cov_grad = np.zeros(shape=out_cov_shp)
    cep50_ls = np.zeros(shape=(num_iterations,))
    cep50_grad = np.zeros(shape=(num_iterations,))

    for ii in np.arange(num_iterations):
        bias_ls[:, ii] = np.mean(np.squeeze(err_ls[:, ii, :]), axis=1)
        bias_grad[:, ii] = np.mean(np.squeeze(err_grad[:, ii, :]), axis=1)

        cov_ls[:, :, ii] = np.cov(np.squeeze(err_ls[:, ii, :])) + bias_ls[:, ii].dot(bias_ls[:, ii].T)
        cov_grad[:, :, ii] = np.cov(np.squeeze(err_grad[:, ii, :])) + bias_grad[:, ii].dot(bias_grad[:, ii].T)

        cep50_ls[ii] = utils.errors.compute_cep50(cov_ls[:, :, ii])  # [km]
        cep50_grad[ii] = utils.errors.compute_cep50(cov_grad[:, :, ii])  # [km]

    # Second subfigure
    iter_ax = np.arange(num_iterations)
    ones_vec = np.ones(shape=(num_iterations, ))
    
    fig_err = plt.figure()
    plt.loglog(iter_ax, cep50_ls, linestyle=':', label='Least-Squares')
    # plt.text(1.2, 4, 'Least-Squares', fontsize=10)
    plt.plot(iter_ax, cep50_grad, '--', label='Gradient Descent')
    # plt.text(2.5, 15, 'Gradient Descent', fontsize=10)
    plt.plot(iter_ax, cep50_ml*ones_vec, label='Max Likelihood')
    plt.plot(iter_ax, cep50_bf*ones_vec, label='BestFix')
    plt.plot(iter_ax, cep50_cnt*ones_vec, label='Centroid')
    # plt.text(15, 2.6, 'Centroid', fontsize=10)
    plt.plot(iter_ax, cep50_inc*ones_vec, label='Incenter')
    # plt.text(3, 35, 'Incenter', fontsize=10)
    plt.plot(iter_ax, crlb_cep50*ones_vec, 'k-.', label='CRLB')
    # plt.text(1.2, 1.8, 'CRLB', fontsize=10)
    
    plt.xlabel('Iteration Number')
    plt.ylabel('$CEP_{50}$ [km]')
    plt.legend(loc='upper right')
    plt.xlim([1, 150])
    plt.ylim([1, 50])

    # Re-engage the warning for numpy underflow
    np.seterr(under='warn')

    return fig_geo, fig_err


def _mc_iteration(args):
    """
    Executes a single iteration of the Monte Carlo simulation in Example 10.1.

    :param args: Dictionary of arguments for monte carlo simulation in Example 10.1. Fields are:
                rng: random number generator
                psi_act: true angle of arrival (radians)
                covar_psi: measurement error covariance matrix
                covar_lower: lower triangular Cholesky decomposition of the measurement error covariance matrix
                num_sensors: number of AOA sensors
                x_sensor: position of AOA sensors
                x_init: initial solution guess (also used as center of search grid for ML and Bestfix)
                x_extent: search grid extent
                epsilon: search grid spacing (also used as stopping condition for LS and GD)
                num_iterations: maximum number of iterations for GD and LS solvers
    :return estimates: Dictionary with estimated target position using several algorithms.  Fields are:
                ml: Maximum Likelihood solution
                bf: Bestfix solution
                centroid: Geometric centroid solution
                incenter: Geometric incenter solution
                gd: Gradient Descent solution
                ls: Least Squares solution


    Nicholas O'Donoughue
    18 March 2022
    """

    # Generate a random measurement
    rng = args['rng']
    psi = args['psi_act'] + args['covar_lower'] @ rng.standard_normal(size=(args['num_sensors'], ))

    # Generate solutions
    res_ml, _, _ = triang.solvers.max_likelihood(x_sensor=args['x_sensor'], psi=psi, cov=args['covar_psi'],
                                                 x_ctr=args['x_init'], search_size=args['x_extent'],
                                                 epsilon=args['epsilon'])
    res_bf, _, _ = triang.solvers.bestfix(x_sensor=args['x_sensor'], psi=psi, cov=args['covar_psi'],
                                          x_ctr=args['x_init'], search_size=args['x_extent'], epsilon=args['epsilon'])
    res_centroid = triang.solvers.centroid(x_sensor=args['x_sensor'], psi=psi)
    res_incenter = triang.solvers.angle_bisector(x_sensor=args['x_sensor'], psi=psi)
    _, res_ls = triang.solvers.least_square(x_sensor=args['x_sensor'], psi=psi, cov=args['covar_psi'],
                                            x_init=args['x_init'], max_num_iterations=args['num_iterations'],
                                            force_full_calc=True)
    _, res_gd = triang.solvers.gradient_descent(x_sensor=args['x_sensor'], psi=psi, cov=args['covar_psi'],
                                                x_init=args['x_init'], max_num_iterations=args['num_iterations'],
                                                force_full_calc=True)

    return {'ml': res_ml, 'bf': res_bf, 'centroid': res_centroid, 'incenter': res_incenter, 'ls': res_ls, 'gd': res_gd}


def example2():
    """
    Executes Example 10.2 and generates one figure
    
    Ported from MATLAB Code

    Nicholas O'Donoughue
    18 May 2021

    :return: figure handle to generated graphic
    """

    # Occasionally, we divide by zero.  This represents invalid scenarios.  The result is
    # expected (returns a NaN), but let's hide the warning to be clean
    np.seterr(invalid='ignore')

    # Define sensor positions
    x_sensor = 10*np.array([[-1, 1], [0, 0]])
    num_dims, num_sensors = np.shape(x_sensor)

    # Define measurement accuracy
    sigma_psi = 2.5*np.pi/180
    covar_psi = sigma_psi**2 * np.eye(num_sensors)  # N x N identity matrix
    covar_inv = (1/sigma_psi**2) * np.eye(num_sensors)  # the inverse of the covariance matrix

    # Find maximum cross-range position at 100 km downrange
    cross_range_vec = np.arange(start=-100, stop=101)
    down_range_vec = 100 * np.ones(shape=np.shape(cross_range_vec))
    x_source = np.concatenate((cross_range_vec[np.newaxis, :], down_range_vec[np.newaxis, :]), axis=0)
    crlb = triang.perf.compute_crlb(x_sensor*1e3, x_source*1e3, cov=covar_inv, cov_is_inverted=True)
    cep50 = utils.errors.compute_cep50(crlb)
    
    good_points = np.argwhere(cep50 <= 25e3)
    max_cross_range = np.amax(np.abs(x_source[0, good_points]))*1e3
    print('Maximum cross range position at 100 km downrange that satisfies CEP < 25 km is {:.2f} km'
          .format(max_cross_range/1e3))

    x_max = 100
    x_vec = np.arange(start=-x_max, step=.25, stop=x_max)
    x_mesh, y_mesh = np.meshgrid(x_vec, x_vec)
    x0 = np.stack((x_mesh.flatten(), y_mesh.flatten()), axis=1).T

    # Compute CRLB
    crlb = triang.perf.compute_crlb(x_sensor*1e3, x0*1e3, covar_inv, cov_is_inverted=True)
    cep50 = np.reshape(utils.errors.compute_cep50(crlb), newshape=x_mesh.shape)

    # Blank out y=0
    nan_mask = np.abs(y_mesh) < 1e-6
    cep50[nan_mask] = np.nan
    
    # Plot
    fig = plt.figure()
    plt.scatter(x_sensor[0, :], x_sensor[1, :], marker='o', label='AOA Sensors')
    contour_levels = [1, 5, 10, 25, 50]
    contour_set = plt.contour(x_mesh, y_mesh, cep50/1e3, levels=contour_levels)
    plt.clabel(contour_set, contour_levels)
    plt.xlabel('x [km]')
    plt.ylabel('y [km]')
    plt.title('Triangulation CRLB RMSE [km]')
    plt.legend(loc='upper right')
    
    plt.ylim([-100, 100])

    # Reactive warnings
    np.seterr(invalid='warn')

    return fig


def example3():
    """
    Executes Example 10.3 and generates one figure.
    
    Ported from MATLAB Code
    
    Nicholas O'Donoughue
    18 May 2021
    
    :return: figure handle to generated graphic
    """
    
    # Define sensor positions
    x_sensor = 10*np.array([[-1, 1, 0], [0, 0, 1]])
    num_dims, num_sensors = np.shape(x_sensor)

    x_max = 100
    x_vec = np.arange(start=-x_max, step=.25, stop=x_max)
    x_mesh, y_mesh = np.meshgrid(x_vec, x_vec)
    x0 = np.stack((x_mesh.flatten(), y_mesh.flatten()), axis=1).T
    
    # Define measurement accuracy
    sigma_psi = 2.5*np.pi/180
    covar_psi = sigma_psi**2 * np.eye(num_sensors)  # N x N identity matrix
    
    # Compute CRLB
    crlb = triang.perf.compute_crlb(x_sensor*1e3, x0*1e3, covar_psi)
    cep50 = np.reshape(utils.errors.compute_cep50(crlb), newshape=np.shape(x_mesh))  # m
    
    good_point = cep50 <= 25e3
    rng_val = np.sqrt(np.sum(np.abs(x0)**2, axis=0))  # km
    max_range = np.max(rng_val[np.reshape(good_point, newshape=np.shape(rng_val))])  # km
    print('Maximum range that satisfies CEP < 25 km is {:.2f} km'.format(max_range))

    # Plot
    fig = plt.figure()
    plt.scatter(x_sensor[0, :], x_sensor[1, :], marker='o', label='AOA Sensors')
    contour_levels = [.1, .5, 1, 5, 10, 25, 50]
    contour_set = plt.contour(x_mesh, y_mesh, cep50/1e3, contour_levels)
    plt.clabel(contour_set, contour_levels)
    plt.xlabel('x [km]')
    plt.ylabel('y [km]')
    plt.title('Triangulation CRLB RMSE [km]')
    plt.legend(loc='upper right')

    return fig
