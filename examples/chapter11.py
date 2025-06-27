import numpy as np
import matplotlib.pyplot as plt
import time
import utils
from utils.covariance import CovarianceMatrix
import tdoa


def run_all_examples():
    """
    Run all chapter 11 examples and return a list of figure handles

    :return figs: list of figure handles
    """

    # Random Number Generator
    rng = np.random.default_rng(0)

    figs = example1(rng)

    return figs


def example1(rng=np.random.default_rng()):
    """
    Executes Example 11.1 and generates two figures

    Ported from MATLAB Code

    Added multiprocessing support to use all available CPU cores

    Nicholas O'Donoughue
    14 Mar 2022

    :param rng: random number generator
    :return fig_geo_a: figure handle for geographic layout
    :return fig_geo_b: figure handle for geographic layout -- zoomed in on target
    :return fig_err: figure handle for error as a function of iteration
    """

    #  Set up TDOA Receiver system
    #  Spacing of 1 km at 60 degree intervals around origin

    # Define Sensor Positions
    baseline = 10e3
    num_sensors = 3
    sensor_pos_angle = np.arange(num_sensors)*2*np.pi/num_sensors + np.pi/2
    x_sensor = baseline * np.array([np.cos(sensor_pos_angle), np.sin(sensor_pos_angle)])

    # Add one at the origin
    x_sensor = np.concatenate((x_sensor, np.zeros(shape=(2, 1))), axis=1)
    _, num_sensors = np.shape(x_sensor)

    # Define Sensor Performance
    time_measurement_standard_deviation = 1e-7
    rng_measurement_standard_deviation = time_measurement_standard_deviation * utils.constants.speed_of_light
    # covar_roa = CovarianceMatrix(rng_measurement_standard_deviation**2 * np.eye(num_sensors))
    covar_rho = CovarianceMatrix(rng_measurement_standard_deviation**2 * (1 + np.eye(num_sensors-1)))

    # Initialize Transmitter Position
    th = rng.random()*2*np.pi
    x_source = 5*baseline*np.array([np.cos(th), np.sin(th)],)

    # Compute Range
    rho_actual = tdoa.model.measurement(x_sensor, x_source)

    # Find Isochrones
    # Could be useful for overlaying isochrones onto plot
    # x_isochrone1 = tdoa.model.draw_isochrone(x_sensor[:, -1], x_sensor[:, 0], dR[0], 1000, 5*baseline)
    # x_isochrone2 = tdoa.model.draw_isochrone(x_sensor[:, -1], x_sensor[:, 1], dR[1], 1000, 5*baseline)
    # x_isochrone3 = tdoa.model.draw_isochrone(x_sensor[:, -1], x_sensor[:, 2], dR[2], 1000, 5*baseline)

    # Set up the Monte Carlo Trial
    num_mc_trials = int(1000)
    num_iterations = int(1000)
    alpha = .3
    beta = .8
    epsilon = 100  # [m] desired iterative search stopping condition
    grid_res = int(500)  # [m] desired grid search resolution
    x_init = 2 * baseline * np.array([np.cos(th + np.pi / 6), np.sin(th + np.pi / 6)])  # km
    x_extent = 5 * baseline

    out_shp = (2, num_mc_trials)
    out_iterative_shp = (2, num_iterations, num_mc_trials)
    x_ml = np.zeros(shape=out_shp)
    x_bf = np.zeros(shape=out_shp)
    x_chan_ho = np.zeros(shape=out_shp)
    x_ls_full = np.zeros(shape=out_iterative_shp)
    x_grad_full = np.zeros(shape=out_iterative_shp)

    print('Performing Monte Carlo simulation for TDOA performance...')
    t_start = time.perf_counter()

    rx_args = {'x_sensor': x_sensor,
               'cov': covar_rho,
               'do_resample': False
               }

    ml_args = {'x_ctr': x_init,
               'search_size': x_extent,
               'epsilon': grid_res
               }

    ls_args = {'x_init': x_init,
               'max_num_iterations': num_iterations,
               'epsilon': epsilon
               }

    gd_args = {'x_init': x_init,
               'max_num_iterations': num_iterations,
               'epsilon': epsilon,
               'alpha': alpha,
               'beta': beta
               }

    mc_args = {'rho_act': rho_actual,
               'num_measurements': num_sensors-1,
               'rng': rng
               }

    iterations_per_marker = 1
    markers_per_row = 40
    iterations_per_row = markers_per_row * iterations_per_marker
    for idx in np.arange(num_mc_trials):
        utils.print_progress(num_mc_trials, idx, iterations_per_marker, iterations_per_row, t_start)

        result = _mc_iteration(rx_args, ml_args, ls_args, gd_args, mc_args)
        x_ml[:, idx] = result['ml']
        x_bf[:, idx] = result['bf']
        x_ls_full[:, :, idx] = result['ls']
        x_grad_full[:, :, idx] = result['gd']
        x_chan_ho[:, idx] = result['chan_ho']

    print('done')
    t_elapsed = time.perf_counter() - t_start
    utils.print_elapsed(t_elapsed)

    fig_geo_a, ax = plt.subplots()
    plt.scatter(x_sensor[0, :]/1e3, x_sensor[1, :]/1e3, marker='o', label='Sensors', linewidth=1)
    plt.scatter(x_source[0]/1e3, x_source[1]/1e3, marker='^', label='Transmitter', linewidth=1)

    # Plot Closed-Form Solutions
    plt.scatter(x_chan_ho[0, 0]/1e3, x_chan_ho[1, 0]/1e3, marker='*', label='Chan-Ho')
    plt.scatter(x_ml[0, 0]/1e3, x_ml[1, 0]/1e3, marker='v', label='Maximum Likelihood')
    plt.scatter(x_bf[0, 0]/1e3, x_bf[1, 0]/1e3, marker='o', label='BestFix')

    # Plot Iterative Solutions
    plt.scatter(x_init[0], x_init[1], marker='x', label='Initial Estimate')
    plt.plot(x_ls_full[0, :, 0]/1e3, x_ls_full[1, :, 0]/1e3, linestyle=':', label='Least Squares')
    plt.plot(x_grad_full[0, :, 0]/1e3, x_grad_full[1, :, 0]/1e3, linestyle='--', label='Grad Descent')
    # plt.text(15, 10, 'Least Squares', fontsize=10)
    # plt.text(-16, 10, 'Grad Descent', fontsize=10)
    plt.xlabel('[km]')
    plt.ylabel('[km]')

    # Compute and Plot CRLB and Error Ellipse Expectations
    err_crlb = np.squeeze(tdoa.perf.compute_crlb(x_sensor, x_source, cov=covar_rho, variance_is_toa=False,
                                                 do_resample=False))
    crlb_cep50 = utils.errors.compute_cep50(err_crlb)/1e3  # [km]
    crlb_ellipse = utils.errors.draw_error_ellipse(x=x_source, covariance=err_crlb, num_pts=100, conf_interval=90)
    plt.plot(crlb_ellipse[0, :]/1e3, crlb_ellipse[1, :]/1e3, linewidth=.5, label='90% Error Ellipse')

    plt.xlabel('Cross-range [km]')
    plt.ylabel('Down-range [km]')
    wd = 7*baseline/1e3
    ht = wd*9/16
    plt.xlim([-wd, wd])
    plt.ylim(np.array([-1, 1])*ht+x_source[1]/1e3/2)
    plt.legend(loc='upper left')
    plt.show()

    # Plot zoomed geometry
    num_iter_to_plot = 1000

    fig_geo_b = plt.figure()
    plt.scatter(x_source[0] / 1e3,
                x_source[1] / 1e3, color='k', marker='^', label='Transmitter')
    plt.plot(x_ls_full[0, :num_iter_to_plot, 0] / 1e3,
             x_ls_full[1, :num_iter_to_plot, 0] / 1e3, ':', marker='x', label='Least Squares', markevery=[-1])
    plt.plot(x_grad_full[0, :num_iter_to_plot, 0] / 1e3,
             x_grad_full[1, :num_iter_to_plot, 0] / 1e3, '-.', marker='+', label='Gradient Descent', markevery=[-1])
    plt.scatter(x_chan_ho[0, 0] / 1e3, x_chan_ho[1, 0] / 1e3, marker='*', label='Chan-Ho')
    plt.scatter(x_bf[0, 0] / 1e3, x_bf[1, 0] / 1e3, marker='o', label='BestFix')
    plt.scatter(x_ml[0, 0] / 1e3, x_ml[1, 0] / 1e3, marker='v', label='Maximum Likelihood')

    plt.plot(crlb_ellipse[0, :] / 1e3, crlb_ellipse[1, :] / 1e3, color='k', linewidth=.5, label='90% Error Ellipse')

    wd = 1.4 * np.max(np.abs(crlb_ellipse[0, :] - x_source[0]))
    ht = wd * 7 / 8
    plt.ylim(np.array([-1, 1]) * ht / 1e3 + x_source[1] / 1e3)
    plt.xlim(np.array([-1, 1]) * wd / 1e3 + x_source[0] / 1e3)
    plt.xlabel('Cross-range [km]')
    plt.ylabel('Down-range [km]')
    plt.legend(loc='best')

    # Compute Error Statistics
    err_ml = x_source[:, np.newaxis] - x_ml
    err_bf = x_source[:, np.newaxis] - x_bf
    err_chan_ho = x_source[:, np.newaxis] - x_chan_ho
    err_ls = x_source[:, np.newaxis, np.newaxis] - x_ls_full
    err_grad = x_source[:, np.newaxis, np.newaxis] - x_grad_full

    bias_ml = np.mean(err_ml, axis=1)
    bias_bf = np.mean(err_bf, axis=1)
    bias_chan_ho = np.mean(err_chan_ho, axis=1)
    cov_ml = np.cov(err_ml) + bias_ml.dot(bias_ml.T)
    cov_bf = np.cov(err_bf) + bias_bf.dot(bias_bf.T)
    cov_chan_ho = np.cov(err_chan_ho) + bias_chan_ho.dot(bias_chan_ho.T)
    cep50_ml = utils.errors.compute_cep50(cov_ml)/1e3
    cep50_bf = utils.errors.compute_cep50(cov_bf)/1e3
    cep50_chan_ho = utils.errors.compute_cep50(cov_chan_ho)/1e3

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

        cep50_ls[ii] = utils.errors.compute_cep50(cov_ls[:, :, ii])/1e3  # [km]
        cep50_grad[ii] = utils.errors.compute_cep50(cov_grad[:, :, ii])/1e3  # [km]

    # Error plot
    iter_ax = np.arange(num_iterations)
    ones_vec = np.ones(shape=(num_iterations,))

    fig_err = plt.figure()
    plt.loglog(iter_ax, cep50_ls, linestyle=':', label='Least-Squares')
    # plt.text(1.2, 4, 'Least-Squares', fontsize=10)
    plt.plot(iter_ax, cep50_grad, '--', label='Gradient Descent')
    # plt.text(2.5, 15, 'Gradient Descent', fontsize=10)
    plt.plot(iter_ax, cep50_ml * ones_vec, label='Max Likelihood')
    plt.plot(iter_ax, cep50_bf * ones_vec, linestyle='-.', label='BestFix')
    plt.plot(iter_ax, cep50_chan_ho * ones_vec, label='Chan-Ho')
    # plt.text(15, 2.6, 'Chan-Ho', fontsize=10)
    plt.plot(iter_ax, crlb_cep50 * ones_vec, 'k-.', label='CRLB')
    # plt.text(1.2, 1.8, 'CRLB', fontsize=10)

    plt.xlabel('Iteration Number')
    plt.ylabel('$CEP_{50}$ [km]')
    plt.legend(loc='upper right')

    return fig_geo_a, fig_geo_b, fig_err


def _mc_iteration(rx_args: dict, ml_args: dict, ls_args: dict, gd_args: dict, mc_args: dict):
    """
    Executes a single iteration of the Monte Carlo simulation in Example 11.1.

    :param args: Dictionary of arguments for monte carlo simulation in Example 11.1. Fields are:
                rng: random number generator
                rho_act: true range difference of arrival (meters)
                covar_rho: measurement error covariance matrix
                covar_lower: lower triangular Cholesky decomposition of the measurement error covariance matrix
                num_measurements: number of TDOA sensor pair measurements
                x_sensor: position of TDOA sensors
                x_init: initial solution guess (also used as center of search grid for ML and Bestfix)
                x_extent: search grid extent
                epsilon: stopping condition for LS and GD
                grid_res: search grid spacing
                num_iterations: maximum number of iterations for GD and LS solvers
    :return estimates: Dictionary with estimated target position using several algorithms.  Fields are:
                ml: Maximum Likelihood solution
                bf: Bestfix solution
                gd: Gradient Descent solution
                ls: Least Squares solution

    Nicholas O'Donoughue
    18 March 2022
    """

    # Generate a random measurement
    rng = mc_args['rng']
    rho = mc_args['rho_act'] + rx_args['cov'].lower @ rng.standard_normal(size=(mc_args['num_measurements'], ))

    # Generate solutions
    res_ml, _, _ = tdoa.solvers.max_likelihood(**rx_args, **ml_args, zeta=rho)
    res_bf, _, _ = tdoa.solvers.bestfix(**rx_args, **ml_args, zeta=rho)
    _, res_ls = tdoa.solvers.least_square(**rx_args, **ls_args, zeta=rho)
    _, res_gd = tdoa.solvers.gradient_descent(**rx_args, **gd_args, zeta=rho)
    res_chan_ho = tdoa.solvers.chan_ho(x_sensor=rx_args['x_sensor'], zeta=rho, cov=rx_args['cov'])

    return {'ml': res_ml, 'ls': res_ls, 'gd': res_gd, 'bf': res_bf, 'chan_ho': res_chan_ho}
