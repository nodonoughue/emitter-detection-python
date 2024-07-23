import numpy as np
import matplotlib.pyplot as plt
import time
import utils
import fdoa


def run_all_examples():
    """
    Run all chapter 12 examples and return a list of figure handles

    :return figs: list of figure handles
    """

    # Random Number Generator
    rng = np.random.default_rng(0)

    figs = example1(rng)

    return figs


def example1(rng=np.random.default_rng()):
    """
    Executes Example 12.1 and generates three figures

    Ported from MATLAB Code

    Added multiprocessing support to use all available CPU cores

    Nicholas O'Donoughue
    2 Nov 2022

    :param rng: random number generator
    :return fig_geo_a: figure handle for geographic layout
    :return fig_geo_b: figure handle for geographic layout -- zoomed in on target
    :return fig_err: figure handle for error as a function of iteration
    """

    # TODO: Debug solvers -- not working correctly (ML works, but not GD or LS; not sure about BF -- is it plotting?)

    #  Set up FDOA Receiver system
    #  Spacing of 10 km at 120 degree intervals around origin
    baseline = 10e3     # m
    std_velocity = 100  # m/s
    num_sensors = 3
    sensor_pos_angle = np.arange(num_sensors)*2*np.pi/num_sensors + np.pi/2
    x_sensor = baseline * np.array([np.cos(sensor_pos_angle), np.sin(sensor_pos_angle)])
    v_sensor = std_velocity * np.array([np.cos(sensor_pos_angle), np.sin(sensor_pos_angle)])

    # Add one at the origin
    x_sensor = np.concatenate((x_sensor, np.zeros(shape=(2, 1))), axis=1)
    v_sensor = np.concatenate((v_sensor, np.array([std_velocity, 0.])[:, np.newaxis]), axis=1)
    _, num_sensors = np.shape(x_sensor)

    # Define Sensor Performance
    freq_error = 3                  # Hz resolution
    transmit_freq = 1e9             # Hz
    fdoa_ref_idx = num_sensors - 1  # Use the last sensor as our reference sensor (the one at the origin)
    rng_rate_standard_deviation = freq_error*utils.constants.speed_of_light/transmit_freq
    covar_sensor = rng_rate_standard_deviation**2 * np.eye(num_sensors)
    covar_rho = rng_rate_standard_deviation**2 * (1 + np.eye(num_sensors-1))
    covar_lower = np.linalg.cholesky(covar_rho)

    # Initialize Transmitter Position
    th = rng.random()*2*np.pi
    x_source = 5*baseline*np.array([np.cos(th), np.sin(th)], )

    # Generate noise free measurement
    rho_actual = fdoa.model.measurement(x_sensor=x_sensor, x_source=x_source, v_sensor=v_sensor, ref_idx=fdoa_ref_idx)

    # Initialize Solvers
    num_mc_trials = int(1000)
    x_init = baseline * x_source / np.abs(x_source)
    x_extent = 5 * baseline
    num_iterations = int(400)
    alpha = .3
    beta = .8
    epsilon = 100  # [m] desired iterative search stopping condition
    grid_res = int(500)  # [m] desired grid search resolution

    out_shp = (2, num_mc_trials)
    out_iterative_shp = (2, num_iterations, num_mc_trials)
    x_ml = np.zeros(shape=out_shp)
    x_bf = np.zeros(shape=out_shp)
    x_ls_full = np.zeros(shape=out_iterative_shp)
    x_grad_full = np.zeros(shape=out_iterative_shp)

    args = {'rho_act': rho_actual,
            'num_measurements': num_sensors - 1,
            'x_sensor': x_sensor,
            'v_sensor': v_sensor,
            'x_init': x_init,
            'x_extent': x_extent,
            'covar_sensor': covar_sensor,
            'covar_rho': covar_rho,
            'covar_lower': covar_lower,
            'epsilon': epsilon,
            'grid_res': grid_res,
            'num_iterations': num_iterations,
            'rng': rng,
            'gd_alpha': alpha,
            'gd_beta': beta}

    print('Performing Monte Carlo simulation for FDOA performance...')
    t_start = time.perf_counter()

    iterations_per_marker = 1
    markers_per_row = 40
    iterations_per_row = markers_per_row * iterations_per_marker
    for idx in np.arange(num_mc_trials):
        if np.mod(idx + 1, iterations_per_marker) == 0:
            print('.', end='')  # Use end='' to prevent the newline

        if np.mod(idx + 1, iterations_per_row) == 0:
            print(' ({}/{}) '.format(idx + 1, num_mc_trials), end='')
            pct_elapsed = idx / num_mc_trials
            t_elapsed = time.perf_counter() - t_start
            utils.print_predicted(t_elapsed, pct_elapsed, do_elapsed=True)

        result = _mc_iteration(args)
        x_ml[:, idx] = result['ml']
        x_bf[:, idx] = result['bf']
        x_ls_full[:, :, idx] = result['ls']
        x_grad_full[:, :, idx] = result['gd']

    print('done')
    t_elapsed = time.perf_counter() - t_start
    utils.print_elapsed(t_elapsed)

    # TODO: Adjust plotting commands to align with chapter 12 output --- this is pasted from chapter 11
    fig_geo_a, ax = plt.subplots()
    sensor_handle = plt.scatter(x_sensor[0, :] / 1e3, x_sensor[1, :] / 1e3,
                                marker='o', label='Sensors')
    plt.scatter(x_source[0] / 1e3, x_source[1] / 1e3,
                marker='^', label='Transmitter')

    for this_x, this_v in zip(x_sensor.T, v_sensor.T):
        plt.arrow(x=this_x[0]/1e3, y=this_x[1]/1e3, dx=this_v[0]/100, dy=this_v[1]/100,
                  width=.01, head_width=.05,
                  color=sensor_handle.get_edgecolor())

    # Plot Closed-Form Solutions
    plt.scatter(x_ml[0, 0] / 1e3, x_ml[1, 0] / 1e3, marker='v', label='Maximum Likelihood')
    plt.scatter(x_bf[0, 0] / 1e3, x_bf[1, 0] / 1e3, marker='o', label='BestFix')

    # Plot Iterative Solutions
    plt.scatter(x_init[0]/1e3, x_init[1]/1e3, marker='x', label='Initial Estimate')
    plt.plot(x_ls_full[0, :, 0] / 1e3, x_ls_full[1, :, 0] / 1e3, linestyle=':', label='Least Squares')
    plt.plot(x_grad_full[0, :, 0] / 1e3, x_grad_full[1, :, 0] / 1e3, linestyle='--', label='Grad Descent')
    plt.xlabel('[km]')
    plt.ylabel('[km]')

    # Compute and Plot CRLB and Error Ellipse Expectations
    err_crlb = np.squeeze(fdoa.perf.compute_crlb(x_sensor=x_sensor, v_sensor=v_sensor,
                                                 x_source=x_source, cov=covar_rho, do_resample=False))
    crlb_cep50 = utils.errors.compute_cep50(err_crlb) / 1e3  # [km]
    crlb_ellipse = utils.errors.draw_error_ellipse(x=x_source, covariance=err_crlb, num_pts=100, conf_interval=90)
    plt.plot(crlb_ellipse[0, :]/1e3, crlb_ellipse[1, :]/1e3, linewidth=.5, label='90% Error Ellipse')

    plt.xlabel('Cross-range [km]')
    plt.ylabel('Down-range [km]')
    # wd = 7 * baseline / 1e3
    # ht = wd * 9 / 16
    # plt.xlim([-wd, wd])
    # plt.ylim(np.array([-1, 1]) * ht + x_source[1] / 1e3 / 2)
    plt.legend()
    plt.show()

    # Plot zoomed geometry
    num_iter_to_plot = 100

    fig_geo_b = plt.figure()
    plt.scatter(x_source[0] / 1e3,
                x_source[1] / 1e3, color='k', marker='^', label='Transmitter')
    plt.plot(x_ls_full[0, :num_iter_to_plot, 0] / 1e3,
             x_ls_full[1, :num_iter_to_plot, 0] / 1e3, '--x', label='LS')
    plt.plot(x_grad_full[0, :num_iter_to_plot, 0] / 1e3,
             x_grad_full[1, :num_iter_to_plot, 0] / 1e3, '-.+', label='Grad Descent')
    plt.scatter(x_bf[0, 0] / 1e3, x_bf[1, 0] / 1e3, marker='o', label='BestFix')
    plt.scatter(x_ml[0, 0] / 1e3, x_ml[1, 0] / 1e3, marker='v', label='Maximum Likelihood')

    plt.plot(crlb_ellipse[0, :] / 1e3, crlb_ellipse[1, :] / 1e3, color='k', linewidth=.5, label='90% Error Ellipse')

    ht = 1.2 * np.max([np.max(np.abs(crlb_ellipse[0, :] - x_source[0])),
                       np.max(np.abs(crlb_ellipse[1, :] - x_source[1]))])
    wd = ht * 1.2
    plt.ylim(np.array([-1, 1]) * ht / 1e3 + x_source[1] / 1e3)
    plt.xlim(np.array([-1, 1]) * wd / 1e3 + x_source[0] / 1e3)
    plt.xlabel('Cross-range [km]')
    plt.ylabel('Down-range [km]')
    plt.legend(loc='best')

    # Compute Error Statistics
    err_ml = x_source[:, np.newaxis] - x_ml
    err_bf = x_source[:, np.newaxis] - x_bf
    err_ls = x_source[:, np.newaxis, np.newaxis] - x_ls_full
    err_grad = x_source[:, np.newaxis, np.newaxis] - x_grad_full

    bias_ml = np.mean(err_ml, axis=1)
    bias_bf = np.mean(err_bf, axis=1)
    cov_ml = np.cov(err_ml) + bias_ml.dot(bias_ml.T)
    cov_bf = np.cov(err_bf) + bias_bf.dot(bias_bf.T)
    cep50_ml = utils.errors.compute_cep50(cov_ml) / 1e3
    cep50_bf = utils.errors.compute_cep50(cov_bf) / 1e3

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

        cep50_ls[ii] = utils.errors.compute_cep50(cov_ls[:, :, ii]) / 1e3  # [km]
        cep50_grad[ii] = utils.errors.compute_cep50(cov_grad[:, :, ii]) / 1e3  # [km]

    # Error plot
    iter_ax = np.arange(num_iterations)
    ones_vec = np.ones(shape=(num_iterations,))

    fig_err = plt.figure()
    plt.loglog(iter_ax, cep50_ls, linestyle=':', label='Least-Squares')
    plt.plot(iter_ax, cep50_grad, '--', label='Gradient Descent')
    plt.plot(iter_ax, cep50_ml * ones_vec, label='Max Likelihood')
    plt.plot(iter_ax, cep50_bf * ones_vec, linestyle='-.', label='BestFix')
    plt.plot(iter_ax, crlb_cep50 * ones_vec, 'k-.', label='CRLB')

    plt.xlabel('Iteration Number')
    plt.ylabel('$CEP_{50}$ [km]')
    plt.legend(loc='upper right')

    return fig_geo_a, fig_geo_b, fig_err


def _mc_iteration(args):
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
                ls: Least Square solution

    Nicholas O'Donoughue
    18 March 2022
    """
    # TODO: Profile MC iteration, and attempt to speed up

    # Generate a random measurement
    rng = args['rng']
    rho = args['rho_act'] + args['covar_lower'] @ rng.standard_normal(size=(args['num_measurements'], 1))

    # Generate solutions
    res_ml, ml_surf, ml_grid = fdoa.solvers.max_likelihood(x_sensor=args['x_sensor'], v_sensor=args['v_sensor'],
                                                           rho=rho, cov=args['covar_rho'],
                                                           x_ctr=args['x_init'], search_size=args['x_extent'],
                                                           epsilon=args['grid_res'], do_resample=False)
    res_bf, bf_surf, bf_grid = fdoa.solvers.bestfix(x_sensor=args['x_sensor'], v_sensor=args['v_sensor'],
                                                    rho=rho, cov=args['covar_rho'],
                                                    x_ctr=args['x_init'], search_size=args['x_extent'],
                                                    epsilon=args['grid_res'], do_resample=False)
    _, res_ls = fdoa.solvers.least_square(x_sensor=args['x_sensor'], v_sensor=args['v_sensor'],
                                          rho=rho, cov=args['covar_rho'],
                                          x_init=args['x_init'], max_num_iterations=args['num_iterations'],
                                          force_full_calc=True, do_resample=False)
    _, res_gd = fdoa.solvers.gradient_descent(x_sensor=args['x_sensor'], v_sensor=args['v_sensor'],
                                              rho=rho, cov=args['covar_rho'],
                                              x_init=args['x_init'], max_num_iterations=args['num_iterations'],
                                              alpha=args['gd_alpha'], beta=args['gd_beta'],
                                              force_full_calc=True, do_resample=False)

    return {'ml': res_ml, 'ls': res_ls, 'gd': res_gd, 'bf': res_bf}
