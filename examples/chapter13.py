import numpy as np
import matplotlib.pyplot as plt
import time
import utils
from utils.covariance import CovarianceMatrix
import hybrid


def run_all_examples():
    """
    Run all chapter 13 examples and return a list of figure handles

    :return figs: list of figure handles
    """

    # Random Number Generator
    rng = np.random.default_rng(0)

    fig1a, fig1b = example1(rng)
    fig2a, fig2b = example2(rng)

    figs = [fig1a, fig1b, fig2a, fig2b]
    return figs


def example1(rng=np.random.default_rng()):
    """
    Executes Example 13.1 and generates two figures

    Ported from MATLAB Code

    Nicholas O'Donoughue
    17 Nov 2022

    :param rng: random number generator
    :return fig_geo: figure handle for geographic layout
    :return fig_err: figure handle for error as a function of iteration
    """

    #  Set up receiver system
    baseline = 10e3
    std_velocity = 100
    x_sensor = baseline*np.array([[0., 1., -1.], np.zeros((3,))])
    v_sensor = std_velocity*np.array([np.zeros((3,)), np.ones((3,))])
    _, num_sensors = np.shape(x_sensor)

    # Define Sensor Performance
    transmit_freq = 1e9             # Hz
    ang_err = 0.2                   # rad (az/el)
    time_err = 100e-9               # sec
    freq_err = 3                    # Hz resolution
    rng_err = utils.constants.speed_of_light * time_err  # m
    rng_rate_err = utils.constants.speed_of_light * freq_err / transmit_freq  # m/s

    # Measurement Error Covariance
    tdoa_ref_idx = num_sensors - 1
    fdoa_ref_idx = num_sensors - 1  # Use the last sensor as our reference sensor (the one at the origin)
    covar_ang = CovarianceMatrix(ang_err**2 * np.eye(num_sensors))
    covar_roa = CovarianceMatrix(rng_err**2 * np.eye(num_sensors))
    covar_rroa = CovarianceMatrix(rng_rate_err**2 * np.eye(num_sensors))
    covar_rdoa = covar_roa.resample(ref_idx=tdoa_ref_idx)
    covar_rrdoa = covar_rroa.resample(ref_idx=fdoa_ref_idx)

    covar_rho = CovarianceMatrix.block_diagonal(covar_ang, covar_rdoa, covar_rrdoa)

    # Initialize Transmitter Position
    x_source = np.ones((2,)) * 30e3
    x_init = np.array([0, 10e3])

    # True measurement vector
    rho_actual = hybrid.model.measurement(x_aoa=x_sensor, x_tdoa=x_sensor, 
                                          x_fdoa=x_sensor, v_fdoa=v_sensor, 
                                          x_source=x_source, 
                                          tdoa_ref_idx=tdoa_ref_idx, fdoa_ref_idx=fdoa_ref_idx)

    # Initialize Solvers
    num_mc_trials = int(1000)
    x_extent = 5 * baseline
    num_iterations = int(1000)
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

    rx_args = {'x_aoa': x_sensor,
               'x_tdoa': x_sensor,
               'tdoa_ref_idx': tdoa_ref_idx,
               'x_fdoa': x_sensor,
               'v_fdoa': v_sensor,
               'fdoa_ref_idx': fdoa_ref_idx,
               'cov': covar_rho,
               'do_resample': False
               }

    ml_args = {'x_ctr': x_init,
               'search_size': x_extent,
               'epsilon': grid_res
               }

    ls_args = {'x_init': x_init,
               'max_num_iterations': num_iterations,
               'epsilon': epsilon,
               'force_full_calc': True
               }

    gd_args = {'x_init': x_init,
               'max_num_iterations': num_iterations,
               'epsilon': epsilon,
               'force_full_calc': True,
               'alpha': alpha,
               'beta': beta
               }

    args = {'rho_act': rho_actual,
            'num_measurements': rho_actual.size,
            'rng': rng,
            'rx_args': rx_args,
            'ml_args': ml_args,
            'gd_args': gd_args,
            'ls_args': ls_args,
            }

    print('Performing Monte Carlo simulation for Hybrid AOA/TDOA/FDOA performance...')
    t_start = time.perf_counter()

    iterations_per_marker = 1
    markers_per_row = 40
    iterations_per_row = markers_per_row * iterations_per_marker
    for idx in np.arange(num_mc_trials):
        utils.print_progress(num_mc_trials, idx, iterations_per_marker, iterations_per_row, t_start)

        result = _mc_iteration(args)
        x_ml[:, idx] = result['ml']
        x_bf[:, idx] = result['bf']
        x_ls_full[:, :, idx] = result['ls']
        x_grad_full[:, :, idx] = result['gd']

    print('done')
    t_elapsed = time.perf_counter() - t_start
    utils.print_elapsed(t_elapsed)

    # Wrap the results in a dictionary for easier passing
    results = {'ml': x_ml,
               'bf': x_bf,
               'ls': x_ls_full,
               'grad': x_grad_full}

    # Call the common plot generator for both examples
    rx_args['x_init'] = x_init  # Add the initial position estimate, for plotting
    rx_args['x_source'] = x_source  # Add the true source position, for plotting
    rx_args['x_sensor'] = x_sensor  # Set the 'x_sensor' flag, to plot all three sensors as a single type
    rx_args['num_iterations'] = num_iterations
    fig_geo, fig_err = _plot_mc_iteration_result(rx_args, results)

    return fig_geo, fig_err


def example2(rng=np.random.default_rng()):
    """
    Executes Example 13.2 and generates two figures

    Ported from MATLAB Code

    Nicholas O'Donoughue
    19 Nov 2022

    :param rng: random number generator
    :return fig_geo: figure handle for geographic layout
    :return fig_err: figure handle for error as a function of iteration
    """

    #  Set up receiver system
    baseline = 10e3
    std_velocity = 100
    x_aoa = baseline * np.array([-1., 1.])
    x_time_freq = baseline * np.array([[-1., 1.], [0., 0.]])
    v_time_freq = std_velocity*np.array([np.zeros((2,)), np.ones((2,))])
    _, num_aoa = utils.safe_2d_shape(x_aoa)
    _, num_time_freq = utils.safe_2d_shape(x_time_freq)

    # Define Sensor Performance
    transmit_freq = 1e9             # Hz
    ang_err = 0.6                   # rad (az/el)
    time_err = 100e-9               # sec
    freq_err = 3                    # Hz resolution
    rng_err = utils.constants.speed_of_light * time_err  # m
    rng_rate_err = utils.constants.speed_of_light * freq_err / transmit_freq  # m/s

    # Measurement Error Covariance
    tdoa_ref_idx = num_time_freq - 1
    fdoa_ref_idx = num_time_freq - 1
    covar_ang = CovarianceMatrix(ang_err**2 * np.eye(num_aoa))
    covar_roa = CovarianceMatrix(rng_err**2 * np.eye(num_time_freq))
    covar_rroa = CovarianceMatrix(rng_rate_err**2 * np.eye(num_time_freq))
    covar_rdoa = covar_roa.resample(ref_idx=tdoa_ref_idx)
    covar_rrdoa = covar_rroa.resample(ref_idx=fdoa_ref_idx)

    covar_rho = CovarianceMatrix.block_diagonal(covar_ang, covar_rdoa, covar_rrdoa)

    # Initialize Transmitter Position
    x_source = np.ones((2,)) * 30e3
    x_init = np.array([0, 10e3])

    # True measurement vector
    rho_actual = hybrid.model.measurement(x_aoa=x_aoa, x_tdoa=x_time_freq,
                                          x_fdoa=x_time_freq, v_fdoa=v_time_freq,
                                          x_source=x_source,
                                          tdoa_ref_idx=tdoa_ref_idx, fdoa_ref_idx=fdoa_ref_idx)

    # Initialize Solvers
    num_mc_trials = int(1000)
    x_extent = 5 * baseline
    num_iterations = int(1000)
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

    rx_args = {'x_aoa': x_aoa,
               'x_tdoa': x_time_freq,
               'tdoa_ref_idx': tdoa_ref_idx,
               'x_fdoa': x_time_freq,
               'v_fdoa': v_time_freq,
               'fdoa_ref_idx': fdoa_ref_idx,
               'cov': covar_rho,
               'do_resample': False
               }

    ml_args = {'x_ctr': x_init,
               'search_size': x_extent,
               'epsilon': grid_res
               }

    ls_args = {'x_init': x_init,
               'max_num_iterations': num_iterations,
               'epsilon': epsilon,
               'force_full_calc': True
               }

    gd_args = {'x_init': x_init,
               'max_num_iterations': num_iterations,
               'epsilon': epsilon,
               'force_full_calc': True,
               'alpha': alpha,
               'beta': beta
               }

    args = {'rx_args': rx_args,  # arguments to pass on to solvers that represent the receiver system
            'ml_args': ml_args,  # arguments to pass on to ML solver
            'ls_args': ls_args,  # arguments to pass on to LS solver
            'gd_args': gd_args,  # arguments to pass on to GD solver
            'rho_act': rho_actual,
            'num_measurements': num_aoa + 2 * (num_time_freq - 1),
            'rng': rng
            }

    print('Performing Monte Carlo simulation for FDOA performance...')
    t_start = time.perf_counter()

    iterations_per_marker = 1
    markers_per_row = 40
    iterations_per_row = markers_per_row * iterations_per_marker
    for idx in np.arange(num_mc_trials):
        utils.print_progress(num_mc_trials, idx, iterations_per_marker, iterations_per_row, t_start)

        result = _mc_iteration(args)
        x_ml[:, idx] = result['ml']
        x_bf[:, idx] = result['bf']
        x_ls_full[:, :, idx] = result['ls']
        x_grad_full[:, :, idx] = result['gd']

    print('done')
    t_elapsed = time.perf_counter() - t_start
    utils.print_elapsed(t_elapsed)

    # Wrap the results in a dictionary for easier passing
    results = {'ml': x_ml,
               'bf': x_bf,
               'ls': x_ls_full,
               'grad': x_grad_full}

    # Call the common plot generator for both examples
    rx_args['x_source'] = x_source  # Add the true source position, for plotting
    rx_args['x_sensor'] = None  # Set the 'x_sensor' flag to none, so that AOA and T/FDOA sensors are plotted separately
    rx_args['x_init'] = x_init  # Add the initial position estimate, for plotting
    rx_args['num_iterations'] = num_iterations
    fig_geo, fig_err = _plot_mc_iteration_result(rx_args, results)

    return fig_geo, fig_err


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
                ls: Least Squares solution

    Nicholas O'Donoughue
    18 March 2022
    """

    # TODO: Check for divergent case and set a flag to ignore this iteration

    # Generate a random measurement
    rng = args['rng']
    zeta_act = args['rho_act']
    cov_lower = args['rx_args']['cov'].lower
    zeta = zeta_act + cov_lower @ rng.standard_normal(size=(args['num_measurements'], ))

    # Generate solutions
    res_ml, ml_surf, ml_grid = hybrid.solvers.max_likelihood(**args['rx_args'], **args['ml_args'], zeta=zeta)
    res_bf, bf_surf, bf_grid = hybrid.solvers.bestfix(**args['rx_args'], **args['ml_args'], zeta=zeta)
    _, res_ls = hybrid.solvers.least_square(**args['rx_args'], **args['ls_args'], zeta=zeta)
    _, res_gd = hybrid.solvers.gradient_descent(**args['rx_args'], **args['gd_args'], zeta=zeta)

    return {'ml': res_ml, 'ls': res_ls, 'gd': res_gd, 'bf': res_bf}


def _plot_mc_iteration_result(args, results):

    # Generate plot of geographic laydown
    fig_geo, ax = plt.subplots()
    if args['x_sensor'] is not None:
        # Plot all sensors together
        time_freq_handle = plt.scatter(args['x_sensor'][0] / 1e3, args['x_sensor'][1] / 1e3,
                                       marker='o', label='Sensor')
    else:
        # Plot AOA and TDOA/FDOA separately
        plt.scatter(args['x_aoa'][0] / 1e3, args['x_aoa'][1] / 1e3, marker='^', label='AOA Sensor')
        time_freq_handle = plt.scatter(args['x_tdoa'][0] / 1e3, args['x_tdoa'][1] / 1e3,
                                       marker='o', label='Time/Freq Sensor')

    # Add velocity arrows to FDOA sensors
    for this_x, this_v in zip(args['x_fdoa'].T, args['v_fdoa'].T):
        plt.arrow(x=this_x[0] / 1e3, y=this_x[1] / 1e3, dx=this_v[0] / 100, dy=this_v[1] / 100,
                  width=.1, head_width=.5, color=time_freq_handle.get_edgecolor())

    plt.scatter(args['x_source'][0] / 1e3, args['x_source'][1] / 1e3, marker='^', label='Transmitter')

    # Plot Closed-Form Solutions
    plt.scatter(results['ml'][0, 0] / 1e3, results['ml'][1, 0] / 1e3, marker='v', label='Maximum Likelihood')
    plt.scatter(results['bf'][0, 0] / 1e3, results['bf'][1, 0] / 1e3, marker='o', label='BestFix')

    # Plot Iterative Solutions
    plt.scatter(args['x_init'][0] / 1e3, args['x_init'][1] / 1e3, marker='x', label='Initial Estimate')
    plt.plot(results['ls'][0, :, 0] / 1e3, results['ls'][1, :, 0] / 1e3,
             linestyle=':', label='Least Squares')
    plt.plot(results['grad'][0, :, 0] / 1e3, results['grad'][1, :, 0] / 1e3,
             linestyle='--', label='Grad Descent')
    plt.xlabel('[km]')
    plt.ylabel('[km]')

    # Compute and Plot CRLB and Error Ellipse Expectations
    err_crlb = np.squeeze(hybrid.perf.compute_crlb(x_aoa=args['x_aoa'], x_tdoa=args['x_tdoa'],
                                                   x_fdoa=args['x_fdoa'], v_fdoa=args['v_fdoa'],
                                                   cov=args['cov'], x_source=args['x_source'],
                                                   tdoa_ref_idx=args['tdoa_ref_idx'],
                                                   fdoa_ref_idx=args['fdoa_ref_idx'],
                                                   do_resample=args['do_resample']))
    crlb_cep50 = utils.errors.compute_cep50(err_crlb)/1e3  # [m]
    crlb_ellipse = utils.errors.draw_error_ellipse(x=args['x_source'], covariance=err_crlb,
                                                   num_pts=100, conf_interval=90)
    plt.plot(crlb_ellipse[0, :] / 1e3, crlb_ellipse[1, :] / 1e3, linewidth=.5, label='90% Error Ellipse')

    plt.xlabel('Cross-range [km]')
    plt.ylabel('Down-range [km]')
    plt.legend()
    plt.show()

    # Compute Error Statistics
    err_ml = args['x_source'][:, np.newaxis] - results['ml']
    err_bf = args['x_source'][:, np.newaxis] - results['bf']
    err_ls = args['x_source'][:, np.newaxis, np.newaxis] - results['ls']
    err_grad = args['x_source'][:, np.newaxis, np.newaxis] - results['grad']

    bias_ml = np.mean(err_ml, axis=1)
    bias_bf = np.mean(err_bf, axis=1)
    cov_ml = np.cov(err_ml) + bias_ml.dot(bias_ml.T)
    cov_bf = np.cov(err_bf) + bias_bf.dot(bias_bf.T)
    cep50_ml = utils.errors.compute_cep50(cov_ml) / 1e3
    cep50_bf = utils.errors.compute_cep50(cov_bf) / 1e3

    out_shp = (2, args['num_iterations'])
    out_cov_shp = (2, 2, args['num_iterations'])
    bias_ls = np.zeros(shape=out_shp)
    bias_grad = np.zeros(shape=out_shp)
    cov_ls = np.zeros(shape=out_cov_shp)
    cov_grad = np.zeros(shape=out_cov_shp)
    cep50_ls = np.zeros(shape=(args['num_iterations'],))
    cep50_grad = np.zeros(shape=(args['num_iterations'],))

    for ii in np.arange(args['num_iterations']):
        bias_ls[:, ii] = np.mean(np.squeeze(err_ls[:, ii, :]), axis=1)
        bias_grad[:, ii] = np.mean(np.squeeze(err_grad[:, ii, :]), axis=1)

        cov_ls[:, :, ii] = np.cov(np.squeeze(err_ls[:, ii, :])) + bias_ls[:, ii].dot(bias_ls[:, ii].T)
        cov_grad[:, :, ii] = np.cov(np.squeeze(err_grad[:, ii, :])) + bias_grad[:, ii].dot(bias_grad[:, ii].T)

        cep50_ls[ii] = utils.errors.compute_cep50(cov_ls[:, :, ii]) / 1e3  # [km]
        cep50_grad[ii] = utils.errors.compute_cep50(cov_grad[:, :, ii]) / 1e3  # [km]

    # Error plot
    iter_ax = np.arange(args['num_iterations'])
    ones_vec = np.ones(shape=(args['num_iterations'],))

    fig_err = plt.figure()
    plt.loglog(iter_ax, cep50_ls, linestyle=':', label='Least-Squares')
    plt.plot(iter_ax, cep50_grad, '--', label='Gradient Descent')
    plt.plot(iter_ax, cep50_ml * ones_vec, label='Max Likelihood')
    plt.plot(iter_ax, cep50_bf * ones_vec, linestyle='-.', label='BestFix')
    plt.plot(iter_ax, crlb_cep50 * ones_vec, 'k-.', label='CRLB')

    plt.xlabel('Iteration Number')
    plt.ylabel('$CEP_{50}$ [km]')
    plt.legend(loc='upper right')

    return fig_geo, fig_err
