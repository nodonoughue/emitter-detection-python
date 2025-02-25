import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy

import utils
import tdoa
import hybrid
import time


_rad2deg = 180.0/np.pi
_deg2rad = np.pi/180.0

def run_all_examples():
    """
    Run all chapter 3 examples and return a list of figure handles

    :return figs: list of figure handles
    """

    return [example1(), example2(), example3(), example4()]


def example1(colors=None):
    """
    Executes Example 3.1.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    17 January 2025

    :param colors: set of colors for plotting
    :return: figure handle to generated graphic
    """

    if colors is None:
        colors = plt.get_cmap("viridis")

    # Initialize Measurement Covariance Matrix
    cov = np.diag(np.array([1, 3, 2, 3, 5])**2)
    c_max = 5**2 + 3**2

    # Generate common reference sets
    cov_first = utils.resample_covariance_matrix(cov, test_idx_vec=0)
    cov_last = utils.resample_covariance_matrix(cov, test_idx_vec=4)

    fig_a = plt.figure()
    plt.imshow(cov_first, vmin=0, vmax=c_max, cmap=colors)
    plt.colorbar()
    plt.title('Ref Index = 0')

    fig_b = plt.figure()
    plt.imshow(cov_last, vmin=0, vmax=c_max, cmap=colors)
    plt.colorbar()
    plt.title('Ref Index = 4')

    # Generate a full measurement set example
    cov_full = utils.resample_covariance_matrix(cov, test_idx_vec='full')

    fig_c = plt.figure()
    plt.imshow(cov_full, vmin=0, vmax=c_max, cmap=colors)
    plt.colorbar()
    plt.title('Ref Index = ''full''')

    # Package figure handles
    return fig_a, fig_b, fig_c


def example2(colors=None):
    """
    Executes Example 3.2.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    17 January 2025

    :return: figure handle to generated graphic
    """

    if colors is None:
        colors = plt.get_cmap("viridis")

    # Initialize sensor coordinates
    num_sensors = 4
    baseline = 10e3
    angle_sensors = np.linspace(start=0, stop=2*np.pi, num=num_sensors) + np.pi/2
    xx_sensors = np.concatenate([[0], np.cos(angle_sensors[:-1])])
    yy_sensors = np.concatenate([[0], np.sin(angle_sensors[:-1])])
    x_sensors = np.array([xx_sensors, yy_sensors]) * baseline
    # 3D Version -- from video
    # zz_sensors = np.ones((num_sensors, ))
    # x_sensors = np.array([xx_sensors, yy_sensors, zz_sensors]) * baseline

    # Define Covariance Matrix
    err_time = 100e-9
    cov_full = err_time**2 * np.eye(num_sensors)

    # Plot geometry
    fig = plt.figure()
    plt.scatter(x_sensors[0, :], x_sensors[1, :], marker='s', label='Sensors', clip_on=False)
    plt.legend(loc='upper left')

    # Define Sensor Pairs, wrap figure handle in a list
    ref_set = [0, 1, 2, 3, 'full']
    figs = [fig]

    # Define search grid for CRLB (targets up to 200 km away from center of TDOA constellation)
    x_ctr = np.array([0., 0.])
    max_offset = 100e3
    search_size = max_offset * np.ones((2,))
    num_grid_points_per_axis = 101
    grid_res = 2*max_offset / num_grid_points_per_axis
    # 3D Version -- from video
    # alt = 5e3
    # x_ctr = np.array([0., 0., alt])                       # It's 3D coordinates, so we need a 3D center point
    # search_size = np.array([max_offset, max_offset, 0])   # don't search the z-dimension, though
    x_source, x_grid, grid_shape = utils.make_nd_grid(x_ctr, search_size, grid_res)
    extent = (x_ctr[0] - max_offset, x_ctr[0] + max_offset, x_ctr[1] - max_offset, x_ctr[1] + max_offset)

    # Use a squeeze operation to ensure that the individual dimension indices in x_grid are 2D
    x_grid = [np.squeeze(this_dim) for this_dim in x_grid]

    # Remove singleton-dimensions from grid_shape so that contourf gets a 2D input
    grid_shape_2d = [i for i in grid_shape if i > 1]

    # Pre-define contour-levels, for consistency
    levels = [.01, 1, 5, 10, 25, 50, 100, 200]

    for this_ref_set in ref_set:
        # Compute CRLB
        this_crlb = tdoa.perf.compute_crlb(x_sensor=x_sensors, x_source=x_source, cov=cov_full, variance_is_toa=True,
                                           ref_idx=this_ref_set)
        # Response should be N x N x 3, where grid_shape = N x N x 1

        # Compute CEP50
        this_cep = utils.errors.compute_cep50(this_crlb)
        # To compute the RMSE instead of CEP50 uncomment the following
        # this_cep = np.sqrt(np.trace(this_crlb, axis0=0, axis1=1))  # Compute the trace along the spatial axes

        # Plot this Result
        this_fig = _plot_contourf(x_grid, extent, grid_shape_2d, this_cep/1e3, x_sensors, None, levels, colors)
        figs.append(this_fig)

    # TODO: Repeat with higher error on one sensor to match what was done in the video?

    return figs


def example3(colors=None):
    """
    Executes Example 3.3.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    17 January 2025

    :param colors:
    :return: figure handle to generated graphic
    """

    if colors is None:
        colors = plt.get_cmap("viridis")

    # Initialize sensor coordinates
    x_sensors = np.array([[-17.5, -12.5, 12.5, 17.5],[0, 0, 0, 0]]) * 1e3  # m
    velocity = 100   # m/s
    heading_1 = 70 * _deg2rad   # CCW from +x; heading for sensors 0 and 1
    heading_2 = 110 * _deg2rad  # CCW from +x; heading for sensors 2 and 3
    v_sensors = np.kron(np.array([np.cos([heading_1, heading_2]),    # Use heading to define cartesian velocity.
                                 np.sin([heading_1, heading_2])]),
                        np.ones((1, 2))) * velocity                  # Use kronecker product to duplicate velocity for
                                                                    # each sensor in FDOA pairs (sensor pairs moving in
                                                                    # formation).
    # 3D Version -- from video; uncomment to use
    # alt = 10e3
    # zz_sensors = alt*np.ones((num_sensors, ))
    # x_sensors = np.concatenate((x_sensors, zz_sensors), axis=0)
    num_dims, num_sensors = utils.safe_2d_shape(x_sensors)

    # Define Covariance Matrix
    freq_hz = 1e9
    lam = utils.constants.speed_of_light / freq_hz
    time_err = 100e-9  # 100 ns
    freq_err = 100     # 10 Hz
    rng_err = utils.constants.speed_of_light * time_err  # meters (ROA)
    rng_rate_err = lam * freq_err                        # meters/second (RROA)
    cov_toa = rng_err ** 2 * np.eye(num_sensors)
    cov_foa = rng_rate_err **2 * np.eye(num_sensors)
    cov_full = scipy.linalg.block_diag(cov_toa, cov_foa)  # Measurement level error (ROA/RROA)

    # Plot geometry
    fig = plt.figure()
    hdl = plt.scatter(x_sensors[0, :], x_sensors[1, :], marker='o', label='Sensors', clip_on=False)
    for this_x, this_v in zip(x_sensors.T, v_sensors.T):  # transpose so the loop steps over sensors, not dimensions
        plt.arrow(x=this_x[0], y=this_x[1],
                  dx=this_v[0] * 10, dy=this_v[1] * 10,
                  width=.05e3, head_width=.2e3,
                  color=hdl.get_edgecolor(), label=None)
    plt.legend(loc='upper left')
    plt.ylim([-10e3, 10e3])

    # Define Sensor Pairs
    ref_set = (np.array([[0, 2],[1, 3]]), np.array([[0, 1, 2],[1, 2, 3]]), 'full')

    # Define search grid for CRLB (targets up to 200 km away from center of TDOA constellation)
    x_ctr = np.zeros((num_dims, ))
    max_offset = 50e3
    search_size = max_offset * np.ones((num_dims,))
    if num_dims > 2:
        search_size[2:] = 0  # only search the first two dimensions, even if a third is defined

    num_grid_points_per_axis = 301
    grid_res = 2 * max_offset / num_grid_points_per_axis

    x_source, x_grid, grid_shape = utils.make_nd_grid(x_ctr, search_size, grid_res)
    extent = (x_ctr[0] - max_offset, x_ctr[0] + max_offset, x_ctr[1] - max_offset, x_ctr[1] + max_offset)

    # Use a squeeze operation to ensure that the individual dimension
    # indices in x_grid are 2D
    x_grid = [np.squeeze(this_dim) for this_dim in x_grid]

    # Remove singleton-dimensions from grid_shape so that contourf gets a 2D input
    grid_shape_2d = [i for i in grid_shape if i > 1]

    # Loop over Sensor Pairs and compute CRLB
    figs = [fig]

    levels = [0.1, 0.5, 1, 5, 10, 25]

    for this_ref in ref_set:

        # Use the same sensors and reference indices for both TDOA and FDOA; set AOA to none
        this_crlb = hybrid.perf.compute_crlb(x_aoa=None, x_tdoa=x_sensors, x_fdoa=x_sensors, v_fdoa=v_sensors,
                                             x_source=x_source,
                                             cov=cov_full, do_resample=True, cov_is_inverted=False,
                                             tdoa_ref_idx=this_ref, fdoa_ref_idx=this_ref)

        this_rmse = np.sqrt(np.trace(this_crlb, axis1=0, axis2=1))

        this_fig = _plot_contourf(x_grid, extent, grid_shape_2d, this_rmse/1e3, x_sensors, v_sensors, levels, colors)
        figs.append(this_fig)

    return figs


def example4(rng=np.random.default_rng()):
    """
    Executes Example 3.4.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    28 January 2025

    :param rng: random number generator
    :return: figure handle to generated graphic
    """

    # Set up sensor and target coordinates
    x_source_ctr = np.array([3, 4]) * 1e3
    num_mc = 200
    offset = 1e2   # Maximum distance from center to a single instance of the source position (per dimension)
    x_source = x_source_ctr[:, np.newaxis] + offset * (-1 + 2 * rng.standard_normal(size=(2, num_mc)))

    x_tdoa = np.array([[1., 3., 4., 5., 2.],[0., .5, 0., .5, -1.]]) * 1e3
    _, num_tdoa = utils.safe_2d_shape(x_tdoa)

    # Initialize error covariance matrix
    time_err = 1e-7         # 100 ns time of arrival error per sensor
    cov_toa = (time_err**2) * np.eye(num_tdoa)
    cov_roa = cov_toa * (utils.constants.speed_of_light**2)

    # TDOA Measurement and Combined Covariance Matrix
    # TODO: Align test_idx and ref_idx among all usages
    z_common = tdoa.model.measurement(x_sensor=x_tdoa, x_source=x_source, ref_idx=None)  # num_tdoa x num_mc
    z_full = tdoa.model.measurement(x_sensor=x_tdoa, x_source=x_source, ref_idx='full')

    cov_z_common = utils.resample_covariance_matrix(cov_roa, test_idx_vec=None)  # RDOA
    cov_z_full = utils.resample_covariance_matrix(cov_roa, test_idx_vec='full')  # RDOA

    # Make sure the covariance matrices are invertible (positive definite)
    cov_z_common = utils.ensure_invertible(cov_z_common)
    cov_z_full = utils.ensure_invertible(cov_z_full)

    # Generate random noise
    noise_sensor = np.linalg.cholesky(cov_roa) @ rng.standard_normal(size=(num_tdoa, num_mc))
    noise_common = utils.resample_noise(noise_sensor, test_idx_vec=None)
    noise_full = utils.resample_noise(noise_sensor, test_idx_vec='full')

    # Noisy Measurements
    zeta_common = z_common + noise_common
    zeta_full = z_full + noise_full

    # ---- Set Up Solution Parameters ----
    # ML Search Parameters
    ml_args = dict(x_ctr=np.array([2.5, 2.5]) * 1e3, search_size=np.array([5, 5]) * 1e3,
                   epsilon=20)  # meters, grid resolution

    # GD and LS Search Parameters
    gd_ls_args = dict(x_init=np.array([1, 1]) * 1e3, epsilon=ml_args['epsilon'], max_num_iterations=200,
                      force_full_calc=True, plot_progress=False)

    # This section differs from the MATLAB code. Profiling shows a speedup if we
    # compute the pseudoinverse of the covariance matrix once, rather than pass in
    # the regular covariance matrix.
    # TODO: Think of a better way to pass in both the inverted and regular covariance matrices; perhaps a custom data structure.
    ml_args['cov_is_inverted'] = True
    gd_ls_args['cov_is_inverted'] = True
    cov_z_common_inv = np.real(scipy.linalg.pinvh(cov_z_common))
    cov_z_full_inv = np.real(scipy.linalg.pinvh(cov_z_full))

    rmse_ml = np.zeros((num_mc,))
    rmse_gd = np.zeros((num_mc, gd_ls_args['max_num_iterations']))
    rmse_ls = np.zeros((num_mc, gd_ls_args['max_num_iterations']))

    rmse_ml_full = np.zeros((num_mc,))
    rmse_gd_full = np.zeros((num_mc, gd_ls_args['max_num_iterations']))
    rmse_ls_full = np.zeros((num_mc, gd_ls_args['max_num_iterations']))

    print('Performing Monte Carlo simulation...')
    t_start = time.perf_counter()

    iterations_per_marker = 1
    markers_per_row = 40
    iterations_per_row = markers_per_row * iterations_per_marker
    res = {}  # Pre-define to avoid a warning later
    for idx in np.arange(num_mc):
        utils.print_progress(num_mc, idx, iterations_per_marker, iterations_per_row, t_start)

        this_source = x_source[:, idx]
        this_zeta_common = zeta_common[:, idx]
        this_zeta_full = zeta_full[:, idx]

        res = _mc_iteration(x_tdoa, this_zeta_common, this_zeta_full, cov_z_common_inv, cov_z_full_inv, ml_args, gd_ls_args)

        rmse_ml[idx] = np.linalg.norm(res['ml'] - this_source)
        rmse_gd[idx, :] = np.linalg.norm(res['gd'] - this_source[:, np.newaxis], axis=0)
        rmse_ls[idx, :] = np.linalg.norm(res['ls'] - this_source[:, np.newaxis], axis=0)

        rmse_ml_full[idx] = np.linalg.norm(res['ml_full'] - this_source)
        rmse_gd_full[idx, :] = np.linalg.norm(res['gd_full'] - this_source[:, np.newaxis], axis=0)
        rmse_ls_full[idx, :] = np.linalg.norm(res['ls_full'] - this_source[:, np.newaxis], axis=0)
    print('done')
    t_elapsed = time.perf_counter() - t_start
    utils.print_elapsed(t_elapsed)

    # Compute average error across Monte Carlo Iterations
    rmse_avg_ml = np.sqrt(np.sum(rmse_ml ** 2) / num_mc)
    rmse_avg_gd = np.sqrt(np.sum(rmse_gd ** 2, axis=0) / num_mc)
    rmse_avg_ls = np.sqrt(np.sum(rmse_ls ** 2, axis=0) / num_mc)
    rmse_avg_ml_full = np.sqrt(np.sum(rmse_ml_full ** 2) / num_mc)
    rmse_avg_gd_full = np.sqrt(np.sum(rmse_gd_full ** 2, axis=0) / num_mc)
    rmse_avg_ls_full = np.sqrt(np.sum(rmse_ls_full ** 2, axis=0) / num_mc)

    fig_err = plt.figure()
    x_arr = np.arange(gd_ls_args['max_num_iterations'])
    hdl_ml = plt.plot(x_arr, rmse_avg_ml * np.ones_like(x_arr), label='Maximum Likelihood')
    hdl_gd = plt.plot(x_arr, rmse_avg_gd, label='Gradient Descent')
    hdl_ls = plt.plot(x_arr, rmse_avg_ls, label='Least Square')
    plt.plot(x_arr, rmse_avg_ml_full * np.ones_like(x_arr), '--', label='Maximum Likelihood (full)',
             marker='o', markevery=10)
    plt.plot(x_arr, rmse_avg_gd_full, '-.', label='Gradient Descent (full)',
             marker='o', markevery=10)
    plt.plot(x_arr, rmse_avg_ls_full, '-.', label='Least Square (full)',
             marker='o', markevery=10)


    plt.xlabel('Number of Iterations')
    plt.ylabel('RMSE [m]')
    plt.yscale('log')
    plt.title('Monte Carlo Geolocation Results')
    plt.xlim([0, x_arr[-1]])
    plt.ylim([100, 4000])

    # ---- Estimate Error Bounds ----
    # CRLB
    crlb_common = np.squeeze(tdoa.perf.compute_crlb(x_source=x_source_ctr, x_sensor=x_tdoa, cov=cov_z_common,
                                                    ref_idx=None, do_resample=False, variance_is_toa=False))
    crlb_full = np.squeeze(tdoa.perf.compute_crlb(x_source=x_source_ctr, x_sensor=x_tdoa, cov=cov_z_full,
                                                  ref_idx='full', do_resample=False, variance_is_toa=False))
    print('CRLB (using common sensor:')
    print('{} m^2'.format(np.matrix(crlb_common)))
    print('CRLB (using full sensor:')
    print('{} m^2'.format(np.matrix(crlb_full)))

    rmse_crlb = np.sqrt(np.trace(crlb_common))
    rmse_crlb_full = np.sqrt(np.trace(crlb_full))

    plt.plot(x_arr, rmse_crlb * np.ones_like(x_arr), '--', color='k', label='CRLB')
    plt.plot(x_arr, rmse_crlb_full * np.ones_like(x_arr), '-.', color='k', label='CRLB (full)')
    plt.legend(loc='upper right')

    # CEP50
    cep50_crlb = utils.errors.compute_cep50(crlb_common)
    cep50_crlb_full = utils.errors.compute_cep50(crlb_full)
    print('CEP50: {:.2f} km ({:.2f} km using full set)'.format(cep50_crlb / 1e3, cep50_crlb_full / 1e3))

    # 90% Error Ellipse
    conf_interval = 90
    crlb_ellipse = utils.errors.draw_error_ellipse(x=x_source[:, -1], covariance=crlb_common, num_pts=101,
                                                   conf_interval=conf_interval)
    # crlb_ellipse_full = utils.errors.draw_error_ellipse(x=x_source[:, -1], covariance=crlb_full, num_pts=101,
    #                                                     conf_interval=conf_interval)

    # ---- Plot Results from the final Monte Carlo iteration----
    # x_init = gd_ls_args['x_init']
    x_ml = res['ml']
    x_ls = res['ls']
    x_gd = res['gd']
    x_ml_full = res['ml_full']
    x_ls_full = res['ls_full']
    x_gd_full = res['gd_full']

    fig_full = plt.figure()
    plt.scatter(x_source[0, -1], x_source[1, -1], marker='x', color='k', label='Target', clip_on=False, zorder=3)
    # plt.scatter(x_tdoa[0], x_tdoa[1], marker='s', color='k', label='Sensors', clip_on=False, zorder=3)

    # Plot Closed-Form Solution
    plt.scatter(x_ml[0], x_ml[1], marker='v', label='Maximum Likelihood', zorder=3)
    plt.scatter(x_ml_full[0], x_ml_full[1], marker='^', label='Maximum Likelihood (full)', zorder=3)

    # Plot Iterative Solutions
    # plt.scatter(x_init[0], x_init[1], marker='x', color='k', label='Initial Estimate')
    plt.plot(x_gd[0], x_gd[1], linestyle='-.', marker='+', markevery=[-1], label='Grad Descent')
    plt.plot(x_gd_full[0], x_gd_full[1], linestyle='-.', marker='+', markevery=[-1], label='Grad Descent (full)')
    plt.plot(x_ls[0], x_ls[1], linestyle='-.', marker='*', markevery=[-1], label='Least Squares')
    plt.plot(x_ls_full[0], x_ls_full[1], linestyle='-.', marker='*', markevery=[-1], label='Least Squares (full)')


    # Overlay Error Ellipse
    plt.plot(crlb_ellipse[0], crlb_ellipse[1], linestyle='--', color='k',
             label='{:d}% Error Ellipse'.format(conf_interval))
    plt.legend(loc='best')

    plt.xlim([0.5e3, 5.5e3])
    plt.ylim([3.2e3, 4.8e3])

    return fig_err, fig_full


def _mc_iteration(x_tdoa, zeta_common, zeta_full, cov_z_common, cov_z_full, ml_args, gd_ls_args):
    """
    Executes a single iteration of the Monte Carlo simulation in Example 3.4.

    :return estimates: Dictionary with estimated target position using several algorithms.  Fields are:
                ml:         Maximum Likelihood solution with default common sensor
                gd:         Gradient Descent solution with default common sensor
                ls:         Least Squares solution with default common sensor
                ml_full     Maximum Likelihood solution with 'full' sensor pairs
                gd_full     Gradient Descent solution with 'full' sensor pairs
                ls_full:    Least Squares solution with 'full' sensor pairs

    Nicholas O'Donoughue
    28 January 2025
    """

    # ---- Apply Various Solvers ----
    # ML Solution
    x_ml, _, _ = tdoa.solvers.max_likelihood(rho=zeta_common, cov=cov_z_common, x_sensor=x_tdoa, ref_idx=None,
                                             do_resample=False, **ml_args)

    # GD Solution
    _, x_gd = tdoa.solvers.gradient_descent(rho=zeta_common, cov=cov_z_common, x_sensor=x_tdoa, ref_idx=None,
                                            do_resample=False, **gd_ls_args)

    # LS Solution
    _, x_ls = tdoa.solvers.least_square(rho=zeta_common, cov=cov_z_common, x_sensor=x_tdoa, ref_idx=None,
                                        do_resample=False, **gd_ls_args)

    # ML Solution -- Full Sensor Pairs
    x_ml_full, _, _ = tdoa.solvers.max_likelihood(rho=zeta_full, cov=cov_z_full, x_sensor=x_tdoa, ref_idx='full',
                                                  do_resample=False, **ml_args)

    # GD Solution -- Full Sensor Pairs
    _, x_gd_full = tdoa.solvers.gradient_descent(rho=zeta_full, cov=cov_z_full, x_sensor=x_tdoa, ref_idx='full',
                                                 do_resample=False, **gd_ls_args)

    # LS Solution -- Full Sensor Pairs
    _, x_ls_full = tdoa.solvers.least_square(rho=zeta_full, cov=cov_z_full, x_sensor=x_tdoa, ref_idx='full',
                                             do_resample=False, ** gd_ls_args)

    return {'ml': x_ml, 'ls': x_ls, 'gd': x_gd, 'ml_full':x_ml_full, 'ls_full': x_ls_full, 'gd_full': x_gd_full}


def _plot_contourf(x_grid, extent, grid_shape_2d, z, x_sensors, v_sensors, levels, colors):
    this_fig = plt.figure()
    hdl = plt.imshow(np.reshape(z, grid_shape_2d), origin='lower', cmap=colors, extent=extent,
                     norm=matplotlib.colors.LogNorm(vmin=levels[0], vmax=levels[-1]))
    plt.colorbar(hdl, format='%d')

    # Unlike in MATLAB, contourf does not draw contour edges. Manually add contours
    hdl2 = plt.contour(x_grid[0], x_grid[1], np.reshape(z, grid_shape_2d), levels=levels,
                      origin='lower', colors='k')
    plt.clabel(hdl2, fontsize=10, colors='w')#, fmt=matplotlib.ticker.LogFormatterExponent)#, fmt=lambda x: f"{x:.1f} km")

    hdl3 = plt.scatter(x_sensors[0, :], x_sensors[1, :], color='w', facecolors='none', marker='o', label='Sensors')
    if v_sensors is not None:
        for this_x, this_v in zip(x_sensors.T, v_sensors.T):  # transpose so the loop steps over sensors, not dimensions
            plt.arrow(x=this_x[0], y=this_x[1],
                      dx=this_v[0]*10, dy=this_v[1]*10,
                      width=.05e3, head_width=.2e3,
                      color=hdl3.get_edgecolor(), label=None)
    plt.grid(False)

    return this_fig