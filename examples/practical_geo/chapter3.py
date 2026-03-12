import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time

from ewgeo.fdoa import FDOAPassiveSurveillanceSystem
from ewgeo.hybrid import HybridPassiveSurveillanceSystem
from ewgeo.tdoa import TDOAPassiveSurveillanceSystem
from ewgeo.utils.constants import speed_of_light
from ewgeo.utils.covariance import CovarianceMatrix
from ewgeo.utils.errors import compute_cep50, draw_error_ellipse, compute_rmse
from ewgeo.utils import print_elapsed, print_progress, resample_noise, SearchSpace

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
    cov = CovarianceMatrix(np.diag(np.array([1, 3, 2, 3, 5])**2))
    c_max = 5**2 + 3**2

    # Generate common reference sets
    cov_first = cov.resample(ref_idx=0)
    cov_last = cov.resample(ref_idx=4)

    fig_a = plt.figure()
    plt.imshow(cov_first.cov, vmin=0, vmax=c_max, cmap=colors)
    plt.colorbar()
    plt.title('Ref Index = 0')
    plt.xticks(range(cov_first.size))
    plt.yticks(range(cov_first.size))
    plt.xlabel('Sensor Pair Indices')
    plt.ylabel('Sensor Pair Indices')

    fig_b = plt.figure()
    plt.imshow(cov_last.cov, vmin=0, vmax=c_max, cmap=colors)
    plt.colorbar()
    plt.title('Ref Index = 4')
    plt.xticks(range(cov_last.size))
    plt.yticks(range(cov_last.size))
    plt.xlabel('Sensor Pair Indices')
    plt.ylabel('Sensor Pair Indices')

    # Generate a full measurement set example
    cov_full = cov.resample(ref_idx='full')

    fig_c = plt.figure()
    plt.imshow(cov_full.cov, vmin=0, vmax=c_max, cmap=colors)
    plt.colorbar()
    plt.title('Ref Index = ''full''')
    plt.xticks(range(cov_full.size))
    plt.yticks(range(cov_full.size))
    plt.xlabel('Sensor Pair Indices')
    plt.ylabel('Sensor Pair Indices')

    # Package figure handles
    return fig_a, fig_b, fig_c


def example2(colors=None, do_video_example=False):
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
    cov_full = CovarianceMatrix(err_time**2 * np.eye(num_sensors))

    # Make the PSS Object
    tdoa = TDOAPassiveSurveillanceSystem(x=x_sensors, ref_idx=None, cov=cov_full, variance_is_toa=True)

    # Plot geometry
    fig = plt.figure()
    plt.scatter(x_sensors[0]/1e3, x_sensors[1]/1e3, marker='s', label='Sensors', clip_on=False)
    plt.legend(loc='upper left')
    plt.xlabel('x [km]')
    plt.ylabel('y [km]')

    # Define Sensor Pairs, wrap figure handle in a list
    ref_set = [0, 1, 2, 3, 'full']
    figs = [fig]

    # Define search grid for CRLB (targets up to 200 km away from center of TDOA constellation)
    x_ctr = np.array([0., 0.])
    max_offset = 100e3
    search_size = max_offset * np.ones((2,))
    num_grid_points_per_axis = 201
    grid_res = 2*max_offset / num_grid_points_per_axis
    # 3D Version -- from video
    # alt = 5e3
    # x_ctr = np.array([0., 0., alt])                       # It's 3D coordinates, so we need a 3D center point
    # search_size = np.array([max_offset, max_offset, 0])   # don't search the z-dimension, though
    search_space = SearchSpace(x_ctr=x_ctr,
                               max_offset=search_size,
                               epsilon=grid_res)
    x_source, x_grid = search_space.x_set, search_space.x_grid

    extent = search_space.get_extent(multiplier=1/1e3)  # scale extent to km

    # Use a squeeze operation to ensure that the individual dimension indices in x_grid are 2D
    x_grid = [np.squeeze(this_dim)/1e3 for this_dim in x_grid] # convert to km

    # Pre-define contour-levels, for consistency
    levels = [.01, 1, 5, 10, 25, 50, 100, 200]

    for this_ref_set in ref_set:
        # Update the PSS system's reference index
        tdoa.ref_idx = this_ref_set

        # Compute CRLB
        this_crlb = tdoa.compute_crlb(x_source=x_source, print_progress=True)
        # Response should be N x N x 3, where grid_shape = N x N x 1

        # Compute CEP50
        this_cep = compute_cep50(this_crlb)
        # To compute the RMSE instead of CEP50 uncomment the following
        # this_cep = compute_rmse(this_crlb)  # Compute the trace along the spatial axes

        # Plot this Result
        this_fig = _plot_contourf(x_grid, extent, search_space.grid_shape, this_cep/1e3, x_sensors/1e3, None, levels, colors, units='km')
        figs.append(this_fig)

    if do_video_example:
        # For the video about Example 3.2, we modified the code to re-run with a larger covariance error on one sensor
        c = cov_full.cov                # Extract the covariance matrix
        c[0, 0] = 10 * c[0, 0]          # Multiply the first sensor's error by 10
        cov_full = CovarianceMatrix(c)  # Make a new covariance matrix object

        # Make a new TDOA object
        tdoa = TDOAPassiveSurveillanceSystem(x=x_sensors, cov=cov_full, ref_idx=None, variance_is_toa=True)
        ref_set = [0, 'full']

        for this_ref_set in ref_set:
            tdoa.ref_idx = this_ref_set

            # Compute CRLB
            this_crlb = tdoa.compute_crlb(x_source=x_source, print_progress=True)

            # Compute CEP50
            this_cep = compute_cep50(this_crlb)

            # Plot this Result
            this_fig = _plot_contourf(x_grid, extent, search_space.grid_shape, this_cep/1e3, x_sensors/1e3, None, levels, colors, units='km')
            figs.append(this_fig)

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
    x_sensors = np.array([[-17.5, -12.5, 12.5, 17.5], [0, 0, 0, 0]]) * 1e3  # m
    velocity = 100   # m/s
    heading_1 = 70 * _deg2rad   # CCW from +x; heading for sensors 0 and 1
    heading_2 = 110 * _deg2rad  # CCW from +x; heading for sensors 2 and 3
    v_sensors = np.kron(np.array([np.cos([heading_1, heading_2]),    # Use heading to define cartesian velocity.
                                 np.sin([heading_1, heading_2])]),
                        np.ones((1, 2))) * velocity
    # Use kronecker product to duplicate velocity for each sensor in FDOA pairs (sensor pairs moving in formation).

    # 3D Version -- from video; uncomment to use
    # alt = 10e3
    # zz_sensors = alt*np.ones((num_sensors, ))
    # x_sensors = np.concatenate((x_sensors, zz_sensors), axis=0)
    num_dims, num_sensors = np.shape(x_sensors)

    # Define Covariance Matrix
    freq_hz = 1e9
    lam = speed_of_light / freq_hz
    time_err = 100e-9  # 100 ns
    freq_err = 100     # 10 Hz
    rng_err = speed_of_light * time_err  # meters (ROA)
    rng_rate_err = lam * freq_err                        # meters/second (RROA)
    cov_roa = CovarianceMatrix(rng_err ** 2 * np.eye(num_sensors))
    cov_rroa = CovarianceMatrix(rng_rate_err ** 2 * np.eye(num_sensors))
    # cov_full = CovarianceMatrix.block_diagonal(cov_roa, cov_rroa)  # Measurement level error (ROA/RROA)

    # Make the PSS Objects
    aoa = None
    tdoa = TDOAPassiveSurveillanceSystem(x=x_sensors, cov=cov_roa, variance_is_toa=False, ref_idx=None)
    fdoa = FDOAPassiveSurveillanceSystem(x=x_sensors, vel=v_sensors, cov=cov_rroa, ref_idx=None)
    hybrid = HybridPassiveSurveillanceSystem(aoa=aoa, tdoa=tdoa, fdoa=fdoa)

    # Plot geometry
    fig = plt.figure()
    hdl = plt.scatter(x_sensors[0]/1e3, x_sensors[1]/1e3, marker='o', label='Sensors', clip_on=False, zorder=3)
    for this_x, this_v in zip(x_sensors.T, v_sensors.T):  # transpose so the loop steps over sensors, not dimensions
        plt.arrow(x=this_x[0]/1e3, y=this_x[1]/1e3,
                  dx=this_v[0]/100, dy=this_v[1]/100,
                  width=.05, head_width=.2,
                  color=hdl.get_edgecolor(), label=None)
    plt.legend(loc='upper left')
    plt.ylim([-10, 10])
    plt.xlabel('x [km]')
    plt.ylabel('y [km]')

    # Define Sensor Pairs
    ref_set = (np.array([[0, 2], [1, 3]]), np.array([[0, 1, 2], [1, 2, 3]]), 'full')

    # Define search grid for CRLB (targets up to 200 km away from center of TDOA constellation)
    x_ctr = np.zeros((num_dims, ))
    max_offset = 50e3
    search_size = max_offset * np.ones((num_dims,))
    if num_dims > 2:
        search_size[2:] = 0  # only search the first two dimensions, even if a third is defined

    num_grid_points_per_axis = 301
    grid_res = 2 * max_offset / num_grid_points_per_axis
    search_space = SearchSpace(x_ctr=x_ctr,
                               max_offset=search_size,
                               epsilon=grid_res)
    x_source, x_grid = search_space.x_set, search_space.x_grid
    extent = search_space.get_extent(multiplier=1/1e3) # convert to km for plotting
    # extent = (x_ctr[0].item() - max_offset,
    #           x_ctr[0].item() + max_offset,
    #           x_ctr[1].item() - max_offset,
    #           x_ctr[1].item() + max_offset)

    # Use a squeeze operation to ensure that the individual dimension
    # indices in x_grid are 2D
    x_grid = [np.squeeze(this_dim)/1e3 for this_dim in x_grid]

    # Loop over Sensor Pairs and compute CRLB
    figs = [fig]

    levels = [0.1, 0.5, 1, 5, 10, 25]

    for this_ref in ref_set:
        tdoa.ref_idx = this_ref
        fdoa.ref_idx = this_ref
        # We updated the reference indices; we need to alert the Hybrid system to update its version
        hybrid.update_reference_indices()

        # Use the same sensors and reference indices for both TDOA and FDOA; set AOA to none
        this_crlb = hybrid.compute_crlb(x_source=x_source, print_progress=True)

        this_rmse = compute_rmse(this_crlb)

        this_fig = _plot_contourf(x_grid, extent, search_space.grid_shape, this_rmse/1e3, x_sensors/1e3, v_sensors/100, levels, colors, units='km')
        figs.append(this_fig)

    return figs


def example4(rng=np.random.default_rng(), mc_params=None):
    """
    Executes Example 3.4.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    28 January 2025

    :param rng: random number generator
    :param mc_params: Optional struct to control Monte Carlo trial size
    :return: figure handle to generated graphic
    """

    # Set up sensor and target coordinates
    x_source_ctr = np.array([3, 4]) * 1e3
    num_monte_carlo = 100 # todo: restore to 200? 1000?
    if mc_params is not None:
        num_monte_carlo = max(int(num_monte_carlo/mc_params['monte_carlo_decimation']),mc_params['min_num_monte_carlo'])
    offset = 1e2   # Maximum distance from center to a single instance of the source position (per dimension)
    x_source = x_source_ctr[:, np.newaxis] + offset * (-1 + 2 * rng.standard_normal(size=(2, num_monte_carlo)))

    x_tdoa = np.array([[1., 3., 4., 5., 2.], [0., .5, 0., .5, -1.]]) * 1e3
    _, num_tdoa = np.shape(x_tdoa)

    # Initialize error covariance matrix
    time_err = 1e-7         # 100 ns time of arrival error per sensor
    cov_toa = CovarianceMatrix((time_err**2) * np.eye(num_tdoa))
    cov_roa = cov_toa.multiply(speed_of_light**2, overwrite=False)

    pss_common = TDOAPassiveSurveillanceSystem(x=x_tdoa, cov=cov_roa, ref_idx=None, variance_is_toa=False)
    pss_full = TDOAPassiveSurveillanceSystem(x=x_tdoa, cov=cov_roa, ref_idx='full', variance_is_toa=False)

    # TDOA Measurement and Combined Covariance Matrix
    z_common = pss_common.measurement(x_source=x_source)  # num_tdoa x num_monte_carlo
    z_full = pss_full.measurement(x_source=x_source)

    # Generate random noise
    noise_sensor = cov_roa.sample(num_samples=num_monte_carlo)
    noise_common = resample_noise(noise_sensor, ref_idx=None)
    noise_full = resample_noise(noise_sensor, ref_idx='full')

    # Noisy Measurements
    zeta_common = z_common + noise_common
    zeta_full = z_full + noise_full

    # ---- Set Up Solution Parameters ----
    # ML Search Parameters
    ml_search = SearchSpace(x_ctr=np.array([2.5, 2.5]) * 1e3,
                            max_offset=np.array([5, 5]) * 1e3,
                            epsilon=20)  # meters, grid resolution

    # GD and LS Search Parameters
    gd_ls_args = dict(x_init=np.array([1, 1]) * 1e3,
                      epsilon=ml_search.epsilon,
                      max_num_iterations=200,
                      force_full_calc=True,
                      plot_progress=False)

    rmse_ml = np.zeros((num_monte_carlo,))
    rmse_gd = np.zeros((num_monte_carlo, gd_ls_args['max_num_iterations']))
    rmse_ls = np.zeros((num_monte_carlo, gd_ls_args['max_num_iterations']))

    rmse_ml_full = np.zeros((num_monte_carlo,))
    rmse_gd_full = np.zeros((num_monte_carlo, gd_ls_args['max_num_iterations']))
    rmse_ls_full = np.zeros((num_monte_carlo, gd_ls_args['max_num_iterations']))

    print('Performing Monte Carlo simulation...')
    t_start = time.perf_counter()

    iterations_per_marker = 1
    markers_per_row = 40
    iterations_per_row = markers_per_row * iterations_per_marker
    for idx in range(num_monte_carlo):
        print_progress(num_monte_carlo, idx, iterations_per_marker, iterations_per_row, t_start)

        this_source = x_source[:, idx]
        this_zeta_common = zeta_common[:, idx]
        this_zeta_full = zeta_full[:, idx]

        res_common = _mc_iteration(pss_common, this_zeta_common, ml_search, gd_ls_args)
        res_full = _mc_iteration(pss_full, this_zeta_full, ml_search, gd_ls_args)

        rmse_ml[idx] = np.linalg.norm(res_common['ml'] - this_source)
        rmse_gd[idx, :] = np.linalg.norm(res_common['gd'] - this_source[:, np.newaxis], axis=0)
        rmse_ls[idx, :] = np.linalg.norm(res_common['ls'] - this_source[:, np.newaxis], axis=0)

        rmse_ml_full[idx] = np.linalg.norm(res_full['ml'] - this_source)
        rmse_gd_full[idx, :] = np.linalg.norm(res_full['gd'] - this_source[:, np.newaxis], axis=0)
        rmse_ls_full[idx, :] = np.linalg.norm(res_full['ls'] - this_source[:, np.newaxis], axis=0)

    print('done')
    t_elapsed = time.perf_counter() - t_start
    print_elapsed(t_elapsed)

    # Compute average error across Monte Carlo Iterations
    rmse_avg_ml = np.sqrt(np.sum(rmse_ml ** 2) / num_monte_carlo)
    rmse_avg_gd = np.sqrt(np.sum(rmse_gd ** 2, axis=0) / num_monte_carlo)
    rmse_avg_ls = np.sqrt(np.sum(rmse_ls ** 2, axis=0) / num_monte_carlo)
    rmse_avg_ml_full = np.sqrt(np.sum(rmse_ml_full ** 2) / num_monte_carlo)
    rmse_avg_gd_full = np.sqrt(np.sum(rmse_gd_full ** 2, axis=0) / num_monte_carlo)
    rmse_avg_ls_full = np.sqrt(np.sum(rmse_ls_full ** 2, axis=0) / num_monte_carlo)

    fig_err = plt.figure()
    x_arr = np.arange(gd_ls_args['max_num_iterations'])
    ml_hdl = plt.plot(x_arr, rmse_avg_ml * np.ones_like(x_arr), label='Maximum Likelihood',
                      marker='x', markevery=10)
    plt.plot(x_arr, rmse_avg_ml_full * np.ones_like(x_arr), '--', label='Maximum Likelihood (full)',
             marker='o', markevery=10, color=ml_hdl[0].get_color(), markerfacecolor=None)
    gd_hdl = plt.plot(x_arr, rmse_avg_gd, label='Gradient Descent',
                      marker='+', markevery=10)
    plt.plot(x_arr, rmse_avg_gd_full, '-.', label='Gradient Descent (full)',
             marker='^', markevery=10, color=gd_hdl[0].get_color(), markerfacecolor=None)
    ls_hdl = plt.plot(x_arr, rmse_avg_ls, label='Least Square',
                      marker='*', markevery=10)
    plt.plot(x_arr, rmse_avg_ls_full, '-.', label='Least Square (full)',
             marker='v', markevery=10, color=ls_hdl[0].get_color(), markerfacecolor=None)

    plt.xlabel('Number of Iterations')
    plt.ylabel('RMSE [m]')
    plt.yscale('log')
    plt.title('Monte Carlo Geolocation Results')
    plt.xlim([0, x_arr[-1]])
    plt.ylim([100, 4000])

    # ---- Estimate Error Bounds ----
    # CRLB
    crlb_common = pss_common.compute_crlb(x_source=x_source_ctr)
    crlb_full = pss_full.compute_crlb(x_source=x_source_ctr)

    print('CRLB (using common sensor:')
    print('{} m^2'.format(crlb_common))
    print('CRLB (using full sensor:')
    print('{} m^2'.format(crlb_full))

    rmse_crlb = compute_rmse(crlb_common)
    rmse_crlb_full = compute_rmse(crlb_full)

    plt.plot(x_arr, rmse_crlb * np.ones_like(x_arr), '-', color='k', label='CRLB')
    plt.plot(x_arr, rmse_crlb_full * np.ones_like(x_arr), '--', color='k', label='CRLB (full)')
    plt.legend(loc='upper right')

    # CEP50
    cep50_crlb = compute_cep50(crlb_common)
    cep50_crlb_full = compute_cep50(crlb_full)
    print('CEP50: {:.2f} km ({:.2f} km using full set)'.format(cep50_crlb / 1e3, cep50_crlb_full / 1e3))

    # 90% Error Ellipse
    conf_interval = 90
    crlb_ellipse = draw_error_ellipse(x=x_source[:, -1], covariance=crlb_common, num_pts=101,
                                                   conf_interval=conf_interval)
    # crlb_ellipse_full = draw_error_ellipse(x=x_source[:, -1], covariance=crlb_full, num_pts=101,
    #                                                     conf_interval=conf_interval)

    # ---- Plot Results from the final Monte Carlo iteration----
    # x_init = gd_ls_args['x_init']
    x_ml = res_common['ml']
    x_ls = res_common['ls']
    x_gd = res_common['gd']
    x_ml_full = res_full['ml']
    x_ls_full = res_full['ls']
    x_gd_full = res_full['gd']

    fig_full = plt.figure()
    plt.scatter(x_source[0, -1]/1e3, x_source[1, -1]/1e3, marker='x', color='k', label='Target', clip_on=False, zorder=3)
    plt.scatter(x_tdoa[0], x_tdoa[1], marker='s', color='k', label='Sensors', clip_on=False, zorder=3,
                facecolor=None)

    # Plot Closed-Form Solution
    plt.scatter(x_ml[0]/1e3, x_ml[1]/1e3, marker='x', label='Maximum Likelihood', zorder=3, color=ml_hdl[0].get_color())
    plt.scatter(x_ml_full[0]/1e3, x_ml_full[1]/1e3, marker='o', label='Maximum Likelihood (full)', zorder=3,
                color=ml_hdl[0].get_color(), facecolor=None)

    # Plot Iterative Solutions
    # plt.scatter(x_init[0]/1e3, x_init[1]/1e3, marker='x', color='k', label='Initial Estimate')
    plt.plot(x_gd[0]/1e3, x_gd[1]/1e3, linestyle='-', marker='+', markevery=[-1], label='Grad Descent',
             color=gd_hdl[0].get_color())
    plt.plot(x_gd_full[0]/1e3, x_gd_full[1]/1e3, linestyle='--', marker='^', markevery=[-1],
             label='Grad Descent (full)', color=gd_hdl[0].get_color(), facecolor=None)
    plt.plot(x_ls[0]/1e3, x_ls[1]/1e3, linestyle='-', marker='*', markevery=[-1], label='Least Squares',
             color=ls_hdl[0].get_color())
    plt.plot(x_ls_full[0]/1e3, x_ls_full[1]/1e3, linestyle='-.', marker='v', markevery=[-1],
             label='Least Squares (full)', color=ls_hdl[0].get_color(), facecolor=None)

    # Overlay Error Ellipse
    plt.plot(crlb_ellipse[0]/1e3, crlb_ellipse[1]/1e3, linestyle='-', color='k',
             label='{:d}% Error Ellipse'.format(conf_interval))
    plt.legend(loc='best')

    plt.xlim([0.5, 5.5])
    plt.ylim([2.6, 4.6])
    plt.xlabel('x [km]')
    plt.ylabel('y [km]')

    return fig_err, fig_full


def _mc_iteration(pss:TDOAPassiveSurveillanceSystem, zeta, ml_search: SearchSpace, gd_ls_args):
    """
    Executes a single iteration of the Monte Carlo simulation in Example 3.4.

    :return estimates: Dictionary with estimated target position using several algorithms.  Fields are:
                ml:         Maximum Likelihood solution with default common sensor
                gd:         Gradient Descent solution with default common sensor
                ls:         Least Squares solution with default common sensor

    Nicholas O'Donoughue
    28 January 2025
    """

    # ---- Apply Various Solvers ----
    # ML Solution
    x_ml, _, _ = pss.max_likelihood(zeta=zeta, search_space=ml_search)

    # GD Solution
    _, x_gd = pss.gradient_descent(zeta=zeta, **gd_ls_args)

    # LS Solution
    _, x_ls = pss.least_square(zeta=zeta, **gd_ls_args)

    return {'ml': x_ml, 'ls': x_ls, 'gd': x_gd}


def _plot_contourf(x_grid, extent, grid_shape, z, x_sensors, v_sensors, levels, colors, units='m'):
    this_fig = plt.figure()
    hdl = plt.imshow(np.reshape(z, grid_shape), origin='lower', cmap=colors, extent=extent,
                     norm=matplotlib.colors.LogNorm(vmin=levels[0], vmax=levels[-1]))
    plt.colorbar(hdl, format='%d')

    # Unlike in MATLAB, contourf does not draw contour edges. Manually add contours
    hdl2 = plt.contour(x_grid[0], x_grid[1], np.reshape(z, grid_shape), levels=levels,
                       origin='lower', colors='k')
    plt.clabel(hdl2, fontsize=10, colors='w')

    hdl3 = plt.scatter(x_sensors[0], x_sensors[1], color='w', facecolors='none', marker='o', label='Sensors',
                       zorder=3)
    if v_sensors is not None:
        for this_x, this_v in zip(x_sensors.T, v_sensors.T):  # transpose so the loop steps over sensors, not dimensions
            plt.arrow(x=this_x[0], y=this_x[1],
                      dx=this_v[0]*10, dy=this_v[1]*10,
                      width=.05, head_width=.4,
                      color=hdl3.get_edgecolor(), label=None)
    plt.grid(False)
    plt.xlabel(f"x [{units}]")
    plt.ylabel(f"y [{units}]")

    return this_fig


if __name__ == '__main__':
    run_all_examples()
    plt.show()
