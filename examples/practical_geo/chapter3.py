import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import utils
import tdoa
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
        this_crlb = tdoa.perf.compute_crlb(x_sensor=x_sensors, x_source=x_source, cov=cov_full, ref_idx=this_ref_set)
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
        colormap = plt.get_cmap("tab10")
        colors = (colormap(0), colormap(1), colormap(2), colormap(3))


    return None

def example4(colors=None):
    """
    Executes Example 3.4.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    17 January 2025

    :param rng: random number generator
    :param colors:
    :return: figure handle to generated graphic
    """

    if colors is None:
        colormap = plt.get_cmap("tab10")
        colors = (colormap(0), colormap(1), colormap(2), colormap(3))


    return None


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
    plt.grid('off')

    return this_fig