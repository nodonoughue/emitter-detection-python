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
        this_fig = plt.figure()
        plt.contourf(x_grid[0], x_grid[1], np.reshape(this_cep/1e3, grid_shape_2d), levels=levels,
                     vmin=levels[0], vmax=levels[-1], origin='lower', norm=matplotlib.colors.LogNorm(),
                     cmap=colors.resampled(len(levels)-1))
        # Unlike in MATLAB, contourf does not draw contour edges. Manually add contours
        hdl = plt.contour(x_grid[0], x_grid[1], np.reshape(this_cep / 1e3, grid_shape_2d), levels=levels,
                          vmin=levels[0], vmax=levels[-1], origin='lower', colors='k')
        plt.clabel(hdl, fontsize=10, colors='w')#, fmt=matplotlib.ticker.LogFormatterExponent)#, fmt=lambda x: f"{x:.1f} km")
        plt.colorbar()
        plt.scatter(x_sensors[0, :], x_sensors[1, :], color='w', facecolors='none', marker='o', label='Sensors')

        figs.append(this_fig)

    # TODO: Repeat with higher error (cov_full = 10*cov_full, ref_set = [1,'full'])?

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

