"""
Draw Figures - Chapter 13

This script generates all the figures that appear in Chapter 13 of the textbook.

Ported from MATLAB Code

Nicholas O'Donoughue
4 December 2022
"""

import utils
import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns
import triang
import tdoa
import fdoa
import hybrid
from examples import chapter13


def make_all_figures(close_figs=False, force_recalc=False):
    """
    Call all the figure generators for this chapter

    :param close_figs: Boolean flag.  If true, will close all figures after generating them; for batch scripting.
                 Default=False
    :param force_recalc: If set to False, will skip any figures that are time-consuming to generate.
    :return: List of figure handles
    """

    # Find the output directory
    prefix = utils.init_output_dir('chapter13')

    # Random Number Generator
    rng = np.random.default_rng(0)

    # Activate seaborn for prettier plots
    sns.set()

    # Colormap
    # colors = plt.get_cmap("tab10")

    # Generate all figures
    fig1 = make_figure_1(prefix)
    plt.show()
    fig2a, fig2b, fig2c, fig2d = make_figure_2(prefix)
    plt.show()
    fig3, fig4 = make_figures_3_4(prefix, rng, force_recalc)
    plt.show()
    fig5, fig6 = make_figures_5_6(prefix, rng, force_recalc)
    plt.show()
    fig7a, fig7b = make_figure_7(prefix)
    plt.show()
    fig8a, fig8b = make_figure_8(prefix)
    plt.show()
    fig9 = make_figure_9(prefix)
    plt.show()

    figs = [fig1, fig2a, fig2b, fig2c, fig2d, fig3, fig4, fig5, fig6, fig7a, fig7b, fig8a, fig8b, fig9]
    if close_figs:
        for fig in figs:
            plt.close(fig)

        return None
    else:
        plt.show()

        return figs


def make_figure_1(prefix=None, colors=None):
    """
    Figure 1, System Drawing

    Ported from MATLAB Code

    Nicholas O'Donoughue
    28 October 2022

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Figure 1, System Drawing
    print('Generating Figure 13.1...')

    x_source = np.array([2., 4.])  # Transmitter/source
    x_sensor = np.array([[0., 0.], [3., 0.]])
    v_sensor = np.array([[1., 0.],  [1., 0.]])
    tdoa_ref_idx = 0
    fdoa_ref_idx = 0

    # Draw Geometry
    fig1 = plt.figure()
    plt.scatter(x_source[0], x_source[1], marker='^', color=colors[0], label='Transmitter')
    plt.scatter(x_sensor[:, 0], x_sensor[:, 1], marker='o', color=colors[1], label='Sensors')
    for this_x, this_v in zip(x_sensor, v_sensor):
        plt.arrow(x=this_x[0], y=this_x[1],
                  dx=this_v[0]/4, dy=this_v[1]/4,
                  width=.01, head_width=.05,
                  color=colors[1])

    for idx, this_x in enumerate(x_sensor):
        plt.text(this_x[0]-.2, this_x[1]-.2, '$S_{}$'.format(idx+1), fontsize=10)

    # True Measurements
    range_act = utils.geo.calc_range(x1=x_sensor.T, x2=x_source)
    psi_act = triang.model.measurement(x_sensor=x_sensor.T, x_source=x_source)
    rdiff = tdoa.model.measurement(x_sensor=x_sensor.T, x_source=x_source, ref_idx=tdoa_ref_idx)
    vdiff = fdoa.model.measurement(x_sensor=x_sensor.T, v_sensor=v_sensor.T, x_source=x_source, v_source=None,
                                   ref_idx=fdoa_ref_idx)

    # Draw DF lines
    this_lob_label = 'AOA Solution'
    for this_x, this_psi, this_range in zip(x_sensor, psi_act, range_act):
        # Plot the LOB
        lob_x = this_x[0] + np.array([0, np.cos(this_psi)]) * 5 * this_range
        lob_y = this_x[1] + np.array([0, np.sin(this_psi)]) * 5 * this_range
        plt.plot(lob_x, lob_y, color=colors[2], linestyle='-', label=this_lob_label)

        # Turn off legend entry for subsequent angles
        this_lob_label = None

    # Draw isochrone
    xy_isochrone = tdoa.model.draw_isochrone(x_sensor[0, :], x_sensor[1, :], range_diff=rdiff,
                                             num_pts=1000, max_ortho=5)
    plt.plot(xy_isochrone[0], xy_isochrone[1], color=colors[3], linestyle=':', label='Isochrone')

    # Draw isodoppler line
    x_isodop, y_isodop = fdoa.model.draw_isodop(x1=x_sensor[0], v1=v_sensor[0], x2=x_sensor[1], v2=v_sensor[1],
                                                vdiff=vdiff, num_pts=1000, max_ortho=5)

    plt.plot(x_isodop, y_isodop, color=colors[4], linestyle='-.', label='Lines of Constant FDOA')

    # Adjust Plot Display
    plt.xlim([-2, 4])
    plt.ylim([-1, 5])
    plt.legend(loc='upper left')

    # Remove the axes for a clean image
    plt.axis('off')

    if prefix is not None:
        fig1.savefig(prefix + 'fig1.png')
        fig1.savefig(prefix + 'fig1.svg')

    return fig1


def make_figure_2(prefix=None):
    """
    Figure 2, Hybrid Error

    Ported from MATLAB Code

    Nicholas O'Donoughue
    4 December 2022

    :param prefix: output directory to place generated figure
    :return: figure handles
    """

    # Figure 2 -- FDOA Error
    print('Generating Figure 13.2...')

    # Make the Sensor and Source Positions
    std_baseline = 10e3
    std_vel = 100
    x_sensor = np.array([[0., 0.], [3., 0.]]) * std_baseline
    v_sensor = np.array([[1., 0.], [1., 0.]]) * std_vel
    num_sensors = np.shape(x_sensor)[0]
    tdoa_ref_idx = num_sensors - 1
    fdoa_ref_idx = num_sensors - 1

    x_source = np.array([2., 4.]) * std_baseline
    transmit_freq = 10e9

    # Calculate the Hybrid Error using each covariance error setting
    x_max = 50e3
    num_pts = 401

    # Figures 13.2a -- Baseline
    print('Generating Figure 13.2a...')
    ang_err = .2  # rad
    time_err = 1000e-9  # sec
    rng_err = utils.constants.speed_of_light * time_err  # m
    freq_err = 100  # Hz
    rng_rate_err = freq_err * utils.constants.speed_of_light / transmit_freq  # m / s

    # Error Covariance Matrices
    covar_ang = ang_err ** 2 * np.eye(num_sensors)
    covar_roa = rng_err ** 2 * np.eye(num_sensors)
    covar_rroa = rng_rate_err ** 2 * np.eye(num_sensors)
    covar_rdoa = utils.resample_covariance_matrix(covar_roa, tdoa_ref_idx)
    covar_rrdoa = utils.resample_covariance_matrix(covar_rroa, fdoa_ref_idx)
    covar_rho = scipy.linalg.block_diag(covar_ang, covar_rdoa, covar_rrdoa)

    fig2a = _make_figure2_subfigure(x_sensor, v_sensor, x_source, covar_rho, x_max, num_pts)

    # Figure 12.3b -- Better DF
    print('Generating Figure 13.2b...')
    ang_err_highres = .02  # rad
    covar_ang_highres = ang_err_highres ** 2 * np.eye(num_sensors)
    covar_rho = scipy.linalg.block_diag(covar_ang_highres, covar_rdoa, covar_rrdoa)

    fig2b = _make_figure2_subfigure(x_sensor, v_sensor, x_source, covar_rho, x_max, num_pts)

    # Figure 12.3c -- Better TDOA
    print('Generating Figure 13.2c...')
    time_err_highres = 100e-9  # sec
    rng_err_highres = utils.constants.speed_of_light * time_err_highres
    covar_roa_highres = rng_err_highres ** 2 * np.eye(num_sensors)
    covar_rdoa_highres = utils.resample_covariance_matrix(covar_roa_highres, tdoa_ref_idx)
    covar_rho = scipy.linalg.block_diag(covar_ang, covar_rdoa_highres, covar_rrdoa)

    fig2c = _make_figure2_subfigure(x_sensor, v_sensor, x_source, covar_rho, x_max, num_pts)

    # Figure 12.3d -- Better FDOA
    print('Generating Figure 13.2d...')
    freq_err_highres = 10  # Hz
    rng_rate_err_highres = freq_err_highres * utils.constants.speed_of_light / transmit_freq  # m / s
    covar_rroa_highres = rng_rate_err_highres ** 2 * np.eye(num_sensors)
    covar_rrdoa_highres = utils.resample_covariance_matrix(covar_rroa_highres, fdoa_ref_idx)
    covar_rho = scipy.linalg.block_diag(covar_ang, covar_rdoa, covar_rrdoa_highres)

    fig2d = _make_figure2_subfigure(x_sensor, v_sensor, x_source, covar_rho, x_max, num_pts)

    if prefix is not None:
        fig2a.savefig(prefix + 'fig2a.png')
        fig2a.savefig(prefix + 'fig2a.svg')

        fig2b.savefig(prefix + 'fig2b.png')
        fig2b.savefig(prefix + 'fig2b.svg')

        fig2c.savefig(prefix + 'fig2c.png')
        fig2c.savefig(prefix + 'fig2c.svg')

        fig2d.savefig(prefix + 'fig2d.png')
        fig2d.savefig(prefix + 'fig2d.svg')

    return fig2a, fig2b, fig2c, fig2d


def _make_figure2_subfigure(x_sensor, v_sensor, x_source, covar_rho, x_max, num_pts):

    # Generate the likelihood; for plotting
    # Alternatively, we could plot the error (eps)
    eps, x_vec,  y_vec = hybrid.model.error(x_aoa=x_sensor.T, x_tdoa=x_sensor.T, x_fdoa=x_sensor.T, v_fdoa=v_sensor.T,
                                            cov=covar_rho, x_source=x_source, x_max=x_max, num_pts=num_pts,
                                            do_resample=False)

    # Open the plot
    fig, ax = plt.subplots()

    # Make the background image using the difference between each pixel's FDOA and the true source's FDOA
    handle_contour = plt.contour(x_vec/1e3, y_vec/1e3, -eps, levels=np.arange(start=-100, step=20, stop=0))
    ax.clabel(handle_contour, fmt='%.0f', colors='k', fontsize=14)

    # Add the sensors and source markers
    handle_sensors = plt.scatter(x_sensor[:, 0]/1e3, x_sensor[:, 1]/1e3,
                                 marker='o', color='k', s=16)
    plt.scatter(x_source[0]/1e3, x_source[1]/1e3, marker='^', s=18)
    plt.text(x_source[0]/1e3 + 2, x_source[1]/1e3 - 2, 'Source', fontsize=12)

    # Annotate the sensors
    for sensor_num in np.arange(np.size(x_sensor, axis=0)):
        this_x = x_sensor[sensor_num]
        this_v = v_sensor[sensor_num]

        # Velocity Arrow
        plt.arrow(x=this_x[0]/1e3, y=this_x[1]/1e3, dx=this_v[0] / 20, dy=this_v[1] / 20,
                  width=.1, head_width=.5, color=handle_sensors.get_edgecolor())

        # Annotation Text
        plt.text(this_x[0]/1e3 - 2, this_x[1]/1e3 - 2, '$S_{}$'.format(sensor_num), fontsize=12)

    # Adjust limits
    plt.xlim([-10, 40])
    plt.ylim([-10, 50])
    plt.grid('on')

    # Remove the axes for a clean image
    # plt.axis('off')

    return fig


def make_figures_3_4(prefix=None, rng=np.random.default_rng(0), force_recalc=False):
    """
    Figures 3 and 4, Example 13.1: Homogeneous (3-mode) Sensors

    Ported from MATLAB Code

    Nicholas O'Donoughue
    4 December 2022

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    if not force_recalc:
        print('Skipping Figures 13.3 and 13.4...')
        return None, None

    print('Generating Figures 13.3 and 13.4...')
    fig3, fig4 = chapter13.example1(rng)

    if prefix is not None:
        fig3.savefig(prefix + 'fig3.png')
        fig3.savefig(prefix + 'fig3.svg')

        fig4.savefig(prefix + 'fig4.png')
        fig4.savefig(prefix + 'fig4.svg')

    return fig3, fig4


def make_figures_5_6(prefix=None, rng=np.random.default_rng(0), force_recalc=False):
    """
    Figures 3 and 4, Example 13.2: Heterogeneous Sensors

    Ported from MATLAB Code

    Nicholas O'Donoughue
    4 December 2022

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    if not force_recalc:
        print('Skipping Figures 13.5 and 13.6...')
        return None, None

    print('Generating Figures 13.5 and 13.6...')
    fig5, fig6 = chapter13.example2(rng)

    if prefix is not None:
        fig5.savefig(prefix + 'fig5.png')
        fig5.savefig(prefix + 'fig5.svg')

        fig6.savefig(prefix + 'fig6.png')
        fig6.savefig(prefix + 'fig6.svg')

    return fig5, fig6


def make_figure_7(prefix):
    """
    Figures 7, CRLB

    Ported from MATLAB Code

    Nicholas O'Donoughue
    4 December 2022

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    print('Generating Figure 13.7...')

    # Make the Sensor and Source Positions
    std_baseline = 10e3
    std_vel = 100
    x_sensor = np.array([[-.5, 0], [.5, 0.]]) * std_baseline
    v_sensor = np.array([[1., 0.], [1., 0.]]) * std_vel
    num_sensors = np.shape(x_sensor)[0]
    tdoa_ref_idx = num_sensors - 1
    fdoa_ref_idx = num_sensors - 1

    # Define Sensor Performance
    transmit_freq = 1e9  # Hz
    ang_err = .06  # rad
    time_err = 1e-7  # 100 ns resolution
    rng_err = utils.constants.speed_of_light * time_err  # m
    freq_err = 10  # Hz
    rng_rate_err = freq_err * utils.constants.speed_of_light / transmit_freq  # m/s

    # Error Covariance Matrices
    covar_ang = ang_err ** 2 * np.eye(num_sensors)
    covar_roa = rng_err ** 2 * np.eye(num_sensors)
    covar_rroa = rng_rate_err ** 2 * np.eye(num_sensors)
    covar_rdoa = utils.resample_covariance_matrix(covar_roa, tdoa_ref_idx)
    covar_rrdoa = utils.resample_covariance_matrix(covar_rroa, fdoa_ref_idx)
    covar_rho = scipy.linalg.block_diag(covar_ang, covar_rdoa, covar_rrdoa)

    # Define source positions
    x_ctr = np.array([0., 0.])
    num_elements = 201
    max_offset = 100e3
    grid_spacing = 2 * max_offset / (num_elements - 1)
    x_set, x_grid, grid_shape = utils.make_nd_grid(x_ctr=x_ctr, max_offset=max_offset, grid_spacing=grid_spacing)

    # Figure 13.7a
    print('Generating Figure 13.7a...')

    # Compute CRLB
    # warning('off','MATLAB:nearlySingularMatrix'); % We know the problem is ill-defined, deactivate the warning
    crlb = hybrid.perf.compute_crlb(x_aoa=x_sensor.T, x_tdoa=x_sensor.T, x_fdoa=x_sensor.T, v_fdoa=v_sensor.T,
                                    x_source=x_set.T, cov=covar_rho, do_resample=False)
    cep50 = np.reshape(utils.errors.compute_cep50(crlb), newshape=grid_shape)
    # warning('on','MATLAB:nearlySingularMatrix'); % Reactivate the singular matrix warning

    # Set up contours
    contour_levels = [.1, 1, 5, 10, 50, 100]
    contour_level_labels = [.1, 1, 5, 10, 50, 100]
    fmt = {}
    for level, label in zip(contour_levels, contour_level_labels):
        fmt[level] = label

    # Draw Figure
    fig7a, ax = plt.subplots()

    handle_sensors = plt.scatter(x_sensor[:, 0] / 1e3, x_sensor[:, 1] / 1e3, marker='o', label='Sensors')
    contour_set = ax.contour(x_grid[0] / 1e3, x_grid[1] / 1e3, cep50 / 1e3, contour_levels)
    ax.clabel(contour_set, contour_set.levels, inline=True, fmt=fmt, fontsize=10)

    # Draw Velocity Arrows
    for this_x, this_v in zip(x_sensor, v_sensor):
        # Velocity Arrow
        plt.arrow(x=this_x[0]/1e3, y=this_x[1]/1e3, dx=this_v[0] / 10, dy=this_v[1] / 10,
                  width=.01, head_width=.05, color=handle_sensors.get_edgecolor())

    # Adjust the Display
    plt.xlabel('Cross-range [km]')
    plt.ylabel('Down-range [km]')
    plt.legend(loc='upper right')
    plt.grid('off')

    # Figure 13.7b -- Repeat with +x Velocity
    print('Generating Figure 13.7b...')
    v_sensor = np.array([[0., 1.], [0., 1.]]) * std_vel

    # warning('off','MATLAB:nearlySingularMatrix'); % We know the problem is ill-defined, deactivate the warning
    crlb = hybrid.perf.compute_crlb(x_aoa=x_sensor.T, x_tdoa=x_sensor.T, x_fdoa=x_sensor.T, v_fdoa=v_sensor.T,
                                    x_source=x_set.T, cov=covar_rho, do_resample=False)
    cep50 = np.reshape(utils.errors.compute_cep50(crlb), newshape=grid_shape)
    # warning('on','MATLAB:nearlySingularMatrix'); % Reactivate the singular matrix warning

    # Draw Figure
    fig7b, ax = plt.subplots()

    handle_sensors = plt.scatter(x_sensor[:, 0] / 1e3, x_sensor[:, 1] / 1e3, marker='o', label='Sensors')
    contour_set = ax.contour(x_grid[0] / 1e3, x_grid[1] / 1e3, cep50 / 1e3, contour_levels)
    ax.clabel(contour_set, contour_set.levels, inline=True, fmt=fmt, fontsize=10)

    # Draw Velocity Arrows
    for this_x, this_v in zip(x_sensor, v_sensor):
        # Velocity Arrow
        plt.arrow(x=this_x[0]/1e3, y=this_x[1]/1e3, dx=this_v[0] / 10, dy=this_v[1] / 10,
                  width=.01, head_width=.05, color=handle_sensors.get_edgecolor())

    # Adjust the Display
    plt.xlabel('Cross-range [km]')
    plt.ylabel('Down-range [km]')
    plt.legend(loc='upper right')
    plt.grid('off')

    if prefix is not None:
        fig7a.savefig(prefix + 'fig7a.png')
        fig7a.savefig(prefix + 'fig7a.svg')

        fig7b.savefig(prefix + 'fig7b.png')
        fig7b.savefig(prefix + 'fig7b.svg')

    return fig7a, fig7b


def make_figure_8(prefix):
    """
    Figures 8, CRLB with TDOA and FDOA sensors

    Ported from MATLAB Code

    Nicholas O'Donoughue
    4 December 2022

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    print('Generating Figure 13.8...')

    # Make the Sensor and Source Positions
    std_baseline = 10e3
    std_vel = 100
    x_sensor = np.array([[-.5, 0], [.5, 0.]]) * std_baseline
    v_sensor = np.array([[1., 0.], [1., 0.]]) * std_vel
    num_sensors = np.shape(x_sensor)[0]
    tdoa_ref_idx = num_sensors - 1
    fdoa_ref_idx = num_sensors - 1

    # Define Sensor Performance
    transmit_freq = 1e9  # Hz
    time_err = 1e-7  # 100 ns resolution
    rng_err = utils.constants.speed_of_light * time_err  # m
    freq_err = 10  # Hz
    rng_rate_err = freq_err * utils.constants.speed_of_light / transmit_freq  # m/s

    # Error Covariance Matrices
    covar_roa = rng_err ** 2 * np.eye(num_sensors)
    covar_rroa = rng_rate_err ** 2 * np.eye(num_sensors)
    covar_rdoa = utils.resample_covariance_matrix(covar_roa, tdoa_ref_idx)
    covar_rrdoa = utils.resample_covariance_matrix(covar_rroa, fdoa_ref_idx)
    covar_rho = scipy.linalg.block_diag(covar_rdoa, covar_rrdoa)

    # Define source positions
    x_ctr = np.array([0., 0.])
    num_elements = 201
    max_offset = 100e3
    grid_spacing = 2 * max_offset / (num_elements - 1)
    x_set, x_grid, grid_shape = utils.make_nd_grid(x_ctr=x_ctr, max_offset=max_offset, grid_spacing=grid_spacing)

    # Figure 13.8a
    print('Generating Figure 13.8a...')

    # Compute CRLB
    # warning('off','MATLAB:nearlySingularMatrix'); % We know the problem is ill-defined, deactivate the warning
    crlb = hybrid.perf.compute_crlb(x_aoa=None, x_tdoa=x_sensor.T, x_fdoa=x_sensor.T, v_fdoa=v_sensor.T,
                                    x_source=x_set.T, cov=covar_rho, do_resample=False)
    cep50 = np.reshape(utils.errors.compute_cep50(crlb), newshape=grid_shape)
    # warning('on','MATLAB:nearlySingularMatrix'); % Reactivate the singular matrix warning

    # Set up contours
    contour_levels = [.1, 1, 5, 10, 50, 100]
    contour_level_labels = [.1, 1, 5, 10, 50, 100]
    fmt = {}
    for level, label in zip(contour_levels, contour_level_labels):
        fmt[level] = label

    # Draw Figure
    fig8a, ax = plt.subplots()

    handle_sensors = plt.scatter(x_sensor[:, 0] / 1e3, x_sensor[:, 1] / 1e3, marker='o', label='Sensors')
    contour_set = ax.contour(x_grid[0] / 1e3, x_grid[1] / 1e3, cep50 / 1e3, contour_levels)
    ax.clabel(contour_set, contour_set.levels, inline=True, fmt=fmt, fontsize=10)

    # Draw Velocity Arrows
    for this_x, this_v in zip(x_sensor, v_sensor):
        # Velocity Arrow
        plt.arrow(x=this_x[0]/1e3, y=this_x[1]/1e3, dx=this_v[0] / 10, dy=this_v[1] / 10,
                  width=.01, head_width=.05, color=handle_sensors.get_edgecolor())

    # Adjust the Display
    plt.xlabel('Cross-range [km]')
    plt.ylabel('Down-range [km]')
    plt.legend(loc='upper right')
    plt.grid('off')

    # Figure 13.8b -- Repeat with +x Velocity
    print('Generating Figure 13.8b...')
    v_sensor = np.array([[0., 1.], [0., 1.]]) * std_vel

    # warning('off','MATLAB:nearlySingularMatrix'); % We know the problem is ill-defined, deactivate the warning
    crlb = hybrid.perf.compute_crlb(x_aoa=None, x_tdoa=x_sensor.T, x_fdoa=x_sensor.T, v_fdoa=v_sensor.T,
                                    x_source=x_set.T, cov=covar_rho, do_resample=False)
    cep50 = np.reshape(utils.errors.compute_cep50(crlb), newshape=grid_shape)
    # warning('on','MATLAB:nearlySingularMatrix'); % Reactivate the singular matrix warning

    # Draw Figure
    fig8b, ax = plt.subplots()

    handle_sensors = plt.scatter(x_sensor[:, 0] / 1e3, x_sensor[:, 1] / 1e3, marker='o', label='Sensors')
    contour_set = ax.contour(x_grid[0] / 1e3, x_grid[1] / 1e3, cep50 / 1e3, contour_levels)
    ax.clabel(contour_set, contour_set.levels, inline=True, fmt=fmt, fontsize=10)

    # Draw Velocity Arrows
    for this_x, this_v in zip(x_sensor, v_sensor):
        # Velocity Arrow
        plt.arrow(x=this_x[0]/1e3, y=this_x[1]/1e3, dx=this_v[0] / 10, dy=this_v[1] / 10,
                  width=.01, head_width=.05, color=handle_sensors.get_edgecolor())

    # Adjust the Display
    plt.xlabel('Cross-range [km]')
    plt.ylabel('Down-range [km]')
    plt.legend(loc='upper right')
    plt.grid('off')

    if prefix is not None:
        fig8a.savefig(prefix + 'fig8a.png')
        fig8a.savefig(prefix + 'fig8a.svg')

        fig8b.savefig(prefix + 'fig8b.png')
        fig8b.savefig(prefix + 'fig8b.svg')

    return fig8a, fig8b


def make_figure_9(prefix):
    """
    Figures 9, CRLB with AOA and TDOA sensors

    Ported from MATLAB Code

    Nicholas O'Donoughue
    4 December 2022

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    print('Generating Figure 13.9...')

    # Make the Sensor and Source Positions
    std_baseline = 10e3
    x_sensor = np.array([[-.5, 0], [.5, 0.]]) * std_baseline
    num_sensors = np.shape(x_sensor)[0]
    tdoa_ref_idx = num_sensors - 1

    # Define Sensor Performance
    ang_err = .06  # rad
    time_err = 1e-7  # 100 ns resolution
    rng_err = utils.constants.speed_of_light * time_err  # m

    # Error Covariance Matrices
    covar_ang = ang_err ** 2 * np.eye(num_sensors)
    covar_roa = rng_err ** 2 * np.eye(num_sensors)
    covar_rdoa = utils.resample_covariance_matrix(covar_roa, tdoa_ref_idx)
    covar_rho = scipy.linalg.block_diag(covar_ang, covar_rdoa)

    # Define source positions
    x_ctr = np.array([0., 0.])
    num_elements = 201
    max_offset = 100e3
    grid_spacing = 2 * max_offset / (num_elements - 1)
    x_set, x_grid, grid_shape = utils.make_nd_grid(x_ctr=x_ctr, max_offset=max_offset, grid_spacing=grid_spacing)

    # Compute CRLB
    # warning('off','MATLAB:nearlySingularMatrix'); % We know the problem is ill-defined, deactivate the warning
    crlb = hybrid.perf.compute_crlb(x_aoa=x_sensor.T, x_tdoa=x_sensor.T, x_fdoa=None, v_fdoa=None,
                                    x_source=x_set.T, cov=covar_rho, do_resample=False)
    cep50 = np.reshape(utils.errors.compute_cep50(crlb), newshape=grid_shape)
    # warning('on','MATLAB:nearlySingularMatrix'); % Reactivate the singular matrix warning

    # Set up contours
    contour_levels = [.1, 1, 5, 10, 50, 100]
    contour_level_labels = [.1, 1, 5, 10, 50, 100]
    fmt = {}
    for level, label in zip(contour_levels, contour_level_labels):
        fmt[level] = label

    # Draw Figure
    fig9, ax = plt.subplots()

    handle_sensors = plt.scatter(x_sensor[:, 0] / 1e3, x_sensor[:, 1] / 1e3, marker='o', label='Sensors')
    contour_set = ax.contour(x_grid[0] / 1e3, x_grid[1] / 1e3, cep50 / 1e3, contour_levels)
    ax.clabel(contour_set, contour_set.levels, inline=True, fmt=fmt, fontsize=10)

    # Adjust the Display
    plt.xlabel('Cross-range [km]')
    plt.ylabel('Down-range [km]')
    plt.legend(loc='upper right')
    plt.grid('off')

    if prefix is not None:
        fig9.savefig(prefix + 'fig9.png')
        fig9.savefig(prefix + 'fig9.svg')

    return fig9
