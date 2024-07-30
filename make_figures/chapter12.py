"""
Draw Figures - Chapter 12

This script generates all the figures that appear in Chapter 12 of the textbook.

Ported from MATLAB Code

Nicholas O'Donoughue
28 October 2022
"""

import utils
import matplotlib.pyplot as plt
import numpy as np
import fdoa
from examples import chapter12


def make_all_figures(close_figs=False, force_recalc=False):
    """
    Call all the figure generators for this chapter

    :param close_figs: Boolean flag.  If true, will close all figures after generating them; for batch scripting.
                 Default=False
    :param force_recalc: If set to False, will skip any figures that are time-consuming to generate.
    :return: List of figure handles
    """

    # Find the output directory
    prefix = utils.init_output_dir('chapter12')
    utils.init_plot_style()

    # Random Number Generator
    # rng = np.random.default_rng(0)

    # Colormap
    # colors = plt.get_cmap("tab10")

    # Generate all figures
    fig1 = make_figure_1(prefix)
    fig2a, fig2b = make_figure_2(prefix)
    fig3a, fig3b, fig3c, fig3d = make_figure_3(prefix)
    fig4 = make_figure_4(prefix)
    fig5 = make_figure_5(prefix)
    fig6a, fig6b, fig6c, fig6d = make_figure_6(prefix)
    fig7a, fig7b, fig8 = make_figures_7_8(prefix, force_recalc)

    figs = [fig1, fig2a, fig2b, fig3a, fig3b, fig3c, fig3d, fig4, fig5, fig6a, fig6b, fig6c, fig6d, fig7a, fig7b, fig8]
    if close_figs:
        for fig in figs:
            plt.close(fig)

        return None
    else:
        plt.show()

        return figs


def make_figure_1(prefix=None):
    """
    Figure 1, System Drawing

    Ported from MATLAB Code

    Nicholas O'Donoughue
    28 October 2022

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    # Figure 1, System Drawing
    print('Generating Figure 12.1...')

    x_source = np.array([0., 2.])  # Transmitter/source
    x_sensor = np.array([[-1., 0.], [0., 0.], [2., 0.]])
    v_sensor = np.array([[1., 1.],  [1., 1.], [1., 1.]])
    num_sensors = np.size(x_sensor, axis=0)

    # Draw Geometry
    fig1 = plt.figure()
    plt.scatter(x_source[0], x_source[1], marker='^', label='Transmitter')
    sensor_markers = plt.scatter(x_sensor[:, 0], x_sensor[:, 1], marker='o', label='Sensors')
    for this_x, this_v in zip(x_sensor, v_sensor):
        plt.arrow(x=this_x[0], y=this_x[1],
                  dx=this_v[0]/4, dy=this_v[1]/4,
                  width=.01, head_width=.05,
                  color=sensor_markers.get_edgecolor())

    for idx in np.arange(num_sensors):
        plt.text(x_sensor[idx, 0]-.2, x_sensor[idx, 1]-.2, '$S_{}$'.format(idx+1), fontsize=10)

    # Draw isodoppler lines for S12 and S23
    iso_doppler_label = 'Lines of Constant FDOA'
    for idx in np.arange(num_sensors-1):
        idx2 = idx+1
        vdiff = utils.geo.calc_doppler_diff(x_source, np.array([0, 0]), x_sensor[idx], v_sensor[idx],
                                            x_sensor[idx2], v_sensor[idx2], utils.constants.speed_of_light)
        x_isodoppler, y_isodoppler = fdoa.model.draw_isodop(x_sensor[idx], v_sensor[idx],
                                                            x_sensor[idx2], v_sensor[idx2], vdiff, 1000, 5)

        plt.plot(x_isodoppler, y_isodoppler, linestyle='-.', label=iso_doppler_label)
        iso_doppler_label = None  # Suppress all but the first label

    plt.xlim([-2, 3])
    plt.ylim([-1, 4])
    plt.legend()

    # Remove the axes for a clean image
    plt.axis('off')

    if prefix is not None:
        fig1.savefig(prefix + 'fig1.png')
        fig1.savefig(prefix + 'fig1.svg')

    return fig1


def make_figure_2(prefix=None):
    """
    Figure 2, IsoDoppler Lines

    Ported from MATLAB Code

    Nicholas O'Donoughue
    28 October 2022

    :param prefix: output directory to place generated figure
    :return: figure handles
    """

    # Figure 2a, IsoDoppler Lines
    print('Generating Figure 12.2a...')

    x_sensor0 = np.array([-2., 0.])
    x_sensor1 = np.array([2., 0.])
    x_set, x_grid, grid_shape = utils.make_nd_grid(x_ctr=(0, 0), max_offset=5, grid_spacing=.01)

    v_sensor0 = np.array([1., 0.])
    v_sensor1 = np.array([1., 0.])
    vt = np.array([0., 0.])

    transmit_freq = 1e9  # Hz
    ddop = utils.geo.calc_doppler_diff(x_set.T, vt, x_sensor0, v_sensor0,
                                       x_sensor1, v_sensor1, transmit_freq)

    fig2a, ax2a = plt.subplots()
    ax2a.contour(x_grid[0], x_grid[1], np.reshape(ddop, newshape=grid_shape), levels=20)
    handle0 = plt.scatter(x_sensor0[0], x_sensor0[1], marker='o', s=10, color='k')
    handle1 = plt.scatter(x_sensor1[0], x_sensor1[1], marker='o', s=10, color='k')

    # Draw Velocity Arrows
    plt.arrow(x=x_sensor0[0], y=x_sensor0[1],
              dx=v_sensor0[0]/2, dy=v_sensor0[1]/2,
              width=.01, head_width=.1,
              color=handle0.get_edgecolor())
    plt.arrow(x=x_sensor1[0], y=x_sensor1[1],
              dx=v_sensor1[0]/2, dy=v_sensor1[1]/2,
              width=.01, head_width=.1,
              color=handle1.get_edgecolor())

    # Annotation Text
    plt.text(x_sensor0[0]-1, x_sensor0[1], '$S_0$', fontsize=12)
    plt.text(x_sensor1[0]-1, x_sensor1[1], '$S_1$', fontsize=12)
    plt.text(3.5,  0, '$f_1 = f_0$', fontsize=12)
    plt.text(-5,   0, '$f_1 = f_0$', fontsize=12)
    plt.text(-.75, 0, '$f_1 > f_0$', fontsize=12)

    # Remove the axes for a clean image
    plt.axis('off')

    # Figure 2b, IsoDoppler Lines
    print('Generating Figure 12.2b...')

    v_sensor0 = np.array([0., 1.])
    v_sensor1 = np.array([0., 1.])
    vt = np.array([0., 0.])

    transmit_freq = 1e9
    ddop = utils.geo.calc_doppler_diff(x_set.T, vt, x_sensor0, v_sensor0,
                                       x_sensor1, v_sensor1, transmit_freq)

    fig2b, ax2b = plt.subplots()
    ax2b.contour(x_grid[0], x_grid[1], np.reshape(ddop, newshape=grid_shape), levels=11)
    handle0 = plt.scatter(x_sensor0[0], x_sensor0[1], marker='o', s=10, color='k')
    handle1 = plt.scatter(x_sensor1[0], x_sensor1[1], marker='o', s=10, color='k')

    # Draw Velocity Arrows
    plt.arrow(x=x_sensor0[0], y=x_sensor0[1],
              dx=v_sensor0[0]/2, dy=v_sensor0[1]/2,
              width=.01, head_width=.1,
              color=handle0.get_edgecolor())
    plt.arrow(x=x_sensor1[0], y=x_sensor1[1],
              dx=v_sensor1[0]/2, dy=v_sensor1[1]/2,
              width=.01, head_width=.1,
              color=handle1.get_edgecolor())

    # Annotation Text
    plt.text(x_sensor0[0], x_sensor0[1]-1, '$S_0$', fontsize=12)
    plt.text(x_sensor1[0], x_sensor1[1]-1, '$S_1$', fontsize=12)
    plt.text(3,  3, '$f_1 > f_0$', fontsize=12)
    plt.text(-3,  3, '$f_1 < f_0$', fontsize=12)
    plt.text(-3, -3, '$f_1 > f_0$', fontsize=12)
    plt.text(3, -3, '$f_1 < f_0$', fontsize=12)

    # Remove the axes for a clean image
    plt.axis('off')

    if prefix is not None:
        fig2a.savefig(prefix + 'fig2a.png')
        fig2a.savefig(prefix + 'fig2a.svg')

        fig2b.savefig(prefix + 'fig2b.png')
        fig2b.savefig(prefix + 'fig2b.svg')

    return fig2a, fig2b


def make_figure_3(prefix):
    """
    Figure 3, FDOA Error

    Ported from MATLAB Code

    Nicholas O'Donoughue
    28 October 2022

    :param prefix: output directory to place generated figure
    :return: figure handles
    """

    # Figure 3a -- FDOA Error
    print('Generating Figure 12.3...')
    x_sensor = np.array([[0., 0.], [-1., 1.], [1., 0.]])
    v_sensor = np.array([[1., 0.], [0., 1.], [0., 1.]])
    num_sensors = np.shape(x_sensor)[0]
    x_source = np.array([1., 3.])
    # f0 = 1e9

    eps1, x_vec1,  y_vec1 = fdoa.model.error(x_sensor=x_sensor[[0, 1], :].T, v_sensor=v_sensor[[0, 1], :].T,
                                             cov=np.eye(2), x_source=x_source, x_max=4., num_pts=1001, do_resample=True)
    eps2, x_vec2, y_vec2 = fdoa.model.error(x_sensor=x_sensor[[0, 2], :].T, v_sensor=v_sensor[[0, 2], :].T,
                                            cov=np.eye(2), x_source=x_source, x_max=4, num_pts=1001, do_resample=True)
    eps3, x_vec3, y_vec3 = fdoa.model.error(x_sensor=x_sensor[[1, 2], :].T, v_sensor=v_sensor[[1, 2], :].T,
                                            cov=np.eye(2), x_source=x_source, x_max=4, num_pts=1001, do_resample=True)
    eps4, x_vec4, y_vec4 = fdoa.model.error(x_sensor=x_sensor.T, v_sensor=v_sensor.T, cov=np.eye(3),
                                            x_source=x_source, x_max=4, num_pts=1001, do_resample=True)

    # Generate Plots
    print('Generating Figures 12.3a...')
    fig3a = _make_figure3_subfigure(eps1, x_vec1, y_vec1, x_sensor, v_sensor, x_source, [int(0), int(1)])
    print('Generating Figure 12.3b...')
    fig3b = _make_figure3_subfigure(eps2, x_vec2, y_vec2, x_sensor, v_sensor, x_source, [int(0), int(2)])
    print('Generating Figure 12.3c...')
    fig3c = _make_figure3_subfigure(eps3, x_vec3, y_vec3, x_sensor, v_sensor, x_source, [int(1), int(2)])
    print('Generating Figure 12.3d...')
    fig3d = _make_figure3_subfigure(eps4, x_vec4, y_vec4, x_sensor, v_sensor, x_source, np.arange(num_sensors))

    if prefix is not None:
        fig3a.savefig(prefix + 'fig3a.png')
        fig3a.savefig(prefix + 'fig3a.svg')

        fig3b.savefig(prefix + 'fig3b.png')
        fig3b.savefig(prefix + 'fig3b.svg')

        fig3c.savefig(prefix + 'fig3c.png')
        fig3c.savefig(prefix + 'fig3c.svg')

        fig3d.savefig(prefix + 'fig3d.png')
        fig3d.savefig(prefix + 'fig3d.svg')

    return fig3a, fig3b, fig3c, fig3d


def _make_figure3_subfigure(eps, x_vec, y_vec, x_sensor, v_sensor, x_source, sensors_to_plot):

    # Open the plot
    fig, ax = plt.subplots()

    # Make the background image using the difference between each pixel's FDOA and the true source's FDOA
    ax.imshow(10 * np.log10(np.flipud(eps)), extent=(x_vec[0], x_vec[-1], y_vec[0], y_vec[-1]), aspect='auto')

    # Add the sensors and source markers
    handle_sensors = plt.scatter(x_sensor[sensors_to_plot, 0], x_sensor[sensors_to_plot, 1],
                                 marker='o', color='k', s=16)
    plt.scatter(x_source[0], x_source[1], marker='^', s=16)
    plt.text(x_source[0] + .25, x_source[1] - .25, 'Source', fontsize=12)

    # Annotate the sensors
    for sensor_num in sensors_to_plot:
        this_x = x_sensor[sensor_num]
        this_v = v_sensor[sensor_num]

        # Velocity Arrow
        plt.arrow(x=this_x[0], y=this_x[1], dx=this_v[0] / 2, dy=this_v[1] / 2,
                  width=.01, head_width=.05, color=handle_sensors.get_edgecolor())

        # Annotation Text
        plt.text(this_x[0] - .5, this_x[1], '$S_{}$'.format(sensor_num), fontsize=12)

    # Remove the axes for a clean image
    plt.axis('off')

    return fig


def make_figure_4(prefix):
    """
    Figure 4, Two-Ship FDOA Configuration

    Ported from MATLAB Code

    Nicholas O'Donoughue
    28 October 2022

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    # Figure 4, Two-Ship Configuration
    print('Generating Figure 12.4...')

    # Plot a notional two-ship (over time)
    x_sensor = np.array([[0., 0.], [.5, -1.]])
    v_sensor = np.array([[0., 1.], [0., 1.]])
    x_offset = np.array([[0., 3.]])  # Position offset between time steps
    num_time_steps = 3

    # Transmitter Position
    x_source = np.array([3, 5])

    # Plot the Sensor Positions
    fig4 = plt.figure()
    for idx_time in np.arange(num_time_steps):
        # Update Sensor Positions
        x_now = x_sensor + idx_time * x_offset

        # Plot all sensors with the same color
        handle_sensors = plt.scatter(x_now[:, 0], x_now[:, 1], marker='o', s=14)
        this_color = handle_sensors.get_edgecolor()[0]

        # Velocity Arrows
        for this_x, this_v in zip(x_now, v_sensor):
            # Velocity Arrow
            plt.arrow(x=this_x[0], y=this_x[1],
                      dx=this_v[0], dy=this_v[1],
                      width=.01, head_width=.1, color=this_color)

        # Caption the current time step
        plt.text(x_now[0, 0] - .25, x_now[0, 1] - .75, '$t_{}$'.format(idx_time),
                 fontsize=12, color=this_color)

        # Draw isodoppler lines
        vdiff = utils.geo.calc_doppler_diff(x0=x_source, v0=np.array([0, 0]),
                                            x1=x_now[0], v1=v_sensor[0],
                                            x2=x_now[1], v2=v_sensor[1],
                                            f=utils.constants.speed_of_light)
        x_isodoppler, y_isodoppler = fdoa.model.draw_isodop(x1=x_now[0], v1=v_sensor[0],
                                                            x2=x_now[1], v2=v_sensor[1],
                                                            vdiff=vdiff, num_pts=1000, max_ortho=20)
        plt.plot(x_isodoppler, y_isodoppler, linestyle='-.', label=None, color=this_color)

    # Add Transmitter to plot -- use zorder=3 to ensure the marker is above the lines
    handle_transmitter = plt.scatter(x_source[0], x_source[1], marker='^', s=20, color='k', zorder=3)
    plt.text(x_source[0]+.1, x_source[1]-.75, 'Transmitter', fontsize=12, color=handle_transmitter.get_edgecolor()[0])

    # Remove axes for a clean image
    plt.axis('off')

    plt.ylim([-5, 15])
    plt.xlim([-5, 5])

    if prefix is not None:
        fig4.savefig(prefix + 'fig4.png')
        fig4.savefig(prefix + 'fig4.svg')

    return fig4


def make_figure_5(prefix):
    """
    Figure 5, Frequency Estimation Error

    Ported from MATLAB Code

    Nicholas O'Donoughue
    28 October 2022

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    # Figure 5, Freq Estimation Error
    print('Generating Figure 12.5...')
    snr_db = np.arange(start=-10, step=.5, stop=30.5)
    # snr_lin = np.power(10, snr_db/10)

    pulse_duration = 1e-3                         # 1ms pulse
    receive_bw = np.array([1e3, 1e6])             # 10 kHz receiver
    sample_time = 1/(2*receive_bw)                # Nyquist sampling rate
    time_bw_product = pulse_duration*receive_bw   # Time-Bandwidth product
    num_samples = pulse_duration/sample_time      # Number of samples

    sigma_f = fdoa.perf.freq_crlb(sample_time[:, np.newaxis],
                                  num_samples[:, np.newaxis],
                                  snr_db[np.newaxis, :])
    sigma_fd = fdoa.perf.freq_diff_crlb(pulse_duration, receive_bw[:, np.newaxis],
                                        snr_db[np.newaxis, :])

    fig5 = plt.figure()
    for idx_bt, this_bt in enumerate(time_bw_product):
        line = plt.plot(snr_db, 2*sigma_f[idx_bt, :], linestyle='-.', linewidth=2-idx_bt/2,
                        label='$\\sigma_f$, BT={}'.format(this_bt))
        plt.plot(snr_db, sigma_fd[idx_bt, :], linestyle='--', linewidth=2-idx_bt/2,
                 label='$\\sigma_{{fd}}$, BT={}'.format(this_bt), color=line[0].get_color())

    plt.yscale('log')
    plt.legend(loc='lower left')
    plt.xlabel('SNR [dB]')
    plt.ylabel('$\\sigma_f$ [Hz]')

    if prefix is not None:
        fig5.savefig(prefix + 'fig5.png')
        fig5.savefig(prefix + 'fig5.svg')

    return fig5


def make_figure_6(prefix):
    """
    Figure 1, System Drawing

    Ported from MATLAB Code

    Nicholas O'Donoughue
    28 October 2022

    :param prefix: output directory to place generated figure
    :return: figure handle
    """
    # Figure 6a
    print('Generating Figure 12.6a...')
    # Plots of the FDOA CRLB for a 3 sensor scenario and 1 microsecond timing
    # error

    # Define Sensor Positions
    baseline = 10e3
    std_vel = 100
    num_sensors = 3
    sensor_pos_angle = np.arange(num_sensors)*2*np.pi/num_sensors + np.pi/2
    x_sensor = baseline * np.array([np.cos(sensor_pos_angle), np.sin(sensor_pos_angle)])
    v_sensor = std_vel * np.array([np.cos(sensor_pos_angle), np.sin(sensor_pos_angle)])

    # Define Sensor Performance
    freq_error = 10  # 1 Hz resolution
    f0 = 1e9
    rng_rate_std_dev = freq_error*utils.constants.speed_of_light/f0
    cov_rroa = rng_rate_std_dev**2 * np.eye(num_sensors)  # covariance matrix structure
    ref_idx = None
    cov_rrdoa = utils.resample_covariance_matrix(cov_rroa, ref_idx)

    # Define source positions
    num_grid_points = 501
    grid_extent = 100e3
    grid_spacing = 2*grid_extent/(num_grid_points-1)
    x_source, x_grid, grid_shape = utils.make_nd_grid(x_ctr=(0., 0.), grid_spacing=grid_spacing, max_offset=grid_extent)

    # Compute CRLB
    crlb = fdoa.perf.compute_crlb(x_sensor, v_sensor, x_source.T, cov_rrdoa, do_resample=False)  # Ndim x Ndim x M^2
    cep50 = np.reshape(utils.errors.compute_cep50(crlb), newshape=grid_shape)

    # Set up contours
    contour_levels = [.1, 1, 5, 10, 50, 100, 1000]
    contour_level_labels = [.1, 1, 5, 10, 50, 100]
    fmt = {}
    for level, label in zip(contour_levels, contour_level_labels):
        fmt[level] = label

    # Draw Figure
    fig6a, ax = plt.subplots()

    # ax=subplot(2,1,1)
    plt.scatter(x_sensor[0, :]/1e3, x_sensor[1, :]/1e3, marker='o', label='Sensors')
    contour_set = ax.contour(x_grid[0]/1e3, x_grid[1]/1e3, cep50/1e3, contour_levels)
    ax.clabel(contour_set, contour_levels, inline=True, fmt=fmt, fontsize=10)
    plt.legend(loc='upper right')

    # Adjust the Display
    plt.xlabel('Cross-range [km]')
    plt.ylabel('Down-range [km]')

    # Figure 6b
    print('Generating Figure 12.6b...')

    # Repeat with +x velocity
    v_sensor = 100 * np.concatenate([np.ones((1, num_sensors)), np.zeros((1, num_sensors))], axis=0)
    crlb = fdoa.perf.compute_crlb(x_sensor, v_sensor, x_source.T, cov_rrdoa, do_resample=False)  # Ndim x Ndim x M^2
    cep50 = np.reshape(utils.errors.compute_cep50(crlb), newshape=grid_shape)

    # Draw Figure
    fig6b, ax = plt.subplots()

    # ax=subplot(2,1,1)
    plt.scatter(x_sensor[0, :]/1e3, x_sensor[1, :]/1e3, marker='o', label='Sensors')
    contour_set = ax.contour(x_grid[0]/1e3, x_grid[1]/1e3, cep50/1e3, contour_levels)
    ax.clabel(contour_set, contour_levels, inline=True, fmt=fmt, fontsize=10)
    plt.legend(loc='upper right')

    # Adjust the Display
    plt.xlabel('Cross-range [km]')
    plt.ylabel('Down-range [km]')

    # Figure 6c,d, Impact of fourth sensor on CRLB
    print('Generating Figure 12.6c...')

    # Add a sensor at the origin
    x_sensor = np.concatenate((x_sensor, np.zeros(shape=(2, 1))), axis=1)
    _, num_sensors = np.shape(x_sensor)
    v_sensor = std_vel * np.array([np.cos(sensor_pos_angle), np.sin(sensor_pos_angle)])
    v_sensor = np.concatenate((v_sensor, np.array([1, 0])[:, np.newaxis]), axis=1)

    # Regenerate covariance matrix
    cov_rroa = rng_rate_std_dev**2 * np.eye(num_sensors)  # covariance matrix structure
    ref_idx = None
    cov_rrdoa = utils.resample_covariance_matrix(cov_rroa, ref_idx)

    # Compute CRLB
    crlb = fdoa.perf.compute_crlb(x_sensor, v_sensor, x_source.T, cov_rrdoa, do_resample=False)  # Ndim x Ndim x M^2
    cep50 = np.reshape(utils.errors.compute_cep50(crlb), newshape=grid_shape)

    # Draw Figure
    fig6c, ax = plt.subplots()

    plt.scatter(x_sensor[0, :]/1e3, x_sensor[1, :]/1e3, marker='o', label='Sensors')
    contour_set = ax.contour(x_grid[0]/1e3, x_grid[1]/1e3, cep50/1e3, contour_levels)
    ax.clabel(contour_set, contour_levels, inline=True, fmt=fmt, fontsize=10)
    plt.legend(loc='upper right')

    # Adjust the Display
    plt.xlabel('Cross-range [km]')
    plt.ylabel('Down-range [km]')

    # Figure 6d
    print('Generating Figure 12.6d...')

    # Repeat with +x velocity
    v_sensor = 100 * np.concatenate([np.ones((1, num_sensors)), np.zeros((1, num_sensors))], axis=0)
    crlb = fdoa.perf.compute_crlb(x_sensor, v_sensor, x_source.T, cov_rrdoa, do_resample=False)  # Ndim x Ndim x M^2
    cep50 = np.reshape(utils.errors.compute_cep50(crlb), newshape=grid_shape)

    # Draw Figure
    fig6d, ax = plt.subplots()

    plt.scatter(x_sensor[0, :]/1e3, x_sensor[1, :]/1e3, marker='o', label='Sensors')
    contour_set = ax.contour(x_grid[0]/1e3, x_grid[1]/1e3, cep50/1e3, contour_levels)
    ax.clabel(contour_set, contour_levels, inline=True, fmt=fmt, fontsize=10)
    plt.legend(loc='upper right')

    # Adjust the Display
    plt.xlabel('Cross-range [km]')
    plt.ylabel('Down-range [km]')

    if prefix is not None:
        fig6a.savefig(prefix + 'fig6a.png')
        fig6a.savefig(prefix + 'fig6a.svg')

        fig6b.savefig(prefix + 'fig6b.png')
        fig6b.savefig(prefix + 'fig6b.svg')

        fig6c.savefig(prefix + 'fig6c.png')
        fig6c.savefig(prefix + 'fig6c.svg')

        fig6d.savefig(prefix + 'fig6d.png')
        fig6d.savefig(prefix + 'fig6d.svg')

    return fig6a, fig6b, fig6c, fig6d


def make_figures_7_8(prefix, force_recalc=False):

    if not force_recalc:
        print('Skipping Figures 12.7 and 12.8... (re-run with force_recalc=True to generate)')
        return None, None, None

    # Figures 7-8, Example FDOA Calculation
    # Figure 7 is geometry
    print('Generating Figures 12.7 and 12.8...')

    fig7a, fig7b, fig8 = chapter12.example1()

    if prefix is not None:
        fig7a.savefig(prefix + 'fig7a.png')
        fig7a.savefig(prefix + 'fig7a.svg')

        fig7b.savefig(prefix + 'fig7b.png')
        fig7b.savefig(prefix + 'fig7b.svg')

        fig8.savefig(prefix + 'fig8.png')
        fig8.savefig(prefix + 'fig8.svg')

    return fig7a, fig7b, fig8
