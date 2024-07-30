"""
Draw Figures - Chapter 11

This script generates all the figures that appear in Chapter 10 of the plt.textbook.

Ported from MATLAB Code

Nicholas O'Donoughue
8 March 2022
"""

import utils
import matplotlib.pyplot as plt
import numpy as np
import scipy
import tdoa
from examples import chapter11


def make_all_figures(close_figs=False, force_recalc=False):
    """
    Call all the figure generators for this chapter

    :param close_figs: Boolean flag.  If true, will close all figures after generating them; for batch scripting.
                 Default=False
    :param force_recalc: If set to False, will skip any figures that are time-consuming to generate.
    :return: List of figure handles
    """

    # Find the output directory
    prefix = utils.init_output_dir('chapter11')
    utils.init_plot_style()

    # Random Number Generator
    rng = np.random.default_rng(0)

    # Colormap
    # cmap = plt.get_cmap("tab10")

    # Generate all figures
    fig1a, fig1b = make_figure_1(prefix)
    fig2 = make_figure_2(prefix)
    fig3 = make_figure_3(prefix)
    fig4 = make_figure_4(prefix)
    fig5 = make_figure_5(prefix)
    fig6a, fig6b = make_figure_6(prefix)
    fig7a, fig7b, fig8 = make_figure_7_8(prefix, rng, force_recalc)
    fig9 = make_figure_9(prefix)
    fig10 = make_figure_10(prefix)

    figs = [fig1a, fig1b, fig2, fig3, fig4, fig5, fig6a, fig6b, fig7a, fig7b, fig8, fig9, fig10]

    if close_figs:
        for fig in figs:
            plt.close(fig)

        return None
    else:
        plt.show()

        return figs


def make_figure_1(prefix=None):
    """
    Figure 1a - TOA Circles
    Figure 1b - TDOA Circles

    Ported from MATLAB Code

    Nicholas O'Donoughue
    8 March 2022

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    # Figure 1a, TOA Circles
    print('Generating Figure 11.1a...')

    # Initialize Locations
    x_sensor1 = np.array([0, 0])
    x_sensor2 = np.array([1.5, .7])
    x_sensor3 = np.array([-.1, .8])
    x_source = np.array([.5, .5])
    
    # Compute Range Vectors
    r1 = utils.geo.calc_range(x_sensor1, x_source)
    r2 = utils.geo.calc_range(x_sensor2, x_source)
    r3 = utils.geo.calc_range(x_sensor3, x_source)
    
    # Generate a unit circle
    th = np.linspace(start=0, stop=2*np.pi, num=1001)
    circle_x_coords = np.cos(th)
    circle_y_coords = np.sin(th)
    x_circle = [circle_x_coords, circle_y_coords]  # 2 x 1001
    
    # Scale the unit circle and offset with position bias
    toa1 = r1 * x_circle + x_sensor1[:, np.newaxis]
    toa2 = r2 * x_circle + x_sensor2[:, np.newaxis]
    toa3 = r3 * x_circle + x_sensor3[:, np.newaxis]
    
    # Draw the circles and position hash marks
    fig1a = plt.figure()
    plt.plot(toa1[0, :], toa1[1, :], linestyle=':', label='Constant TOA')
    plt.plot(toa2[0, :], toa2[1, :], linestyle=':', label=None)
    plt.plot(toa3[0, :], toa3[1, :], linestyle=':', label=None)
    
    # Overlay plt.text
    plt.text(x_sensor1[0] + .05, x_sensor1[1] - .1, r'$S_1$')
    plt.text(x_sensor2[0] + .05, x_sensor2[1] - .1, r'$S_2$')
    plt.text(x_sensor3[0] + .05, x_sensor3[1] + .1, r'$S_3$')
    
    # plt.text(x_source[0] + .1, x_source[1] + .1, r'$T_1$')
    
    # Plot radials
    plt.plot(x_sensor1[0] + [0, r1*np.cos(5*np.pi/4)], 
             x_sensor1[1] + [0, r1*np.sin(5*np.pi/4)], color='k', linewidth=.5, label=None)
    plt.plot(x_sensor2[0] + [0, r2*np.cos(5*np.pi/4)], 
             x_sensor2[1] + [0, r2*np.sin(5*np.pi/4)], color='k', linewidth=.5, label=None)
    plt.plot(x_sensor3[0] + [0, r3*np.cos(3*np.pi/4)], 
             x_sensor3[1] + [0, r3*np.sin(3*np.pi/4)], color='k', linewidth=.5, label=None)
    
    plt.text(x_sensor1[0] + r1/2*np.cos(5*np.pi/4)+.05,
             x_sensor1[1] + r1/2*np.sin(5*np.pi/4)-.1, r'$R_1$')
    plt.text(x_sensor2[0] + r2/2*np.cos(5*np.pi/4)+.05,
             x_sensor2[1] + r2/2*np.sin(5*np.pi/4)-.1, r'$R_2$')
    plt.text(x_sensor3[0] + r3/2*np.cos(3*np.pi/4),
             x_sensor3[1] + r3/2*np.sin(3*np.pi/4), r'$R_3$')
    
    # Position Markers (with legend)
    plt.scatter([x_sensor1[0], x_sensor2[0], x_sensor3[0]],
                [x_sensor1[1], x_sensor2[1], x_sensor3[1]], marker='o', label='Sensors')
    plt.scatter(x_source[0], x_source[1], marker='^', color='black', label='Transmitter')
    
    # Adjust Axes
    plt.xlim([-2, 3])
    plt.ylim([-1, 2])
    plt.legend(loc='lower right')
    plt.axis('off')

    # Figure 1b TDOA Circles
    print('Generating Figure 11.1b...')

    # Initialize Detector/Source Locations
    x_sensor1 = np.array([0, 0])
    x_sensor2 = np.array([.8, .2])
    x_sensor3 = np.array([1, 1])
    x_source = np.array([.1, .9])
    
    # Compute Ranges
    r1 = utils.geo.calc_range(x_sensor1, x_source)
    r2 = utils.geo.calc_range(x_sensor2, x_source)
    r3 = utils.geo.calc_range(x_sensor3, x_source)
    
    # Find Isochrones
    xy_isochrone1 = tdoa.model.draw_isochrone(x_sensor1, x_sensor2, range_diff=r2 - r1, num_pts=1000, max_ortho=3)
    xy_isochrone2 = tdoa.model.draw_isochrone(x_sensor2, x_sensor3, range_diff=r3 - r2, num_pts=1000, max_ortho=3)
    
    # Draw Figure
    fig1b = plt.figure()
    
    # Isochrones
    plt.plot(xy_isochrone1[0], xy_isochrone1[1], color='k', linestyle=':', label='Isochrone')
    plt.plot(xy_isochrone2[0], xy_isochrone2[1], color='k', linestyle=':', label=None)
    
    # Isochrone Labels
    plt.text(np.mean([x_sensor1[0], x_sensor2[0]]),
             np.mean([x_sensor1[1], x_sensor2[1]]) - .2, r'$TDOA_{1,2}$')
    plt.text(np.mean([x_sensor2[0], x_sensor3[0]]) + .3,
             np.mean([x_sensor2[1], x_sensor3[1]]), r'$TDOA_{2,3}$')
    
    # Position Markers
    plt.scatter([x_sensor1[0], x_sensor2[0], x_sensor3[0]],
                [x_sensor1[1], x_sensor2[1], x_sensor3[1]], color='k', linestyle='-', linewidth=1, label=None)
    plt.scatter([x_sensor1[0], x_sensor2[0], x_sensor3[0]],
                [x_sensor1[1], x_sensor2[1], x_sensor3[1]], marker='o', label='Sensors')
    plt.scatter(x_source[0], x_source[1], marker='^', label='Transmitter')
    
    # Position Labels
    plt.text(x_sensor1[0]+.05, x_sensor1[1]-.1, r'$S_1$')
    plt.text(x_sensor2[0]+.05, x_sensor2[1]-.1, r'$S_2$')
    plt.text(x_sensor3[0]+.05, x_sensor3[1]-.1, r'$S_3$')
    
    # Adjust Axes
    plt.xlim([-2, 3])
    plt.ylim([-1, 2])
    plt.legend(loc='lower right')
    plt.axis('off')

    if prefix is not None:
        fig1a.savefig(prefix + 'fig1a.png')
        fig1a.savefig(prefix + 'fig1a.svg')

        fig1b.savefig(prefix + 'fig1b.png')
        fig1b.savefig(prefix + 'fig1b.svg')
        
    return fig1a, fig1b


def make_figure_2(prefix=None):
    """
    Figure 2 - Isochrones

    Ported from MATLAB Code

    Nicholas O'Donoughue
    8 March 2022

    :param prefix: output directory to place generated figure
    :return: figure handle
    """
    # Figure 2, Isochrones Plots
    print('Generating Figure 11.2...')

    x_sensor1 = np.array([0, 0])
    x_sensor2 = np.array([1, 0])

    # Define set of RangeDiffs
    num_isochrones = 15
    range_diff_vec = np.linspace(start=-.9, stop=.9, num=num_isochrones)
    num_points = 1000  # number of points for isochrone drawing

    # Initialize Figure
    fig2 = plt.figure()

    # Plot isochrones
    for range_diff in range_diff_vec:
        xy_isochrone = tdoa.model.draw_isochrone(x_sensor1, x_sensor2, range_diff=range_diff, num_pts=num_points,
                                                 max_ortho=3)
        plt.plot(xy_isochrone[0], xy_isochrone[1], label=None)

    # Plot Markers
    plt.plot([x_sensor1[0], x_sensor2[0]], [x_sensor1[1], x_sensor2[1]], marker='o', color='black', label='Sensors')

    # Label the sensors
    plt.text(x_sensor1[0]-.1, x_sensor1[1]-.05, r'$S_1$')
    plt.text(x_sensor2[0]+.05, x_sensor2[1]-.05, r'$S_2$')

    # Label the regimes
    plt.text(.24, 1.2, r'$R_1 < R_2$')
    plt.text(.55, 1.2, r'$R_1 > R_2$')
    plt.arrow(x=.55, y=1, dx=.15, dy=0, width=.015, color='black')
    plt.arrow(x=.45, y=1, dx=-.15, dy=0, width=.015, color='black')

    # Adjust Display
    plt.xlim([-0.5, 1.5])
    plt.ylim([-1.5, 1.5])
    plt.axis('off')

    if prefix is not None:
        fig2.savefig(prefix + 'fig2.png')
        fig2.savefig(prefix + 'fig2.svg')

    return fig2


def make_figure_3(prefix=None, rng=np.random):
    """
    Figure 3 - Plot of leading edge detection

    Ported from MATLAB Code

    Nicholas O'Donoughue
    8 March 2022

    :param prefix: output directory to place generated figure
    :param rng: random number generator [Optional]
    :return: figure handle
    """

    print('Generating Figure 11.3...')

    # Noise signal
    s2 = .1
    signal_len = 1024
    n = np.sqrt(s2)*rng.standard_normal((signal_len, ))

    # Signal
    signal_energy = 10
    t0 = int(np.floor(.3*signal_len))  # Start position of chirp
    chirp_len = np.floor(signal_len/2)    # Length of chirp
    t_chirp = np.arange(start=0, stop=chirp_len)
    p = np.sqrt(signal_energy)*scipy.signal.chirp(t=t_chirp, f0=0., t1=chirp_len, f1=.1, method='linear', phi=-90)
    y = np.zeros(shape=(signal_len, ))
    y[t0+t_chirp.astype(int)] = p

    fig3 = plt.figure()
    plt.plot(np.arange(signal_len), n, label='Noise')
    plt.plot(np.arange(signal_len), y, label='Signal')

    # Lines
    eta = 10*s2
    tt = np.argmax(np.abs(y) > eta)  # return the index of the first instance of |y| > eta
    plt.plot([1, signal_len], [eta, eta], linestyle='--', label='Threshold')
    plt.plot([tt, tt], [-np.sqrt(signal_energy), np.sqrt(signal_energy)], linestyle=':', label=r'\tau_i')
    # plt.legend(loc='upper left')

    plt.text(10, 1.2, 'Threshold')
    plt.text(10, -1.2, 'Noise')
    plt.text(830, 2.5, 'Signal')
    plt.text(300, 3, r'$\tau_i$')
    plt.xlim([1, signal_len])
    plt.axis('off')

    if prefix is not None:
        fig3.savefig(prefix + 'fig3.png')
        fig3.savefig(prefix + 'fig3.svg')

    return fig3


def make_figure_4(prefix=None, rng=np.random.default_rng()):
    """
    Figure 4 - Illustration of Cross-Correlation TDOA Processing

    Ported from MATLAB Code

    Nicholas O'Donoughue
    8 March 2022

    :param prefix: output directory to place generated figure
    :param rng: random number generator [Optional]
    :return: figure handle
    """

    print('Generating Figure 11.4...')

    # Noise signal
    s2 = 5
    noise_len = 1024
    noise = np.sqrt(s2/2)*(rng.standard_normal(size=(noise_len, )) + 1j * rng.standard_normal(size=(noise_len, )))
    # noise2 = np.sqrt(s2/2)*(rng.standard_normal(size=(noise_len, )) + 1j * rng.standard_normal(size=(noise_len, )))

    # Signal
    signal_energy = 100
    t0 = int(np.floor(.3*noise_len))  # Start position of chirp
    chirp_len = np.floor(noise_len/2)    # Length of chirp
    t_chirp = np.arange(chirp_len)
    p = np.sqrt(signal_energy/2)*scipy.signal.chirp(t_chirp, f0=0, t1=chirp_len, f1=.1, method='linear', phi=-90)
    y = np.zeros(shape=(noise_len, ), dtype=complex)
    y[t0+np.arange(chirp_len).astype(int)] = p+p*np.exp(1j*np.pi/2)

    # Sensors Received Signals
    s1 = noise+y
    s2 = noise+np.roll(y, 100)

    # Time Index
    fs = 1e4
    t = np.arange(noise_len)/fs
    tau = t-np.mean(t)

    sx = np.fft.fftshift(np.fft.ifft(np.fft.fft(s1)*np.conj(np.fft.fft(s2))))/noise_len

    fig4, (ax0, ax1, ax2) = plt.subplots(3)

    # Subplot 1
    ax0.plot(t*1e3, np.real(s1))
    ax0.set_title('Sensor 1')
    ax0.plot([36, 36], [-10, 10], 'k:')
    ax0.text(32, -5, r'$\tau_1$')
    ax0.set_ylim([-10, 10])
    ax0.set_xlim([0, (noise_len-1)*1e3/fs])

    # Subplot 2
    ax1.plot(t*1e3, np.real(s2))
    ax1.set_title('Sensor 2')
    ax1.plot([46, 46], [-10, 10], 'k:')
    ax1.text(41, -5, r'$\tau_2$')
    ax1.set_xlim([0, (noise_len-1)*1e3/fs])
    ax1.set_ylim([-10, 10])

    # Subplot 3
    ax2.plot(tau*1e3, np.abs(sx))
    ax2.plot([-10, -10], [0, 40], 'k:')
    # ax2.plot([0, 0], [0, 40], 'k:')
    ax2.text(-25, 30, r'$\tau_{1,2}$')
    # ax2.text(5, 20, '0')
    ax2.set_xlabel('Time Difference of Arrival [ms]')
    ax2.set_title('Cross Correlation')
    ax2.set_xlim(np.array([-1., 1.])*(noise_len-1)*1e3/fs)

    # Remove axis ticks to clean up plot
    for ax in (ax0, ax1, ax2):
        ax.set_yticklabels([])
        ax.set_xticklabels([])

    if prefix is not None:
        fig4.savefig(prefix + 'fig4.png')
        fig4.savefig(prefix + 'fig4.svg')

    return fig4


def make_figure_5(prefix=None):
    """
    Figure 5 - Plot of the TDOA variance for peak detection and cross-correlation

    Ported from MATLAB Code

    Nicholas O'Donoughue
    8 March 2022

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    print('Generating Figure 11.5...')

    # Link-16 Pulse Parameters
    #   Chip Duration = 200 ns (implies 5 MHz bandwidth)
    #     Assume flat pass-band shape
    #   6.4 microseconds per pulse

    # Define Pulse Parameters
    bw = np.array([5e6, 1e9])
    bw_rms = bw/np.sqrt(3)
    pulse_len_vec = np.array([6.4e-6, 10e-6])
    snr_db = np.linspace(start=0, stop=60, num=100)
    # snr_linear = 10 ** (snr_db/10)

    # Compute the variances
    var_peak = tdoa.model.toa_error_peak_detection(snr_db)
    var_cross_corr = tdoa.model.toa_error_cross_corr(snr_db[np.newaxis, :], bw[:, np.newaxis],
                                                     pulse_len_vec[:, np.newaxis], bw_rms[:, np.newaxis])

    # Draw the figure
    fig5 = plt.figure()

    plt.semilogy(snr_db, np.sqrt(var_peak)*1e6, label='Peak Detection')
    plt.text(20, .9e5, 'Peak Detection', rotation=-13)
    plt.semilogy(snr_db, np.sqrt(var_cross_corr[0, :])*1e6,
                 label="Cross-Correlation, BT={}".format(pulse_len_vec[0]*bw[0]))
    plt.text(20, .9, 'Cross-Correlation, BT=32', rotation=-13)
    plt.semilogy(snr_db, np.sqrt(var_cross_corr[1, :])*1e6,
                 label="Cross-Correlation, BT={}".format(pulse_len_vec[1]*bw[1]))
    plt.text(20, .001, 'Cross-Correlation, BT=10,000', rotation=-13)

    # Label Axes
    plt.ylabel(r'$\sigma$ [$\mu$s]')
    plt.xlabel('SNR [dB]')
    # plt.legend(loc='upper right')

    # Save Output
    if prefix is not None:
        fig5.savefig(prefix + "fig5.svg")
        fig5.savefig(prefix + "fig5.png")

    return fig5


def make_figure_6(prefix=None):
    """
    Figures 6a and 6b - Plot of TDOA CRLB for 3 (6a) and 4 (6b)
    sensor scenarios with 1 microsecond timing error

    Ported from MATLAB Code

    Nicholas O'Donoughue
    8 March 2022

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    # Define Sensor Positions
    baseline = 10e3
    num_sensors = 3
    sensor_angles = np.arange(num_sensors)*2*np.pi/num_sensors + np.pi/2
    x_sensor = baseline * np.array([np.cos(sensor_angles), np.sin(sensor_angles)])

    # Define Sensor Performance
    timing_error = 1e-7
    range_error = utils.constants.speed_of_light * timing_error
    cov_roa = range_error**2 * np.eye(num_sensors)
    # In the book, we manually resampled.  But we now have a utility that will
    # automatically resample the covariance matrix.
    # cov_tdoa = timing_error**2 * (1 + np.eye(num_sensors-1))

    # Define source positions
    num_grid_points = 501
    x_vec = np.linspace(start=-100, stop=100, num=num_grid_points)*1e3
    y_vec = np.linspace(start=-100, stop=100, num=num_grid_points)*1e3
    x_full, y_full = np.meshgrid(x_vec, y_vec)
    x_source = np.array([np.ravel(x_full), np.ravel(y_full)])  # shape=(2 x num_grid_points**2)

    # Compute CRLB
    crlb = tdoa.perf.compute_crlb(x_sensor, x_source, cov_roa, do_resample=True)  # Ndim x Ndim x M^2
    cep50 = np.reshape(utils.errors.compute_cep50(crlb), (num_grid_points, num_grid_points))

    # Set up contours
    contour_levels = [.1, 1, 5, 10, 50, 100, 1000]
    contour_level_labels = [.1, 1, 5, 10, 50, 100]
    fmt = {}
    for level, label in zip(contour_levels, contour_level_labels):
        fmt[level] = label

    # Draw Figure
    print('Generating Figure 11.6a...')
    fig6a, ax = plt.subplots()

    plt.plot(x_sensor[0, :]/1e3, x_sensor[1, :]/1e3, 'ro', label='Sensors')
    contour_set = ax.contour(x_full/1e3, y_full/1e3, cep50/1e3, contour_levels)
    ax.clabel(contour_set, contour_levels, inline=True, fmt=fmt, fontsize=10)
    plt.legend(loc='upper right')

    # Adjust the Display
    plt.xlabel('Cross-range [km]')
    plt.ylabel('Down-range [km]')

    # Figure 6b, Impact of fourth sensor on CRLB

    # Add a sensor at the origin
    x_sensor1 = np.concatenate((x_sensor, np.zeros(shape=(2, 1))), axis=1)
    _, num_sensors = np.shape(x_sensor1)

    # Adjust Sensor Performance Vector
    timing_error = 1e-7
    range_error = utils.constants.speed_of_light * timing_error
    cov_roa = range_error**2 * np.eye(num_sensors)
    # In the book, we manually resampled.  But we now have a utility that will
    # automatically resample the covariance matrix.
    # cov_tdoa = timing_error**2 * (1 + np.eye(num_sensors-1))

    crlb2 = tdoa.perf.compute_crlb(x_sensor1, x_source, cov_roa, do_resample=True)  # Ndim x Ndim x M**2
    cep50 = np.reshape(utils.errors.compute_cep50(crlb2), [num_grid_points, num_grid_points])

    # Draw the figure
    print('Generating Figure 11.6b...')
    fig6b, ax = plt.subplots()

    plt.plot(x_sensor1[0, :]/1e3, x_sensor1[1, :]/1e3, 'ro', label='Sensors')
    contour_set = ax.contour(x_full/1e3, y_full/1e3, cep50/1e3, contour_levels)
    ax.clabel(contour_set, contour_levels, inline=True, fmt=fmt, fontsize=10)
    plt.legend(loc='upper right')

    # Adjust the Display
    plt.xlabel('Cross-range [km]')
    plt.ylabel('Down-range [km]')

    if prefix is not None:
        fig6a.savefig(prefix + 'fig6a.svg')
        fig6a.savefig(prefix + 'fig6a.png')

        fig6b.savefig(prefix + 'fig6b.svg')
        fig6b.savefig(prefix + 'fig6b.png')

    return fig6a, fig6b


def make_figure_7_8(prefix=None, rng=np.random.default_rng(), force_recalc=False):
    """
    Figures 7 and 8 - Example TDOA Calculation

    Figure 7 is the geometry, Figure 8 is the estimate error for each iteration

    Ported from MATLAB Code

    Nicholas O'Donoughue
    8 March 2022

    :param prefix: output directory to place generated figure
    :param rng: random number generator
    :param force_recalc: optional flag (default=True), if False then the example does not run
    :return: figure handle
    """

    if not force_recalc:
        print('Skipping Figures 11.7 and 11.8 (re-run with force_recalc=True to generate)...')
        return None, None, None

    print('Generating Figures 11.7 and 11.8 (using Example 11.1)...')
    fig7a, fig7b, fig8 = chapter11.example1(rng)

    if prefix is not None:
        fig7a.savefig(prefix + 'fig7a.svg')
        fig7a.savefig(prefix + 'fig7a.png')

        fig7b.savefig(prefix + 'fig7b.svg')
        fig7b.savefig(prefix + 'fig7b.png')

        fig8.savefig(prefix + 'fig8.svg')
        fig8.savefig(prefix + 'fig8.png')

    return fig7a, fig7b, fig8


def make_figure_9(prefix=None):
    """
    Figure 9 - Plot of false isochrones

    Ported from MATLAB Code

    Nicholas O'Donoughue
    8 March 2022

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    # Define positions
    x_sensor1 = np.array([0, 0])
    x_sensor2 = np.array([1, 0])
    x_source1 = np.array([-.5, 1])
    x_source2 = np.array([1, 2])

    # Ranges
    r11 = utils.geo.calc_range(x_source1, x_sensor1)
    r12 = utils.geo.calc_range(x_source1, x_sensor2)
    r21 = utils.geo.calc_range(x_source2, x_sensor1)
    r22 = utils.geo.calc_range(x_source2, x_sensor2)

    # Compute Isochrones
    num_points = 1000
    xy_isochrone1 = tdoa.model.draw_isochrone(x_sensor1, x_sensor2, r12-r11, num_points, 3)
    xy_isochrone2 = tdoa.model.draw_isochrone(x_sensor1, x_sensor2, r22-r21, num_points, 3)
    # This isochrone is a result of erroneous TDOA comparison
    xy_isochrone_error = tdoa.model.draw_isochrone(x_sensor1, x_sensor2, r22-r11, num_points, 3)

    # Draw Figure
    fig9, ax = plt.subplots()

    # Transmitter/Sensor Locations
    plt.plot([x_sensor1[0], x_sensor2[0]], [x_sensor1[1], x_sensor2[1]], 'o', label='Sensors')
    plt.plot([x_source1[0], x_source2[0]], [x_source1[1], x_source2[1]], '^', markersize=8, label='Transmitters')

    # Transmitter/Sensor Labels
    plt.text(x_sensor1[0]-.4, x_sensor1[1]-.1, '$S_1$')
    plt.text(x_sensor2[0]+.1, x_sensor2[1]-.1, '$S_2$')

    plt.text(x_source1[0]+.1, x_source1[1]+.1, '$T_a$')
    plt.text(x_source2[0]+.1, x_source2[1]+.2, '$T_b$')

    # Isochrones
    plt.plot(xy_isochrone1[0], xy_isochrone1[1], '--', label=None)
    plt.plot(xy_isochrone2[0], xy_isochrone2[1], '--', label=None)
    plt.plot(xy_isochrone_error[0], xy_isochrone_error[1], ':', label='False Isochrone')

    # Isochrone Labels
    label_y_location = -1.5
    label_x_offset = .2
    ind1 = np.argmin(np.abs(xy_isochrone1[1]-label_y_location))
    plt.text(xy_isochrone1[0][ind1] + label_x_offset, label_y_location, '$TDOA_a$')

    ind2 = np.argmin(np.abs(xy_isochrone2[1]-label_y_location))
    plt.text(xy_isochrone2[0][ind2] + label_x_offset, label_y_location, '$TDOA_b$')

    ind_error = np.argmin(np.abs(xy_isochrone_error[1]-label_y_location))
    plt.text(xy_isochrone_error[0][ind_error] + label_x_offset, label_y_location, '$TDOA_{err}$')

    # Adjust Plot
    plt.legend(loc='upper left')

    # Remove axis ticks and grid
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if prefix is not None:
        fig9.savefig(prefix + 'fig9.svg')
        fig9.savefig(prefix + 'fig9.png')

    return fig9


def make_figure_10(prefix=None):
    """
    Figure 10 - Plot of false locations

    Ported from MATLAB Code

    Nicholas O'Donoughue
    8 March 2022

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    # Transmitter/Sensor Locations
    x_sensor1 = np.array([0., 0.])
    x_sensor2 = np.array([.8, .2])
    x_sensor3 = np.array([1., 1.])

    x_source1 = np.array([1, 1.5])
    x_source2 = np.array([-.5, 1.5])

    # Ranges
    r11 = utils.geo.calc_range(x_source1, x_sensor1)
    r12 = utils.geo.calc_range(x_source1, x_sensor2)
    r22 = utils.geo.calc_range(x_source2, x_sensor2)
    r23 = utils.geo.calc_range(x_source2, x_sensor3)

    # False Isochrones
    num_points = 1000
    x_isochrone1, y_isochrone1 = tdoa.model.draw_isochrone(x_sensor1, x_sensor2, r12-r11, num_points, 3)
    x_isochrone2, y_isochrone2 = tdoa.model.draw_isochrone(x_sensor2, x_sensor3, r23-r22, num_points, 3)

    # Find False Solution
    isochrone_distance = np.sqrt((x_isochrone1[:, np.newaxis] - x_isochrone2[np.newaxis, :])**2
                                 + (y_isochrone1[:, np.newaxis] - y_isochrone2[np.newaxis, :])**2)
    crossing_index = isochrone_distance.argmin()
    row = int(crossing_index / num_points)
    col = crossing_index % num_points
    x_false_solution = .5*np.array([x_isochrone1[row] + x_isochrone2[col],
                                    y_isochrone1[row] + y_isochrone2[col]])

    # Draw Figure
    fig10, ax = plt.subplots()

    # Emitter/Sensor Locations
    plt.plot([x_sensor1[0], x_sensor2[0], x_sensor3[0]],
             [x_sensor1[1], x_sensor2[1], x_sensor3[1]], 'o', label='Sensors')
    plt.plot([x_source1[0], x_source2[0]],
             [x_source1[1], x_source2[1]], '^', markersize=8, label='Transmitters')

    # Transmitter/Sensor Labels
    plt.text(x_sensor1[0]+.05, x_sensor1[1]+.1, r'$S_1$')
    plt.text(x_sensor2[0]+.05, x_sensor2[1]+.1, r'$S_2$')
    plt.text(x_sensor3[0]+.05, x_sensor3[1]+.1, r'$S_3$')

    plt.text(x_source1[0]+.1, x_source1[1], r'$T_a$')
    plt.text(x_source2[0]+.1, x_source2[1], r'$T_b$')

    # False Isochrones
    plt.plot(x_isochrone1, y_isochrone1, '--', label=None)
    plt.plot(x_isochrone2, y_isochrone2, '--', label=None)

    # Isochrone Labels
    label_x_location = 2
    label_y_offset = .2
    ind1 = np.argmin(np.abs(x_isochrone1-label_x_location))
    ind2 = np.argmin(np.abs(x_isochrone2-label_x_location))

    plt.text(label_x_location, y_isochrone1[ind1] + label_y_offset, r'$TDOA_{12,a}$')
    plt.text(label_x_location, y_isochrone2[ind2] + label_y_offset, r'$TDOA_{23,b}$')

    # False Solution
    plt.plot(x_false_solution[0], x_false_solution[1], '^', markersize=8,
             label='False TDOA Solution')

    # Adjust Display
    plt.legend(loc='lower left')
    plt.ylim([-1.5, 2.5])
    plt.xlim([-2.5, 3.5])

    # Remove axis ticks and grid
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if prefix is not None:
        fig10.savefig(prefix + 'fig10.svg')
        fig10.savefig(prefix + 'fig10.png')

    return fig10
