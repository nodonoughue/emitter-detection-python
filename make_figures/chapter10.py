"""
Draw Figures - Chapter 10

This script generates all the figures that appear in Chapter 10 of the textbook.

Ported from MATLAB Code

Nicholas O'Donoughue
19 May 2021
"""

import utils
import matplotlib.pyplot as plt
import numpy as np
import triang
from examples import chapter10


def make_all_figures(close_figs=False, force_recalc=False):
    """
    Call all the figure generators for this chapter

    :close_figs: Boolean flag.  If true, will close all figures after generating them; for batch scripting.
                 Default=False
    :force_recalc: If set to False, will skip any figures that are time-consuming to generate.
    :return: List of figure handles
    """

    # Find the output directory
    prefix = utils.init_output_dir('chapter10')
    utils.init_plot_style()

    # Random Number Generator
    rng = np.random.default_rng(0)

    # Colormap
    cmap = plt.get_cmap("tab10")

    # Generate all figures
    fig1 = make_figure_1(prefix, cmap)
    fig2 = make_figure_2(prefix, cmap)
    fig3 = make_figure_3(prefix)
    fig4 = make_figure_4(prefix)
    fig5a, fig5b = make_figure_5(prefix, rng, force_recalc)
    fig6 = make_figure_6(prefix)
    fig7 = make_figure_7(prefix)

    figs = [fig1, fig2, fig3, fig4, fig5a, fig5b, fig6, fig7]

    if close_figs:
        for fig in figs:
            plt.close(fig)

        return None
    else:
        plt.show()

        return figs


def make_figure_1(prefix=None, cmap=None):
    """
    Figure 1, LOB Intersection
    Compute an isochrone between sensors 1 and 2, then draw an AOA slice from
    sensor 3

    Ported from MATLAB Code

    Nicholas O'Donoughue
    19 May 2021

    :param prefix: output directory to place generated figure
    :param cmap: colormap
    :return: figure handle
    """

    if cmap is None:
        cmap = plt.get_cmap("tab10")

    print('Generating Figure 10.1...')

    # Initialize Detector/Source Locations
    x_sensor = np.array([[0, 1, -1], [0, 1, 0]])
    x_source = np.array([[.1], [.9]])

    # Error Values
    ang_err = 5*np.pi/180

    # Compute Ranges and Angles
    range_act = utils.geo.calc_range(x_sensor, x_source)
    psi_act = triang.model.measurement(x_sensor, x_source)

    # Start first figure; geometry
    fig1 = plt.figure()

    # Geometry
    for idx_sensor, this_psi in enumerate(psi_act):
        # Vector from sensor to source
        this_x = np.expand_dims(x_sensor[:, idx_sensor], axis=1)
        this_range = range_act[idx_sensor]
        this_color = cmap(idx_sensor)

        # Find AOA
        lob = this_x + np.array([[0, np.cos(this_psi)], [0, np.sin(this_psi)]]) * 5 * this_range
        lob_err1 = this_x + np.array([[0, np.cos(this_psi + ang_err)], [0, np.sin(this_psi + ang_err)]])*5*this_range
        lob_err0 = this_x + np.array([[0, np.cos(this_psi - ang_err)], [0, np.sin(this_psi - ang_err)]])*5*this_range
        lob_fill1 = np.concatenate((lob_err1, np.fliplr(lob_err0), np.expand_dims(lob_err1[:, 0], axis=1)), axis=1)

        # Plot the Uncertainty Interval
        if idx_sensor == 0:
            this_uc_label = 'Uncertainty Interval'
            this_lob_label = 'AOA Solution'
        else:
            this_uc_label = None
            this_lob_label = None

        plt.fill(lob_fill1[0, :], lob_fill1[1, :], linestyle='--', alpha=.1, edgecolor='k',
                 facecolor=this_color, label=this_uc_label)

        # Plot the LOB
        plt.plot(lob[0, :], lob[1, :], linestyle='-', color=this_color, label=this_lob_label)

    # Position Markers
    plt.scatter(x_sensor[0, :], x_sensor[1, :], marker='o', label='Sensors', zorder=3)

    # Plot the points and lobs
    plt.scatter(x_source[0], x_source[1], marker='^', label='Transmitter', zorder=3)

    # Position Labels
    for idx_sensor in np.arange(np.shape(x_sensor)[1]):
        plt.text(x_sensor[0, idx_sensor], x_sensor[1, idx_sensor], '$S_{:d}$'.format(idx_sensor))

    # Adjust Axes
    plt.legend(loc='lower right')
    plt.ylim([-.5, 1.5])
    plt.xlim([-1.5, 2])
    plt.axis('off')

    if prefix is not None:
        fig1.savefig(prefix + 'fig1.png')
        fig1.savefig(prefix + 'fig1.svg')

    return fig1


def make_figure_2(prefix=None, cmap=None):
    """
    Figure 2 - This time with angles

    Ported from MATLAB Code

    Nicholas O'Donoughue
    19 May 2021

    :param prefix: output directory to place generated figure
    :param cmap: colormap
    :return: figure handle
    """

    if cmap is None:
        cmap = plt.get_cmap("tab10")

    print('Generating Figure 10.2...')

    # Initialize Detector/Source Locations
    x_sensor = np.array([[0, 1, -1], [0, 1, 0]])
    x_source = np.array([[.1], [.9]])

    # Error Values
    ang_err = 5*np.pi/180

    # Compute Ranges and Angles
    range_act = utils.geo.calc_range(x_sensor, x_source)
    psi_act = triang.model.measurement(x_sensor, x_source)

    # Start first figure; geometry
    fig2 = plt.figure()

    # Geometry
    for idx_sensor, this_psi in enumerate(psi_act):
        # Vector from sensor to source
        this_x = np.expand_dims(x_sensor[:, idx_sensor], axis=1)
        this_range = range_act[idx_sensor]
        this_color = cmap(idx_sensor)

        # Find AOA
        lob = this_x + np.array([[0, np.cos(this_psi)], [0, np.sin(this_psi)]]) * 5 * this_range
        lob_err1 = this_x + np.array([[0, np.cos(this_psi + ang_err)], [0, np.sin(this_psi + ang_err)]])*5*this_range
        lob_err0 = this_x + np.array([[0, np.cos(this_psi - ang_err)], [0, np.sin(this_psi - ang_err)]])*5*this_range
        lob_fill1 = np.concatenate((lob_err1, np.fliplr(lob_err0), np.expand_dims(lob_err1[:, 0], axis=1)), axis=1)

        # Plot the Uncertainty Interval
        if idx_sensor == 0:
            this_uc_label = 'Uncertainty Interval'
            this_lob_label = 'AOA Solution'
        else:
            this_uc_label = None
            this_lob_label = None

        plt.fill(lob_fill1[0, :], lob_fill1[1, :], linestyle='--', alpha=.1, edgecolor='k',
                 facecolor=this_color, label=this_uc_label)

        # Plot the LOB
        plt.plot(lob[0, :], lob[1, :], linestyle='-', color=this_color, label=this_lob_label)

        # Plot the Angle Marker
        angle_rad = 0.2
        rad_axis = this_x + np.asarray([[0, 3*angle_rad], [0, 0]])
        plt.plot(rad_axis[0, :], rad_axis[1, :], color=this_color, linewidth=.5)
        th_vec = np.linspace(start=0, stop=this_psi, num=100)
        arc = this_x + angle_rad*np.stack((np.cos(th_vec), np.sin(th_vec)))
        plt.plot(arc[0, :], arc[1, :], color=this_color, linestyle='-', linewidth=.5)
        plt.text(this_x[0] + angle_rad, this_x[1] + .6*angle_rad*np.sign(this_psi), r"$\psi_{:d}$".format(idx_sensor))

    # Position Markers
    plt.scatter(x_sensor[0, :], x_sensor[1, :], marker='o', label='Sensors', zorder=3)

    # Plot the points and lobs
    plt.scatter(x_source[0], x_source[1], marker='^', label='Transmitter', zorder=3)

    # Position Labels
    for idx_sensor in np.arange(np.shape(x_sensor)[1]):
        plt.text(x_sensor[0, idx_sensor], x_sensor[1, idx_sensor], '$S_{:d}$'.format(idx_sensor))

    # Adjust Axes
    plt.legend(loc='lower right')
    plt.ylim([-.5, 1.5])
    plt.xlim([-1.5, 2])
    plt.axis('off')

    if prefix is not None:
        fig2.savefig(prefix + 'fig2.png')
        fig2.savefig(prefix + 'fig2.svg')

    return fig2


def make_figure_3(prefix=None):
    """
    Figure 3 - Geometric Solutions

    Ported from MATLAB Code

    Nicholas O'Donoughue
    19 May 2021

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    print('Generating Figure 10.3...')

    # Define triangle
    x_sensor = np.array([[0, 1, .5], [0, 1, 1.5]])
    num_dims, num_sensors = np.shape(x_sensor)

    # Initialize all 3 plots
    fig3, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)

    for this_ax in (ax0, ax1):
        this_ax.plot(np.append(x_sensor[0, :], x_sensor[0, 0]),
                     np.append(x_sensor[1, :], x_sensor[1, 0]), label=None)

    # First subplot - midpoint method

    # Compute midpoints
    midpoints = .5 * (np.roll(x_sensor, shift=1, axis=1) + np.roll(x_sensor, shift=-1, axis=1))

    for idx_sensor in np.arange(num_sensors):
        this_vertex = x_sensor[:, idx_sensor]
        this_midpoint = midpoints[:, idx_sensor]

        ax0.plot([this_vertex[0], this_midpoint[0]],
                 [this_vertex[1], this_midpoint[1]], label=None, linewidth=0.5)

    # Second subplot -- Angle Bisector method

    # Compute angle bisectors
    dx_fwd = np.roll(x_sensor, shift=-1, axis=1) - x_sensor
    th_fwd = np.arctan2(dx_fwd[1, :], dx_fwd[0, :])
    dx_back = np.roll(x_sensor, shift=1, axis=1) - x_sensor
    th_back = np.arctan2(dx_back[1, :], dx_back[0, :])
    th_bisector = .5*(th_fwd+th_back)
    flipped = th_bisector < th_fwd
    th_bisector[flipped] = th_bisector[flipped] + np.pi

    # Find Intersection on Opposite Edge
    rng_fwd = np.sqrt(np.sum(np.abs(dx_fwd)**2, axis=0))
    rng_opp = np.roll(rng_fwd, shift=-1)
    rng_rev = np.roll(rng_fwd, shift=1)

    rng_bisector = np.sqrt(rng_fwd*rng_rev*((rng_fwd+rng_rev)**2-rng_opp**2) / (rng_fwd+rng_rev)**2)

    # Plot
    for idx_sensor in np.arange(num_sensors):
        this_x = x_sensor[:, idx_sensor]
        this_rng = rng_bisector[idx_sensor]
        this_th = th_bisector[idx_sensor]

        ax1.plot(this_x[0] + this_rng*np.array([0, np.cos(this_th)]),
                 this_x[1] + this_rng*np.array([0, np.sin(this_th)]),
                 linewidth=.5, label=None)

    ax0.axis('off')
    ax1.axis('off')

    if prefix is not None:
        fig3.savefig(prefix + 'fig3.png')
        fig3.savefig(prefix + 'fig3.svg')

    return fig3


def make_figure_4(prefix=None):
    """
    Figure 4 - Example Results

    Ported from MATLAB Code

    Nicholas O'Donoughue
    19 May 2021

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    print('Generating Figure 10.4...')

    # Test LOB intersection
    x_sensor = np.array([[0, 1, 2], [2, 0, 0]])
    n_dims, num_sensors = np.shape(x_sensor)

    psi = np.array([65, 75, 95])*np.pi/180
    this_range = 20

    # Error checking, make sure num_sensors matches the length of psi
    assert num_sensors == np.size(psi), 'Number of sensor positions does not equal number of AOA measurements'

    # Plot the points and lobs
    fig4 = plt.figure()
    # Geometry
    for idx_sensor, this_psi in enumerate(psi):
        # Vector from sensor to source
        this_x = np.expand_dims(x_sensor[:, idx_sensor], axis=1)
        this_color = 'k'

        # Find AOA
        lob = this_x + np.array([[0, np.cos(this_psi)], [0, np.sin(this_psi)]]) * 5 * this_range

        # Plot the Uncertainty Interval
        if idx_sensor == 0:
            this_lob_label = 'AOA Solution'
        else:
            this_lob_label = None

        # Plot the LOB
        plt.plot(lob[0, :], lob[1, :], linestyle='-', color=this_color, label=this_lob_label)
        plt.scatter(lob[0, 0], lob[1, 0], marker='.', color=this_color, label=None)

    # Position Labels
    for idx_sensor in np.arange(np.shape(x_sensor)[1]):
        plt.text(x_sensor[0, idx_sensor]+.1, x_sensor[1, idx_sensor], '$S_{:d}$'.format(idx_sensor))

    # Look for intersections
    x_centroid = triang.solvers.centroid(x_sensor, psi)
    x_angle_bisector = triang.solvers.angle_bisector(x_sensor, psi)

    plt.plot(x_centroid[0], x_centroid[1], marker='x', label='Centroid')
    plt.plot(x_angle_bisector[0], x_angle_bisector[1], marker='+', label='Incenter')
    plt.legend(loc='upper left')
    plt.xlim([-1.5, 5.5])
    plt.ylim([-.5, 10.5])
    plt.axis('off')

    if prefix is not None:
        fig4.savefig(prefix + 'fig4.png')
        fig4.savefig(prefix + 'fig4.svg')

    return fig4


def make_figure_5(prefix=None, rng=np.random.default_rng(), force_recalc=True):
    """
    Figure 5 - Example Solution

    Ported from MATLAB Code

    Nicholas O'Donoughue
    19 May 2021

    :param prefix: output directory to place generated figure
    :param rng: random number generator
    :param force_recalc: optional flag (default=True), if False then the example does not run
    :return fig5a: figure handle for first subfigure
    :return fig5b: figure handle for second subfigure
    """

    if not force_recalc:
        print('Skipping Figure 10.5... (re-run with force_recalc=True to generate)')
        return None, None

    print('Generating Figure 10.5 (a and b)...')

    fig5a, fig5b = chapter10.example1(rng)

    if prefix is not None:
        plt.figure(fig5a)
        fig5a.savefig(prefix + 'fig5a.png')
        fig5a.savefig(prefix + 'fig5a.svg')

        plt.figure(fig5b)
        fig5b.savefig(prefix + 'fig5b.png')
        fig5b.savefig(prefix + 'fig5b.svg')

    return fig5a, fig5b


def make_figure_6(prefix=None):
    """
    Figure 6 - 2 Sensor Configuration

    Ported from MATLAB Code

    Nicholas O'Donoughue
    19 May 2021

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    print('Generating Figure 10.6 (using Example 10.2)...')

    fig6 = chapter10.example2()

    if prefix is not None:
        fig6.savefig(prefix + 'fig6.png')
        fig6.savefig(prefix + 'fig6.svg')

    return fig6


def make_figure_7(prefix=None):
    """
    Figure 7 - 3 Sensor Configuration

    Ported from MATLAB Code

    Nicholas O'Donoughue
    19 May 2021

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    print('Generating Figure 10.7 (using Example 10.3)...')

    fig7 = chapter10.example3()

    if prefix is not None:
        fig7.savefig(prefix + 'fig7.png')
        fig7.savefig(prefix + 'fig7.svg')

    return fig7
