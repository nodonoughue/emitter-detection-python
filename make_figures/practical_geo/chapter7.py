"""
Draw Figures - Chapter 7

This script generates all the figures that appear in Chapter 7 of the textbook.

Ported from MATLAB Code

Nicholas O'Donoughue
28 June 2025
"""

from triang import DirectionFinder
from tdoa import TDOAPassiveSurveillanceSystem
import utils
from utils.covariance import CovarianceMatrix
from utils import tracker
import matplotlib.pyplot as plt
import numpy as np

from examples.practical_geo import chapter7

_rad2deg = utils.unit_conversions.convert(1, "rad", "deg")
_deg2rad = utils.unit_conversions.convert(1, "deg", "rad")


def make_all_figures(close_figs=False, force_recalc=False):
    """
    Call all the figure generators for this chapter

    :param close_figs: Boolean flag.  If true, will close all figures after generating them; for batch scripting.
                       Default=False
    :param force_recalc: optional flag (default=True), if False then the example does not run
    :return: List of figure handles
    """

    # Reset the random number generator, to ensure reproducibility
    # rng = np.random.default_rng()

    # Find the output directory
    prefix = utils.init_output_dir('practical_geo/chapter7')
    utils.init_plot_style()

    # Generate all figures
    figs1_2 = make_figures_1_2(prefix, force_recalc)
    figs3_4 = make_figures_3_4(prefix, force_recalc)
    fig5 = make_figure_5(prefix, force_recalc)
    fig6 = make_figure_6(prefix)
    fig7 = make_figure_7(prefix)
    fig8 = make_figure_8(prefix)

    figs = list(figs1_2) +  list(figs3_4) + list(fig5) + list(fig6) + list(fig7) + list(fig8)
    if close_figs:
        [plt.close(fig) for fig in figs]
        return None
    else:
        # Display the plots
        plt.show()

    return figs


def make_figures_1_2(prefix=None, force_recalc=False):
    """
    Figures 7.1 and 7.2 from Example 7.1

    :param prefix: output directory to place generated figure
    :param force_recalc: optional flag (default=True), if False then the example does not run
    :return: handle
    """

    if not force_recalc:
        print('Skipping Figures 7.1 and 7.2 (re-run with force_recalc=True to generate)...')
        return None,

    print('Generating Figures 7.1 and 7.2 (Example 7.1)...')

    figs = chapter7.example1()

    # Output to file
    if prefix is not None:
        labels = ['fig1', 'fig2a', 'fig2b']
        if len(labels) != len(figs):
            print('**Error saving figure 7.1 and 7.2; unexpected number of figures returned from Example 7.1.')
        else:
            for fig, label in zip(figs, labels):
                fig.savefig(prefix + label + '.svg')
                fig.savefig(prefix + label + '.png')

    return figs

def make_figures_3_4(prefix=None, force_recalc=False):
    """
    Figures 7.3 and 7.4 from Example 7.2

    :param prefix: output directory to place generated figure
    :param force_recalc: optional flag (default=True), if False then the example does not run
    :return: handle
    """

    if not force_recalc:
        print('Skipping Figures 7.3 and 7.4 (re-run with force_recalc=True to generate)...')
        return None,

    print('Generating Figures 7.3 and 7.4 (Example 7.2)...')

    figs = chapter7.example2()

    # Output to file
    if prefix is not None:
        labels = ['fig3', 'fig4a', 'fig4b']
        if len(labels) != len(figs):
            print('**Error saving figure 7.3 and 7.4; unexpected number of figures returned from Example 7.2.')
        else:
            for fig, label in zip(figs, labels):
                fig.savefig(prefix + label + '.svg')
                fig.savefig(prefix + label + '.png')

    return figs


def make_figure_5(prefix=None, force_recalc=False):
    """
    Figure 7.5 from Example 7.3

    :param prefix: output directory to place generated figure
    :param force_recalc: optional flag (default=True), if False then the example does not run
    :return: handle
    """

    if not force_recalc:
        print('Skipping Figure 7.5 (re-run with force_recalc=True to generate)...')
        return None,

    print('Generating Figures 7.5 (Example 7.3)...')

    figs = chapter7.example3()

    # Output to file
    if prefix is not None:
        labels = ['fig5a', 'fig5b']
        if len(labels) != len(figs):
            print('**Error saving figure 7.5; unexpected number of figures returned from Example 7.3.')
        else:
            for fig, label in zip(figs, labels):
                fig.savefig(prefix + label + '.svg')
                fig.savefig(prefix + label + '.png')

    return figs

def make_figure_6(prefix=None):
    """
    Figure 7.6

    :param prefix: output directory to place generated figure
    :return: handle
    """

    print('Generating Figures 7.6...')

    # Illustration of TDOA changes over time
    x_tgt = np.array([0, 0])
    x_tdoa = np.array([[-5, 5, 15],
                       [-50, -40, -50]])*1e3
    v_tdoa = np.array([[0, 0, 0],
                       [1e3, 1e3, 1e3]])

    t = np.arange(1000)/10  # 0 to 100 in increments of .1

    _, num_sensors = utils.safe_2d_shape(x_tdoa)
    num_t = len(t)

    x_tdoa_full = x_tdoa[:, :, np.newaxis] + v_tdoa[:, :, np.newaxis] * np.reshape(t, shape=(1, 1, num_t))

    # Plot Geometry
    fig6a = plt.figure()
    plt.plot(x_tgt[0], x_tgt[1], '^', label='Target')
    tdoa_label='TDOA Sensor'
    for this_x, this_x_full in zip(x_tdoa, x_tdoa_full):
        hdl=plt.plot(this_x_full[0], this_x_full[1], label=tdoa_label)
        tdoa_label = None  # clear the label; only the first one gets a legend entry
        plt.scatter(this_x[0], this_x[1], marker='o', color=hdl[0].get_color(), label=None)
    plt.legend(loc='upper left')

    # Compute TDOA as a function of time
    tdoa = TDOAPassiveSurveillanceSystem(x=x_tdoa, ref_idx=None, cov=CovarianceMatrix(np.eye(num_sensors)))
    zeta = np.zeros((tdoa.num_measurements, num_t))
    for idx in np.arange(num_t):
        this_x = x_tdoa_full[:, :, idx]
        tdoa.pos = this_x
        zeta[:, idx] = tdoa.measurement(x_source=x_tgt)

    fig6b=plt.figure()
    plt.plot(t, zeta)
    plt.legend('TDOA_{1,2}','TDOA_{1,3}')
    plt.grid(True)
    plt.xlabel('Time [s]')
    plt.ylabel('Range Difference Measurement [m]')

    # Output to file
    figs = [fig6a, fig6b]
    if prefix is not None:
        labels = ['fig6a', 'fig6b']
        for fig, label in zip(figs, labels):
            fig.savefig(prefix + label + '.svg')
            fig.savefig(prefix + label + '.png')

    return figs


def make_figure_7(prefix=None, force_recalc=False):
    """
    Figure 7.7 from Example 7.4

    :param prefix: output directory to place generated figure
    :param force_recalc: optional flag (default=True), if False then the example does not run
    :return: handle
    """

    if not force_recalc:
        print('Skipping Figure 7.7 (re-run with force_recalc=True to generate)...')
        return None,

    print('Generating Figure 7.7 (Example 7.4)...')

    figs = chapter7.example4()

    # Output to file
    if prefix is not None:
        labels = ['fig7a', 'fig7b']
        if len(labels) != len(figs):
            print('**Error saving figure 7.7; unexpected number of figures returned from Example 7.4.')
        else:
            for fig, label in zip(figs, labels):
                fig.savefig(prefix + label + '.svg')
                fig.savefig(prefix + label + '.png')

    return figs

def make_figure_8(prefix=None, rng=np.random.default_rng()):
    """
    Figure 7.8

    :param prefix: output directory to place generated figure
    :param rng: random number generator
    :return: handle
    """

    print('Generating Figure 7.8...')

    x_tgt = np.array([0, 10e3])

    x_init = np.array([0, 0])
    v_init = np.array([0, 50])
    a_init = np.array([1, 0])  # m/s^2
    t_turn = 100    # s, at this point, flip the acceleration
    t_full = 2*t_turn
    dt = 1
    t_vec = np.arange(start=dt, step=dt, stop=t_full)
    a_aoa = np.concatenate((np.array([0, 0]), a_init * ((t_vec < t_turn) - (t_vec > t_turn))), axis=1)
    v_aoa = v_init + np.cumsum(a_aoa * dt, axis=1)
    x_aoa = x_init + np.cumsum(v_aoa * dt, axis=1)

    theta_unc = 5 # +/- 5 degree uncertainty interval
    cov_df = CovarianceMatrix((theta_unc*_deg2rad)**2)
    aoa = DirectionFinder(x=x_aoa, do_2d_aoa=False, cov=cov_df)

    fig8a=plt.figure()
    hdl_traj = plt.plot(x_aoa[0], x_aoa[1], label='Sensor Trajectory')

    # Draw bearings at time markers
    idx_set = [1, np.floor(len(t_vec)/2), len(t_vec)]
    label_fill = 'Uncertainty Interval'
    label_lob = 'LOB'
    color = hdl_traj[0].get_color()
    for ii, this_idx in enumerate(idx_set):
        # Grab Position
        this_x = x_aoa[:, this_idx]
        this_v = v_aoa[:,this_idx]
        this_bng = np.atan2(this_v[1], this_v[0])

        # Make a triangular patch
        marker_radius = 250
        num_pts = 3
        vertex_theta = np.pi+np.arange(start=0, step=2*np.pi/num_pts, stop=2*np.pi*(num_pts-1)/num_pts)-this_bng
        marker_x = this_x[0] + marker_radius*np.cos(vertex_theta)
        marker_y = this_x[1] + marker_radius*np.sin(vertex_theta)

        # Draw Icon
        plt.scatter(marker_x, marker_y, s=10, marker='^', edgecolor='k', facecolor=color, label=None)

        # Draw LOB with uncertainty
        aoa.pos = this_x
        psi = aoa.measurement(x_source=x_tgt)
        psi_high = psi + theta_unc * _deg2rad
        psi_low =  psi - theta_unc * _deg2rad

        xy_lob = aoa.draw_lobs(zeta=psi, x_source=x_tgt, scale=5)
        xy_lob_high = aoa.draw_lobs(zeta=psi_high, x_source=x_tgt, scale=5)
        xy_lob_low = aoa.draw_lobs(zeta=psi_low, x_source=x_tgt, scale=5)

        # Make a patch; unlike MATLAB, we don't have to close it
        lob_fill = np.concatenate((xy_lob_high,xy_lob_low[:, -1]), axis=1)
        fill_patch = plt.Polygon(lob_fill, linestyle='--', edgecolor='k', facecolor=color, alpha=.2,
                                 label=label_fill)
        fig8a.gca().add_patch(fill_patch)
        plt.plot(xy_lob[0], xy_lob[1], '-.', label=label_lob,color=color)
        label_fill = None
        label_lob = None

    plt.plot(x_tgt[0], x_tgt[1],'o', label='Target')
    plt.legend(loc='lower right')
    plt.ylim([-1e3, 11e3])
    plt.xlim([-7.5e3, 13.5e3])

    # Model CEP over time

    # Since we can't do geolocation with the first measurement, let's
    # initialize the track manually
    x_prev = np.array([0, 1e3])
    p_prev = np.diag([1e3, 10e3])**2

    cep_vec = np.zeros_like(t_vec)
    for idx in np.arange(t_vec):
        this_x_aoa = x_aoa[:,idx]
        aoa.pos = this_x_aoa

        z_fun = aoa.measurement
        h_fun = lambda x: aoa.jacobian(x).T

        # ToDo: Make a noisy_measurement function for PSS classes and use it
        this_psi = aoa.measurement(x_tgt) + aoa.cov.lower @ rng.standard_normal(1)

        this_x, this_p = tracker.ekf_update(x_prev, p_prev, this_psi, aoa.cov.cov, z_fun, h_fun)
        cep_vec[idx] = utils.errors.compute_cep50(this_p)

        x_prev = this_x
        p_prev = this_p

    fig8b=plt.figure()
    plt.plot(t_vec, cep_vec/1e3)
    plt.xlabel('Time [s]')
    plt.ylabel('$CEP_{50}$ [km]')

    # Output to file
    figs = [fig8a, fig8b]
    if prefix is not None:
        labels = ['fig8', 'fig9']
        if len(labels) != len(figs):
            print('**Error saving figure 7.8; unexpected number of figures.')
        else:
            for fig, label in zip(figs, labels):
                fig.savefig(prefix + label + '.svg')
                fig.savefig(prefix + label + '.png')

    return figs