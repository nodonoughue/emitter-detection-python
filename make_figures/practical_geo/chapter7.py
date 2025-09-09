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


def make_all_figures(close_figs=False, mc_params=None):
    """
    Call all the figure generators for this chapter

    :param close_figs: Boolean flag.  If true, will close all figures after generating them; for batch scripting.
                       Default=False
    :param mc_params: Optional struct to control Monte Carlo trial size
    :return: List of figure handles
    """

    # Reset the random number generator, to ensure reproducibility
    # rng = np.random.default_rng()

    # Find the output directory
    prefix = utils.init_output_dir('practical_geo/chapter7')
    utils.init_plot_style()

    # Generate all figures
    figs1_2 = make_figures_1_2(prefix, mc_params)
    figs3_4 = make_figures_3_4(prefix, mc_params)
    fig5 = make_figure_5(prefix, mc_params)
    fig7 = make_figure_7(prefix)
    fig8 = make_figure_8(prefix, mc_params)
    fig10 = make_figure_10(prefix)

    figs = list(figs1_2) +  list(figs3_4) + list(fig5) + list(fig7) + list(fig8) + list(fig10)
    if close_figs:
        [plt.close(fig) for fig in figs]
        return None
    else:
        # Display the plots
        plt.show()

    return figs


def make_figures_1_2(prefix=None, mc_params=None):
    """
    Figures 7.1 and 7.2 from Example 7.1

    :param prefix: output directory to place generated figure
    :param mc_params: Optional struct to control Monte Carlo trial size
    :return: handle
    """

    if mc_params is not None and 'force_recalc' in mc_params and not mc_params['force_recalc']:
        print('Skipping Figures 7.1 and 7.2 (re-run with mc_params[\'force_recalc\']=True to generate)...')
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

def make_figures_3_4(prefix=None, mc_params=None):
    """
    Figures 7.3 and 7.4 from Example 7.2

    :param prefix: output directory to place generated figure
    :param mc_params: Optional struct to control Monte Carlo trial size
    :return: handle
    """

    if mc_params is not None and 'force_recalc' in mc_params and not mc_params['force_recalc']:
        print('Skipping Figures 7.3 and 7.4 (re-run with mc_params[\'force_recalc\']=True to generate)...')
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


def make_figure_5(prefix=None, mc_params=None):
    """
    Figure 7.5 from Example 7.3

    :param prefix: output directory to place generated figure
    :param mc_params: Optional struct to control Monte Carlo trial size
    :return: handle
    """

    if mc_params is not None and 'force_recalc' in mc_params and not mc_params['force_recalc']:
        print('Skipping Figure 7.5 (re-run with mc_params[\'force_recalc\']=True to generate)...')
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

def make_figure_7(prefix=None):
    """
    Figure 7.7

    :param prefix: output directory to place generated figure
    :return: handle
    """

    print('Generating Figures 7.7...')

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
    fig7a = plt.figure()
    plt.plot(x_tgt[0], x_tgt[1], '^', label='Target')
    tdoa_label='TDOA Sensor'
    for idx in range(num_sensors):
        this_x = np.squeeze(x_tdoa[:, idx])
        this_x_full = np.squeeze(x_tdoa_full[:, idx, :])
        hdl=plt.plot(this_x_full[0], this_x_full[1], label=tdoa_label)
        tdoa_label = None  # clear the label; only the first one gets a legend entry
        plt.scatter(this_x[0], this_x[1], marker='o', color=hdl[0].get_color(), label=None)
    plt.legend(loc='upper left')
    plt.axis('equal')

    # Compute TDOA as a function of time
    tdoa = TDOAPassiveSurveillanceSystem(x=x_tdoa, ref_idx=None, cov=CovarianceMatrix(np.eye(num_sensors)))
    zeta = np.zeros((tdoa.num_measurements, num_t))
    for idx in np.arange(num_t):
        this_x = x_tdoa_full[:, :, idx]
        tdoa.pos = this_x
        zeta[:, idx] = tdoa.measurement(x_source=x_tgt)

    fig7b=plt.figure()
    plt.plot(t, zeta.T)
    plt.legend(['$TDOA_{1,2}$','$TDOA_{1,3}$'])
    plt.grid(True)
    plt.xlabel('Time [s]')
    plt.ylabel('Range Difference Measurement [m]')

    # Output to file
    figs = [fig7a, fig7b]
    if prefix is not None:
        labels = ['fig7a', 'fig7b']
        for fig, label in zip(figs, labels):
            fig.savefig(prefix + label + '.svg')
            fig.savefig(prefix + label + '.png')

    return figs


def make_figure_8(prefix=None, mc_params=None):
    """
    Figure 7.8 from Example 7.4

    :param prefix: output directory to place generated figure
    :param mc_params: Optional struct to control Monte Carlo trial size
    :return: handle
    """

    if mc_params is not None and 'force_recalc' in mc_params and not mc_params['force_recalc']:
        print('Skipping Figure 7.8 (re-run with mc_params[\'force_recalc\']=True to generate)...')
        return None,

    print('Generating Figure 7.8 (Example 7.4)...')

    figs = chapter7.example4()

    # Output to file
    if prefix is not None:
        labels = ['fig8a', 'fig8b']
        if len(labels) != len(figs):
            print('**Error saving figure 7.7; unexpected number of figures returned from Example 7.4.')
        else:
            for fig, label in zip(figs, labels):
                fig.savefig(prefix + label + '.svg')
                fig.savefig(prefix + label + '.png')

    return figs

def make_figure_10(prefix=None, rng=np.random.default_rng()):
    """
    Figure 7.10

    :param prefix: output directory to place generated figure
    :param rng: random number generator
    :return: handle
    """

    print('Generating Figure 7.10...')

    x_tgt = np.array([0, 10e3])

    x_init = np.array([0, 0])
    v_init = np.array([0, 50])
    a_init = np.array([1, 0])  # m/s^2
    t_turn = 100    # s, at this point, flip the acceleration
    t_full = 2*t_turn
    dt = 1
    t_vec = np.arange(start=dt, step=dt, stop=t_full)
    turn_dir = np.sign(t_turn - t_vec)
    a_aoa = np.concatenate((np.array([[0], [0]]), a_init[:, np.newaxis] * turn_dir[np.newaxis, :]), axis=1)
    v_aoa = v_init[:, np.newaxis] + np.cumsum(a_aoa * dt, axis=1)
    x_aoa = x_init[:, np.newaxis] + np.cumsum(v_aoa * dt, axis=1)

    theta_unc = 5 # +/- 5 degree uncertainty interval
    cov_df = CovarianceMatrix([(theta_unc*_deg2rad)**2])
    aoa = DirectionFinder(x=x_aoa, do_2d_aoa=False, cov=cov_df)

    fig10a=plt.figure()
    hdl_traj = plt.plot(x_aoa[0], x_aoa[1], label='Sensor Trajectory')

    # Draw bearings at time markers
    idx_set = [1, np.floor(len(t_vec)/2).astype(int), len(t_vec)]
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
        vertex_theta = np.pi+np.arange(start=0, step=2*np.pi/num_pts, stop=2*np.pi)-this_bng
        marker_x = this_x[0] + marker_radius*np.cos(vertex_theta)
        marker_y = this_x[1] + marker_radius*np.sin(vertex_theta)

        # Draw Icon
        plt.fill(marker_x, marker_y, edgecolor='k', facecolor=color, label=None)

        # Draw LOB with uncertainty
        aoa.pos = this_x
        psi = aoa.measurement(x_source=x_tgt)
        psi_high = psi + theta_unc * _deg2rad
        psi_low =  psi - theta_unc * _deg2rad

        # Generate all the LOBs at once; return will be 2 x 2 x num_sensors x num_cases. There's only one sensor
        xy_lobs = aoa.draw_lobs(zeta=np.concatenate((psi[:, np.newaxis],
                                                     psi_high[:, np.newaxis],
                                                     psi_low[:, np.newaxis]), axis=1), x_source=x_tgt, scale=5)
        xy_lob = xy_lobs[:, :, 0, 0]
        xy_lob_high = xy_lobs[:, :, 0, 1] # aoa.draw_lobs(zeta=psi_high, x_source=x_tgt, scale=5)[0,:,:,0]
        xy_lob_low = xy_lobs[:, :, 0, 2] #aoa.draw_lobs(zeta=psi_low, x_source=x_tgt, scale=5)[0,:,:,0]

        # Make a patch; unlike MATLAB, we don't have to close it
        lob_fill = np.concatenate((xy_lob_high,xy_lob_low[:, [-1]]), axis=1)
        fill_patch = plt.Polygon(lob_fill.T, linestyle='--', edgecolor='k', facecolor=color, alpha=.2,
                                 label=label_fill)
        fig10a.gca().add_patch(fill_patch)
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

    lower = np.sqrt(aoa.cov.cov)  # the covariance matrix is a scalar; so just take the square root

    cep_vec = np.zeros_like(t_vec)
    for idx in np.arange(t_vec.size):
        this_x_aoa = x_aoa[:,idx]
        aoa.pos = this_x_aoa

        z_fun = aoa.measurement
        h_fun = lambda x: aoa.jacobian(x).T

        # ToDo: Make a noisy_measurement function for PSS classes and use it
        this_psi = aoa.measurement(x_tgt) + lower @ rng.standard_normal(1)

        this_x, this_p = tracker.ekf_update(x_prev, p_prev, this_psi, aoa.cov.cov, z_fun, h_fun)
        cep_vec[idx] = utils.errors.compute_cep50(this_p)

        x_prev = this_x
        p_prev = this_p

    fig10b=plt.figure()
    plt.semilogy(t_vec, cep_vec)
    plt.xlabel('Time [s]')
    plt.ylabel('$CEP_{50}$ [m]')

    # Output to file
    figs = [fig10a, fig10b]
    if prefix is not None:
        labels = ['fig10a', 'fig10b']
        if len(labels) != len(figs):
            print('**Error saving figure 7.10; unexpected number of figures.')
        else:
            for fig, label in zip(figs, labels):
                fig.savefig(prefix + label + '.svg')
                fig.savefig(prefix + label + '.png')

    return figs


if __name__ == "__main__":
    make_all_figures(close_figs=False, mc_params={'force_recalc': True})
