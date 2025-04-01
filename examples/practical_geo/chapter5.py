import numpy as np
import matplotlib.pyplot as plt
import scipy

import tdoa.model
import utils
import hybrid
import time
from utils.covariance import CovarianceMatrix
import triang

_rad2deg = utils.unit_conversions.convert(1, "rad", "deg")
_deg2rad = utils.unit_conversions.convert(1, "deg", "rad")


def run_all_examples():
    """
    Run all chapter 5 examples and return a list of figure handles

    :return figs: list of figure handles
    """

    return list(example1()) + list(example2()) + list(example3()) + list(example4()) + list(example5())


def example1(do_mod_cov=False):
    """
    Executes Example 5.1.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    1 April 2025

    :param do_mod_cov: boolean; if True then the covariance matrix is modified as discussed in Video 5.1.
    :return: figure handle to generated graphic
    """

    # Set up sensors
    x_aoa = np.array([[-2, 2],
                      [0, 0]])
    _, num_aoa = utils.safe_2d_shape(x_aoa)

    # Define received signals and covariance matrix
    psi = np.array([80, 87]) * _deg2rad
    cov = CovarianceMatrix(np.diag([0.1, 1.0]))  # second sensor is 10x worse than first sensor
    if do_mod_cov:
        cov = CovarianceMatrix(np.diag([0.05, 0.2]))
    x_init = np.array([0, 1])  # initial guess

    # Plot the scenario
    def _make_laydown():
        this_fig = plt.figure()
        hdl0 = plt.scatter(x_aoa[0, 0], x_aoa[1, 0], marker='^', label='AOA Sensors')
        hdl1 = plt.scatter(x_aoa[0, 1], x_aoa[1, 1], marker='^', label=None)
        plt.grid(True)

        # LOBs
        xy_lob = triang.model.draw_lob(x_aoa[:, 0], psi[0], scale=35)
        plt.plot(xy_lob[0], xy_lob[1], color=hdl0.get_edgecolor())
        xy_lob = triang.model.draw_lob(x_aoa[:, 1], psi[1], scale=35)
        plt.plot(xy_lob[0], xy_lob[1], color=hdl1.get_edgecolor())

        return this_fig

    # Make the first figure; just the laydown
    figs = [_make_laydown()]
    plt.legend()

    # Gradient Descent Solution; Unconstrained
    gd_args = {'x_sensor':x_aoa, 'psi': psi, 'cov': cov, 'x_init': x_init}
    x_gd, x_gd_full = triang.solvers.gradient_descent(**gd_args)

    # Gradient Descent Solution; Constrained
    y_soln = 25
    a, _ = utils.constraints.fixed_cartesian('y', y_soln)
    constraint_arg = {'eq_constraints': [a]}
    x_gd_const, x_gd_full_const = triang.solvers.gradient_descent(**gd_args, **constraint_arg)

    # Report Output and Generate Figure
    print('Gradient Descent Solvers...')
    print('Unconstrained Solution: ({:.2f}, {:.2f})'.format(x_gd[0], x_gd[1]))
    print('Constrained Solution: ({:.2f}, {:.2f})'.format(x_gd_const[0], x_gd_const[1]))

    figs.append(_make_laydown())
    plt.plot(x_gd_full[0], x_gd_full[1], '--o', markevery=[-1], label='GD (Unconstrained)')
    plt.plot(x_gd_full_const[0], x_gd_full_const[1], '--s', markevery=[-1], label='GD (Constrained')
    plt.legend()

    ## Do it again with LS
    x_ls, x_ls_full = triang.solvers.least_square(**gd_args)
    x_ls_const, x_ls_full_const = triang.solvers.least_square(**gd_args, **constraint_arg)

    # Report Output and Generate Figure
    print('Least Squares Solvers...')
    print('Unconstrained Solution: ({:.2f}, {:.2f})'.format(x_ls[0], x_ls[1]))
    print('Constrained Solution: ({:.2f}, {:.2f})'.format(x_ls_const[0], x_ls_const[1]))

    figs.append(_make_laydown())
    plt.plot(x_gd_full[0], x_gd_full[1], '--o', markevery=[-1], label='GD (Unconstrained)')
    plt.plot(x_gd_full_const[0], x_gd_full_const[1], '--s', markevery=[-1], label='GD (Constrained')
    plt.plot(x_ls_full[0], x_ls_full[1], '--^', markevery=[-1], label='LS (Unconstrained)')
    plt.plot(x_ls_full_const[0], x_ls_full_const[1], '--v', markevery=[-1], label='LS (Constrained')
    plt.legend()

    return figs


def example2():
    """
    Executes Example 5.2.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    7 February 2025

    :param do_mod_pos: boolean, if True then the
    :return: figure handle to generated graphic
    """

    def _ex2_inner(this_x_tdoa, this_x_init, title=None):
        # Kernel of example 2; called multiple times with different sensor positions
        _, num_tdoa = utils.safe_2d_shape(this_x_tdoa)

        # Define target position
        tgt_alt = 100.
        x_tgt = np.array([-10e3, 40e3, tgt_alt])

        # Sensor Accuracy
        time_err = 1e-7
        roa_var = (utils.constants.speed_of_light*time_err)**2
        cov = CovarianceMatrix(np.eye(num_tdoa)*roa_var)

        # Measurement and Noise
        ref_idx = num_tdoa-1
        z = tdoa.model.measurement(x_sensor=this_x_tdoa, x_source=x_tgt, ref_idx=ref_idx)
        noise = cov.lower @ np.random.randn(num_tdoa)
        cov_z = cov.resample(ref_idx=ref_idx)
        noise_z = utils.resample_noise(noise, ref_idx=ref_idx)
        zeta = z + noise_z

        # Solve for Target Position
        gd_args = {'x_sensor': this_x_tdoa, 'rho': zeta, 'cov': cov_z, 'x_init': this_x_init}
        x_gd, x_gd_full = tdoa.solvers.gradient_descent(**gd_args)

        a, _ = utils.constraints.fixed_alt(alt_val=tgt_alt, geo_type='flat')
        x_gd_alt, x_gd_full_alt = tdoa.solvers.gradient_descent(**gd_args, eq_constraints=[a])
        if title is not None:
            print(title)
        print('Unconstrained Solution: ({:.2f} km E, {:.2f} km N, {:.2f} km U)'.format(x_gd[0]/1e3,
                                                                                       x_gd[1]/1e3,
                                                                                       x_gd[2]/1e3))
        print('Constrained Solution: ({:.2f} km E, {:.2f} km N, {:.2f} km U)'.format(x_gd_alt[0]/1e3,
                                                                                     x_gd_alt[1]/1e3,
                                                                                     x_gd_alt[2]/1e3))

        # Initialize the plot
        this_fig, this_ax = plt.subplots(subplot_kw=dict(projection='3d'))
        this_ax.stem(this_x_tdoa[0], this_x_tdoa[1], this_x_tdoa[2], basefmt='grey', linefmt='grey',
                     markerfmt='+', label='Sensors')
        this_ax.stem([x_tgt[0]], [x_tgt[1]], [x_tgt[2]], basefmt='grey', linefmt='grey', markerfmt='^',
                     label='Target')

        # Add isochrones
        iso_label = 'Isochrones'
        for this_x, this_zeta in zip(this_x_tdoa.T, zeta):
            if np.all(this_x == x_tdoa[:, ref_idx]):
                continue

            iso = tdoa.model.draw_isochrone(x1=this_x[:2, np.newaxis], x2=x_tdoa[:2, ref_idx],
                                            range_diff=this_zeta, num_pts=101, max_ortho=40e3)
            plt.plot(iso[0], iso[1], '--k', label=iso_label)
            iso_label=None

        # Plot GD solution
        this_ax.plot(x_gd_full[0], x_gd_full[1], x_gd_full[2], '-.s', markevery=[-1],label='GD (Unconstrained)')
        this_ax.plot(x_gd_full_alt[0], x_gd_full_alt[1], x_gd_full_alt[2], '-.s', markevery=[-1], label='GD (Constrained)')

        this_ax.set_xlim([-20e3, 20e3])
        this_ax.set_ylim([0e3, 50e3])
        this_ax.set_zlim([0e3, 1.2e3])

        if title is not None:
            plt.title(title)
        plt.legend()
        this_ax.set_xlabel('x [m]')
        this_ax.set_ylabel('y [m]')
        this_ax.set_zlabel('z [m]')

        # Set the view angle
        this_ax.azim = -45
        this_ax.elev = 10

        return this_fig

    # Set up sensors
    alt1 = 1e3
    x_init = np.array([0, 10e3, alt1])
    x_tdoa = np.array([[-15e3, -5e3, 5e3, 15e3],
                       [0., 0., 0., 0.],
                       [alt1, alt1, alt1, alt1]])

    figs =[_ex2_inner(x_tdoa, x_init, title='Example 5.2')]

    # Try again with better elevation support
    alt2 = 2*alt1
    x_tdoa[2] = [alt1, alt2, alt1, alt2]
    figs.append(_ex2_inner(x_tdoa, x_init, title='Better Altitude Support'))

    # Video 5.2 modified altitude again
    alt2 = 0.5*alt1
    alt3=0*alt1
    x_tdoa[2] = [alt2, alt1, alt3, alt2]
    figs.append(_ex2_inner(x_tdoa, x_init, title='Video 5.2 Version'))

    return figs


def example3():
    """
    Executes Example 5.3.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    7 February 2025

    :return: figure handle to generated graphic
    """

    return []


def example4():
    """
    Executes Example 5.4.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    7 February 2025

    :return: figure handle to generated graphic
    """

    return []


def example5():
    """
    Executes Example 5.5.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    7 February 2025

    :return: figure handle to generated graphic
    """

    return []
