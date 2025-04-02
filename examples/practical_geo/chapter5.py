import numpy as np
import matplotlib.pyplot as plt
import scipy

import tdoa.model
import utils
import hybrid
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

            iso = tdoa.model.draw_isochrone(x1=x_tdoa[:2, ref_idx], x2=this_x[:2],
                                            range_diff=this_zeta, num_pts=101, max_ortho=40e3)
            plt.plot(iso[0], iso[1], '--k', linewidth=0.5, label=iso_label)
            iso_label=None

        # Plot GD solution
        this_ax.plot(x_gd_full[0], x_gd_full[1], x_gd_full[2], '-.s', markevery=[-1], label='GD (Unconstrained)')
        this_ax.plot(x_gd_full_alt[0], x_gd_full_alt[1], x_gd_full_alt[2], '-.s', markevery=[-1], label='GD (Constrained)')

        this_ax.set_xlim([-20e3, 20e3])
        this_ax.set_ylim([0e3, 50e3])
        this_ax.set_zlim([0e3, 2.1e3])
        this_ax.set_clip_on(True)

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

    # Set up scene
    ref_lla = np.array([20., -150., 0.])  # deg lat, deg lon, m alt
    x_aoa = np.zeros((3,1))               # meters, ENU
    x_tdoa = np.array([[20e3, 25e3],
                       np.zeros((2,)),
                       np.zeros((2,))])   # meters, ENU
    _, num_tdoa = utils.safe_2d_shape(x_tdoa)
    ref_idx = num_tdoa - 1  # index of TDOA reference sensor

    tgt_az = 30.    # degrees E of N
    tgt_rng = 50e3  # meters
    tgt_alt = 10e3  # meters

    x_tgt = np.array([tgt_rng * np.sin(tgt_az * _deg2rad),
                      tgt_rng * np.cos(tgt_az * _deg2rad),
                      tgt_alt])  # meters, ENU

    # Errors
    err_aoa = 3 * _deg2rad
    err_toa = 1e-6
    err_roa = utils.constants.speed_of_light * err_toa

    cov_aoa = err_aoa**2 * np.eye(2)  # 2D AOA measurement covariance
    cov_roa = err_roa**2 * np.eye(num_tdoa)  # ROA measurement covariance
    cov_raw = CovarianceMatrix(scipy.linalg.block_diag(cov_aoa, cov_roa))  # convert to Covariance Matrix object
    cov_msmt = cov_raw.resample_hybrid(x_aoa=x_aoa, x_tdoa=x_tdoa, do_2d_aoa=True, tdoa_ref_idx=ref_idx)

    # CRLB Computation
    crlb_args = {'x_source':x_tgt,
                 'cov':cov_msmt,
                 'x_aoa':x_aoa,
                 'x_tdoa':x_tdoa,
                 'do_2d_aoa':True,
                 'tdoa_ref_idx': ref_idx,
                 'do_resample': False}
    crlb_raw = hybrid.perf.compute_crlb(**crlb_args)

    _, a_grad = utils.constraints.fixed_alt(tgt_alt, geo_type='flat')
    crlb_fix = hybrid.perf.compute_crlb(**crlb_args, eq_constraints_grad=[a_grad])

    print('CRLB (unconstrained):')
    with np.printoptions(precision=0):
        print(crlb_raw)
    print('CRLB (constrained):')
    with np.printoptions(precision=0, suppress=True):
        print(crlb_fix)

    ## Plot for x/y grid
    # Initialize grid
    max_offset = int(10e3)
    num_pts = 201
    grid_res = 2*max_offset / (num_pts-1)
    x_set, x_grid, out_shape = utils.make_nd_grid(x_ctr=x_tgt,
                                                  max_offset=max_offset*np.array([1, 1, 0]),
                                                  grid_spacing=grid_res)

    # Compute CRLB across grid
    crlb_args['x_source'] = x_set  # replace singular source point with grid of potential source points
    crlb_args['print_progress'] = True  # turn on progress tracker; these may take some time
    crlb_raw_grid = hybrid.perf.compute_crlb(**crlb_args)
    crlb_fix_grid = hybrid.perf.compute_crlb(**crlb_args, eq_constraints_grad=[a_grad])

    # Compute RMSE of each grid point
    rmse_raw = np.reshape(np.sqrt(np.trace(crlb_raw_grid, axis1=0, axis2=1)), newshape=out_shape)
    rmse_fix = np.reshape(np.sqrt(np.trace(crlb_fix_grid, axis1=0, axis2=1)), newshape=out_shape)

    # Plot RMSE
    fig, axes = plt.subplots(ncols=2)
    contour_levels = np.arange(20)
    extent = ((x_tgt[0] - max_offset)/1e3,
              (x_tgt[0] + max_offset)/1e3,
              (x_tgt[1] - max_offset)/1e3,
              (x_tgt[1] + max_offset)/1e3)

    # Unconstrained on axes[0] and Constrained on axes[1]
    for this_ax, this_z in zip(axes, [rmse_raw, rmse_fix]):
        # Begin with the RMSE Background Plot
        hdl_img = this_ax.imshow(this_z.squeeze()/1e3, origin='lower', cmap='viridis_r', extent=extent,
                                 vmin=0, vmax=contour_levels[-1])

        # Unlike in MATLAB, contourf does not draw contour edges. Manually add contours
        hdl_contour = this_ax.contour(x_grid[0].squeeze()/1e3, x_grid[1].squeeze()/1e3, this_z.squeeze()/1e3,
                                      levels=contour_levels,origin='lower', colors='k')
        plt.clabel(hdl_contour, fontsize=10, colors='k')

        # Add a target scatterer, legend, and axis labels
        this_ax.scatter(x_tgt[0]/1e3, x_tgt[1]/1e3, color='k', facecolors='k', marker='^', label='Target')
        this_ax.set_xlabel('E [km]')
        this_ax.set_ylabel('N [km]')
        this_ax.legend(loc='upper left')

    # Colorbar and subplot titles
    fig.colorbar(hdl_img, ax=axes, location='bottom', label='RMSE [km]')
    axes[0].set_title('Unconstrained')
    axes[1].set_title('Constrained')

    return fig


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
