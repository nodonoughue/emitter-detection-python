import numpy as np
import matplotlib.pyplot as plt
import scipy

import tdoa.model
import utils
import hybrid
from utils.covariance import CovarianceMatrix
from utils.coordinates import ecef_to_enu, ecef_to_lla, enu_to_ecef, lla_to_ecef
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
    gd_args = {'x_sensor': x_aoa, 'psi': psi, 'cov': cov, 'x_init': x_init}
    x_gd, x_gd_full = triang.solvers.gradient_descent(**gd_args)

    # Gradient Descent Solution; Constrained
    y_soln = 25.
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

    # Do it again with LS
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
            iso_label = None

        # Plot GD solution
        this_ax.plot(x_gd_full[0], x_gd_full[1], x_gd_full[2], '-.s', markevery=[-1], label='GD (Unconstrained)')
        this_ax.plot(x_gd_full_alt[0], x_gd_full_alt[1], x_gd_full_alt[2], '-.o', markevery=[-1],
                     label='GD (Constrained)')

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

    figs = [_ex2_inner(x_tdoa, x_init, title='Example 5.2')]

    # Try again with better elevation support
    alt2 = 2*alt1
    x_tdoa[2] = [alt1, alt2, alt1, alt2]
    figs.append(_ex2_inner(x_tdoa, x_init, title='Better Altitude Support'))

    # Video 5.2 modified altitude again
    alt2 = 0.5*alt1
    alt3 = 0*alt1
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
    # ref_lla = np.array([20., -150., 0.])  # deg lat, deg lon, m alt
    x_aoa = np.zeros((3, 1))               # meters, ENU
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
    crlb_args = {'x_source': x_tgt,
                 'cov': cov_msmt,
                 'x_aoa': x_aoa,
                 'x_tdoa': x_tdoa,
                 'do_2d_aoa': True,
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

    # Plot for x/y grid
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
    for this_ax, this_z, this_title in zip(axes, [rmse_raw, rmse_fix], ['Unconstrained', 'Constrained']):
        # Begin with the RMSE Background Plot
        hdl_img = this_ax.imshow(this_z.squeeze()/1e3, origin='lower', cmap='viridis_r', extent=extent,
                                 vmin=0, vmax=contour_levels[-1])

        # Unlike in MATLAB, contourf does not draw contour edges. Manually add contours
        hdl_contour = this_ax.contour(x_grid[0].squeeze()/1e3, x_grid[1].squeeze()/1e3, this_z.squeeze()/1e3,
                                      levels=contour_levels, origin='lower', colors='k')
        plt.clabel(hdl_contour, fontsize=10, colors='k')

        # Add a target scatterer, legend, and axis labels
        this_ax.scatter(x_tgt[0]/1e3, x_tgt[1]/1e3, color='k', facecolors='k', marker='^', label='Target')
        this_ax.set_xlabel('E [km]')
        this_ax.set_ylabel('N [km]')
        this_ax.set_title(this_title)
        this_ax.legend(loc='upper left')

    # Colorbar and subplot titles
    # noinspection PyUnboundLocalVariable
    fig.colorbar(hdl_img, ax=axes, location='bottom', label='RMSE [km]')

    return [fig]


def example4():
    """
    Executes Example 5.4.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    3 April 2025

    :return: figure handle to generated graphic
    """

    # Set up scene
    ref_lla = np.array([25., -15., 0.])  # deg lat, deg lon, m alt
    x_aoa_enu = np.array([[0, 50e3, 0],
                          [0, 0, 50e3],
                          10*np.ones((3,))])  # meters, ENU
    x_aoa_ecef = np.array(enu_to_ecef(east=x_aoa_enu[0], north=x_aoa_enu[1], up=x_aoa_enu[2],
                                      lat_ref=ref_lla[0], lon_ref=ref_lla[1], alt_ref=ref_lla[2],
                                      dist_units='m', angle_units='deg'))  # convert tuple output to an array
    _, num_aoa = utils.safe_2d_shape(x_aoa_ecef)

    sat_lla = np.array([27, -13, 575e3])  # deg lat, deg lon, m alt
    x_tgt_ecef = np.array(lla_to_ecef(lat=sat_lla[0], lon=sat_lla[1], alt=sat_lla[2],
                                      angle_units='deg', dist_units='m'))
    x_tgt_enu = np.array(ecef_to_enu(x=x_tgt_ecef[0], y=x_tgt_ecef[1], z=x_tgt_ecef[2],
                                     lat_ref=ref_lla[0], lon_ref=ref_lla[1], alt_ref=ref_lla[2],
                                     angle_units='deg', dist_units='m'))

    # Build Constraints
    # Note: bounded_alt returns a list of two one-sided inequality constraints; no need to wrap it in a list
    # when passing to gradient_descent
    alt_low = 500e3
    alt_high = 600e3
    b = utils.constraints.bounded_alt(alt_min=alt_low, alt_max=alt_high, geo_type='ellipse')

    # Measurement Errors
    err_aoa = 3 * _deg2rad
    cov_aoa = CovarianceMatrix(err_aoa ** 2 * np.eye(2*num_aoa))  # 2D AOA measurement covariance

    # Noisy Measurement
    z = triang.model.measurement(x_sensor=x_aoa_ecef, x_source=x_tgt_ecef, do_2d_aoa=True)
    n = cov_aoa.lower @ np.random.randn(2*num_aoa)
    zeta = z + n

    # Solvers
    init_alt = 500e3
    x_init = np.array(lla_to_ecef(lat=ref_lla[0], lon=ref_lla[1], alt=init_alt,
                                  angle_units='deg', dist_units='m'))

    gd_args = {'x_init': x_init, 'x_sensor': x_aoa_ecef, 'cov': cov_aoa, 'psi': zeta, 'do_2d_aoa': True}
    x_gd, x_gd_full = triang.solvers.gradient_descent(**gd_args)
    x_gd_bound, x_gd_bound_full = triang.solvers.gradient_descent(**gd_args, ineq_constraints=b)

    # Convert Solutions to LLA and Print
    x_gd_lla = np.array(ecef_to_lla(x_gd[0], x_gd[1], x_gd[2],
                                    angle_units='deg', dist_units='m'))
    print('Unconstrained Solution: {:.2f} deg N, {:.2f} deg W, {:.2f} km'.format(x_gd_lla[0],
                                                                                 np.fabs(x_gd_lla[1]),
                                                                                 x_gd_lla[2]/1e3))
    gd_err = np.linalg.norm(x_gd-x_tgt_ecef)/1e3
    print('   Error: {:.2f} km'.format(gd_err))

    x_gd_bound_lla = np.array(ecef_to_lla(x_gd_bound[0], x_gd_bound[1], x_gd_bound[2],
                                          angle_units='deg', dist_units='m'))
    print('Constrained Solution: {:.2f} deg N, {:.2f} deg W, {:.2f} km'.format(x_gd_bound_lla[0],
                                                                               np.fabs(x_gd_bound_lla[1]),
                                                                               x_gd_bound_lla[2] / 1e3))
    gd_bound_err = np.linalg.norm(x_gd_bound - x_tgt_ecef) / 1e3
    print('   Error: {:.2f} km'.format(gd_bound_err))

    # Plot in ENU Coordinates
    x_gd_enu = np.array(ecef_to_enu(x_gd_full[0], x_gd_full[1], x_gd_full[2],
                                    lat_ref=ref_lla[0], lon_ref=ref_lla[1], alt_ref=ref_lla[2],
                                    angle_units='deg', dist_units='m'))
    x_gd_bound_enu = np.array(ecef_to_enu(x_gd_bound_full[0], x_gd_bound_full[1], x_gd_bound_full[2],
                                          lat_ref=ref_lla[0], lon_ref=ref_lla[1], alt_ref=ref_lla[2],
                                          angle_units='deg', dist_units='m'))

    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.stem(x_aoa_enu[0]/1e3, x_aoa_enu[1]/1e3, x_aoa_enu[2]/1e3, basefmt='grey', linefmt='grey',
            markerfmt='+', label='Sensors')
    ax.stem([x_tgt_enu[0]/1e3], [x_tgt_enu[1]/1e3], [x_tgt_enu[2]/1e3], basefmt='grey', linefmt='grey',
            markerfmt='^', label='Target')
    ax.plot(x_gd_enu[0]/1e3, x_gd_enu[1]/1e3, x_gd_enu[2]/1e3, marker='s', markevery=[-1], label='GD (Unconstrained)')
    ax.plot(x_gd_bound_enu[0] / 1e3, x_gd_bound_enu[1] / 1e3, x_gd_bound_enu[2] / 1e3, marker='o', markevery=[-1],
            label='GD (Constrained)')

    plt.legend()

    ax.set_xlabel('x [km]')
    ax.set_ylabel('y [km]')
    ax.set_zlabel('z [km]')

    # Set the view angle
    ax.azim = -45
    ax.elev = 10

    return [fig]


def example5():
    """
    Executes Example 5.5.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    7 February 2025

    :return: figure handle to generated graphic
    """

    # Set up scenario
    baseline = 10e3
    num_tdoa = 4
    tdoa_angle = np.pi/6 + 2*np.pi/3 * np.arange(num_tdoa-1)
    x_tdoa = baseline * np.array([np.cos(tdoa_angle), np.sin(tdoa_angle), np.zeros_like(tdoa_angle)])
    x_tdoa = np.concatenate((np.zeros((3, 1)), x_tdoa), axis=1)  # add a sensor at the origin

    # Errors
    err_time = 3e-7
    err_range = utils.constants.speed_of_light * err_time
    cov_roa = CovarianceMatrix(err_range**2 * np.eye(num_tdoa))
    ref_idx = None
    cov_rdoa = cov_roa.resample(ref_idx=ref_idx)

    # Target Coordinates
    tgt_range = 100e3
    tgt_alt = utils.unit_conversions.convert(40e3, from_unit='ft', to_unit='m')
    x_tgt = np.array(utils.coordinates.correct_enu(e_ground=tgt_range, n_ground=0., u_ground=tgt_alt))

    # External Prior
    x_prior = np.array(utils.coordinates.correct_enu(e_ground=95e3, n_ground=10e3, u_ground=10e3))
    cov_prior = np.array([[5., 1., 0.], [1., 50., 0.], [0., 0., 10.]])*1e6

    def prior(x):
        # x is (n_dim x n_position) array of potential source positions; compute mvnpdf for each, but don't bother
        # with cross-terms
        return np.array([scipy.stats.multivariate_normal.pdf(this_x, mean=x_prior, cov=cov_prior) for this_x in x.T])

    # Measurement
    z = tdoa.model.measurement(x_sensor=x_tdoa, x_source=x_tgt, ref_idx=ref_idx)
    noise = cov_rdoa.lower @ np.random.randn(num_tdoa-1)
    zeta = z + noise

    # Solution
    x_center = x_tgt
    grid_size = np.array([50e3, 50e3, 0])
    epsilon = 250
    extent = (float(x_tgt[0] - grid_size[0]) / 1e3,
              float(x_tgt[0] + grid_size[0]) / 1e3,
              float(x_tgt[1] - grid_size[1]) / 1e3,
              float(x_tgt[1] + grid_size[1]) / 1e3)  # cast each entry to a float to avoid a PyCharm type warning later

    ml_args = {'x_sensor': x_tdoa, 'rho': zeta, 'cov': cov_rdoa, 'x_ctr': x_center, 'search_size': grid_size,
               'epsilon': epsilon, 'ref_idx': ref_idx}
    x_ml, score, x_grid = tdoa.solvers.max_likelihood(**ml_args)
    x_ml_prior, score_prior, _ = tdoa.solvers.max_likelihood(**ml_args, prior=prior, prior_wt=0.5)

    print('Solution w/o prior: {:.2f} km, {:.2f} km, {:.2f} km'.format(x_ml[0]/1e3, x_ml[1]/1e3, x_ml[2]/1e3))
    print('    Error: {:.2f} km'.format(np.linalg.norm(x_ml-x_tgt)/1e3))
    print('Solution w/prior: {:.2f} km, {:.2f} km, {:.2f} km'.format(x_ml_prior[0]/1e3,
                                                                     x_ml_prior[2]/1e3,
                                                                     x_ml_prior[2]/1e3))
    print('    Error: {:.2f} km'.format(np.linalg.norm(x_ml_prior - x_tgt)/1e3))

    # Plot
    def _do_plot(this_ell, title, do_prior=False):
        this_fig = plt.figure()
        im = plt.imshow(this_ell, origin='lower', extent=extent, cmap='viridis', label=None, vmin=-50, vmax=0)
        plt.scatter(x_tdoa[0]/1e3, x_tdoa[1]/1e3, marker='o', label='Sensors')
        plt.scatter(x_tgt[0]/1e3, x_tgt[1]/1e3, marker='^', label='Target')
        plt.scatter(x_ml[0]/1e3, x_ml[1]/1e3, marker='s', label='Estimate')

        if do_prior:
            ell_prior = utils.errors.draw_error_ellipse(x_prior[:2], cov_prior[:2, :2], num_pts=100, conf_interval=90)
            plt.scatter(x_prior[0]/1e3, x_prior[1]/1e3, marker='v', label='Prior')
            plt.plot(ell_prior[0]/1e3, ell_prior[1]/1e3, 'w-.', label='Prior Confidence (90%)')
            plt.scatter(x_ml_prior[0]/1e3, x_ml_prior[1]/1e3, marker='d', label='Estimate (w/prior)')

        plt.grid(True, color='w')
        plt.xlabel('x [km]')
        plt.ylabel('y [km]')
        plt.legend(loc='upper left', fontsize='small')
        plt.title(title)

        this_fig.colorbar(im, shrink=0.6)

        return this_fig

    out_shape = np.shape(np.squeeze(x_grid[0]))
    fig1 = _do_plot(np.reshape(score, out_shape), title='Likelihood Estimate w/o Prior', do_prior=False)
    fig2 = _do_plot(np.reshape(score_prior, out_shape), title='Likelihood Estimate w/Prior', do_prior=True)

    return [fig1, fig2]
