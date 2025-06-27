import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import tdoa.model
import utils
import hybrid
from utils.covariance import CovarianceMatrix
import triang

_rad2deg = utils.unit_conversions.convert(1, "rad", "deg")
_deg2rad = utils.unit_conversions.convert(1, "deg", "rad")


def run_all_examples():
    """
    Run all chapter 6 examples and return a list of figure handles

    :return figs: list of figure handles
    """

    return list(example1())


def example1():
    """
    Executes Example 6.1.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    8 April 2025

    :return: figure handle to generated graphic
    """
    # Set up sensors
    x_aoa = np.array([[2, 2, 0], [2, -1, 0]])
    _, n_aoa = np.shape(x_aoa)

    x_tgt = np.array([5, 3])

    # Define received signals and covariance matrix
    alpha = np.array([5, 10, -5])*_deg2rad  # AOA bias
    psi = triang.model.measurement(x_sensor=x_aoa, x_source=x_tgt, do_2d_aoa=False)
    psi_bias = triang.model.measurement(x_sensor=x_aoa, x_source=x_tgt, do_2d_aoa=False, bias=alpha)

    # Plot the scenario
    fig = plt.figure()
    hdl_sensors=plt.scatter(x_aoa[0], x_aoa[1], marker='s', label='Sensors')
    hdl_target=plt.scatter(x_tgt[0], x_tgt[1], marker='^', label='Target')
    plt.grid(True)

    # Draw the LOBs
    xy_lob = triang.model.draw_lob(x_sensor=x_aoa, psi=psi, x_source=x_tgt, scale=1.5)
    label_set = [None]*n_aoa
    label_set[0] = 'LOB (w/o bias)'
    plt.plot(xy_lob[0], xy_lob[1], color=hdl_target.get_facecolor(), label=label_set)

    xy_lob_bias = triang.model.draw_lob(x_sensor=x_aoa, psi=psi_bias, x_source=x_tgt, scale=1.5)
    label_set[0] = 'LOB (w/bias)'
    plt.plot(xy_lob_bias[0], xy_lob_bias[1], '--', color=hdl_sensors.get_facecolor(), label=label_set)

    plt.legend(loc='upper left')
    plt.xlim(-1, 6)
    plt.ylim(-1, 5)

    return [fig]


def example2():
    """
    Executes Example 6.2.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    8 April 2025

    :return: figure handle to generated graphic
    """

    # Set up sensors
    x_tdoa = np.array([[0, 2, 0],[2, -2, 0]])  # avg position(reported)
    _, n_tdoa = utils.safe_2d_shape(x_tdoa)

    cov_pos_full = .1 * np.eye(2 * n_tdoa)  # position covar; all are IID
    cov_pos_lower = np.linalg.cholesky(cov_pos_full, upper=False)

    # Generate a random set of TDOA positions
    x_tdoa_actual = x_tdoa + np.reshape(cov_pos_lower @ np.random.randn(2 * n_tdoa, ), newshape=(2, n_tdoa))

    # Generate Measurements
    x_tgt = np.array([6, 3])

    zeta = tdoa.model.measurement(x_sensor=x_tdoa, x_source=x_tgt, ref_idx=n_tdoa-1)
    zeta_unc = tdoa.model.measurement(x_sensor=x_tdoa_actual, x_source=x_tgt, ref_idx=n_tdoa-1)

    print('Measurements from sensors 1-3 (w.r.t sensor 0):')
    print('Nominal Positions: {:.2f} m, {:.2f} m'.format(zeta[0], zeta[1]))
    print('Random Positions:  {:.2f} m, {:.2f} m'.format(zeta_unc[0], zeta_unc[1]))

    # Plot Scenario
    fig = plt.figure()
    hdl_nominal = plt.scatter(x_tdoa[0], x_tdoa[1], marker='s', label='Sensors (nominal positions)')
    hdl_true = plt.scatter(x_tdoa_actual[0], x_tdoa_actual[1], marker='o', label='Sensors (true positions)')
    plt.scatter(x_tgt[0], x_tgt[1], marker='^', label='Target', zorder=3)  # use zorder=3 to place in front of lines
    plt.grid(True)

    # Draw the Isochrones
    label_nominal = 'Isochrone (nominal positions)'
    label_actual = 'Isochrone (true positions)'
    for idx in np.arange(n_tdoa - 1):
        xy_iso = tdoa.model.draw_isochrone(x_ref=x_tdoa[:, -1], x_test=x_tdoa[:, idx],
                                           range_diff=zeta_unc[idx], num_pts=101, max_ortho=8)
        xy_iso_unc = tdoa.model.draw_isochrone(x_ref=x_tdoa_actual[:, -1], x_test=x_tdoa_actual[:, idx],
                                               range_diff=zeta_unc[idx], num_pts=101, max_ortho=8)

        plt.plot(xy_iso[0], xy_iso[1], '-', color=hdl_nominal.get_facecolor(), label=label_nominal)
        plt.plot(xy_iso_unc[0], xy_iso_unc[1], '--', color=hdl_true.get_facecolor(), label=label_actual)

        label_nominal = None
        label_actual = None

    plt.legend(loc='lower right')
    plt.xlim(-1, 8)
    plt.ylim(-3, 4)

    return [fig]


def example3():
    """
    Executes Example 6.3.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    8 April 2025

    :return: figure handle to generated graphic
    """

    # Set up sensors
    x_aoa = np.array([[2., 2., 0.], [2.,  -1., 0.]])
    num_dims, n_aoa = utils.safe_2d_shape(x_aoa)

    x_tdoa = np.array([[0., 2., 0.], [2., -1., 0.]])  # avg position (reported)
    _, n_tdoa = utils.safe_2d_shape(x_tdoa)

    # Find matching sensors
    dist = utils.geo.calc_range(x1=x_aoa, x2=x_tdoa)
    idx_aoa, idx_tdoa = np.where(dist == 0.)

    # Build position covariance matrix
    cov_pos_1d = np.eye(n_tdoa+n_aoa)  # position covariance; all are sensor positions are IID
    for this_idx_aoa, this_idx_tdoa in zip(idx_aoa, idx_tdoa):
        # Add off-diagonal 1's to all entries corresponding to a matching
        # TDOA/AOD sensor, to ensure that random position errors are matched
        cov_pos_1d[this_idx_aoa, n_aoa+this_idx_tdoa] = 1
        cov_pos_1d[n_aoa+this_idx_tdoa,this_idx_aoa] = 1

    # At this point, cov_pos_1d is the correlation coefficient of all
    # sensor position errors in a given dimension.  We must now expand
    # the matrix to account for (a) actual position variance and (b) the
    # number of spatial dimensions in the problem.
    cov_pos_single = .1*np.eye(num_dims)  # covariance across dimensions, for a single sensor
    cov_pos = np.kron(cov_pos_1d, cov_pos_single) # combined covariance of all sensor coordinates

    # Generate a random set of AOA and TDOA positions
    # L=chol(C_beta,'lower'); -- This will fail, because C_beta is not positive
    # definite (it has some eigenvalues that are zero)
    singular_vectors, singular_values, _ = np.linalg.svd(cov_pos)
    cov_lower = singular_vectors @ np.diag(np.sqrt(singular_values))
    epsilon = np.reshape(cov_lower @ np.random.randn(num_dims*(n_aoa+n_tdoa), 1), newshape=(n_aoa+n_tdoa, num_dims)).T

    # Grab the position offsets and add to the reported TDOA and AOA positions
    x_aoa_true = x_aoa + epsilon[:, :n_aoa]  # first n_dim x n_aoa errors belong to the AOA sensors
    x_tdoa_true = x_tdoa + epsilon[:, n_aoa:]  # latter n_dim x n_tdoa errors belong to the TDOA sensors

    # Let's verify that sensors 2 and 4 are still colocated
    dist_perturbed = utils.geo.calc_range(x1=x_aoa_true, x2=x_tdoa_true)
    assert np.all(np.fabs(dist_perturbed[idx_aoa, idx_tdoa]) < 1e-6), 'Error generating correlated sensor perturbations.'

    # Generate Measurements
    x_tgt = np.array([6, 3])

    alpha_aoa = np.array([5, 10, -5])*_deg2rad  # AOA bias
    msmt_args = {'x_fdoa': None,
                 'v_fdoa': None,
                 'x_source': x_tgt,
                 'tdoa_ref_idx': n_tdoa-1,
                 'do_2d_aoa': False}
    zeta = hybrid.model.measurement(x_aoa=x_aoa, x_tdoa=x_tdoa, **msmt_args)  # free of pos unc and bias
    zeta_unc = hybrid.model.measurement(x_aoa=x_aoa_true, x_tdoa=x_tdoa_true, **msmt_args)  # with pos unc, no bias
    zeta_unc_bias = hybrid.model.measurement(x_aoa=x_aoa_true, x_tdoa=x_tdoa_true, angle_bias=alpha_aoa, **msmt_args)
    # with pos unc and bias

    print('Measurements from ideal sensors (AOA, AOA, AOA, RDOA, RDOA):\n[', end='')
    [print('{:.2f}, '.format(this_zeta), end='') for this_zeta in zeta]
    print(']\nWith pos unc:\n[', end='')
    [print('{:.2f}, '.format(this_zeta), end='') for this_zeta in zeta_unc]
    print(']\nWith pos unc and bias:\n[', end='')
    [print('{:.2f}, '.format(this_zeta), end='') for this_zeta in zeta_unc_bias]
    print(']')

    # Plot Scenario
    fig = plt.figure()
    hdl_nominal_a = plt.scatter(x_aoa[0], x_aoa[1], marker='s', label='AOA Sensors (nominal positions)')
    hdl_nominal_t = plt.scatter(x_tdoa[0], x_tdoa[1], marker='s', label='TDOA Sensors (nominal positions)')
    hdl_true_a = plt.scatter(x_aoa_true[0], x_aoa_true[1], marker='o', label='AOA Sensors (true positions)')
    hdl_true_t = plt.scatter(x_tdoa_true[0], x_tdoa_true[1], marker='o', label='TDOA Sensors (true positions)')
    plt.scatter(x_tgt[0], x_tgt[1],marker='^', color='k', label='Target')
    plt.grid(True)

    # Draw the Isochrones and LOBs -- Truth
    xy_lob = triang.model.draw_lob(x_sensor=x_aoa, psi=zeta_unc_bias[:n_aoa], x_source=x_tgt, scale=1.5)
    # xy_lob_bias = triang.model.draw_lob(x_sensor=x_aoa_true, psi=zeta_unc_bias[:n_aoa], x_source=x_tgt, scale=1.5)
    label_set = [None] * n_aoa
    label_set[0] = 'LOB (nominal positions w/bias)'
    plt.plot(xy_lob[0],xy_lob[1], color=hdl_nominal_a.get_facecolor(), label=label_set)
    # label_set[0] = 'LOB (true positions w/bias)'
    # plt.plot(xy_lob_bias[0], xy_lob_bias[1],'-.', color=hdl_true_a.get_facecolor(), label=label_set)

    label_nominal = 'Isochrone (nominal positions w/bias)'
    # label_true = 'Isochrone (true positions w/bias)'
    for idx in np.arange(n_tdoa-1):
        xy_iso = tdoa.model.draw_isochrone(x_tdoa[:, -1], x_tdoa[:, idx], range_diff=zeta_unc_bias[n_aoa+idx],
                                           num_pts=101, max_ortho=8)
        # xy_iso_true = tdoa.model.draw_isochrone(x_tdoa_true[:, -1], x_tdoa_true[:, idx], range_diff=zeta_unc_bias[n_aoa + idx],
        #                                    num_pts=101, max_ortho=8)

        plt.plot(xy_iso[0], xy_iso[1], color=hdl_nominal_t.get_facecolor(), label=label_nominal)
        # plt.plot(xy_iso_true[0], xy_iso_true[1], color=hdl_true_t.get_facecolor(), label=label_true)
        label_nominal = 'None'
        # label_true = 'None'

    plt.legend(loc='lower right')
    plt.xlim(-1, 8)
    plt.ylim(-3, 4)

    return [fig]


def example4(do_iterative=False):
    """
    Executes Example 6.4.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    29 April 2025

    :param do_iterative: Boolean flag; if True this example executes the modifications discussed in Video 6.2
    :return: figure handle to generated graphic
    """

    # Set up sensors
    x_tdoa = np.array([[2, 0,  4, 0],
                       [2, 2, 0, 0]])*1e3 # avg position (reported)
    n_dim, n_tdoa = utils.safe_2d_shape(x_tdoa)

    # Generate Measurements
    x_tgt = np.array([6, 3])*1e3

    tdoa_bias = np.array([10, 30, -20, 60])  # TOA bias
    tdoa_args = {'x_sensor': x_tdoa,
                 'x_source': x_tgt,
                 'ref_idx': n_tdoa-1}
    z = tdoa.model.measurement(**tdoa_args, bias=tdoa_bias)  # free of pos unc, w/bias
    z_true = tdoa.model.measurement(**tdoa_args)  # free of pos unc, w/o bias

    err_toa = 100e-9
    cov_toa = CovarianceMatrix((err_toa**2)*np.eye(n_tdoa))
    cov_roa = cov_toa.multiply(val=utils.constants.speed_of_light**2, overwrite=False)
    cov_rdoa = cov_roa.resample(ref_idx=n_tdoa-1)
    lower_rdoa = cov_rdoa.lower

    noise = lower_rdoa @ np.random.randn(n_tdoa-1, )
    zeta = z + noise
    zeta_true = z_true + noise

    # Compute Log Likelihood
    x_ctr = np.array([5e3, 5e3])
    search_size = 5e3
    grid_res = .05e3
    x_set, x_grid, out_shape = utils.make_nd_grid(x_ctr=x_ctr, max_offset=search_size, grid_spacing=grid_res)
    extent = ((x_ctr[0]-search_size)/1e3, (x_ctr[0]+search_size)/1e3,
              (x_ctr[1]-search_size)/1e3, (x_ctr[1]+search_size)/1e3)

    tdoa_args = {'x_sensor': x_tdoa,
                 'x_source': x_set,
                 'ref_idx': n_tdoa-1,
                 'do_resample': True,
                 'cov': cov_roa,
                 'variance_is_toa': False}
    ell = tdoa.model.log_likelihood(zeta=zeta, **tdoa_args)
    ell_true = tdoa.model.log_likelihood(zeta=zeta_true, **tdoa_args)

    # Plot Scenario
    levels = [-1000, -100, -50, -20, -10, -5, 0]
    def _make_plot(this_ell, x_plot, label_set, title=None):
        this_fig, this_ax = plt.subplots()

        cmap = cm.get_cmap('viridis')  # Choose a colormap
        cmap.set_under('k', alpha=0)  # Set values below vmin to transparent black

        hdl = plt.imshow(this_ell, origin='lower', cmap=cmap, extent=extent)
        plt.clim([-100, 0])
        plt.colorbar(hdl, format='%d')

        # Unlike in MATLAB, contourf does not draw contour edges. Manually add contours
        hdl2 = plt.contour(x_grid[0]/1e3, x_grid[1]/1e3, this_ell, levels=levels,
                           origin='lower', colors='k')
        plt.clabel(hdl2, fontsize=10, colors='k')

        # Scatter plots
        for this_x, this_label in zip(x_plot, label_set):
            plt.scatter(this_x[0]/1e3, this_x[1]/1e3, label=this_label, clip_on=False)

        plt.grid(True)
        plt.legend(loc='upper left')

        plt.xlim([0, 10])
        plt.ylim([0, 10])

        if title is not None:
            plt.title(title)

        return this_fig

    ell_true_plot = np.reshape(ell_true, out_shape) - np.max(ell_true)
    ell_plot = np.reshape(ell, out_shape) - np.max(ell)
    figs= [_make_plot(ell_true_plot, [x_tdoa, x_tgt], ['Sensors', 'Target']),
           _make_plot(ell_plot, [x_tdoa, x_tgt], ['Sensors', 'Target'])]

    # ML Solver (baseline)
    x_ctr = np.array([5, 5])*1e3
    search_size = 5e3
    grid_res = .1e3

    solver_args = {'x_sensor': x_tdoa,
                   'cov': cov_rdoa,
                   'x_ctr': x_ctr,
                   'search_size': search_size,
                   'epsilon': grid_res,
                   'ref_idx': n_tdoa-1,
                   'do_resample': False,
                   'variance_is_toa': False}
    x_est_true, _, _ = tdoa.solvers.max_likelihood(zeta=zeta_true, **solver_args)
    x_est, _, _ = tdoa.solvers.max_likelihood(zeta=zeta, **solver_args)

    print('True ML Est.: ({:.2f}, {:.2f}) km, error: {:.2f} km'.format(x_est_true[0]/1e3, x_est_true[1]/1e3,
                                                                      np.linalg.norm(x_est_true-x_tgt)/1e3))
    print('Biased ML Est.: ({:.2f}, {:.2f}) km, error: {:.2f} km'.format(x_est[0]/1e3, x_est[1]/1e3,
                                                                         np.linalg.norm(x_est-x_tgt)/1e3))

    # ML Solver with Bias Estimation
    # In this solver, we'll let the ML search grid include possible bias terms, but not sensor position uncertainties
    th_ctr = np.concatenate((x_ctr,              # unknown x/y coordinates of source (start at center of search grid)
                             np.zeros(n_tdoa,),  # unknown ROA bias terms (start at zero bias)
                             x_tdoa.ravel()))    # unknown x/y coordinates of sensors (start at nominal positions)
    search_size = np.concatenate((5e3*np.ones(2,),          # x/y search grid at +/- 5 km
                                 80*np.ones(n_tdoa-1,),     # assume as much as 80 meters of RDOA error
                                 np.zeros(1, ),  # the reference sensor can be assumed to have no bias (for simplicity)
                                 np.zeros(n_tdoa*n_dim,)))  # assume no sensor position uncertainty
    epsilon = np.concatenate((500*np.ones(2,),      # search target positions with 500 m accuracy
                              10*np.ones(n_tdoa,),  # search bias terms with 10 m RDOA accuracy
                              np.ones(n_tdoa*n_dim,)))  # resolution for x_tdoa doesn't matter; search_size is 0
    cov_beta = CovarianceMatrix(.001*np.eye(n_tdoa*n_dim))  # position covariance error

    unc_solver_args = solver_args
    unc_solver_args['x_ctr'] = th_ctr               # overwrite search parameters using new, larger versions with
    unc_solver_args['search_size'] = search_size    # dimensions for bias and sensor position uncertainty
    unc_solver_args['epsilon'] = epsilon
    unc_solver_args['cov_pos'] = cov_beta

    x_est_bias, bias_est, x_tdoa_est, _, _  = tdoa.solvers.max_likelihood_uncertainty(zeta=zeta, **unc_solver_args)
    err_km = np.linalg.norm(x_est_bias-x_tgt)/1e3
    print('ML Est. w/Uncertainty: ({:.2f}, {:.2f}) km, error: {:.2f} km'.format(x_est_bias[0]/1e3,
                                                                                x_est_bias[1]/1e3,
                                                                                err_km))

    with np.printoptions(precision=3, suppress=True):
        print('True range bias: (', end='')
        print(*tdoa_bias, sep=', ', end=') m\n')
        print('Estimated range bias: (', end='')
        print(*bias_est[:n_tdoa], sep=', ', end=') m\n')

    # Plot Solutions
    figs.append(_make_plot(ell_plot, [x_tdoa, x_tgt, x_est_true],
                           ['Sensors', 'Target', 'ML Est.']))
    figs.append(_make_plot(ell_plot, [x_tdoa, x_tgt, x_est, x_est_bias],
                           ['Sensors', 'Target', 'ML Est.', 'ML Est. w/uncertainty']))

    if do_iterative:
        iter_solver_args = {'x_sensor': x_tdoa,
                            'cov': cov_rdoa,
                            'th_init': x_ctr,
                            'epsilon': grid_res,
                            'ref_idx': n_tdoa-1,
                            'do_resample': False,
                            'variance_is_toa': False}

        # Iterative Solvers
        x_est_gd, x_est_gd_full, bias_est_gd = tdoa.solvers.gradient_descent(zeta=zeta, **iter_solver_args)
        x_est_ls, x_est_ls_full, bias_est_ls = tdoa.solvers.least_square(zeta=zeta, **iter_solver_args)

        with np.printoptions(precision=3, suppress=True):
            print('GD Est. range bias: (', end='')
            print(*bias_est_gd, sep=', ', end=') m\n')
            print('LS Est. range bias: (', end='')
            print(*bias_est_ls, sep=', ', end=') m\n')

        figs.append(_make_plot(ell_plot, [x_tdoa, x_tgt, x_est_true, x_est_bias, x_est_gd, x_est_ls],
                               ['Sensors', 'Target', 'ML Est.', 'ML Est. w/unc.', 'GD Est.', 'LS Est.']))

    return figs


def example5():
    """
    Executes Example 6.5.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    29 April 2025

    :return: figure handle to generated graphic
    """

    # Set up sensors
    x_tdoa = np.array([[-1, 0, 1],
                       [0, 1, 0]])*1e3
    x_fdoa = x_tdoa
    v_fdoa = np.array([[0, 0, 0],
                       [500, 500, 500]])

    n_dim, n_tdoa = utils.safe_2d_shape(x_tdoa)
    _, n_fdoa = utils.safe_2d_shape(x_fdoa)

    # Generate Random Velocity Errors
    cov_vel = CovarianceMatrix(100**2 * np.eye(n_dim * n_fdoa))
    vel_err = np.reshape(cov_vel.lower() @ np.random.randn(n_dim*n_fdoa, 1), (n_dim, n_fdoa))
    v_fdoa_actual = v_fdoa + vel_err

    # Generate Measurements
    x_source = np.array([-3, 4]) * 1e3
    x_cal = np.array([-2, -1, 0, 1, 2],
                     [-5, -5, -5, -5, -5]) * 1e3
    _, num_cal = utils.safe_2d_shape(x_cal)

    system_args = {'x_aoa': None,
                   'x_tdoa': x_tdoa,
                   'x_fdoa': x_fdoa,
                   'v_fdoa': v_fdoa}
    system_args_truth = system_args
    system_args_truth['v_fdoa'] = v_fdoa_actual

    z = hybrid.model.measurement(x_source=x_source, **system_args_truth)
    z_cal = hybrid.model.measurement(x_source=x_cal, **system_args_truth)

    # Build sensor-level covariance matrix
    err_time = 100e-9
    err_freq = 1
    freq_hz = 10e9
    lam = utils.constants.speed_of_light / freq_hz  # wavelength
    cov_toa = CovarianceMatrix(err_time**2 * np.eye(n_tdoa))
    cov_roa = cov_toa.multiply(utils.constants.speed_of_light**2)
    cov_foa = CovarianceMatrix(err_freq**2 * np.eye(n_fdoa))
    cov_rroa = cov_foa.multiply(lam**2)

    cov_rdoa = cov_roa.resample()
    cov_rrdoa = cov_rroa.resample()
    cov_tf = CovarianceMatrix.block_diagonal(cov_rdoa, cov_rrdoa)

    # Generate Noise
    noise = cov_tf.lower() @ np.random.randn(n_tdoa+n_fdoa-2, n_cal + 1)
    zeta = z + noise[:, 0]
    zeta_cal = z_cal + noise[:, 1:]

    # Estimate Position
    x_init = np.array([0, 5])*1e3
    x_est, x_est_full = hybrid.solvers.gradient_descent(zeta=zeta, cov=cov_tf, x_init=x_init, **system_args)
    x_est_cal, x_est_cal_full, _, _ = hybrid.solvers.gradient_descent(zeta=zeta, cov=cov_tf, x_init=x_init,
                                                                      x_cal=x_cal, zeta_cal=zeta_cal, **system_args)
    
    # Plot Scenario

    # Bonus: FDOA-only Cal

    return []
