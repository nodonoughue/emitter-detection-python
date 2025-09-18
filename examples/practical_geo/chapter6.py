import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from ewgeo.fdoa import FDOAPassiveSurveillanceSystem
from ewgeo.hybrid import HybridPassiveSurveillanceSystem
from ewgeo.tdoa import TDOAPassiveSurveillanceSystem
from ewgeo.triang import DirectionFinder
from ewgeo.utils import make_nd_grid, safe_2d_shape, SearchSpace
from ewgeo.utils.constants import speed_of_light
from ewgeo.utils.covariance import CovarianceMatrix
from ewgeo.utils.geo import calc_range
from ewgeo.utils.unit_conversions import convert

_rad2deg = convert(1, "rad", "deg")
_deg2rad = convert(1, "deg", "rad")


def run_all_examples():
    """
    Run all chapter 6 examples and return a list of figure handles

    :return figs: list of figure handles
    """

    return list(example1()) + list(example2()) + list(example3()) + list(example4()) + list(example5(True))


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

    aoa = DirectionFinder(x=x_aoa, cov=None, do_2d_aoa=False)

    # Define received signals and covariance matrix
    alpha = np.array([5, 10, -5])*_deg2rad  # AOA bias
    psi = aoa.measurement(x_source=x_tgt)
    psi_bias = aoa.measurement(x_source=x_tgt, bias=alpha)
    # Alternatively, we can set aoa's bias to have it persist...
    # aoa.bias = alpha
    # psi_bias = aoa.measurement(x_source=x_tgt)

    # Plot the scenario
    fig = plt.figure()
    hdl_sensors=plt.scatter(x_aoa[0], x_aoa[1], marker='s', label='Sensors')
    hdl_target=plt.scatter(x_tgt[0], x_tgt[1], marker='^', label='Target')
    plt.grid(True)

    # Draw the LOBs
    xy_lobs = aoa.draw_lobs(zeta=psi, x_source=x_tgt, scale=1.5)
    # response is (2, 2, 3, 1) for (n_dims, start:end, num_sensors, num_cases)
    xy_lobs = np.squeeze(xy_lobs).transpose((2, 0, 1)) # reshape to (num_sensors, num_dims, 2)
    label = 'LOB (w/o bias)'
    for xy_lob in xy_lobs:
        plt.plot(xy_lob[0], xy_lob[1], color=hdl_target.get_facecolor(), label=label)
        label = None

    xy_lobs = aoa.draw_lobs(zeta=psi_bias, x_source=x_tgt, scale=1.5)
    # response is (2, 2, 3, 1) for (n_dims, start:end, num_sensors, num_cases)
    xy_lobs = np.squeeze(xy_lobs).transpose((2, 0, 1)) # reshape to (num_sensors, num_dims, 2)
    label = 'LOB (w/bias)'
    for xy_lob in xy_lobs:
        plt.plot(xy_lob[0], xy_lob[1], '--', color=hdl_sensors.get_facecolor(), label=label)
        label = None

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
    _, n_tdoa = safe_2d_shape(x_tdoa)

    cov_pos_full = CovarianceMatrix(.1 * np.eye(2 * n_tdoa))  # position covar; all are IID

    # Generate a random set of TDOA positions
    x_tdoa_actual = x_tdoa + np.reshape(cov_pos_full.sample(), shape=(2, n_tdoa))

    # Generate PSS System
    tdoa = TDOAPassiveSurveillanceSystem(x=x_tdoa, cov=None, ref_idx=None)

    # Generate Measurements
    x_tgt = np.array([6, 3])

    zeta = tdoa.measurement(x_source=x_tgt)
    zeta_unc = tdoa.measurement(x_sensor=x_tdoa_actual, x_source=x_tgt)  # manually specify actual sensor pos.

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
    xy_iso_nominal = tdoa.draw_isochrones(range_diff=zeta_unc, num_pts=101, max_ortho=8)
    xy_iso_actual = tdoa.draw_isochrones(range_diff=zeta_unc, num_pts=101, max_ortho=8, x_sensor=x_tdoa_actual)

    for xy_iso, xy_iso_unc in zip(xy_iso_nominal, xy_iso_actual):
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
    num_dims, n_aoa = safe_2d_shape(x_aoa)

    x_tdoa = np.array([[0., 2., 0.], [2., -1., 0.]])  # avg position (reported)
    _, n_tdoa = safe_2d_shape(x_tdoa)

    # Find matching sensors
    dist = calc_range(x1=x_aoa, x2=x_tdoa)
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
    cov_obj = CovarianceMatrix(cov_pos)

    # Generate a random set of AOA and TDOA positions
    # L=chol(C_beta,'lower'); -- This will fail, because C_beta is not positive
    # definite (it has some eigenvalues that are zero)
    epsilon = np.reshape(cov_obj.sample(), shape=(n_aoa+n_tdoa, num_dims)).T

    # Grab the position offsets and add to the reported TDOA and AOA positions
    x_aoa_true = x_aoa + epsilon[:, :n_aoa]  # first n_dim x n_aoa errors belong to the AOA sensors
    x_tdoa_true = x_tdoa + epsilon[:, n_aoa:]  # latter n_dim x n_tdoa errors belong to the TDOA sensors
    x_sensor_true = np.concatenate((x_aoa_true, x_tdoa_true), axis=1)

    # Let's verify that sensors 2 and 4 are still colocated
    dist_perturbed = calc_range(x1=x_aoa_true, x2=x_tdoa_true)
    assert np.all(np.fabs(dist_perturbed[idx_aoa, idx_tdoa]) < 1e-6), 'Error generating correlated sensor perturbations.'

    # Initialize the PSS
    aoa = DirectionFinder(x=x_aoa, cov=None, do_2d_aoa=False)
    tdoa = TDOAPassiveSurveillanceSystem(x=x_tdoa, cov=None, ref_idx=None)
    hybrid = HybridPassiveSurveillanceSystem(aoa=aoa, tdoa=tdoa, fdoa=None)

    # Generate Measurements
    x_tgt = np.array([6, 3])
    alpha_aoa = np.array([5, 10, -5])*_deg2rad  # AOA bias
    bias = np.concatenate((alpha_aoa, np.zeros((tdoa.num_sensors, ))), axis=0)

    zeta = hybrid.measurement(x_source=x_tgt)
    zeta_unc = hybrid.measurement(x_source=x_tgt, x_sensor=x_sensor_true)
    zeta_unc_bias = hybrid.measurement(x_source=x_tgt, x_sensor=x_sensor_true, bias=bias)
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
    plt.scatter(x_aoa_true[0], x_aoa_true[1], marker='o', label='AOA Sensors (true positions)')
    plt.scatter(x_tdoa_true[0], x_tdoa_true[1], marker='o', label='TDOA Sensors (true positions)')
    plt.scatter(x_tgt[0], x_tgt[1],marker='^', color='k', label='Target')
    plt.grid(True)

    # Draw the Isochrones and LOBs -- Truth
    xy_lobs = aoa.draw_lobs(zeta=zeta_unc_bias[:n_aoa], x_source=x_tgt, scale=1.5)
    xy_lobs = np.squeeze(xy_lobs).transpose((2, 0, 1))
    # xy_lob_bias = aoa.draw_lobs(x_sensor=x_aoa_true, psi=zeta_unc_bias[:n_aoa], x_source=x_tgt, scale=1.5)
    label = 'LOB (nominal positions w/bias)'
    for xy_lob in xy_lobs:
        plt.plot(xy_lob[0],xy_lob[1], color=hdl_nominal_a.get_facecolor(), label=label)
        label = None

    label_nominal = 'Isochrone (nominal positions w/bias)'
    # label_true = 'Isochrone (true positions w/bias)'
    xy_isos = tdoa.draw_isochrones(range_diff=zeta_unc_bias[aoa.num_measurements:], num_pts=101, max_ortho=8)
    for xy_iso in xy_isos:
        plt.plot(xy_iso[0], xy_iso[1], color=hdl_nominal_t.get_facecolor(), label=label_nominal)
        label_nominal = None

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
    n_dim, n_tdoa = safe_2d_shape(x_tdoa)

    tdoa_bias = np.array([10, 30, -20, 60])  # TOA bias

    err_toa = 100e-9
    cov_toa = CovarianceMatrix((err_toa**2)*np.eye(n_tdoa))
    cov_roa = cov_toa.multiply(val=speed_of_light**2, overwrite=False)

    tdoa = TDOAPassiveSurveillanceSystem(x=x_tdoa, cov=cov_roa, ref_idx=None, variance_is_toa=False)

    # Generate Measurements
    x_tgt = np.array([6, 3])*1e3

    z = tdoa.measurement(x_source=x_tgt, bias=tdoa_bias)  # free of pos unc, w/bias
    z_true = tdoa.measurement(x_source=x_tgt)  # free of pos unc, w/o bias
    noise = tdoa.cov.sample()
    zeta = z + noise
    zeta_true = z_true + noise

    # Compute Log Likelihood
    x_ctr = np.array([5e3, 5e3])
    search_size = 5e3
    grid_res = .05e3
    search_space = SearchSpace(x_ctr=x_ctr,
                               max_offset=search_size,
                               epsilon=grid_res)
    x_set, x_grid, out_shape = make_nd_grid(search_space)
    extent = ((x_ctr[0]-search_size)/1e3, (x_ctr[0]+search_size)/1e3,
              (x_ctr[1]-search_size)/1e3, (x_ctr[1]+search_size)/1e3)

    ell = tdoa.log_likelihood(zeta=zeta, x_source=x_set)
    ell_true = tdoa.log_likelihood(zeta=zeta_true, x_source=x_set)

    # Plot Scenario
    levels = [-1000, -100, -50, -20, -10, -5, 0]
    def _make_plot(this_ell, x_plot, label_set, title=None):
        this_fig, this_ax = plt.subplots()

        cmap = matplotlib.colormaps['viridis']  # Choose a colormap
        cmap.set_under('k', alpha=0)  # Set values below vmin to transparent black

        hdl = plt.imshow(this_ell, origin='lower', cmap=cmap, extent=extent)
        plt.clim(-100, 0)
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

    ml_search = SearchSpace(x_ctr=x_ctr,
                            max_offset=search_size,
                            epsilon=grid_res)
    x_est_true, _, _ = tdoa.max_likelihood(zeta=zeta_true, search_space=ml_search, print_progress=True)
    x_est, _, _ = tdoa.max_likelihood(zeta=zeta, search_space=ml_search, print_progress=True)

    print('True ML Est.: ({:.2f}, {:.2f}) km, error: {:.2f} km'.format(x_est_true[0]/1e3, x_est_true[1]/1e3,
                                                                      np.linalg.norm(x_est_true-x_tgt)/1000.))
    print('Biased ML Est.: ({:.2f}, {:.2f}) km, error: {:.2f} km'.format(x_est[0]/1e3, x_est[1]/1e3,
                                                                         np.linalg.norm(x_est-x_tgt)/1000))

    # ML Solver with Bias Estimation
    bias_search = SearchSpace(x_ctr=np.zeros((tdoa.num_sensors, )),
                              max_offset=np.array([80]*tdoa.num_measurements+[0]), # assume the ref sensor has no bias
                              epsilon=10)

    x_est_bias, _, _, th_est  = tdoa.max_likelihood_uncertainty(zeta=zeta,
                                                                source_search=ml_search,
                                                                bias_search=bias_search,
                                                                do_sensor_bias=True,
                                                                do_sensor_pos=False,
                                                                do_sensor_vel=False,
                                                                print_progress=True)

    err_km = np.linalg.norm(x_est_bias-x_tgt)/1000
    print('ML Est. w/Uncertainty: ({:.2f}, {:.2f}) km, error: {:.2f} km'.format(x_est_bias[0]/1e3,
                                                                                x_est_bias[1]/1e3,
                                                                                err_km))

    with np.printoptions(precision=3, suppress=True):
        print('True range bias: (', end='')
        print(*tdoa_bias, sep=', ', end=') m\n')
        print('Estimated range bias: (', end='')
        print(*th_est['bias'][:n_tdoa], sep=', ', end=') m\n')

    # Plot Solutions
    figs.append(_make_plot(ell_plot, [x_tdoa, x_tgt, x_est_true],
                           ['Sensors', 'Target', 'ML Est.']))
    figs.append(_make_plot(ell_plot, [x_tdoa, x_tgt, x_est, x_est_bias],
                           ['Sensors', 'Target', 'ML Est.', 'ML Est. w/uncertainty']))

    if do_iterative:
        iter_solver_args = {'x_init': x_ctr,
                            'epsilon': grid_res}

        # Iterative Solvers
        x_est_gd, x_est_gd_full, bias_est_gd = tdoa.gradient_descent(zeta=zeta, **iter_solver_args)
        x_est_ls, x_est_ls_full, bias_est_ls = tdoa.least_square(zeta=zeta, **iter_solver_args)

        with np.printoptions(precision=3, suppress=True):
            print('GD Est. range bias: (', end='')
            print(*bias_est_gd, sep=', ', end=') m\n')
            print('LS Est. range bias: (', end='')
            print(*bias_est_ls, sep=', ', end=') m\n')

        figs.append(_make_plot(ell_plot, [x_tdoa, x_tgt, x_est_true, x_est_bias, x_est_gd, x_est_ls],
                               ['Sensors', 'Target', 'ML Est.', 'ML Est. w/unc.', 'GD Est.', 'LS Est.']))

    return figs


def example5(do_vel_only_cal=False):
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

    n_dim, n_tdoa = safe_2d_shape(x_tdoa)
    _, n_fdoa = safe_2d_shape(x_fdoa)

    # Generate Random Velocity Errors
    cov_vel = CovarianceMatrix(100**2 * np.eye(n_dim * n_fdoa))
    vel_err = np.reshape(cov_vel.sample(), (n_dim, n_fdoa))
    v_fdoa_actual = v_fdoa + vel_err

    # Build sensor-level covariance matrix
    err_time = 100e-9
    err_freq = 1
    freq_hz = 10e9
    lam = speed_of_light / freq_hz  # wavelength
    cov_toa = CovarianceMatrix(err_time**2 * np.eye(n_tdoa))
    cov_roa = cov_toa.multiply(speed_of_light**2, overwrite=False)
    cov_foa = CovarianceMatrix(err_freq**2 * np.eye(n_fdoa))
    cov_rroa = cov_foa.multiply(lam**2, overwrite=False)

    cov_rdoa = cov_roa.resample()
    cov_rrdoa = cov_rroa.resample()
    cov_tf = CovarianceMatrix.block_diagonal(cov_rdoa, cov_rrdoa)

    # Construct PSS Object
    tdoa = TDOAPassiveSurveillanceSystem(x=x_tdoa, cov=cov_roa, variance_is_toa=False, ref_idx=None)
    fdoa = FDOAPassiveSurveillanceSystem(x=x_fdoa, vel=v_fdoa, cov=cov_rroa, ref_idx=None)
    hybrid = HybridPassiveSurveillanceSystem(tdoa=tdoa, fdoa=fdoa)

    # Generate Measurements using the True Sensor Velocities
    x_source = np.array([-3, 4]) * 1e3
    x_cal = np.array([[-2, -1, 0, 1, 2],
                      [-5, -5, -5, -5, -5]]) * 1e3
    _, num_cal = safe_2d_shape(x_cal)

    z = hybrid.measurement(x_source=x_source, v_sensor=v_fdoa_actual)
    z_cal = hybrid.measurement(x_source=x_cal, v_sensor=v_fdoa_actual)

    # Generate Noise
    noise = cov_tf.sample(num_samples=None)  # providing 'None' ensured a 1d vector response
    noise_cal = cov_tf.sample(num_samples=num_cal)
    zeta = z + noise
    zeta_cal = z_cal + noise_cal

    # Estimate Position
    x_init = np.array([0, 5])*1e3
    cal_data = {'zeta_cal': zeta_cal,
                'x_cal': x_cal,
                'do_pos_cal': True,
                'do_vel_cal': True,
                'do_bias_cal': False}  # don't bother calibrating across measurement biases; let's just do pos/vel
    _, x_est = hybrid.gradient_descent(zeta=zeta, x_init=x_init)
    _, x_est_cal = hybrid.gradient_descent(zeta=zeta, x_init=x_init, cal_data=cal_data)
    
    # Plot Scenario
    fig = plt.figure()
    tdoa.plot_sensors(marker='^', label='Sensors', clip_on=False, zorder=3)
    plt.scatter(x_source[0], x_source[1], marker='o', label='Target', clip_on=False, zorder=3)
    plt.scatter(x_cal[0], x_cal[1], marker='v', label='Calibration Sources', clip_on=False, zorder=3)

    # Plot Iterative Solutions
    plt.scatter(x_init[0], x_init[1], marker='x', color='k', label='Initial Estimate')
    plt.plot(x_est[0], x_est[1], linestyle='--', marker='s', markevery=[-1], label='Solution (w/o cal)')
    plt.plot(x_est_cal[0], x_est_cal[1], linestyle='--', marker='s', markevery=[-1], label='Solution (w/pos & vel cal)')

    # Bonus: FDOA-only Cal
    if do_vel_only_cal:
        # To restrict calibration to velocity alone, we simply adjust the flags in cal_data
        cal_data['do_pos_cal'] = False  # turn off sensor position calibration; data will be used only for velocity cal
        _, x_est_fdoa_cal = hybrid.gradient_descent(zeta=zeta, x_init=x_init, cal_data=cal_data)

        # Plot the scenario
        plt.plot(x_est_fdoa_cal[0], x_est_fdoa_cal[1], linestyle='-.', marker='s', markevery=[-1],
                 label='Solution (w/velocity cal)')

    return fig,


if __name__ == '__main__':
    run_all_examples()
    plt.show()
