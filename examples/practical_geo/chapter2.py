import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
import utils
import triang
import tdoa
import fdoa
import hybrid
import time

_rad2deg = 180.0/np.pi
_deg2rad = np.pi/180.0

# Sensor and Source Positions
# These sensor positions are used for all three examples in Chapter 2.
x_source = np.array([3, 3]) * 1e3
x_aoa = np.array([[4], [0]]) * 1e3
x_tdoa = np.array([[1, 3], [0, 0.5]]) * 1e3
x_fdoa = np.array([[0, 0], [1, 2]]) * 1e3
v_fdoa = np.array([[1, 1], [-1, -1]]) * np.sqrt(0.5) * 300  # 300 m/s, at -45 deg heading


def run_all_examples():
    """
    Run all chapter 2 examples and return a list of figure handles

    :return figs: list of figure handles
    """

    return [example1(), example2(), example3()]


def _plt_markers(do_source=True, do_aoa=False, do_tdoa=False, do_fdoa=False, colors=('w', 'k', 'k', 'k')):
    """
    Add markers for Source, AOA, TDOA, and FDOA sensors to a plot; optionally specify colors

    Nicholas O'Donoughue
    16 January 2025

    :param colors: 4-element Tuple of colors to use for markers (source, aoa, tdoa, fdoa). Default=('w', 'k', 'k', 'k')
    """
    if do_source:
        plt.scatter(x_source[0], x_source[1], marker='x', color=colors[0], label='Target', clip_on=False)
    if do_aoa:
        plt.scatter(x_aoa[0], x_aoa[1], marker='o', color=colors[1], label='AOA Sensors', clip_on=False)
    if do_tdoa:
        plt.scatter(x_tdoa[0], x_tdoa[1], marker='s', color=colors[2], label='TDOA Sensors', clip_on=False)
    if do_fdoa:
        plt.scatter(x_fdoa[0], x_fdoa[1], marker='^', color=colors[3], label='FDOA Sensors', clip_on=False)
        for this_x, this_v in zip(x_fdoa.T, v_fdoa.T):  # transpose so the loop steps over sensors, not dimensions
            plt.arrow(x=this_x[0], y=this_x[1],
                      dx=this_v[0], dy=this_v[1],
                      width=.01e3, head_width=.05e3,
                      color=colors[3])


def _make_err_covariance(err_aoa=None, err_time=None, err_freq=None, f0=1.0, tdoa_ref_idx=None, fdoa_ref_idx=None):
    """

    :param err_aoa: Angle-of-Arrival measurement error (in deg)
    :param err_time: Time-of-Arrival measurement error (in seconds)
    :param err_freq: Frequency measurement error (in Hz)
    :param f0: Operating frequency (in Hz)
    :param tdoa_ref_idx: Index of TDOA reference sensor
    :param fdoa_ref_idx: Index of FDOA reference sensor
    :return cov_z: Measurement error covariance (angle, time, freq)
    :return cov_z_out: Sensor pair error covariance (angle, time-difference, freq-difference)
    """
    # Count the number of sensors in each type
    _, num_aoa = utils.safe_2d_shape(x_aoa)
    _, num_tdoa = utils.safe_2d_shape(x_tdoa)
    _, num_fdoa = utils.safe_2d_shape(x_fdoa)

    # Define Error Covariance Matrix
    cov_arr = []  # list for keeping track of which covariance matrices are defined
    if err_aoa is not None:
        cov_psi = (err_aoa * _deg2rad) ** 2.0 * np.eye(num_aoa) # rad ^ 2
        cov_arr.append(cov_psi)  # Add to cov_arr, for use in block_diag

    if err_time is not None:
        err_r = err_time * utils.constants.speed_of_light
        cov_r = (err_r ** 2.0) * np.eye(num_tdoa)  # m ^ 2
        cov_arr.append(cov_r)

    if err_freq is not None:
        rr_err = err_freq * utils.constants.speed_of_light / f0  # (m/s)
        cov_rr = (rr_err ** 2) * np.eye(num_fdoa)  # (m/s)^2
        cov_arr.append(cov_rr)

    cov_z = block_diag(*cov_arr)  # Sensor-level errors. Note the *cov_arr tells python to unpack the list
    cov_z_out = hybrid.model.resample_hybrid_covariance_matrix(cov_z, x_aoa, x_tdoa, x_fdoa, tdoa_ref_idx, fdoa_ref_idx)

    return cov_z, cov_z_out


def example1(colors=None):
    """
    Executes Example 2.1.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    9 January 2025

    :param colors: set of colors for plotting
    :return: figure handle to generated graphic
    """

    # For the initial laydown plot, borrow code from Figure 13.1 of the 2019 text
    if colors is None:
        colormap = plt.get_cmap("tab10")
        colors = (colormap(0), colormap(1), colormap(2), colormap(3))

    # Figure 1, System Drawing

    # Draw Geometry
    fig1 = plt.figure()
    _plt_markers(do_aoa=True, do_tdoa=True, do_fdoa=True, colors=colors)

    # True Measurements
    psi_act = triang.model.measurement(x_sensor=x_aoa, x_source=x_source)
    range_diff = tdoa.model.measurement(x_sensor=x_tdoa, x_source=x_source, ref_idx=None)
    velocity_diff = fdoa.model.measurement(x_sensor=x_fdoa, v_sensor=v_fdoa, x_source=x_source, v_source=None,
                                           ref_idx=None)

    # Draw DF line
    xy_lob = triang.model.draw_lob(x_sensor=x_aoa, psi=psi_act, x_source=x_source, scale=5)
    plt.plot(xy_lob[0], xy_lob[1], color=colors[1], linestyle='-', label='AOA Solution')

    # Draw isochrone
    # Transpose the x_tdoa array before indexing; so [0] and [1] refer to sensors, not dimensions
    xy_isochrone = tdoa.model.draw_isochrone(x_tdoa.T[1], x_tdoa.T[0], range_diff=range_diff, num_pts=1000, max_ortho=5e3)
    plt.plot(xy_isochrone[0], xy_isochrone[1], color=colors[2], linestyle=':', label='TDOA Solution')

    # Draw isodoppler line
    # Transpose the x_fdoa|v_fdoa arrays before indexing; so [0] and [1] refer to sensors, not dimensions
    xy_isodoppler = fdoa.model.draw_isodop(x1=x_fdoa.T[1], v1=v_fdoa.T[1], x2=x_fdoa.T[0], v2=v_fdoa.T[0],
                                                        vdiff=velocity_diff, num_pts=1000, max_ortho=5e3)
    plt.plot(xy_isodoppler[0], xy_isodoppler[1], color=colors[3], linestyle='-.', label='FDOA Solution')

    # Adjust Plot Display
    plt.ylim([-0.5e3, 4e3])
    plt.xlim([-.5e3, 5.5e3])
    plt.legend(loc='upper right')

    # Remove the axes for a clean image
    plt.axis('off')

    # --- Compute Variances and Print ---
    c = utils.constants.speed_of_light
    err_aoa = 3  # deg
    cov_psi = (err_aoa * _deg2rad) ** 2  # rad^2
    print('AOA Measurement: {:.2f} deg'.format(psi_act[0] * _rad2deg))
    print('AOA Covariance: {:.4f} rad^2'.format(cov_psi))

    err_time = 1e-7  # 100 ns timing error
    err_r = err_time * c
    _, num_tdoa = utils.safe_2d_shape(x_tdoa)
    cov_r = err_r ** 2 * np.eye(num_tdoa)  # m^2
    print('TDOA Measurement: {:.2f} m'.format(range_diff[0]))
    print('TDOA Covariance:')
    print('{} m^2'.format(np.matrix(cov_r)))

    freq_err = 10  # Hz
    f0 = 1e9  # Hz
    rr_err = freq_err * c / f0  # (m/s)
    _, num_fdoa = utils.safe_2d_shape(x_fdoa)
    cov_rr = rr_err ** 2 * np.eye(num_fdoa)  # (m/s)^2
    print('FDOA Measurement: {:.2f} m/s'.format(velocity_diff[0]))
    print('FDOA Covariance:')
    print('{} m^2/s^2'.format(np.matrix(cov_rr)))

    z = hybrid.model.measurement(x_aoa=x_aoa, x_tdoa=x_tdoa, x_fdoa=x_fdoa, v_fdoa=v_fdoa, x_source=x_source)
    # cov_z = block_diag([cov_psi], cov_r, cov_rr)
    cov_z, _ = _make_err_covariance(err_aoa, err_time, freq_err, f0)

    # Set Up Search Grid
    x_grid = np.arange(-0.5, 5.5, 0.02) * 1e3
    y_grid = np.arange(0, 4, 0.02) * 1e3
    xx, yy = np.meshgrid(x_grid, y_grid)
    x_test_pos = np.vstack((xx.ravel(), yy.ravel()))
    grid_extent=(float(x_grid[0]), float(x_grid[-1]), float(y_grid[0]), float(y_grid[-1]))

    # Log-Likelihood Figure Generator
    def _make_subfigure(ell, do_aoa=False, do_tdoa=False, do_fdoa=False):
        _fig, _ax = plt.subplots()
        plt.imshow(ell, extent=grid_extent, origin='lower', cmap=cm['viridis_r'], vmin=-20, vmax=0)
        plt.colorbar()
        _plt_markers(do_aoa=do_aoa, do_tdoa=do_tdoa, do_fdoa=do_fdoa)

        plt.ylim([0, 4e3])
        plt.xlim([-.5e3, 5.5e3])
        plt.clim(-20, 0)
        plt.legend(loc='upper right')

        return _fig

    # Plot AOA Likelihood
    ell_aoa = triang.model.log_likelihood(x_aoa=x_aoa, psi=psi_act, cov=cov_psi, x_source=x_test_pos,
                                          do_2d_aoa=False).reshape(xx.shape)
    fig2 = _make_subfigure(ell_aoa, do_aoa=True)

    # TDOA
    ell_tdoa = tdoa.model.log_likelihood(x_sensor=x_tdoa, rho=range_diff, cov=cov_r,
                                         x_source=x_test_pos, do_resample=True).reshape(xx.shape)
    fig3 = _make_subfigure(ell_tdoa, do_tdoa=True)

    # FDOA
    ell_fdoa = fdoa.model.log_likelihood(x_sensor=x_fdoa, v_sensor=v_fdoa, rho_dot=velocity_diff, cov=cov_rr,
                                         x_source=x_test_pos, do_resample=True).reshape(xx.shape)
    fig4 = _make_subfigure(ell_fdoa, do_fdoa=True)

    # Hybrid
    ell_hybrid = hybrid.model.log_likelihood(x_aoa=x_aoa, x_tdoa=x_tdoa, x_fdoa=x_fdoa, v_fdoa=v_fdoa,
                                             zeta=z, cov=cov_z, x_source=x_test_pos, v_source=None,
                                             do_resample=True).reshape(xx.shape)
    fig5 = _make_subfigure(ell_hybrid, do_aoa=True, do_tdoa=True, do_fdoa=True)

    # Package figure handles
    return fig1, fig2, fig3, fig4, fig5


def example2(colors=None):
    """
    Executes Example 2.2.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    16 January 2025

    :return: figure handle to generated graphic
    """

    # For the initial laydown plot, borrow code from Figure 13.1 of the 2019 text
    if colors is None:
        colormap = plt.get_cmap("tab10")
        colors = (colormap(0), colormap(1), colormap(2), colormap(3))



    # Take Hybrid measurement and Define combined covariance matrix
    tdoa_ref_idx=1
    fdoa_ref_idx=1
    z = hybrid.model.measurement(x_source=x_source, x_aoa=x_aoa, x_tdoa=x_tdoa, x_fdoa=x_fdoa, v_fdoa=v_fdoa,
                                 tdoa_ref_idx=tdoa_ref_idx, fdoa_ref_idx=fdoa_ref_idx)

    # Generate Error Covariance
    err_aoa = 3  # deg
    err_time = 1e-7  # 100 ns timing error
    err_freq = 10  # Hz
    f0 = 1e9  # Hz
    cov_z, cov_z_out = _make_err_covariance(err_aoa, err_time, err_freq, f0,
                                            tdoa_ref_idx=tdoa_ref_idx, fdoa_ref_idx=fdoa_ref_idx)

    # Generate Random Noise
    cov_lower = np.linalg.cholesky(cov_z_out, upper=False)
    num_msmt, _ = utils.safe_2d_shape(cov_lower)
    noise = cov_lower @ np.random.randn(num_msmt, )

    # Combine Noise with Perfect Measurement
    zeta = z + noise

    # ---- Set Up Solution Parameters ----
    # ML Search Parameters
    x_ctr = np.array([2.5, 2.5]) * 1e3
    grid_size = np.array([5, 5]) * 1e3
    grid_res = 25  # meters, grid resolution

    # GD and LS Search Parameters
    x_init = np.array([1, 1]) * 1e3
    epsilon = grid_res  # desired convergence step size, same as grid resolution
    max_num_iterations = 100
    force_full_calc = True
    plot_progress = False

    # ---- Apply Various Solvers ----
    # ML Solution
    x_ml, _, _ = hybrid.solvers.max_likelihood(zeta=zeta, cov=cov_z_out, x_aoa=x_aoa, x_tdoa=x_tdoa,
                                               x_fdoa=x_fdoa, v_fdoa=v_fdoa, x_ctr=x_ctr, search_size=grid_size,
                                               epsilon=epsilon, do_resample=False, cov_is_inverted=False)

    # GD Solution
    x_gd, x_gd_full = hybrid.solvers.gradient_descent(zeta=zeta, cov=cov_z_out, x_aoa=x_aoa, x_tdoa=x_tdoa,
                                                      x_fdoa=x_fdoa, v_fdoa=v_fdoa, x_init=x_init, epsilon=epsilon,
                                                      max_num_iterations=max_num_iterations, do_resample=False,
                                                      force_full_calc=force_full_calc, plot_progress=plot_progress)

    # LS Solution
    x_ls, x_ls_full = hybrid.solvers.least_square(zeta=zeta, cov=cov_z_out, x_aoa=x_aoa, x_tdoa=x_tdoa,
                                                  x_fdoa=x_fdoa, v_fdoa=v_fdoa, x_init=x_init, epsilon=epsilon,
                                                  max_num_iterations=max_num_iterations, do_resample=False,
                                                  force_full_calc=force_full_calc, plot_progress=plot_progress)

    # ---- Plot Results ----
    def _make_plot():
        fig, _ = plt.subplots()
        _plt_markers(do_aoa=True, do_tdoa=True, do_fdoa=True, colors=colors)

        # Plot Closed-Form Solution
        plt.scatter(x_ml[0], x_ml[1], marker='v', label='Maximum Likelihood')

        # Plot Iterative Solutions
        plt.scatter(x_init[0], x_init[1], marker='x', color='k', label='Initial Estimate')
        plt.plot(x_ls_full[0, :], x_ls_full[1, :], linestyle=':', label='Least Squares')
        plt.plot(x_gd_full[0, :], x_gd_full[1, :], linestyle='--', label='Grad Descent')

        plt.legend(loc='best')

        return fig

    fig_full = _make_plot()
    plt.xlim([-.5e3, 5.5e3])
    plt.ylim([0, 4e3])

    fig_zoom = _make_plot()
    plt.xlim([2e3, 3.5e3])
    plt.ylim([2e3, 3.5e3])

    return fig_full, fig_zoom


def example3(rng=np.random.default_rng(), colors=None):
    """
    Executes Example 2.3.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    16 January 2025

    :param rng: random number generator
    :param colors:
    :return: figure handle to generated graphic
    """

    # For the initial laydown plot, borrow code from Figure 13.1 of the 2019 text
    if colors is None:
        colormap = plt.get_cmap("tab10")
        colors = (colormap(0), colormap(1), colormap(2), colormap(3))

    # Generate Error Covariance
    err_aoa = 3  # deg
    err_time = 1e-7  # 100 ns timing error
    err_freq = 10  # Hz
    f0 = 1e9  # Hz
    tdoa_ref_idx = 1
    fdoa_ref_idx = 1
    cov_z, cov_z_out = _make_err_covariance(err_aoa, err_time, err_freq, f0,
                                            tdoa_ref_idx=tdoa_ref_idx, fdoa_ref_idx=fdoa_ref_idx)

    # Take Hybrid measurement
    z = hybrid.model.measurement(x_source=x_source, x_aoa=x_aoa, x_tdoa=x_tdoa, x_fdoa=x_fdoa, v_fdoa=v_fdoa,
                                 tdoa_ref_idx=tdoa_ref_idx, fdoa_ref_idx=fdoa_ref_idx)

    # Generate Random Noise
    cov_lower = np.linalg.cholesky(cov_z_out, upper=False)
    num_msmt, _ = utils.safe_2d_shape(cov_lower)

    # ---- Set Up Solution Parameters ----
    # ML Search Parameters
    ml_args = {'x_ctr' : np.array([2.5, 2.5]) * 1e3,
               'search_size' : np.array([5, 5]) * 1e3,
               'epsilon' : 25}  # meters, grid resolution

    # GD and LS Search Parameters
    gd_ls_args = {'x_init' : np.array([1, 1]) * 1e3,
                  'epsilon' : ml_args['epsilon'],  # desired convergence step size, same as grid resolution
                  'max_num_iterations' : 50,
                  'force_full_calc' : True,
                  'plot_progress' : False}

    res = _mc_iteration(z, num_msmt, tdoa_ref_idx, fdoa_ref_idx, cov_z_out, cov_lower, rng, ml_args, gd_ls_args)

    # ---- Estimate Error Bounds ----
    # CRLB
    crlb = hybrid.perf.compute_crlb(x_source=x_source, x_aoa=x_aoa, x_tdoa=x_tdoa, x_fdoa=x_fdoa, v_fdoa=v_fdoa,
                                    cov=cov_z_out, cov_is_inverted=False, do_resample=False)
    print('CRLB: {}'.format(crlb))

    # RMSE
    rmse_crlb = np.sqrt(np.trace(crlb))
    print('RMSE: {:.2f} km'.format(rmse_crlb/1e3))

    # CEP50
    cep50_crlb = utils.errors.compute_cep50(crlb)
    print('CEP50: {:.2f} km'.format(cep50_crlb/1e3))

    # 90% Error Ellipse
    conf_interval = 90
    crlb_ellipse = utils.errors.draw_error_ellipse(x=x_source, covariance=crlb, num_pts=101,
                                                   conf_interval=conf_interval)

    # ---- Plot Results ----
    x_init = gd_ls_args['x_init']
    x_ml = res['ml']
    x_ls = res['ls']
    x_gd = res['gd']

    def _make_plot():
        fig, _ = plt.subplots()
        _plt_markers(do_aoa=True, do_tdoa=True, do_fdoa=True, colors=colors)

        # Plot Closed-Form Solution
        plt.scatter(x_ml[0], x_ml[1], marker='v', label='Maximum Likelihood')

        # Plot Iterative Solutions
        plt.scatter(x_init[0], x_init[1], marker='x', color='k', label='Initial Estimate')
        plt.plot(x_ls[0, :], x_ls[1, :], linestyle=':', label='Least Squares')
        plt.plot(x_gd[0, :], x_gd[1, :], linestyle='--', label='Grad Descent')

        # Overlay Error Ellipse
        plt.plot(crlb_ellipse[0, :], crlb_ellipse[1, :], linestyle='--', color='k',
                 label='{:d}% Error Ellipse'.format(conf_interval))
        plt.legend(loc='best')

        return fig

    fig_full = _make_plot()
    plt.xlim([-.5e3, 5.5e3])
    plt.ylim([0, 4e3])

    fig_zoom = _make_plot()
    plt.xlim([2e3, 3.5e3])
    plt.ylim([2e3, 3.5e3])

    return fig_full, fig_zoom

def example3_mc(rng=np.random.default_rng(), colors=None):
    """
    Executes a modified version of Example 2.3 with Monte-Carlo trials, as discussed in the video walkthrough of
    Example 2.3.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    16 January 2025

    :param rng: random number generator
    :param colors:
    :return: figure handle to generated graphic
    """

# For the initial laydown plot, borrow code from Figure 13.1 of the 2019 text
    if colors is None:
        colormap = plt.get_cmap("tab10")
        colors = (colormap(0), colormap(1), colormap(2), colormap(3))

    # Generate Error Covariance
    err_aoa = 3  # deg
    err_time = 1e-7  # 100 ns timing error
    err_freq = 10  # Hz
    f0 = 1e9  # Hz
    tdoa_ref_idx = 1
    fdoa_ref_idx = 1
    cov_z, cov_z_out = _make_err_covariance(err_aoa, err_time, err_freq, f0,
                                            tdoa_ref_idx=tdoa_ref_idx, fdoa_ref_idx=fdoa_ref_idx)

    # Take Hybrid measurement
    z = hybrid.model.measurement(x_source=x_source, x_aoa=x_aoa, x_tdoa=x_tdoa, x_fdoa=x_fdoa, v_fdoa=v_fdoa,
                                 tdoa_ref_idx=tdoa_ref_idx, fdoa_ref_idx=fdoa_ref_idx)

    # Pre-compute covariance matrix decomposition for noise generation
    cov_lower = np.linalg.cholesky(cov_z_out, upper=False)
    num_msmt, _ = utils.safe_2d_shape(cov_lower)

    # ---- Set Up Solution Parameters ----
    # ML Search Parameters
    ml_args = {'x_ctr' : np.array([2.5, 2.5]) * 1e3,
               'search_size' : np.array([5, 5]) * 1e3,
               'epsilon' : 25}  # meters, grid resolution

    # GD and LS Search Parameters
    gd_ls_args = {'x_init' : np.array([1, 1]) * 1e3,
                  'epsilon' : ml_args['epsilon'],  # desired convergence step size, same as grid resolution
                  'max_num_iterations' : 50,
                  'force_full_calc' : True,
                  'plot_progress' : False}

    # Monte Carlo Iteration
    num_mc_trials = 100
    rmse_ml = np.zeros((num_mc_trials, ))
    rmse_gd = np.zeros((num_mc_trials, gd_ls_args['max_num_iterations']))
    rmse_ls = np.zeros((num_mc_trials, gd_ls_args['max_num_iterations']))

    print('Performing Monte Carlo simulation...')
    t_start = time.perf_counter()

    iterations_per_marker = 1
    markers_per_row = 40
    iterations_per_row = markers_per_row * iterations_per_marker
    for idx in np.arange(num_mc_trials):
        utils.print_progress(num_mc_trials, idx, iterations_per_marker, iterations_per_row, t_start)

        res = _mc_iteration(z, num_msmt, tdoa_ref_idx, fdoa_ref_idx, cov_z_out, cov_lower, rng, ml_args, gd_ls_args)

        rmse_ml[idx] = np.linalg.norm(res['ml']-x_source)
        rmse_gd[idx, :] = np.linalg.norm(res['gd']-x_source[:, np.newaxis], axis=0)
        rmse_ls[idx, :] = np.linalg.norm(res['ls']-x_source[:, np.newaxis], axis=0)

    print('done')
    t_elapsed = time.perf_counter() - t_start
    utils.print_elapsed(t_elapsed)

    # Compute average error across Monte Carlo Iterations
    rmse_avg_ml = np.sqrt(np.sum(rmse_ml**2)/num_mc_trials)
    rmse_avg_gd = np.sqrt(np.sum(rmse_gd**2, axis=0)/num_mc_trials)
    rmse_avg_ls = np.sqrt(np.sum(rmse_ls**2, axis=0)/num_mc_trials)

    fig_err = plt.figure()
    x_arr = np.arange(gd_ls_args['max_num_iterations'])
    plt.plot(x_arr, rmse_avg_ml*np.ones_like(x_arr), label='Maximum Likelihood')
    plt.plot(x_arr, rmse_avg_gd, label='Gradient Descent')
    plt.plot(x_arr, rmse_avg_ls, label='Least Square')

    plt.xlabel('Number of Iterations')
    plt.ylabel('RMSE [m]')
    plt.title('Monte Carlo Geolocation Results')

    # ---- Estimate Error Bounds ----
    # CRLB
    crlb = hybrid.perf.compute_crlb(x_source=x_source, x_aoa=x_aoa, x_tdoa=x_tdoa, x_fdoa=x_fdoa, v_fdoa=v_fdoa,
                                    cov=cov_z_out, cov_is_inverted=False, do_resample=False)
    rmse_crlb = np.sqrt(np.trace(crlb))
    plt.plot(x_arr, rmse_crlb*np.ones_like(x_arr), '--', color='k', label='CRLB')

    # CEP50
    cep50_crlb = utils.errors.compute_cep50(crlb)
    print('CEP50: {:.2f} km'.format(cep50_crlb/1e3))

    # 90% Error Ellipse
    conf_interval = 90
    crlb_ellipse = utils.errors.draw_error_ellipse(x=x_source, covariance=crlb, num_pts=101,
                                                   conf_interval=conf_interval)

    # ---- Plot Results ----
    x_init = gd_ls_args['x_init']
    x_ml = res['ml']
    x_ls = res['ls']
    x_gd = res['gd']

    def _make_plot():
        fig, _ = plt.subplots()
        _plt_markers(do_aoa=True, do_tdoa=True, do_fdoa=True, colors=colors)

        # Plot Closed-Form Solution
        plt.scatter(x_ml[0], x_ml[1], marker='v', label='Maximum Likelihood')

        # Plot Iterative Solutions
        plt.scatter(x_init[0], x_init[1], marker='x', color='k', label='Initial Estimate')
        plt.plot(x_ls[0, :], x_ls[1, :], linestyle=':', label='Least Squares')
        plt.plot(x_gd[0, :], x_gd[1, :], linestyle='--', label='Grad Descent')

        # Overlay Error Ellipse
        plt.plot(crlb_ellipse[0, :], crlb_ellipse[1, :], linestyle='--', color='k',
                 label='{:d}% Error Ellipse'.format(conf_interval))
        plt.legend(loc='best')

        return fig

    fig_full = _make_plot()
    plt.xlim([-.5e3, 5.5e3])
    plt.ylim([0, 4e3])

    return fig_err, fig_full

def _mc_iteration(z, num_measurements, tdoa_ref_idx, fdoa_ref_idx, covar, covar_lower, rng, ml_args, gd_ls_args):
    """
    Executes a single iteration of the Monte Carlo simulation in Example 2.3.

    :return estimates: Dictionary with estimated target position using several algorithms.  Fields are:
                ml: Maximum Likelihood solution
                gd: Gradient Descent solution
                ls: Least Square solution

    Nicholas O'Donoughue
    16 January 2025
    """

    # Generate a random measurement
    zeta = z + covar_lower @ rng.standard_normal(size=(num_measurements, ))

    # ---- Apply Various Solvers ----
    # ML Solution
    x_ml, _, _ = hybrid.solvers.max_likelihood(zeta=zeta, cov=covar, x_aoa=x_aoa, x_tdoa=x_tdoa,
                                               x_fdoa=x_fdoa, v_fdoa=v_fdoa, tdoa_ref_idx=tdoa_ref_idx,
                                               fdoa_ref_idx=fdoa_ref_idx, do_resample=False, cov_is_inverted=False,
                                               **ml_args)

    # GD Solution
    _, x_gd = hybrid.solvers.gradient_descent(zeta=zeta, cov=covar, x_aoa=x_aoa, x_tdoa=x_tdoa,
                                              x_fdoa=x_fdoa, v_fdoa=v_fdoa, do_resample=False,
                                              tdoa_ref_idx=tdoa_ref_idx, fdoa_ref_idx=fdoa_ref_idx,
                                              **gd_ls_args)

    # LS Solution
    _, x_ls = hybrid.solvers.least_square(zeta=zeta, cov=covar, x_aoa=x_aoa, x_tdoa=x_tdoa,
                                          x_fdoa=x_fdoa, v_fdoa=v_fdoa, do_resample=False,
                                          tdoa_ref_idx=tdoa_ref_idx, fdoa_ref_idx=fdoa_ref_idx,
                                          **gd_ls_args)


    return {'ml': x_ml, 'ls': x_ls, 'gd': x_gd}
