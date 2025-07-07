import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
import utils
from utils import SearchSpace
from triang import DirectionFinder
from tdoa import TDOAPassiveSurveillanceSystem
from fdoa import FDOAPassiveSurveillanceSystem
from hybrid import HybridPassiveSurveillanceSystem
import time
from utils.covariance import CovarianceMatrix

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


def _plt_markers(pss, do_source=True, do_aoa=False, do_tdoa=False, do_fdoa=False, colors=('w', 'k', 'k', 'k')):
    """
    Add markers for Source, AOA, TDOA, and FDOA sensors to a plot; optionally specify colors

    Nicholas O'Donoughue
    16 January 2025

    :param colors: 4-element Tuple of colors to use for markers (source, aoa, tdoa, fdoa). Default=('w', 'k', 'k', 'k')
    """
    if do_source:
        plt.scatter(x_source[0], x_source[1], marker='x', color=colors[0], label='Target', clip_on=False)
    if do_aoa:
        pss.aoa.plot_sensors(marker='o', color=colors[1], label='AOA Sensors', clip_on=False)
    if do_tdoa:
        pss.tdoa.plot_sensors(marker='s', color=colors[2], label='TDOA Sensors', clip_on=False)
    if do_fdoa:
        pss.fdoa.plot_sensors(marker='^', color=colors[3], label='FDOA Sensors', clip_on=False)
        for this_x, this_v in zip(x_fdoa.T, v_fdoa.T):  # transpose so the loop steps over sensors, not dimensions
            plt.arrow(x=this_x[0], y=this_x[1],
                      dx=this_v[0], dy=this_v[1],
                      width=.01e3, head_width=.05e3,
                      color=colors[3])


def _make_pss_systems(err_aoa=None, err_time=None, err_freq=None, f0=1.0, tdoa_ref_idx=None, fdoa_ref_idx=None) \
        -> HybridPassiveSurveillanceSystem:
    """

    :param err_aoa: Angle-of-Arrival measurement error (in deg)
    :param err_time: Time-of-Arrival measurement error (in seconds)
    :param err_freq: Frequency measurement error (in Hz)
    :param f0: Operating frequency (in Hz)
    :param tdoa_ref_idx: Index of TDOA reference sensor
    :param fdoa_ref_idx: Index of FDOA reference sensor
    :return pss: Hybrid PSS system object
    """

    # Count the number of sensors in each type
    num_dim, num_aoa = utils.safe_2d_shape(x_aoa)
    _, num_tdoa = utils.safe_2d_shape(x_tdoa)
    _, num_fdoa = utils.safe_2d_shape(x_fdoa)

    # Define Error Covariance Matrix
    if err_aoa is not None:
        cov_psi = (err_aoa * _deg2rad) ** 2.0 * np.eye(num_aoa)  # rad ^ 2
        aoa_pss = DirectionFinder(x=x_aoa, cov=CovarianceMatrix(cov_psi), do_2d_aoa=num_dim>2)
    else:
        aoa_pss = None

    if err_time is not None:
        err_r = err_time * utils.constants.speed_of_light
        cov_r = (err_r ** 2.0) * np.eye(num_tdoa)  # m ^ 2
        tdoa_pss = TDOAPassiveSurveillanceSystem(x=x_tdoa, cov=CovarianceMatrix(cov_r), variance_is_toa=False,
                                                 ref_idx=tdoa_ref_idx)
    else:
        tdoa_pss = None

    if err_freq is not None:
        rr_err = err_freq * utils.constants.speed_of_light / f0  # (m/s)
        cov_rr = (rr_err ** 2) * np.eye(num_fdoa)  # (m/s)^2
        fdoa_pss = FDOAPassiveSurveillanceSystem(x=x_fdoa, vel=v_fdoa, cov=CovarianceMatrix(cov_rr),
                                                 ref_idx=fdoa_ref_idx)
    else:
        fdoa_pss = None

    pss = HybridPassiveSurveillanceSystem(aoa=aoa_pss, tdoa=tdoa_pss, fdoa=fdoa_pss)
    return pss


def _make_plot(pss: HybridPassiveSurveillanceSystem, x_ml, x_ls, x_gd, x_init, colors=None,
               crlb_ellipse=None, conf_interval=None):
    fig, _ = plt.subplots()
    _plt_markers(pss, do_aoa=True, do_tdoa=True, do_fdoa=True, colors=colors)

    # Plot Closed-Form Solution
    plt.scatter(x_ml[0], x_ml[1], marker='v', label='Maximum Likelihood')

    # Plot Iterative Solutions
    plt.scatter(x_init[0], x_init[1], marker='x', color='k', label='Initial Estimate')
    plt.plot(x_ls[0], x_ls[1], linestyle=':', marker='o', markevery=[-1], label='Least Squares')
    plt.plot(x_gd[0], x_gd[1], linestyle='--', marker='s', markevery=[-1], label='Grad Descent')

    # Overlay Error Ellipse
    if crlb_ellipse is not None:
        plt.plot(crlb_ellipse[0, :], crlb_ellipse[1, :], linestyle='--', color='k',
                 label='{:d}% Error Ellipse'.format(conf_interval))

    plt.legend(loc='best')

    return fig


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

    # Initialize Sensors
    err_aoa = 3  # deg
    err_time = 1e-7  # 100 ns timing error
    err_freq = 10  # Hz
    f0 = 1e9  # Hz
    pss = _make_pss_systems(err_aoa=err_aoa, err_time=err_time, err_freq=err_freq, f0=f0) # result stored in _pss

    # Draw Geometry
    fig1 = plt.figure()
    _plt_markers(pss, do_aoa=True, do_tdoa=True, do_fdoa=True, colors=colors)

    # True Measurements
    psi_act = pss.aoa.measurement(x_source=x_source)
    range_diff = pss.tdoa.measurement(x_source=x_source)
    velocity_diff = pss.fdoa.measurement(x_source=x_source, v_source=None)
    z = pss.measurement(x_source=x_source)
    # z_test = np.concatenate((psi_act, range_diff, velocity_diff), axis=0)
    # assert np.sum(np.fabs(z-z_test))<=1e-10, "Measurement mismatch; something went wrong."

    # Draw DF line
    xy_lobs = pss.aoa.draw_lobs(zeta=psi_act, x_source=x_source, scale=5)
    lob_label = 'AOA Solution'
    for xy_lob in xy_lobs:
        plt.plot(xy_lob[0], xy_lob[1], color=colors[1], linestyle='-', label=lob_label)
        lob_label=None

    # Draw isochrone
    # Transpose the x_tdoa array before indexing; so [0] and [1] refer to sensors, not dimensions
    xy_isochrones = pss.tdoa.draw_isochrones(range_diff=range_diff, num_pts=1000, max_ortho=5e3)
    for xy_isochrone in xy_isochrones:
        plt.plot(xy_isochrone[0], xy_isochrone[1], color=colors[2], linestyle=':', label='TDOA Solution')

    # Draw isodoppler line
    # Transpose the x_fdoa|v_fdoa arrays before indexing; so [0] and [1] refer to sensors, not dimensions
    xy_isodopplers = pss.fdoa.draw_isodoppler(vel_diff=velocity_diff, num_pts=1000, max_ortho=5e3)
    for xy_isodoppler in xy_isodopplers:
        plt.plot(xy_isodoppler[0], xy_isodoppler[1], color=colors[3], linestyle='-.', label='FDOA Solution')

    # Adjust Plot Display
    plt.ylim([-0.5e3, 4e3])
    plt.xlim([-.5e3, 5.5e3])
    plt.legend(loc='upper right')

    # Remove the axes for a clean image
    plt.axis('off')

    # --- Compute Variances and Print ---
    print('AOA Measurement: {:.2f} deg'.format(psi_act[0] * _rad2deg))
    print('AOA Covariance: {} rad^2'.format(pss.aoa.cov.cov))

    print('TDOA Measurement: {:.2f} m'.format(range_diff[0]))
    print('TDOA Covariance:')
    print('{} m^2'.format(np.matrix(pss.tdoa.cov_raw.cov)))

    print('FDOA Measurement: {:.2f} m/s'.format(velocity_diff[0]))
    print('FDOA Covariance:')
    print('{} m^2/s^2'.format(np.matrix(pss.fdoa.cov_raw.cov)))

    # Set Up Search Grid
    x_grid = np.arange(-0.5, 5.5, 0.02) * 1e3
    y_grid = np.arange(0, 4, 0.02) * 1e3
    xx, yy = np.meshgrid(x_grid, y_grid)
    x_test_pos = np.vstack((xx.ravel(), yy.ravel()))
    grid_extent = (float(x_grid[0]), float(x_grid[-1]), float(y_grid[0]), float(y_grid[-1]))

    # Log-Likelihood Figure Generator
    def _make_subfigure(ell, do_aoa=False, do_tdoa=False, do_fdoa=False):
        _fig, _ax = plt.subplots()
        plt.imshow(ell, extent=grid_extent, origin='lower', cmap=cm['viridis_r'], vmin=-20, vmax=0)
        plt.colorbar()
        _plt_markers(pss, do_aoa=do_aoa, do_tdoa=do_tdoa, do_fdoa=do_fdoa)

        plt.ylim([0, 4e3])
        plt.xlim([-.5e3, 5.5e3])
        plt.clim(-20, 0)
        plt.legend(loc='upper right')

        return _fig

    # Plot AOA Likelihood
    ell_aoa = pss.aoa.log_likelihood(zeta=psi_act, x_source=x_test_pos).reshape(xx.shape)
    fig2 = _make_subfigure(ell_aoa, do_aoa=True)

    # TDOA
    ell_tdoa = pss.tdoa.log_likelihood(zeta=range_diff, x_source=x_test_pos).reshape(xx.shape)
    fig3 = _make_subfigure(ell_tdoa, do_tdoa=True)

    # FDOA
    ell_fdoa = pss.fdoa.log_likelihood(zeta=velocity_diff, x_source=x_test_pos).reshape(xx.shape)
    fig4 = _make_subfigure(ell_fdoa, do_fdoa=True)

    # Hybrid
    ell_hybrid = pss.log_likelihood(zeta=z, x_source=x_test_pos, v_source=None).reshape(xx.shape)
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

    # Generate Error Covariance
    err_aoa = 3  # deg
    err_time = 1e-7  # 100 ns timing error
    err_freq = 10  # Hz
    f0 = 1e9  # Hz
    tdoa_ref_idx = 1
    fdoa_ref_idx = 1
    pss = _make_pss_systems(err_aoa=err_aoa, err_time=err_time, err_freq=err_freq, f0=f0,
                            tdoa_ref_idx=tdoa_ref_idx, fdoa_ref_idx=fdoa_ref_idx)

    # Take Hybrid measurement and Define combined covariance matrix
    z = pss.measurement(x_source=x_source)

    # Generate Random Noise
    cov_lower = pss.cov.lower
    num_measurements = pss.num_measurements
    noise = cov_lower @ np.random.randn(num_measurements, )

    # Combine Noise with Perfect Measurement
    zeta = z + noise

    # ---- Set Up Solution Parameters ----
    # ML Search Parameters
    search_space = SearchSpace(x_ctr=np.array([2.5, 2.5]) * 1e3,
                               epsilon=25, max_offset=np.array([5, 5])*1e3)

    # GD and LS Search Parameters
    x_init = np.array([1, 1]) * 1e3
    gd_ls_args = {
        'x_init': x_init,
        'epsilon': search_space.epsilon,  # desired convergence step size, same as grid resolution
        'max_num_iterations': 100,
        'force_full_calc': True,
        'plot_progress': False
    }
    # ---- Apply Various Solvers ----
    # ML Solution
    x_ml, _, _ = pss.max_likelihood(zeta=zeta, search_space=search_space)

    # GD Solution
    x_gd, x_gd_full = pss.gradient_descent(zeta=zeta, **gd_ls_args)

    # LS Solution
    x_ls, x_ls_full = pss.least_square(zeta=zeta, **gd_ls_args)

    # ---- Plot Results ----
    fig_full = _make_plot(pss, x_ml=x_ml, x_ls=x_ls_full, x_gd=x_gd_full, x_init=x_init, colors=colors,
                          crlb_ellipse=None)
    plt.xlim([-.5e3, 5.5e3])
    plt.ylim([-.5e3, 4e3])

    fig_zoom = _make_plot(pss, x_ml=x_ml, x_ls=x_ls_full, x_gd=x_gd_full, x_init=x_init, colors=colors,
                          crlb_ellipse=None)
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
    pss = _make_pss_systems(err_aoa=err_aoa, err_time=err_time, err_freq=err_freq, f0=f0,
                            tdoa_ref_idx=tdoa_ref_idx, fdoa_ref_idx=fdoa_ref_idx)

    # Take Hybrid measurement
    z = pss.measurement(x_source=x_source)

    # ---- Set Up Solution Parameters ----
    # ML Search Parameters
    ml_args = {'x_ctr': np.array([2.5, 2.5]) * 1e3,
               'search_size': np.array([5, 5]) * 1e3,
               'epsilon': 20}  # meters, grid resolution

    # GD and LS Search Parameters
    gd_ls_args = {'x_init': np.array([1, 1]) * 1e3,
                  'epsilon': ml_args['epsilon'],  # desired convergence step size, same as grid resolution
                  'max_num_iterations': 50,
                  'force_full_calc': True,
                  'plot_progress': False}

    # Perform a single Monte-Carlo iteration (executing all solvers)
    res = _mc_iteration(z, pss, rng, ml_args, gd_ls_args)

    # ---- Estimate Error Bounds ----
    # CRLB
    crlb = pss.compute_crlb(x_source=x_source)
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

    fig_full = _make_plot(pss, x_ml=x_ml, x_ls=x_ls, x_gd=x_gd, x_init=x_init, colors=colors,
                          crlb_ellipse=crlb_ellipse, conf_interval=conf_interval)
    plt.xlim([-.5e3, 5.5e3])
    plt.ylim([0, 4e3])

    fig_zoom = _make_plot(pss, x_ml=x_ml, x_ls=x_ls, x_gd=x_gd, x_init=x_init, colors=colors,
                          crlb_ellipse=crlb_ellipse, conf_interval=conf_interval)
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
    pss = _make_pss_systems(err_aoa=err_aoa, err_time=err_time, err_freq=err_freq, f0=f0,
                            tdoa_ref_idx=tdoa_ref_idx, fdoa_ref_idx=fdoa_ref_idx)

    # Take Hybrid measurement
    z = pss.measurement(x_source=x_source)

    # ---- Set Up Solution Parameters ----
    # ML Search Parameters
    ml_args = {'x_ctr': np.array([2.5, 2.5]) * 1e3,
               'search_size': np.array([5, 5]) * 1e3,
               'epsilon': 25}

    # GD and LS Search Parameters
    gd_ls_args = {'x_init': np.array([1, 1]) * 1e3,
                  'epsilon': ml_args['epsilon'],  # desired convergence step size, same as grid resolution
                  'max_num_iterations': 50,
                  'force_full_calc': True,
                  'plot_progress': False}

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
    res = {}
    for idx in np.arange(num_mc_trials):
        utils.print_progress(num_mc_trials, idx, iterations_per_marker, iterations_per_row, t_start)

        res = _mc_iteration(z, pss, rng, ml_args, gd_ls_args)

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
    crlb = pss.compute_crlb(x_source=x_source)
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

    fig_full = _make_plot(pss, x_ml=x_ml, x_ls=x_ls, x_gd=x_gd, x_init=x_init, colors=colors,
                          crlb_ellipse=crlb_ellipse, conf_interval=conf_interval)
    plt.xlim([-.5e3, 5.5e3])
    plt.ylim([0, 4e3])

    return fig_err, fig_full


def _mc_iteration(z, pss: HybridPassiveSurveillanceSystem, rng, ml_search:SearchSpace, gd_ls_args):
    """
    Executes a single iteration of the Monte Carlo simulation in Example 2.3.

    :return estimates: Dictionary with estimated target position using several algorithms.  Fields are:
                ml: Maximum Likelihood solution
                gd: Gradient Descent solution
                ls: Least Squares solution

    Nicholas O'Donoughue
    16 January 2025
    """

    # Generate a random measurement
    zeta = z + pss.cov.lower @ rng.standard_normal(size=(pss.num_measurements, ))

    # ---- Apply Various Solvers ----
    # ML Solution
    x_ml, _, _ = pss.max_likelihood(zeta=zeta, search_space=ml_search)

    # GD Solution
    _, x_gd = pss.gradient_descent(zeta=zeta, **gd_ls_args)

    # LS Solution
    _, x_ls = pss.least_square(zeta=zeta, **gd_ls_args)

    return {'ml': x_ml, 'ls': x_ls, 'gd': x_gd}
