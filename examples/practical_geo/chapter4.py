import matplotlib.pyplot as plt
import numpy as np
import time

from ewgeo.fdoa import FDOAPassiveSurveillanceSystem
from ewgeo.hybrid import HybridPassiveSurveillanceSystem
from ewgeo.tdoa import TDOAPassiveSurveillanceSystem
from ewgeo.triang import DirectionFinder
from ewgeo.utils import print_elapsed, print_progress, SearchSpace, remove_outliers
from ewgeo.utils.constants import speed_of_light
from ewgeo.utils.coordinates import lla_to_enu
from ewgeo.utils.covariance import CovarianceMatrix
from ewgeo.utils.errors import compute_cep50, draw_error_ellipse, compute_rmse
from ewgeo.utils.unit_conversions import convert, kph_to_mps

_rad2deg = convert(1, "rad", "deg")
_deg2rad = convert(1, "deg", "rad")
_kph2mps = kph_to_mps(1)


def run_all_examples():
    """
    Run all chapter 4 examples and return a list of figure handles

    :return figs: list of figure handles
    """

    return list(example1()) + list(example2())


def example1(mc_params=None):
    """
    Executes Example 4.1.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    7 February 2025

    :return: figure handle to generated graphic
    """

    # Set up sensors
    x_sensor_enu = np.array([[-5, 0, 5],
                             [0, 0, 0],
                             [0, 0, 0]])*1e3
    v_sensor_enu = np.array([[80, 80, 80],
                            [0, 0, 0],
                            [0, 0, 0]])*_kph2mps  # Convert to SI units (m/s)

    # Reference coordinate for ENU
    ref_lat = 5  # deg Lat (N)
    ref_lon = -15  # deg Lon (W)
    ref_alt = 10e3  # alt (m)

    # Measurement Errors
    do_2d_aoa = True
    err_aoa_deg = 2
    err_tdoa_s = 1e-6
    err_fdoa_hz = 10

    # Source Position and Velocity
    x_source_lla = np.array([4.5, -(14 + 40/60), 0])  # deg Lat (N), deg Lon (E), alt (m)
    v_source_enu = np.array([0, 0, 0])
    f_source_hz = 1e9

    # Convert positions and velocity to ENU
    e_source, n_source, u_source = lla_to_enu(x_source_lla[0], x_source_lla[1], x_source_lla[2],
                                              ref_lat, ref_lon, ref_alt,
                                              angle_units='deg', dist_units='m')
    x_source_enu = np.array([e_source, n_source, u_source])

    # Sensor Selection
    num_dims, num_tdoa = np.shape(x_sensor_enu)
    num_aoa = num_tdoa
    num_fdoa = num_aoa
    ref_tdoa = num_tdoa - 1
    ref_fdoa = num_fdoa - 1

    # Error Covariance Matrix
    if do_2d_aoa:
        num_aoa = 2 * num_aoa  # double the number of aoa "sensors" to account for the second measurement from each

    cov_psi = (err_aoa_deg*_deg2rad)**2 * np.eye(num_aoa)  # rad^2
    cov_r = (err_tdoa_s*speed_of_light)**2 * np.eye(num_tdoa)  # m^2
    cov_rr = (err_fdoa_hz*speed_of_light/f_source_hz)**2 * np.eye(num_fdoa)  # m^2/s^2

    # Make the PSS Objects
    aoa = DirectionFinder(x=x_sensor_enu, do_2d_aoa=do_2d_aoa, cov=CovarianceMatrix(cov_psi))
    tdoa = TDOAPassiveSurveillanceSystem(x=x_sensor_enu, cov=CovarianceMatrix(cov_r), variance_is_toa=False, ref_idx=ref_tdoa)
    fdoa = FDOAPassiveSurveillanceSystem(x=x_sensor_enu, vel=v_sensor_enu, cov=CovarianceMatrix(cov_rr), ref_idx=ref_fdoa)
    hybrid = HybridPassiveSurveillanceSystem(aoa=aoa, tdoa=tdoa, fdoa=fdoa)

    # Monte Carlo parameters
    num_monte_carlo = 1000
    if mc_params is not None:
        num_monte_carlo = max(int(num_monte_carlo/mc_params['monte_carlo_decimation']),mc_params['min_num_monte_carlo'])

    # Generate Noisy data
    zeta = hybrid.noisy_measurement(x_source=x_source_enu, v_source=v_source_enu, num_samples=num_monte_carlo)

    # GD and LS Search Parameters
    x_init_enu = np.array([1, 1, 0]) * 1e3
    # [xx, yy, zz] = enu2ecef(x_init_enu(1), x_init_enu(2), x_init_enu(3), ref_lat, ref_lon, ref_alt)
    # x_init_ecef = [xx,yy,zz] # reassemble ecef coordinates into a single variable

    # LS Search Parameters
    max_num_iterations = 100
    ls_args = {'x_init': x_init_enu,
               'epsilon': 100,  # desired convergence step size, same as grid resolution
               'max_num_iterations': max_num_iterations,
               'force_full_calc': True,
               'plot_progress': False}

    error = np.zeros((num_monte_carlo, ls_args['max_num_iterations']))
    x_ls_iters = np.zeros((num_dims, max_num_iterations))  # pre-initialize iterative LS solution
    print('Performing Monte Carlo simulation...')
    t_start = time.perf_counter()

    iterations_per_marker = 10
    markers_per_row = 40
    iterations_per_row = markers_per_row * iterations_per_marker

    for idx in range(num_monte_carlo):
        print_progress(num_total=num_monte_carlo, curr_idx=idx, iterations_per_marker=iterations_per_marker,
                             iterations_per_row=iterations_per_row, t_start=t_start)

        # LS Solution
        _, x_ls_iters = hybrid.least_square(zeta=zeta[:, idx], **ls_args)

        error[idx] = np.sqrt(np.sum(np.abs(x_ls_iters-x_source_enu[:, np.newaxis])**2, 0))

    print('done')
    t_elapsed = time.perf_counter() - t_start
    print_elapsed(t_elapsed)

    # Remove outliers
    error = remove_outliers(error)
    num_monte_carlo_actual, _ = np.shape(error)

    # Plot RMSE and CRLB
    rmse_ls = np.sqrt(np.sum(error**2, 0)/num_monte_carlo_actual)

    fig2 = plt.figure()
    plt.plot(np.arange(max_num_iterations), rmse_ls, label='Least Squares')
    plt.yscale('log')

    # Compute the CRLB
    crlb = hybrid.compute_crlb(x_source=x_source_enu)

    # Compute and display the RMSE
    rmse_crlb = compute_rmse(crlb)

    plt.plot([0, max_num_iterations-1], [rmse_crlb, rmse_crlb], 'k', label='CRLB')
    plt.xlabel('Iteration Number')
    plt.ylabel('RMSE [m]')
    plt.title('Monte Carlo Geolocation Results')
    plt.ylim([1e4, 1e5])
    plt.legend(loc='upper right')

    # Compute and display the CEP50
    cep50 = compute_cep50(crlb)
    print('CEP50: {:.2f} km'.format(cep50/1e3))

    # Generate the 90% error ellipse from the CRLB
    crlb_ellipse = draw_error_ellipse(x_source_enu[0:2], crlb, 101, 50)

    # Plot Result
    fig1 = plt.figure()
    plt.scatter(x_source_enu[0]/1e3, x_source_enu[1]/1e3, marker='x', label='Target', zorder=3)
    plt.scatter(x_sensor_enu[0]/1e3, x_sensor_enu[1]/1e3, marker='s', label='Sensor', zorder=3)

    plt.plot(x_ls_iters[0]/1e3, x_ls_iters[1]/1e3, '-', markevery=[-1], marker='*', label='LS Solution')
    plt.plot(crlb_ellipse[0]/1e3, crlb_ellipse[1]/1e3, '--', label='CRLB')

    plt.grid(True)
    plt.legend(loc='upper right')
    plt.xlabel('East [km]')
    plt.ylabel('North [km]')

    # Package figure handles
    return fig1, fig2


def example2(colors=None):
    """
    Executes Example 4.2.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    7 February 2025

    :return: figure handle to generated graphic
    """

    if colors is None:
        colors = plt.get_cmap("viridis_r")

    # Set up sensors
    num_sensors=4
    sensor_lat = np.array([26 + 52/60, 27 + 52/60, 23 + 19/60, 24 + 19/60])
    sensor_lon = np.array([-(68 + 47/60), -(72 + 36/60), -(69 + 47/60), -(74 + 36/60)])
    sensor_alt = 500e3 * np.ones((num_sensors,))

    # Center of observation area
    ref_lat = 26  # deg Lat (N)
    ref_lon = -71  # deg Lon (E)
    ref_alt = 0  # alt (m)

    e_sensor, n_sensor, u_sensor = lla_to_enu(sensor_lat, sensor_lon, sensor_alt,
                                              ref_lat, ref_lon, ref_alt,
                                              angle_units="deg", dist_units="m")
    x_sensor_enu = np.array([e_sensor, n_sensor, u_sensor]) # shape: (3, n_sensor)

    # Define sensor velocity in ENU
    v_abs = 7.61*1e3  # m/s, based on 500 km orbital height above Earth
    heading_deg = 60  # deg N from E
    heading_rad = heading_deg * _deg2rad
    v0 = v_abs * np.array([np.sin(heading_rad), np.cos(heading_rad), 0]) # shape: (3, )
    v_sensor_enu = np.repeat(v0[:, np.newaxis], num_sensors, axis=1)

    # Build grid of positions within 500 km of source position (ENU origin)
    v_source_enu = np.zeros(shape=(3, ))  # source is stationary
    search_space = SearchSpace(x_ctr=np.zeros((3, )),
                               max_offset=500e3 * np.array([1., 1., 0.]),
                               points_per_dim=201)
    x_source_enu, x_grid = search_space.x_set, search_space.x_grid
    extent = search_space.get_extent(axes=(0, 1), multiplier=1/1e3)

    # Use a squeeze operation to ensure that the individual dimension
    # indices in x_grid are 2D
    x_grid = [np.squeeze(this_dim) for this_dim in x_grid]

    # Measurement Error
    err_aoa_rad = 3 * _deg2rad           # rad
    err_r = 1e-5 * speed_of_light        # meters
    err_rr = 100 * speed_of_light / 1e9  # meters/second

    # Error Covariance Matrix
    _eye = np.eye(num_sensors)
    _eye2 = np.eye(2*num_sensors)
    cov_psi = CovarianceMatrix(err_aoa_rad**2 * _eye2)  # rad^2
    cov_r = CovarianceMatrix(err_r**2 * _eye)           # m^2
    cov_rr = CovarianceMatrix(err_rr**2 * _eye)         # m^2/s^2

    # Make the PSS objects
    ref_fdoa = ref_tdoa = num_sensors - 1
    aoa = DirectionFinder(x=x_sensor_enu, do_2d_aoa=True, cov=cov_psi)
    tdoa = TDOAPassiveSurveillanceSystem(x=x_sensor_enu, cov=cov_r, ref_idx=ref_tdoa, variance_is_toa=False)
    fdoa = FDOAPassiveSurveillanceSystem(x=x_sensor_enu, vel=v_sensor_enu, cov=cov_rr, ref_idx=ref_fdoa)
    hybrid = HybridPassiveSurveillanceSystem(aoa=aoa, tdoa=tdoa, fdoa=fdoa)

    # Compute the CRLB
    crlb = hybrid.compute_crlb(v_source= v_source_enu, x_source=x_source_enu, print_progress=True)

    # Compute and display the RMSE
    rmse_crlb = compute_rmse(crlb)
    levels = np.arange(15)
    color_lims = [5, 10]

    # Plotting
    fig = plt.figure()
    plt.imshow(np.reshape(rmse_crlb/1e3, search_space.grid_shape), origin='lower', cmap=colors, extent=extent,
               vmin=color_lims[0], vmax=color_lims[-1])
    plt.colorbar(format=lambda x, _: f"{x:.0f} km")

    # Add contours
    hdl2 = plt.contour(x_grid[0]/1e3, x_grid[1]/1e3, np.reshape(rmse_crlb/1e3, search_space.grid_shape),
                       origin='lower', colors='k', levels=levels)
    plt.clabel(hdl2, fontsize=10, colors='k', fmt='%.0f km')

    hdl3 = plt.scatter(x_sensor_enu[0]/1e3, x_sensor_enu[1]/1e3, color='w', facecolors='none', marker='o',
                       label='Sensors', zorder=3)
    for this_x, this_v in zip(x_sensor_enu.T,
                              v_sensor_enu.T):  # transpose so the loop steps over sensors, not dimensions
        plt.arrow(x=this_x[0]/1e3, y=this_x[1]/1e3,
                  dx=this_v[0]/100, dy=this_v[1]/100,
                  width=5, head_width=10,
                  color=hdl3.get_edgecolor(), label=None, zorder=3)
    plt.grid(True)

    plt.xlabel('East [km]')
    plt.ylabel('North [km]')

    return fig,

# TODO: Make a version of Example 4.2 that plots on a globe. Use basemap?
# https://jakevdp.github.io/PythonDataScienceHandbook/04.13-geographic-data-with-basemap.html

# TODO: Implement the code from book2_vid4_1.m


if __name__ == '__main__':
    run_all_examples()
    plt.show()
