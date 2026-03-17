import matplotlib.pyplot as plt
import numpy as np
import time

from ewgeo.fdoa import FDOAPassiveSurveillanceSystem
from ewgeo.hybrid import HybridPassiveSurveillanceSystem
from ewgeo.tdoa import TDOAPassiveSurveillanceSystem
from ewgeo.triang import DirectionFinder
from ewgeo.utils import print_elapsed, print_progress, SearchSpace, remove_outliers
from ewgeo.utils.constants import speed_of_light
from ewgeo.utils.coordinates import lla_to_enu, enu_to_lla, lla_to_ecef, enu_to_aer
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

    return list(example1()) + list(example2()) + list(example2_globe()) + list(example2_globe_3d()) + list(vid4_1())


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
                               points_per_dim=np.array([201, 201, 1]))
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

def example2_globe(colors=None):
    """
    Executes Example 4.2, displaying the CRLB heatmap overlaid on a geographic map
    using Cartopy with a PlateCarree projection and Lat/Lon axes.

    Requires cartopy: pip install cartopy

    Nicholas O'Donoughue
    16 March 2026

    :return: figure handle to generated graphic
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    if colors is None:
        colors = plt.get_cmap("viridis_r")

    # Set up sensors (identical to example2)
    num_sensors = 4
    sensor_lat = np.array([26 + 52/60, 27 + 52/60, 23 + 19/60, 24 + 19/60])
    sensor_lon = np.array([-(68 + 47/60), -(72 + 36/60), -(69 + 47/60), -(74 + 36/60)])
    sensor_alt = 500e3 * np.ones((num_sensors,))

    ref_lat = 26
    ref_lon = -71
    ref_alt = 0

    e_sensor, n_sensor, u_sensor = lla_to_enu(sensor_lat, sensor_lon, sensor_alt,
                                              ref_lat, ref_lon, ref_alt,
                                              angle_units="deg", dist_units="m")
    x_sensor_enu = np.array([e_sensor, n_sensor, u_sensor])

    v_abs = 7.61e3
    heading_rad = 60 * _deg2rad
    v0 = v_abs * np.array([np.sin(heading_rad), np.cos(heading_rad), 0])
    v_sensor_enu = np.repeat(v0[:, np.newaxis], num_sensors, axis=1)

    v_source_enu = np.zeros(shape=(3,))
    search_space = SearchSpace(x_ctr=np.zeros((3,)),
                               max_offset=500e3 * np.array([1., 1., 0.]),
                               points_per_dim=np.array([201, 201, 1]))
    x_source_enu, x_grid = search_space.x_set, search_space.x_grid
    x_grid = [np.squeeze(this_dim) for this_dim in x_grid]

    err_aoa_rad = 3 * _deg2rad
    err_r = 1e-5 * speed_of_light
    err_rr = 100 * speed_of_light / 1e9

    _eye = np.eye(num_sensors)
    _eye2 = np.eye(2 * num_sensors)
    cov_psi = CovarianceMatrix(err_aoa_rad**2 * _eye2)
    cov_r = CovarianceMatrix(err_r**2 * _eye)
    cov_rr = CovarianceMatrix(err_rr**2 * _eye)

    ref_fdoa = ref_tdoa = num_sensors - 1
    aoa = DirectionFinder(x=x_sensor_enu, do_2d_aoa=True, cov=cov_psi)
    tdoa = TDOAPassiveSurveillanceSystem(x=x_sensor_enu, cov=cov_r, ref_idx=ref_tdoa, variance_is_toa=False)
    fdoa = FDOAPassiveSurveillanceSystem(x=x_sensor_enu, vel=v_sensor_enu, cov=cov_rr, ref_idx=ref_fdoa)
    hybrid = HybridPassiveSurveillanceSystem(aoa=aoa, tdoa=tdoa, fdoa=fdoa)

    crlb = hybrid.compute_crlb(v_source=v_source_enu, x_source=x_source_enu, print_progress=True)
    rmse_crlb = compute_rmse(crlb)
    levels = np.arange(15)
    color_lims = [5, 10]

    # Convert ENU grid to Lat/Lon for geographic axes
    lat_grid, lon_grid, _ = enu_to_lla(x_grid[0], x_grid[1], np.zeros_like(x_grid[0]),
                                        ref_lat, ref_lon, ref_alt,
                                        angle_units='deg', dist_units='m')
    rmse_grid = np.reshape(rmse_crlb / 1e3, search_space.grid_shape)

    # Build map
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN, zorder=0)
    ax.add_feature(cfeature.LAND, zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=1)
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    ax.set_extent([lon_grid.min()-7, lon_grid.max()+7, lat_grid.min()-7, lat_grid.max()+7],
                  crs=ccrs.PlateCarree())

    # CRLB heatmap
    mesh = ax.pcolormesh(lon_grid, lat_grid, rmse_grid,
                         cmap=colors, vmin=color_lims[0], vmax=color_lims[-1],
                         transform=ccrs.PlateCarree(), zorder=2)
    fig.colorbar(mesh, ax=ax, format=lambda x, _: f"{x:.0f} km")

    # Contours
    contour_set = ax.contour(lon_grid, lat_grid, rmse_grid,
                             levels=levels, colors='k',
                             transform=ccrs.PlateCarree(), zorder=3)
    ax.clabel(contour_set, fontsize=10, colors='k', fmt='%.0f km')

    # Sensor markers
    hdl3 = ax.scatter(sensor_lon, sensor_lat, color='w', facecolors='none', marker='o',
                      label='Sensors', transform=ccrs.PlateCarree(), zorder=5)

    # Velocity arrows: project ENU arrow tip to Lat/Lon, then quiver
    tip_e = x_sensor_enu[0] + v_sensor_enu[0] / 100
    tip_n = x_sensor_enu[1] + v_sensor_enu[1] / 100
    tip_lat, tip_lon, _ = enu_to_lla(tip_e, tip_n, np.zeros(num_sensors),
                                      ref_lat, ref_lon, ref_alt,
                                      angle_units='deg', dist_units='m')
    ax.quiver(sensor_lon, sensor_lat,
              tip_lon - sensor_lon, tip_lat - sensor_lat,
              color=hdl3.get_edgecolor()[0],
              transform=ccrs.PlateCarree(), zorder=5,
              angles='xy', scale_units='xy', scale=1)

    ax.set_title('Hybrid CRLB RMSE [km]')
    ax.legend(loc='upper right')

    return fig,


def example2_globe_3d(colors=None):
    """
    Executes Example 4.2 as a 3D scene: the hybrid CRLB heatmap is drawn as a coloured
    surface at ground level, Natural Earth coastlines are overlaid at z=0, and the four
    satellites are rendered as stem plots rising from their sub-satellite footprints to
    their true orbital altitude (500 km).

    Requires cartopy: pip install cartopy

    Nicholas O'Donoughue
    16 March 2026

    :return: figure handle to generated graphic
    """
    import cartopy.feature as cfeature
    import matplotlib.colors as mcolors
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers the '3d' projection

    if colors is None:
        colors = plt.get_cmap("viridis_r")

    # Set up sensors (identical to example2)
    num_sensors = 4
    sensor_lat = np.array([26 + 52/60, 27 + 52/60, 23 + 19/60, 24 + 19/60])
    sensor_lon = np.array([-(68 + 47/60), -(72 + 36/60), -(69 + 47/60), -(74 + 36/60)])
    sensor_alt = 500e3 * np.ones((num_sensors,))

    ref_lat = 26
    ref_lon = -71
    ref_alt = 0

    e_sensor, n_sensor, u_sensor = lla_to_enu(sensor_lat, sensor_lon, sensor_alt,
                                              ref_lat, ref_lon, ref_alt,
                                              angle_units="deg", dist_units="m")
    x_sensor_enu = np.array([e_sensor, n_sensor, u_sensor])

    v_abs = 7.61e3
    heading_rad = 60 * _deg2rad
    v0 = v_abs * np.array([np.sin(heading_rad), np.cos(heading_rad), 0])
    v_sensor_enu = np.repeat(v0[:, np.newaxis], num_sensors, axis=1)

    v_source_enu = np.zeros(shape=(3,))
    search_space = SearchSpace(x_ctr=np.zeros((3,)),
                               max_offset=500e3 * np.array([1., 1., 0.]),
                               points_per_dim=np.array([201, 201, 1]))
    x_source_enu, x_grid = search_space.x_set, search_space.x_grid
    x_grid = [np.squeeze(this_dim) for this_dim in x_grid]

    err_aoa_rad = 3 * _deg2rad
    err_r = 1e-5 * speed_of_light
    err_rr = 100 * speed_of_light / 1e9

    _eye = np.eye(num_sensors)
    _eye2 = np.eye(2 * num_sensors)
    cov_psi = CovarianceMatrix(err_aoa_rad**2 * _eye2)
    cov_r = CovarianceMatrix(err_r**2 * _eye)
    cov_rr = CovarianceMatrix(err_rr**2 * _eye)

    ref_fdoa = ref_tdoa = num_sensors - 1
    aoa = DirectionFinder(x=x_sensor_enu, do_2d_aoa=True, cov=cov_psi)
    tdoa = TDOAPassiveSurveillanceSystem(x=x_sensor_enu, cov=cov_r, ref_idx=ref_tdoa, variance_is_toa=False)
    fdoa = FDOAPassiveSurveillanceSystem(x=x_sensor_enu, vel=v_sensor_enu, cov=cov_rr, ref_idx=ref_fdoa)
    hybrid = HybridPassiveSurveillanceSystem(aoa=aoa, tdoa=tdoa, fdoa=fdoa)

    crlb = hybrid.compute_crlb(v_source=v_source_enu, x_source=x_source_enu, print_progress=True)
    rmse_crlb = compute_rmse(crlb)
    levels = np.arange(15)
    color_lims = [5, 10]

    # All spatial axes in km
    e_grid_km = x_grid[0] / 1e3
    n_grid_km = x_grid[1] / 1e3
    rmse_grid = np.reshape(rmse_crlb / 1e3, search_space.grid_shape)
    e_sens_km = x_sensor_enu[0] / 1e3
    n_sens_km = x_sensor_enu[1] / 1e3
    alt_km = sensor_alt[0] / 1e3  # 500 km; all sensors share the same orbital altitude

    # Build figure
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    # CRLB surface at z=0
    # Downsample by 4 for rendering performance (contour lines supply the fine detail).
    # Face colours are the average of the four surrounding grid-point values.
    s = 4
    eg = e_grid_km[::s, ::s]
    ng = n_grid_km[::s, ::s]
    rg = rmse_grid[::s, ::s]
    norm = mcolors.Normalize(vmin=color_lims[0], vmax=color_lims[1])
    fc = colors(norm(0.25 * (rg[:-1, :-1] + rg[1:, :-1] + rg[:-1, 1:] + rg[1:, 1:])))
    ax.plot_surface(eg, ng, np.zeros_like(eg),
                    facecolors=fc, shade=False, alpha=0.9, zorder=1)

    sm = plt.cm.ScalarMappable(cmap=colors, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, format=lambda x, _: f"{x:.0f} km", shrink=0.5, pad=0.1)

    # CRLB contour lines floating 0.5 km above the surface for visibility
    ax.contour(e_grid_km, n_grid_km, rmse_grid,
               levels=levels, colors='k', linewidths=0.8,
               zdir='z', offset=0.5, zorder=3)

    # Coastlines: convert Natural Earth lon/lat to ENU (km) and draw at z=0.
    # Only segments with at least one point within 600 km of the origin are drawn.
    coast = cfeature.NaturalEarthFeature('physical', 'coastline', '110m')
    for geom in coast.geometries():
        for line in (geom.geoms if hasattr(geom, 'geoms') else [geom]):
            coords = np.array(line.coords)
            lons_c, lats_c = coords[:, 0], coords[:, 1]
            e_c, n_c, _ = lla_to_enu(lats_c, lons_c, np.zeros_like(lats_c),
                                      ref_lat, ref_lon, ref_alt,
                                      angle_units='deg', dist_units='m')
            e_c_km, n_c_km = e_c / 1e3, n_c / 1e3
            if np.any((np.abs(e_c_km) < 600) & (np.abs(n_c_km) < 600)):
                ax.plot(e_c_km, n_c_km, np.zeros_like(e_c_km),
                        color='k', linewidth=0.5, alpha=0.7, zorder=2)

    # Sensor stems: vertical stalk from sub-satellite footprint to orbital altitude,
    # with a '+' marker on the ground and an 'o' marker at the satellite.
    for e, n in zip(e_sens_km, n_sens_km):
        ax.plot([e, e], [n, n], [0, alt_km],
                color='k', linewidth=0.8, zorder=5)
        ax.scatter([e], [n], [0], color='k', marker='+', s=30, linewidths=0.8, zorder=5)
    ax.scatter(e_sens_km, n_sens_km, np.full(num_sensors, alt_km),
               color='k', marker='o', s=40, zorder=5, label='Sensors')

    # Velocity arrows at orbital altitude
    tip_e_km = (x_sensor_enu[0] + v_sensor_enu[0] / 100) / 1e3
    tip_n_km = (x_sensor_enu[1] + v_sensor_enu[1] / 100) / 1e3
    ax.quiver(e_sens_km, n_sens_km, np.full(num_sensors, alt_km),
              tip_e_km - e_sens_km, tip_n_km - n_sens_km, np.zeros(num_sensors),
              color='w', arrow_length_ratio=0.3, zorder=5)

    ax.set_xlim([-500, 500])
    ax.set_ylim([-500, 500])
    ax.set_zlim([0, 600])
    ax.set_xlabel('East [km]')
    ax.set_ylabel('North [km]')
    ax.set_zlabel('Altitude [km]')
    ax.set_title('Hybrid CRLB RMSE [km]')
    ax.legend(loc='upper right')

    return fig,


def vid4_1():
    """
    Demonstrates coordinate transformations between LLA, ECEF, ENU, and AER
    using the Lincoln Memorial and US Capitol Building as example points.

    Distances computed three ways (ECEF, ENU, AER) are printed and cross-checked.
    Bearing and elevation from the Lincoln Memorial to the Capitol are also reported.

    A terrain cross-section is then sampled from the NASADEM GeoTIFF
    (nasadem_washingtondc.tif) using rasterio and scipy, and two LOS profile
    figures are generated: one for a ground-level observer and one for an
    observer 10 ft (3.048 m) above ground.

    Ported from book2_vid4_1.m

    Nicholas O'Donoughue
    17 March 2026

    :return: (fig1, fig2) — LOS profile figures for ground-level and 10 ft observer
    """
    import rasterio
    from scipy.ndimage import map_coordinates
    from pathlib import Path

    # Coordinates for the Lincoln Memorial (used as ENU reference origin)
    ref_lat, ref_lon, ref_alt = 38.88939, -77.05006, 0.0

    ref_x, ref_y, ref_z = lla_to_ecef(ref_lat, ref_lon, ref_alt, angle_units='deg', dist_units='m')
    ref_ecef = np.array([ref_x, ref_y, ref_z])

    # Coordinates for the Capitol Building
    test_lat, test_lon, test_alt = 38.89017, -77.00909, 0.0

    test_x, test_y, test_z = lla_to_ecef(test_lat, test_lon, test_alt, angle_units='deg', dist_units='m')
    test_ecef = np.array([test_x, test_y, test_z])

    # Compute distance via ECEF
    dist_km = np.linalg.norm(test_ecef - ref_ecef) / 1e3
    print(f'Distance from Lincoln Memorial to Capitol Building is {dist_km:.2f} km.')

    # Compare to Google Maps approximation (2.3 mi)
    meters_per_mile = 1609.344
    dist_gmap_km = 2.3 * meters_per_mile / 1e3
    print(f'Distance estimated via Google Maps is {dist_gmap_km:.2f} km.')

    # Compute distance via ENU
    e, n, u = lla_to_enu(test_lat, test_lon, test_alt,
                         ref_lat, ref_lon, ref_alt,
                         angle_units='deg', dist_units='m')
    dist_enu_km = np.linalg.norm(np.array([e, n, u])) / 1e3
    print(f'Distance computed via ENU coordinates is {dist_enu_km:.2f} km.')
    print(f'Error between ECEF/ENU distance calculations is: {abs(dist_km - dist_enu_km)*1e3:.4f} m.')

    # Compute distance via AER
    az, el, rng = enu_to_aer(e, n, u, angle_units='deg')
    print(f'Distance computed via AER coordinates is {rng/1e3:.2f} km.')
    print(f'Error between ECEF/AER distance calculations is: {abs(rng - dist_km*1e3):.4f} m.')
    print(f'Bearing from Lincoln Memorial to U.S. Capitol Building is {az:.2f} deg.')
    print(f'Elevation angle of U.S. Capitol as seen by Lincoln Memorial is {el:.4f} deg.')

    # -------------------------------------------------------------------------
    # LOS terrain profile (equivalent to MATLAB los2)
    # -------------------------------------------------------------------------
    dem_path = Path(__file__).parent.parent.parent / 'data' / 'nasadem_washingtondc.tif'

    # Sample N evenly-spaced points along the great-circle path (linear in
    # lat/lon is accurate enough over this ~3.7 km distance).
    n_pts = 500
    lats_path = np.linspace(ref_lat, test_lat, n_pts)
    lons_path = np.linspace(ref_lon, test_lon, n_pts)

    # Load DEM and sample elevations using bilinear interpolation
    with rasterio.open(str(dem_path)) as src:
        dem_data = src.read(1).astype(float)
        nodata = src.nodata
        tf = src.transform   # affine: tf.a = pixel_width, tf.e = pixel_height (<0), tf.c = left, tf.f = top

    dem_data[dem_data == nodata] = np.nan

    # Convert lat/lon path to fractional pixel coordinates
    # col = (lon - left) / pixel_width_deg
    # row = (lat - top)  / pixel_height_deg   (pixel_height_deg is negative → rows increase downward)
    cols = (lons_path - tf.c) / tf.a
    rows = (lats_path - tf.f) / tf.e
    terrain_elev = map_coordinates(dem_data, [rows, cols], order=1, mode='nearest')

    # Ground distances (m) from Lincoln Memorial along the path
    e_path, n_path, _ = lla_to_enu(lats_path, lons_path, np.zeros(n_pts),
                                   ref_lat, ref_lon, ref_alt,
                                   angle_units='deg', dist_units='m')
    dist_path_km = np.sqrt(e_path**2 + n_path**2) / 1e3

    obs_terrain_elev = terrain_elev[0]   # terrain height at Lincoln Memorial
    tgt_terrain_elev = terrain_elev[-1]  # terrain height at Capitol Building

    def _make_los_figure(h_obs_m, title):
        """Build one LOS profile figure for a given observer height above ground."""
        obs_elev = obs_terrain_elev + h_obs_m   # absolute observer elevation
        tgt_elev = tgt_terrain_elev              # target at ground level

        # Straight LOS line from observer to target (linear in elevation vs. distance)
        los_elev = np.linspace(obs_elev, tgt_elev, n_pts)

        # Terrain is blocking the LOS wherever it rises above the LOS line
        blocked = terrain_elev > los_elev

        fig, ax = plt.subplots()
        ax.fill_between(dist_path_km, terrain_elev, color='saddlebrown', alpha=0.6, label='Terrain')
        ax.plot(dist_path_km, terrain_elev, 'k', linewidth=0.8)
        ax.plot(dist_path_km, los_elev, 'r-', linewidth=1.5, label='Line of Sight')
        if np.any(blocked):
            ax.fill_between(dist_path_km, los_elev, terrain_elev,
                            where=blocked, color='red', alpha=0.4, label='LOS blocked')
        ax.set_xlabel('Distance from Lincoln Memorial [km]')
        ax.set_ylabel('Elevation [m]')
        ax.set_title(title)
        ax.legend(loc='upper left')
        ax.grid(True)
        return fig

    fig1 = _make_los_figure(0.,        'LOS Profile: Ground-Level Observer (0 ft AGL)')
    fig2 = _make_los_figure(10*0.3048, 'LOS Profile: Observer 10 ft Above Ground')

    return fig1, fig2


if __name__ == '__main__':
    run_all_examples()
    plt.show()
