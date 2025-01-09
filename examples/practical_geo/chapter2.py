import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
import utils
import triang
import tdoa
import fdoa
import hybrid

_rad2deg = 180.0/np.pi
_deg2rad = np.pi/180.0


def run_all_examples():
    """
    Run all chapter 2 examples and return a list of figure handles

    :return figs: list of figure handles
    """

    return [example1(), example2(), example3()]


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

    # Define source and sensor positions
    x_source = np.array([3, 3]) * 1e3

    x_aoa = np.array([[4], [0]]) * 1e3
    x_tdoa = np.array([[1, 3], [0, 0.5]]) * 1e3
    x_fdoa = np.array([[0, 0], [1, 2]]) * 1e3
    v_fdoa = np.array([[1, 1], [-1, -1]]) * np.sqrt(0.5) * 300  # 300 m/s, at -45 deg heading

    # Draw Geometry
    fig1 = plt.figure()
    def _plt_markers(do_aoa=False, do_tdoa=False, do_fdoa=False, _colors=('w', 'k', 'k', 'k')):
        plt.scatter(x_source[0], x_source[1], marker='x', color=_colors[0], label='Target')
        if do_aoa:
            plt.scatter(x_aoa[0], x_aoa[1], marker='o', color=_colors[1], label='AOA Sensors')
        if do_tdoa:
            plt.scatter(x_tdoa[0], x_tdoa[1], marker='s', color=_colors[2], label='TDOA Sensors')
        if do_fdoa:
            plt.scatter(x_fdoa[0], x_fdoa[1], marker='^', color=_colors[3], label='FDOA Sensors')
            for this_x, this_v in zip(x_fdoa.T, v_fdoa.T):  # transpose so the loop steps over sensors, not dimensions
                plt.arrow(x=this_x[0], y=this_x[1],
                          dx=this_v[0]/4, dy=this_v[1]/4,
                          width=.01, head_width=.05,
                          color=_colors[3])

    _plt_markers(do_aoa=True, do_tdoa=True, do_fdoa=True, _colors=colors)

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
    cov_z = block_diag([cov_psi], cov_r, cov_rr)

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


def example2():
    """
    Executes Example 2.2.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    9 January 2025

    :return: figure handle to generated graphic
    """

    return None


def example3():
    """
    Executes Example 2.3.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    9 January 2025

    :return: figure handle to generated graphic
    """

    return None