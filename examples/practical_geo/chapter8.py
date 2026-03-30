import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – registers '3d' projection
import time
from pathlib import Path

from ewgeo.tdoa import TDOAPassiveSurveillanceSystem
from ewgeo.triang import DirectionFinder
from ewgeo import tracker
from ewgeo.tracker import State
from ewgeo.tracker.transition import (ConstantVelocityMotionModel,
                                      BallisticMotionModel,
                                      ConstantTurnMotionModel)
from ewgeo.utils import print_progress, print_elapsed
from ewgeo.utils.constants import speed_of_light
from ewgeo.utils.constraints import fixed_alt
from ewgeo.utils.covariance import CovarianceMatrix
from ewgeo.utils.errors import draw_error_ellipse
from ewgeo.utils.unit_conversions import convert

_ft2m = convert(1, from_unit="ft", to_unit="m")
_rad2deg = convert(1, from_unit="rad", to_unit="deg")
_deg2rad = convert(1, from_unit="deg", to_unit="rad")

# Default output directory for animated GIFs produced by this chapter
_FIGS_DIR = Path(__file__).resolve().parent.parent.parent / 'figures' / 'practical_geo' / 'chapter8'

def run_all_examples():
    """
    Run all chapter 8 examples and return a list of figure handles

    :return figs: list of figure handles
    """

    return list(example1()) + list(example2()) + list(example3()) + list(example4()) + example1_vid() + example2_vid()


def example1(mc_params=None):
    """
    Executes Example 8.1.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    29 June 2025

    :return: figure handle to generated graphic
    """

    # Define sensor positions
    x_tdoa = np.array([[5e3,   0,  0, -5e3],
                       [  0, 5e3,  0,    0],
                       [30,  60, 30,   60]])
    num_dims, n_tdoa = np.shape(x_tdoa)

    # ===  Define target trajectory
    x_tgt_init = np.array([-50e3, 100e3, 20e3*_ft2m])
    vel = 200

    t_e_leg = 3*60 # turn at 3 min
    turn_rad = 50e3
    t_turn = np.pi/2*turn_rad/vel
    t_s_leg = 6*60

    t_inc = 10 # 10 seconds between track updates

    # Due East
    t_e_vec = np.arange(start=0, stop=t_e_leg+t_inc, step=t_inc)
    x_e_leg = x_tgt_init[:, np.newaxis] + np.array([vel, 0, 0])[:, np.newaxis] * t_e_vec[np.newaxis, :]

    # Turn to South
    t_turn_vec = np.arange(start=t_inc, step=t_inc, stop=t_turn)
    angle_turn = np.pi/2 * t_turn_vec/t_turn
    x_turn = x_e_leg[:, [-1]] + turn_rad * np.array([np.sin(angle_turn), np.cos(angle_turn)-1, np.zeros_like(angle_turn)])

    # Due South
    t_s_vec = np.arange(start=t_inc, step=t_inc, stop=t_s_leg+t_inc)
    x_s_leg = x_turn[:, [-1]] + np.array([0, -vel, 0])[:, np.newaxis] * t_s_vec[np.newaxis, :]

    # Combine legs
    x_tgt_full = np.concatenate((x_e_leg, x_turn, x_s_leg), axis=1)
    t_vec = np.arange(x_tgt_full.shape[1]) * t_inc
    num_time = len(t_vec)

    # plt.plot Geometry
    fig1 = plt.figure()
    plt.scatter(x_tdoa[0]/1e3, x_tdoa[1]/1e3, marker='o', label='Sensors')
    plt.plot(x_tgt_full[0]/1e3,x_tgt_full[1]/1e3, marker='v', markevery=[-1], label='Aircraft')
    plt.grid(True)

    # ===  Measurement Statistics
    ref_idx = 0
    sigma_toa = 10e-9
    cov_toa = (sigma_toa**2) * np.eye(n_tdoa)
    cov_roa = CovarianceMatrix(speed_of_light ** 2 * cov_toa)
    tdoa = TDOAPassiveSurveillanceSystem(x=x_tdoa, cov=cov_roa, ref_idx=ref_idx, variance_is_toa=False)

    # ===  Generate Measurements
    z = tdoa.measurement(x_tgt_full)
    noise = tdoa.cov.sample(num_samples=num_time)
    zeta = z + noise

    # ===  Set Up Tracker
    sigma_a = 1

    motion_model = tracker.MotionModel.make_motion_model('cv',num_dims,sigma_a**2)
    state_space = motion_model.state_space
    motion_model.update_time_step(t_inc)
    f = motion_model.f # generate state transition matrix
    q = motion_model.q # generate process noise covariance matrix

    msmt_model = tracker.MeasurementModel(pss=tdoa, state_space=motion_model.state_space)

    # ===  Initialize Track State
    x_pred = State(state_space=state_space, time=t_vec[0], state=None, covar=None)

    # Initialize position with TDOA estimate from first measurement
    x_init = np.array([0, 50e3, 5e3])
    epsilon = 100
    x_pred.position, _ = tdoa.gradient_descent(zeta=zeta[:, 0], x_init=x_init, epsilon=epsilon)
    x_pred.position_covar = tdoa.compute_crlb(x_source=x_pred.position).multiply(10, overwrite=False)

    # Bound initial velocity uncertainty by assumed max velocity of 340 m/s
    # (Mach 1 at sea level)
    max_vel = 340
    x_pred.velocity_covar = 10*max_vel**2*np.eye(num_dims)

    # ===  Step Through Time
    print('Iterating through EKF tracker time steps...')
    t_start = time.perf_counter()
    iterations_per_marker = 1
    markers_per_row = 40
    iterations_per_row = markers_per_row * iterations_per_marker

    x_ekf_est = np.zeros((num_dims,num_time))
    x_ekf_pred = np.zeros((num_dims, num_time))
    rmse_cov_est = np.zeros((num_time, ))
    rmse_cov_pred = np.zeros((num_time, ))
    num_ell_pts = 101 # number of points for ellipse drawing
    x_ell_est = np.zeros((2,num_ell_pts,num_time))
    for idx in range(num_time):
        print_progress(num_time, idx, iterations_per_marker, iterations_per_row, t_start)

        # Grab Current Measurement
        this_zeta = zeta[:, idx]

        # Update Position Estimate
        x_est = msmt_model.ekf_update(x_pred, this_zeta)

        # Predict state to the next time step
        x_pred = tracker.kf_predict(x_est, q, f)

        # Output the current prediction/estimation state
        x_ekf_est[:,idx] = x_est.position
        x_ekf_pred[:,idx] = x_pred.position

        rmse_cov_est[idx] = x_est.position_covar.rmse
        rmse_cov_pred[idx] = x_pred.position_covar.rmse

        # Draw an error ellipse
        x_est_xyz = x_est.position
        p_est_xyz = x_est.position_covar.cov
        x_est_xy = x_est_xyz[:2]
        p_est_xy = CovarianceMatrix(p_est_xyz[:2, :2])
        x_ell_est[:, :, idx] = draw_error_ellipse(x_est_xy, p_est_xy, num_ell_pts)

    print('done')
    t_elapsed = time.perf_counter() - t_start
    print_elapsed(t_elapsed)

    plt.scatter(x_init[0]/1e3, x_init[1]/1e3, marker='+',label='Initial Position Estimate')
    plt.plot(x_ekf_est[0]/1e3,x_ekf_est[1]/1e3,'--',label='EKF (est.)')
    plt.plot(x_ekf_pred[0]/1e3,x_ekf_pred[1]/1e3,'--',label='EKF (pred.)')
    plt.grid(True)
    plt.xlabel('x [km]')
    plt.ylabel('y [km]')
    plt.legend(loc='lower left')

    # ===  Compute Error
    err_pred = x_ekf_pred[:, :-2] - x_tgt_full[:, 1:-1]
    err_est = x_ekf_est - x_tgt_full

    rmse_pred = np.sqrt(np.sum(np.fabs(err_pred)**2, axis=0))
    rmse_est = np.sqrt(np.sum(np.fabs(err_est)**2,axis=0))

    fig2=plt.figure()
    plt.plot(t_vec,rmse_cov_est/1e3, label='RMSE (est. cov.)')
    plt.plot(t_vec[2:], rmse_cov_pred[:-2]/1e3, label='RMSE (pred. cov)')
    plt.plot(t_vec,rmse_est/1e3, '--', label='RMSE (est. act.)')
    plt.plot(t_vec[2:], rmse_pred/1e3, '--',label='RMSE (pred. act.)')
    plt.grid(True)
    plt.xlabel('Time [sec]')
    plt.ylabel('Error [km]')
    plt.yscale('log')
    plt.legend(loc='upper right')

    # === Repeat for Statistical Certainty
    num_monte_carlo = 1000
    if mc_params is not None:
        num_monte_carlo = max(int(num_monte_carlo/mc_params['monte_carlo_decimation']),mc_params['min_num_monte_carlo'])
    sse_pred = np.zeros((num_monte_carlo, num_time-1))
    sse_est = np.zeros((num_monte_carlo, num_time))
    sse_cov_pred = np.zeros((num_monte_carlo, num_time))
    sse_cov_est = np.zeros((num_monte_carlo, num_time))
    print('Repeating tracker test for {:d} Monte Carlo trials...'.format(num_monte_carlo))
    t_start = time.perf_counter()
    iterations_per_marker = 10
    markers_per_row = 40
    iterations_per_row = markers_per_row * iterations_per_marker

    for idx_mc in range(num_monte_carlo):
        print_progress(num_monte_carlo, idx_mc, iterations_per_marker, iterations_per_row, t_start)

        # Generate Measurements
        noise = tdoa.cov.sample(num_samples=num_time)
        zeta = z + noise

        # Initialize Track State
        x_pred = State(state_space=state_space, time=t_vec[0], state=None, covar=None)

        x_init = np.array([0, 50e3, 5e3])
        epsilon = 100
        x_pred.position, _ = tdoa.gradient_descent(zeta[:, 0], x_init=x_init,epsilon=epsilon)
        x_pred.position_covar = tdoa.compute_crlb(x_source=x_pred.position)

        # Bound initial velocity uncertainty by assumed max velocity of 340 m/s
        # (Mach 1 at sea level)
        x_pred.velocity_covar = 10 * max_vel**2*np.eye(num_dims)

        # Step Through Time
        x_ekf_est = np.zeros((num_dims,num_time))
        x_ekf_pred = np.zeros((num_dims, num_time))
        for idx in np.arange(num_time):
            # Grab Current Measurement
            this_zeta = zeta[:, idx]

            # Update Position Estimate
            x_est = msmt_model.ekf_update(x_pred, this_zeta, tdoa.cov)

            # Predict state to the next time step
            x_pred = tracker.kf_predict(x_est, q, f)

            # Output the current prediction/estimation state
            x_ekf_est[:, idx] = x_est.position
            x_ekf_pred[:, idx] = x_pred.position

            sse_cov_est[idx_mc, idx] = x_est.position_covar.trace
            sse_cov_pred[idx_mc, idx] = x_pred.position_covar.trace

        err_pred = x_ekf_pred[:, :-1] - x_tgt_full[:, 1:]
        err_est = x_ekf_est - x_tgt_full

        sse_pred[idx_mc,:] = np.sum(np.fabs(err_pred)**2, axis=0)
        sse_est[idx_mc,:] = np.sum(np.fabs(err_est)**2, axis=0)

    print('done')
    t_elapsed = time.perf_counter() - t_start
    print_elapsed(t_elapsed)

    rmse_pred = np.sqrt(np.mean(sse_pred, axis=0))
    rmse_est = np.sqrt(np.mean(sse_est, axis=0))

    rmse_cov_est = np.sqrt(np.mean(sse_cov_est, axis=0))
    rmse_cov_pred = np.sqrt(np.mean(sse_cov_pred, axis=0))

    fig3=plt.figure()
    hdl_est = plt.plot(t_vec, rmse_cov_est/1e3, label='RMSE (est. cov.)')
    hdl_pred = plt.plot(t_vec[1:], rmse_cov_pred[:-1]/1e3, label='RMSE (pred. cov)')
    plt.plot(t_vec, rmse_est/1e3, '--', label='RMSE (est. act.)', color=hdl_est[0].get_color())
    plt.plot(t_vec[1:], rmse_pred/1e3, '--', label='RMSE (pred. act.)', color=hdl_pred[0].get_color())
    plt.grid(True)
    plt.xlabel('Time [sec]')
    plt.ylabel('Error [km]')
    plt.yscale('log')

    plt.legend(loc='upper right')

    # === Return Figure Handles
    figs = [fig1, fig2, fig3]

    return figs


def example2(rng=np.random.default_rng()):
    """
    Executes Example 8.2.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    29 June 2025

    :param rng: random number generator
    :return: figure handle to generated graphic
    """
    print('Example 8.2...\n')

    # ===  Define ship and sensor positions
    t_inc = 30
    t_max = 15*60
    t_vec = np.arange(start=0, step=t_inc, stop=t_max+t_inc)  # seconds
    num_time = len(t_vec)

    # Origin is 0,0.  Alt is 10 km. Velocity is +y at 100 m/s.
    # Let's make it a function for simplicity
    x_aoa_full = np.array([0, 0, 10e3])[:, np.newaxis] + np.array([0, 100, 0])[:, np.newaxis] * t_vec[np.newaxis, :]

    # Origin is 50 km, 50 km. Alt is 0 m.  Velocity is -x at 5 m/s.
    ship_accel_power = .05
    a_tgt_full = np.concatenate((-ship_accel_power + 2*ship_accel_power*rng.uniform(low=0, high=1, size=(2, num_time)), np.zeros((1, num_time))), axis=0)
    v_tgt_full = np.array([-20, 0, 0])[:, np.newaxis] + np.cumsum(a_tgt_full*t_inc,axis=1)
    x_tgt_full = np.array([50e3, 50e3, 0])[:, np.newaxis] + np.cumsum(v_tgt_full*t_inc, axis=1)

    beam_plot_times = [0,1,2,np.floor(num_time/2).astype(int),num_time-1]
    fig1=plt.figure()
    plt.plot(x_aoa_full[0]/1e3, x_aoa_full[1]/1e3, '-^', markevery=beam_plot_times, label='AOA Sensor Trajectory')
    plt.plot(x_tgt_full[0]/1e3, x_tgt_full[1]/1e3, '-<', markevery=beam_plot_times, label='Target Trajectory')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.xlabel('x [km]')
    plt.ylabel('y [km]')

    # ===  Measurement Statistics
    sigma_theta = 1 # deg
    sigma_psi = sigma_theta*_deg2rad # rad
    num_msmt=2 # az/el
    cov_psi = CovarianceMatrix(sigma_psi**2*np.eye(num_msmt)) # measurement error covariance
    aoa = DirectionFinder(x_aoa_full[:, 0], cov=cov_psi, do_2d_aoa=True)

    # Add DF overlays
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fill_color = colors[5]
    label_fill='DF Error'
    for idx_overlay in beam_plot_times:
        this_ac_pos = x_aoa_full[:2, idx_overlay]
        this_tgt_pos = x_tgt_full[:2,idx_overlay]

        # Az/El
        # Since we only fed in 2D sensor/tgt positions, that overrides the do_2d_aoa flag and forces a return
        # of just azimuth measurement
        psi_true = aoa.measurement(x_source=this_tgt_pos, x_sensor=this_ac_pos)
        psi_err_plus = psi_true + sigma_psi
        psi_err_minus = psi_true - sigma_psi

        # Lobs are returned as num_dim x 2 x num_sensor x num_cases
        zeta_full = np.concatenate((psi_err_plus[:, np.newaxis], psi_err_minus[:, np.newaxis]), axis=1)
        lobs = aoa.draw_lobs(x_source=this_tgt_pos, zeta=zeta_full, scale=2, x_sensor=this_ac_pos)
        lob_plus = np.squeeze(lobs[:, :, 0, 0])
        lob_minus = np.squeeze(lobs[:, :, 0, 1])
        lob_fill = np.concatenate((lob_plus,lob_minus[:, [-1]]), axis=1)
        fill_patch = plt.Polygon(lob_fill.T/1e3, linestyle='--', edgecolor='k', facecolor=fill_color, alpha=.3,
                                 label=label_fill)
        fig1.gca().add_patch(fill_patch)
        label_fill = None

    plt.xlim([-5,55])
    plt.ylim([-5,105])

    # ===  Generate Measurements
    zeta = np.zeros((aoa.num_measurements, num_time))
    for idx in range(num_time):
        zeta[:, idx] = aoa.noisy_measurement(x_source=x_tgt_full[:, idx],
                                             x_sensor=x_aoa_full[:, idx])

    # ===  Set Up Tracker
    # Track in x/y, even though a/c and tgt in x/y/z
    sigma_a = .05
    num_dims = 3 # number of dimensions to use in state

    motion_model = tracker.MotionModel.make_motion_model('cv',num_dims,sigma_a**2)
    state_space = motion_model.state_space
    motion_model.update_time_step(t_inc)
    f = motion_model.f # generate state transition matrix
    q = motion_model.q # generate process noise covariance matrix

    # ===  Initialize Track State
    # Initialize position with AOA estimate from first three measurement
    x_init_guess = np.array([10e3, 10e3, 0])
    bnd, bnd_grad = fixed_alt(0, 'flat')
    x_init, _ = aoa.gradient_descent(zeta=zeta[:, 0], x_init=x_init_guess,
                                     eq_constraints=[bnd], epsilon=1e-3)
    p_init = aoa.compute_crlb(x_source=x_init, eq_constraints_grad=[bnd_grad]).cov
    init_state = State(state_space=state_space, time=t_vec[0], state=None, covar=None)
    init_state.position = x_init
    init_state.position_covar = p_init

    # Bound initial velocity uncertainty by assumed max velocity of 10 m/s
    # (~20 knots)
    max_vel = 10
    p_vel = max_vel**2*np.eye(num_dims)
    init_state.velocity = 0.
    init_state.velocity_covar = p_vel

    def constrain_state(s: State):
        # Constrain a state, in-place, to have no vertical component or vertical velocity
        s.position[-1] = 0.  # no altitude
        s.velocity[-1] = 0.  # no vertical motion
        s.position_covar.cov[-1, :] = 0.  # no altitude error
        s.position_covar.cov[:, -1] = 0.
        s.velocity_covar.cov[-1, :] = 0.  # no vertical velocity error
        s.velocity_covar.cov[:, -1] = 0.
        return
    constrain_state(init_state)

    msmt_model = tracker.MeasurementModel(pss=aoa, state_space=state_space)

    # ===  Step Through Time
    print('Iterating through EKF tracker time steps...')
    markers_per_row = 40
    iter_per_marker = 1
    iter_per_row = markers_per_row * iter_per_marker

    t_start = time.perf_counter()

    x_ekf_est = np.zeros((num_dims,num_time))
    x_ekf_pred = np.zeros((num_dims, num_time))
    rmse_cov_est = np.zeros((num_time,))
    rmse_cov_pred = np.zeros((num_time,))
    num_ell_pts = 101 # number of points for ellipse drawing
    x_ell_est = np.zeros((2,num_ell_pts,num_time))
    for idx in range(num_time):
        print_progress(num_total=num_time, curr_idx=idx, iterations_per_marker=iter_per_marker,
                       iterations_per_row=iter_per_row, t_start=t_start)

        # Grab Current Measurement
        this_zeta = zeta[:, idx]

        # Update sensor positions
        aoa.pos = x_aoa_full[:num_dims, idx]

        # Update Position Estimate
        if idx==0:
            # Start with the initial state we computed before the loop began
            x_est = init_state
        else:
            # Update the prior prediction with the current measurement
            x_est = tracker.ekf_update(x_pred, this_zeta, aoa.cov, msmt_model.measurement, msmt_model.jacobian)

            # Enforce known altitude
            constrain_state(x_est)

        # Predict state to the next time step
        x_pred = tracker.kf_predict(x_est, q, f)

        # Enforce known altitude
        constrain_state(x_pred)

        # Output the current prediction/estimation state
        x_ekf_est[:, idx] = x_est.position
        x_ekf_pred[:,idx] = x_pred.position

        rmse_cov_est[idx] = x_est.position_covar.rmse
        rmse_cov_pred[idx]= x_pred.position_covar.rmse

        # Draw an error ellipse
        p_pos_xy = CovarianceMatrix(x_est.position_covar.cov[:2, :2])
        x_ell_est[:, :, idx] = draw_error_ellipse(x=x_est.position[:2],
                                                  covariance=p_pos_xy,
                                                  num_pts=num_ell_pts)

    print('done')
    t_elapsed = time.perf_counter() - t_start
    print_elapsed(t_elapsed)

    plt.plot(x_init[0]/1e3, x_init[1]/1e3, '+',label='Initial Position Estimate')
    plt.plot(x_ekf_est[0]/1e3,x_ekf_est[1]/1e3,'-',label='EKF (est.)')
    plt.plot(x_ekf_pred[0]/1e3,x_ekf_pred[1]/1e3,'-',label='EKF (pred.)')
    plt.grid(True)
    plt.legend()

    # ===  Zoomed plt.plot on Target
    fig2=plt.figure()
    plt.plot(x_tgt_full[0]/1e3, x_tgt_full[1]/1e3,'-<', markevery=[-1], label='Target Trajectory')
    plt.plot(x_init[0]/1e3, x_init[1]/1e3, '+',label='Initial Position Estimate')
    plt.plot(x_ekf_est[0]/1e3,x_ekf_est[1]/1e3,'-',label='EKF (est.)')
    plt.plot(x_ekf_pred[0]/1e3,x_ekf_pred[1]/1e3,'-',label='EKF (pred.)')
    plt.grid(True)
    plt.xlim([30,50])
    plt.legend(loc='lower left')
    plt.xlabel('x [km]')
    plt.ylabel('y [km]')

    ## Compute Error
    err_pred = x_ekf_pred[:, :-1] - x_tgt_full[:num_dims, 1:]
    err_est = x_ekf_est - x_tgt_full[:num_dims]

    rmse_pred = np.sqrt(np.sum(np.fabs(err_pred)**2, axis=0))
    rmse_est = np.sqrt(np.sum(np.fabs(err_est)**2, axis=0))

    fig3=plt.figure()
    hdl0 =plt.plot(t_vec,rmse_cov_est/1e3,label='RMSE (est. cov.)')

    hdl1 =plt.plot(t_vec[1:], rmse_cov_pred[:-1]/1e3, label='RMSE (pred. cov)')
    # set(gca,'ColorOrderIndex',1)
    plt.plot(t_vec,rmse_est/1e3,'--',label='RMSE (est. act.)', color=hdl0[0].get_color())
    plt.plot(t_vec[1:], rmse_pred/1e3, '--', label='RMSE (pred. act.)', color=hdl1[0].get_color())
    plt.grid(True)
    plt.xlabel('Time [sec]')
    plt.ylabel('Error [km]')
    plt.legend(loc='upper right')
    plt.yscale('log')

    # ===  Return Figure Handles
    figs = [fig1, fig2, fig3]
    return figs


def example3(rng=np.random.default_rng(0)):
    """
    Example 8.3: Ballistic Trajectory Tracking.

    A projectile is launched and tracked using four ground-based TDOA sensors.
    Two EKF variants are compared:

    - Constant-Velocity (CV) model: gravity is unmodelled; must rely on large
      process noise to follow the curved descent.  This inflates estimation noise
      throughout the trajectory.
    - Ballistic motion model: gravity is an explicit deterministic forcing term on
      the state mean.  A small kinematic process noise is sufficient, yielding
      accurate tracking through ascent and descent.

    :param rng: numpy random-number generator (seeded for reproducibility)
    :return: list of figure handles [trajectory profile, RMSE vs time]
    """

    # --- Target trajectory: ballistic, 3-D --------------------------------
    g_scalar = -9.80665         # m/s²  (downward, applied to z-axis)
    t_inc    = 1.0              # seconds between updates
    v_init   = np.array([100., 80., 200.])   # initial velocity [m/s]: east, north, up
    x_init   = np.array([0., 0., 0.])        # launch point (ground level)

    t_flight = -2. * v_init[2] / g_scalar    # total flight time ≈ 40.8 s
    t_vec    = np.arange(0., t_flight, t_inc)
    num_time = len(t_vec)

    gravity_vec  = np.array([0., 0., g_scalar])
    x_tgt_full = (x_init[:, np.newaxis]
                  + v_init[:, np.newaxis] * t_vec[np.newaxis, :]
                  + 0.5 * gravity_vec[:, np.newaxis] * t_vec[np.newaxis, :] ** 2)

    # --- TDOA sensors: cross pattern at 3 km (ground level) ---------------
    x_tdoa = np.array([[ 3e3,  0.,  -3e3,  0. ],
                       [ 0.,   3e3,  0.,  -3e3],
                       [ 10.,  10.,  10.,  10.]])
    n_tdoa = x_tdoa.shape[1]

    ref_idx   = 0
    sigma_toa = 300e-9    # 300 ns → ~90 m RDOA noise
    cov_toa   = CovarianceMatrix(sigma_toa ** 2 * np.eye(n_tdoa))
    tdoa      = TDOAPassiveSurveillanceSystem(x=x_tdoa, cov=cov_toa, ref_idx=ref_idx,
                                              variance_is_toa=True)
    zeta = tdoa.noisy_measurement(x_tgt_full)

    # --- Shared initial conditions ----------------------------------------
    x_init = np.array([300., -200., 200.])   # intentional offset [m] ENU
    v_init = np.array([50., 50., 100.])      # intentional wrong [m/s] ENU

    # CV tracker: large process noise to compensate for unmodelled gravity
    mm_cv = ConstantVelocityMotionModel(num_dims=3, process_covar=50. ** 2)
    x0_cv = State(state_space=mm_cv.state_space, time=0, state=None, covar=None)
    x0_cv.position = x_init         # start with perturbed position
    x0_cv.velocity = v_init         # start with perturbed velocity assumption
    x0_cv.position_covar = np.eye(3) * (500. **2)
    x0_cv.velocity_covar = np.eye(3) * (300. **2)

    # Ballistic tracker: gravity is modelled → small kinematic process noise suffices
    mm_bal = BallisticMotionModel(process_covar=3. ** 2)
    # as it happens, the BallisticMotionModel uses the same state space as CV
    x0_bal = State(state_space=mm_bal.state_space, time=x0_cv.time,
                   state=x0_cv.state, covar=x0_cv.covar) # copy the initial conditions

    # Measurement Model for Trackers
    # Both Ballistic and CV motion models use the same state space, so we can use a common
    # measurement model for both.
    msmt = tracker.MeasurementModel(pss=tdoa, state_space=mm_bal.state_space)

    # Run the tracker loop on both trackers
    x_cv = []
    x_bal = []
    for idx in range(num_time):
        # Grab current measurement and predict states forward to the current time
        t_now = t_vec[idx]
        this_zeta = zeta[:, idx]
        if idx==0:
            # On the first iteration, use our initial state as the predicted
            # state
            x_cv_pred = x0_cv
            x_bal_pred = x0_bal
        else:
            # Predict the prior estimated state forward to the current time
            x_cv_pred = mm_cv.predict(x_cv[-1], t_now)
            x_bal_pred = mm_bal.predict(x_bal[-1], t_now)

        # Update the estimates with new data
        x_cv_est = msmt.ekf_update(x_cv_pred, this_zeta)
        x_bal_est = msmt.ekf_update(x_bal_pred, this_zeta)

        # Append the states
        x_cv.append(x_cv_est)
        x_bal.append(x_bal_est)

    # Extract the position from each set of estimated states
    x_cv  = np.array([x.position for x in x_cv]).T
    x_bal = np.array([x.position for x in x_bal]).T

    # --- Figure 1: range–altitude profile ---------------------------------
    rng_tgt = np.hypot(x_tgt_full[0], x_tgt_full[1])
    rng_cv  = np.hypot(x_cv[0],  x_cv[1])
    rng_bal = np.hypot(x_bal[0], x_bal[1])

    fig1, ax1 = plt.subplots()
    ax1.plot(rng_tgt / 1e3, x_tgt_full[2] / 1e3, 'k-',  linewidth=2, label='True trajectory')
    ax1.plot(rng_cv  / 1e3, x_cv[2]  / 1e3,       '--',  label='CV model')
    ax1.plot(rng_bal / 1e3, x_bal[2] / 1e3,       '-.',  label='Ballistic model')
    ax1.set_xlabel('Horizontal range [km]')
    ax1.set_ylabel('Altitude [km]')
    ax1.legend()
    ax1.grid(True)
    ax1.set_title('Example 8.3: Ballistic Trajectory – Range/Altitude Profile')

    # --- Figure 2: 3-D position RMSE vs time ------------------------------
    rmse_cv  = np.sqrt(np.sum((x_cv  - x_tgt_full) ** 2, axis=0))
    rmse_bal = np.sqrt(np.sum((x_bal - x_tgt_full) ** 2, axis=0))

    fig2, ax2 = plt.subplots()
    ax2.semilogy(t_vec, rmse_cv  / 1e3, label='CV model')
    ax2.semilogy(t_vec, rmse_bal / 1e3, label='Ballistic model')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('3-D RMSE [km]')
    ax2.legend()
    ax2.grid(True)
    ax2.set_title('Example 8.3: Position RMSE vs Time')

    return [fig1, fig2]


def example4(rng=np.random.default_rng(0)):
    """
    Example 8.4: Constant-Turn Aircraft Tracking.

    An aircraft executes a sustained coordinated level turn (one full circle)
    tracked by four ground-based TDOA sensors.  Two EKF variants are compared:

    - Constant-Velocity (CV) model: the centripetal acceleration is unmodelled;
      a large process noise is required to follow the turn, producing noisy estimates.
    - Constant-Turn (CT) model: the turn rate ω is a tracked state.  A small
      kinematic process noise suffices, yielding smooth, accurate estimates
      throughout the manoeuvre.

    The CT model tracks ω from an initial value of zero, converging to the true
    turn rate after a few observations.

    :param rng: numpy random-number generator (seeded for reproducibility)
    :return: list of figure handles [x-y trajectory, RMSE vs time, ω estimate]
    """

    # --- Target trajectory: constant-rate level turn ----------------------
    omega_true = np.pi / 60.       # rad/s  → one full circle in 120 s (standard-rate turn)
    v0         = 200.              # m/s
    alt        = 3e3               # m  (constant altitude)
    R          = v0 / omega_true   # turn radius ≈ 3820 m

    t_inc    = 5.0                         # seconds between updates
    t_max    = 2. * np.pi / omega_true     # 120 s  (full circle)
    t_vec    = np.arange(0., t_max, t_inc)
    num_time = len(t_vec)

    # Exact CT trajectory starting at [0, 0, alt] heading north (+y)
    x_tgt_full = np.array([
        -R * (1. - np.cos(omega_true * t_vec)),   # px: circles left (west)
         R *       np.sin(omega_true * t_vec),    # py
        np.full(num_time, alt)                    # pz: level
    ])
    vx_true = -v0 * np.sin(omega_true * t_vec)
    vy_true =  v0 * np.cos(omega_true * t_vec)

    # --- TDOA sensors: rectangle surrounding the turn circle --------------
    # Circle centre ≈ (−R, 0, alt); bounding box x ∈ [−2R, 0], y ∈ [−R, R]
    x_tdoa = np.array([[ 2e3, -9e3, -9e3,  2e3],
                       [ 5e3,  5e3, -5e3, -5e3],
                       [  0.,   0.,   0.,   0.]])
    n_tdoa = x_tdoa.shape[1]

    sigma_toa = 10e-9    # 10 ns → ~3 m RDOA noise
    cov_toa   = CovarianceMatrix(sigma_toa ** 2 * np.eye(n_tdoa))
    tdoa      = TDOAPassiveSurveillanceSystem(x=x_tdoa, cov=cov_toa,
                                              variance_is_toa=True)

    zeta    = tdoa.noisy_measurement(x_tgt_full)

    # --- Shared initial conditions ----------------------------------------
    x_init = np.array([400., -400., 100.])

    # CV tracker
    # CV: tuned for a non-maneuvering target (σ_a = 1 m/s²). The centripetal
    # acceleration is ~10.5 m/s², so each 5-second prediction step introduces a
    # ~130 m position error — roughly 5× the CV process-noise σ_pos of ~25 m.
    # This systematic mismatch causes visible filter lag during the turn.
    mm_cv = ConstantVelocityMotionModel(num_dims=3, process_covar=1. ** 2)
    x0_cv = State(state_space=mm_cv.state_space, time=0, state=None, covar=None)
    x0_cv.position = x_tgt_full[:, 0] + x_init
    x0_cv.velocity = [0., v0, 0.]
    x0_cv.position_covar = (500. ** 2) * np.eye(3)
    x0_cv.velocity_covar = (100. ** 2) * np.eye(3)
    msmt_cv = tracker.MeasurementModel(pss=tdoa, state_space=mm_cv.state_space)

    # CT tracker: ω initially unknown → start at 0 with generous uncertainty
    sigma_omega_init = 0.1  # rad/s  (covers ±2σ of the true ω ≈ 0.052 rad/s)
    # CT: same kinematic σ_a as CV for a fair comparison; ω treated as nearly
    # constant (process_covar_omega = 1e-6 → σ_ω ≈ 0.002 rad/s per step),
    # which forces the filter to maintain its ω estimate between updates.
    mm_ct = ConstantTurnMotionModel(num_dims=3, process_covar=2. ** 2,
                                    process_covar_omega=1e-3)
    x0_ct = State(state_space=mm_ct.state_space, time=0, state=None, covar=None)
    x0_ct.position = x_tgt_full[:, 0] + x_init
    x0_ct.velocity = [0., v0, 0.]
    x0_ct.state[-1] = 0.  # ω unknown at start
    x0_ct.position_covar = (500. ** 2) * np.eye(3)
    x0_ct.velocity_covar = (100. ** 2) * np.eye(3)
    x0_ct.covar.cov[-1, -1] = sigma_omega_init ** 2
    msmt_ct = tracker.MeasurementModel(pss=tdoa, state_space=mm_ct.state_space)

    # --- EKF Loop -------------------------------------------------------
    x_cv = []
    x_ct = []
    for idx in range(num_time):
        # Grab the current time and measurement
        t_now = t_vec[idx]
        this_zeta = zeta[:, idx]
        # Predict the tracks forward to current time
        if idx==0:
            x_cv_pred = x0_cv
            x_ct_pred = x0_ct
        else:
            x_cv_pred = mm_cv.predict(x_cv[-1], t_now)
            x_ct_pred = mm_ct.predict(x_ct[-1], t_now)

        # EKF Update with new measurement
        x_cv_est = msmt_cv.ekf_update(x_cv_pred, this_zeta)
        x_ct_est = msmt_ct.ekf_update(x_ct_pred, this_zeta)

        # Append track estimates to the list
        x_cv.append(x_cv_est)
        x_ct.append(x_ct_est)

    # Extract parameters from state lists
    x_cv = np.array([x.position for x in x_cv]).T      # extract positions
    omega_est = np.array([x.state[-1] for x in x_ct])  # extract turn rate
    x_ct = np.array([x.position for x in x_ct]).T      # extract position

    # --- Figure 1: x-y trajectory -----------------------------------------
    fig1, ax1 = plt.subplots()
    ax1.plot(x_tgt_full[0] / 1e3, x_tgt_full[1] / 1e3, 'k-', linewidth=2,
             label='True trajectory')
    ax1.plot(x_cv[0] / 1e3, x_cv[1] / 1e3,  '--', label='CV model')
    ax1.plot(x_ct[0] / 1e3, x_ct[1] / 1e3,  '-.', label='CT model')
    ax1.scatter(*x_tdoa[:2] / 1e3, marker='^', zorder=5, label='Sensors')
    ax1.set_xlabel('x [km]')
    ax1.set_ylabel('y [km]')
    ax1.set_aspect('equal')
    ax1.legend()
    ax1.grid(True)
    ax1.set_title('Example 8.4: Constant-Turn Aircraft – x-y Trajectory')

    # --- Figure 1b: 3-D isometric trajectory ------------------------------
    fig1b = plt.figure()
    ax1b = fig1b.add_subplot(111, projection='3d')
    ax1b.plot(x_tgt_full[0] / 1e3, x_tgt_full[1] / 1e3, x_tgt_full[2] / 1e3,
              'k-', linewidth=2, label='True trajectory')
    ax1b.plot(x_cv[0] / 1e3, x_cv[1] / 1e3, x_cv[2] / 1e3,
              '--', label='CV model')
    ax1b.plot(x_ct[0] / 1e3, x_ct[1] / 1e3, x_ct[2] / 1e3,
              '-.', label='CT model')
    ax1b.scatter(x_tdoa[0] / 1e3, x_tdoa[1] / 1e3, x_tdoa[2] / 1e3,
                 marker='^', zorder=5, label='Sensors')
    ax1b.set_xlabel('x [km]')
    ax1b.set_ylabel('y [km]')
    ax1b.set_zlabel('z [km]')
    ax1b.view_init(elev=30, azim=45)
    ax1b.legend()
    ax1b.grid(True)
    ax1b.set_title('Example 8.4: Constant-Turn Aircraft – 3-D Trajectory')

    # --- Figure 2: 3-D position RMSE vs time ------------------------------
    rmse_cv = np.sqrt(np.sum((x_cv - x_tgt_full) ** 2, axis=0))
    rmse_ct = np.sqrt(np.sum((x_ct - x_tgt_full) ** 2, axis=0))

    fig2, ax2 = plt.subplots()
    ax2.semilogy(t_vec, rmse_cv / 1e3, label='CV model')
    ax2.semilogy(t_vec, rmse_ct / 1e3, label='CT model')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('3-D RMSE [km]')
    ax2.legend()
    ax2.grid(True)
    ax2.set_title('Example 8.4: Position RMSE vs Time')

    # --- Figure 3: estimated turn rate vs time ----------------------------
    fig3, ax3 = plt.subplots()
    ax3.axhline(np.degrees(omega_true), color='k', linewidth=2, label='True ω')
    ax3.plot(t_vec, np.degrees(omega_est), label='CT estimate')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Turn rate [deg/s]')
    ax3.legend()
    ax3.grid(True)
    ax3.set_title('Example 8.4: CT Model – Estimated Turn Rate')

    return [fig1, fig1b, fig2, fig3]


def example1_vid(save_path=_FIGS_DIR / 'example1_vid.gif'):
    """
    Animated version of Example 8.1.

    Runs the same single-target TDOA EKF as example1(), then produces a
    2×2 animated figure showing how the tracker evolves over time:

      Panel 1 (top-left):  RDOA measurements vs time with a moving marker.
      Panel 2 (top-right): Position RMSE and predicted CEP50 vs time (semilogy).
      Panel 3 (bot-left):  Full x-y map — sensor positions, full truth trajectory
                           (faint), error ellipse, and a fading tail of the last
                           num_tail EKF estimates.
      Panel 4 (bot-right): Zoomed x-y view that follows the target.

    Returns a single figure.  The animation object is stored on the figure as
    fig._anim to prevent garbage collection.

    :return: [fig]
    """
    from matplotlib.animation import FuncAnimation

    # ===  Sensor and target setup (same as example1 / MATLAB vid8_1) ===
    x_tdoa = np.array([[5e3,   0,  0, -5e3],
                       [  0, 5e3,  0,    0],
                       [30,  60, 30,   60]])
    num_dims, n_tdoa = np.shape(x_tdoa)

    x_tgt_init = np.array([-50e3, 100e3, 20e3 * _ft2m])
    vel = 200

    t_e_leg  = 3 * 60
    turn_rad = 50e3
    t_turn   = np.pi / 2 * turn_rad / vel
    t_s_leg  = 6 * 60
    t_inc    = 10

    t_e_vec    = np.arange(0, t_e_leg + t_inc, t_inc)
    x_e_leg    = x_tgt_init[:, np.newaxis] + np.array([vel, 0, 0])[:, np.newaxis] * t_e_vec
    t_turn_vec = np.arange(t_inc, t_turn, t_inc)
    angle_turn = np.pi / 2 * t_turn_vec / t_turn
    x_turn     = x_e_leg[:, [-1]] + turn_rad * np.array(
        [np.sin(angle_turn), np.cos(angle_turn) - 1, np.zeros_like(angle_turn)])
    t_s_vec    = np.arange(t_inc, t_s_leg + t_inc, t_inc)
    x_s_leg    = x_turn[:, [-1]] + np.array([0, -vel, 0])[:, np.newaxis] * t_s_vec

    x_tgt_full = np.concatenate((x_e_leg, x_turn, x_s_leg), axis=1)
    t_vec      = np.arange(x_tgt_full.shape[1]) * t_inc
    num_time   = len(t_vec)

    ref_idx   = 0
    sigma_toa = 10e-9
    cov_toa   = CovarianceMatrix(sigma_toa ** 2 * np.eye(n_tdoa))
    cov_roa   = CovarianceMatrix(speed_of_light ** 2 * cov_toa.cov)
    tdoa      = TDOAPassiveSurveillanceSystem(x=x_tdoa, cov=cov_roa,
                                              ref_idx=ref_idx,
                                              variance_is_toa=False)
    z    = tdoa.measurement(x_tgt_full)
    zeta = z + tdoa.cov.sample(num_samples=num_time)

    sigma_a      = 10   # match MATLAB vid8_1 (larger than example1's sigma_a=1)
    motion_model = tracker.MotionModel.make_motion_model('cv', num_dims, sigma_a ** 2)
    state_space  = motion_model.state_space
    motion_model.update_time_step(t_inc)
    f = motion_model.f
    q = motion_model.q

    msmt_model = tracker.MeasurementModel(pss=tdoa, state_space=state_space)

    x_pred = tracker.State(state_space=state_space, time=t_vec[0], state=None, covar=None)
    x_init = np.array([0., 50e3, 5e3])
    x_pred.position, _ = tdoa.gradient_descent(zeta=zeta[:, 0], x_init=x_init, epsilon=100)
    x_pred.position_covar = tdoa.compute_crlb(x_source=x_pred.position).multiply(10, overwrite=False)
    max_vel = 340
    x_pred.velocity_covar = 10 * max_vel ** 2 * np.eye(num_dims)

    # ===  EKF loop — store all estimates, errors, and ellipses ===
    num_ell_pts  = 101
    x_ekf_est    = np.zeros((num_dims, num_time))
    rmse_est     = np.zeros(num_time)
    rmse_cov_est = np.zeros(num_time)
    x_ell_est    = np.zeros((2, num_ell_pts, num_time))

    for idx in range(num_time):
        this_zeta = zeta[:, idx]
        x_est  = msmt_model.ekf_update(x_pred, this_zeta)
        x_pred = motion_model.predict(x_est, new_time=t_vec[idx]+t_inc)

        x_ekf_est[:, idx] = x_est.position
        err = np.linalg.norm(x_est.position - x_tgt_full[:, idx])
        rmse_est[idx]     = err
        rmse_cov_est[idx] = x_est.position_covar.rmse

        p_xy = CovarianceMatrix(x_est.position_covar.cov[:2, :2])
        x_ell_est[:, :, idx] = draw_error_ellipse(x_est.position[:2], p_xy, num_ell_pts)

    # ===  Figure setup ===
    num_tail = 10
    num_msmt = zeta.shape[0]
    offset   = 10e3   # zoom half-width [m]
    scale    = 1e3

    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    ax_msmt, ax_err, ax_xy, ax_zoom = axs.flatten()

    # Panel 1: full RDOA time series (static) + animated current-time marker
    for ch in range(num_msmt):
        ax_msmt.plot(t_vec, zeta[ch], label='Measurements' if ch == 0 else None)
    marker_msmt = ax_msmt.scatter([t_vec[num_tail]] * num_msmt, zeta[:, num_tail],
                                  c='k', zorder=5, s=30)
    ax_msmt.set_xlabel('Time [s]')
    ax_msmt.set_ylabel('RDOA [m]')
    ax_msmt.legend(loc='upper right')
    ax_msmt.grid(True)

    # Panel 2: RMSE + CEP50 (static) + animated markers
    ax_err.semilogy(t_vec, rmse_est,     label='Measured')
    ax_err.semilogy(t_vec, rmse_cov_est, label='Predicted (CEP50)')
    marker_err = ax_err.scatter([t_vec[num_tail]], [rmse_est[num_tail]],     c='k', zorder=5, s=30)
    marker_cep = ax_err.scatter([t_vec[num_tail]], [rmse_cov_est[num_tail]], c='k', zorder=5, s=30)
    ax_err.set_xlabel('Time [s]')
    ax_err.set_ylabel('Error [m]')
    ax_err.legend(loc='upper right')
    ax_err.grid(True)

    # Panel 3: full x-y map
    ax_xy.scatter(x_tdoa[0] / scale, x_tdoa[1] / scale, marker='^', zorder=5, label='Sensors')
    ax_xy.plot(x_tgt_full[0] / scale, x_tgt_full[1] / scale, 'k-', alpha=0.2)
    K0      = num_tail
    tail_x0 = x_ekf_est[0, :K0 + 1] / scale
    tail_y0 = x_ekf_est[1, :K0 + 1] / scale
    tail_xy,    = ax_xy.plot(tail_x0, tail_y0, 'b-o', ms=3, label='Estimated Position')
    ell_xy,     = ax_xy.plot(x_ell_est[0, :, K0] / scale, x_ell_est[1, :, K0] / scale,
                             'r-.', label='Error Ellipse')
    tgt_xy,     = ax_xy.plot([x_tgt_full[0, K0] / scale], [x_tgt_full[1, K0] / scale],
                             's', ms=8, label='Target')
    ax_xy.set_xlabel('x [km]')
    ax_xy.set_ylabel('y [km]')
    ax_xy.legend(loc='lower left')
    ax_xy.grid(True)

    # Panel 4: zoomed x-y
    tail_zoom, = ax_zoom.plot(tail_x0, tail_y0, 'b-o', ms=3, label='Estimated Position')
    ell_zoom,  = ax_zoom.plot(x_ell_est[0, :, K0] / scale, x_ell_est[1, :, K0] / scale,
                              'r-.')
    tgt_zoom,  = ax_zoom.plot([x_tgt_full[0, K0] / scale], [x_tgt_full[1, K0] / scale],
                              's', ms=8)
    ax_zoom.set_xlabel('x [km]')
    ax_zoom.set_ylabel('y [km]')
    ax_zoom.legend(loc='lower left')
    ax_zoom.grid(True)
    ax_zoom.set_xlim(x_tgt_full[0, K0] / scale + offset / scale * np.array([-1, 1]))
    ax_zoom.set_ylim(x_tgt_full[1, K0] / scale + offset / scale * np.array([-1, 1]))

    plt.tight_layout()

    # ===  Animation ===
    def _update(frame):
        K     = num_tail + frame
        start = max(0, K - num_tail + 1)

        # Panel 1: current-time measurement dots
        offsets = np.column_stack([np.full(num_msmt, t_vec[K]), zeta[:, K]])
        marker_msmt.set_offsets(offsets)

        # Panel 2: RMSE marker
        marker_err.set_offsets([[t_vec[K], rmse_est[K]]])
        marker_cep.set_offsets([[t_vec[K], rmse_cov_est[K]]])

        # Panel 3 & 4: tail, ellipse, target
        tx = x_ekf_est[0, start:K + 1] / scale
        ty = x_ekf_est[1, start:K + 1] / scale
        tail_xy.set_data(tx, ty)
        tail_zoom.set_data(tx, ty)

        ell_xy.set_data(x_ell_est[0, :, K] / scale, x_ell_est[1, :, K] / scale)
        ell_zoom.set_data(x_ell_est[0, :, K] / scale, x_ell_est[1, :, K] / scale)

        tgt_x = x_tgt_full[0, K] / scale
        tgt_y = x_tgt_full[1, K] / scale
        tgt_xy.set_data([tgt_x], [tgt_y])
        tgt_zoom.set_data([tgt_x], [tgt_y])

        ax_zoom.set_xlim(tgt_x + 2 * offset / scale * np.array([-1, 1]))
        ax_zoom.set_ylim(tgt_y + 2 * offset / scale * np.array([-1, 1]))

        return [marker_msmt, marker_err, marker_cep,
                tail_xy, tail_zoom, ell_xy, ell_zoom, tgt_xy, tgt_zoom]

    anim = FuncAnimation(fig, _update, frames=num_time - num_tail,
                         interval=50, blit=False)
    fig._anim = anim   # prevent garbage collection
    fig.suptitle('Example 8.1: TDOA EKF Tracker (animated)')

    if save_path is not None:
        from matplotlib.animation import PillowWriter
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        print(f'Saving animation to {save_path} ...')
        anim.save(str(save_path), writer=PillowWriter(fps=20), dpi=100)
        print('done.')

    return [fig]


def example2_vid(save_path=_FIGS_DIR / 'example2_vid.gif', rng=np.random.default_rng()):
    """
    Animated version of Example 8.2.

    Runs the same AOA EKF as example2() using the vid parameters
    (t_inc=15 s, sigma_theta=0.2 deg), then produces a 2×2 animated figure:

      Panel 1 (top-left):  AOA measurements (az + el in degrees) vs time.
      Panel 2 (top-right): Position RMSE and predicted CEP50 vs time (semilogy).
      Panel 3 (bot-left):  Full x-y map — moving sensor, LOB uncertainty wedge,
                           error ellipse, and EKF tail.
      Panel 4 (bot-right): Zoomed x-y view that follows the target.

    Returns a single figure.  The animation object is stored on ``fig._anim``.

    :return: [fig]
    """
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Polygon as MplPolygon
    from ewgeo.triang.model import draw_lob as _draw_lob

    # ===  Sensor / target setup (MATLAB vid8_2 parameters) ===
    t_inc    = 15
    t_max    = 15 * 60
    t_vec    = np.arange(0, t_max + t_inc, t_inc)
    num_time = len(t_vec)

    x_aoa_full = (np.array([0., 0., 10e3])[:, np.newaxis]
                  + np.array([0., 100., 0.])[:, np.newaxis] * t_vec)

    ship_accel_power = 0.05
    a_tgt = np.concatenate(
        (-ship_accel_power + 2 * ship_accel_power * rng.uniform(size=(2, num_time)),
         np.zeros((1, num_time))), axis=0)
    v_tgt_full = np.array([-20., 0., 0.])[:, np.newaxis] + np.cumsum(a_tgt * t_inc, axis=1)
    x_tgt_full = np.array([50e3, 50e3, 0.])[:, np.newaxis] + np.cumsum(v_tgt_full * t_inc, axis=1)

    sigma_theta = 0.2                  # deg — match MATLAB vid8_2
    sigma_psi   = sigma_theta * _deg2rad
    num_msmt    = 2                    # az + el
    cov_psi     = CovarianceMatrix(sigma_psi ** 2 * np.eye(num_msmt))
    aoa         = DirectionFinder(x_aoa_full[:, 0], cov=cov_psi, do_2d_aoa=True)

    z = np.column_stack([
        aoa.measurement(x_source=x_tgt_full[:, i], x_sensor=x_aoa_full[:, i])
        for i in range(num_time)
    ])
    zeta = z + sigma_psi * rng.standard_normal((num_msmt, num_time))

    sigma_a      = 0.05
    num_dims     = 3
    motion_model = tracker.MotionModel.make_motion_model('cv', num_dims, sigma_a ** 2)
    state_space  = motion_model.state_space
    pos_slice    = state_space.pos_slice
    vel_slice    = state_space.vel_slice
    motion_model.update_time_step(t_inc)
    f = motion_model.f
    q = motion_model.q

    x_aoa_init = x_aoa_full[:, :3]
    zeta_init  = np.ravel(zeta[:, :3])
    cov_init   = CovarianceMatrix.block_diagonal(*[cov_psi] * 3)
    aoa_init   = DirectionFinder(x=x_aoa_init, cov=cov_init, do_2d_aoa=True)
    bnd, bnd_grad = fixed_alt(0, 'flat')
    x_init, _  = aoa_init.gradient_descent(zeta=zeta_init, x_init=np.array([10e3, 10e3, 0.]),
                                            eq_constraints=[bnd], epsilon=100)
    p_init = aoa_init.compute_crlb(x_source=x_init, eq_constraints_grad=[bnd_grad]).cov

    _x0 = np.zeros(state_space.num_states)
    _p0 = np.zeros((state_space.num_states, state_space.num_states))
    _x0[pos_slice] = x_init[:num_dims]
    _p0[pos_slice, pos_slice] = p_init[:num_dims, :num_dims]
    max_vel = 30
    _p0_vel = max_vel ** 2 * np.eye(num_dims)
    _p0_vel[-1, :] = 0;  _p0_vel[:, -1] = 0
    _p0[vel_slice, vel_slice] = _p0_vel
    x_pred = tracker.State(state_space=state_space, time=t_vec[0],
                           state=_x0, covar=CovarianceMatrix(_p0))

    msmt_model = tracker.MeasurementModel(pss=aoa, state_space=state_space)

    # ===  EKF loop — store estimates, errors, ellipses, and LOB fill polygons ===
    num_ell_pts  = 101
    x_ekf_est    = np.zeros((num_dims, num_time))
    rmse_est     = np.zeros(num_time)
    rmse_cov_est = np.zeros(num_time)
    x_ell_est    = np.zeros((2, num_ell_pts, num_time))
    lob_fills    = []          # list of (2, 4) polygon vertex arrays

    def _zero_alt_rows(st):
        st.state[pos_slice][-1] = 0.;  st.state[vel_slice][-1] = 0.
        for sl in (pos_slice, vel_slice):
            st.covar.cov[sl, sl][-1, :] = 0.
            st.covar.cov[sl, sl][:, -1] = 0.

    for idx in range(num_time):
        this_zeta = zeta[:, idx]
        aoa.pos   = x_aoa_full[:num_dims, idx]

        x_est  = tracker.ekf_update(x_pred, this_zeta, aoa.cov,
                                     msmt_model.measurement, msmt_model.jacobian)
        _zero_alt_rows(x_est)
        x_pred = tracker.kf_predict(x_est, q, f)
        _zero_alt_rows(x_pred)

        x_ekf_est[:, idx] = x_est.position
        rmse_est[idx]     = np.linalg.norm(x_est.position - x_tgt_full[:num_dims, idx])
        rmse_cov_est[idx] = x_est.position_covar.rmse

        p_xy = CovarianceMatrix(x_est.position_covar.cov[:2, :2])
        x_ell_est[:, :, idx] = draw_error_ellipse(x_est.position[:2], p_xy, num_ell_pts)

        # LOB uncertainty wedge at this time step (azimuth channel only)
        az      = this_zeta[0]
        s2d     = x_aoa_full[:2, idx]
        t2d     = x_tgt_full[:2, idx]
        lob_p   = _draw_lob(s2d, az + sigma_psi, x_source=t2d, scale=5)[:, :, 0]   # (2,2)
        lob_m   = _draw_lob(s2d, az - sigma_psi, x_source=t2d, scale=5)[:, :, 0]
        # polygon: sensor → minus_end → plus_end → sensor (closed triangle/wedge)
        poly_pts = np.hstack([lob_m, lob_p[:, [1]], lob_m[:, [0]]])   # (2, 4)
        lob_fills.append(poly_pts)

    # ===  Figure setup ===
    num_tail  = 10
    offset    = 10e3
    scale_km  = 1e3
    colors    = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fill_color = colors[5] if len(colors) > 5 else 'tab:purple'

    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    ax_msmt, ax_err, ax_xy, ax_zoom = axs.flatten()

    # Panel 1: AOA measurements (degrees)
    labels = ['Azimuth', 'Elevation']
    for ch in range(num_msmt):
        ax_msmt.plot(t_vec, np.degrees(zeta[ch]), label=labels[ch])
    marker_msmt = ax_msmt.scatter([t_vec[num_tail]] * num_msmt,
                                   np.degrees(zeta[:, num_tail]), c='k', zorder=5, s=30)
    ax_msmt.set_xlabel('Time [s]')
    ax_msmt.set_ylabel('AOA [deg]')
    ax_msmt.legend(loc='upper right')
    ax_msmt.grid(True)

    # Panel 2: RMSE + CEP50
    ax_err.semilogy(t_vec, rmse_est,     label='Measured')
    ax_err.semilogy(t_vec, rmse_cov_est, label='Predicted (CEP50)')
    marker_err = ax_err.scatter([t_vec[num_tail]], [rmse_est[num_tail]],     c='k', zorder=5, s=30)
    marker_cep = ax_err.scatter([t_vec[num_tail]], [rmse_cov_est[num_tail]], c='k', zorder=5, s=30)
    ax_err.set_xlabel('Time [s]')
    ax_err.set_ylabel('Error [m]')
    ax_err.legend(loc='upper right')
    ax_err.grid(True)

    # Panel 3: full x-y map
    K0 = num_tail
    ax_xy.plot(x_tgt_full[0] / scale_km, x_tgt_full[1] / scale_km, 'k-', alpha=0.2)
    ax_xy.set_xlim([-5, 55]);  ax_xy.set_ylim([-5, 105])

    poly0 = lob_fills[K0] / scale_km
    fill_patch = MplPolygon(poly0.T, closed=True, facecolor=fill_color,
                             edgecolor='none', alpha=0.3, label='LOB uncertainty')
    ax_xy.add_patch(fill_patch)

    sensor_dot, = ax_xy.plot([x_aoa_full[0, K0] / scale_km], [x_aoa_full[1, K0] / scale_km],
                              '^', ms=10, label='Sensor')
    tail_xy,    = ax_xy.plot(x_ekf_est[0, :K0 + 1] / scale_km,
                              x_ekf_est[1, :K0 + 1] / scale_km, 'b-o', ms=3,
                              label='Estimated Position')
    ell_xy,     = ax_xy.plot(x_ell_est[0, :, K0] / scale_km,
                              x_ell_est[1, :, K0] / scale_km, 'r-.', label='Error Ellipse')
    tgt_xy,     = ax_xy.plot([x_tgt_full[0, K0] / scale_km], [x_tgt_full[1, K0] / scale_km],
                              's', ms=8, label='Target')
    ax_xy.set_xlabel('x [km]');  ax_xy.set_ylabel('y [km]')
    ax_xy.legend(loc='lower right');  ax_xy.grid(True)

    # Panel 4: zoomed x-y
    fill_patch_zoom = MplPolygon(poly0.T, closed=True, facecolor=fill_color,
                                  edgecolor='none', alpha=0.3)
    ax_zoom.add_patch(fill_patch_zoom)

    tail_zoom,  = ax_zoom.plot(x_ekf_est[0, :K0 + 1] / scale_km,
                                x_ekf_est[1, :K0 + 1] / scale_km, 'b-o', ms=3)
    ell_zoom,   = ax_zoom.plot(x_ell_est[0, :, K0] / scale_km,
                                x_ell_est[1, :, K0] / scale_km, 'r-.')
    tgt_zoom,   = ax_zoom.plot([x_tgt_full[0, K0] / scale_km], [x_tgt_full[1, K0] / scale_km],
                                's', ms=8)
    ax_zoom.set_xlabel('x [km]');  ax_zoom.set_ylabel('y [km]')
    ax_zoom.grid(True)
    ax_zoom.set_xlim(x_tgt_full[0, K0] / scale_km + offset / scale_km * np.array([-1, 1]))
    ax_zoom.set_ylim(x_tgt_full[1, K0] / scale_km + offset / scale_km * np.array([-1, 1]))

    plt.tight_layout()

    # ===  Animation ===
    def _update(frame):
        K     = num_tail + frame
        start = max(0, K - num_tail + 1)

        marker_msmt.set_offsets(
            np.column_stack([np.full(num_msmt, t_vec[K]), np.degrees(zeta[:, K])]))
        marker_err.set_offsets([[t_vec[K], rmse_est[K]]])
        marker_cep.set_offsets([[t_vec[K], rmse_cov_est[K]]])

        tx = x_ekf_est[0, start:K + 1] / scale_km
        ty = x_ekf_est[1, start:K + 1] / scale_km
        tail_xy.set_data(tx, ty)
        tail_zoom.set_data(tx, ty)

        ell_xy.set_data(x_ell_est[0, :, K] / scale_km, x_ell_est[1, :, K] / scale_km)
        ell_zoom.set_data(x_ell_est[0, :, K] / scale_km, x_ell_est[1, :, K] / scale_km)

        tgt_x = x_tgt_full[0, K] / scale_km
        tgt_y = x_tgt_full[1, K] / scale_km
        tgt_xy.set_data([tgt_x], [tgt_y])
        tgt_zoom.set_data([tgt_x], [tgt_y])

        sensor_dot.set_data([x_aoa_full[0, K] / scale_km], [x_aoa_full[1, K] / scale_km])

        poly_k = lob_fills[K] / scale_km
        fill_patch.set_xy(poly_k.T)
        fill_patch_zoom.set_xy(poly_k.T)

        ax_zoom.set_xlim(tgt_x + 2 * offset / scale_km * np.array([-1, 1]))
        ax_zoom.set_ylim(tgt_y + 2 * offset / scale_km * np.array([-1, 1]))

        return [marker_msmt, marker_err, marker_cep,
                tail_xy, tail_zoom, ell_xy, ell_zoom,
                tgt_xy, tgt_zoom, sensor_dot,
                fill_patch, fill_patch_zoom]

    anim = FuncAnimation(fig, _update, frames=num_time - num_tail,
                         interval=100, blit=False)
    fig._anim = anim
    fig.suptitle('Example 8.2: AOA EKF Tracker (animated)')

    if save_path is not None:
        from matplotlib.animation import PillowWriter
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        print(f'Saving animation to {save_path} ...')
        anim.save(str(save_path), writer=PillowWriter(fps=20), dpi=100)
        print('done.')

    return [fig]


if __name__ == '__main__':
    run_all_examples()
    plt.show()
