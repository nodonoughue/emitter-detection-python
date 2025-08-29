import numpy as np
import matplotlib.pyplot as plt
import time
import utils
from utils.covariance import CovarianceMatrix
from utils import tracker
from triang import DirectionFinder
from tdoa import TDOAPassiveSurveillanceSystem

_ft2m = utils.unit_conversions.convert(1, from_unit="ft", to_unit="m")
_rad2deg = utils.unit_conversions.convert(1, from_unit="rad", to_unit="deg")
_deg2rad = utils.unit_conversions.convert(1, from_unit="deg", to_unit="rad")
_speed_of_light = utils.constants.speed_of_light

def run_all_examples():
    """
    Run all chapter 8 examples and return a list of figure handles

    :return figs: list of figure handles
    """

    return list(example1()) + list(example2())


def example1(rng=np.random.default_rng()):
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
    num_dims, n_tdoa = utils.safe_2d_shape(x_tdoa)

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
    plt.scatter(x_tdoa[0], x_tdoa[1], marker='o', label='Sensors')
    plt.plot(x_tgt_full[0],x_tgt_full[1], marker='v', markevery=[-1], label='Aircraft')
    plt.grid(True)

    # ===  Measurement Statistics
    ref_idx = 0
    sigma_toa = 10e-9
    cov_toa = (sigma_toa**2) * np.eye(n_tdoa)
    cov_roa = CovarianceMatrix(_speed_of_light**2*cov_toa)
    tdoa = TDOAPassiveSurveillanceSystem(x=x_tdoa, cov=cov_roa, ref_idx=ref_idx, variance_is_toa=False)

    # ===  Generate Measurements
    z = tdoa.measurement(x_tgt_full)
    noise = tdoa.cov.lower @ rng.standard_normal((tdoa.num_measurements, num_time))
    zeta = z + noise

    # crlb = tdoa.computeCRLB(x_tdoa, x_tgt_full, C_roa, ref_idx, false, true)
    # rmse_crlb = sqrt(arrayfun(@(i) trace(crlb(:,:,i)), 1:size(crlb,3)))

    # ===  Set Up Tracker
    sigma_a = 1

    f_fun, q_fun, state_space = tracker.make_kinematic_model('cv',num_dims,sigma_a**2)
    num_states = state_space['num_states']
    pos_slice = state_space['pos_slice']
    vel_slice = state_space['vel_slice']
    f = f_fun(t_inc) # generate state transition matrix
    q = q_fun(t_inc) # generate process noise covariance matrix

    z_fun, h_fun = tracker.make_measurement_model(pss=tdoa, state_space=state_space)
    # msmt function and linearized msmt function

    # ===  Initialize Track State
    x_pred = np.zeros((num_states,))
    p_pred = np.zeros((num_states, num_states))

    # Initialize position with TDOA estimate from first measurement
    x_init = np.array([0, 50e3, 5e3])
    epsilon = 100
    x_pred[pos_slice], _ = tdoa.gradient_descent(zeta=zeta[:, 0], x_init=x_init, epsilon=epsilon)
    p_pred[pos_slice, pos_slice] = 10*tdoa.compute_crlb(x_source=x_pred[pos_slice])

    # Bound initial velocity uncertainty by assumed max velocity of 340 m/s
    # (Mach 1 at sea level)
    max_vel = 340
    p_pred[vel_slice, vel_slice] = 10*max_vel**2*np.eye(num_dims)

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
    for idx in np.arange(num_time):
        utils.print_progress(num_time, idx, iterations_per_marker, iterations_per_row, t_start)

        # Grab Current Measurement
        this_zeta = zeta[:, idx]

        # Update Position Estimate
        # Previous prediction stored in x_pred, P_pred
        # Updated estimate will be stored in x_est, P_est
        x_est, p_est = tracker.ekf_update(x_pred, p_pred, this_zeta, tdoa.cov.cov, z_fun, h_fun)

        # Predict state to the next time step
        x_pred, p_pred = tracker.kf_predict(x_est, p_est, q, f)

        # Output the current prediction/estimation state
        x_ekf_est[:,idx] = x_est[pos_slice]
        x_ekf_pred[:,idx] = x_pred[pos_slice]

        rmse_cov_est[idx] = np.sqrt(np.linalg.trace(p_est[pos_slice,pos_slice]))
        rmse_cov_pred[idx] = np.sqrt(np.linalg.trace(p_pred[pos_slice,pos_slice]))

        # Draw an error ellipse
        x_est_xyz = x_est[pos_slice]
        p_est_xyz = p_est[pos_slice, pos_slice]
        x_est_xy = x_est_xyz[:2]
        p_est_xy = p_est_xyz[:2, :2]
        x_ell_est[:, :, idx] = utils.errors.draw_error_ellipse(x_est_xy, p_est_xy, num_ell_pts)

    print('done')
    t_elapsed = time.perf_counter() - t_start
    utils.print_elapsed(t_elapsed)

    plt.scatter(x_init[0], x_init[1], marker='+',label='Initial Position Estimate')
    plt.plot(x_ekf_est[0],x_ekf_est[1],'--',label='EKF (est.)')
    plt.plot(x_ekf_pred[0],x_ekf_pred[1],'--',label='EKF (pred.)')
    plt.grid(True)
    plt.legend(loc='lower left')

    # plt.plot some error ellipses
    #for idx=1:10:num_time
    #    this_ell = squeeze(x_ell_est(:,:,idx))
    #    hdl = patch(this_ell[0], this_ell[1],hdl_est.Color,'FaceAlpha',.2,label='1$\sigma$ Error (est.)')
    #    if idx~=1
    #        utils.excludeFromLegend(hdl)
    #    end
    #end

    # ===  Compute Error
    err_pred = x_ekf_pred[:, :-2] - x_tgt_full[:, 1:-1]
    err_est = x_ekf_est - x_tgt_full

    rmse_pred = np.sqrt(np.sum(np.fabs(err_pred)**2, axis=0))
    rmse_est = np.sqrt(np.sum(np.fabs(err_est)**2,axis=0))

    fig2=plt.figure()
    plt.plot(t_vec,rmse_cov_est, label='RMSE (est. cov.)')
    plt.plot(t_vec[2:], rmse_cov_pred[:-2], label='RMSE (pred. cov)')
    plt.plot(t_vec,rmse_est, '--', label='RMSE (est. act.)')
    plt.plot(t_vec[2:], rmse_pred, '--',label='RMSE (pred. act.)')
    plt.grid(True)
    plt.xlabel('Time [sec]')
    plt.ylabel('Error [m]')
    plt.yscale('log')
    plt.legend(loc='upper right')

    # ===  Repeat for Statistical Certainty
    num_mc = 1000
    sse_pred = np.zeros((num_mc, num_time-1))
    sse_est = np.zeros((num_mc, num_time))
    sse_cov_pred = np.zeros((num_mc, num_time))
    sse_cov_est = np.zeros((num_mc, num_time))
    print('Repeating tracker test for {:d} Monte Carlo trials...'.format(num_mc))
    t_start = time.perf_counter()
    iterations_per_marker = 1
    markers_per_row = 40
    iterations_per_row = markers_per_row * iterations_per_marker

    for idx_mc in np.arange(num_mc):
        utils.print_progress(num_mc, idx_mc, iterations_per_marker, iterations_per_row, t_start)

        # Generate Measurements
        noise = tdoa.cov.lower @ rng.standard_normal((tdoa.num_measurements, num_time))
        zeta = z + noise

        # Initialize Track State
        x_pred = np.zeros((num_states,))
        p_pred = np.zeros((num_states, num_states))

        # Initialize position with TDOA estimate from first measurement
        x_init = np.array([0, 50e3, 5e3])
        epsilon = 100
        x_pred[pos_slice], _ = tdoa.gradient_descent(zeta[:, 0], x_init=x_init,epsilon=epsilon)
        p_pred[pos_slice, pos_slice] = tdoa.compute_crlb(x_source=x_pred[pos_slice])

        # Bound initial velocity uncertainty by assumed max velocity of 340 m/s
        # (Mach 1 at sea level)
        p_pred[vel_slice, vel_slice] = max_vel**2*np.eye(num_dims)

        # Step Through Time
        x_ekf_est = np.zeros((num_dims,num_time))
        x_ekf_pred = np.zeros((num_dims, num_time))
        for idx in np.arange(num_time):
            # Grab Current Measurement
            this_zeta = zeta[:, idx]

            # Update Position Estimate
            # Previous prediction stored in x_pred, P_pred
            # Updated estimate will be stored in x_est, P_est
            x_est, p_est = tracker.ekf_update(x_pred, p_pred, this_zeta, tdoa.cov.cov, z_fun, h_fun)

            # Predict state to the next time step
            x_pred, p_pred = tracker.kf_predict(x_est, p_est, q, f)

            # Output the current prediction/estimation state
            x_ekf_est[:, idx] = x_est[pos_slice]
            x_ekf_pred[:, idx] = x_pred[pos_slice]

            sse_cov_est[idx_mc, idx] = np.linalg.trace(p_est[pos_slice, pos_slice])
            sse_cov_pred[idx_mc, idx] = np.linalg.trace(p_pred[pos_slice, pos_slice])

        err_pred = x_ekf_pred[:, :-1] - x_tgt_full[:, 1:]
        err_est = x_ekf_est - x_tgt_full

        sse_pred[idx_mc,:] = np.sum(np.fabs(err_pred)**2, axis=0)
        sse_est[idx_mc,:] = np.sum(np.fabs(err_est)**2, axis=0)

    print('done')
    t_elapsed = time.perf_counter() - t_start
    utils.print_elapsed(t_elapsed)

    rmse_pred = np.sqrt(np.mean(sse_pred, axis=0))
    rmse_est = np.sqrt(np.mean(sse_est, axis=0))

    rmse_cov_est = np.sqrt(np.mean(sse_cov_est, axis=0))
    rmse_cov_pred = np.sqrt(np.mean(sse_cov_pred, axis=0))

    fig3=plt.figure()
    hdl_est = plt.plot(t_vec, rmse_cov_est, label='RMSE (est. cov.)')
    hdl_pred = plt.plot(t_vec[1:], rmse_cov_pred[:-1], label='RMSE (pred. cov)')
    plt.plot(t_vec, rmse_est, '--', label='RMSE (est. act.)', color=hdl_est[0].get_color())
    plt.plot(t_vec[1:], rmse_pred, '--', label='RMSE (pred. act.)', color=hdl_pred[0].get_color())
    plt.grid(True)
    plt.xlabel('Time [sec]')
    plt.ylabel('Error [m]')
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
    plt.plot(x_aoa_full[0], x_aoa_full[1], '-^', markevery=beam_plot_times, label='AOA Sensor Trajectory')
    plt.plot(x_tgt_full[0], x_tgt_full[1], '-<', markevery=beam_plot_times, label='Target Trajectory')
    plt.grid(True)
    plt.legend(loc='upper right')

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
        fill_patch = plt.Polygon(lob_fill.T, linestyle='--', edgecolor='k', facecolor=fill_color, alpha=.3,
                                 label=label_fill)
        fig1.gca().add_patch(fill_patch)
        label_fill = None

    plt.xlim([-5e3,55e3])
    plt.ylim([-5e3,105e3])

    # ===  Generate Measurements
    do2daoa = aoa.do_2d_aoa
    num_msmt = aoa.num_measurements
    z = np.array([aoa.measurement(x_source=this_x_source.T, x_sensor=this_x_sensor.T) 
                  for this_x_source, this_x_sensor in zip(x_tgt_full.T, x_aoa_full.T)]).T
    noise = sigma_psi * rng.standard_normal((num_msmt, num_time))
    zeta = z + noise

    # ===  Set Up Tracker
    # Track in x/y, even though a/c and tgt in x/y/z
    sigma_a = .05
    num_dims = 3 # number of dimensions to use in state

    f_fun, q_fun, state_space = tracker.make_kinematic_model('cv',num_dims,sigma_a**2)
    num_states = state_space['num_states']
    pos_slice = state_space['pos_slice']
    vel_slice = state_space['vel_slice']
    f = f_fun(t_inc) # generate state transition matrix
    q = q_fun(t_inc) # generate process noise covariance matrix

    # ===  Initialize Track State
    x_pred = np.zeros((num_states,))
    p_pred = np.zeros((num_states, num_states))

    # Initialize position with AOA estimate from first three measurement
    x_aoa_init = x_aoa_full[:, :3]
    zeta_init = np.ravel(zeta[:,:3])
    cov_init = CovarianceMatrix.block_diagonal(*[cov_psi]*3)

    aoa_init = DirectionFinder(x=x_aoa_init, cov=cov_init, do_2d_aoa=do2daoa)
    
    x_init_guess = np.array([10e3, 10e3, 0])
    bnd, bnd_grad = utils.constraints.fixed_alt(0, 'flat')
    x_init, _ = aoa_init.gradient_descent(zeta=zeta_init, x_init=x_init_guess, eq_constraints=[bnd], epsilon=100)
    p_init = aoa_init.compute_crlb(x_source=x_init, eq_constraints_grad=[bnd_grad])

    x_pred[pos_slice] = x_init[:num_dims]
    p_pred[pos_slice, pos_slice] = p_init[:num_dims, :num_dims]

    # Bound initial velocity uncertainty by assumed max velocity of 10 m/s
    # (~20 knots)
    max_vel = 30
    p_vel = max_vel**2*np.eye(num_dims)
    p_vel[-1, :] = 0
    p_vel[:, -1] = 0
    p_pred[vel_slice, vel_slice] = p_vel

    # ===  Step Through Time
    print('Iterating through EKF tracker time steps...')
    markers_per_row = 40
    iter_per_marker = 10
    iter_per_row = markers_per_row * iter_per_marker

    # at least 1 iteration per marker, no more than 100 iterations per marker
    t_start = time.perf_counter()
    
    x_ekf_est = np.zeros((num_dims,num_time))
    x_ekf_pred = np.zeros((num_dims, num_time))
    rmse_cov_est = np.zeros((num_time,))
    rmse_cov_pred = np.zeros((num_time,))
    num_ell_pts = 101 # number of points for ellipse drawing
    x_ell_est = np.zeros((2,num_ell_pts,num_time))
    for idx in np.arange(num_time):
        utils.print_progress(num_total=num_time, curr_idx=idx, iterations_per_marker=iter_per_marker,
                             iterations_per_row=iter_per_row, t_start=t_start)

        # Grab Current Measurement
        this_zeta = zeta[:, idx]

        # Update msmt function
        this_x_aoa = x_aoa_full[:num_dims, idx]
        aoa.pos = this_x_aoa
        
        z_fun, h_fun = tracker.make_measurement_model(pss=aoa, state_space=state_space)

        # Update Position Estimate
        # Previous prediction stored in x_pred, P_pred
        # Updated estimate will be stored in x_est, P_est
        x_est, p_est = tracker.ekf_update(x_pred, p_pred, this_zeta, aoa.cov.cov, z_fun, h_fun)

        # Enforce known altitude
        pos_est = x_est[pos_slice]
        pos_est[-1] = 0
        x_est[pos_slice] = pos_est
        vel_est = x_est[vel_slice]
        vel_est[-1] = 0
        x_est[vel_slice] = vel_est
        p_pos = p_est[pos_slice, pos_slice]
        p_pos[-1, :] = 0
        p_pos[:, -1] = 0
        p_est[pos_slice, pos_slice] = p_pos
        p_vel = p_est[vel_slice, vel_slice]
        p_vel[-1, :] = 0
        p_vel[:, -1] = 0
        p_est[vel_slice, vel_slice] = p_vel

        # Predict state to the next time step
        x_pred, p_pred = tracker.kf_predict(x_est, p_est, q, f)

        # Enforce known altitude
        pos_pred = x_pred[pos_slice]
        pos_pred[-1] = 0
        x_pred[pos_slice] = pos_pred
        vel_pred = x_pred[vel_slice]
        vel_pred[-1] = 0
        x_pred[vel_slice] = vel_pred
        p_pos = p_pred[pos_slice, pos_slice]
        p_pos[-1, :] = 0
        p_pos[:, -1] = 0
        p_pred[pos_slice, pos_slice] = p_pos
        p_vel = p_pred[vel_slice, vel_slice]
        p_vel[-1, :] = 0
        p_vel[:, -1] = 0
        p_pred[vel_slice, vel_slice] = p_vel

        # Output the current prediction/estimation state
        pos_est = x_est[pos_slice]
        p_pos_est = p_est[pos_slice, pos_slice]
        x_ekf_est[:, idx] = pos_est
        x_ekf_pred[:,idx] = x_pred[pos_slice]

        rmse_cov_est[idx] = np.sqrt(np.linalg.trace(p_est[pos_slice, pos_slice]))
        rmse_cov_pred[idx]= np.sqrt(np.linalg.trace(p_pred[pos_slice, pos_slice]))

        # Draw an error ellipse
        x_ell_est[:, :, idx] = utils.errors.draw_error_ellipse(x=pos_est[:2],
                                                               covariance=p_pos_est[:2, :2],
                                                               num_pts=num_ell_pts)

    print('done')
    t_elapsed = time.perf_counter() - t_start
    utils.print_elapsed(t_elapsed)
        
    plt.plot(x_init[0], x_init[1], '+',label='Initial Position Estimate')
    plt.plot(x_ekf_est[0],x_ekf_est[1],'-',label='EKF (est.)')
    plt.plot(x_ekf_pred[0],x_ekf_pred[1],'-',label='EKF (pred.)')
    plt.grid(True)
    plt.legend()

    # plt.plot some error ellipses
    #for idx=1:10:num_time
    #    this_ell = squeeze(x_ell_est(:,:,idx))
    #    hdl = patch(this_ell[0], this_ell[1],hdl_est.Color,'FaceAlpha',.2,label='1$\sigma$ Error (est.)')
    #    if idx~=1
    #        utils.excludeFromLegend(hdl)
    #    end
    #end

    # ===  Zoomed plt.plot on Target
    fig2=plt.figure()
    plt.plot(x_tgt_full[0], x_tgt_full[1],'-<', markevery=[-1], label='Target Trajectory')
    plt.plot(x_init[0], x_init[1], '+',label='Initial Position Estimate')
    plt.plot(x_ekf_est[0],x_ekf_est[1],'-',label='EKF (est.)')
    plt.plot(x_ekf_pred[0],x_ekf_pred[1],'-',label='EKF (pred.)')
    plt.grid(True)
    plt.xlim([30e3,50e3])
    plt.legend(loc='lower left')

    ## Compute Error
    err_pred = x_ekf_pred[:, :-1] - x_tgt_full[:num_dims, 1:]
    err_est = x_ekf_est - x_tgt_full[:num_dims]

    rmse_pred = np.sqrt(np.sum(np.fabs(err_pred)**2, axis=0))
    rmse_est = np.sqrt(np.sum(np.fabs(err_est)**2, axis=0))

    fig3=plt.figure()
    plt.plot(t_vec,rmse_cov_est,label='RMSE (est. cov.)')

    plt.plot(t_vec[1:], rmse_cov_pred[:-1], label='RMSE (pred. cov)')
    # set(gca,'ColorOrderIndex',1)
    plt.plot(t_vec,rmse_est,'--',label='RMSE (est. act.)')
    plt.plot(t_vec[1:], rmse_pred, '--', label='RMSE (pred. act.)')
    plt.grid(True)
    plt.xlabel('Time [sec]')
    plt.ylabel('Error [m]')
    plt.legend(loc='upper right')
    plt.yscale('log')

    # ===  Return Figure Handles
    figs = [fig1, fig2, fig3]
    return figs


if __name__ == '__main__':
    run_all_examples()
    plt.show()
