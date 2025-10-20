import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
from scipy.linalg import block_diag
import time

from ewgeo.tdoa import TDOAPassiveSurveillanceSystem
from ewgeo.tracker import StateSpace, State, Track, Tracker
from ewgeo.tracker.association import NNAssociator, GNNAssociator, PDAAssociator
from ewgeo.tracker.deleter import MissedDetectionDeleter
from ewgeo.tracker.initiator import SinglePointMeasurementInitiator
from ewgeo.tracker.measurement import Measurement, MeasurementModel
from ewgeo.tracker.promoter import MofNPromoter
from ewgeo.tracker.transition import MotionModel, ConstantVelocityMotionModel, ConstantAccelerationMotionModel
from ewgeo.triang import DirectionFinder
from ewgeo.utils import safe_2d_shape, print_progress, print_elapsed
from ewgeo.utils.constants import speed_of_light
from ewgeo.utils.covariance import CovarianceMatrix
from ewgeo.utils.system import PassiveSurveillanceSystem
from ewgeo.utils.unit_conversions import convert

_ft2m = convert(1, from_unit="ft", to_unit="m")
_rad2deg = convert(1, from_unit="rad", to_unit="deg")
_deg2rad = convert(1, from_unit="deg", to_unit="rad")

def run_all_examples():
    """
    Run all chapter 9 examples and return a list of figure handles

    :return figs: list of figure handles
    """

    return list(example1()) + list(example2()) + list(example3()) + list(example4())


def __init_example1():
    """
    Initialize the tracks, measurement model, and kinematic model for example 1

    :return scenario: dictionary containing the initial states, kinematic model, measurement model, and tracks
    """
    # Define starting states
    ids = ['1', '2', '3']
    state_vecs = [[0, 2e3, 0, 100],
                  [1e3, 2e3, 80, -10],
                  [1e3, 1.3e3, 70, 50]]
    state_covars = [
        CovarianceMatrix(np.array([[1e4, 1e2, 0, 0],
                                   [1e2, 2e4, 0, 0],
                                   [0, 0, 2e2, 0],
                                   [0, 0, 0, 2e2]])),
        CovarianceMatrix(np.array([[4e4, -1e4, 0, 0],
                                   [-1e4, 1e4, 0, 0],
                                   [0, 0, 1e3, -5e2],
                                   [0, 0, -5e2, 5e2]])),
        CovarianceMatrix(np.array([[2e4, 1e4, 0, 0],
                                   [1e4, 4e4, 0, 0],
                                   [0, 0, 6e2, 4e2],
                                   [0, 0, 4e2, 6e2]]))]

    # Define the State Space; easiest method is to instantiate
    # a transition model
    process_covar = np.diag([25, 25])
    transition = ConstantVelocityMotionModel(num_dims=2, process_covar=process_covar)
    state_space = transition.state_space

    # Make the initial track states
    states = [State(state_space=state_space, time=0, state=s, covar=c) for (s, c) in zip(state_vecs, state_covars)]

    # Initialize a Track with each one
    tracks = [Track(track_id=i, initial_state=s) for (i, s) in zip(ids, states)]

    # Define Measurement System
    x_aoa = np.array([[750, 300],
                      [200, 800]])
    c_zeta = CovarianceMatrix(np.power(np.diag([3, 3])*_deg2rad,2))
    pss = DirectionFinder(x=x_aoa, cov=c_zeta, do_2d_aoa=False)

    # Define measurements
    zeta = [[1.811, 1.652],
             [1.253, 0.803],
             [1.140, 0.726],
             [1.679, 1.454]]
    new_time: float = 5
    measurements: list[Measurement] = [Measurement(time=new_time, sensor=pss, zeta=z) for z in zeta]

    return {'tracks': tracks,
            'transition': transition,
            'pss': pss,
            'measurements': measurements}

def example1():
    """
    Executes Example 9.1.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    10 October 2025

    :return: figure handle to generated graphic
    """

    # Initialize and unpack the example scenario
    scenario: dict = __init_example1()
    tracks: list[Track] = scenario['tracks']
    transition: MotionModel = scenario['transition']
    state_space: StateSpace = transition.state_space
    pss: PassiveSurveillanceSystem = scenario['pss']
    measurements: list[Measurement] = scenario['measurements']
    new_time = measurements[0].time

    # For plotting, let's generate some new states from these measurements
    x_source = [pss.least_square(m.zeta, x_init=np.array([500,1000])) for m in measurements]
    new_states = [State(state_space=state_space,
                        time=new_time,
                        state=np.concat((x[0], [0,0]), axis=0))
                  for x in x_source]
    coords = list(zip(*[s.position for s in new_states]))

    # Build a state covariance error from the CRLB
    for s in new_states:
        s.covar = CovarianceMatrix(block_diag(pss.compute_crlb(s.position),1e6*np.eye(2)))

    # Predict the states forward and replot
    predicted_states = [transition.predict(t.curr_state, new_time=new_time) for t in tracks]

    figs = []
    fig, ax = plt.subplots()
    figs.append(fig)
    plot_args = {'marker': 'v'}
    [t.plot(ax=ax, predicted_state=p, **plot_args) for (t, p) in zip(tracks, predicted_states)]
    pss.plot_sensors(ax=ax, label='DF System')
    [s.plot(ax=ax, color='k', do_pos=False, do_vel=False, do_cov=True, label=None) for s in new_states]
    plt.scatter(*coords, s=15, marker='^', color='k', label='New States (truth)')
    plt.title('Predicted Track States with new Measurements')
    plt.legend()
    plt.xlim([-500,2000])
    plt.ylim([-500,3000])

    # Set up the Nearest Neighbor Association Scheme
    associator = NNAssociator(motion_model=transition, gate_probability=.75)
    hypotheses, _ = associator.associate(tracks=tracks, measurements=measurements, print_table=True)

    # Make a plot for the hypothesis assignments
    fig, ax = plt.subplots()
    figs.append(fig)
    [h.update_track(ax=ax) for h in hypotheses.values()]
    plt.scatter(*coords, s=25, color='k', label='New States (truth)')
    plt.legend()
    plt.title('Predicted and Updated Track States after NN Association')

    fig, ax = plt.subplots()
    figs.append(fig)
    [t.plot(ax=ax, **plot_args) for t in tracks]
    pss.plot_sensors(ax=ax, label='DF System')
    plt.legend()
    plt.title('Updated Trackers after NN Association')
    plt.xlim([-500,2000])
    plt.ylim([-500,3000])

    return figs


def example2():
    """
    Executes Example 9.2.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    13 October 2025

    :return: figure handle to generated graphic
    """

    # Initialize and unpack the example scenario
    scenario = __init_example1()
    tracks = scenario['tracks']
    transition = scenario['transition']
    pss = scenario['pss']
    measurements = scenario['measurements']

    # Set up the Global Nearest Neighbor Association Scheme
    associator = GNNAssociator(motion_model=transition, gate_probability=.75)
    hypotheses, _ = associator.associate(tracks=tracks, measurements=measurements, print_table=True)
    [h.update_track() for h in hypotheses.values()]

    # Make a plot for the hypothesis assignments
    figs = []
    fig, ax = plt.subplots()
    figs.append(fig)
    [t.plot(ax=ax, marker='v') for t in tracks]
    pss.plot_sensors(ax=ax, label='DF System')
    plt.legend()
    plt.title('Updated Trackers after GNN Association')
    plt.xlim([-500, 2000])
    plt.ylim([-500, 3000])

    return figs

def example3():
    """
    Executes Example 9.3.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    13 October 2025

    :return: figure handle to generated graphic
    """

    # Initialize and unpack the example scenario
    scenario = __init_example1()
    tracks = scenario['tracks']
    transition = scenario['transition']
    pss = scenario['pss']
    measurements = scenario['measurements']

    # Set up the Global Nearest Neighbor Association Scheme
    associator = PDAAssociator(motion_model=transition, gate_probability=.75)
    hypotheses, _ = associator.associate(tracks=tracks, measurements=measurements, print_table=True)
    [h.update_track() for h in hypotheses.values()]

    # Make a plot for the hypothesis assignments
    figs = []
    fig, ax = plt.subplots()
    figs.append(fig)
    [t.plot(ax=ax, marker='v') for t in tracks]
    pss.plot_sensors(ax=ax, label='DF System')
    plt.legend()
    plt.title('Updated Trackers after PDAF Association')
    plt.xlim([-500, 2000])
    plt.ylim([-500, 3000])

    return figs

def _make_tgt_1(max_time: float = 600):
    # ===  Define target trajectory
    x_tgt_init = np.array([-50e3, 100e3, 20e3 * _ft2m])
    vel = 200

    t_e_leg = 3 * 60  # turn at 3 min
    turn_rad = 50e3
    t_turn = np.pi / 2 * turn_rad / vel
    t_s_leg = max_time

    t_inc = 10  # 10 seconds between track updates

    # Due East
    t_e_vec = np.arange(start=0, stop=t_e_leg + t_inc, step=t_inc)
    x_e_leg = x_tgt_init[:, np.newaxis] + np.array([vel, 0, 0])[:, np.newaxis] * t_e_vec[np.newaxis, :]

    # Turn to South
    t_turn_vec = np.arange(start=t_inc, step=t_inc, stop=t_turn)
    angle_turn = np.pi / 2 * t_turn_vec / t_turn
    x_turn = x_e_leg[:, [-1]] + turn_rad * np.array(
        [np.sin(angle_turn), np.cos(angle_turn) - 1, np.zeros_like(angle_turn)])

    # Due South
    t_s_vec = np.arange(start=t_inc, step=t_inc, stop=t_s_leg + t_inc)
    x_s_leg = x_turn[:, [-1]] + np.array([0, -vel, 0])[:, np.newaxis] * t_s_vec[np.newaxis, :]

    # Combine legs
    x_tgt_full = np.concatenate((x_e_leg, x_turn, x_s_leg), axis=1)
    t_vec = np.arange(x_tgt_full.shape[1]) * t_inc

    # Enforce max_time
    mask = t_vec <= max_time
    t_vec = t_vec[mask]
    x_tgt_full = x_tgt_full[:, mask]

    return t_vec, x_tgt_full

def _make_tgt_2(max_time: float = 600):
    # ===  Define target trajectory
    x_tgt_init = np.array([-50e3, 50e3, 20e3 * _ft2m])
    vel = 210

    t_ne_leg = 3 * 60  # turn at 3 min
    turn_rad = 50e3
    t_turn = np.pi * turn_rad / vel
    t_sw_leg = max_time

    t_inc = 10  # 10 seconds between track updates

    # Due North-East
    t_ne_vec = np.arange(start=0, stop=t_ne_leg + t_inc, step=t_inc)
    x_ne_leg = x_tgt_init[:, np.newaxis] + vel*np.array([np.sqrt(2)/2, np.sqrt(2)/2, 0])[:, np.newaxis] * t_ne_vec[np.newaxis, :]

    # Make a 180-degree turn, rotate and offset it to the correct starting position
    t_turn_vec = np.arange(start=t_inc, step=t_inc, stop=t_turn)
    angle_inc = np.pi / len(t_turn_vec)
    angle_turn = np.arange(start=3*np.pi/4, step=-angle_inc, stop=-np.pi/4-angle_inc)# start due south, end due north
    x_turn0 = np.array([np.cos(angle_turn), np.sin(angle_turn), np.zeros_like(angle_turn)])
    # scale the radius and add the offset
    x_turn = x_ne_leg[:, [-1]] + turn_rad * (x_turn0 - x_turn0[:, [0]])

    # Due South-West
    t_sw_vec = np.arange(start=t_inc, step=t_inc, stop=t_sw_leg + t_inc)
    x_sw_leg = x_turn[:, [-1]] - vel*np.array([np.sqrt(2)/2, np.sqrt(2)/2, 0])[:, np.newaxis] * t_sw_vec[np.newaxis, :]

    # Combine legs
    x_tgt_full = np.concatenate((x_ne_leg, x_turn, x_sw_leg), axis=1)
    t_vec = np.arange(x_tgt_full.shape[1]) * t_inc

    # Enforce max_time
    mask = t_vec <= max_time
    t_vec = t_vec[mask]
    x_tgt_full = x_tgt_full[:, mask]

    return t_vec, x_tgt_full

def _make_tgt_3(max_time: float = 600):
    # ===  Define target trajectory
    x_tgt_init = np.array([-50e3, 150e3, 20e3 * _ft2m])
    vel = 170

    t_se_leg = max_time  # turn at 3 min
    t_inc = 10  # 10 seconds between track updates

    # Due South-East
    t_se_vec = np.arange(start=0, stop=t_se_leg + t_inc, step=t_inc)
    x_se_leg = x_tgt_init[:, np.newaxis] + vel*np.array([np.sqrt(2)/2, -np.sqrt(2)/2, 0])[:, np.newaxis] * t_se_vec[np.newaxis, :]

    return t_se_vec, x_se_leg


def example4():
    # Make the targets
    max_time = 900 # seconds
    tgt_1 = _make_tgt_1(max_time)
    tgt_2 = _make_tgt_2(max_time)
    tgt_3 = _make_tgt_3(max_time)
    tgts = [tgt_1, tgt_2, tgt_3] # each one has a tuple of time, pos

    # Make figure handles
    fig, axs = plt.subplots(2, 2)

    # Plot the geometry
    scale=1000
    tgt_hdls = [axs[0, 0].plot(t[1][0]/scale, t[1][1]/scale, label=f"Target {idx}")[0] for idx, t in enumerate([tgt_1, tgt_2, tgt_3])]
    tgt_colors = [h.get_color() for h in tgt_hdls]

    # Initialize the PSS
    x_tdoa = np.array([[5e3, 0, 0, -5e3],
                       [0, 5e3, 0, 0],
                       [30, 60, 30, 60]])
    num_dims, n_tdoa = safe_2d_shape(x_tdoa)
    ref_idx = 0
    sigma_toa = 10e-9
    cov_toa = (sigma_toa ** 2) * np.eye(n_tdoa)
    cov_roa = CovarianceMatrix(speed_of_light ** 2 * cov_toa)
    tdoa = TDOAPassiveSurveillanceSystem(x=x_tdoa, cov=cov_roa, ref_idx=ref_idx, variance_is_toa=False)

    # Truth measurements
    for tgt, color in zip(tgts, tgt_colors):
        # tgt[0] is the (N, ) time vec
        # tgt[1] is the (2,N) position vec
        # color is the color of the corresponding truth position plot

        zeta = tdoa.measurement(x_source=tgt[1])
        axs[0,1].plot(tgt[0], zeta[0,:]/scale, color=color)
        axs[1,0].plot(tgt[0], zeta[1,:]/scale, color=color)
        axs[1,1].plot(tgt[0], zeta[2,:]/scale, color=color)

    # Add axis labels
    [ax.grid(True) for ax in axs.flatten()]
    axs[0,0].set_xlabel('East (km)', fontsize=8)
    axs[0,0].set_ylabel('North (km)', fontsize=8)
    axs[0,0].set_title('Target Trajectories')
    [ax.set_xlabel('Time (s)', fontsize=8) for ax in axs.flatten()[2:]]
    axs[0,1].set_ylabel('$\\tau_{0,1}$ (km)', fontsize=8)
    axs[0,1].set_title('TDOA for Sensors 0, 1')
    axs[1,0].set_ylabel('$\\tau_{0,2}$ (km)', fontsize=8)
    axs[1,0].set_title('TDOA for Sensors 0, 2')
    axs[1,0].set_ylabel('$\\tau_{0,3}$ (km)', fontsize=8)
    axs[1,1].set_title('TDOA for Sensors 0, 3')
    plt.tight_layout()

    # Initialize the Tracker
    # plot_dims = np.s_[:2] # x/y are the plot axes
    transition = ConstantAccelerationMotionModel(num_dims=3,process_covar=0.1**2)
    msmt_model = MeasurementModel(state_space=transition.state_space, pss=tdoa)
    tracker = Tracker(transition=transition, msmt_model=msmt_model,
                      initiator=SinglePointMeasurementInitiator(msmt_model=msmt_model),
                      associator=NNAssociator(motion_model=transition, gate_probability=.9),
                      deleter=MissedDetectionDeleter(num_missed_detections=3),
                      promoter=MofNPromoter(num_hits=3, num_chances=5),
                      do_plotting=False, keep_all_tracks=True)

    # Make truth state objects; we'll update their states over time
    truth_states = [State(state_space=transition.state_space, state=None, time=0, covar=None) for _ in tgts]

    # Run the tracker, one step at a time
    time_vec = tgts[0][0]
    truth_label = 'Noisy Truth Measurements'
    fa_label = 'False Alarm Measurements'
    num_fa_per_step = 15

    # Print progress
    iterations_per_marker = 1
    iterations_per_row = 40 * iterations_per_marker
    total_iterations = len(time_vec)
    t_start = time.perf_counter()
    print(f"Running tracker across {total_iterations} time steps...")

    for idx in range(len(time_vec)):
        print_progress(total_iterations, idx, iterations_per_marker, iterations_per_row, t_start)

        # Update the truth position of each target
        [setattr(s, 'position', x[1][:, idx]) for s, x in zip(truth_states, tgts)]

        # Generate Noisy Measurements
        truth_msmts = [msmt_model.measurement(s, noise=True) for s in truth_states]

        # Add the truth measurements to the plots
        axs[0, 1].scatter(time_vec[idx]*np.ones(len(truth_msmts)), [m.zeta[0]/scale for m in truth_msmts],
                          marker = 'v', color = 'b', alpha=0.5, label = truth_label)
        axs[1, 0].scatter(time_vec[idx]*np.ones(len(truth_msmts)), [m.zeta[1]/scale for m in truth_msmts],
                          marker = 'v', color = 'b', alpha=0.5, label = truth_label)
        axs[1, 1].scatter(time_vec[idx]*np.ones(len(truth_msmts)), [m.zeta[2]/scale for m in truth_msmts],
                          marker = 'v', color = 'b', alpha=0.5, label = truth_label)

        # Generate false alarm measurements
        fa_msmt = msmt_model.false_alarm(max_val=4.5e3, num=num_fa_per_step, time=time_vec[idx].item())

        # Add the false alarms to the plots
        axs[0, 1].scatter(time_vec[idx]*np.ones(num_fa_per_step),
                          [m.zeta[0]/scale for m in fa_msmt], marker = '^', color = 'gray', alpha=0.1, label = fa_label)
        axs[1, 0].scatter(time_vec[idx]*np.ones(num_fa_per_step),
                          [m.zeta[1]/scale for m in fa_msmt], marker = '^', color = 'gray', alpha=0.1, label = fa_label)
        axs[1, 1].scatter(time_vec[idx]*np.ones(num_fa_per_step),
                          [m.zeta[2]/scale for m in fa_msmt], marker = '^', color = 'gray', alpha=0.1, label = fa_label)

        # Feed them to the tracker; shuffle the measurements
        measurements = truth_msmts[:] + fa_msmt[:]
        shuffle(measurements) # shuffle is done in place
        tracker.update(measurements=measurements)

        truth_label = None
        fa_label = None

    # Print all the tracks, including tentative ones
    trk_label = 'Firm Tracks'
    for t in tracker.all_tracks():
        axs[0,0].plot(t, do_cov=False, do_vel=False, linestyle='--', label=trk_label)
        trk_label = None

    trk_label = 'Tentative Tracks'
    for t in tracker.all_tentative_tracks():
        axs[0,0].plot(t, do_cov=False, do_vel=False, linestyle=':', linewidth=0.25, label=trk_label)
        trk_label = None

    print('done.')
    print_elapsed(time.perf_counter()-t_start)

    [ax.legend() for ax in axs.flatten()]

    return [fig]

if __name__ == '__main__':
    run_all_examples()
    plt.show()
