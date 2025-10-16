"""
Draw Figures - Chapter 9

This script generates all the figures that appear in Chapter 9 of the textbook.

Ported from MATLAB Code

Nicholas O'Donoughue
13 October 2025
"""

import matplotlib.pyplot as plt
import numpy as np

from ewgeo.tdoa import TDOAPassiveSurveillanceSystem
from ewgeo.tracker import State, Track
from ewgeo.tracker.association import Hypothesis, MissedDetectionHypothesis, NNAssociator
from ewgeo.tracker.measurement import MeasurementModel
from ewgeo.tracker.transition import ConstantVelocityMotionModel
from ewgeo.triang import DirectionFinder
from ewgeo.utils import init_output_dir, init_plot_style
from ewgeo.utils.covariance import CovarianceMatrix
from ewgeo.utils.errors import draw_error_ellipse
from ewgeo.utils.unit_conversions import convert

from examples.practical_geo import chapter9

_deg2rad = convert(1, from_unit="deg", to_unit="rad")

def make_all_figures(close_figs=False):
    """
    Call all the figure generators for this chapter

    :param close_figs: Boolean flag.  If true, will close all figures after generating them; for batch scripting.
                       Default=False
    :return: List of figure handles
    """

    # Find the output directory
    prefix = init_output_dir('practical_geo/chapter9')
    init_plot_style()

    # Generate all figures
    figs = []
    figs.extend(make_figure_2(prefix))
    figs.extend(make_figure_3(prefix))
    figs.extend(make_figure_4_5(prefix))
    figs.extend(make_figure_6(prefix))
    figs.extend(make_figure_7(prefix))
    figs.extend(make_figure_8(prefix))

    if close_figs:
        [plt.close(fig) for fig in figs]
        return None
    else:
        # Display the plots
        plt.show()

    return figs


def make_figure_2(prefix=None):
    """
    Figure 9.2

    :param prefix: output directory to place generated figure
    :return: handle
    """

    print('Generating Figure 9.2...')

    # Plot a track, its prediction and error covariance, and a set of four measurements nearby

    # Initialize the states for the track
    transition = ConstantVelocityMotionModel(num_dims=2, process_covar=np.eye(2)*100)
    state_vecs = np.array([[0.0, 0.0, 1.0, 1.0],
                           [0.5, 0.5, 1.0, 0.0],
                           [1.0, 0.5, 0.5, 1.0],
                           [1.25, 1.0, 0.5, 0.5]])*1e3
    time_step = 0.5
    time = np.arange(len(state_vecs), dtype=float)*time_step
    states = [State(state_space=transition.state_space, time=t, state=s) for t, s in zip(time, state_vecs)]
    states[-1].covar = CovarianceMatrix(np.diag([2.0, 0.75, 0.4, 0.4])*1e4)
    track = Track(states=states)

    new_time: float = time[-1].item() + time_step  # use the .item() command to force the output to be a scalar float
    prediction = transition.predict(s=track, new_time=new_time) # predict state for the next measurement
    gate_probability = 0.95

    # Instantiate a PSS System
    pss = TDOAPassiveSurveillanceSystem(x=np.array([[-1.0, -1.0, -1.0],[0.0, 0.5, 1.0]])*1e3,
                                        cov=CovarianceMatrix(np.diag([1.0, 1.0, 1.0])*1e2),
                                        variance_is_toa=False,
                                        ref_idx=0)
    msmt_model = MeasurementModel(pss=pss, state_space=transition.state_space)

    # Find the predicted measurement and measurement error covariance (for drawing an ellipse)
    prediction_msmt = msmt_model.measurement(state=prediction, noise=False)
    h_mat = msmt_model.jacobian(prediction)
    pred_cov = prediction.covar.cov
    msmt_cov = pss.cov.cov
    innov_covar = h_mat @ pred_cov @ h_mat.T + msmt_cov
    pred_gate = draw_error_ellipse(x=prediction_msmt.zeta, covariance=innov_covar, conf_interval=gate_probability)

    # Generate some random measurements
    num_meas = 10
    num_states = transition.state_space.num_states
    cov = np.eye(num_states) * 5e4
    state_vecs = np.random.multivariate_normal(mean=prediction.state, cov=cov, size=num_meas)
    truth_states = [State(state_space=transition.state_space, time=new_time, state=x) for x in state_vecs]
    truth_msmts = [msmt_model.measurement(state=s, noise=False) for s in truth_states]
    msmt_to_state_dict = dict(zip(truth_msmts, truth_states))

    # Make hypotheses
    hypotheses = [Hypothesis(track=track, measurement=m, motion_model=transition) for m in truth_msmts]

    # Apply the distance gate
    [h.apply_distance_gate(gate_probability) for h in hypotheses]
    valid_msmts = [h.measurement for h in hypotheses if h.is_valid]
    invalid_msmts = [h.measurement for h in hypotheses if not h.is_valid]
    valid_states = [msmt_to_state_dict[m] for m in valid_msmts]
    invalid_states = [msmt_to_state_dict[m] for m in invalid_msmts]

    # Plot the track and prediction
    figs = []
    fig, ax = plt.subplots()
    figs.append(fig)
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    track.plot(ax=ax, predicted_state=prediction, do_cov=True, do_vel=False,
               cov_ellipse_confidence=gate_probability, marker='o', color=default_colors[0])
    pss.plot_sensors(ax=ax, label='DF System', marker='+', color=default_colors[1])
    label='Valid Measurements'
    for s in valid_states:
        s.plot(ax=ax, do_cov=False, do_vel=False, label=label, marker='v', color=default_colors[2])
        label=None
    label='Invalid Measurements'
    for s in invalid_states:
        s.plot(ax=ax, do_cov=False, do_vel=False, label=label, marker='^', color=default_colors[3])
        label=None
    plt.grid(True)
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Track, Prediction, and Measurements')

    # Plot the Measurements and Prediction in Zeta-space
    fig, ax = plt.subplots()
    figs.append(fig)
    ax.scatter(prediction_msmt.zeta[0], prediction_msmt.zeta[1], marker='o', color=default_colors[0], label='Prediction')
    ax.plot(pred_gate[0], pred_gate[1], color=default_colors[0], label='Acceptance Gate')
    label='Valid Measurements'
    for m in valid_msmts:
        ax.scatter(m.zeta[0], m.zeta[1], marker='v', label=label, color=default_colors[2])
        label=None
    label='Invalid Measurements'
    for m in invalid_msmts:
        ax.scatter(m.zeta[0], m.zeta[1], marker='^', label=label, color=default_colors[3])
        label=None
    plt.grid(True)
    plt.xlabel('$\\tau_{0,1}$')
    plt.ylabel('$\\tau_{0,2}$')
    plt.legend()
    plt.title('Prediction and Measurements in Zeta-space')
    # ToDo: color-code measurements based on gate acceptance

    # Output to file
    if prefix is not None:
        labels = ['fig2a', 'fig2b']
        if len(labels) != len(figs):
            print('**Error saving figure 9.2; unexpected number of figures generated.')
        else:
            for fig, label in zip(figs, labels):
                fig.savefig(prefix + label + '.svg')
                fig.savefig(prefix + label + '.png')

    return figs

def make_figure_3(prefix=None):
    """
    Figure 9.3

    :param prefix: output directory to place generated figure
    :return: handle
    """

    print('Generating Figure 9.3...')

    # Define starting states
    ids = ['1','2','3']
    state_vecs = [[0, 2e3, 0, 100],
                  [1e3, 2e3, 70,-10],
                  [1e3, 1.3e3, 70, 50]]
    state_covars = [CovarianceMatrix(np.array([[1e4, 1e2, 0, 0],
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
    process_covar = np.diag([25,25])
    transition = ConstantVelocityMotionModel(num_dims=2, process_covar=process_covar)
    state_space = transition.state_space

    # Make the initial track states
    states = [State(state_space=state_space, time=0, state=s, covar=c) for (s, c) in zip(state_vecs, state_covars)]

    # Initialize a Track with each one and predict forward to the next measurement
    tracks = [Track(track_id=i, initial_state=s) for (i, s) in zip(ids, states)]
    new_time = 5
    predicted_states = [transition.predict(t.curr_state, new_time=new_time) for t in tracks]

    # Define Measurement System
    x_aoa = np.array([[750, 300],
                      [200, 800]])
    c_zeta = CovarianceMatrix(np.power(np.diag([3, 3])*_deg2rad,2))
    pss = DirectionFinder(x=x_aoa, cov=c_zeta, do_2d_aoa=False)
    msmt_model = MeasurementModel(pss=pss, state_space=state_space)

    # Let's make some new random states based on the predicted states for each track, and a few clutter-type states
    new_states = []
    num_meas_per_track = 3 # let's make two possible associations with each track
    for p in predicted_states:
        x = np.random.multivariate_normal(mean=p.state, cov=p.covar.cov, size=num_meas_per_track)
        s = [State(state=xx, state_space=transition.state_space, time=new_time) for xx in x]
        new_states.extend(s)

    num_background_msmt = 15
    x = np.random.multivariate_normal(mean=np.array([1e3,1e3,0,0]), cov=np.diag([5e4,5e4,100,100]), size=num_background_msmt)
    new_states.extend([State(state=xx, state_space=transition.state_space, time=new_time) for xx in x])
    measurements = [msmt_model.measurement(s) for s in new_states]

    # x/y plot
    figs = []
    fig, ax = plt.subplots()
    figs.append(fig)
    [t.plot(ax=ax, predicted_state=p) for (t, p) in zip(tracks, predicted_states)]
    pss.plot_sensors(ax=ax, label='DF System')
    [s.plot(ax=ax, marker='v', color='k', do_pos=True, do_vel=False, do_cov=False, label=None) for s in new_states]
    plt.title('Predicted Track States with new Measurements')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')

    # Measurement Plot
    fig, ax = plt.subplots()
    figs.append(fig)
    plt.scatter([m.zeta[0] for m in measurements],
                [m.zeta[1] for m in measurements],
                marker='v', color='k', label='Measurements')

    # Set up the Nearest Neighbor Association Scheme
    gate_probability=.75
    associator = NNAssociator(motion_model=transition, gate_probability=gate_probability)
    hypotheses, _ = associator.associate(tracks=tracks, measurements=measurements, print_table=True)

    trk_label = 'Predicted Track Measurement'
    gate_label = 'Association Gate'
    msmt_label = 'Associated Measurement'
    for t, p in zip(tracks, predicted_states):
        h = hypotheses[t]  # find the corresponding hypothesis

        # Plot a line from the predicted measurement to the selected one
        z = msmt_model.measurement(p).zeta
        hdl=plt.plot([z[0], h.measurement.zeta[0]], [z[1], h.measurement.zeta[1]], linestyle='-',
                     label=None)

        # Plot the predicted measurement directly
        plt.scatter(z[0], z[1], marker='o', color=hdl[0].get_color(), label=trk_label)
        trk_label = None
        pred_gate = draw_error_ellipse(x=h.measurement_prediction.zeta,
                                       covariance=h.innovation_covar.cov,
                                       conf_interval=gate_probability)
        plt.plot(pred_gate[0], pred_gate[1], color=hdl[0].get_color(), linestyle='--',
                 label=gate_label)
        gate_label=None
        plt.scatter(h.measurement.zeta[0], h.measurement.zeta[1], marker='v', color=hdl[0].get_color(),
                    label=msmt_label)
        msmt_label=None

    plt.xlabel('$\\theta_0$ [rad]')
    plt.ylabel('$\\theta_1$ [rad]')
    plt.title('Predicted Measurements and Associated Measurements')
    plt.legend()

    # Output to file
    if prefix is not None:
        labels = ['fig3a', 'fig3b']
        if len(labels) != len(figs):
            print('**Error saving figure 9.3; unexpected number of figures generated.')
        else:
            for fig, label in zip(figs, labels):
                fig.savefig(prefix + label + '.svg')
                fig.savefig(prefix + label + '.png')

    return figs

def make_figure_4_5(prefix=None):
    """
    Figure 9.4 and 9.5, from Example 9.1

    :param prefix: output directory to place generated figure
    :return: handle
    """

    print('Generating Figures 9.4 and 9.5...')

    figs = chapter9.example1()

    # Output to file
    if prefix is not None:
        labels = ['fig4a', 'fig5a', 'fig5b']
        if len(labels) != len(figs):
            print('**Error saving figures 9.4 and 9.5; unexpected number of figures generated.')
        else:
            for fig, label in zip(figs, labels):
                fig.savefig(prefix + label + '.svg')
                fig.savefig(prefix + label + '.png')

    return figs

def make_figure_6(prefix=None):
    """
    Figure 9.6, from Example 9.2

    :param prefix: output directory to place generated figure
    :return: handle
    """

    print('Generating Figure 9.6...')

    figs = chapter9.example2()

    # Output to file
    if prefix is not None:
        labels = ['fig6']
        if len(labels) != len(figs):
            print('**Error saving figure 9.6; unexpected number of figures generated.')
        else:
            for fig, label in zip(figs, labels):
                fig.savefig(prefix + label + '.svg')
                fig.savefig(prefix + label + '.png')

    return figs

def make_figure_7(prefix=None):
    """
    Figure 9.7, from Example 9.3

    :param prefix: output directory to place generated figure
    :return: handle
    """

    print('Generating Figure 9.7...')

    figs = chapter9.example3()

    # Output to file
    if prefix is not None:
        labels = ['fig7']
        if len(labels) != len(figs):
            print('**Error saving figure 9.7; unexpected number of figures generated.')
        else:
            for fig, label in zip(figs, labels):
                fig.savefig(prefix + label + '.svg')
                fig.savefig(prefix + label + '.png')

    return figs

def make_figure_8(prefix=None):
    """
    Figure 9.8

    :param prefix: output directory to place generated figure
    :return: handle
    """

    print('Generating Figure 9.8...')

    # Initialize the states for the track
    transition = ConstantVelocityMotionModel(num_dims=2,
                                                            process_covar=np.eye(2) * 1000)
    state_vecs = np.array([[0.0, 0.0, 1.0, 1.0],
                           [0.5, 0.5, 1.0, 0.0],
                           [1.0, 0.5, 0.5, 1.0],
                           [1.25, 1.0, 0.5, 0.5]]) * 1e3
    time_step = 1
    time = np.arange(len(state_vecs)) * time_step
    states = [State(state_space=transition.state_space, time=t, state=s) for t, s in zip(time, state_vecs)]
    states[-1].covar = CovarianceMatrix(np.diag([2.0, 0.75, 0.4, 0.4]) * 1e4)
    track = Track(states=states,id='0')

    num_missed_detections = 3

    # Plot the track
    figs = []
    fig, ax = plt.subplots()
    figs.append(fig)
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    track.plot(ax=ax, color=default_colors[0], label='Initial Track', do_cov=True, do_vel=False, marker='o')
    last_state = track.curr_state
    cov_label='State Error Covariance'
    for idx in range(num_missed_detections):
        h = MissedDetectionHypothesis(track=track,
                                      likelihood=1.0,
                                      motion_model=transition,
                                      time = track.curr_time + time_step)
        h.update_track()
        ax.plot(*zip((last_state.position, track.curr_state.position)),linestyle='--',label=None,color=default_colors[idx+1])
        track.curr_state.plot(ax=ax, color=default_colors[idx+1], do_pos=True, do_cov=False,
                              label=f"{idx+1} Missed Detection")
        track.curr_state.plot(ax=ax, color=default_colors[idx+1],do_pos=False,do_cov=True,label=cov_label,linestyle='--')
        cov_label=None
        last_state=track.curr_state
    plt.grid(True)
    plt.legend()
    plt.title('Error Covariance Growth as Missed Detections Accumulate')
    # Output to file
    if prefix is not None:
        labels = ['fig8']
        if len(labels) != len(figs):
            print('**Error saving figure 9.8; unexpected number of figures generated.')
        else:
            for fig, label in zip(figs, labels):
                fig.savefig(prefix + label + '.svg')
                fig.savefig(prefix + label + '.png')

    return figs

if __name__ == "__main__":
    make_all_figures(close_figs=False)
