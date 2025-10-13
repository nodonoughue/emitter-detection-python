import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag


import ewgeo.tracker as ewt
from ewgeo.tracker import StateSpace, Track
from ewgeo.tracker.association import NNAssociator, GNNAssociator, PDAAssociator
from ewgeo.tracker.measurement import Measurement
from ewgeo.tracker.transition import MotionModel
from ewgeo.triang import DirectionFinder
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

    return list(example1()) + list(example2()) + list(example3())


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
    transition = ewt.transition.ConstantVelocityMotionModel(num_dims=2, process_covar=process_covar)
    state_space = transition.state_space

    # Make the initial track states
    states = [ewt.State(state_space=state_space, time=0, state=s, covar=c) for (s, c) in zip(state_vecs, state_covars)]

    # Initialize a Track with each one
    tracks = [ewt.track.Track(track_id=i, initial_state=s) for (i, s) in zip(ids, states)]

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
    new_states = [ewt.State(state_space=state_space,
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
    hypotheses = associator.associate(tracks=tracks, measurements=measurements, print_table=True)

    # Make a plot for the hypothesis assignments
    fig, ax = plt.subplots()
    figs.append(fig)
    [h.update_track(ax=ax) for h in hypotheses.values()]
    plt.scatter(*coords, s=25, color='k', label='New States (truth)')
    plt.legend()

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
    hypotheses = associator.associate(tracks=tracks, measurements=measurements, print_table=True)
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
    hypotheses = associator.associate(tracks=tracks, measurements=measurements, print_table=True)
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

if __name__ == '__main__':
    run_all_examples()
    plt.show()
