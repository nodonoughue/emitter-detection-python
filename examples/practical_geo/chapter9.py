import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
import time

import ewgeo.tracker as ewt
from ewgeo.tracker.association import NNAssociator
from ewgeo.triang import DirectionFinder
from ewgeo.utils.covariance import CovarianceMatrix
from ewgeo.utils.unit_conversions import convert

_ft2m = convert(1, from_unit="ft", to_unit="m")
_rad2deg = convert(1, from_unit="rad", to_unit="deg")
_deg2rad = convert(1, from_unit="deg", to_unit="rad")

def run_all_examples():
    """
    Run all chapter 9 examples and return a list of figure handles

    :return figs: list of figure handles
    """

    return list(example1())


def example1():
    """
    Executes Example 9.1.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    10 October 2025

    :return: figure handle to generated graphic
    """

    # Define starting states
    ids = ['001','002','003']
    state_vecs = [[0, 2e3, 0, 100],
                  [1e3, 2e3, 70,-10],
                  [1e3, 1.3e3, 70, 50]]
    state_covars = [
        CovarianceMatrix(np.array([[1e4, 1e2, 0, 0],
                                   [1e2, 2e4, 0, 0],
                                   [0, 0, 2e2, 0],
                                   [0, 0, 0, 2e2]])),
        CovarianceMatrix(np.array([[4e4, -1e4, 0, 0],
                                   [-1e4, 1e4, 0, 0],
                                   [0, 0, 1e3, -5e2],
                                   [0, 0, -5e2, 2e2]])),
        CovarianceMatrix(np.array([[2e4, 1e4, 0, 0],
                                   [1e4, 4e4, 0, 0],
                                   [0, 0, 6e2, 4e2],
                                   [0, 0, 4e2, 6e2]]))]

    # Define the State Space; easiest method is to instantiate
    # a transition model
    process_covar = np.diag([25,25])
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
    msmt = ewt.measurement.MeasurementModel(state_space=state_space, pss=pss)

    # Plot the initial laydown
    plot_dims = np.s_[:2] # just plot the first two dimensions
    plot_args = {'marker':'^',
                 'markerfacecolor':'none'}

    figs = []
    fig, ax = plt.subplots()
    figs.append(fig)
    hdls=[t.plot(ax=ax, plot_dims=plot_dims, do_vel=True, do_cov=True, **plot_args) for t in tracks]
    colors = [hdl[0].get_color() for hdl in hdls] # get the track colors
    pss.plot_sensors(ax=ax, label='DF System')
    plt.legend()
    plt.xlim([-500,2000])
    plt.ylim([-500,3000])

    # Define measurements
    new_state_vecs = np.array([[150, 2650, 25, 100],
                               [1450,1775,75,-25],
                               [1650,1740,25,50]])
    new_time = 5
    new_states = [ewt.State(state_space=state_space,
                            time=new_time,
                            state=new_s) for new_s in new_state_vecs]
    # Build a state covariance error from the CRLB
    for s in new_states:
        s.covar = CovarianceMatrix(block_diag(pss.compute_crlb(s.position),1e6*np.eye(2)))

    coords=list(zip(*[s.position for s in new_states]))
    measurements = [msmt.measurement(s, noise=False) for s in new_states]

    # Predict the states forward and replot
    predicted_states = [transition.predict(t.curr_state, new_time=new_time) for t in tracks]

    fig, ax = plt.subplots()
    figs.append(fig)
    [t.plot(ax=ax, plot_dims=plot_dims, predicted_state=p, **plot_args) for (t, p) in zip(tracks, predicted_states)]
    pss.plot_sensors(ax=ax, label='DF System')
    [s.plot(ax=ax, plot_dims=plot_dims, color='k', do_pos=True, do_vel=False, do_cov=True, label=None) for s in new_states]
    plt.scatter(*coords, s=25, color='k', label='New States (truth)')

    # [pss.plot_lobs(ax=ax, zeta=m.zeta, plot_args={'color': 'k', 'linestyle': '-.','label':None}, scale=2e3) for m in measurements]
    plt.title('Predicted Track States with new Measurements')
    plt.legend()
    plt.xlim([-500,2000])
    plt.ylim([-500,3000])

    # Set up the Nearest Neighbor Association Scheme
    associator = NNAssociator(motion_model=transition, gate_probability=.75)
    hypotheses = associator.associate(tracks=tracks, measurements=measurements)

    [h.update_track() for h in hypotheses.values()]

    figs = []
    fig, ax = plt.subplots()
    figs.append(fig)
    [t.plot(ax=ax, plot_dims=plot_dims, **plot_args) for t in tracks]
    pss.plot_sensors(ax=ax, label='DF System')
    plt.legend()
    plt.title('Updated Trackers after NN Association')
    plt.xlim([-500,2000])
    plt.ylim([-500,3000])

    return [fig]


if __name__ == '__main__':
    run_all_examples()
    plt.show()
