import numpy as np
import matplotlib.pyplot as plt
import time

from ewgeo.tdoa import TDOAPassiveSurveillanceSystem
import ewgeo.tracker as ewt
from ewgeo.triang import DirectionFinder
from ewgeo.utils import print_progress, print_elapsed, safe_2d_shape
from ewgeo.utils.constants import speed_of_light
from ewgeo.utils.covariance import CovarianceMatrix
from ewgeo.utils.errors import draw_error_ellipse
from ewgeo.utils import tracker
from ewgeo.utils.unit_conversions import convert

_ft2m = convert(1, from_unit="ft", to_unit="m")
_rad2deg = convert(1, from_unit="rad", to_unit="deg")
_deg2rad = convert(1, from_unit="deg", to_unit="rad")

def run_all_examples():
    """
    Run all chapter 8 examples and return a list of figure handles

    :return figs: list of figure handles
    """

    return list(example1())


def example1():
    """
    Executes Example 9.1.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    29 June 2025

    :return: figure handle to generated graphic
    """

    # Define starting states
    ids = ['001','002','003']
    state_vecs = [[0, 2e3, 0, 100],
                  [1e3, 2e3, 100, 0],
                  [1e3, 1.5e3, 50, 50]]
    state_covars = [
        CovarianceMatrix(np.array([[2e4, 1e2, 0, 0],
                                   [1e2, 1e4, 0, 0],
                                   [0, 0, 3e3, 1e2],
                                   [0, 0, 1e2, 1e3]])),
        CovarianceMatrix(np.array([[2e4, 1e2, 0, 0],
                                   [1e2, 1e4, 0, 0],
                                   [0, 0, 3e3, 1e2],
                                   [0, 0, 1e2, 1e3]])),
        CovarianceMatrix(np.array([[2e4, 1e3, 0, 0],
                                   [1e3, 1e5, 0, 0],
                                   [0, 0, 3e3, 1e2],
                                   [0, 0, 1e2, 1e3]]))]

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
    x_aoa = np.array([[500, 300],
                      [200, 800]])
    c_zeta = CovarianceMatrix(np.power(np.diag([2, 2])*_deg2rad,2))
    pss = DirectionFinder(x=x_aoa, cov=c_zeta, do_2d_aoa=False)
    msmt = ewt.measurement.MeasurementModel(state_space=state_space, pss=pss)

    # Plot the initial laydown
    plot_axes = np.s_[:2] # just plot the first two dimensions
    plot_args = {'marker':'^',
                 'markerfacecolor':'none'}

    figs = []
    fig, ax = plt.subplots()
    figs.append(fig)
    hdls=[t.plot(ax=ax, plot_axes=plot_axes, **plot_args) for t in tracks]
    colors = [hdl[0].get_color() for hdl in hdls] # get the track colors
    pss.plot_sensors(ax=ax, label='DF System')
    plt.legend()

    # Define measurements
    new_state_vecs = np.array([[150, 2450, 25, 100],
                               [1250,1900,75,-25],
                               [1350,1800,25,50]])
    new_time = 5
    new_states = [ewt.State(state_space=state_space,
                            time=new_time,
                            state=new_s) for new_s in new_state_vecs]
    coords=list(zip(*[s.position for s in new_states]))
    measurements = [msmt.measurement(s, noise=True) for s in new_states]

    # Predict the states forward and replot
    predicted_states = [transition.predict(t.curr_state, new_time=new_time) for t in tracks]

    fig, ax = plt.subplots()
    figs.append(fig)
    hdls = [t.plot(ax=ax, plot_axes=plot_axes, predicted_state=p, **plot_args) for (t, p) in zip(tracks, predicted_states)]
    pss.plot_sensors(ax=ax, label='DF System')
    plt.scatter(*coords, s=100, color='k', label='New States (truth)')
    [pss.plot_lobs(ax=ax, zeta=m.zeta, plot_args={'color': 'k', 'linestyle': '-.','label':None}, scale=2e3) for m in measurements]
    plt.legend()


    return [fig]


if __name__ == '__main__':
    run_all_examples()
    plt.show()
