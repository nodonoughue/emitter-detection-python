from matplotlib import pyplot as plt
import numpy as np
from typing import MutableSequence

from .states import State, StateSpace
from ..utils.errors import draw_error_ellipse


class Track:
    """
    Collection of states representing the track of an emitter over time
    """
    # Parameters
    initial_state: State
    states: MutableSequence[State]
    num_dims: int
    track_id: str

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if key == 'initial_state':
                self.states = [value]

            setattr(self, key, value)

    @property
    def curr_time(self)->float:
        return self.curr_state.time

    @property
    def curr_state(self)->State:
        return self.states[-1]

    def append(self, state=State) -> None:
        self.states.append(state)
        self.curr_time = state.time

    def copy(self, **kwargs):
        # Initialize a new track using all the current track's properties
        new_track = Track(**self.__dict__)
        for key, value in kwargs.items():
            new_track.__setattr__(key, value)
        return new_track

    def plot(self, ax: plt.Axes, plot_axes: slice=np.s_[:],
             predicted_state: State=None,
             **kwargs):

        # Pull the appropriate state dimensions from each state
        # in the track's history
        coords = list(zip(*[s.position[plot_axes] for s in self.states]))

        # Line plot
        hdl=ax.plot(*coords, **kwargs, label='Track {}'.format(self.track_id))

        if predicted_state is not None:
            # Predicted state
            pred_coords = [[c[-1], p] for c, p in zip(coords, predicted_state.position[plot_axes])]
            ax.plot(*pred_coords, label=None, linestyle='--', color=hdl[0].get_color())

            # Velocity and Covariance of predicted state
            predicted_state.plot(ax=ax, plot_dims=plot_axes, do_pos=False, do_vel=True, do_cov=True, color=hdl[0].get_color())
        else:
            # Velocity and Covariance of final state
            self.curr_state.plot(ax=ax, plot_dims=plot_axes, do_pos=False, do_vel=True, do_cov=True, color=hdl[0].get_color())

        return hdl
