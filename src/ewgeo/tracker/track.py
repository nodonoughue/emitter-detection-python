from matplotlib import pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.quiver import Quiver

from .states import State, StateSpace


class Track:
    """
    Collection of states representing the track of an emitter over time
    """
    # Parameters
    initial_state: State
    states: list[State]
    track_id: str = ""
    num_missed_detections: int = 0
    num_updates: int = 0

    def __init__(self, **kwargs):
        """
        Initialize a Track. Pass ``initial_state`` as a keyword argument to seed the state list.

        :param initial_state: (keyword) Initial State object; sets states=[initial_state] and num_updates=1
        :param track_id: (keyword) Optional identifier string for the track (default: "")
        """
        for key, value in kwargs.items():
            if key == 'initial_state':
                self.states = [value]
                self.num_updates = 1

            setattr(self, key, value)

    def __str__(self):
        if self.track_id is not None:
            return f'Track {self.track_id}: {self.curr_state.state} @ {self.curr_time:.2f}s'
        else:
            return f'Track {id(self)}: {self.curr_state.state} @ {self.curr_time:.2f}s'

    @property
    def state_space(self)->StateSpace:
        return self.curr_state.state_space

    @property
    def num_dims(self)->int:
        return self.state_space.num_dims

    @property
    def curr_time(self)->float:
        return self.curr_state.time

    @property
    def curr_state(self)->State:
        return self.states[-1]

    def append(self, state: State, missed_detection: bool=False) -> None:
        """
        Append a new state to this track.

        :param state: New State to append
        :param missed_detection: If True, increments the consecutive missed-detection counter and does not
                                 increment num_updates. If False, resets the counter and increments num_updates.
        """
        self.states.append(state)
        if missed_detection:
            self.num_missed_detections += 1
        else:
            self.num_missed_detections = 0
            self.num_updates += 1

    def copy(self, **kwargs):
        """
        Return a shallow copy of this track, optionally overriding attributes via kwargs.
        The states list is shallow-copied so the original State objects are shared.
        A '_0' suffix is appended to the track_id unless track_id is provided in kwargs.
        """
        # Initialize a new track using all the current track's properties
        if 'track_id' not in kwargs:
            kwargs['track_id'] = self.track_id + '_0'

        new_track = object.__new__(Track)
        new_track.__dict__.update(self.__dict__) # copy all attributes
        new_track.states = self.states[:] # shallow-copy of list; original state objects preserved
        for key, value in kwargs.items():
            new_track.__setattr__(key, value)

        return new_track

    def plot(self, ax: plt.Axes, plot_dims: slice= np.s_[:],
             predicted_state: State=None,
             do_vel: bool=False, do_cov: bool=True,
             scale: float=1, cov_ellipse_confidence: float=.75,
             **kwargs)-> tuple[Line2D | None, Line2D | None, Line2D | None, Quiver | None]:
        """
        Plot the states on the provided axis, with an optional scale factor (for converting from m to km, etc.).

        :param ax: Axes object to plot on
        :param plot_dims: Slice representing which spatial axes to plot (optional)
        :param predicted_state: Optional predicted state to append to the track plot.
        :param do_vel: (default=False) If true, then a velocity arrow will be appended to the final (or predicted) state
        :param do_cov: (default=False) If true, then a covariance ellipse will be plotted around the final (or
        predicted) state
        :param scale: (default=1) Scale the track plot by this factor
        :param cov_ellipse_confidence: (default=0.75) Confidence interval for covariance ellipse visualization
        :param kwargs: Keyword arguments to pass to ax.plot()
        """
        # Pull the appropriate state dimensions from each state
        # in the track's history
        coords = list(zip(*[s.position[plot_dims] / scale for s in self.states]))

        plot_args = {}
        if 'label' not in kwargs.keys():
            plot_args['label'] = f"Track {self.track_id}"
        # Line plot
        trk_hdl=ax.plot(*coords, **kwargs, **plot_args)[0]

        if predicted_state is not None:
            # Predicted state
            pred_coords = [[c[-1], p/scale] for c, p in zip(coords, predicted_state.position[plot_dims])]
            if 'label' in kwargs.keys():
                if kwargs['label'] is None:
                    this_label=None
                else:
                    this_label = f"{kwargs['label']} Predicted State"
            else:
                this_label = "Predicted State"

            trk_pred_hdl = ax.plot(*pred_coords, linestyle='--', marker='o', markevery=[-1],
                                   color=trk_hdl.get_color(), label=this_label)

            # Velocity and Covariance of predicted state
            _, trk_err_hdl, trk_vel_hdl =  predicted_state.plot(ax=ax, plot_dims=plot_dims,
                                                                do_pos=False, do_vel=do_vel, do_cov=do_cov,
                                                                color=trk_hdl.get_color(),
                                                                cov_ellipse_confidence=cov_ellipse_confidence,
                                                                scale=scale,
                                                                linestyle='--')
        else:
            trk_pred_hdl = None

            # Velocity and Covariance of final state
            _, trk_err_hdl, trk_vel_hdl = self.curr_state.plot(ax=ax, plot_dims=plot_dims,
                                                               do_pos=False, do_vel=do_vel, do_cov=do_cov,
                                                               color=trk_hdl.get_color(),
                                                               cov_ellipse_confidence=cov_ellipse_confidence,
                                                               scale=scale,
                                                               linestyle='-')

        return trk_hdl, trk_pred_hdl, trk_err_hdl, trk_vel_hdl

    def update_plot(self,
                    trk_hdl: Line2D | None, trk_pred_hdl: Line2D | None, trk_err_hdl: Line2D | None,
                    trk_vel_hdl: Quiver | None,
                    plot_dims: slice= np.s_[:],
                    predicted_state: State=None,
                    do_vel: bool=False, do_cov: bool=True,
                    scale: float=1, cov_ellipse_confidence: float=.75):
        """
        Update existing plot handles with the current track history (for animation).

        :param trk_hdl: Existing track history Line2D handle to update
        :param trk_pred_hdl: Existing predicted-state Line2D handle to update, or None
        :param trk_err_hdl: Existing covariance ellipse Line2D handle to update, or None
        :param trk_vel_hdl: Existing velocity Quiver handle to update, or None
        :param plot_dims: Slice representing which spatial axes to plot (default: all)
        :param predicted_state: Optional predicted state to draw as a dashed extension of the track
        :param do_vel: If True, update the velocity arrow (default: False)
        :param do_cov: If True, update the covariance ellipse (default: True)
        :param scale: Divide all coordinates by this factor before plotting (default: 1)
        :param cov_ellipse_confidence: Confidence interval for covariance ellipse visualization (default: 0.75)
        """

        # Pull the appropriate state dimensions from each state
        # in the track's history and update trk_hdl
        coords = list(zip(*[s.position[plot_dims] / scale for s in self.states]))
        trk_hdl.set_data(*coords)

        if predicted_state is not None:
            pred_coords = [[c[-1], p/scale] for c, p in zip(coords, predicted_state.position[plot_dims])]
            trk_pred_hdl.set_data(*pred_coords)

            # Velocity and Covariance of predicted state
            predicted_state.update_plot(trk_hdl=None, trk_err_hdl=trk_err_hdl, trk_vel_hdl=trk_vel_hdl,
                                        plot_dims=plot_dims, do_pos=False, do_vel=do_vel, do_cov=do_cov,
                                        cov_ellipse_confidence=cov_ellipse_confidence, scale=scale)
        else:
            if trk_pred_hdl is not None:
                # Clear the data if there's no predicted state
                trk_pred_hdl.set_data([], [])

            # Velocity and Covariance of final state
            self.curr_state.update_plot(trk_hdl=None, trk_err_hdl=trk_err_hdl, trk_vel_hdl=trk_vel_hdl,
                                        plot_dims=plot_dims, do_pos=False, do_vel=do_vel, do_cov=do_cov,
                                        cov_ellipse_confidence=cov_ellipse_confidence, scale=scale)

        return