from matplotlib import pyplot as plt
import numpy as np

from .states import State, StateSpace


class Track:
    """
    Collection of states representing the track of an emitter over time
    """
    # Parameters
    initial_state: State
    states: list[State]
    num_dims: int
    track_id: str = ""
    num_missed_detections: int = 0

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if key == 'initial_state':
                self.states = [value]

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
    def curr_time(self)->float:
        return self.curr_state.time

    @property
    def curr_state(self)->State:
        return self.states[-1]

    def append(self, state=State, missed_detection: bool=False) -> None:
        self.states.append(state)
        if missed_detection:
            self.num_missed_detections += 1
        else:
            self.num_missed_detections = 0

    def copy(self, **kwargs):
        # Initialize a new track using all the current track's properties
        if 'track_id' not in kwargs:
            # Make sure we don't copy the track ID
            kwargs['track_id'] = None
        new_track = Track(**self.__dict__)
        for key, value in kwargs.items():
            new_track.__setattr__(key, value)
        return new_track

    def plot(self, ax: plt.Axes, plot_dims: slice= np.s_[:],
             predicted_state: State=None,
             do_vel: bool=False, do_cov: bool=True,
             scale: float=1, cov_ellipse_confidence: float=.75,
             **kwargs):
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
        hdl=ax.plot(*coords, **kwargs, **plot_args)

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

            ax.plot(*pred_coords, linestyle='--', color=hdl[0].get_color(), label=this_label)

            # Velocity and Covariance of predicted state
            predicted_state.plot(ax=ax, plot_dims=plot_dims, do_pos=True, do_vel=do_vel, do_cov=do_cov,
                                 color=hdl[0].get_color(), cov_ellipse_confidence=cov_ellipse_confidence, scale=scale,
                                 linestyle='--')
        else:
            # Velocity and Covariance of final state
            self.curr_state.plot(ax=ax, plot_dims=plot_dims, do_pos=False, do_vel=do_vel, do_cov=do_cov,
                                 color=hdl[0].get_color(), cov_ellipse_confidence=cov_ellipse_confidence, scale=scale)

        return hdl
