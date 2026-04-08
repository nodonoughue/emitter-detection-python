from matplotlib.animation import Animation
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.quiver import Quiver
import numpy as np

from .association import Associator, Hypothesis, MissedDetectionHypothesis
from .initiator import Initiator
from .deleter import Deleter
from .measurement import Measurement
from .promoter import Promoter
from .track import Track

pause_interval=0.1 # seconds per frame update

class Tracker:
    # Parameters
    keep_all_tracks: bool = False
    deleter: Deleter = None
    initiator: Initiator = None
    promoter: Promoter = None
    associator: Associator

    # Track/State Properties
    tracks: list[Track]
    deleted_tracks: list[Track]     # only populated if keep_all_tracks is True
    _tentative_tracks: list[Track]  # not visible outside this class
    _failed_tracks: list[Track]     # tentative tracks that failed to get promoted
    _latest_measurements: list[Measurement]

    # Printing properties
    print_status: bool = False

    # Plotting properties
    do_plotting: bool = False
    _fig = None
    _ax = None
    plot_tentative_tracks: bool = False
    plot_state_error: bool = False
    plot_state_velocity: bool = False
    plot_is_initialized: bool = False
    plot_measurements: bool = False
    track_handles: dict[Track, tuple[Line2D | None, Line2D | None, Line2D | None, Quiver | None]] = {}
    msmt_handle: PathCollection = None
    animator: Animation = None

    def __init__(self,
                 associator: Associator,
                 initiator: Initiator,
                 promoter: Promoter,
                 deleter: Deleter,
                 **kwargs):
        """
        :param associator: Associator that matches measurements to tracks (NN, GNN, or PDA)
        :param initiator: Initiator that seeds new tentative tracks from unassociated measurements
        :param promoter: Promoter that decides when tentative tracks become firm tracks
        :param deleter: Deleter that decides when firm or tentative tracks are dropped
        :param kwargs: Additional keyword arguments set as instance attributes (e.g., ``keep_all_tracks``,
                       ``print_status``, ``do_plotting``, ``plot_tentative_tracks``, etc.)
        """
        self.associator = associator
        self.initiator = initiator
        self.promoter = promoter
        self.deleter = deleter
        self.deleted_tracks = []
        self.tracks = []
        self._tentative_tracks = []
        self._failed_tracks = []
        self._latest_measurements = []

        # Plotting variables
        self.track_handles = {}
        self._fig = None
        self._ax = None

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def all_tracks(self):
        return self.tracks + self.deleted_tracks

    @property
    def all_tentative_tracks(self):
        return self._tentative_tracks + self._failed_tracks

    def update(self, measurements, curr_time: float = None, reuse_measurements: bool=False):
        """
        Process one scan of measurements through the full tracker pipeline:
        associate → promote → initiate → delete.

        :param measurements: List of Measurement objects from the current scan
        :param curr_time: Timestamp of the current scan [seconds]; inferred from measurements[0].time if None
        :param reuse_measurements: If True, pass the full measurement set to tentative tracks even if
                                   those measurements were already used to update firm tracks
        """
        if curr_time is None and measurements:
            curr_time = measurements[0].time

        if self.print_status:
            print(f"Updating tracker with {len(measurements)} measurements...")

        # Copy the measurements, for record-keeping
        self._latest_measurements = measurements[:]

        # Associate measurements with existing tracks
        # Returns any unused measurements
        unassociated_measurements = self.update_existing_tracks(measurements=measurements, curr_time=curr_time)

        # Associate measurements with tentative tracks, look for any that can be promoted
        if reuse_measurements:
            # Pass the full set of measurements to the tentative tracks
            measurements_for_tentative_tracks = measurements[:]
        else:
            # Only pass those measurements that were not used to update firm tracks
            measurements_for_tentative_tracks = unassociated_measurements[:]
        unassociated_measurements_2 = self.promote(measurements=measurements_for_tentative_tracks, curr_time=curr_time)

        # Create new tracks with unused measurements; only include those that did not get used for either
        # firm or tentative tracks
        measurements_for_new_tracks = [m for m in unassociated_measurements if m in unassociated_measurements_2]
        self.initiate(measurements=measurements_for_new_tracks)

        # Delete tracks
        self.delete()

        # ---- Live plot update
        if self.do_plotting:
            if self._ax is None:
                raise RuntimeError("Plotting is enabled, but no axes exist."
                                   "Call tracker.setup_plot() before starting the update loop.")
            self.plot(ax=self._ax)
            plt.draw()
            plt.pause(pause_interval)
        return

    def update_existing_tracks(self, measurements: list[Measurement], curr_time: float = None) -> list[Measurement]:
        """
        Associate measurements with firm tracks and apply Kalman filter updates.

        :param measurements: New measurements from the current scan
        :param curr_time: Current scan timestamp [seconds]; forwarded to the associator for missed-detection coasting
        :return: List of measurements that were not associated with any firm track
        """
        if len(self.tracks) == 0:
            # Nothing to do
            return measurements

        # Generate hypotheses
        hypothesis_dict, unassoc_msmts = self.associator.associate(tracks=self.tracks, measurements=measurements, curr_time=curr_time)

        num_coasted_tracks = np.sum([1 for h in hypothesis_dict.values() if isinstance(h,MissedDetectionHypothesis)])
        if num_coasted_tracks > 0:
            pass

        # Update the hypotheses
        hypotheses = hypothesis_dict.values()
        [h.update_track() for h in hypotheses]

        # Apply per-track motion constraints to the freshly updated states
        for h in hypotheses:
            if not isinstance(h, MissedDetectionHypothesis):
                t = h.track
                if t.max_velocity is not None or t.max_acceleration is not None:
                    t.curr_state.constrain_motion(t.max_velocity, t.max_acceleration)

        if self.print_status:
            print(f"...{len(hypothesis_dict)-num_coasted_tracks} tracks updated, {num_coasted_tracks} tracks coasted...")
            print(f"...{len(unassoc_msmts)} measurements were not associated with any tracks...")

        # Return the unused measurements
        return unassoc_msmts

    def promote(self, measurements: list[Measurement], curr_time: float = None) -> list[Measurement]:
        """
        Associate measurements with tentative tracks, update them, and run the promoter to move
        qualifying tentative tracks to the firm track list (or discard failing ones).

        :param measurements: Measurements available for tentative track association
        :param curr_time: Current scan timestamp [seconds]; forwarded to the associator for coasting
        :return: List of measurements that were not associated with any tentative track
        """
        if len(self._tentative_tracks) == 0:
            # Nothing to do
            return measurements

        # for t in self._tentative_tracks:
        #     print(f"  Track {t.track_id}: P_trace={np.trace(t.curr_state.covar.cov):.4e}")

        # Generate hypotheses to match measurements to the tentative tracks
        hypothesis_dict, unassoc_msmt = self.associator.associate(tracks=self._tentative_tracks,
                                                                  measurements=measurements,
                                                                  curr_time=curr_time)

        # Update the tracks associated with these hypotheses
        tentative_hypotheses = hypothesis_dict.values()
        [h.update_track() for h in tentative_hypotheses]

        # Apply per-track motion constraints to the freshly updated tentative states
        for h in tentative_hypotheses:
            if not isinstance(h, MissedDetectionHypothesis):
                t = h.track
                if t.max_velocity is not None or t.max_acceleration is not None:
                    t.curr_state.constrain_motion(t.max_velocity, t.max_acceleration)

        num_updated_tracks = len([h for h in tentative_hypotheses if not isinstance(h, MissedDetectionHypothesis)])

        # Any hypotheses that are not a MissedDetectionHypothesis can be passed to
        # the promoter for evaluation
        tracks_to_test = [h.track for h in tentative_hypotheses]
        tracks_to_promote, tracks_to_remove = self.promoter.promote(tracks=tracks_to_test)

        # Add the promoted tracks to the track list and remove them from the tentative tracks list
        for t in tracks_to_promote:
            self.tracks.append(t)
            self._tentative_tracks.remove(t)

        for t in tracks_to_remove:
            self._tentative_tracks.remove(t)
            self._failed_tracks.append(t)

        if self.print_status:
            print(f"...{num_updated_tracks} tentative tracks updated...")
            print(f"...{len(tracks_to_promote)} tentative tracks promoted...")
            print(f"...{len(tracks_to_remove)} tentative tracks dropped...")

        return unassoc_msmt

    def initiate(self, measurements: list[Measurement]):
        """
        Seed new tentative tracks from measurements that were not associated with any existing track.

        :param measurements: Unassociated measurements from the current scan
        """
        if len(measurements) == 0:
            # Nothing to do
            return

        new_tracks, _ = self.initiator.initiate(measurements=measurements)
        self._tentative_tracks.extend(new_tracks)
        if self.print_status:
            print(f"...{len(new_tracks)} new tentative tracks created...")

        return

    def delete(self):
        """
        Run the deleter against all firm and tentative tracks, removing those that exceed the
        missed-detection threshold. Deleted tracks are moved to ``deleted_tracks`` if
        ``keep_all_tracks`` is True.
        """
        # Test the firm tracks
        tracks_to_delete = self.deleter.delete(tracks=self.tracks)

        # Remove the tracks by creating a new list that excludes them
        self.tracks = [t for t in self.tracks if t not in tracks_to_delete]

        if self.keep_all_tracks:
            self.deleted_tracks.extend(tracks_to_delete)

        # Repeat with the tentative tracks
        tentative_tracks_to_delete = self.deleter.delete(tracks=self._tentative_tracks)

        # Remove the tracks by creating a new list that excludes them
        self._tentative_tracks = [t for t in self._tentative_tracks if t not in tentative_tracks_to_delete]

        if self.keep_all_tracks:
            self.deleted_tracks.extend(tentative_tracks_to_delete)

        if self.print_status:
            print(f"...{len(tracks_to_delete)+len(tentative_tracks_to_delete)} tracks dropped...")

        return

    def setup_plot(self, ax: Axes = None, **kwargs) -> Axes:
        """
        Initialize the live-plot axes. Must be called before the update loop when ``do_plotting=True``.

        :param ax: Existing Axes to draw on; a new figure and axes are created if None
        :param kwargs: Keyword arguments forwarded to ``plt.subplots()`` when creating a new axes
        :return: The Axes object used for plotting
        """
        if ax is not None:
            self._ax = ax
            self._fig = ax.get_figure()
        else:
            self._fig, self._ax = plt.subplots(**kwargs)
        self.plot_is_initialized=True
        return self._ax

    def plot(self, ax: Axes=None, plot_dims: slice=np.s_[:], hypotheses: dict[Track, Hypothesis]=None,
             scale: float=1)-> Axes:
        """
        Plot all the track states to the axis
        """

        # Make a plot if ax=None
        if ax is None:
            _, ax = plt.subplots()

        # Grab the tracks to plot
        tracks_to_plot = self.tracks
        if self.plot_tentative_tracks:
            tracks_to_plot.extend(self._tentative_tracks)

        # Loop over tracks
        for track in tracks_to_plot:
            if hypotheses is not None and track in hypotheses:
                predicted_state = hypotheses[track].predicted_state
            else:
                predicted_state = None

            if track not in self.track_handles:
                # Need a new plot
                handles = track.plot(ax=ax, plot_dims=plot_dims, predicted_state=predicted_state, scale=scale,
                                     do_vel=self.plot_state_velocity, do_cov=self.plot_state_error)
                self.track_handles[track] = handles
            else:
                # Update the plot
                trk_hdl, trk_pred_hdl, trk_err_hdl, trk_vel_hdl = self.track_handles[track]

                track.update_plot(trk_hdl=trk_hdl, trk_pred_hdl=trk_pred_hdl, trk_err_hdl=trk_err_hdl,
                                  trk_vel_hdl=trk_vel_hdl,
                                  do_vel=self.plot_state_velocity, do_cov=self.plot_state_error,
                                  plot_dims=plot_dims, predicted_state=predicted_state, scale=scale)

        # Add Measurements
        if self.plot_measurements:
            # Grab coordinates from the measurements
            msmt_coords = np.array([m.sensor.least_square(zeta=m.zeta, x_init=None)[plot_dims]
                                    for m in self._latest_measurements])

            # Plot
            if self.msmt_handle is None:
                # Need a new plot
                self.msmt_handle = ax.scatter(msmt_coords)
            else:
                # Update the plot
                self.msmt_handle.set_offsets(msmt_coords)
        return ax
