from abc import ABC, abstractmethod
from collections.abc import Mapping
import numpy as np
from matplotlib import pyplot as plt
from numpy import typing as npt
# from prettytable import PrettyTable
from scipy.stats import chi2, multivariate_normal
from scipy.optimize import linear_sum_assignment

from . import State
from .measurement import Measurement, MeasurementModel
from .track import Track
from .transition import MotionModel
from ..utils.constraints import snap_to_constraints
from ..utils.covariance import CovarianceMatrix
from ..utils.system import PassiveSurveillanceSystem

# Cell colors for use with prettytable
pt_red = "\033[0;31;40m"
pt_blue = "\033[0;34;40m"
pt_reset = "\033[0m"

class Hypothesis:
    """
    Based on the Stone Soup class of the same name, a Hypothesis is simply the association
    of a measurement with a track. It consists of three parameters:

    :param measurement: the new measurement that is being incorporated
    :param track: the existing track to which the measurement will be appended
    """
    track: Track                                # Track under test
    _measurement: Measurement | None             # Measurement to associate to Track
    _measurement_model: MeasurementModel | None
    _motion_model: MotionModel | None

    # Computed parameters
    _is_valid: bool = True
    _distance: float | None = None
    _pred: State | None = None
    _msmt_pred: Measurement | None = None
    _state_from_msmt: State | None = None
    _innov: npt.ArrayLike | None = None
    _innov_covar: CovarianceMatrix | None = None
    _likelihood: float | None = None
    _log_likelihood: float | None = None

    def __init__(self, track: Track, measurement: Measurement | None, motion_model: MotionModel | None = None):
        """
        :param track: Existing track to evaluate this hypothesis against
        :param measurement: New measurement to tentatively associate with the track, or None for a missed detection
        :param motion_model: MotionModel used to predict the track forward to the measurement time
        """
        self.track = track
        self.measurement = measurement
        self.motion_model = motion_model

    def compute_gate_size(self, gate_probability: float)->float:
        """
        Compute the Acceptance Gate size from the gate probability and the measurement size in the
        assigned track.

        The distance metric lambda will be distributed as a Chi-Square random variable with num_measurements degrees
        of freedom, where num_measurements is the size of the measurement vector.
        """
        if self.measurement is None:
            return np.inf
        else:
            return chi2.ppf(gate_probability, self.measurement.size)

    def apply_distance_gate(self, gate_probability: float):
        """
        Apply an association window based on the desired gate probability. First, compute the track's gate size,
        based on the dimension of the measurement, then apply it.

        If the distance is greater than the gate size, replace with np.inf
        """
        if self.distance is not None and self.distance > self.compute_gate_size(gate_probability):
            self.invalidate()

    def invalidate(self):
        self._is_valid = False
        self._distance = np.inf
        self._likelihood = 0.
        self._log_likelihood = -np.inf

    @property
    def is_valid(self) -> bool:
        # This gets set to false if a gate is applied, and this fails that gate.
        return self._is_valid

    @property
    def predicted_state(self) -> State:
        if self._motion_model is None:
            raise ValueError('Hypothesis must have a motion model to generate a predicted state')

        if self._pred is None and self.measurement is not None:
            # Generate the predicted state
            self._pred = self.motion_model.predict(s=self.track.curr_state, new_time=self.measurement.time)
        return self._pred

    @property
    def measurement_prediction(self) -> Measurement:
        if self._msmt_pred is None and self.measurement is not None:
            # Use the predicted state to generate the predicted measurement
            self._msmt_pred = self.measurement_model.measurement(self.predicted_state, noise=False)

        return self._msmt_pred

    @property
    def innovation(self) -> npt.NDArray:
        if self._innov is None and self._measurement is not None:
            self._innov = self.measurement.zeta - self.measurement_prediction.zeta
        return self._innov

    @property
    def innovation_covar(self) -> CovarianceMatrix | None:
        if self._innov_covar is None and self._measurement is not None:
            # Innovation covariance is computed from the measurement matrix, predicted state covariance, and
            # measurement covariance
            h_mat = self.measurement_model.jacobian(self.predicted_state)
            pred_cov = self.predicted_state.covar.cov
            msmt_cov = self.measurement.sensor.cov.cov
            self._innov_covar = CovarianceMatrix(h_mat @ pred_cov @ h_mat.T + msmt_cov)
        return self._innov_covar

    @property
    def distance(self) -> float:
        if self._distance is None:
            if self.measurement is None:
                return np.inf
            else:
                # Mahalanobis distance squared: y' S^{-1} y ~ chi2(n)
                self._distance = self.innovation_covar.solve_aca(self.innovation.T)
        return self._distance

    @property
    def likelihood(self) -> float:
        if self._likelihood is None and self.measurement is not None:
            # Compute a new likelihood, based on the assumption that the innovation is a multivariate Gaussian with
            # zero-mean and a covariance given by the innovation covariance.
            self._likelihood = multivariate_normal.pdf(x=self.innovation,
                                                       mean=np.zeros_like(self.innovation),
                                                       cov=self.innovation_covar.cov)
        return self._likelihood

    @property
    def log_likelihood(self) -> float:
        if self._log_likelihood is None and self.measurement is not None:
            # Compute a new likelihood, based on the assumption that the innovation is a multivariate Gaussian with
            # zero-mean and a covariance given by the innovation covariance.
            self._log_likelihood = multivariate_normal.logpdf(x=self.innovation,
                                                              mean=np.zeros_like(self.innovation),
                                                              cov=self.innovation_covar.cov)
        return self._log_likelihood

    @property
    def measurement(self):
        return self._measurement

    @measurement.setter
    def measurement(self, value: Measurement):
        self._measurement = value
        if value is not None:
            self.measurement_model = MeasurementModel(state_space=self.track.state_space, pss=value.sensor)

        # Clear the dependent parameters
        self.clear_dependent_parameters()

    @property
    def motion_model(self):
        return self._motion_model

    @motion_model.setter
    def motion_model(self, value: MotionModel):
        self._motion_model = value
        self.clear_dependent_parameters()

    @property
    def state_from_measurement(self):
        if self._state_from_msmt is None:
            self._state_from_msmt = self.measurement_model.state_from_measurement(self.measurement)

        return self._state_from_msmt

    def clear_dependent_parameters(self):
        """Reset all lazily-computed properties so they are recalculated on next access."""
        self._state_from_msmt = None
        self._distance = None
        self._pred = None
        self._msmt_pred = None
        self._innov = None
        self._innov_covar = None
        self._likelihood = None
        self._log_likelihood = None
        self._is_valid = True

    def override_likelihood(self, likelihood: float):
        """Manually set the likelihood and log-likelihood, bypassing the lazy computation."""
        self._likelihood = likelihood
        self._log_likelihood = np.log(likelihood)

    def update_track(self, spawn_new_track: bool=False, ax: plt.Axes=None, plot_dims: slice=np.s_[:])->Track:
        """
        Apply this hypothesis to produce a Kalman filter update on the associated track.

        Predicts the track to the measurement time, computes the Kalman gain, and appends the
        updated state to the track (or a copy, if spawn_new_track is True).

        :param spawn_new_track: If True, operate on a shallow copy of the track rather than in place
        :param ax: Optional Axes to plot the predicted state, update step, and updated state on
        :param plot_dims: Slice selecting which spatial dimensions to plot (default: all)
        :return: The updated Track (either the original or its copy)
        """
        # Apply this hypothesis and update the track
        if spawn_new_track:
            t = self.track.copy()
        else:
            t = self.track

        # Plot the results
        if ax is not None:
            hdl = t.plot(ax=ax, do_vel=False, do_cov=True, predicted_state=self.predicted_state, marker='^')

        # Compute the Kalman Gain, which is the product of the predicted state covariance, the transpose of the
        # measurement Jacobian, and the inverse of the innovation covariance
        prediction_state_covar = self.predicted_state.covar.cov
        measurement_jacobian = self.measurement_model.jacobian(self.predicted_state)
        innovation_covar_inv = self.innovation_covar.inv
        kalman_gain = prediction_state_covar @ measurement_jacobian.T @ innovation_covar_inv

        # Update the Estimate
        new_state_vec = self.predicted_state.state + kalman_gain @ self.innovation
        new_state_covar = (np.eye(t.curr_state.size) - kalman_gain @ measurement_jacobian) @ prediction_state_covar

        # Snap position to inequality constraints (e.g. altitude bounds) if the motion
        # model carries them. The covariance is left as-is (project-then-filter approximation).
        ineq = getattr(self._motion_model, 'ineq_constraints', None) or \
               getattr(self.measurement_model, 'ineq_constraints', None)
        if ineq is not None:
            n = t.curr_state.state_space.num_dims
            pos_sl = t.curr_state.state_space.pos_slice
            pos = new_state_vec[pos_sl].reshape(n, 1)
            new_state_vec[pos_sl] = snap_to_constraints(pos, ineq_constraints=ineq).ravel()

        # Make a new State object and add it to the track
        new_state = State(t.curr_state.state_space, self.measurement.time, new_state_vec, CovarianceMatrix(new_state_covar))
        if ax is not None:
            # Plot a dashed line from the current state to the new one
            plt.plot(*zip(t.curr_state.position[plot_dims], new_state.position[plot_dims]),
                     color=hdl[0].get_color(), linestyle=':', label=f'Track {self.track.track_id}, Updated State')
            new_state.plot(ax=ax, do_vel=False, do_cov=True, color=hdl[0].get_color(), linestyle=':')

        t.append(new_state)
        return t

    def __str__(self):
        return f'Hypothesis({self.track}, {self.measurement}), distance = {self.distance}, likelihood = {self.likelihood}'

class MissedDetectionHypothesis(Hypothesis):
    """
    Hypothesis representing a missed detection: the track is coasted (predicted forward) with
    no measurement update. The distance and likelihood are supplied at construction and held fixed.
    """
    def __init__(self, track: Track, motion_model: MotionModel, sensor: PassiveSurveillanceSystem | None,
                 time: float, gate_probability: float = None, num_msmt_dims: int = None,
                 distance: float = None):
        """
        :param track: Existing track to coast
        :param motion_model: MotionModel used to predict the track forward to ``time``
        :param sensor: Sensor associated with this scan (may be None when no measurements exist)
        :param time: Timestamp to coast the track to [seconds]
        :param gate_probability: Chi-square gate probability (e.g., 0.99); used with num_msmt_dims
                                 to compute the gate threshold chi2.ppf(gate_probability, num_msmt_dims).
                                 Ignored when distance is provided explicitly.
        :param num_msmt_dims: Measurement vector dimension; combined with gate_probability to
                              compute the chi-square gate threshold. When None (e.g., no measurements
                              are available in the current scan), distance defaults to 0.0, meaning
                              this hypothesis always wins any cost comparison.
        :param distance: Override the inferred distance directly (e.g., PDA uses 1 − Pd·Pg as a
                         probability weight rather than a Mahalanobis threshold). When None the value
                         is computed from gate_probability and num_msmt_dims.
        """
        dummy_measurement = Measurement(sensor=sensor, time=time, zeta=np.array([0.0]))
        super().__init__(track, dummy_measurement, motion_model)

        if distance is None:
            if gate_probability is not None and num_msmt_dims is not None:
                distance = chi2.ppf(gate_probability, num_msmt_dims)
            else:
                distance = 0.0  # no competing measurements; this hypothesis always wins

        self._distance = distance
        self._likelihood = distance  # also use it as the likelihood
        self._log_likelihood = np.log(distance) if distance > 0 else -np.inf
        self._innov = None
        self._innov_covar = None

    @property
    def innovation(self) -> npt.NDArray:
        """
        By definition, the innovation of a missed detection hypothesis is zero
        """
        return np.zeros_like(self.measurement_prediction.zeta)

    @property
    def innovation_covar(self) -> CovarianceMatrix | None:
        """
        By definition, the innovation covariance of a missed detection hypothesis is the
        same as the predicted state measurement covariance, there is no additional measurement covariance.
        """
        if self._innov_covar is None and self._measurement is not None:
            # Innovation covariance is computed from the measurement matrix, predicted state covariance, and
            # measurement covariance
            h_mat = self.measurement_model.jacobian(self.predicted_state)
            pred_cov = self.predicted_state.covar.cov
            self._innov_covar = CovarianceMatrix(h_mat @ pred_cov @ h_mat.T)
        return self._innov_covar

    @property
    def distance(self) -> float:
        """Return the fixed distance supplied at construction (typically 1 − gate_probability)."""
        return self._distance

    @property
    def likelihood(self) -> float:
        return self._likelihood

    @property
    def log_likelihood(self) -> float:
        return self._log_likelihood

    def update_track(self, spawn_new_track: bool = False, ax: plt.Axes = None, plot_dims: slice = np.s_[:])->Track:
        """
        In the case of a missed detection, the track update is simply the predicted state.
        """

        # Apply this hypothesis and update the track
        if spawn_new_track:
            t = self.track.copy()
        else:
            t = self.track

        # Plot the results
        if ax is not None:
            hdl = t.plot(ax=ax, do_vel=False, do_cov=True, predicted_state=self.predicted_state, marker='^')

        # Add the predicted state to the track
        new_state = self.predicted_state

        if ax is not None:
            # Plot a dashed line from the current state to the new one
            plt.plot(*zip(t.curr_state.position[plot_dims], new_state.position[plot_dims]),
                     color=hdl[0].get_color(), linestyle=':', label=f'Track {self.track.track_id}, Updated State')
            new_state.plot(ax=ax, do_vel=False, do_cov=True, color=hdl[0].get_color(), linestyle=':')

        t.append(new_state, missed_detection=True)
        return t
class GMMHypothesis(Hypothesis):
    """
    Compound hypothesis that is a Gaussian Mixture Model of individual hypotheses, each with a measurement and
    weight associated. Used by the PDA associator to represent the full set of gated measurement hypotheses
    for one track as a single weighted mixture.
    """
    _hypotheses: list[Hypothesis]
    _weights: npt.ArrayLike

    def __init__(self, hypotheses: list[Hypothesis], weights: npt.ArrayLike = None, motion_model: MotionModel=None):
        """
        :param hypotheses: List of individual Hypothesis (or MissedDetectionHypothesis) objects to combine
        :param weights: Normalized association weights, one per hypothesis; uniform if None
        :param motion_model: Unused (inherited track's motion model is used); kept for API consistency
        """
        super().__init__(track=hypotheses[0].track, measurement=None)
        self._hypotheses = hypotheses
        self._weights = weights if weights is not None else np.ones(len(hypotheses))

    def update_track(self, spawn_new_track: bool=False, ax: plt.Axes = None, plot_dims: slice = np.s_[:])->Track:
        """
        Reduce the Gaussian Mixture of hypotheses to a single hypothesis and return that.
        """

        # Spawn a new track, if needed
        if spawn_new_track:
            t = self.track.copy()
        else:
            t = self.track

        if len(self._hypotheses) == 1 and isinstance(self._hypotheses[0], MissedDetectionHypothesis):
            # The only one is a missed detection hypothesis
            new_state = self._hypotheses[0].update_track(spawn_new_track=True).curr_state
            t.append(new_state, missed_detection=True)
        else:
            # First, compute the state as the weighted sum of the updated states; spawn a new track so we don't impact
            # the existing one
            states = [h.update_track(spawn_new_track=True).curr_state for h in self._hypotheses]
            state_vecs = np.asarray([s.state for s in states])
            updated_state_vec = np.average(state_vecs, axis=0, weights=self._weights)

            # Calculate the covariance
            delta_state = state_vecs - updated_state_vec
            covars = np.stack([s.covar.cov for s in states], axis=2)
            covar = np.sum(covars*self._weights, axis=2) + delta_state.T @ (self._weights[:, np.newaxis] * delta_state)
            updated_state_covar = CovarianceMatrix(covar)

            # Wrap it in a new state
            new_state = State(states[0].state_space, states[0].time, updated_state_vec, updated_state_covar)


            # Append the state
            t.append(new_state)

        # Plot the results
        if ax is not None:
            hdl = t.plot(ax=ax, do_vel=False, do_cov=True, predicted_state=self.predicted_state, marker='^')
            # Plot a dashed line from the current state to the new one
            plt.plot(*zip(t.curr_state.position[plot_dims], new_state.position[plot_dims]),
                     color=hdl[0].get_color(), linestyle=':', label=f'Track {self.track.track_id}, Updated State')
            new_state.plot(ax=ax, do_vel=False, do_cov=True, color=hdl[0].get_color(), linestyle=':')

        return t

    @property
    def predicted_state(self) -> State:
        # Should be the same predicted state for all nested hypotheses; just return one
        return self._hypotheses[0].predicted_state

    @property
    def measurement_prediction(self) -> Measurement:
        return self._hypotheses[0].measurement_prediction

class Association(Mapping):
    """
    Bidirectional mapping between Track objects and Measurement objects, supporting lookup in either direction.
    Implements the ``Mapping`` interface so it can be iterated like a dict (iterates over tracks).
    """
    _tracks_to_measurements: dict[Track, Measurement]
    _measurements_to_tracks: dict[Measurement, Track]

    def __init__(self, assoc: dict):
        """
        :param assoc: A dict mapping either Track→Measurement or Measurement→Track; the reverse mapping
                      is built automatically. Pass None to create an empty Association.
        """
        # Initialize with a dictionary mapping either measurements to tracks or tracks to measurements
        if assoc is None:
            return

        key_types = set(type(k) for k in assoc.keys())
        if Track in key_types:
            self._tracks_to_measurements = assoc
            for k, v in assoc.items():
                self._measurements_to_tracks[v] = k
        elif Measurement in key_types:
            self._measurements_to_tracks = assoc
            for k, v in assoc.items():
                self._tracks_to_measurements[v] = k

    def add_association(self, t: Track, m: Measurement):
        """Register a Track–Measurement pair in both internal lookup dicts."""
        self._tracks_to_measurements[t] = m
        self._measurements_to_tracks[m] = t

    def __getitem__(self, item):
        if isinstance(item, Measurement):
            return self._measurements_to_tracks[item]
        elif isinstance(item, Track):
            return self._tracks_to_measurements[item]
        else:
            raise TypeError('Association can only be indexed by Measurement or Track')

    def __iter__(self):
        return iter(self._tracks_to_measurements)

    def __len__(self):
        return len(self._tracks_to_measurements)

    def __repr__(self):
        return f'{type(self)}({self._tracks_to_measurements})'

    def __str__(self):
        return str(self._tracks_to_measurements)


class Associator(ABC):
    """
    An object that can be used to associate measurements with tracks.

    This abstract class defines the methods required.
    """
    gate_probability: float = None
    motion_model: MotionModel = None

    def __init__(self, motion_model: MotionModel, gate_probability: float=None):
        """
        :param motion_model: MotionModel used to predict each track to the current measurement time
        :param gate_probability: Chi-square gate probability for the acceptance gate (e.g., 0.99);
                                 if None the class-level default is used
        """
        self.motion_model = motion_model
        if gate_probability is not None:
            self.gate_probability = gate_probability

    @abstractmethod
    def associate(self, tracks: list[Track],
                  measurements: list[Measurement],
                  curr_time: float = None)-> tuple[dict[Track, Hypothesis], list[Measurement]]:
        pass



class NNAssociator(Associator):
    """
    Nearest-Neighbor (NN) associator. Assigns each track independently to its closest measurement
    (by Mahalanobis distance squared y'S⁻¹y) that passes the acceptance gate. Earlier tracks take
    priority: a measurement already assigned to a track is unavailable to later tracks.
    If no measurements are provided and ``curr_time`` is given, every track receives a
    MissedDetectionHypothesis coasting it to ``curr_time``.
    """

    def associate(self, tracks: list[Track],
                  measurements: list[Measurement],
                  curr_time: float = None,
                  print_table: bool=False)-> tuple[dict[Track, Hypothesis], list[Measurement]]:
        """
        Run NN association for one scan.

        :param tracks: Active tracks to associate
        :param measurements: New measurements from the current scan
        :param curr_time: Current scan timestamp [seconds]; required when ``measurements`` is empty
                          so that missed-detection hypotheses can be coasted to the right time
        :param print_table: If True, print a distance table to stdout (currently stubbed out)
        :return: Tuple of (track→hypothesis dict, list of unassociated measurements)
        """
        hypotheses = {}
        unassociated_measurements = measurements[:]
        if len(measurements) == 0:
            if curr_time is None or not tracks:
                return hypotheses, unassociated_measurements
            # No measurements: coast every track via a missed-detection hypothesis
            for track in tracks:
                hypotheses[track] = MissedDetectionHypothesis(track=track,
                                                              motion_model=self.motion_model,
                                                              sensor=None,
                                                              time=curr_time,
                                                              gate_probability=self.gate_probability)
            return hypotheses, unassociated_measurements

        curr_time = measurements[0].time

        if print_table:
            pass
            # table = PrettyTable()
            # table.field_names = ['Track'] + [f"Msmt {i}" for i in range(len(measurements))]
            # table.float_format = ".2"

        for track in tracks:
            # Generate a hypothesis for each track; we'll start with the null hypothesis.
            # The null cost must equal the gate threshold so it loses to any measurement that
            # passes the gate (d² < gate_threshold) and wins when all measurements fail.
            null_hypothesis = MissedDetectionHypothesis(track=track,
                                                        motion_model=self.motion_model,
                                                        sensor=measurements[0].sensor,
                                                        time=curr_time,
                                                        gate_probability=self.gate_probability,
                                                        num_msmt_dims=measurements[0].size)

            # There are no more measurements to associate; we need to use the missed detection hypothesis
            if not measurements:
                hypotheses[track] = null_hypothesis
                continue

            # Generate a set of candidate hypotheses
            this_hypotheses = [Hypothesis(track=track, measurement=m, motion_model=self.motion_model) for m in measurements]
            for h in this_hypotheses:
                gate_size = h.compute_gate_size(self.gate_probability)
                # print(f"    Track {track.track_id}: distance={h.distance:.4f}, "
                #       f"gate={gate_size:.4f}")

            if all([h.distance > h.compute_gate_size(self.gate_probability) for h in this_hypotheses]):
                pass

            # print('Generating hypotheses for track ', track.track_id, '...')
            # [print(h) for h in this_hypotheses]
            if print_table:
                pass
                # table.add_row([track.__str__()] + [h.distance for h in this_hypotheses])

            [h.apply_distance_gate(self.gate_probability) for h in this_hypotheses]
            already_associated = [h.measurement for h in hypotheses.values()]
            [h.invalidate() for h in this_hypotheses if h.measurement in already_associated]

            # Compute the Mahalanobis distances and find the best one
            d = np.asarray([h.distance for h in this_hypotheses])


            # Make sure the acceptance gate is valid
            if np.isfinite(np.amin(d)):
                idx = np.argmin(d, axis=None)
                # Add the association, removing it from the list of measurements
                best_hypothesis = this_hypotheses[idx]
                hypotheses[track] = best_hypothesis
                unassociated_measurements.remove(best_hypothesis.measurement)
                # print(f'...NN={best_hypothesis}')
            else:
                # All measurements failed the acceptance gate test;
                # Use the null hypothesis
                hypotheses[track] = null_hypothesis
                # print('...NN=null')

        total_cost = np.sum([h.distance for h in hypotheses.values()])
        if print_table:
            pass
            # print(f"Nearest Neighbor Association Distances (total distance={total_cost:.2f})")
            # print(table)

        # Convert to an Association object and return
        return hypotheses, unassociated_measurements


class GNNAssociator(Associator):
    """
    Global Nearest-Neighbor (GNN) associator. Solves the full assignment problem across all
    tracks and measurements simultaneously using the Hungarian algorithm (Munkres), minimizing
    total Mahalanobis distance. Each track and each measurement is assigned at most once.
    A null (missed-detection) column is added for each track so every track gets a hypothesis
    even if no valid measurement is available.
    If no measurements are provided and ``curr_time`` is given, every track receives a
    MissedDetectionHypothesis.
    """

    def associate(self, tracks: list[Track],
                  measurements: list[Measurement],
                  curr_time: float = None,
                  print_table: bool=False) -> tuple[dict[Track, Hypothesis], list[Measurement]]:
        """
        Run GNN association for one scan.

        :param tracks: Active tracks to associate
        :param measurements: New measurements from the current scan
        :param curr_time: Current scan timestamp [seconds]; required when ``measurements`` is empty
                          so that missed-detection hypotheses can be coasted to the right time
        :param print_table: If True, print a distance table to stdout (currently stubbed out)
        :return: Tuple of (track→hypothesis dict, list of unassociated measurements)
        """
        num_tracks = len(tracks)
        num_measurements = len(measurements)
        if num_tracks == 0:
            return {}, measurements
        if num_measurements == 0:
            if curr_time is None:
                return {}, measurements
            # No measurements: coast every track via a missed-detection hypothesis
            hypotheses = {}
            for track in tracks:
                hypotheses[track] = MissedDetectionHypothesis(track=track,
                                                              motion_model=self.motion_model,
                                                              sensor=None,
                                                              time=curr_time,
                                                              gate_probability=self.gate_probability)
            return hypotheses, []

        curr_time = measurements[0].time

        if print_table:
            pass
            # table = PrettyTable()
            # table.field_names = ['Track'] + [f"Msmt {i}" for i in range(len(measurements))]
            # table.float_format = ".2"

        # Generate the full set of hypotheses and record their distances
        hypotheses = []
        null_hypotheses = []
        large_value = 1e10 # mahalanobis distance for detections that are outside the gate
        distance = np.full((num_tracks, num_measurements + num_tracks), large_value)

        for index, track in enumerate(tracks):
            this_hypotheses = [Hypothesis(track=track, measurement=m, motion_model=self.motion_model) for m in measurements]

            if print_table:
                pass
                # table.add_row([track.__str__()] + [h.distance for h in this_hypotheses])
            [h.apply_distance_gate(self.gate_probability) for h in this_hypotheses]
            for j, h in enumerate(this_hypotheses):
                distance[index, j] = h.distance if np.isfinite(h.distance) else large_value

            # Null hypothesis: cost equals the gate threshold so it loses to any measurement
            # that passes the gate (d² < gate_threshold) and wins when all measurements fail.
            null_hyp = MissedDetectionHypothesis(track=track, motion_model=self.motion_model,
                                                 sensor=measurements[0].sensor,
                                                 time=curr_time,
                                                 gate_probability=self.gate_probability,
                                                 num_msmt_dims=measurements[0].size)
            null_hypotheses.append(null_hyp)
            distance[index, num_measurements + index] = null_hyp.distance

            # Add to the nested list and distance array
            hypotheses.append(this_hypotheses)

        # Convert to a 2D matrix and apply the Munkres Algorithm via scipy.optimize
        row_ind, col_ind = linear_sum_assignment(distance)
        total_cost = distance[row_ind, col_ind].sum()

        # Collect the valid hypotheses
        good_hypotheses = {}
        unassociated_measurements = measurements[:]
        for r, c in zip(row_ind, col_ind):
            if c < num_measurements:
                # Assigned to a real measurement
                good_hypotheses[tracks[r]] = hypotheses[r][c]
                unassociated_measurements.remove(hypotheses[r][c].measurement)
            else:
                # Assigned to a null hypothesis
                good_hypotheses[tracks[r]] = null_hypotheses[r]

        if print_table:
            pass
            # print(f"Global Nearest Neighbor Association Distances (total distance={total_cost:.2f})")
            # print(table)

        return good_hypotheses, unassociated_measurements


class PDAAssociator(Associator):
    """
    Probabilistic Data Association (PDA) associator. For each track, all measurements that pass
    the acceptance gate are retained and combined into a single GMMHypothesis, weighted by their
    normalized likelihoods. A missed-detection hypothesis weighted by (1 − Pd·Pg) is always
    included. The resulting GMMHypothesis reduces to a single Gaussian-mixture state update.
    """
    detection_probability: float = 1.0

    def __init__(self, motion_model: MotionModel, gate_probability: float=None, detection_probability: float=1.0):
        """
        :param motion_model: MotionModel used to predict each track to the measurement time
        :param gate_probability: Chi-square gate probability for the acceptance gate (e.g., 0.99)
        :param detection_probability: Probability that the target produces a measurement (Pd); default 1.0
        """
        super().__init__(motion_model, gate_probability)
        self.detection_probability = detection_probability

    def associate(self, tracks: list[Track],
                  measurements: list[Measurement],
                  curr_time: float = None,
                  print_table: bool=False)-> tuple[dict[Track, GMMHypothesis], list[Measurement]]:
        """
        Run PDA association for one scan.

        :param tracks: Active tracks to associate
        :param measurements: New measurements from the current scan
        :param curr_time: Unused (measurement time is read from measurements[0].time); kept for API consistency
        :param print_table: If True, print a likelihood table to stdout (currently stubbed out)
        :return: Tuple of (track→GMMHypothesis dict, list of unassociated measurements)
        """

        if print_table:
            pass
            # table = PrettyTable()
            # table.field_names = ['Track', 'Miss'] + [f"Msmt {i}" for i in range(len(measurements))]
            # table.float_format = ".2"

        hypotheses = {}
        unassociated_measurements = measurements[:]
        for track in tracks:
            # Initialize a Null Hypothesis
            p_miss = 1 - self.detection_probability*self.gate_probability
            null_hypothesis = MissedDetectionHypothesis(track=track,
                                                        motion_model=self.motion_model,
                                                        sensor=measurements[0].sensor,
                                                        time=measurements[0].time,
                                                        gate_probability=self.gate_probability,
                                                        num_msmt_dims=measurements[0].size,
                                                        distance=p_miss)
            # Generate the full set of hypotheses and apply the acceptance gate
            this_hypotheses = [Hypothesis(track=track, measurement=m, motion_model=self.motion_model) for m in measurements]
            [h.apply_distance_gate(self.gate_probability) for h in this_hypotheses]
            init_likelihoods = [h.likelihood for h in this_hypotheses]

            # Keep only those hypotheses that passed the acceptance gate
            good_hypotheses = [h for h in this_hypotheses if h.is_valid]
            good_hypotheses.append(null_hypothesis) # Add a missed detection hypothesis

            # Remove any measurements that are used in this PDA filter from the set of unassociated ones
            for h in good_hypotheses:
                if h.measurement in unassociated_measurements:
                    unassociated_measurements.remove(h.measurement)

            # Normalize the hypotheses
            likelihoods = np.asarray([h.likelihood for h in good_hypotheses])
            total_wt = np.sum(likelihoods)
            likelihoods /= total_wt

            # Make a compound hypothesis
            this_hypothesis = GMMHypothesis(good_hypotheses, likelihoods, motion_model=self.motion_model)
            hypotheses[track] = this_hypothesis

            if print_table:
                pass
                # table.add_row([track.__str__(), null_hypothesis.likelihood] + init_likelihoods)

        if print_table:
            pass
            # print('PDA Associator Table of Likelihoods')
            # print(table)

        return hypotheses, unassociated_measurements
