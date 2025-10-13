from abc import ABC, abstractmethod
from collections.abc import Mapping
import numpy as np
from matplotlib import pyplot as plt
from numpy import typing as npt
from prettytable import PrettyTable
from scipy.stats import chi2, multivariate_normal
from scipy.optimize import linear_sum_assignment

from . import State
from .measurement import Measurement, MeasurementModel
from .track import Track
from .transition import MotionModel
from ..utils.covariance import CovarianceMatrix

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
    _innov: npt.ArrayLike | None = None
    _innov_covar: CovarianceMatrix | None = None
    _likelihood: float | None = None
    _log_likelihood: float | None = None

    def __init__(self, track: Track, measurement: Measurement | None, motion_model: MotionModel | None = None):
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
    def innovation(self) -> npt.ArrayLike:
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
                # Mahalanobis Distance
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

    def clear_dependent_parameters(self):
        self._distance = None
        self._pred = None
        self._msmt_pred = None
        self._innov = None
        self._innov_covar = None
        self._likelihood = None
        self._log_likelihood = None
        self._is_valid = True

    def override_likelihood(self, likelihood: float):
        self._likelihood = likelihood
        self._log_likelihood = np.log(likelihood)

    def update_track(self, spawn_new_track: bool=False, ax: plt.Axes=None, plot_dims: slice=np.s_[:])->Track:
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

        # Make a new State object and add it to the track
        new_state = t.curr_state.copy(state=new_state_vec, covar=CovarianceMatrix(new_state_covar), time=self.measurement.time)
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
    def __init__(self, track: Track, measurement_model: MeasurementModel, motion_model: MotionModel, likelihood: float, time: float):
        dummy_measurement = Measurement(sensor=measurement_model.pss, time=time, zeta=np.zeros(measurement_model.num_measurement_dimensions, ))
        super().__init__(track, dummy_measurement, motion_model)

        # The innovation of a missed detection is zero, set some dummy values for likelihood, as well
        self._likelihood = likelihood
        self._log_likelihood = np.log(likelihood)
        self._innov = np.zeros_like(track.curr_state.state)
        self._innov_covar = self.predicted_state.covar.copy()

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

        # Update the Estimate
        new_state_vec = self.predicted_state.state + kalman_gain @ self.innovation
        new_state_covar = (np.eye(t.curr_state.size) - kalman_gain @ measurement_jacobian) @ prediction_state_covar

        # Make a copy of the predicted state, and add it to the track
        new_state = self.predicted_state.copy()

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
    weight associated.
    """
    _hypotheses: list[Hypothesis]
    _weights: npt.ArrayLike

    def __init__(self, hypotheses: list[Hypothesis], weights: npt.ArrayLike = None, motion_model: MotionModel=None):
        super().__init__(track=hypotheses[0].track, measurement=None)
        self._hypotheses = hypotheses
        self._weights = weights if weights is not None else np.ones(len(hypotheses))

        self.reduce()

    def reduce(self):
        """
        Reduce the Gaussian Mixture of hypotheses to a single hypothesis and return that.
        """

        # First, compute the innovation as the weighted sum of innovations
        innovations = np.asarray([h.innovation for h in self._hypotheses])
        self._innov = np.average(innovations, axis=0, weights=self._weights)

        # Calculate the covariance
        delta_innov = innovations - self._innov
        covars = np.stack([h.innovation_covar.cov for h in self._hypotheses], axis=2)
        covar = np.sum(covars*self._weights, axis=2) + self._weights * delta_innov @ delta_innov.T
        self._innov_covar = CovarianceMatrix(covar)

        # Make a dummy measurement -- this is needed so that update_track will function
        new_state = State(state_space=self._hypotheses[0].track.state_space,
                          time=self._hypotheses[0].measurement.time,
                          state=self.predicted_state.state + self._innov,
                          covar=self._innov_covar)
        self._measurement = Measurement(sensor=None, time=self._hypotheses[0].measurement.time,
                                        zeta=self._measurement_model.measurement(new_state))

    @property
    def predicted_state(self) -> State:
        # Should be the same predicted state for all nested hypotheses; just return one
        return self._hypotheses[0].predicted_state

    @property
    def measurement_prediction(self) -> Measurement:
        return self._hypotheses[0].measurement_prediction

class Association(Mapping):
    _tracks_to_measurements: dict[Track, Measurement]
    _measurements_to_tracks: dict[Measurement, Track]

    def __init__(self, assoc: dict):
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
        self.motion_model = motion_model
        if gate_probability is not None:
            self.gate_probability = gate_probability

    @abstractmethod
    def associate(self, measurements: list[Measurement], tracks: list[Track])-> dict[Track, Hypothesis]:
        pass



class NNAssociator(Associator):

    def associate(self, tracks: list[Track], measurements: list[Measurement], print_table: bool=False)-> dict[Track, Hypothesis]:
        # TODO: Test
        hypotheses = {}

        if print_table:
            table = PrettyTable()
            table.field_names = ['Track'] + [f"Msmt {i}" for i in range(len(measurements))]
            table.float_format = ".2"

        for track in tracks:
            # Generate a hypothesis for each track; we'll start with the null hypothesis
            null_hypothesis = Hypothesis(track=track, measurement=None)

            # There are no more measurements to associate; we need to use the missed detection hypothesis
            if not measurements:
                hypotheses[track] = null_hypothesis
                continue

            # Generate a set of candidate hypotheses
            this_hypotheses = [Hypothesis(track=track, measurement=m, motion_model=self.motion_model) for m in measurements]
            # print('Generating hypotheses for track ', track.track_id, '...')
            # [print(h) for h in this_hypotheses]
            if print_table:
                table.add_row([track.__str__()] + [h.distance for h in this_hypotheses])

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
                # print(f'...NN={best_hypothesis}')
            else:
                # All measurements failed the acceptance gate test;
                # Use the null hypothesis
                hypotheses[track] = null_hypothesis
                # print('...NN=null')

        total_cost = np.sum([h.distance for h in hypotheses.values()])
        if print_table:
            print(f"Nearest Neighbor Association Distances (total distance={total_cost:.2f})")
            print(table)

        # Convert to an Association object and return
        return hypotheses


class GNNAssociator(Associator):

    def associate(self, measurements: list[Measurement], tracks: list[Track], print_table: bool=False)-> dict[Track, Hypothesis]:
        # TODO: Test

        if print_table:
            table = PrettyTable()
            table.field_names = ['Track'] + [f"Msmt {i}" for i in range(len(measurements))]
            table.float_format = ".2"

        # Generate the full set of hypotheses and record their distances
        hypotheses = []
        distance = np.zeros((len(tracks), len(measurements)+1))
        for index, track in enumerate(tracks):
            this_hypotheses = [Hypothesis(track=track, measurement=m, motion_model=self.motion_model) for m in measurements]

            if print_table:
                table.add_row([track.__str__()] + [h.distance for h in this_hypotheses])
            [h.apply_distance_gate(self.gate_probability) for h in this_hypotheses]
            this_distance = [h.distance for h in this_hypotheses]

            # Add a null hypothesis, set its distance to 1 more than the max (finite) value in the array
            this_hypotheses.append(Hypothesis(track=track, measurement=None))
            this_distance.append(1+np.max(this_distance, initial=0.0, where=np.isfinite(this_distance)))

            # Add to the nested list and distance array
            hypotheses.append(this_hypotheses)
            distance[index] = np.asarray(this_distance)

        # Convert to a 2D matrix and apply the Munkres Algorithm via scipy.optimize
        row_ind, col_ind = linear_sum_assignment(distance)
        total_cost = distance[row_ind, col_ind].sum()

        # Collect the valid hypotheses
        good_hypotheses = {}
        if print_table:
            data_rows = table.rows

        for r, c in zip(row_ind, col_ind):
            good_hypotheses[tracks[r]] = hypotheses[r][c]

        if print_table:
            print(f"Global Nearest Neighbor Association Distances (total distance={total_cost:.2f})")
            print(table)

        return good_hypotheses


class PDAAssociator(Associator):
    detection_probability: float = 1.0

    def __init__(self, motion_model: MotionModel, gate_probability: float=None, detection_probability: float=1.0):
        super().__init__(motion_model, gate_probability)
        self.detection_probability = detection_probability
        
    def associate(self, tracks: list[Track], measurements: list[Measurement], print_table: bool=False) -> dict[Track, GMMHypothesis] :
        if print_table:
            table = PrettyTable()
            table.field_names = ['Track', 'Miss'] + [f"Msmt {i}" for i in range(len(measurements))]
            table.float_format = ".2"

        hypotheses = {}
        for track in tracks:
            # Generate the full set of hypotheses and apply the acceptance gate
            this_hypotheses = [Hypothesis(track=track, measurement=m, motion_model=self.motion_model) for m in measurements]
            [h.apply_distance_gate(self.gate_probability) for h in this_hypotheses]
            init_likelihoods = [h.likelihood for h in this_hypotheses]

            # Keep only those hypotheses that passed the acceptance gate
            good_hypotheses = [h for h in this_hypotheses if h.is_valid]
            null_hypothesis = Hypothesis(track=track, measurement=None)
            null_hypothesis.override_likelihood(1 - self.detection_probability*self.gate_probability)
            good_hypotheses.append(null_hypothesis) # Add a missed detection hypothesis

            # Normalize the hypotheses
            likelihoods = np.asarray([h.likelihood for h in good_hypotheses])
            total_wt = np.sum(likelihoods)
            likelihoods /= total_wt

            # Make a compound hypothesis
            this_hypothesis = GMMHypothesis(good_hypotheses, likelihoods, motion_model=self.motion_model)
            hypotheses[track] = this_hypothesis

            if print_table:
                table.add_row([track.__str__(), null_hypothesis.likelihood] + init_likelihoods)

        if print_table:
            print('PDA Associator Table of Likelihoods')
            print(table)

        return hypotheses
