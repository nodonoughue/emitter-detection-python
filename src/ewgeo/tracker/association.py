from abc import ABC, abstractmethod
from collections.abc import Mapping
import numpy as np
from numpy import typing as npt
from scipy.ndimage import measurements
from scipy.stats import chi2
from scipy.optimize import linear_sum_assignment

from .measurement import Measurement
from .track import Track
from ..utils.covariance import CovarianceMatrix


class Hypothesis:
    """
    Based on the Stone Soup class of the same name, a Hypothesis is simply the association
    of a measurement with a track. It consists of three parameters:

    :param measurement: the new measurement that is being incorporated
    :param track: the existing track to which the measurement will be appended
    :param measurement_prediction: the predicted measurement from the associated track
    """
    track: Track                                # Track under test
    measurement: Measurement                    # Measurement to associate to Track
    measurement_prediction: npt.ArrayLike       # Predicted measurement

    def __init__(self, track: Track, measurement: Measurement):
        self.track = track
        self.measurement = measurement
        self.parse()

    def parse(self):
        """
        Parse the hypothesis; predict the track forward to the measurement's timestamp,
        then record the measurement prediction.
        """
        self.track.predict(new_time=self.measurement.time)
        self.measurement_prediction = self.track.pred_measurement

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

    def __init__(self, gate_probability: float=None):
        if gate_probability is not None:
            self.gate_probability = gate_probability

    @abstractmethod
    def associate(self, measurements: list[Measurement], tracks: list[Track])-> Association:
        pass

    def compute_gate_size(self, measurement: Measurement)->float:
        """
        Compute the Acceptance Gate size from the gate probability and the measurement size in the
        assigned track.

        The distance metric lambda will be distributed as a Chi-Square random variable with num_measurements degrees
        of freedom, where num_measurements is the size of the measurement vector.
        """
        chi2_dof = len(measurement.zeta)
        return chi2.ppf(self.gate_probability, chi2_dof)

class NNeighborAssociator(Associator):

    def associate(self, measurements: list[Measurement], tracks: list[Track])-> Association:
        # TODO: Test
        result = {}
        for track in tracks:
            # There are no more measurements to associate; quit
            if not measurements: break

            # Find the best measurement for this track
            d = [track.compute_distance(m) for m in measurements]
            idx = np.argmin(d, axis=None)

            # Make sure the acceptance gate is valid
            if self.gate_probability:
                lam = self.compute_gate_size(measurements[idx])
                if d[idx] <= lam:
                    # Add the association, removing it from the list of measurements
                    result[track] = measurements.pop([idx])
                else:
                    # The best measurement failed the acceptance gate test;
                    # don't add any measurements to the result for this track.
                    #
                    # It needs to coast.
                    pass

        # Convert to an Association object and return
        return Association(result)


class GNNAssociator(Associator):

    def associate(self, measurements: list[Measurement], tracks: list[Track])-> Association:
        # TODO: Test
        # Compute all the distances; it's a nested list that we convert to a 2d array
        d = np.asarray([t.compute_distance(m) for t,m in zip(tracks, measurements)])

        # Apply the acceptance gate, if defined
        if self.gate_probability:
            # Set any measurements that fall outside a track's acceptance gate to np.inf
            lam = self.compute_gate_size(measurements[0])
            d[d>lam] = np.inf

        # Convert to a 2D matrix and apply the Munkres Algorithm via scipy.optimize
        row_ind, col_ind = linear_sum_assignment(d)

        # Parse result and make an Association Object
        result = {}
        for r, c in zip(row_ind, col_ind):
            result[tracks[r]] = measurements[c]

        return Association(result)


# TODO: PDAF / JPDAF
class PDAFAssociator(Associator):

    def hypothesize(self, tracks: list[Track], measurements: list[Measurement]):
        hypotheses = []
        for track in tracks:
            # Check the Acceptance Gate
            if self.gate_probability:
                lam = self.compute_gate_size(measurements[-1])
                valid_measurements = [m for m in measurements if track.compute_distance(m) <= lam]
            else:
                valid_measurements = measurements

            # Generate Hypotheses
            track_hypotheses = track.hypothesize(valid_measurements)

            # Normalize the hypotheses
            total_wt = np.sum(np.array([h.likelihood for h in track_hypotheses]))
            for h in track_hypotheses:
                h.likelihood = h.likelihood / total_wt




