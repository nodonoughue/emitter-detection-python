from abc import ABC, abstractmethod
from collections.abc import Mapping

from .measurement import Measurement
from .track import Track

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

class Associator(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def associate(self, measurements: list[Measurement], tracks: list[Track])-> Association:
        pass

# TODO: Association Window / Gating
# TODO: Nearest Neighbor / GNN
# TODO: PDAF / JPDAF