from abc import ABC, abstractmethod

from .association import Associator, MissedDetectionHypothesis
from .measurement import Measurement
from .track import Track

class Initiator(ABC):
    @abstractmethod
    def initiate(self, measurements: list[Measurement]) -> list[Track]:
        pass

class Promoter(ABC):
    @abstractmethod
    def promote(self, tracks: list[Track]) -> list[Track]:
        pass

class Deleter(ABC):
    @abstractmethod
    def delete(self, tracks: list[Track]) -> list[Track]:
        pass

class TrackManager:
    # Parameters
    keep_all_tracks: bool = False
    deleter = None
    initiator = None
    promoter = None
    associator: Associator

    # Track/State Properties
    tracks: list[Track]
    _tentative_tracks: list[Track]  # not visible outside this class
    deleted_tracks: list[Track]     # only populated if keep_all_tracks is True

    def __init__(self, associator: Associator, initiator, promoter, deleter, keep_all_tracks: bool=False):
        self.associator = associator
        self.initiator = initiator
        self.promoter = promoter
        self.deleter = deleter
        self.keep_all_tracks = keep_all_tracks
        self.deleted_tracks = []
        self.tracks = []
        self._tentative_tracks = []

    @property
    def all_tracks(self):
        return self.tracks + self.deleted_tracks

    def execute(self, measurements, reuse_measurements: bool=False):

        # Associate measurements with existing tracks
        # Returns any unused measurements
        unassociated_measurements = self.update(measurements=measurements)

        # Associate measurements with tentative tracks, look for any that can be promoted
        if reuse_measurements:
            # Pass the full set of measurements to the tentative tracks
            measurements_for_tentative_tracks = measurements[:]
        else:
            # Only pass those measurements that were not used to update firm tracks
            measurements_for_tentative_tracks = unassociated_measurements[:]
        unassociated_measurements_2 = self.promote(measurements=measurements_for_tentative_tracks)

        # Create new tracks with unused measurements; only include those that did not get used for either
        # firm or tentative tracks
        measurements_for_new_tracks = [m for m in unassociated_measurements if m in unassociated_measurements_2]
        self.initiate(measurements=measurements_for_new_tracks)

        # Delete tracks
        self.delete()

    def update(self, measurements: list[Measurement]) -> list[Measurement]:

        # Generate hypotheses
        hypothesis_dict, unassoc_msmts = self.associator.associate(tracks=self.tracks, measurements=measurements)

        # Update the hypotheses
        hypotheses = hypothesis_dict.values()
        [h.update_track() for h in hypotheses]

        # Return the unused measurements
        return unassoc_msmts

    def promote(self, measurements: list[Measurement]) -> list[Measurement]:
        # Generate hypotheses to match measurements to the tentative tracks
        hypothesis_dict, unassoc_msmt = self.associator.associate(tracks=self._tentative_tracks,
                                                                  measurements=measurements)

        # Update the tracks associated with these hypotheses
        tentative_hypotheses = hypothesis_dict.values()
        [h.update_track() for h in tentative_hypotheses]

        # Any hypotheses that are not a MissedDetectionHypothesis can be passed to
        # the promoter for evaluation
        tracks_to_test = [h.track for h in tentative_hypotheses if not isinstance(h, MissedDetectionHypothesis)]
        tracks_to_promote = self.promoter.promote(tracks=tracks_to_test)

        # Add the promoted tracks to the track list and remove them from the tentative tracks list
        for t in tracks_to_promote:
            self.tracks.append(t)
            self._tentative_tracks.remove(t)

        return unassoc_msmt

    def initiate(self, measurements: list[Measurement]):
        new_tracks = self.initiator.initiate(measurements=measurements)
        self._tentative_tracks.extend(new_tracks)

    def delete(self):
        # Test the firm tracks
        tracks_to_delete = self.deleter.delete(tracks=self.tracks)

        # Remove the tracks by creating a new list that excludes them
        self.tracks = [t for t in self.tracks if t not in tracks_to_delete]

        if self.keep_all_tracks:
            self.deleted_tracks.extend(tracks_to_delete)

        # Repeat with the tentative tracks
        tracks_to_delete = self.deleter.delete(tracks=self._tentative_tracks)

        # Remove the tracks by creating a new list that excludes them
        self._tentative_tracks = [t for t in self._tentative_tracks if t not in tracks_to_delete]

        if self.keep_all_tracks:
            self.deleted_tracks.extend(tracks_to_delete)

        return
