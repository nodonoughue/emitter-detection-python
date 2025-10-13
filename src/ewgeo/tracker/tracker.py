import numpy.typing as npt

from .measurement import Measurement
from .track import Track

# TODO: Make an Initiator class -- update the text
# TODO: Make a Deleter class -- update the text
# TODO: Make a Promoter class -- update the text

class TrackManager:
    curr_time: float
    tracks: list[Track]
    _tracklets: list[Track]  # not visible outside this class
    deleted_tracks: list[Track]
    deleted_tracklets: list[Track]
    unassociated_measurements: list[Measurement]

    # deleter: Deleter
    # promoter: Promoter
    # initiator: Initiator


    def __init__(self):
        pass

    def associate_to_tracks(self, measurements: list[Measurement]):
        """
        Accept a list of measurements.

        Unassociated measurements are added to the list unassociated_measurements.
        """
        pass

    def initialize_new_tracks(self):
        """
        Form new tracks from unassociated_measurements.

        Any measurements that are used to form a track are removed from the list.
        Any new tracks are added to the list tracklets.
        """
        pass

    def prune_unassociated_measurements(self):
        """
        Remove any measurements that were not used to form a track and have gotten stale.
        """
        pass

    def promote_tracks(self):
        """
        Look for tracklets that should be promoted to full tracks
        """
        pass

    def delete_tracks(self):
        """
        Look for tracks that should be deleted; move them to deleted_tracks.
        """

        # Find tracks to delete
        tracks_to_delete = []
        for track in self.tracks:
            if track.is_stale:
                tracks_to_delete.append(track)

        # Delete them
        for track in tracks_to_delete:
            self.tracks.remove(track)
            self.deleted_tracks.append(track)

        # Do the same for tracklets
        tracks_to_delete = []
        for track in self._tracklets:
            if track.is_stale:
                tracks_to_delete.append(track)

        # Delete them
        for track in tracks_to_delete:
            self._tracklets.remove(track)
            self.deleted_tracklets.append(track)

        return
