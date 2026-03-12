from abc import ABC, abstractmethod

from ewgeo.tracker import Track


class Deleter(ABC):
    """Abstract base class for track deletion logic."""

    @abstractmethod
    def delete(self, tracks: list[Track]) -> list[Track]:
        pass

class MissedDetectionDeleter(Deleter):
    """
    Deletes tracks that have accumulated more than ``num_missed_detections`` consecutive
    missed detections (i.e., coast steps with no measurement update).
    """
    num_missed_detections: int

    def __init__(self, num_missed_detections: int):
        """
        :param num_missed_detections: Maximum number of consecutive missed detections allowed;
                                      tracks exceeding this threshold are returned for deletion
        """
        self.num_missed_detections = num_missed_detections

    def delete(self, tracks: list[Track]) -> list[Track]:
        """
        :param tracks: Tracks to evaluate
        :return: List of tracks whose consecutive missed-detection count exceeds the threshold
        """
        return [t for t in tracks if t.num_missed_detections > self.num_missed_detections]