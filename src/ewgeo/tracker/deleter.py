from abc import ABC, abstractmethod

from ewgeo.tracker import Track


class Deleter(ABC):
    @abstractmethod
    def delete(self, tracks: list[Track]) -> list[Track]:
        pass

class MissedDetectionDeleter(Deleter):
    num_missed_detections: int

    def __init__(self, num_missed_detections: int):
        self.num_missed_detections = num_missed_detections

    def delete(self, tracks: list[Track]) -> list[Track]:
        return [t for t in tracks if t.num_missed_detections > self.num_missed_detections]