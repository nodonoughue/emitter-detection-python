from abc import ABC, abstractmethod

from .measurement import Measurement, MeasurementModel
from .track import Track


class Initiator(ABC):
    @abstractmethod
    def initiate(self, measurements: list[Measurement]) -> list[Track]:
        pass

class SinglePointMeasurementInitiator(Initiator):
    """
    This initiator type accepts a list of measurements and instantiates a track for each one.
    """
    msmt_model: MeasurementModel

    def __init__(self, msmt_model: MeasurementModel):
        self.msmt_model = msmt_model

    def initiate(self, measurements: list[Measurement], track_id: int=0) -> list[Track]:
        tracks = []

        for m in measurements:
            # Determine a position and/or velocity for this measurement
            s = self.msmt_model.state_from_measurement(m)

            # Initialize a track object
            t = Track(initial_state=s, id=track_id)
            track_id += 1
            tracks.append(t)

        return tracks
