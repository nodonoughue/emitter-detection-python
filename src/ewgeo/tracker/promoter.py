from abc import ABC, abstractmethod

from ewgeo.tracker import Track


class Promoter(ABC):
    @abstractmethod
    def promote(self, tracks: list[Track]) -> tuple[list[Track], list[Track]]:
        pass

class MofNPromoter(Promoter):
    num_hits: int
    num_chances: int

    def __init__(self, num_hits: int, num_chances: int):
        self.num_hits = num_hits
        self.num_chances = num_chances

    def promote(self, tracks: list[Track]) -> tuple[list[Track], list[Track]]:
        to_promote = []
        to_delete = []

        for t in tracks:
            if t.num_hits >= self.num_hits:
                # The track has the required number of hits; promote it
                to_promote.append(t)
            elif len(t.states) >= self.num_chances:
                # The track has gotten the required number of chances, and is still tentative.
                # Delete it
                to_delete.append(t)

        return to_promote, to_delete