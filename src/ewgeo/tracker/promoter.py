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
            # Print statements that are helpful for debugging
            # print(f"  [{t.track_id}] num_updates={t.num_updates}, "
            #       f"len(states)={len(t.states)}, "
            #       f"num_hits={self.num_hits}, "
            #       f"num_chances={self.num_chances}")
            if t.num_updates >= self.num_hits:
                # The track has the required number of hits; promote it
                # print(f"    -> PROMOTE")
                to_promote.append(t)
            elif len(t.states) >= self.num_chances:
                # The track has gotten the required number of chances, and is still tentative.
                # Delete it
                # print(f"    -> DELETE")
                to_delete.append(t)
            # else:
                # print(f"    -> KEEP TENTATIVE")

        return to_promote, to_delete