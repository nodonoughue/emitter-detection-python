from abc import ABC, abstractmethod

from ewgeo.tracker import Track


class Promoter(ABC):
    """Abstract base class for track promotion logic."""

    @abstractmethod
    def promote(self, tracks: list[Track]) -> tuple[list[Track], list[Track]]:
        pass

class MofNPromoter(Promoter):
    """
    M-of-N promoter: promotes a tentative track to firm status once it accumulates
    at least ``num_hits`` confirmed updates, and drops it if it reaches ``num_chances``
    total states without achieving enough hits.
    """
    num_hits: int
    num_chances: int

    def __init__(self, num_hits: int, num_chances: int):
        """
        :param num_hits: Minimum number of confirmed measurement updates required to promote a track
        :param num_chances: Maximum number of state updates (hits + misses) allowed before a
                            tentative track is discarded if it has not yet been promoted
        """
        self.num_hits = num_hits
        self.num_chances = num_chances

    def promote(self, tracks: list[Track]) -> tuple[list[Track], list[Track]]:
        """
        Evaluate each tentative track and classify it.

        :param tracks: Tentative tracks to evaluate
        :return: Tuple of (tracks to promote to firm status, tracks to discard)
        """
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