from ewgeo.tracker import Track
from ewgeo.tracker.promoter import MofNPromoter


def make_track(num_updates, num_states, track_id="t"):
    """Build a minimal Track with controlled num_updates and len(states)."""
    t = object.__new__(Track)
    t.track_id = track_id
    t.num_updates = num_updates
    t.states = [None] * num_states
    return t


def test_promote_when_enough_hits():
    """A track with num_updates clearly above num_hits is promoted."""
    promoter = MofNPromoter(num_hits=3, num_chances=5)
    t = make_track(num_updates=4, num_states=4)
    to_promote, to_delete = promoter.promote([t])
    assert t in to_promote
    assert t not in to_delete


def test_delete_when_out_of_chances():
    """A track that has exhausted its chances without enough hits is deleted."""
    promoter = MofNPromoter(num_hits=3, num_chances=5)
    t = make_track(num_updates=1, num_states=6)
    to_promote, to_delete = promoter.promote([t])
    assert t in to_delete
    assert t not in to_promote


def test_keep_tentative():
    """A track with too few hits and chances remaining is kept (neither promoted nor deleted)."""
    promoter = MofNPromoter(num_hits=3, num_chances=5)
    t = make_track(num_updates=1, num_states=2)
    to_promote, to_delete = promoter.promote([t])
    assert t not in to_promote
    assert t not in to_delete


def test_promote_at_exact_boundary():
    """num_updates == num_hits exactly should promote (uses >=, not >)."""
    promoter = MofNPromoter(num_hits=3, num_chances=5)
    t = make_track(num_updates=3, num_states=3)
    to_promote, to_delete = promoter.promote([t])
    assert t in to_promote
    assert t not in to_delete


def test_delete_at_exact_boundary():
    """len(states) == num_chances exactly should delete (uses >=, not >)."""
    promoter = MofNPromoter(num_hits=3, num_chances=5)
    t = make_track(num_updates=1, num_states=5)
    to_promote, to_delete = promoter.promote([t])
    assert t in to_delete
    assert t not in to_promote


def test_promote_takes_priority():
    """A track meeting both thresholds is promoted, not deleted (if/elif ordering)."""
    promoter = MofNPromoter(num_hits=3, num_chances=5)
    t = make_track(num_updates=3, num_states=5)
    to_promote, to_delete = promoter.promote([t])
    assert t in to_promote
    assert t not in to_delete


def test_mixed_tracks():
    """One promote, one delete, one keep — all routed correctly in a single call."""
    promoter = MofNPromoter(num_hits=3, num_chances=5)
    t_promote = make_track(num_updates=3, num_states=3, track_id="promote")
    t_delete  = make_track(num_updates=1, num_states=5, track_id="delete")
    t_keep    = make_track(num_updates=1, num_states=2, track_id="keep")
    to_promote, to_delete = promoter.promote([t_promote, t_delete, t_keep])
    assert to_promote == [t_promote]
    assert to_delete  == [t_delete]
    assert t_keep not in to_promote
    assert t_keep not in to_delete


def test_empty_input():
    """An empty track list returns two empty lists."""
    promoter = MofNPromoter(num_hits=3, num_chances=5)
    to_promote, to_delete = promoter.promote([])
    assert to_promote == []
    assert to_delete  == []
