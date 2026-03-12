from ewgeo.tracker import Track
from ewgeo.tracker.deleter import MissedDetectionDeleter


def make_track(num_missed_detections, track_id="t"):
    """Build a minimal Track with a controlled num_missed_detections count."""
    t = object.__new__(Track)
    t.track_id = track_id
    t.num_missed_detections = num_missed_detections
    return t


def test_delete_above_threshold():
    """A track with num_missed_detections clearly above the threshold is returned."""
    deleter = MissedDetectionDeleter(num_missed_detections=3)
    t = make_track(num_missed_detections=5)
    to_delete = deleter.delete([t])
    assert t in to_delete


def test_keep_at_exact_threshold():
    """A track at exactly the threshold is NOT deleted (uses >, not >=)."""
    deleter = MissedDetectionDeleter(num_missed_detections=3)
    t = make_track(num_missed_detections=3)
    to_delete = deleter.delete([t])
    assert t not in to_delete


def test_keep_below_threshold():
    """A track below the threshold is not returned."""
    deleter = MissedDetectionDeleter(num_missed_detections=3)
    t = make_track(num_missed_detections=1)
    to_delete = deleter.delete([t])
    assert t not in to_delete


def test_mixed_tracks():
    """Only the track above the threshold is returned; at-threshold and below are kept."""
    deleter = MissedDetectionDeleter(num_missed_detections=3)
    t_above = make_track(num_missed_detections=4, track_id="above")
    t_at    = make_track(num_missed_detections=3, track_id="at")
    t_below = make_track(num_missed_detections=1, track_id="below")
    to_delete = deleter.delete([t_above, t_at, t_below])
    assert to_delete == [t_above]


def test_empty_input():
    """An empty track list returns an empty list."""
    deleter = MissedDetectionDeleter(num_missed_detections=3)
    assert deleter.delete([]) == []


def test_all_survive():
    """All tracks below threshold — nothing is returned."""
    deleter = MissedDetectionDeleter(num_missed_detections=3)
    tracks = [make_track(i, track_id=str(i)) for i in range(3)]
    assert deleter.delete(tracks) == []


def test_all_deleted():
    """All tracks above threshold — all are returned."""
    deleter = MissedDetectionDeleter(num_missed_detections=3)
    tracks = [make_track(i, track_id=str(i)) for i in range(4, 7)]
    to_delete = deleter.delete(tracks)
    assert to_delete == tracks
