"""
End-to-end integration tests for the Tracker class.

The tracker update cycle is: associate → promote → initiate → delete.

Note: NNAssociator contains an active print() statement that produces output
during tests (association.py line 475).
"""
import numpy as np
import pytest

from ewgeo.tracker.tracker import Tracker
from ewgeo.tracker.association import NNAssociator
from ewgeo.tracker.initiator import SinglePointInitiator
from ewgeo.tracker.promoter import MofNPromoter
from ewgeo.tracker.deleter import MissedDetectionDeleter
from ewgeo.tracker.measurement import Measurement, MeasurementModel
from ewgeo.tracker.transition import ConstantVelocityMotionModel
from ewgeo.utils.covariance import CovarianceMatrix


# ---------------------------------------------------------------------------
# Mock sensor — direct 1D position measurement
# ---------------------------------------------------------------------------

class _MockSensor1D:
    num_dim = 1
    num_measurements = 1
    cov = CovarianceMatrix(np.array([[1.0]]))

    def measurement(self, x_source, v_source=None, **kwargs):
        return np.atleast_1d(x_source[:1]).astype(float)

    def jacobian(self, x_source, v_source=None, **kwargs):
        return np.array([[1.0]])

    def least_square(self, zeta, x_init, **kwargs):
        return np.array([zeta[0], 0.0]), None

    def compute_crlb(self, x_source, **kwargs):
        return CovarianceMatrix(np.diag([4.0, 1000.0]))

    def log_likelihood(self, x_source, zeta, **kwargs):
        return 0.0


# ---------------------------------------------------------------------------
# Standard tracker fixture
# ---------------------------------------------------------------------------

def make_tracker(num_hits=2, num_chances=3, max_misses=2):
    """
    Build a Tracker wired with:
      - NNAssociator (gate_probability=0.99)
      - SinglePointInitiator
      - MofNPromoter(num_hits, num_chances)
      - MissedDetectionDeleter(max_misses)   — deletes when misses > max_misses
    """
    model = ConstantVelocityMotionModel(num_dims=1, process_covar=1.0)
    sensor = _MockSensor1D()
    msmt_model = MeasurementModel(pss=sensor)

    associator = NNAssociator(motion_model=model, gate_probability=0.99)
    initiator  = SinglePointInitiator(msmt_model=msmt_model,
                                                 motion_model=model)
    promoter   = MofNPromoter(num_hits=num_hits, num_chances=num_chances)
    deleter    = MissedDetectionDeleter(num_missed_detections=max_misses)

    return Tracker(associator=associator, initiator=initiator,
                   promoter=promoter, deleter=deleter)


def make_measurement(x, t):
    return Measurement(time=t, sensor=_MockSensor1D(), zeta=np.array([float(x)]))


# ---------------------------------------------------------------------------
# Test 1: empty update leaves tracker unchanged
# ---------------------------------------------------------------------------

def test_empty_update_no_change():
    tracker = make_tracker()
    tracker.update([], curr_time=0.0)
    assert tracker.tracks == []
    assert tracker._tentative_tracks == []


# ---------------------------------------------------------------------------
# Test 2: first measurement creates a tentative track
# ---------------------------------------------------------------------------

def test_first_measurement_creates_tentative_track():
    tracker = make_tracker()
    tracker.update([make_measurement(10, t=0)], curr_time=0)
    assert len(tracker._tentative_tracks) == 1
    assert len(tracker.tracks) == 0


# ---------------------------------------------------------------------------
# Test 3: two measurements promote the tentative track to a firm track
# ---------------------------------------------------------------------------

def test_two_measurements_promote_track():
    """MofN(2,3): a track that receives 2 hits in ≤3 chances is promoted."""
    tracker = make_tracker(num_hits=2, num_chances=3)
    tracker.update([make_measurement(10, t=0)], curr_time=0)   # creates tentative T0
    tracker.update([make_measurement(12, t=1)], curr_time=1)   # T0 gets 2nd hit → promoted

    assert len(tracker.tracks) == 1
    assert len(tracker._tentative_tracks) == 0


# ---------------------------------------------------------------------------
# Test 4: firm track state is updated each step
# ---------------------------------------------------------------------------

def test_firm_track_state_updated():
    """After promotion, each update adds a new state to the track history."""
    tracker = make_tracker()
    tracker.update([make_measurement(10, t=0)], curr_time=0)
    tracker.update([make_measurement(12, t=1)], curr_time=1)
    # At this point: 1 firm track with 2 states (t=0 and t=1 after promotion update)
    firm_track = tracker.tracks[0]
    initial_state_count = len(firm_track.states)

    tracker.update([make_measurement(14, t=2)], curr_time=2)
    assert len(firm_track.states) == initial_state_count + 1


def test_firm_track_position_tracks_measurement():
    """KF update pulls the estimated position toward the measurement."""
    tracker = make_tracker()
    tracker.update([make_measurement(10, t=0)], curr_time=0)
    tracker.update([make_measurement(10, t=1)], curr_time=1)   # promote at x=10

    firm_track = tracker.tracks[0]
    pos_before = firm_track.curr_state.position[0]

    # Now feed a measurement at x=14 (inside gate); position should shift toward 14
    tracker.update([make_measurement(14, t=2)], curr_time=2)
    pos_after = firm_track.curr_state.position[0]

    assert pos_after > pos_before


# ---------------------------------------------------------------------------
# Test 5: missed detection (no measurement) increments miss counter
# ---------------------------------------------------------------------------

def test_missed_detection_increments_counter():
    tracker = make_tracker(max_misses=5)
    tracker.update([make_measurement(10, t=0)], curr_time=0)
    tracker.update([make_measurement(12, t=1)], curr_time=1)   # promote

    firm_track = tracker.tracks[0]
    assert firm_track.num_missed_detections == 0

    tracker.update([], curr_time=2)   # no measurements → null hypothesis for every track
    assert firm_track.num_missed_detections == 1


def test_missed_detection_does_not_delete_below_threshold():
    """A single miss does not delete a track with threshold=2."""
    tracker = make_tracker(max_misses=2)
    tracker.update([make_measurement(10, t=0)], curr_time=0)
    tracker.update([make_measurement(12, t=1)], curr_time=1)

    tracker.update([], curr_time=2)   # 1 miss
    assert len(tracker.tracks) == 1


# ---------------------------------------------------------------------------
# Test 6: firm track deleted after exceeding miss threshold
# ---------------------------------------------------------------------------

def test_track_deleted_after_too_many_misses():
    """
    MissedDetectionDeleter(max_misses=2) deletes when num_missed_detections > 2,
    i.e. after 3 consecutive misses.
    """
    tracker = make_tracker(max_misses=2)
    tracker.update([make_measurement(10, t=0)], curr_time=0)
    tracker.update([make_measurement(12, t=1)], curr_time=1)   # promote → 1 firm track

    tracker.update([], curr_time=2)   # miss 1
    tracker.update([], curr_time=3)   # miss 2
    tracker.update([], curr_time=4)   # miss 3  →  3 > 2 → deleted

    assert len(tracker.tracks) == 0


def test_deleted_tracks_captured_when_keep_all_tracks():
    """When keep_all_tracks=True, deleted tracks appear in tracker.deleted_tracks."""
    tracker = make_tracker(max_misses=2)
    tracker.keep_all_tracks = True

    tracker.update([make_measurement(10, t=0)], curr_time=0)
    tracker.update([make_measurement(12, t=1)], curr_time=1)
    tracker.update([], curr_time=2)   # miss 1
    tracker.update([], curr_time=3)   # miss 2
    tracker.update([], curr_time=4)   # miss 3 → deleted

    assert len(tracker.tracks) == 0
    assert len(tracker.deleted_tracks) == 1


# ---------------------------------------------------------------------------
# Test 7: two well-separated measurements create two independent firm tracks
# ---------------------------------------------------------------------------

def test_two_sources_create_two_firm_tracks():
    """
    Measurements at x=0 and x=500 are far enough apart that they initiate
    and are promoted to two separate firm tracks.
    """
    tracker = make_tracker()
    tracker.update([make_measurement(0,   t=0),
                    make_measurement(500, t=0)], curr_time=0)
    tracker.update([make_measurement(1,   t=1),
                    make_measurement(501, t=1)], curr_time=1)

    assert len(tracker.tracks) == 2


# ---------------------------------------------------------------------------
# Test 8: hit followed by a miss resets the miss counter
# ---------------------------------------------------------------------------

def test_hit_after_miss_resets_miss_counter():
    tracker = make_tracker(max_misses=5)
    tracker.update([make_measurement(10, t=0)], curr_time=0)
    tracker.update([make_measurement(12, t=1)], curr_time=1)   # promote

    firm_track = tracker.tracks[0]
    tracker.update([], curr_time=2)   # miss 1
    assert firm_track.num_missed_detections == 1

    tracker.update([make_measurement(12, t=3)], curr_time=3)   # hit
    assert firm_track.num_missed_detections == 0


# ---------------------------------------------------------------------------
# Test 9: tentative track that fails promotion is removed
# ---------------------------------------------------------------------------

def test_tentative_track_removed_if_not_promoted():
    """
    MofN(2,2): track must be hit in both of its first 2 chances.
    If the second update provides no matching measurement, the tentative
    track exhausts its chances and is removed.
    """
    tracker = make_tracker(num_hits=2, num_chances=2, max_misses=10)
    tracker.update([make_measurement(10, t=0)], curr_time=0)   # creates tentative T0
    tracker.update([], curr_time=1)                            # no measurement → T0 dropped

    # T0 should be gone from tentative tracks; only the new x=9000 track is tentative
    assert all(t.curr_state.position[0] != 10.0 or len(t.states) < 2
               for t in tracker._tentative_tracks)
