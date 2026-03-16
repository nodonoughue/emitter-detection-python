import numpy as np

from ewgeo.tracker.track import Track
from ewgeo.tracker.states import State
from ewgeo.tracker.transition import ConstantVelocityMotionModel
from ewgeo.utils.covariance import CovarianceMatrix


def equal_to_tolerance(x, y, tol=1e-10):
    return np.all(np.fabs(np.array(x) - np.array(y)) < tol)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_state(x, vx=0.0, t=0.0):
    model = ConstantVelocityMotionModel(num_dims=1, process_covar=1.0)
    return State(model.state_space, time=t,
                 state=np.array([x, vx], dtype=float),
                 covar=CovarianceMatrix(np.eye(2)))


def make_track(x=0.0, vx=0.0, t=0.0, track_id='T0'):
    return Track(initial_state=make_state(x, vx, t), track_id=track_id)


# ===========================================================================
# __init__ and basic properties
# ===========================================================================

def test_track_init_sets_states_list():
    """initial_state keyword populates states with one element."""
    track = make_track(x=5.0, vx=1.0, t=0.0)
    assert len(track.states) == 1


def test_track_init_sets_num_updates():
    """num_updates starts at 1 when initial_state is supplied."""
    track = make_track()
    assert track.num_updates == 1


def test_track_init_sets_num_missed_to_zero():
    """Missed-detection counter starts at zero."""
    track = make_track()
    assert track.num_missed_detections == 0


def test_track_curr_state_is_last():
    """curr_state returns the most-recently-appended state."""
    track = make_track(x=5.0, vx=1.0)
    s2 = make_state(10.0, 2.0, t=1.0)
    track.states.append(s2)
    assert track.curr_state is s2


def test_track_curr_time_matches_last_state():
    """curr_time is the timestamp of the current state."""
    track = make_track(x=5.0, vx=1.0, t=3.5)
    assert track.curr_time == 3.5


def test_track_state_space_delegates_to_curr_state():
    """state_space property returns the state_space of curr_state."""
    track = make_track()
    assert track.state_space is track.curr_state.state_space


def test_track_num_dims():
    """num_dims reports the spatial dimensionality (1 for a 1D CV model)."""
    track = make_track()
    assert track.num_dims == 1


# ===========================================================================
# append
# ===========================================================================

def test_track_append_hit_increments_num_updates():
    """A detected update increments num_updates."""
    track = make_track()
    track.append(make_state(1.0, t=1.0), missed_detection=False)
    assert track.num_updates == 2


def test_track_append_hit_resets_miss_counter():
    """A hit after a miss resets num_missed_detections to zero."""
    track = make_track()
    track.append(make_state(1.0, t=1.0), missed_detection=True)
    assert track.num_missed_detections == 1
    track.append(make_state(2.0, t=2.0), missed_detection=False)
    assert track.num_missed_detections == 0


def test_track_append_consecutive_misses_accumulate():
    """Consecutive missed detections accumulate the counter."""
    track = make_track()
    track.append(make_state(1.0, t=1.0), missed_detection=True)
    track.append(make_state(2.0, t=2.0), missed_detection=True)
    assert track.num_missed_detections == 2


def test_track_append_miss_does_not_increment_num_updates():
    """A missed detection must not increment num_updates."""
    track = make_track()
    track.append(make_state(1.0, t=1.0), missed_detection=True)
    assert track.num_updates == 1


def test_track_append_updates_curr_state():
    """After append, curr_state and curr_time reflect the new state."""
    track = make_track(x=0.0, t=0.0)
    s2 = make_state(5.0, t=1.0)
    track.append(s2)
    assert track.curr_state is s2
    assert track.curr_time == 1.0


# ===========================================================================
# copy
# ===========================================================================

def test_track_copy_default_id_gets_suffix():
    """copy() appends '_0' to the track_id by default."""
    track = make_track(track_id='T1')
    copy = track.copy()
    assert copy.track_id == 'T1_0'


def test_track_copy_custom_id():
    """copy(track_id=...) overrides the default suffix."""
    track = make_track(track_id='T1')
    copy = track.copy(track_id='T1_copy')
    assert copy.track_id == 'T1_copy'


def test_track_copy_shares_state_objects():
    """Shallow copy: State objects in the copied list are the same references."""
    track = make_track()
    copy = track.copy()
    assert copy.states[0] is track.states[0]


def test_track_copy_independent_list():
    """Appending to the copy's state list does not affect the original."""
    track = make_track()
    copy = track.copy()
    copy.append(make_state(1.0, t=1.0))
    assert len(track.states) == 1
    assert len(copy.states) == 2


def test_track_copy_independent_miss_counter():
    """Incrementing the copy's miss counter does not affect the original."""
    track = make_track()
    copy = track.copy()
    copy.append(make_state(1.0, t=1.0), missed_detection=True)
    assert track.num_missed_detections == 0
    assert copy.num_missed_detections == 1


def test_track_copy_independent_num_updates():
    """Incrementing the copy's update count does not affect the original."""
    track = make_track()
    copy = track.copy()
    copy.append(make_state(1.0, t=1.0), missed_detection=False)
    assert track.num_updates == 1
    assert copy.num_updates == 2
