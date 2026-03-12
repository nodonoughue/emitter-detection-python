import numpy as np
import pytest

from ewgeo.tracker.states import State, StateSpace


# ---------------------------------------------------------------------------
# Module-level fixtures
# ---------------------------------------------------------------------------

def make_cv_state_space():
    """2D constant-velocity state space: [px, py, vx, vy]"""
    ss = StateSpace()
    ss.num_dims = 2
    ss.num_states = 4
    ss.has_pos = True
    ss.has_vel = True
    ss.has_accel = False
    ss.pos_slice = np.s_[:2]
    ss.vel_slice = np.s_[2:4]
    ss.pos_vel_slice = np.s_[:4]
    ss.accel_slice = None
    return ss


def make_ca_state_space():
    """2D constant-acceleration state space: [px, py, vx, vy, ax, ay]"""
    ss = StateSpace()
    ss.num_dims = 2
    ss.num_states = 6
    ss.has_pos = True
    ss.has_vel = True
    ss.has_accel = True
    ss.pos_slice = np.s_[:2]
    ss.vel_slice = np.s_[2:4]
    ss.pos_vel_slice = np.s_[:4]
    ss.accel_slice = np.s_[4:6]
    return ss


def make_pos_only_state_space():
    """1D position-only state space: [px]"""
    ss = StateSpace()
    ss.num_dims = 1
    ss.num_states = 1
    ss.has_pos = True
    ss.has_vel = False
    ss.has_accel = False
    ss.pos_slice = np.s_[:1]
    ss.vel_slice = None
    ss.pos_vel_slice = np.s_[:1]
    ss.accel_slice = None
    return ss


CV_VEC = np.array([1.0, 2.0, 3.0, 4.0])   # [px, py, vx, vy]
CA_VEC = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # [px, py, vx, vy, ax, ay]

# 4x4 covariance for CV; off-diagonal terms to make slicing non-trivial
CV_COV = np.array([[4., 1., 0., 0.],
                   [1., 4., 0., 0.],
                   [0., 0., 1., 0.],
                   [0., 0., 0., 1.]])

# 6x6 covariance for CA
CA_COV = np.diag([4., 4., 1., 1., 0.25, 0.25])


def equal_to_tolerance(x, y, tol=1e-6) -> bool:
    if np.any(np.shape(x) != np.shape(y)):
        return False
    return all([abs(xx - yy) < tol for xx, yy in zip(np.ravel(x), np.ravel(y))])


# ---------------------------------------------------------------------------
# StateSpace tests
# ---------------------------------------------------------------------------

def test_statespace_pos_component():
    ss = make_cv_state_space()
    assert equal_to_tolerance(ss.pos_component(CV_VEC), np.array([1.0, 2.0]))


def test_statespace_vel_component():
    ss = make_cv_state_space()
    assert equal_to_tolerance(ss.vel_component(CV_VEC), np.array([3.0, 4.0]))


def test_statespace_vel_component_none_when_no_vel():
    ss = make_pos_only_state_space()
    assert ss.vel_component(np.array([5.0])) is None


def test_statespace_accel_component():
    ss = make_ca_state_space()
    assert equal_to_tolerance(ss.accel_component(CA_VEC), np.array([5.0, 6.0]))


def test_statespace_accel_component_none_when_no_accel():
    ss = make_cv_state_space()
    assert ss.accel_component(CV_VEC) is None


def test_statespace_pos_vel_component():
    ss = make_ca_state_space()
    assert equal_to_tolerance(ss.pos_vel_component(CA_VEC), np.array([1.0, 2.0, 3.0, 4.0]))


def test_statespace_pos_vel_component_falls_back_to_pos():
    """When has_vel is False, pos_vel_component should return just position."""
    ss = make_pos_only_state_space()
    assert equal_to_tolerance(ss.pos_vel_component(np.array([7.0])), np.array([7.0]))


def test_statespace_copy_preserves_attributes():
    ss = make_cv_state_space()
    ss2 = ss.copy()
    assert ss2.num_dims == ss.num_dims
    assert ss2.num_states == ss.num_states
    assert ss2.has_vel == ss.has_vel
    assert ss2.has_accel == ss.has_accel


def test_statespace_copy_override():
    """copy() should accept keyword overrides."""
    ss = make_cv_state_space()
    ss2 = ss.copy(num_dims=3)
    assert ss2.num_dims == 3
    assert ss.num_dims == 2  # original unchanged


# ---------------------------------------------------------------------------
# State construction
# ---------------------------------------------------------------------------

def test_state_init():
    ss = make_cv_state_space()
    s = State(ss, time=1.5, state=CV_VEC)
    assert s.time == 1.5
    assert equal_to_tolerance(s.state, CV_VEC)


def test_state_init_none_state_gives_zeros():
    ss = make_cv_state_space()
    s = State(ss, time=0.0, state=None)
    assert equal_to_tolerance(s.state, np.zeros(4))


def test_state_size():
    ss = make_cv_state_space()
    s = State(ss, time=0.0, state=CV_VEC)
    assert s.size == 4


def test_state_has_vel_delegates_to_state_space():
    ss = make_cv_state_space()
    s = State(ss, time=0.0, state=CV_VEC)
    assert s.has_vel is True

    ss2 = make_pos_only_state_space()
    s2 = State(ss2, time=0.0, state=np.array([1.0]))
    assert s2.has_vel is False


def test_state_has_accel_delegates_to_state_space():
    ss_cv = make_cv_state_space()
    ss_ca = make_ca_state_space()
    assert State(ss_cv, 0.0, CV_VEC).has_accel is False
    assert State(ss_ca, 0.0, CA_VEC).has_accel is True


# ---------------------------------------------------------------------------
# State property getters
# ---------------------------------------------------------------------------

def test_state_position():
    ss = make_cv_state_space()
    s = State(ss, time=0.0, state=CV_VEC)
    assert equal_to_tolerance(s.position, np.array([1.0, 2.0]))


def test_state_velocity():
    ss = make_cv_state_space()
    s = State(ss, time=0.0, state=CV_VEC)
    assert equal_to_tolerance(s.velocity, np.array([3.0, 4.0]))


def test_state_velocity_none_when_no_vel():
    ss = make_pos_only_state_space()
    s = State(ss, time=0.0, state=np.array([1.0]))
    assert s.velocity is None


def test_state_acceleration():
    ss = make_ca_state_space()
    s = State(ss, time=0.0, state=CA_VEC)
    assert equal_to_tolerance(s.acceleration, np.array([5.0, 6.0]))


def test_state_acceleration_none_when_no_accel():
    ss = make_cv_state_space()
    s = State(ss, time=0.0, state=CV_VEC)
    assert s.acceleration is None


def test_state_pos_vel():
    ss = make_ca_state_space()
    s = State(ss, time=0.0, state=CA_VEC)
    assert equal_to_tolerance(s.pos_vel, np.array([1.0, 2.0, 3.0, 4.0]))


# ---------------------------------------------------------------------------
# State property setters
# ---------------------------------------------------------------------------

def test_state_position_setter():
    ss = make_cv_state_space()
    s = State(ss, time=0.0, state=CV_VEC.copy())
    s.position = np.array([10.0, 20.0])
    assert equal_to_tolerance(s.position, np.array([10.0, 20.0]))
    assert equal_to_tolerance(s.velocity, np.array([3.0, 4.0]))  # velocity unchanged


def test_state_velocity_setter():
    ss = make_cv_state_space()
    s = State(ss, time=0.0, state=CV_VEC.copy())
    s.velocity = np.array([30.0, 40.0])
    assert equal_to_tolerance(s.velocity, np.array([30.0, 40.0]))
    assert equal_to_tolerance(s.position, np.array([1.0, 2.0]))  # position unchanged


def test_state_velocity_setter_raises_when_no_vel():
    ss = make_pos_only_state_space()
    s = State(ss, time=0.0, state=np.array([1.0]))
    with pytest.raises(ValueError):
        s.velocity = np.array([5.0])


def test_state_acceleration_setter():
    ss = make_ca_state_space()
    s = State(ss, time=0.0, state=CA_VEC.copy())
    s.acceleration = np.array([50.0, 60.0])
    assert equal_to_tolerance(s.acceleration, np.array([50.0, 60.0]))
    assert equal_to_tolerance(s.position, np.array([1.0, 2.0]))  # position unchanged


def test_state_acceleration_setter_raises_when_no_accel():
    ss = make_cv_state_space()
    s = State(ss, time=0.0, state=CV_VEC.copy())
    with pytest.raises(ValueError):
        s.acceleration = np.array([1.0, 2.0])


# ---------------------------------------------------------------------------
# Covariance submatrix extraction
# ---------------------------------------------------------------------------

def test_state_position_covar():
    ss = make_cv_state_space()
    s = State(ss, time=0.0, state=CV_VEC, covar=CV_COV)
    pos_cov = s.position_covar
    expected = CV_COV[:2, :2]
    assert equal_to_tolerance(pos_cov.cov, expected)


def test_state_velocity_covar():
    ss = make_cv_state_space()
    s = State(ss, time=0.0, state=CV_VEC, covar=CV_COV)
    vel_cov = s.velocity_covar
    expected = CV_COV[2:4, 2:4]
    assert equal_to_tolerance(vel_cov.cov, expected)


def test_state_acceleration_covar():
    ss = make_ca_state_space()
    s = State(ss, time=0.0, state=CA_VEC, covar=CA_COV)
    accel_cov = s.acceleration_covar
    expected = CA_COV[4:6, 4:6]
    assert equal_to_tolerance(accel_cov.cov, expected)


def test_state_pos_vel_covar():
    ss = make_ca_state_space()
    s = State(ss, time=0.0, state=CA_VEC, covar=CA_COV)
    pv_cov = s.pos_vel_covar
    expected = CA_COV[:4, :4]
    assert equal_to_tolerance(pv_cov.cov, expected)


def test_state_position_covar_none_when_no_covar():
    ss = make_cv_state_space()
    s = State(ss, time=0.0, state=CV_VEC)
    assert s.position_covar is None


def test_state_velocity_covar_none_when_no_vel():
    ss = make_pos_only_state_space()
    s = State(ss, time=0.0, state=np.array([1.0]))
    assert s.velocity_covar is None


def test_state_acceleration_covar_none_when_no_accel():
    ss = make_cv_state_space()
    s = State(ss, time=0.0, state=CV_VEC, covar=CV_COV)
    assert s.acceleration_covar is None
