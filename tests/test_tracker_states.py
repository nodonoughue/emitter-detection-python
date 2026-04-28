import numpy as np
import pytest

from ewgeo.tracker.states import State, CartesianStateSpace, PolarKinematicStateSpace


# ---------------------------------------------------------------------------
# Module-level fixtures
# ---------------------------------------------------------------------------

def make_cv_state_space():
    """2D constant-velocity state space: [px, py, vx, vy]"""
    return CartesianStateSpace(num_dims=2, has_vel=True, has_accel=False)


def make_ca_state_space():
    """2D constant-acceleration state space: [px, py, vx, vy, ax, ay]"""
    return CartesianStateSpace(num_dims=2, has_vel=True, has_accel=True)


def make_pos_only_state_space():
    """1D position-only state space: [px]"""
    return CartesianStateSpace(num_dims=1, has_vel=False, has_accel=False)


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


def test_cartesian_state_space_cv_properties():
    """CartesianStateSpace CV: correct num_states, slices, and flags."""
    ss = CartesianStateSpace(num_dims=2, has_vel=True, has_accel=False)
    assert ss.num_dims == 2
    assert ss.num_states == 4
    assert ss.has_pos is True
    assert ss.has_vel is True
    assert ss.has_accel is False
    assert ss.pos_slice == np.s_[:2]
    assert ss.vel_slice == np.s_[2:4]
    assert ss.pos_vel_slice == np.s_[:4]
    assert ss.accel_slice is None


def test_cartesian_state_space_ca_properties():
    """CartesianStateSpace CA: 3-block layout, accel_slice present."""
    ss = CartesianStateSpace(num_dims=2, has_vel=True, has_accel=True)
    assert ss.num_states == 6
    assert ss.has_accel is True
    assert ss.accel_slice == np.s_[4:6]


def test_cartesian_state_space_cj_properties():
    """CartesianStateSpace CJ: 4-block layout (jerk block)."""
    ss = CartesianStateSpace(num_dims=2, has_vel=True, has_accel=True, has_jerk=True)
    assert ss.num_states == 8


def test_cartesian_state_space_pos_only_properties():
    """CartesianStateSpace position-only: vel and accel slices are None."""
    ss = CartesianStateSpace(num_dims=1, has_vel=False, has_accel=False)
    assert ss.num_states == 1
    assert ss.has_vel is False
    assert ss.vel_slice is None
    assert ss.accel_slice is None


def test_cartesian_state_space_jerk_without_accel_raises():
    """has_jerk=True without has_accel=True must raise ValueError."""
    with pytest.raises(ValueError):
        CartesianStateSpace(num_dims=2, has_vel=True, has_accel=False, has_jerk=True)


def test_cartesian_state_space_is_immutable():
    """Properties have no setters; assigning to one must raise AttributeError."""
    ss = CartesianStateSpace(num_dims=2, has_vel=True, has_accel=False)
    with pytest.raises(AttributeError):
        ss.num_dims = 3


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


# ---------------------------------------------------------------------------
# PolarKinematicStateSpace tests
# ---------------------------------------------------------------------------

def test_polar_kinematic_ct_2d_properties():
    """CT 2D yaw-only: [px, py, vx, vy, ω] — 5 states."""
    ss = PolarKinematicStateSpace(num_dims=2, has_vel=True, has_accel=False, num_turn_dims=1)
    assert ss.num_dims == 2
    assert ss.num_states == 5
    assert ss.has_vel is True
    assert ss.has_accel is False
    assert ss.has_turn_rate is True
    assert ss.num_turn_dims == 1
    assert ss.pos_slice == np.s_[:2]
    assert ss.vel_slice == np.s_[2:4]
    assert ss.accel_slice is None
    assert ss.turn_rate_slice == np.s_[4:5]


def test_polar_kinematic_ctra_2d_properties():
    """CTRA 2D yaw-only: [px, py, vx, vy, ax, ay, ω] — 7 states."""
    ss = PolarKinematicStateSpace(num_dims=2, has_vel=True, has_accel=True, num_turn_dims=1)
    assert ss.num_states == 7
    assert ss.has_accel is True
    assert ss.accel_slice == np.s_[4:6]
    assert ss.turn_rate_slice == np.s_[6:7]


def test_polar_kinematic_ct_3d_yaw_pitch_properties():
    """CT 3D yaw+pitch: [px, py, pz, vx, vy, vz, ωyaw, ωpitch] — 8 states."""
    ss = PolarKinematicStateSpace(num_dims=3, has_vel=True, has_accel=False, num_turn_dims=2)
    assert ss.num_states == 8
    assert ss.num_turn_dims == 2
    assert ss.turn_rate_slice == np.s_[6:8]


def test_polar_kinematic_ctra_3d_yaw_pitch_properties():
    """CTRA 3D yaw+pitch: [px,py,pz, vx,vy,vz, ax,ay,az, ωyaw,ωpitch] — 11 states."""
    ss = PolarKinematicStateSpace(num_dims=3, has_vel=True, has_accel=True, num_turn_dims=2)
    assert ss.num_states == 11
    assert ss.turn_rate_slice == np.s_[9:11]


def test_polar_kinematic_turn_rate_component():
    """turn_rate_component extracts the correct scalar from the state vector."""
    ss = PolarKinematicStateSpace(num_dims=2, has_accel=False, num_turn_dims=1)
    x = np.array([1., 2., 3., 4., 0.1])   # ω = 0.1
    assert equal_to_tolerance(ss.turn_rate_component(x), np.array([0.1]))


def test_polar_kinematic_turn_rate_component_yaw_pitch():
    """Two-element turn_rate_component for yaw+pitch case."""
    ss = PolarKinematicStateSpace(num_dims=3, has_accel=False, num_turn_dims=2)
    x = np.array([1., 2., 3., 4., 5., 6., 0.1, 0.05])
    assert equal_to_tolerance(ss.turn_rate_component(x), np.array([0.1, 0.05]))


def test_cartesian_state_space_has_no_turn_rate():
    """CartesianStateSpace.has_turn_rate is False; turn_rate_component returns None."""
    ss = CartesianStateSpace(num_dims=2, has_vel=True)
    assert ss.has_turn_rate is False
    assert ss.turn_rate_slice is None
    assert ss.turn_rate_component(np.array([1., 2., 3., 4.])) is None


def test_polar_kinematic_no_vel_raises():
    with pytest.raises(ValueError):
        PolarKinematicStateSpace(num_dims=2, has_vel=False)


def test_polar_kinematic_invalid_num_turn_dims_raises():
    with pytest.raises(ValueError):
        PolarKinematicStateSpace(num_dims=2, num_turn_dims=3)


def test_polar_kinematic_yaw_pitch_requires_3d():
    with pytest.raises(ValueError):
        PolarKinematicStateSpace(num_dims=2, num_turn_dims=2)


# ---------------------------------------------------------------------------
# State.constrain_motion
# ---------------------------------------------------------------------------

def test_constrain_motion_velocity_below_limit_unchanged():
    """Velocity well within limit — state vector must not change."""
    ss = make_cv_state_space()
    s = State(ss, 0.0, CV_VEC.copy())
    original = s.state.copy()
    s.constrain_motion(max_velocity=100.0)
    np.testing.assert_array_equal(s.state, original)


def test_constrain_motion_velocity_clipped_to_limit():
    """Velocity exceeds limit — magnitude is scaled to max_velocity, direction preserved."""
    ss = make_cv_state_space()
    # velocity = [3, 4] → speed = 5
    s = State(ss, 0.0, np.array([0., 0., 3., 4.]))
    s.constrain_motion(max_velocity=2.5)
    v = s.state[ss.vel_slice]
    assert abs(np.linalg.norm(v) - 2.5) < 1e-10
    # Direction unchanged: (3,4)/5 * 2.5 = (1.5, 2.0)
    np.testing.assert_allclose(v, [1.5, 2.0], atol=1e-10)


def test_constrain_motion_acceleration_clipped():
    """Acceleration exceeds limit — magnitude is scaled down, direction preserved."""
    ss = make_ca_state_space()
    # accel = [0, 10] → mag = 10
    s = State(ss, 0.0, np.array([0., 0., 1., 1., 0., 10.]))
    s.constrain_motion(max_acceleration=5.0)
    a = s.state[ss.accel_slice]
    assert abs(np.linalg.norm(a) - 5.0) < 1e-10
    np.testing.assert_allclose(a, [0., 5.], atol=1e-10)


def test_constrain_motion_both_limits_applied():
    """Both velocity and acceleration are clipped independently."""
    ss = make_ca_state_space()
    # vel=[3,4] speed=5, accel=[0,10] mag=10
    s = State(ss, 0.0, np.array([0., 0., 3., 4., 0., 10.]))
    s.constrain_motion(max_velocity=2.5, max_acceleration=5.0)
    assert abs(np.linalg.norm(s.state[ss.vel_slice]) - 2.5) < 1e-10
    assert abs(np.linalg.norm(s.state[ss.accel_slice]) - 5.0) < 1e-10


def test_constrain_motion_no_vel_state_space_ignored():
    """max_velocity is silently ignored when the state space has no velocity."""
    ss = make_pos_only_state_space()
    s = State(ss, 0.0, np.array([99.]))
    s.constrain_motion(max_velocity=1.0)  # must not raise
    assert s.state[0] == 99.


def test_constrain_motion_none_limits_are_noop():
    """Passing None for both limits leaves the state unchanged."""
    ss = make_ca_state_space()
    s = State(ss, 0.0, CA_VEC.copy())
    original = s.state.copy()
    s.constrain_motion(max_velocity=None, max_acceleration=None)
    np.testing.assert_array_equal(s.state, original)


def test_constrain_motion_position_untouched():
    """Position block is never modified regardless of limits."""
    ss = make_cv_state_space()
    s = State(ss, 0.0, np.array([10., 20., 300., 400.]))
    s.constrain_motion(max_velocity=1.0)
    np.testing.assert_array_equal(s.position, [10., 20.])


# ---------------------------------------------------------------------------
# StateSpace.is_equal
# ---------------------------------------------------------------------------

def test_is_equal_reflexive():
    """A state space must equal itself."""
    ss = make_cv_state_space()
    assert ss.is_equal(ss)


def test_is_equal_identical_cartesian_cv():
    """Two independently constructed identical CV spaces are equal."""
    assert make_cv_state_space().is_equal(make_cv_state_space())


def test_is_equal_identical_cartesian_ca():
    """Two independently constructed identical CA spaces are equal."""
    assert make_ca_state_space().is_equal(make_ca_state_space())


def test_is_equal_different_num_dims():
    """Different num_dims → not equal (fails on first check)."""
    ss2 = CartesianStateSpace(num_dims=2, has_vel=True)
    ss3 = CartesianStateSpace(num_dims=3, has_vel=True)
    assert not ss2.is_equal(ss3)


def test_is_equal_different_has_vel():
    """CV (has_vel=True) vs position-only (has_vel=False) → not equal."""
    assert not make_cv_state_space().is_equal(make_pos_only_state_space())


def test_is_equal_different_has_accel():
    """CV (no accel) vs CA (with accel) with same num_dims → not equal."""
    assert not make_cv_state_space().is_equal(make_ca_state_space())


def test_is_equal_cartesian_vs_polar():
    """CartesianStateSpace and PolarKinematicStateSpace are never equal
    (has_turn_rate differs)."""
    cartesian = CartesianStateSpace(num_dims=2, has_vel=True)
    polar = PolarKinematicStateSpace(num_dims=2, has_vel=True, num_turn_dims=1)
    assert not cartesian.is_equal(polar)


def test_is_equal_identical_polar():
    """Two independently constructed identical PolarKinematic spaces are equal."""
    ss_a = PolarKinematicStateSpace(num_dims=3, has_vel=True, has_accel=False, num_turn_dims=1)
    ss_b = PolarKinematicStateSpace(num_dims=3, has_vel=True, has_accel=False, num_turn_dims=1)
    assert ss_a.is_equal(ss_b)


def test_is_equal_different_num_turn_dims():
    """PolarKinematic with 1 vs 2 turn dims are not equal."""
    ss1 = PolarKinematicStateSpace(num_dims=3, has_vel=True, num_turn_dims=1)
    ss2 = PolarKinematicStateSpace(num_dims=3, has_vel=True, num_turn_dims=2)
    assert not ss1.is_equal(ss2)


def test_is_equal_non_statespace_returns_false():
    """Passing a non-StateSpace object always returns False."""
    ss = make_cv_state_space()
    assert not ss.is_equal(None)
    assert not ss.is_equal("cv")
    assert not ss.is_equal(42)
