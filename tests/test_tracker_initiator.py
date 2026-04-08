"""
Tests for tracker/initiator.py.

Note: tests involving NNAssociator will produce stdout output due to an active
print() statement in NNAssociator.associate() (association.py line 475).
"""
import numpy as np
import pytest

from ewgeo.tracker.initiator import (
    SinglePointMeasurementInitiator,
    TwoPointInitiator,
    ThreePointInitiator,
)
from ewgeo.tracker.association import NNAssociator
from ewgeo.tracker.measurement import Measurement, MeasurementModel
from ewgeo.tracker.states import (State, StateSpace, adapt_cartesian_state,
                                  PolarKinematicStateSpace)
from ewgeo.tracker.track import Track
from ewgeo.tracker.transition import (
    ConstantVelocityMotionModel,
    ConstantAccelerationMotionModel,
    ConstantTurnMotionModel,
)
from ewgeo.utils.covariance import CovarianceMatrix


def equal_to_tolerance(x, y, tol=1e-6):
    return np.all(np.fabs(np.array(x) - np.array(y)) < tol)


# ---------------------------------------------------------------------------
# Mock PSS — supports state_from_measurement for CV 1D
# ---------------------------------------------------------------------------

class _MockSensor1D:
    """
    Direct 1D position sensor.  Enough to drive MeasurementModel.state_from_measurement
    and the Hypothesis machinery used by the associator.
    """
    num_dim = 1
    num_measurements = 1
    cov = CovarianceMatrix(np.array([[1.0]]))

    def measurement(self, x_source, v_source=None, **kwargs):
        return np.atleast_1d(x_source[:1]).astype(float)

    def jacobian(self, x_source, v_source=None, **kwargs):
        return np.array([[1.0]])   # shape (num_dim, num_measurements)

    def least_square(self, zeta, x_init, **kwargs):
        # For z=x, optimal estimate is [zeta, 0] (position, zero velocity)
        return np.array([zeta[0], 0.0]), None

    def compute_crlb(self, x_source, **kwargs):
        # Position variance 4; velocity variance 1000 (poorly known)
        return CovarianceMatrix(np.diag([4.0, 1000.0]))

    def log_likelihood(self, x_source, zeta, **kwargs):
        return 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_cv_model():
    return ConstantVelocityMotionModel(num_dims=1, process_covar=1.0)


def make_cv_measurement_model():
    model = make_cv_model()
    return MeasurementModel(state_space=model.state_space, pss=_MockSensor1D())


def make_measurement(x, t=0.0):
    return Measurement(time=t, sensor=_MockSensor1D(), zeta=np.array([float(x)]))


def make_ca_state(x, t, covar_diag=0.1):
    """Make a 1D CA state [x, vx, ax] with only position set."""
    model = ConstantAccelerationMotionModel(num_dims=1, process_covar=1.0)
    ss = model.state_space
    state_vec = np.zeros(ss.num_states)
    state_vec[ss.pos_slice] = x
    covar = CovarianceMatrix(np.eye(ss.num_states) * covar_diag)
    return State(ss, time=t, state=state_vec, covar=covar)


# ---------------------------------------------------------------------------
# SinglePointMeasurementInitiator
# ---------------------------------------------------------------------------

def test_single_point_creates_one_track_per_measurement():
    initiator = SinglePointMeasurementInitiator(msmt_model=make_cv_measurement_model())
    measurements = [make_measurement(5), make_measurement(10)]
    tracks, _ = initiator.initiate(measurements, next_track_id=0)
    assert len(tracks) == 2


def test_single_point_track_ids_are_sequential():
    initiator = SinglePointMeasurementInitiator(msmt_model=make_cv_measurement_model())
    measurements = [make_measurement(5), make_measurement(10)]
    tracks, next_id = initiator.initiate(measurements, next_track_id=7)
    assert tracks[0].track_id == 7
    assert tracks[1].track_id == 8
    assert next_id == 9


def test_single_point_track_position_matches_measurement():
    initiator = SinglePointMeasurementInitiator(msmt_model=make_cv_measurement_model())
    tracks, _ = initiator.initiate([make_measurement(5.0)], next_track_id=0)
    assert equal_to_tolerance(tracks[0].curr_state.position, [5.0])


def test_single_point_empty_measurements_returns_empty():
    initiator = SinglePointMeasurementInitiator(msmt_model=make_cv_measurement_model())
    tracks, next_id = initiator.initiate([], next_track_id=3)
    assert tracks == []
    assert next_id == 3


# ---------------------------------------------------------------------------
# TwoPointInitiator — static helpers
# ---------------------------------------------------------------------------

def test_build_state_with_velocity_sets_vel_component():
    model = make_cv_model()
    s = State(model.state_space, time=1.0,
              state=np.array([5.0, 0.0]),
              covar=CovarianceMatrix(np.eye(2)))
    result = TwoPointInitiator._build_state_with_velocity(s, vel_est=np.array([2.5]))
    assert equal_to_tolerance(result, [5.0, 2.5])


def test_build_state_with_velocity_does_not_modify_original():
    model = make_cv_model()
    s = State(model.state_space, time=1.0, state=np.array([5.0, 0.0]))
    TwoPointInitiator._build_state_with_velocity(s, vel_est=np.array([2.5]))
    assert equal_to_tolerance(s.state, [5.0, 0.0])


def test_build_initial_covariance_pos_var():
    """Position variance = 10 * crlb_pos."""
    model = make_cv_model()
    s = State(model.state_space, time=1.0,
              state=np.array([5.0, 0.0]),
              covar=CovarianceMatrix(np.diag([4.0, 1000.0])))
    covar = TwoPointInitiator._build_initial_covariance(s, dt=1.0)
    # pos_var = 10 * 4 = 40
    assert equal_to_tolerance(covar.cov[0, 0], 40.0)


def test_build_initial_covariance_vel_var_scales_with_dt():
    """vel_var = pos_var / dt^2, so larger dt → smaller velocity uncertainty."""
    model = make_cv_model()
    s = State(model.state_space, time=1.0,
              state=np.array([5.0, 0.0]),
              covar=CovarianceMatrix(np.diag([4.0, 1000.0])))
    covar_dt1 = TwoPointInitiator._build_initial_covariance(s, dt=1.0)
    covar_dt2 = TwoPointInitiator._build_initial_covariance(s, dt=2.0)
    # dt=1: vel_var = 40/1 = 40; dt=2: vel_var = 40/4 = 10
    assert equal_to_tolerance(covar_dt1.cov[1, 1], 40.0)
    assert equal_to_tolerance(covar_dt2.cov[1, 1], 10.0)


# ---------------------------------------------------------------------------
# TwoPointInitiator — integration
# ---------------------------------------------------------------------------

def test_two_point_first_call_returns_no_tracks():
    """First call buffers measurements; no confirmed tracks yet."""
    mm = make_cv_measurement_model()
    assoc = NNAssociator(motion_model=make_cv_model(), gate_probability=0.99)
    initiator = TwoPointInitiator(msmt_model=mm, associator=assoc)

    tracks, _ = initiator.initiate([make_measurement(10, t=0)], next_track_id=0)
    assert tracks == []
    assert len(initiator._buffer_tracks) == 1


def test_two_point_second_call_returns_track_with_velocity():
    """
    Two calls, one measurement each:
      call 1: m1 at x=10, t=0  → buffered
      call 2: m2 at x=12, t=1  → associated with m1, vel=(12-10)/1=2.0
    """
    mm = make_cv_measurement_model()
    assoc = NNAssociator(motion_model=make_cv_model(), gate_probability=0.99)
    initiator = TwoPointInitiator(msmt_model=mm, associator=assoc)

    initiator.initiate([make_measurement(10, t=0)], next_track_id=0)
    tracks, _ = initiator.initiate([make_measurement(12, t=1)], next_track_id=1)

    assert len(tracks) == 1
    assert equal_to_tolerance(tracks[0].curr_state.position, [12.0])
    assert equal_to_tolerance(tracks[0].curr_state.velocity, [2.0])


def test_two_point_unmatched_second_call_goes_to_new_buffer():
    """
    If the second measurement is far from the buffered track and fails the gate,
    both end up in the buffer and no confirmed track is produced.
    """
    mm = make_cv_measurement_model()
    assoc = NNAssociator(motion_model=make_cv_model(), gate_probability=0.99)
    initiator = TwoPointInitiator(msmt_model=mm, associator=assoc)

    initiator.initiate([make_measurement(10, t=0)], next_track_id=0)
    # x=9000 is very far — will fail the gate against the buffered track at x=10
    tracks, _ = initiator.initiate([make_measurement(9000, t=1)], next_track_id=1)

    assert tracks == []


# ---------------------------------------------------------------------------
# ThreePointInitiator._build_track — finite difference estimates
# ---------------------------------------------------------------------------

def test_build_track_position():
    """Third-point position is stored in the confirmed track."""
    initiator = ThreePointInitiator(msmt_model=None, associator=None)
    s1 = make_ca_state(0.0, t=0)
    s2 = make_ca_state(1.5, t=1)
    s3 = make_ca_state(4.0, t=2)
    track = initiator._build_track(s1, s2, s3)
    assert track is not None
    assert equal_to_tolerance(track.curr_state.position, [4.0])


def test_build_track_velocity_central_difference():
    """vel = (x3 - x1) / (2*dt).  For x=[0,1.5,4], dt=1: vel=2.0."""
    initiator = ThreePointInitiator(msmt_model=None, associator=None)
    s1 = make_ca_state(0.0, t=0)
    s2 = make_ca_state(1.5, t=1)
    s3 = make_ca_state(4.0, t=2)
    track = initiator._build_track(s1, s2, s3)
    assert track is not None
    assert equal_to_tolerance(track.curr_state.velocity, [2.0])


def test_build_track_acceleration_second_difference():
    """acc = (x3 - 2*x2 + x1) / dt^2.  For x=[0,1.5,4], dt=1: acc=1.0."""
    initiator = ThreePointInitiator(msmt_model=None, associator=None)
    s1 = make_ca_state(0.0, t=0)
    s2 = make_ca_state(1.5, t=1)
    s3 = make_ca_state(4.0, t=2)
    track = initiator._build_track(s1, s2, s3)
    assert track is not None
    assert equal_to_tolerance(track.curr_state.acceleration, [1.0])


def test_build_track_returns_none_for_inconsistent_dt():
    """If dt1 and dt2 differ by more than 10%, _build_track returns None."""
    initiator = ThreePointInitiator(msmt_model=None, associator=None)
    s1 = make_ca_state(0.0, t=0)
    s2 = make_ca_state(1.5, t=1)
    s3 = make_ca_state(4.0, t=3)   # dt2=2, dt1=1  →  |1-2|/2=0.5 > 0.1
    assert initiator._build_track(s1, s2, s3) is None


def test_build_track_covariance_structure():
    """
    With covar_diag=0.1 and dt=1:
      pos_var  = 10 * 0.1 = 1.0
      vel_var  = 1.0 / (2 * 1^2) = 0.5
      acc_var  = 6.0 * 1.0 / (1^4)  = 6.0
    Covariance diagonal should be [1.0, 0.5, 6.0].
    """
    initiator = ThreePointInitiator(msmt_model=None, associator=None)
    s1 = make_ca_state(0.0, t=0, covar_diag=0.1)
    s2 = make_ca_state(1.5, t=1, covar_diag=0.1)
    s3 = make_ca_state(4.0, t=2, covar_diag=0.1)
    track = initiator._build_track(s1, s2, s3)
    assert track is not None
    cov_diag = np.diag(track.curr_state.covar.cov)
    assert equal_to_tolerance(cov_diag, [1.0, 0.5, 6.0])


# ---------------------------------------------------------------------------
# adapt_cartesian_state
# ---------------------------------------------------------------------------

def make_cv_state_1d(x=5.0, v=2.0, t=1.0, pos_var=4.0, vel_var=1000.0):
    """1D CV state [x, vx] with diagonal covariance."""
    model = ConstantVelocityMotionModel(num_dims=1, process_covar=1.0)
    ss = model.state_space
    state_vec = np.array([x, v])
    covar = CovarianceMatrix(np.diag([pos_var, vel_var]))
    return State(ss, time=t, state=state_vec, covar=covar)


def make_ct_ss_1d():
    """1D CT-like state space [x, vx, ω] built directly (CT model requires ≥2D)."""
    return PolarKinematicStateSpace(num_dims=1, has_accel=False, num_turn_dims=1)


def make_ct_ss_2d():
    """2D CT state space [px, py, vx, vy, ω]."""
    return ConstantTurnMotionModel(num_dims=2, process_covar=1.0).state_space


def test_adapt_cartesian_copies_position():
    """Adapted position equals source position."""
    src = make_cv_state_1d(x=7.5)
    adapted = adapt_cartesian_state(src, make_ct_ss_1d())
    assert equal_to_tolerance(adapted.position, [7.5])


def test_adapt_cartesian_copies_velocity():
    """Adapted velocity equals source velocity."""
    src = make_cv_state_1d(x=5.0, v=3.0)
    adapted = adapt_cartesian_state(src, make_ct_ss_1d())
    assert equal_to_tolerance(adapted.velocity, [3.0])


def test_adapt_cartesian_turn_rate_is_zero():
    """Turn-rate mean initialises to zero."""
    src = make_cv_state_1d()
    ct_ss = make_ct_ss_1d()
    adapted = adapt_cartesian_state(src, ct_ss)
    assert equal_to_tolerance(ct_ss.turn_rate_component(adapted.state), [0.0])


def test_adapt_cartesian_state_size():
    """Output state vector has target_ss.num_states entries."""
    ct_ss = make_ct_ss_1d()
    adapted = adapt_cartesian_state(make_cv_state_1d(), ct_ss)
    assert adapted.size == ct_ss.num_states


def test_adapt_cartesian_turn_rate_variance():
    """Turn-rate variance block equals sigma_turn_rate**2."""
    ct_ss = make_ct_ss_1d()
    adapted = adapt_cartesian_state(make_cv_state_1d(pos_var=4.0, vel_var=1000.0),
                                    ct_ss, sigma_turn_rate=0.3)
    tr_sl = ct_ss.turn_rate_slice
    assert equal_to_tolerance(adapted.covar.cov[tr_sl, tr_sl], [[0.09]])  # 0.3**2


def test_adapt_cartesian_copies_pos_covariance():
    """Position covariance block is copied from source."""
    ct_ss = make_ct_ss_1d()
    adapted = adapt_cartesian_state(make_cv_state_1d(pos_var=4.0, vel_var=1000.0), ct_ss)
    assert equal_to_tolerance(adapted.covar.cov[ct_ss.pos_slice, ct_ss.pos_slice], [[4.0]])


def test_adapt_cartesian_no_covar_gives_none():
    """When source has no covariance, the adapted state also has None."""
    model = ConstantVelocityMotionModel(num_dims=1, process_covar=1.0)
    src = State(model.state_space, time=0.0, state=np.array([5.0, 2.0]), covar=None)
    adapted = adapt_cartesian_state(src, make_ct_ss_1d())
    assert adapted.covar is None


def test_adapt_cartesian_preserves_timestamp():
    """Adapted state keeps the source timestamp."""
    adapted = adapt_cartesian_state(make_cv_state_1d(t=42.5), make_ct_ss_1d())
    assert adapted.time == 42.5


# ---------------------------------------------------------------------------
# SinglePointMeasurementInitiator with target_state_space
# ---------------------------------------------------------------------------

def test_single_point_with_target_ss_produces_ct_state():
    """Tracks produced with target_state_space have the CT state size."""
    # Use 1D PolarKinematicStateSpace so it matches the 1D mock sensor
    ct_ss = make_ct_ss_1d()
    initiator = SinglePointMeasurementInitiator(
        msmt_model=make_cv_measurement_model(),
        target_state_space=ct_ss,
    )
    tracks, _ = initiator.initiate([make_measurement(5.0)], next_track_id=0)
    assert len(tracks) == 1
    assert tracks[0].curr_state.size == ct_ss.num_states


def test_single_point_with_target_ss_has_zero_turn_rate():
    """Turn-rate component of the adapted state is zero."""
    ct_ss = make_ct_ss_1d()
    initiator = SinglePointMeasurementInitiator(
        msmt_model=make_cv_measurement_model(),
        target_state_space=ct_ss,
    )
    tracks, _ = initiator.initiate([make_measurement(5.0)], next_track_id=0)
    tr = ct_ss.turn_rate_component(tracks[0].curr_state.state)
    assert equal_to_tolerance(tr, [0.0])


# ---------------------------------------------------------------------------
# TwoPointInitiator with target_state_space
# ---------------------------------------------------------------------------

def test_two_point_with_target_ss_produces_ct_state():
    """Confirmed tracks have CT state size when target_state_space is set."""
    ct_ss = make_ct_ss_1d()
    mm = make_cv_measurement_model()
    assoc = NNAssociator(motion_model=make_cv_model(), gate_probability=0.99)
    initiator = TwoPointInitiator(msmt_model=mm, associator=assoc,
                                  target_state_space=ct_ss)

    initiator.initiate([make_measurement(10, t=0)], next_track_id=0)
    tracks, _ = initiator.initiate([make_measurement(12, t=1)], next_track_id=1)

    assert len(tracks) == 1
    assert tracks[0].curr_state.size == ct_ss.num_states


def test_two_point_with_target_ss_velocity_preserved():
    """Velocity estimate is preserved after adaptation to CT state space."""
    ct_ss = make_ct_ss_1d()
    mm = make_cv_measurement_model()
    assoc = NNAssociator(motion_model=make_cv_model(), gate_probability=0.99)
    initiator = TwoPointInitiator(msmt_model=mm, associator=assoc,
                                  target_state_space=ct_ss)

    initiator.initiate([make_measurement(10, t=0)], next_track_id=0)
    tracks, _ = initiator.initiate([make_measurement(12, t=1)], next_track_id=1)

    assert equal_to_tolerance(tracks[0].curr_state.velocity, [2.0])


# ---------------------------------------------------------------------------
# ThreePointInitiator with target_state_space
# ---------------------------------------------------------------------------

def test_three_point_with_target_ss_produces_ct_state():
    """Confirmed tracks from ThreePointInitiator have CT state size."""
    ct_ss = make_ct_ss_1d()
    initiator = ThreePointInitiator(msmt_model=None, associator=None,
                                    target_state_space=ct_ss)
    s1 = make_ca_state(0.0, t=0)
    s2 = make_ca_state(1.5, t=1)
    s3 = make_ca_state(4.0, t=2)
    # _build_track uses s3.state_space (CA), then adaptation replaces it with ct_ss
    full_track = initiator._build_track(s1, s2, s3)
    # Direct adaptation path (bypasses the initiate() association loop)
    adapted = adapt_cartesian_state(full_track.curr_state, ct_ss)
    assert adapted.size == ct_ss.num_states
    assert equal_to_tolerance(adapted.velocity, [2.0])


# ---------------------------------------------------------------------------
# _propagate_covariance bug fix: no crash for CV (no accel) state space
# ---------------------------------------------------------------------------

def test_propagate_covariance_no_accel_does_not_crash():
    """_propagate_covariance must not crash when state_space.has_accel is False."""
    cv_ss = ConstantVelocityMotionModel(num_dims=1, process_covar=1.0).state_space
    initiator = ThreePointInitiator(msmt_model=None, associator=None)
    pos_var = np.array([[1.0]])
    # This should not raise even though cv_ss.accel_slice is None
    covar = initiator._propagate_covariance(pos_var, dt=1.0, state_space=cv_ss)
    # pos block was scaled; size matches CV num_states (2)
    assert covar.shape == (cv_ss.num_states, cv_ss.num_states)


# ---------------------------------------------------------------------------
# target_max_velocity / target_max_acceleration caps
# ---------------------------------------------------------------------------

def test_two_point_build_covariance_vel_uncapped_when_no_max():
    """Without target_max_velocity the velocity variance equals pos_var / dt^2."""
    model = make_cv_model()
    s = State(model.state_space, time=1.0,
              state=np.array([5.0, 0.0]),
              covar=CovarianceMatrix(np.diag([4.0, 1000.0])))
    covar = TwoPointInitiator._build_initial_covariance(s, dt=1.0,
                                                        target_max_velocity=None)
    # pos_var = 10 * 4 = 40; vel_var = 40 / 1^2 = 40
    assert equal_to_tolerance(covar.cov[1, 1], 40.0)


def test_two_point_build_covariance_vel_capped_when_below_crlb():
    """target_max_velocity=3 caps vel_var at 3^2=9 (CRLB-derived would be 40)."""
    model = make_cv_model()
    s = State(model.state_space, time=1.0,
              state=np.array([5.0, 0.0]),
              covar=CovarianceMatrix(np.diag([4.0, 1000.0])))
    covar = TwoPointInitiator._build_initial_covariance(s, dt=1.0,
                                                        target_max_velocity=3.0)
    assert covar.cov[1, 1] <= 3.0 ** 2 + 1e-9


def test_two_point_build_covariance_vel_unchanged_when_max_is_large():
    """target_max_velocity larger than the CRLB estimate leaves vel_var unchanged."""
    model = make_cv_model()
    s = State(model.state_space, time=1.0,
              state=np.array([5.0, 0.0]),
              covar=CovarianceMatrix(np.diag([4.0, 1000.0])))
    # CRLB-derived vel_var = 40; max_velocity=10 -> cap = 100 > 40, no change
    covar_uncapped = TwoPointInitiator._build_initial_covariance(s, dt=1.0)
    covar_capped   = TwoPointInitiator._build_initial_covariance(s, dt=1.0,
                                                                 target_max_velocity=10.0)
    assert equal_to_tolerance(covar_capped.cov[1, 1], covar_uncapped.cov[1, 1])


def test_two_point_constructor_stores_max_velocity():
    mm = make_cv_measurement_model()
    assoc = NNAssociator(motion_model=make_cv_model(), gate_probability=0.99)
    initiator = TwoPointInitiator(msmt_model=mm, associator=assoc,
                                  target_max_velocity=250.0)
    assert initiator.target_max_velocity == 250.0


def test_three_point_propagate_vel_capped():
    """target_max_velocity=2 caps vel_var (normally pos_var/(2*dt^2) = 0.5) ... but
    pos_var=1 gives vel_var=0.5 < 2^2=4, so no cap should trigger.
    Use a large pos_var so the cap is needed."""
    ca_ss = ConstantAccelerationMotionModel(num_dims=1, process_covar=1.0).state_space
    initiator = ThreePointInitiator(msmt_model=None, associator=None,
                                    target_max_velocity=2.0)
    # pos_var = 100 -> vel_var = 100/(2*1) = 50 >> 4 -> should be capped to 4
    pos_var = np.array([[100.0]])
    covar = initiator._propagate_covariance(pos_var, dt=1.0, state_space=ca_ss)
    vel_idx = ca_ss.vel_slice
    assert covar[vel_idx, vel_idx][0, 0] <= 2.0 ** 2 + 1e-9


def test_three_point_propagate_accel_capped():
    """target_max_acceleration=1 caps acc_var (normally 6*pos_var/dt^4 = 600) to 1."""
    ca_ss = ConstantAccelerationMotionModel(num_dims=1, process_covar=1.0).state_space
    initiator = ThreePointInitiator(msmt_model=None, associator=None,
                                    target_max_acceleration=1.0)
    pos_var = np.array([[100.0]])
    covar = initiator._propagate_covariance(pos_var, dt=1.0, state_space=ca_ss)
    accel_idx = ca_ss.accel_slice
    assert covar[accel_idx, accel_idx][0, 0] <= 1.0 ** 2 + 1e-9


def test_three_point_propagate_no_cap_when_below_bound():
    """No cap when both bounds are larger than the CRLB-derived variances."""
    ca_ss = ConstantAccelerationMotionModel(num_dims=1, process_covar=1.0).state_space
    initiator_no_cap = ThreePointInitiator(msmt_model=None, associator=None)
    initiator_large  = ThreePointInitiator(msmt_model=None, associator=None,
                                           target_max_velocity=1000.0,
                                           target_max_acceleration=1000.0)
    pos_var = np.array([[1.0]])
    c1 = initiator_no_cap._propagate_covariance(pos_var, dt=1.0, state_space=ca_ss)
    c2 = initiator_large._propagate_covariance(pos_var, dt=1.0, state_space=ca_ss)
    assert equal_to_tolerance(c1, c2)
