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
from ewgeo.tracker.states import State, StateSpace
from ewgeo.tracker.track import Track
from ewgeo.tracker.transition import (
    ConstantVelocityMotionModel,
    ConstantAccelerationMotionModel,
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
