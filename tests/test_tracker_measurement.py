import numpy as np
import pytest
from scipy.stats import multivariate_normal

from ewgeo.tracker.measurement import Measurement, MeasurementModel, kf_update, ekf_update
from ewgeo.tracker.states import State, StateSpace
from ewgeo.utils.covariance import CovarianceMatrix


def equal_to_tolerance(x, y, tol=1e-10):
    return np.all(np.fabs(np.array(x) - np.array(y)) < tol)


# ---------------------------------------------------------------------------
# Minimal PSS stub for MeasurementModel tests
# ---------------------------------------------------------------------------

class _MockPSS:
    """
    Minimal stub that satisfies the interface MeasurementModel needs.
    Two position dimensions, two scalar measurements (one per dim).
    The jacobian is just the 2x2 identity: each measurement maps directly
    to the corresponding position component.
    """
    num_dim = 2
    num_measurements = 2
    _jac = np.eye(2)  # shape (num_dim, num_measurements)

    def jacobian(self, x_source, v_source=None, **kwargs):
        return self._jac


class _MockPSSFull:
    """
    Complete mock PSS: z = x (identity), Gaussian LL with R = I.
    Supports measurement(), jacobian(), and log_likelihood().
    """
    num_dim = 2
    num_measurements = 2
    cov = CovarianceMatrix(np.eye(2))

    def measurement(self, x_source, v_source=None, **kwargs):
        return np.array(x_source, dtype=float)

    def jacobian(self, x_source, v_source=None, **kwargs):
        return np.eye(2)

    def log_likelihood(self, x_source, zeta, v_source=None, **kwargs):
        return float(multivariate_normal.logpdf(zeta, mean=x_source, cov=np.eye(2)))

    def noisy_measurement(self, x_source, v_source=None, **kwargs):
        return np.array(x_source, dtype=float) + np.random.randn(2)


class _MockPSSWithVel:
    """
    Stub where the jacobian also returns velocity rows (shape 2*num_dim x num_measurements).
    """
    num_dim = 2
    num_measurements = 2
    _jac = np.vstack([np.eye(2), 0.5 * np.eye(2)])  # shape (4, 2)

    def jacobian(self, x_source, v_source=None, **kwargs):
        return self._jac


def make_cv_state_space():
    """2D CV state space: [px, py, vx, vy]"""
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


def make_pos_only_state_space():
    """2D position-only state space: [px, py]"""
    ss = StateSpace()
    ss.num_dims = 2
    ss.num_states = 2
    ss.has_pos = True
    ss.has_vel = False
    ss.has_accel = False
    ss.pos_slice = np.s_[:2]
    ss.vel_slice = None
    ss.pos_vel_slice = np.s_[:2]
    ss.accel_slice = None
    return ss


# ---------------------------------------------------------------------------
# Measurement class
# ---------------------------------------------------------------------------

def test_measurement_stores_fields():
    zeta = np.array([1.0, 2.0])
    m = Measurement(time=3.0, sensor=None, zeta=zeta)
    assert m.time == 3.0
    assert m.sensor is None
    assert np.array_equal(m.zeta, zeta)


def test_measurement_size():
    m = Measurement(time=0.0, sensor=None, zeta=np.array([1.0, 2.0, 3.0]))
    assert m.size == 3


# ---------------------------------------------------------------------------
# kf_update — 1D state, scalar measurement
# ---------------------------------------------------------------------------

def test_kf_update_1d_state_updated():
    """State estimate moves toward the measurement."""
    x = np.array([10.0])
    P = CovarianceMatrix(np.array([[4.0]]))
    H = np.array([[1.0]])
    R = CovarianceMatrix(np.array([[1.0]]))
    zeta = np.array([12.0])

    x_new, _ = kf_update(x, P, zeta, R, H)
    # K = 4/5 = 0.8;  x = 10 + 0.8*2 = 11.6
    assert equal_to_tolerance(x_new, [11.6])


def test_kf_update_1d_covariance_reduced():
    """Posterior covariance is smaller than prior."""
    x = np.array([10.0])
    P = CovarianceMatrix(np.array([[4.0]]))
    H = np.array([[1.0]])
    R = CovarianceMatrix(np.array([[1.0]]))
    zeta = np.array([12.0])

    _, P_new = kf_update(x, P, zeta, R, H)
    # P = (1 - K*H)*P = (1 - 0.8)*4 = 0.8
    assert equal_to_tolerance(P_new.cov, [[0.8]])


def test_kf_update_perfect_measurement_collapses_covariance():
    """R→0 means we trust measurement fully; posterior variance → 0."""
    x = np.array([0.0])
    P = CovarianceMatrix(np.array([[100.0]]))
    H = np.array([[1.0]])
    R = CovarianceMatrix(np.array([[1e-12]]))
    zeta = np.array([5.0])

    x_new, P_new = kf_update(x, P, zeta, R, H)
    assert equal_to_tolerance(x_new, [5.0], tol=1e-6)
    assert P_new.cov[0, 0] < 1e-9


# ---------------------------------------------------------------------------
# kf_update — 2D state, position-only measurement
# ---------------------------------------------------------------------------

def test_kf_update_2d_state_position_only():
    """Position component updated; velocity component unchanged."""
    x = np.array([10.0, 5.0])
    P = CovarianceMatrix(np.diag([4.0, 1.0]))
    H = np.array([[1.0, 0.0]])   # measure position only
    R = CovarianceMatrix(np.array([[1.0]]))
    zeta = np.array([12.0])

    x_new, P_new = kf_update(x, P, zeta, R, H)
    # K = [4, 0]^T / 5;  x = [10+0.8*2, 5+0*2] = [11.6, 5.0]
    assert equal_to_tolerance(x_new, [11.6, 5.0])


def test_kf_update_2d_covariance_structure():
    """Position variance reduced; velocity variance unchanged."""
    x = np.array([10.0, 5.0])
    P = CovarianceMatrix(np.diag([4.0, 1.0]))
    H = np.array([[1.0, 0.0]])
    R = CovarianceMatrix(np.array([[1.0]]))
    zeta = np.array([12.0])

    _, P_new = kf_update(x, P, zeta, R, H)
    # (I - K@H) @ P = diag([0.8, 1.0])
    assert equal_to_tolerance(P_new.cov, np.diag([0.8, 1.0]))


def test_kf_update_returns_covariance_matrix_instance():
    x = np.array([0.0])
    P = CovarianceMatrix(np.eye(1))
    R = CovarianceMatrix(np.eye(1))
    H = np.eye(1)
    _, P_new = kf_update(x, P, np.array([1.0]), R, H)
    assert isinstance(P_new, CovarianceMatrix)


# ---------------------------------------------------------------------------
# ekf_update — linear case matches kf_update
# ---------------------------------------------------------------------------

def test_ekf_update_linear_matches_kf_update():
    """For a linear z_fun/h_fun, ekf_update and kf_update give identical results."""
    H = np.array([[1.0, 0.0]])
    x = np.array([10.0, 5.0])
    P = CovarianceMatrix(np.diag([4.0, 1.0]))
    R = CovarianceMatrix(np.array([[1.0]]))
    zeta = np.array([12.0])

    x_kf, P_kf = kf_update(x, P, zeta, R, H)
    x_ekf, P_ekf = ekf_update(x, P, zeta, R,
                               z_fun=lambda s: H @ s,
                               h_fun=lambda s: H)

    assert equal_to_tolerance(x_ekf, x_kf)
    assert equal_to_tolerance(P_ekf.cov, P_kf.cov)


def test_ekf_update_returns_covariance_matrix_instance():
    H = np.eye(1)
    x = np.array([0.0])
    P = CovarianceMatrix(np.eye(1))
    R = CovarianceMatrix(np.eye(1))
    _, P_new = ekf_update(x, P, np.array([1.0]), R,
                          z_fun=lambda s: H @ s,
                          h_fun=lambda s: H)
    assert isinstance(P_new, CovarianceMatrix)


# ---------------------------------------------------------------------------
# MeasurementModel.jacobian
# ---------------------------------------------------------------------------

def test_measurement_model_jacobian_position_only():
    """
    Position-only PSS with identity jacobian on a CV state space.
    Expected H = [[1,0,0,0],[0,1,0,0]]: measurements map to positions, zeros for velocity.
    """
    ss = make_cv_state_space()
    mm = MeasurementModel(state_space=ss, pss=_MockPSS())
    s = State(ss, time=0.0, state=np.array([1., 2., 0., 0.]))

    H = mm.jacobian(s)

    expected = np.array([[1., 0., 0., 0.],
                         [0., 1., 0., 0.]])
    assert H.shape == (2, 4)
    assert equal_to_tolerance(H, expected)


def test_measurement_model_jacobian_with_vel_rows():
    """
    PSS returns rows for both position and velocity.
    Expected: pos columns get pos rows, vel columns get vel rows.
    _MockPSSWithVel: j = [[1,0],[0,1],[0.5,0],[0,0.5]]
    After transpose and placement:
      h[:, :2] = [[1,0],[0,1]]    (pos)
      h[:, 2:] = [[0.5,0],[0,0.5]] (vel)
    """
    ss = make_cv_state_space()
    mm = MeasurementModel(state_space=ss, pss=_MockPSSWithVel())
    s = State(ss, time=0.0, state=np.array([1., 2., 0.5, 0.5]))

    H = mm.jacobian(s)

    expected = np.array([[1., 0., 0.5, 0. ],
                         [0., 1., 0.,  0.5]])
    assert H.shape == (2, 4)
    assert equal_to_tolerance(H, expected)


def test_measurement_model_jacobian_pos_only_state_space():
    """
    Position-only state space: H should be just the pos block (2x2 identity).
    """
    ss = make_pos_only_state_space()
    mm = MeasurementModel(state_space=ss, pss=_MockPSS())
    s = State(ss, time=0.0, state=np.array([3., 4.]))

    H = mm.jacobian(s)

    assert H.shape == (2, 2)
    assert equal_to_tolerance(H, np.eye(2))


# ---------------------------------------------------------------------------
# MeasurementModel.num_measurement_dimensions / num_state_dimensions
# ---------------------------------------------------------------------------

def test_measurement_model_num_measurement_dimensions():
    """num_measurement_dimensions matches pss.num_measurements."""
    ss = make_cv_state_space()
    mm = MeasurementModel(state_space=ss, pss=_MockPSSFull())
    assert mm.num_measurement_dimensions == 2


def test_measurement_model_num_state_dimensions():
    """num_state_dimensions matches state_space.num_states."""
    ss = make_cv_state_space()
    mm = MeasurementModel(state_space=ss, pss=_MockPSSFull())
    assert mm.num_state_dimensions == 4


# ---------------------------------------------------------------------------
# MeasurementModel.measurement (noiseless)
# ---------------------------------------------------------------------------

def test_measurement_model_measurement_returns_measurement_instance():
    ss = make_cv_state_space()
    mm = MeasurementModel(state_space=ss, pss=_MockPSSFull())
    s = State(ss, time=2.0, state=np.array([3., 4., 0.5, 0.5]))
    m = mm.measurement(s, noise=False)
    assert isinstance(m, Measurement)


def test_measurement_model_measurement_noiseless_zeta():
    """Noiseless measurement returns zeta equal to the state's position."""
    ss = make_cv_state_space()
    mm = MeasurementModel(state_space=ss, pss=_MockPSSFull())
    s = State(ss, time=2.0, state=np.array([3., 4., 0.5, 0.5]))
    m = mm.measurement(s, noise=False)
    assert equal_to_tolerance(m.zeta, [3., 4.])


def test_measurement_model_measurement_time_matches_state():
    """Returned Measurement carries the state's timestamp."""
    ss = make_cv_state_space()
    mm = MeasurementModel(state_space=ss, pss=_MockPSSFull())
    s = State(ss, time=7.5, state=np.array([1., 2., 0., 0.]))
    m = mm.measurement(s, noise=False)
    assert m.time == 7.5


# ---------------------------------------------------------------------------
# MeasurementModel.log_likelihood / log_likelihood_from_measurement
# ---------------------------------------------------------------------------

def test_measurement_model_log_likelihood_higher_at_truth():
    """LL must be higher when candidate state matches the observation state."""
    ss = make_cv_state_space()
    mm = MeasurementModel(state_space=ss, pss=_MockPSSFull())
    s_true   = State(ss, time=0., state=np.array([3., 4., 0., 0.]))
    s_offset = State(ss, time=0., state=np.array([6., 8., 0., 0.]))
    ll_true   = mm.log_likelihood(s_true, s_true)
    ll_offset = mm.log_likelihood(s_true, s_offset)
    assert ll_true > ll_offset


def test_measurement_model_log_likelihood_from_measurement_higher_at_match():
    """log_likelihood_from_measurement scores higher for a matching state."""
    ss = make_cv_state_space()
    mm = MeasurementModel(state_space=ss, pss=_MockPSSFull())
    s_true   = State(ss, time=0., state=np.array([3., 4., 0., 0.]))
    s_offset = State(ss, time=0., state=np.array([6., 8., 0., 0.]))
    m = mm.measurement(s_true, noise=False)
    ll_match  = mm.log_likelihood_from_measurement(s_true,   m)
    ll_offset = mm.log_likelihood_from_measurement(s_offset, m)
    assert ll_match > ll_offset
