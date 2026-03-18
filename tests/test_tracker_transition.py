import numpy as np
import pytest

from ewgeo.tracker.transition import (
    ConstantVelocityMotionModel,
    ConstantAccelerationMotionModel,
    ConstantJerkMotionModel,
    ConstantTurnMotionModel,
    MotionModel,
    kf_predict,
    ekf_predict,
)
from ewgeo.tracker.states import State
from ewgeo.utils.covariance import CovarianceMatrix


def equal_to_tolerance(x, y, tol=1e-10):
    return np.all(np.fabs(np.array(x) - np.array(y)) < tol)


# ---------------------------------------------------------------------------
# CV transition matrix
# ---------------------------------------------------------------------------

def test_cv_transition_matrix_1d_dt1():
    model = ConstantVelocityMotionModel(num_dims=1, process_covar=1.0)
    F = model.make_transition_matrix(time_delta=1.0)
    expected = np.array([[1., 1.],
                         [0., 1.]])
    assert equal_to_tolerance(F, expected)


def test_cv_transition_matrix_2d_dt2():
    model = ConstantVelocityMotionModel(num_dims=2, process_covar=1.0)
    F = model.make_transition_matrix(time_delta=2.0)
    expected = np.array([[1., 0., 2., 0.],
                         [0., 1., 0., 2.],
                         [0., 0., 1., 0.],
                         [0., 0., 0., 1.]])
    assert equal_to_tolerance(F, expected)


# ---------------------------------------------------------------------------
# CV process noise matrix
# ---------------------------------------------------------------------------

def test_cv_process_noise_1d_dt1():
    model = ConstantVelocityMotionModel(num_dims=1, process_covar=1.0)
    Q = model.make_process_covariance_matrix(process_covar=1.0, time_delta=1.0)
    expected = np.array([[0.25, 0.5],
                         [0.5,  1.0]])
    assert equal_to_tolerance(Q.cov, expected)


def test_cv_process_noise_scales_with_process_covar():
    model = ConstantVelocityMotionModel(num_dims=1, process_covar=2.0)
    Q = model.make_process_covariance_matrix(process_covar=2.0, time_delta=1.0)
    expected = np.array([[0.5, 1.0],
                         [1.0, 2.0]])
    assert equal_to_tolerance(Q.cov, expected)


# ---------------------------------------------------------------------------
# CA transition matrix
# ---------------------------------------------------------------------------

def test_ca_transition_matrix_1d_dt1():
    model = ConstantAccelerationMotionModel(num_dims=1, process_covar=1.0)
    F = model.make_transition_matrix(time_delta=1.0)
    expected = np.array([[1., 1., 0.5],
                         [0., 1., 1. ],
                         [0., 0., 1. ]])
    assert equal_to_tolerance(F, expected)


def test_ca_transition_matrix_1d_dt2():
    model = ConstantAccelerationMotionModel(num_dims=1, process_covar=1.0)
    F = model.make_transition_matrix(time_delta=2.0)
    expected = np.array([[1., 2., 2.],
                         [0., 1., 2.],
                         [0., 0., 1.]])
    assert equal_to_tolerance(F, expected)


# ---------------------------------------------------------------------------
# CA process noise matrix
# ---------------------------------------------------------------------------

def test_ca_process_noise_1d_dt1():
    model = ConstantAccelerationMotionModel(num_dims=1, process_covar=1.0)
    Q = model.make_process_covariance_matrix(process_covar=1.0, time_delta=1.0)
    expected = np.array([[0.25, 0.5, 0.5],
                         [0.5,  1.0, 1.0],
                         [0.5,  1.0, 1.0]])
    assert equal_to_tolerance(Q.cov, expected)


# ---------------------------------------------------------------------------
# CJ transition matrix
# ---------------------------------------------------------------------------

def test_cj_transition_matrix_1d_dt1():
    model = ConstantJerkMotionModel(num_dims=1, process_covar=1.0)
    F = model.make_transition_matrix(time_delta=1.0)
    expected = np.array([[1., 1., 0.5,     1/6],
                         [0., 1., 1.,      0.5],
                         [0., 0., 1.,      1. ],
                         [0., 0., 0.,      1. ]])
    assert equal_to_tolerance(F, expected)


# ---------------------------------------------------------------------------
# CJ process noise matrix
# ---------------------------------------------------------------------------

def test_cj_process_noise_1d_dt1():
    model = ConstantJerkMotionModel(num_dims=1, process_covar=1.0)
    Q = model.make_process_covariance_matrix(process_covar=1.0, time_delta=1.0)
    expected = np.array([[1/252, 1/72, 1/30, 1/24],
                         [1/72,  1/20, 1/8,  1/6 ],
                         [1/30,  1/8,  1/3,  1/2 ],
                         [1/24,  1/6,  1/2,  1.0 ]])
    assert equal_to_tolerance(Q.cov, expected)


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------

def test_predict_no_covariance():
    """predict() with no covariance leaves covar as None on the new state."""
    model = ConstantVelocityMotionModel(num_dims=1, process_covar=1.0)
    s = State(model.state_space, time=0.0, state=np.array([10., 5.]))
    s_new = model.predict(s, new_time=1.0)
    assert equal_to_tolerance(s_new.state, [15., 5.])
    assert s_new.covar is None
    assert s_new.time == 1.0


def test_predict_with_covariance():
    """predict() propagates covariance correctly: P_new = F P F^T + Q."""
    model = ConstantVelocityMotionModel(num_dims=1, process_covar=1.0)
    P0 = CovarianceMatrix(np.eye(2))
    s = State(model.state_space, time=0.0, state=np.array([10., 5.]), covar=P0)

    s_new = model.predict(s, new_time=1.0)

    F = np.array([[1., 1.], [0., 1.]])
    Q = np.array([[0.25, 0.5], [0.5, 1.0]])
    P_expected = F @ np.eye(2) @ F.T + Q  # [[2.25, 1.5], [1.5, 2.0]]

    assert equal_to_tolerance(s_new.state, [15., 5.])
    assert equal_to_tolerance(s_new.covar.cov, P_expected)


def test_predict_same_time_returns_unchanged_state():
    """predict() at the same time returns the original state unchanged."""
    model = ConstantVelocityMotionModel(num_dims=1, process_covar=1.0)
    s = State(model.state_space, time=5.0, state=np.array([10., 5.]))
    s_new = model.predict(s, new_time=5.0)
    assert s_new is s


def test_predict_preserves_state_space():
    """predict() new state uses the same state_space object."""
    model = ConstantVelocityMotionModel(num_dims=2, process_covar=1.0)
    s = State(model.state_space, time=0.0, state=np.array([1., 2., 3., 4.]))
    s_new = model.predict(s, new_time=1.0)
    assert s_new.state_space is s.state_space


# ---------------------------------------------------------------------------
# kf_predict()
# ---------------------------------------------------------------------------

def test_kf_predict_state_and_covariance():
    """kf_predict returns correct x_pred and P_pred."""
    F = np.array([[1., 1.], [0., 1.]])
    P = CovarianceMatrix(np.eye(2))
    Q = CovarianceMatrix(np.array([[0.25, 0.5], [0.5, 1.0]]))
    x = np.array([10., 5.])

    x_pred, P_pred = kf_predict(x, P, Q, F)

    assert equal_to_tolerance(x_pred, [15., 5.])
    P_expected = F @ np.eye(2) @ F.T + Q.cov
    assert equal_to_tolerance(P_pred.cov, P_expected)


def test_kf_predict_returns_covariance_matrix_instance():
    """kf_predict wraps the result in a CovarianceMatrix."""
    F = np.eye(2)
    P = CovarianceMatrix(np.eye(2))
    Q = CovarianceMatrix(np.eye(2))
    _, P_pred = kf_predict(np.zeros(2), P, Q, F)
    assert isinstance(P_pred, CovarianceMatrix)


# ---------------------------------------------------------------------------
# ekf_predict()
# ---------------------------------------------------------------------------

def test_ekf_predict_linear_case_matches_kf_predict():
    """For a linear system, ekf_predict and kf_predict produce identical results."""
    F = np.array([[1., 1.], [0., 1.]])
    P = CovarianceMatrix(np.eye(2))
    Q = CovarianceMatrix(np.array([[0.25, 0.5], [0.5, 1.0]]))
    x = np.array([10., 5.])

    x_kf, P_kf = kf_predict(x, P, Q, F)
    x_ekf, P_ekf = ekf_predict(x, P, Q, f_fun=lambda s: F @ s, g_fun=lambda s: F)

    assert equal_to_tolerance(x_ekf, x_kf)
    assert equal_to_tolerance(P_ekf.cov, P_kf.cov)


# ---------------------------------------------------------------------------
# make_motion_model() factory
# ---------------------------------------------------------------------------

def test_make_motion_model_cv():
    m = MotionModel.make_motion_model('cv', num_dims=2, process_covar=1.0)
    assert isinstance(m, ConstantVelocityMotionModel)
    assert m.num_dims == 2


def test_make_motion_model_ca():
    m = MotionModel.make_motion_model('ca', num_dims=2, process_covar=1.0)
    assert isinstance(m, ConstantAccelerationMotionModel)


def test_make_motion_model_cj():
    m = MotionModel.make_motion_model('cj', num_dims=2, process_covar=1.0)
    assert isinstance(m, ConstantJerkMotionModel)


def test_make_motion_model_long_names():
    assert isinstance(MotionModel.make_motion_model('constant_velocity', 1, 1.0),
                      ConstantVelocityMotionModel)
    assert isinstance(MotionModel.make_motion_model('constant_acceleration', 1, 1.0),
                      ConstantAccelerationMotionModel)
    assert isinstance(MotionModel.make_motion_model('constant_jerk', 1, 1.0),
                      ConstantJerkMotionModel)


def test_make_motion_model_invalid_raises():
    with pytest.raises(ValueError):
        MotionModel.make_motion_model('banana', num_dims=2, process_covar=1.0)


# ---------------------------------------------------------------------------
# ConstantTurnMotionModel — construction and properties
# ---------------------------------------------------------------------------

def test_ct_is_nonlinear():
    m = ConstantTurnMotionModel(num_dims=2, process_covar=1.0, process_covar_omega=0.1)
    assert m.is_linear is False


def test_ct_state_space_2d():
    m = ConstantTurnMotionModel(num_dims=2, process_covar=1.0, process_covar_omega=0.1)
    assert m.num_dims == 2
    assert m.num_states == 5          # [px, py, vx, vy, ω]
    assert m.state_space.has_turn_rate is True


def test_ct_state_space_3d():
    m = ConstantTurnMotionModel(num_dims=3, process_covar=1.0, process_covar_omega=0.1)
    assert m.num_dims == 3
    assert m.num_states == 7          # [px, py, pz, vx, vy, vz, ω]


def test_ct_invalid_num_dims_raises():
    with pytest.raises(ValueError):
        ConstantTurnMotionModel(num_dims=1, process_covar=1.0)


def test_ct_make_transition_matrix_raises():
    m = ConstantTurnMotionModel(num_dims=2, process_covar=1.0)
    with pytest.raises(NotImplementedError):
        m.make_transition_matrix(1.0)


# ---------------------------------------------------------------------------
# ConstantTurnMotionModel — transition function
# ---------------------------------------------------------------------------

def test_ct_zero_turn_rate_recovers_cv():
    """With ω = 0, CT reduces to constant velocity (straight line)."""
    m   = ConstantTurnMotionModel(num_dims=2, process_covar=1.0)
    dt  = 0.5
    x   = np.array([0., 0., 10., 5., 0.])    # [px, py, vx, vy, ω=0]
    f   = m.make_transition_function(dt)
    x_new = f(x)
    expected = np.array([5., 2.5, 10., 5., 0.])
    assert equal_to_tolerance(x_new, expected)


def test_ct_quarter_turn():
    """ω = π/2 rad/s, dt = 1 s → 90° turn; velocity rotates 90°."""
    m   = ConstantTurnMotionModel(num_dims=2, process_covar=1.0)
    dt  = 1.0
    om  = np.pi / 2
    x   = np.array([0., 0., 1., 0., om])   # moving in +x, turning left
    f   = m.make_transition_function(dt)
    x_new = f(x)
    # After 90° turn: velocity should point in +y
    assert equal_to_tolerance(x_new[2], 0., tol=1e-10)    # vx' ≈ 0
    assert equal_to_tolerance(x_new[3], 1., tol=1e-10)    # vy' = 1
    # ω unchanged
    assert equal_to_tolerance(x_new[4], om)


def test_ct_turn_rate_unchanged():
    """Turn rate ω is propagated unchanged by the transition."""
    m = ConstantTurnMotionModel(num_dims=2, process_covar=1.0)
    om = 0.3
    x  = np.array([1., 2., 3., 4., om])
    x_new = m.make_transition_function(0.1)(x)
    assert equal_to_tolerance(x_new[4], om)


def test_ct_3d_z_propagates_as_cv():
    """In 3D, the z-component should advance as pz' = pz + dt * vz."""
    m  = ConstantTurnMotionModel(num_dims=3, process_covar=1.0)
    dt = 0.5
    x  = np.array([0., 0., 0., 1., 0., 2., 0.1])   # vz=2, ω=0.1
    x_new = m.make_transition_function(dt)(x)
    assert equal_to_tolerance(x_new[2], 0. + 0.5 * 2.)   # pz' = 0 + 0.5*2 = 1.0
    assert equal_to_tolerance(x_new[5], 2.)               # vz unchanged


# ---------------------------------------------------------------------------
# ConstantTurnMotionModel — Jacobian
# ---------------------------------------------------------------------------

def test_ct_jacobian_shape_2d():
    m = ConstantTurnMotionModel(num_dims=2, process_covar=1.0)
    x = np.array([0., 0., 1., 0., 0.5])
    J = m.make_jacobian(x, time_delta=0.1)
    assert J.shape == (5, 5)


def test_ct_jacobian_shape_3d():
    m = ConstantTurnMotionModel(num_dims=3, process_covar=1.0)
    x = np.array([0., 0., 0., 1., 0., 0., 0.5])
    J = m.make_jacobian(x, time_delta=0.1)
    assert J.shape == (7, 7)


def test_ct_jacobian_zero_omega_matches_cv():
    """At ω = 0 the Jacobian should match the CV transition matrix."""
    m  = ConstantTurnMotionModel(num_dims=2, process_covar=1.0)
    dt = 1.0
    x  = np.array([0., 0., 1., 0., 0.])   # vx=1, ω=0
    J  = m.make_jacobian(x, time_delta=dt)
    # CV F for 2D, dt=1: [[1,0,1,0,*],[0,1,0,1,*],[0,0,1,0,*],[0,0,0,1,*],[0,0,0,0,1]]
    assert equal_to_tolerance(J[0, 2], 1.0)   # ∂px'/∂vx = dt = 1
    assert equal_to_tolerance(J[0, 3], 0.0)   # ∂px'/∂vy = 0  (no rotation)
    assert equal_to_tolerance(J[2, 2], 1.0)   # ∂vx'/∂vx = cos(0) = 1
    assert equal_to_tolerance(J[2, 3], 0.0)   # ∂vx'/∂vy = -sin(0) = 0


def test_ct_jacobian_numerical_check():
    """Jacobian matches finite-difference approximation."""
    m   = ConstantTurnMotionModel(num_dims=2, process_covar=1.0)
    dt  = 0.3
    x   = np.array([1., 2., 3., 4., 0.5])
    J   = m.make_jacobian(x, time_delta=dt)
    f   = m.make_transition_function(dt)
    eps = 1e-6
    J_fd = np.zeros((5, 5))
    for i in range(5):
        xp, xm = x.copy(), x.copy()
        xp[i] += eps
        xm[i] -= eps
        J_fd[:, i] = (f(xp) - f(xm)) / (2 * eps)
    assert equal_to_tolerance(J, J_fd, tol=1e-6)


# ---------------------------------------------------------------------------
# ConstantTurnMotionModel — process noise
# ---------------------------------------------------------------------------

def test_ct_process_noise_shape_2d():
    m = ConstantTurnMotionModel(num_dims=2, process_covar=1.0, process_covar_omega=0.5)
    Q = m.make_process_covariance_matrix(time_delta=1.0)
    assert Q.cov.shape == (5, 5)


def test_ct_process_noise_shape_3d():
    m = ConstantTurnMotionModel(num_dims=3, process_covar=1.0, process_covar_omega=0.5)
    Q = m.make_process_covariance_matrix(time_delta=1.0)
    assert Q.cov.shape == (7, 7)


def test_ct_process_noise_omega_block():
    """The turn-rate variance element equals process_covar_omega * dt."""
    dt = 0.5
    m  = ConstantTurnMotionModel(num_dims=2, process_covar=1.0, process_covar_omega=0.4)
    Q  = m.make_process_covariance_matrix(time_delta=dt)
    assert equal_to_tolerance(Q.cov[-1, -1], 0.4 * dt)


# ---------------------------------------------------------------------------
# ConstantTurnMotionModel — predict()
# ---------------------------------------------------------------------------

def test_ct_predict_propagates_state():
    """predict() returns a State object with the nonlinear transition applied."""
    m   = ConstantTurnMotionModel(num_dims=2, process_covar=1.0, process_covar_omega=0.1)
    ss  = m.state_space
    dt  = 1.0
    om  = np.pi / 2
    x0  = np.array([0., 0., 1., 0., om])
    s0  = State(ss, time=0.0, state=x0)
    s1  = m.predict(s0, new_time=dt)
    assert isinstance(s1, State)
    assert s1.time == dt
    # velocity should have rotated 90°
    assert equal_to_tolerance(s1.state[2], 0., tol=1e-10)
    assert equal_to_tolerance(s1.state[3], 1., tol=1e-10)


def test_ct_predict_propagates_covariance():
    """predict() propagates covariance via the EKF Jacobian."""
    m   = ConstantTurnMotionModel(num_dims=2, process_covar=1.0, process_covar_omega=0.1)
    ss  = m.state_space
    P0  = CovarianceMatrix(np.eye(5))
    x0  = np.array([0., 0., 1., 0., 0.1])
    s0  = State(ss, time=0.0, state=x0, covar=P0)
    s1  = m.predict(s0, new_time=0.5)
    assert s1.covar is not None
    # Posterior covariance must be positive definite (all eigenvalues > 0)
    assert np.all(np.linalg.eigvalsh(s1.covar.cov) > 0)


def test_ct_predict_no_covar_returns_none_covar():
    """If the input State has no covariance, predict() returns a State with no covariance."""
    m  = ConstantTurnMotionModel(num_dims=2, process_covar=1.0, process_covar_omega=0.1)
    ss = m.state_space
    s0 = State(ss, time=0.0, state=np.array([0., 0., 1., 0., 0.2]))
    s1 = m.predict(s0, new_time=0.5)
    assert s1.covar is None


def test_make_motion_model_ct():
    m = MotionModel.make_motion_model('ct', num_dims=2, process_covar=1.0)
    assert isinstance(m, ConstantTurnMotionModel)


def test_make_motion_model_constant_turn():
    m = MotionModel.make_motion_model('constant_turn', num_dims=2, process_covar=1.0)
    assert isinstance(m, ConstantTurnMotionModel)
