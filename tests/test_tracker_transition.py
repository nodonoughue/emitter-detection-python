import numpy as np
import pytest

from ewgeo.tracker.transition import (
    ConstantVelocityMotionModel,
    ConstantAccelerationMotionModel,
    ConstantJerkMotionModel,
    ConstantTurnMotionModel,
    BallisticMotionModel,
    ConstantTurnRateAccelerationMotionModel,
    MotionModel,
    kf_predict,
    ekf_predict,
    ukf_predict,
)
from ewgeo.tracker.states import State, CartesianStateSpace
from ewgeo.utils.covariance import CovarianceMatrix

# 1-D constant-velocity state space: [pos, vel] (2 states)
_SS_1D_CV = CartesianStateSpace(num_dims=1, has_vel=True)


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
    expected = np.array([[1/36,  1/12, 1/6,  1/6 ],
                         [1/12,  1/4,  1/2,  1/2 ],
                         [1/6,   1/2,  1.0,  1.0 ],
                         [1/6,   1/2,  1.0,  1.0 ]])
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

    result = kf_predict(State(_SS_1D_CV, 0., x, P), Q, F)

    assert equal_to_tolerance(result.state, [15., 5.])
    P_expected = F @ np.eye(2) @ F.T + Q.cov
    assert equal_to_tolerance(result.covar.cov, P_expected)


def test_kf_predict_returns_covariance_matrix_instance():
    """kf_predict wraps the result in a CovarianceMatrix."""
    F = np.eye(2)
    P = CovarianceMatrix(np.eye(2))
    Q = CovarianceMatrix(np.eye(2))
    result = kf_predict(State(_SS_1D_CV, 0., np.zeros(2), P), Q, F)
    assert isinstance(result.covar, CovarianceMatrix)


# ---------------------------------------------------------------------------
# ekf_predict()
# ---------------------------------------------------------------------------

def test_ekf_predict_linear_case_matches_kf_predict():
    """For a linear system, ekf_predict and kf_predict produce identical results."""
    F = np.array([[1., 1.], [0., 1.]])
    P = CovarianceMatrix(np.eye(2))
    Q = CovarianceMatrix(np.array([[0.25, 0.5], [0.5, 1.0]]))
    x = np.array([10., 5.])

    s_est = State(_SS_1D_CV, 0., x, P)
    result_kf  = kf_predict(s_est, Q, F)
    result_ekf = ekf_predict(s_est, Q,
                             transition_fun=lambda s, dt: State(s.state_space, s.time + dt, F @ s.state, s.covar),
                             jacobian_fun=lambda s, dt: F,
                             time_step=0.0)

    assert equal_to_tolerance(result_ekf.state, result_kf.state)
    assert equal_to_tolerance(result_ekf.covar.cov, result_kf.covar.cov)


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
    f   = _ct_f_fun(m, dt)
    x_new = f(x)
    expected = np.array([5., 2.5, 10., 5., 0.])
    assert equal_to_tolerance(x_new, expected)


def test_ct_quarter_turn():
    """ω = π/2 rad/s, dt = 1 s → 90° turn; velocity rotates 90°."""
    m   = ConstantTurnMotionModel(num_dims=2, process_covar=1.0)
    dt  = 1.0
    om  = np.pi / 2
    x   = np.array([0., 0., 1., 0., om])   # moving in +x, turning left
    f   = _ct_f_fun(m, dt)
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
    x_new = _ct_f_fun(m, 0.1)(x)
    assert equal_to_tolerance(x_new[4], om)


def test_ct_3d_z_propagates_as_cv():
    """In 3D, the z-component should advance as pz' = pz + dt * vz."""
    m  = ConstantTurnMotionModel(num_dims=3, process_covar=1.0)
    dt = 0.5
    x  = np.array([0., 0., 0., 1., 0., 2., 0.1])   # vz=2, ω=0.1
    x_new = _ct_f_fun(m, dt)(x)
    assert equal_to_tolerance(x_new[2], 0. + 0.5 * 2.)   # pz' = 0 + 0.5*2 = 1.0
    assert equal_to_tolerance(x_new[5], 2.)               # vz unchanged


# ---------------------------------------------------------------------------
# ConstantTurnMotionModel — Jacobian
# ---------------------------------------------------------------------------

def test_ct_jacobian_shape_2d():
    m = ConstantTurnMotionModel(num_dims=2, process_covar=1.0)
    x = np.array([0., 0., 1., 0., 0.5])
    J = m.transition_matrix(State(m.state_space, 0., x, None), time_delta=0.1)
    assert J.shape == (5, 5)


def test_ct_jacobian_shape_3d():
    m = ConstantTurnMotionModel(num_dims=3, process_covar=1.0)
    x = np.array([0., 0., 0., 1., 0., 0., 0.5])
    J = m.transition_matrix(State(m.state_space, 0., x, None), time_delta=0.1)
    assert J.shape == (7, 7)


def test_ct_jacobian_zero_omega_matches_cv():
    """At ω = 0 the Jacobian should match the CV transition matrix."""
    m  = ConstantTurnMotionModel(num_dims=2, process_covar=1.0)
    dt = 1.0
    x  = np.array([0., 0., 1., 0., 0.])   # vx=1, ω=0
    J  = m.transition_matrix(State(m.state_space, 0., x, None), time_delta=dt)
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
    J   = m.transition_matrix(State(m.state_space, 0., x, None), time_delta=dt)
    f   = _ct_f_fun(m, dt)
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


# ---------------------------------------------------------------------------
# BallisticMotionModel — construction and properties
# ---------------------------------------------------------------------------

def test_ballistic_is_linear():
    m = BallisticMotionModel(process_covar=1.0)
    assert m.is_linear is True


def test_ballistic_state_space():
    m = BallisticMotionModel(process_covar=1.0)
    assert m.num_dims == 3
    assert m.num_states == 6       # [px, py, pz, vx, vy, vz]
    assert m.state_space.has_vel is True
    assert m.state_space.has_accel is False


def test_ballistic_default_gravity_vec():
    m = BallisticMotionModel(process_covar=1.0)
    assert equal_to_tolerance(m.gravity_vec, [0., 0., -9.80665])


def test_ballistic_scalar_gravity():
    m = BallisticMotionModel(process_covar=1.0, gravity=-5.0)
    assert equal_to_tolerance(m.gravity_vec, [0., 0., -5.0])


def test_ballistic_vector_gravity():
    g = np.array([0., -9.81, 0.])
    m = BallisticMotionModel(process_covar=1.0, gravity=g)
    assert equal_to_tolerance(m.gravity_vec, g)


def test_ballistic_invalid_gravity_raises():
    with pytest.raises(ValueError):
        BallisticMotionModel(process_covar=1.0, gravity=np.array([1., 2.]))


# ---------------------------------------------------------------------------
# BallisticMotionModel — transition matrix and process noise
# ---------------------------------------------------------------------------

def test_ballistic_transition_matrix_matches_cv_3d():
    """Transition matrix must equal the 3-D CV matrix."""
    m   = BallisticMotionModel(process_covar=1.0)
    cv  = ConstantVelocityMotionModel(num_dims=3, process_covar=1.0)
    dt  = 0.5
    assert equal_to_tolerance(m.make_transition_matrix(dt),
                               cv.make_transition_matrix(dt))


def test_ballistic_process_noise_matches_cv_3d():
    """Process noise matrix must equal the 3-D CV Q."""
    m   = BallisticMotionModel(process_covar=1.0)
    cv  = ConstantVelocityMotionModel(num_dims=3, process_covar=1.0)
    dt  = 0.5
    assert equal_to_tolerance(m.make_process_covariance_matrix(time_delta=dt).cov,
                               cv.make_process_covariance_matrix(time_delta=dt).cov)


# ---------------------------------------------------------------------------
# BallisticMotionModel — predict()
# ---------------------------------------------------------------------------

def test_ballistic_predict_zero_gravity_matches_cv():
    """With gravity=0 the ballistic model must give identical results to CV."""
    dt = 1.0
    x0 = np.array([0., 0., 100., 10., 0., 50.])
    m_b  = BallisticMotionModel(process_covar=1.0, gravity=0.0, time_delta=dt)
    m_cv = ConstantVelocityMotionModel(num_dims=3, process_covar=1.0, time_delta=dt)
    ss   = m_b.state_space
    s0   = State(ss, time=0.0, state=x0)
    s_b  = m_b.predict(s0, new_time=dt)
    s_cv = m_cv.predict(s0, new_time=dt)
    assert equal_to_tolerance(s_b.state, s_cv.state)


def test_ballistic_predict_vz_decremented_by_gravity():
    """vz must decrease by g*dt each step."""
    g  = -9.80665
    dt = 1.0
    m  = BallisticMotionModel(process_covar=1.0, gravity=g, time_delta=dt)
    ss = m.state_space
    x0 = np.array([0., 0., 0., 0., 0., 0.])
    s0 = State(ss, time=0.0, state=x0)
    s1 = m.predict(s0, new_time=dt)
    assert equal_to_tolerance(s1.state[5], g * dt)     # vz' = g*dt


def test_ballistic_predict_pz_parabolic():
    """pz must follow the parabolic free-fall equation pz' = pz + vz*dt + 0.5*g*dt²."""
    g  = -9.80665
    dt = 2.0
    vz0 = 50.0
    pz0 = 1000.0
    m  = BallisticMotionModel(process_covar=1.0, gravity=g, time_delta=dt)
    ss = m.state_space
    x0 = np.array([0., 0., pz0, 0., 0., vz0])
    s0 = State(ss, time=0.0, state=x0)
    s1 = m.predict(s0, new_time=dt)
    expected_pz = pz0 + vz0 * dt + 0.5 * g * dt ** 2
    assert equal_to_tolerance(s1.state[2], expected_pz)


def test_ballistic_predict_xy_unaffected_by_gravity():
    """Horizontal components must be unaffected by gravity."""
    dt = 1.0
    m  = BallisticMotionModel(process_covar=1.0, time_delta=dt)
    ss = m.state_space
    x0 = np.array([10., 20., 0., 3., 4., 0.])
    s0 = State(ss, time=0.0, state=x0)
    s1 = m.predict(s0, new_time=dt)
    assert equal_to_tolerance(s1.state[0], 10. + 3. * dt)   # px' = px + vx*dt
    assert equal_to_tolerance(s1.state[1], 20. + 4. * dt)   # py' = py + vy*dt
    assert equal_to_tolerance(s1.state[3], 3.)               # vx unchanged
    assert equal_to_tolerance(s1.state[4], 4.)               # vy unchanged


def test_ballistic_predict_covariance_positive_definite():
    """Posterior covariance must be positive definite."""
    m  = BallisticMotionModel(process_covar=1.0)
    ss = m.state_space
    P0 = CovarianceMatrix(np.eye(6))
    x0 = np.array([0., 0., 1000., 10., 0., 50.])
    s0 = State(ss, time=0.0, state=x0, covar=P0)
    s1 = m.predict(s0, new_time=0.5)
    assert np.all(np.linalg.eigvalsh(s1.covar.cov) > 0)


def test_ballistic_predict_covariance_matches_cv():
    """Covariance must propagate identically to CV (gravity does not affect covariance)."""
    dt = 1.0
    x0 = np.array([0., 0., 0., 1., 1., 1.])
    P0 = CovarianceMatrix(np.eye(6))
    m_b  = BallisticMotionModel(process_covar=1.0, time_delta=dt)
    m_cv = ConstantVelocityMotionModel(num_dims=3, process_covar=1.0, time_delta=dt)
    ss   = m_b.state_space
    s0   = State(ss, time=0.0, state=x0, covar=P0)
    s_b  = m_b.predict(s0, new_time=dt)
    s_cv = m_cv.predict(s0, new_time=dt)
    assert equal_to_tolerance(s_b.covar.cov, s_cv.covar.cov)


def test_make_motion_model_ballistic():
    m = MotionModel.make_motion_model('ballistic', num_dims=3, process_covar=1.0)
    assert isinstance(m, BallisticMotionModel)


# ---------------------------------------------------------------------------
# ConstantTurnRateAccelerationMotionModel — construction and properties
# ---------------------------------------------------------------------------

def test_ctra_is_not_linear():
    m = ConstantTurnRateAccelerationMotionModel(num_dims=2, process_covar=1.0)
    assert m.is_linear is False


def test_ctra_state_space_2d():
    from ewgeo.tracker.states import PolarKinematicStateSpace
    m = ConstantTurnRateAccelerationMotionModel(num_dims=2, process_covar=1.0)
    assert isinstance(m.state_space, PolarKinematicStateSpace)
    assert m.state_space.num_states == 7    # [px,py,vx,vy,ax,ay,ω]
    assert m.state_space.has_accel is True
    assert m.state_space.has_turn_rate is True


def test_ctra_state_space_3d():
    m = ConstantTurnRateAccelerationMotionModel(num_dims=3, process_covar=1.0)
    assert m.state_space.num_states == 10   # [px,py,pz,vx,vy,vz,ax,ay,az,ω]


def test_ctra_invalid_num_dims_raises():
    with pytest.raises(ValueError):
        ConstantTurnRateAccelerationMotionModel(num_dims=1, process_covar=1.0)


def test_ctra_make_transition_matrix_raises():
    m = ConstantTurnRateAccelerationMotionModel(num_dims=2, process_covar=1.0)
    with pytest.raises(NotImplementedError):
        m.make_transition_matrix()


# ---------------------------------------------------------------------------
# ConstantTurnRateAccelerationMotionModel — transition function
# ---------------------------------------------------------------------------

def test_ctra_transition_zero_omega_equals_ca_2d():
    """At ω=0 CTRA transition collapses to CA kinematics."""
    dt = 1.0
    m  = ConstantTurnRateAccelerationMotionModel(num_dims=2, process_covar=1.0)
    # state: [px, py, vx, vy, ax, ay, ω]
    x  = np.array([0., 0., 10., 5., 2., 1., 0.])
    f  = m.make_transition_function(dt)
    x_new = f(x)
    # CA expected (ω=0, fallback sow=dt, com=0):
    # px' = 0 + dt*10 + 0.5*dt²*2 = 11
    # py' = 0 + dt*5  + 0.5*dt²*1 = 5.5
    # vx' = 10 + dt*2 = 12
    # vy' =  5 + dt*1 = 6
    assert equal_to_tolerance(x_new[:2], np.array([11., 5.5]))
    assert equal_to_tolerance(x_new[2:4], np.array([12., 6.]))
    assert equal_to_tolerance(x_new[4:6], np.array([2., 1.]))   # accel unchanged
    assert equal_to_tolerance(x_new[-1], 0.0)                   # ω unchanged


def test_ctra_transition_zero_accel_equals_ct_2d():
    """At ax=ay=0 CTRA transition collapses to CT kinematics."""
    dt    = 0.5
    omega = np.pi / 4   # 45 deg/s
    vx, vy = 10., 0.
    m  = ConstantTurnRateAccelerationMotionModel(num_dims=2, process_covar=1.0)
    x  = np.array([0., 0., vx, vy, 0., 0., omega])
    f  = m.make_transition_function(dt)
    x_new = f(x)
    odt   = omega * dt
    sow   = np.sin(odt) / omega
    com   = (1.0 - np.cos(odt)) / omega
    # Pure CT prediction
    assert equal_to_tolerance(x_new[0], sow * vx - com * vy, tol=1e-9)
    assert equal_to_tolerance(x_new[1], com * vx + sow * vy, tol=1e-9)
    assert equal_to_tolerance(x_new[2], np.cos(odt) * vx - np.sin(odt) * vy, tol=1e-9)
    assert equal_to_tolerance(x_new[3], np.sin(odt) * vx + np.cos(odt) * vy, tol=1e-9)


def test_ctra_transition_3d_z_propagates_as_ca():
    """3D z sub-system uses CA (pz' = pz + dt*vz + 0.5*dt²*az, vz' = vz + dt*az)."""
    dt = 1.0
    m  = ConstantTurnRateAccelerationMotionModel(num_dims=3, process_covar=1.0)
    # state: [px,py,pz, vx,vy,vz, ax,ay,az, ω]
    x  = np.zeros(10)
    x[5] = 5.     # vz
    x[8] = -2.    # az
    f  = m.make_transition_function(dt)
    x_new = f(x)
    assert equal_to_tolerance(x_new[2], 0 + 1.0 * 5. + 0.5 * 1. ** 2 * (-2.), tol=1e-9)  # pz'
    assert equal_to_tolerance(x_new[5], 5. + 1.0 * (-2.), tol=1e-9)                       # vz'


def test_ctra_transition_function_returns_state():
    """transition_function() returns a State with advanced time and correct state values."""
    dt = 1.0
    m  = ConstantTurnRateAccelerationMotionModel(num_dims=2, process_covar=1.0)
    x0 = np.array([1., 2., 5., 3., 0.5, -0.5, 0.0])
    s0 = State(m.state_space, time=3.0, state=x0, covar=None)
    s1 = m.transition_function(s0, dt)
    assert isinstance(s1, State)
    assert s1.time == 3.0 + dt
    # Values must match the raw make_transition_function output
    f  = m.make_transition_function(dt)
    assert equal_to_tolerance(s1.state, f(x0))


def test_ctra_transition_function_caches_q():
    """transition_function() caches self.q when time_delta changes."""
    m  = ConstantTurnRateAccelerationMotionModel(num_dims=2, process_covar=1.0,
                                                 process_covar_omega=0.2)
    x0 = np.array([0., 0., 1., 0., 0., 0., 0.1])
    s0 = State(m.state_space, time=0.0, state=x0, covar=None)
    assert m.q is None    # nothing cached yet
    m.transition_function(s0, 1.0)
    assert m.q is not None
    assert m.q.cov.shape == (7, 7)
    # Bottom-right element should equal process_covar_omega * dt = 0.2 * 1.0
    assert equal_to_tolerance(m.q.cov[-1, -1], 0.2 * 1.0)


def test_ctra_transition_function_updates_q_on_new_dt():
    """transition_function() refreshes self.q when called with a different time_delta."""
    m  = ConstantTurnRateAccelerationMotionModel(num_dims=2, process_covar=1.0,
                                                 process_covar_omega=1.0)
    x0 = np.array([0., 0., 1., 0., 0., 0., 0.0])
    s0 = State(m.state_space, time=0.0, state=x0, covar=None)
    m.transition_function(s0, 1.0)
    q_before = m.q.cov[-1, -1]   # should be 1.0 * 1.0 = 1.0
    s1 = State(m.state_space, time=1.0, state=m.transition_function(s0, 1.0).state, covar=None)
    m.transition_function(s1, 2.0)
    q_after = m.q.cov[-1, -1]    # should be 1.0 * 2.0 = 2.0
    assert equal_to_tolerance(q_before, 1.0)
    assert equal_to_tolerance(q_after,  2.0)


# ---------------------------------------------------------------------------
# ConstantTurnRateAccelerationMotionModel — Jacobian
# ---------------------------------------------------------------------------

def test_ctra_jacobian_shape_2d():
    m = ConstantTurnRateAccelerationMotionModel(num_dims=2, process_covar=1.0)
    x = np.array([0., 0., 5., 0., 1., 0., 0.1])
    J = m.transition_matrix(x, time_delta=1.0)
    assert J.shape == (7, 7)


def test_ctra_jacobian_shape_3d():
    m = ConstantTurnRateAccelerationMotionModel(num_dims=3, process_covar=1.0)
    x = np.zeros(10)
    J = m.transition_matrix(x, time_delta=1.0)
    assert J.shape == (10, 10)


def test_ctra_jacobian_omega_zero_limit():
    """Jacobian at ω=0 should not contain NaN or Inf (Taylor fallback)."""
    m = ConstantTurnRateAccelerationMotionModel(num_dims=2, process_covar=1.0)
    x = np.array([0., 0., 5., 3., 1., 0.5, 0.])
    J = m.transition_matrix(x, time_delta=1.0)
    assert not np.any(np.isnan(J))
    assert not np.any(np.isinf(J))


def test_ctra_jacobian_finite_difference_2d():
    """Jacobian should match finite-difference approximation."""
    dt    = 0.5
    omega = 0.3
    m     = ConstantTurnRateAccelerationMotionModel(num_dims=2, process_covar=1.0)
    x0    = np.array([1., 2., 4., -3., 0.5, -0.5, omega])
    f     = m.make_transition_function(dt)
    J_an  = m.transition_matrix(x0, dt)

    eps  = 1e-5
    ns   = len(x0)
    J_fd = np.zeros((ns, ns))
    for j in range(ns):
        xp = x0.copy(); xp[j] += eps
        xm = x0.copy(); xm[j] -= eps
        J_fd[:, j] = (f(xp) - f(xm)) / (2 * eps)

    assert np.allclose(J_an, J_fd, atol=1e-6)


# ---------------------------------------------------------------------------
# ConstantTurnRateAccelerationMotionModel — process noise
# ---------------------------------------------------------------------------

def test_ctra_process_noise_shape_2d():
    m = ConstantTurnRateAccelerationMotionModel(num_dims=2, process_covar=1.0,
                                                process_covar_omega=0.1)
    Q = m.make_process_covariance_matrix(time_delta=1.0)
    assert Q.cov.shape == (7, 7)


def test_ctra_process_noise_shape_3d():
    m = ConstantTurnRateAccelerationMotionModel(num_dims=3, process_covar=1.0)
    Q = m.make_process_covariance_matrix(time_delta=1.0)
    assert Q.cov.shape == (10, 10)


def test_ctra_process_noise_omega_block():
    """Bottom-right element of Q should equal process_covar_omega * dt."""
    dt = 2.0
    pw = 0.25
    m  = ConstantTurnRateAccelerationMotionModel(num_dims=2, process_covar=1.0,
                                                 process_covar_omega=pw)
    Q  = m.make_process_covariance_matrix(time_delta=dt)
    assert equal_to_tolerance(Q.cov[-1, -1], pw * dt)


def test_ctra_kinematic_block_matches_ca_2d():
    """Top-left 6×6 of CTRA Q should equal the CA Q for the same parameters."""
    dt = 1.0
    m_ctra = ConstantTurnRateAccelerationMotionModel(num_dims=2, process_covar=1.0)
    m_ca   = ConstantAccelerationMotionModel(num_dims=2, process_covar=1.0)
    Q_ctra = m_ctra.make_process_covariance_matrix(time_delta=dt).cov
    Q_ca   = m_ca.make_process_covariance_matrix(time_delta=dt).cov
    assert equal_to_tolerance(Q_ctra[:6, :6], Q_ca)


# ---------------------------------------------------------------------------
# ConstantTurnRateAccelerationMotionModel — predict()
# ---------------------------------------------------------------------------

def test_ctra_predict_state_shape_2d():
    dt = 1.0
    m  = ConstantTurnRateAccelerationMotionModel(num_dims=2, process_covar=1.0)
    x0 = np.array([0., 0., 5., 0., 0., 0., 0.2])
    cov = CovarianceMatrix(np.eye(7))
    s  = State(m.state_space, time=0.0, state=x0, covar=cov)
    s_new = m.predict(s, dt)
    assert s_new.state.shape == (7,)
    assert s_new.covar.cov.shape == (7, 7)


def test_ctra_predict_covariance_positive_definite():
    dt = 1.0
    m  = ConstantTurnRateAccelerationMotionModel(num_dims=2, process_covar=1.0,
                                                 process_covar_omega=0.1)
    x0 = np.array([0., 0., 5., 3., 1., 0., 0.2])
    cov = CovarianceMatrix(np.eye(7))
    s  = State(m.state_space, time=0.0, state=x0, covar=cov)
    s_new = m.predict(s, dt)
    eigvals = np.linalg.eigvalsh(s_new.covar.cov)
    assert np.all(eigvals > -1e-10)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def test_make_motion_model_ctra():
    m = MotionModel.make_motion_model('ctra', num_dims=2, process_covar=1.0)
    assert isinstance(m, ConstantTurnRateAccelerationMotionModel)


def test_make_motion_model_ctra_long_key():
    m = MotionModel.make_motion_model('constant_turn_rate_acceleration',
                                      num_dims=3, process_covar=1.0)
    assert isinstance(m, ConstantTurnRateAccelerationMotionModel)


# ---------------------------------------------------------------------------
# ukf_predict
# ---------------------------------------------------------------------------

def _cv_f_fun(F):
    """Return a closure that applies the linear CV transition matrix."""
    return lambda x: F @ x


def _ct_f_fun(m, dt):
    """Return a closure that applies the CT nonlinear transition to a state vector (ndarray → ndarray)."""
    def f(x_arr):
        s_tmp = State(m.state_space, 0., x_arr, None)
        return m.transition_function(s_tmp, dt).state
    return f


def test_ukf_predict_output_shapes():
    """ukf_predict returns a State with x of shape (n,) and CovarianceMatrix of size (n,n)."""
    m  = ConstantVelocityMotionModel(num_dims=2, process_covar=1.0)
    dt = 1.0
    m.update_time_step(dt)
    x  = np.array([0., 0., 1., 0.])
    P  = CovarianceMatrix(np.eye(4))
    f  = _cv_f_fun(m.f)
    result = ukf_predict(State(m.state_space, 0., x, P), m.q, f)
    assert result.state.shape == (4,)
    assert result.covar.cov.shape == (4, 4)


def test_ukf_predict_linear_matches_kf():
    """For a linear transition function UKF and KF should agree to machine precision."""
    m  = ConstantVelocityMotionModel(num_dims=3, process_covar=1.0)
    dt = 0.5
    m.update_time_step(dt)
    x  = np.array([100., -50., 200., 10., -5., 0.])
    P  = CovarianceMatrix(np.diag([1e4, 1e4, 1e4, 1e2, 1e2, 1e2]))
    f_fun = _cv_f_fun(m.f)

    s_est = State(m.state_space, 0., x, P)
    result_kf  = kf_predict(s_est, m.q, m.f)
    result_ukf = ukf_predict(s_est, m.q, f_fun, alpha=1.0, beta=2., kappa=0.)

    assert np.allclose(result_ukf.state, result_kf.state, atol=1e-8)
    assert np.allclose(result_ukf.covar.cov, result_kf.covar.cov, atol=1e-8)


def test_ukf_predict_nonlinear_ct_close_to_ekf():
    """For a mildly nonlinear CT step UKF and EKF should give close (not identical) results."""
    dt    = 1.0
    omega = 0.1   # rad/s — mild nonlinearity
    m     = ConstantTurnMotionModel(num_dims=2, process_covar=1.0,
                                    process_covar_omega=0.01)
    m.update_time_step(dt)

    x = np.array([0., 0., 10., 0., omega])
    P = CovarianceMatrix(np.diag([1e4, 1e4, 1e2, 1e2, 0.01]))

    f_fun_ukf = _ct_f_fun(m, dt)

    s_est = State(m.state_space, 0., x, P)
    result_ekf = ekf_predict(s_est, m.q, m.transition_function, m.transition_matrix, time_step=dt)
    result_ukf = ukf_predict(s_est, m.q, f_fun_ukf, time_step=dt)

    # Means should agree closely for mild nonlinearity
    assert np.allclose(result_ukf.state, result_ekf.state, atol=0.5)
    # Covariances should have the same order of magnitude
    assert np.allclose(np.diag(result_ukf.covar.cov), np.diag(result_ekf.covar.cov), rtol=0.1)


def test_ukf_predict_covariance_positive_semidefinite():
    """Predicted covariance from UKF should be positive semidefinite."""
    m  = ConstantTurnMotionModel(num_dims=2, process_covar=2.0,
                                 process_covar_omega=0.01)
    dt = 2.0
    m.update_time_step(dt)
    x  = np.array([0., 0., 15., 5., 0.05])
    P  = CovarianceMatrix(np.diag([1e4, 1e4, 1e2, 1e2, 0.01]))
    f  = _ct_f_fun(m, dt)
    result = ukf_predict(State(m.state_space, 0., x, P), m.q, f, time_step=dt)
    eigvals = np.linalg.eigvalsh(result.covar.cov)
    assert np.all(eigvals > -1e-10)


def test_ukf_predict_alpha_affects_result():
    """Changing alpha should change the predicted covariance for a nonlinear model."""
    m  = ConstantTurnMotionModel(num_dims=2, process_covar=1.0,
                                 process_covar_omega=0.01)
    dt = 1.0
    m.update_time_step(dt)
    x  = np.array([0., 0., 10., 0., 0.3])   # larger ω → stronger nonlinearity
    P  = CovarianceMatrix(np.diag([1e4, 1e4, 1e2, 1e2, 0.1]))
    f  = _ct_f_fun(m, dt)
    s_est = State(m.state_space, 0., x, P)
    result1 = ukf_predict(s_est, m.q, f, alpha=1e-3)
    result2 = ukf_predict(s_est, m.q, f, alpha=1.0)
    assert not np.allclose(result1.covar.cov, result2.covar.cov, atol=1e-6)


def test_ukf_predict_near_singular_covariance():
    """UKF should not crash when P is near-singular (Cholesky fallback path)."""
    m  = ConstantVelocityMotionModel(num_dims=2, process_covar=1.0)
    dt = 1.0
    m.update_time_step(dt)
    x  = np.array([0., 0., 1., 0.])
    # Near-singular P: one eigenvalue essentially zero
    P  = CovarianceMatrix(np.diag([1e4, 1e4, 1., 1e-12]))
    f  = _cv_f_fun(m.f)
    result = ukf_predict(State(m.state_space, 0., x, P), m.q, f)
    assert result.state.shape == (4,)
    assert np.all(np.isfinite(result.covar.cov))
