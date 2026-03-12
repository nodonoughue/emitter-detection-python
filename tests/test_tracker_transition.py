import numpy as np
import pytest

from ewgeo.tracker.transition import (
    ConstantVelocityMotionModel,
    ConstantAccelerationMotionModel,
    ConstantJerkMotionModel,
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
