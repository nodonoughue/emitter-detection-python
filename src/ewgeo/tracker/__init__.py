import numpy as np
import numpy.typing as npt

# Directly import several classes for easier reference
from .states import *
from .track import *
from .transition import MotionModel
from .measurement import MeasurementModel

# Import all submodules
from . import association
from . import transition
from . import measurement
from . import tracker

def kf_update(x_prev: npt.ArrayLike, p_prev: npt.ArrayLike,
              zeta: npt.ArrayLike, cov: npt.ArrayLike, h: npt.ArrayLike) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Conduct a Kalman Filter update, given the previous state estimate and covariance, a measurement, and
    the measurement matrix.

    :param x_prev: Previous state estimate, shape: (n_states, )
    :param p_prev: Previous state error covariance, shape: (n_states, n_states)
    :param zeta: Measurement vector, shape: (n_meas, )
    :param cov: Measurement error covariance, shape: (n_meas, n_meas)
    :param h: Measurement matrix, shape: (n_meas, n_states)
    :return x: Updated state estimate, shape: (n_states, )
    :return p: Updated state error covariance, shape: (n_states, n_states)
    """
    # Evaluate the Measurement and Jacobian at x_prev
    z = h @ x_prev

    # Compute the Innovation (or Residual)
    y = zeta - z

    # Compute the Innovation Covariance
    s = h @ p_prev @ h.T + cov

    # Compute the Kalman Gain
    k = p_prev@h.T/s

    # Update the Estimate
    x = x_prev + k @ y
    p = (np.eye(p_prev.shape[0]) - (k @ h)) @ p_prev

    return x, p


def kf_predict(x_est, p_est, q, f):
    """
    Conduct a Kalman Filter prediction, given the current estimated state and covariance, a transition matrix, and
    the process noise covariance.

    :param x_est: Current state estimate, shape: (n_states, )
    :param p_est: Current state error covariance, shape: (n_states, n_states)
    :param q: Process noise covariance, shape: (n_states, n_states)
    :param f: Transition matrix, shape: (n_states, n_states)
    :return x_pred: Predicted state estimate, shape: (n_states, )
    :return p_pred: Predicted state error covariance, shape: (n_states, n_states)
    """

    # Predict the next state
    x_pred = f @ x_est

    # Predict the next state error covariance
    p_pred = f @ p_est @ f.T + q

    return x_pred, p_pred

def ekf_update(x_prev, p_prev, zeta, cov, z_fun, h_fun):
    """
    Conduct an Extended Kalman Filter update, given the previous state estimate and covariance, a measurement function,
    and the measurement matrix function.

    :param x_prev: Previous state estimate, shape: (n_states, )
    :param p_prev: Previous state error covariance, shape: (n_states, n_states)
    :param zeta: Measurement vector, shape: (n_meas, )
    :param cov: Measurement error covariance, shape: (n_meas, n_meas)
    :param z_fun: Function handle for measurement evaluation, returns a vector of shape (n_meas, )
    :param h_fun: Function handle for measurement matrix evaluation, returns a matrix of shape (n_meas, n_states)
    :return x: Updated state estimate, shape: (n_states, )
    :return p: Updated state error covariance, shape: (n_states, n_states)
    """

    # Evaluate the Measurement and Jacobian at x_prev
    z = z_fun(x_prev)
    h = h_fun(x_prev)

    # Compute the Innovation (or Residual)
    y = zeta - z

    # Compute the Innovation Covariance
    s = h @ p_prev @ h.T + cov

    # Compute the Kalman Gain
    k = p_prev @ h.T @ np.linalg.inv(s)

    # Update the Estimate
    x = x_prev + k @ y
    p = (np.eye(p_prev.shape[0])- (k @ h)) @ p_prev

    return x, p

def ekf_predict(x_est, p_est, q, f_fun, g_fun):
    """
    Conduct an Extended Kalman Filter prediction, given the current estimated state and covariance, a function handle
    to generate the predicted state, and a function handle to generate the transition matrix.

    :param x_est: Current state estimate, shape: (n_states, )
    :param p_est: Current state error covariance, shape: (n_states, n_states)
    :param q: Process noise covariance, shape: (n_states, n_states)
    :param f_fun: Transition function handle, returns an array of shape: (n_states, )
    :param g_fun: Transition matrix function handle, returns a matrix of shape: (n_states, n_states)
    :return x_pred: Predicted state estimate, shape: (n_states, )
    :return p_pred: Predicted state error covariance, shape: (n_states, n_states)
    """

    # Forward prediction of state
    x_pred = f_fun(x_est)

    # Forward prediction of state error covariance
    f = g_fun(x_est)
    p_pred = f @ p_est @ np.transpose(f) + q

    return x_pred, p_pred
