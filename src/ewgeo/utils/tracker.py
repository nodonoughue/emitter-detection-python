import numpy as np

from . import safe_2d_shape
from .system import PassiveSurveillanceSystem


def kf_update(x_prev, p_prev, zeta, cov, h):
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

    # Predict the next state
    x_pred = f @ x_est

    # Predict the next state error covariance
    p_pred = f @ p_est @ f.T + q

    return x_pred, p_pred

def ekf_update(x_prev, p_prev, zeta, cov, z_fun, h_fun):
    """

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
    Conduct the Predict stage for an Extended Kalman Filter

    :param x_est: Current state estimate
    :param p_est:
    :param q:
    :param f_fun: Function handle for measurements
    :param g_fun: Function handle for generation of the system matrix G
    """

    # Forward prediction of state
    x_pred = f_fun(x_est)

    # Forward prediction of state error covariance
    f = g_fun(x_est)
    p_pred = f @ p_est @ np.transpose(f) + q

    return x_pred, p_pred

def make_kinematic_model(model_type: str, num_dims: int=3, process_covar: np.ndarray or None=None):
    """

    :param model_type:
    :param num_dims:
    :param process_covar:
    :return f:
    :return q:
    :return state_space:
    """

    # Parse Covariance Input
    if process_covar is None:
        process_covar = np.eye(num_dims)
    elif np.isscalar(process_covar) or np.size(process_covar) == 1:
        process_covar = process_covar*np.eye(num_dims)

    # Initialize Output
    state_space = {'num_dims': num_dims,
                   'num_states': None,
                   'has_pos': True,
                   'has_vel': None,
                   'pos_slice': None,
                   'vel_slice': None}

    # Define Kinematic Model and Process Noise
    model_type = model_type.lower() # ensure it's lowercase for ease of comparison
    if model_type == 'cv' or model_type == 'constant velocity':
        # Position and Velocity are tracked states
        # Acceleration is assumed zero-mean Gaussian
        state_space['num_states'] = 2*num_dims
        state_space['pos_slice'] = np.s_[:num_dims]
        state_space['vel_slice'] = np.s_[num_dims:2*num_dims]
        state_space['has_vel'] = True

        def f(t):
            return np.block([[np.eye(num_dims), t*np.eye(num_dims)],
                             [np.zeros((num_dims, num_dims)), np.eye(num_dims)]])

        def q(t):
            return np.block([[.25*t**4*process_covar, .5*t**3*process_covar],
                             [.5*t**3*process_covar, t**2*process_covar]])

    elif model_type == 'ca' or model_type == 'constant acceleration':
        # Position, Velocity, and Acceleration are tracked states
        # Acceleration is assumed to have non-zero-mean Gaussian
        # distribution
        #
        # State model is:
        # [px, py, pz, vx, vy, vz, ax, ay, az]'
        state_space['num_states'] = 3*num_dims
        state_space['pos_slice'] = np.s_[:num_dims]
        state_space['vel_slice'] = np.s_[num_dims:2*num_dims]
        state_space['has_vel'] = True

        def f(t):
            return np.block([[np.eye(num_dims), t*np.eye(num_dims), .5*t**2*np.eye(num_dims)],
                             [np.zeros((num_dims, num_dims)), np.eye(num_dims), t*np.eye(num_dims)],
                             [np.zeros((num_dims, 2*num_dims)), np.eye(num_dims)]])

        # Process noise covariance
        # This assumes that accel_var is the same in all dimensions
        def q(t):
            return np.block([[.25*t**4*process_covar, .5*t**3*process_covar, .5*t**2*process_covar],
                             [.5*t**3*process_covar,    t**2*process_covar,      t*process_covar],
                             [.5*t**2*process_covar,      t*process_covar,        process_covar]])

    elif model_type == 'cj' or model_type == 'constant jerk':
        # Position, Velocity, and Acceleration are tracked states
        # Acceleration is assumed to have non-zero-mean Gaussian
        # distribution
        #
        # State model is:
        # [px, py, pz, vx, vy, vz, ax, ay, az, jx, jy, jz]'
        state_space['num_states'] = 4*num_dims
        state_space['pos_slice'] = np.s_[:num_dims]
        state_space['vel_slice'] = np.s_[num_dims:2*num_dims]
        state_space['has_vel'] = True

        def f(t):
            return np.block([[np.eye(num_dims), t*np.eye(num_dims), .5*t**2*np.eye(num_dims), 1/6*t**3*np.eye(num_dims)],
                             [np.zeros((num_dims, num_dims)), np.eye(num_dims), t*np.eye(num_dims), .5*t**2*np.eye(num_dims)],
                             [np.zeros((num_dims, 2*num_dims)), np.eye(num_dims), t*np.eye(num_dims)],
                             [np.zeros((num_dims, 3*num_dims)), np.eye(num_dims)]])

        # Process noise covariance
        # This assumes that accel_var is the same in all dimensions
        def q(t):
            return np.block([[t**7/252*process_covar, t**6/72*process_covar, t**5/30*process_covar, t**4/24*process_covar],
                             [t**6/72*process_covar,  t**5/20*process_covar, t**4/8*process_covar,  t**3/6*process_covar],
                             [t**5/30*process_covar,  t**4/8*process_covar,  t**3/3*process_covar,  t**2/2*process_covar],
                             [t**4/24*process_covar,  t**3/6*process_covar,  t**2/2*process_covar,      t*process_covar]])

    # Implementation of the aerodynamic and ballistic models is left to
    # readers as an exercise
    elif model_type == 'brv' or model_type == 'ballistic reentry':
        raise NotImplementedError('%s kinematic model not yet implemented.', model_type)
    elif model_type == 'marv' or model_type == 'maneuvering reentry':
        raise NotImplementedError('%s kinematic model not yet implemented.', model_type)
    elif model_type == 'aero':
        raise NotImplementedError('%s kinematic model not yet implemented.', model_type)
    elif model_type == 'ballistic':
        raise NotImplementedError('%s kinematic model not yet implemented.', model_type)
    else:
        raise NotImplementedError('%s kinematic model option not recognized.', model_type)

    return f, q, state_space

def make_measurement_model(pss:PassiveSurveillanceSystem, state_space:dict):

    # Define functions to sample the position/velocity components of the target state
    def pos_component(x):
        return x[state_space['pos_slice']]
    def vel_component(x):
        return x[state_space['vel_slice']] if state_space['has_vel'] else None

    # Non-Linear Measurement Function
    def z_fun(x):
        return pss.measurement(x_source=pos_component(x), v_source=vel_component(x))

    # Measurement Function Jacobian
    def h_fun(x):
        j = pss.jacobian(x_source=pos_component(x), v_source=vel_component(x))
        # Jacobian may be either w.r.t. position-only (pss.num_dim rows) or pos/vel,
        # depending on which type of pss we're calling.

        # Build the H matrix
        _, num_source_pos = safe_2d_shape(x)
        h = np.zeros((pss.num_measurements, state_space['num_states'], num_source_pos))
        h[:, state_space['pos_slice'], :] = np.transpose(j[:pss.num_dim, :])[:, :, np.newaxis]
        if state_space['has_vel'] and j.shape[0] > pss.num_dim:
            # The state space has velocity components, and the pss returned rows for
            # the jacobian w.r.t. velocity.
            h[: state_space['vel_slice'], :] = np.transpose(j[pss.num_dim:, :])[:, :, np.newaxis]

        if num_source_pos == 1:
            # Collapse it to 2D, there's no need for the third dimension
            h = np.reshape(h, (pss.num_measurements, state_space['num_states']))
        return h

    return z_fun, h_fun