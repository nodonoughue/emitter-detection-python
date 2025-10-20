import numpy as np
import numpy.typing as npt

from ..utils.covariance import CovarianceMatrix
from ..utils.system import PassiveSurveillanceSystem
from .states import StateSpace, State


class Measurement:
    time: float
    sensor: PassiveSurveillanceSystem | None
    zeta: npt.ArrayLike

    def __init__(self, time: float, sensor: PassiveSurveillanceSystem | None, zeta: npt.ArrayLike):
        self.time = time
        self.sensor = sensor
        self.zeta = zeta

    @property
    def size(self):
        return len(self.zeta)

    def __str__(self):
        return f"Measurement at t={self.time}: {self.zeta}"

class MeasurementModel:
    """
    Parent class to capture the measurement model for a KF/EKF-system.

    Used to update a track given new measurements
    """
    state_space: StateSpace
    pss: PassiveSurveillanceSystem

    def __init__(self, state_space: StateSpace, pss: PassiveSurveillanceSystem):
        self.state_space = state_space
        self.pss = pss

    @property
    def num_measurement_dimensions(self):
        return self.pss.num_measurements

    @property
    def num_state_dimensions(self):
        return self.state_space.num_states

    def false_alarm(self, max_val: npt.ArrayLike, num: int, time: float = None):
        return [Measurement(time=time,
                            sensor=self.pss,
                            zeta=np.random.uniform(low=-max_val,
                                                   high=max_val,
                                                   size=(self.num_measurement_dimensions, ))) for _ in range(num)]

    def measurement(self, state: State, noise: npt.ArrayLike | bool=False) -> Measurement:
        """
        Return the measurement associated with the provided state.

        :param state: State of the target at which to compute the measurement
        :param noise: if it is a bool, then random noise will be generated if it is True and nothing if it is False;
                      if it is a numpy array, then it will be added directly to the result.
        """
        args = {'x_source': self.state_space.pos_component(state.state),
                'v_source': self.state_space.vel_component(state.state)}
        if noise == True:
            z = self.pss.noisy_measurement(**args)
        else:
            z = self.pss.measurement(**args)
            if isinstance(noise, np.ndarray):
                z = z + noise

        return Measurement(zeta=z, sensor=self.pss, time=state.time)

    def jacobian(self, state: State) -> npt.NDArray:
        j = self.pss.jacobian(x_source=self.state_space.pos_component(state.state),
                              v_source=self.state_space.vel_component(state.state))

        # Jacobian may be either w.r.t. position-only (pss.num_dim rows) or pos/vel,
        # depending on which type of pss we're calling.

        # Build the H matrix
        h = np.zeros((self.num_measurement_dimensions, self.num_state_dimensions))
        h[:, self.state_space.pos_slice] = np.transpose(j[:self.pss.num_dim, :])
        if self.state_space.has_vel and j.shape[0] > self.pss.num_dim:
            # The state space has velocity components, and the pss returned rows for
            # the jacobian w.r.t. velocity.
            h[: self.state_space.vel_slice] = np.transpose(j[self.pss.num_dim:, :])

        return h

    def log_likelihood(self, state1: State, state2: State) -> float:
        """
        Return the log-likelihood of the measurement at state1 given the state2 as the underlying truth.
        """

        # Determine the measurement that comes from state1
        m = self.measurement(state1)

        # Compute the log likelihood of the measurement given the state is state2
        return self.log_likelihood_from_measurement(state=state2, measurement=m)

    def log_likelihood_from_measurement(self, state: State, measurement: Measurement) -> float:
        return self.pss.log_likelihood(x_source=self.state_space.pos_component(state.state),
                                       v_source=self.state_space.vel_component(state.state),
                                       zeta=measurement.zeta)

    def state_from_measurement(self, m: Measurement, truth_state: bool=False)-> State:
        """
        Attempt to populate a state from a measurement.

        We will first generate an empty state, based on the state space, then attempt to populate the
        position and velocity using the least square solver from the underlying PassiveSurveillanceSystem object stored
        in self.pss.
        """

        # Use the PSS object's least square estimator to come up with an estimated position
        init_pos_vel = np.zeros((2*self.state_space.num_dims, ))
        pos_vel_est, _ = self.pss.least_square(zeta=m.zeta, x_init=init_pos_vel)

        # Convert to a state vector
        init_state_vec = np.zeros((self.state_space.num_states, ))
        init_state_vec[self.state_space.pos_vel_slice] = pos_vel_est

        # Initialize the state without a covariance matrix
        s = State(state_space=self.state_space, time=m.time, state=init_state_vec)

        if not truth_state:
            # Compute the CRLB and put it on top of the covariance matrix object
            crlb = self.pss.compute_crlb(x_source=s.pos_vel)
            init_covar = 1e6*np.eye(self.state_space.num_states)
            init_covar[:crlb.shape[0], :crlb.shape[1]] = crlb

            s.covar = CovarianceMatrix(init_covar)

        return s

# =============== Elementary Kalman Filter and Extended Kalman Filter Update Functions ================
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