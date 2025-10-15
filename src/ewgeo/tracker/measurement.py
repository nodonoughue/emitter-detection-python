import numpy as np
import numpy.typing as npt

from ..utils.system import PassiveSurveillanceSystem
from .states import StateSpace, State


class Measurement:
    time: float
    sensor: PassiveSurveillanceSystem or None
    zeta: npt.ArrayLike

    def __init__(self, time: float, sensor: PassiveSurveillanceSystem or None, zeta: npt.ArrayLike):
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

