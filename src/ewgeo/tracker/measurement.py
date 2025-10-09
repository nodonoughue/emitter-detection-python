from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt

from ..utils import safe_2d_shape
from ..utils.system import PassiveSurveillanceSystem
from ..utils.covariance import CovarianceMatrix
from .states import StateSpace, State


class Measurement:
    time: float
    sensor: PassiveSurveillanceSystem
    zeta: npt.ArrayLike

    def __init__(self, time: float, sensor: PassiveSurveillanceSystem, zeta: npt.ArrayLike):
        self.time = time
        self.sensor = sensor
        self.zeta = zeta

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
        :param noise: if a bool, then random noise will be generated if True; if a numpy array, then it will be added directly to the result.
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

    def jacobian(self, state: State) -> npt.ArrayLike:
        j = self.pss.jacobian(x_source=self.state_space.pos_component(state.state),
                              v_source=self.state_space.vel_component(state.state))

        # Jacobian may be either w.r.t. position-only (pss.num_dim rows) or pos/vel,
        # depending on which type of pss we're calling.

        # Build the H matrix
        h = np.zeros((self.num_measurement_dimensions, self.num_state_dimensions))
        h[:, self.state_space.pos_slice, :] = np.transpose(j[:self.pss.num_dim, :])[:, :, np.newaxis]
        if self.state_space.has_vel and j.shape[0] > self.pss.num_dim:
            # The state space has velocity components, and the pss returned rows for
            # the jacobian w.r.t. velocity.
            h[: self.state_space.vel_slice, :] = np.transpose(j[self.pss.num_dim:, :])[:, :, np.newaxis]

        return h

    def log_likelihood(self, state1: State, state2: State) -> float:
        """
        Return the log-likelihood of the measurement at state1 given the state2 as the underlying truth.
        """

        # Determine the measurement that comes from state2
        m = self.measurement(state1)

        # Compute the log likelihood of state1
        return self.log_likelihood_from_measurement(state=state1, measurement=m)

    def log_likelihood_from_measurement(self, state: State, measurement: Measurement) -> float:
        return self.pss.log_likelihood(x_source=self.state_space.pos_component(state.state),
                                       v_source=self.state_space.vel_component(state.state),
                                       zeta=measurement.zeta)

class Updater(ABC):
    """
    Abstract class for a kinematic model that can update a State to some new time.
    """
    measurement_model: MeasurementModel

    def __init__(self, measurement_model: MeasurementModel):
        self.measurement_model = measurement_model

    @abstractmethod
    def predict_measurement(self, predicted_state: State, measurement_model: MeasurementModel=None, measurement_noise=True):
        """
        Predict the measurement implied by the predicted_state, returns a Measurement object with the predicted
        mean and the covariance matrix of the measurement prediction.
        """
        # Compute the predicted measurement and next step's innovation covariance, to assist
        # with data association
        z_fun, h_fun = self.measurements[-1].make_measurement_model(self.motion_model.state_space)
        self.pred_measurement = z_fun(self.pred_state)
        h_mtx = h_fun(self.pred_state)
        msmt_cov = self.measurements[-1].sensor.cov.cov
        self.innov_covar = CovarianceMatrix(h_mtx @ self.pred_covar @ h_mtx.T + msmt_cov)




