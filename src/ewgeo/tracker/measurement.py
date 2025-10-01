import numpy as np
import numpy.typing as npt

from ewgeo.utils import safe_2d_shape
from ewgeo.utils.system import PassiveSurveillanceSystem
from ewgeo.utils.covariance import CovarianceMatrix
from .kinematic import StateSpace


class Measurement:
    time: npt.floating
    sensor: PassiveSurveillanceSystem
    zeta: npt.ArrayLike
    position_estimate: npt.ArrayLike
    velocity_estimate: npt.ArrayLike or None
    error_covariance_estimate: CovarianceMatrix  # represents either the pos_estimate (if vel is none) or combined pos/vel estimates

    def __init__(self, time: npt.floating, sensor: PassiveSurveillanceSystem, zeta: npt.ArrayLike):
        pass

    def make_measurement_model(self, state_space: StateSpace):

        # Define functions to sample the position/velocity components of the target state
        def pos_component(x):
            return x[state_space.pos_slice]

        def vel_component(x):
            return x[state_space.vel_slice] if state_space.has_vel else None

        # Non-Linear Measurement Function
        def z_fun(x):
            return self.sensor.measurement(x_source=pos_component(x), v_source=vel_component(x))

        # Measurement Function Jacobian
        def h_fun(x):
            j = self.sensor.jacobian(x_source=pos_component(x), v_source=vel_component(x))
            # Jacobian may be either w.r.t. position-only (pss.num_dim rows) or pos/vel,
            # depending on which type of pss we're calling.

            # Build the H matrix
            _, num_source_pos = safe_2d_shape(x)
            h = np.zeros((self.sensor.num_measurements, state_space.num_states, num_source_pos))
            h[:, state_space.pos_slice, :] = np.transpose(j[:self.sensor.num_dim, :])[:, :, np.newaxis]
            if state_space.has_vel and j.shape[0] > self.sensor.num_dim:
                # The state space has velocity components, and the pss returned rows for
                # the jacobian w.r.t. velocity.
                h[: state_space.vel_slice, :] = np.transpose(j[self.sensor.num_dim:, :])[:, :, np.newaxis]

            if num_source_pos == 1:
                # Collapse it to 2D, there's no need for the third dimension
                h = np.reshape(h, (self.sensor.num_measurements, state_space.num_states))
            return h

        return z_fun, h_fun