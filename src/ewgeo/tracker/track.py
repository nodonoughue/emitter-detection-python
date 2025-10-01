import numpy as np
import numpy.typing as npt

from ewgeo.utils.covariance import CovarianceMatrix
from .kinematic import MotionModel, StateSpace
from .measurement import Measurement


class Track:
    """
    Parent class to keep track of data for a track and implement basic track maintenance functions, such as updating
    with new measurements and predicting forward.


    """
    # Data Inputs
    measurements: list[Measurement]

    # Track History
    time_history: npt.ArrayLike  # a list or a numpy array
    state_history: list[npt.ArrayLike]
    covar_history: list[CovarianceMatrix]

    # Current Track State
    curr_time: npt.floating
    curr_state: npt.ArrayLike
    curr_covar: CovarianceMatrix

    # Predicted Track State
    pred_time: npt.floating
    pred_state: npt.ArrayLike
    pred_covar: CovarianceMatrix

    # Track Parameters
    motion_model: MotionModel
    max_num_misses: int = 3  # number of updates without an associated measurement before track should be deleted
    num_misses: int = 0      # current number of missed updates
    is_stale: bool = False   # flag setting whether this track is stale or not
    is_mature: bool = False  # flag recording whether the track is mature, or is still a tracklet
    tracklet_m_of_n: list[int] = [3, 5]  # MofN track criteria to promote from tracklet to full track

    def __init__(self, motion_model: MotionModel):
        self.motion_model = motion_model
        self.measurements = []
        self.time_history = []

    def coast(self):
        """
        There was no associated measurement, so the track coasts. This is like an update where the new measurement
        gave no new information.
        """
        self._update_bookkeeping(self.pred_time, None, self.pred_state, self.pred_covar)

    def initiate(self, measurements: list[Measurement] or Measurement):
        """

        Initiate a track from a single measurement or a list of measurements
        """

        # Wrap the measurement in a list, if it's not already there
        if not isinstance(measurements, list):
            measurements = [measurements]

        first_msmt = measurements.pop(0)

        # Grab the motion model's state space
        state_space = self.motion_model.state_space  # StateSpace

        # Initialize a blank state
        curr_state = np.zeros((state_space.num_states, ))
        curr_state[state_space.pos_slice] = first_msmt.position_estimate
        curr_state[state_space.vel_slice] = first_msmt.velocity_estimate if state_space.has_vel and first_msmt.velocity_estimate is not None else None

        # Initialize a blank state error covariance; initialized with a very large posterior error covariance
        curr_cov = 1e9*np.eye(state_space.num_states)
        if first_msmt.velocity_estimate is not None and state_space.has_vel:
            # The covariance matrix estimate should be the same size as pos_vel_slice
            cov_slice = state_space.pos_vel_slice
        else:
            cov_slice = state_space.pos_slice
        curr_cov[cov_slice, cov_slice] = first_msmt.error_covariance_estimate.cov

        # Update the parameters
        self.curr_time = first_msmt.time
        self.curr_state = curr_state
        self.curr_covar = CovarianceMatrix(curr_cov)
        self.measurements.append(first_msmt)
        self.time_history.append(self.curr_time)
        self.state_history.append(self.curr_state)
        self.covar_history.append(self.curr_covar)

        # Iterate over any remaining initial measurements
        for msmt in measurements:
            self.predict(msmt.time)
            self.update(msmt)

    def predict(self, new_time: npt.floating):
        """
        Conduct the Predict stage for a Kalman or Extended Kalman Filter using the motion model(s) stored in this
        track.

        Updates pred_state and pred_covar, using the values in curr_state and curr_covar.
        """

        # Predict the state forward and record its time
        time_delta = new_time - self.curr_time
        self.pred_state, self.pred_covar = self.motion_model.predict(self.curr_state, self.curr_covar, time_delta)
        self.pred_time = new_time

        return

    def update(self, measurement: Measurement):
        """
        Update the current tracker. Use the list of predicted states and covariances to update the current state.

        If multiple motion models are used, then each prediction is tested, and the one that results in the smallest
        state covariance matrix (as measured by the trace of curr_covar) will be kept.
        """

        # Clear the current state
        self.curr_state = None
        self.curr_covar = CovarianceMatrix(np.inf)

        # Evaluate the Measurement and Jacobian at x_prev
        z_fun, h_fun = measurement.make_measurement_model(self.motion_model.state_space)
        z = z_fun(self.pred_state)
        h = h_fun(self.pred_state)

        # Compute the Innovation (or Residual)
        y = measurement.zeta - z

        # Compute the Innovation Covariance
        s = h @ self.pred_covar.cov @ h.T + measurement.error_covariance_estimate.cov

        # Compute the Kalman Gain
        k = self.pred_covar.cov @ h.T @ np.linalg.inv(s)

        # Update the Estimate
        new_state = self.pred_state + k @ y
        new_covar = (np.eye(self.motion_model.state_space.num_states) - (k @ h)) @ self.pred_covar.cov

        # Store the estimate and update the time
        self._update_bookkeeping(measurement.time, measurement, new_state, CovarianceMatrix(new_covar))

    def _update_bookkeeping(self, time: npt.floating, measurement: Measurement or None, state: npt.ArrayLike, covar: CovarianceMatrix):
        # Set the Current State
        self.curr_time = time
        self.curr_state = state
        self.curr_covar = covar

        # Append to state/time history
        self.time_history.append(time)
        self.state_history.append(state)
        self.covar_history.append(covar)

        # Add the measurement, if there was one, and update miss counter/stale parameters
        if measurement is not None:
            self.measurements.append(measurement)
            self.num_misses = 0  # reset the miss counter
            self.is_stale = False
        else:
            # Add to the miss counter
            self.num_misses = self.num_misses + 1
            if self.num_misses >= self.max_num_misses:
                self.is_stale = True

        # Check track quality and determine if this track should be considered mature
        if not self.is_mature:
            if len(self.measurements) >= self.tracklet_m_of_n[0] and len(self.time_history) <= self.tracklet_m_of_n[1]:
                # We've hit the required number of measurements in the M-of-N window
                self.is_mature = True
            elif len(self.time_history) >= self.tracklet_m_of_n[1]:
                # Enough time has passed, and we haven't yet gotten the required number of tracks, let's mark ourselves
                # stale
                self.is_stale = True
