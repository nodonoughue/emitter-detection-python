from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt

from ewgeo.utils.covariance import CovarianceMatrix

class StateSpace:
    num_dims: int
    num_states: int
    has_pos: bool
    has_vel: bool
    has_accel: bool
    pos_slice: slice or None
    vel_slice: slice or None
    pos_vel_slice: slice or None
    accel_slice: slice or None

    def __init__(self):
        pass

class MotionModel(ABC):
    """
    Abstract class for defining motion models.
    """
    state_space: StateSpace

    # Process Equations
    time_delta: npt.floating
    f: npt.ArrayLike                # Transition Matrix
    process_covar: npt.ArrayLike
    q: npt.ArrayLike                # Process Noise

    def __init__(self):
        pass

    @abstractmethod
    def make_transition_matrix(self, time_delta: npt.floating):
        pass

    @abstractmethod
    def make_process_covariance_matrix(self, process_covar: npt.ArrayLike, time_delta: npt.floating):
        pass

    def predict(self, previous_state: npt.ArrayLike, previous_covar: CovarianceMatrix, time_delta=None):

        # Look up or compute the process noise and transition matrices
        if time_delta is None or time_delta == self.time_delta:
            f = self.f
            q = self.q
        else:
            f = self.make_transition_matrix(time_delta)
            q = self.make_process_covariance_matrix(self.process_covar, time_delta)

        # To predict forward, just pre-multiply the previous state with the transition matrix
        new_state = f @ previous_state
        new_covar = CovarianceMatrix(f @ previous_covar.cov @ np.transpose(f) + q)
        return new_state, new_covar

    def update_time_step(self, time_delta):
        if time_delta is None:
            # Clear everything
            self.time_delta = None
            self.f = None
            self.q = None
        elif time_delta == self.time_delta and self.f is not None and self.q is not None:
            pass  # nothing to do; no change
        else:
            # Update the transition and process noise mapping matrices
            self.f = self.make_transition_matrix(self.time_delta)
            if self.process_covar is None:
                self.q = None
            else:
                self.q = self.make_process_covariance_matrix(self.process_covar, self.time_delta)
        return

class ConstantVelocityMotionModel(MotionModel):
    """
    Position and Velocity are tracked states. There is no tracking of acceleration.
    Velocity is assumed to have non-zero-mean Gaussian distribution.
    """
    def __init__(self, num_dims: int, process_covar: npt.ArrayLike=None, time_delta: npt.floating=None):
        super().__init__()

        state_space = StateSpace()
        state_space.num_states = 2*num_dims
        state_space.pos_slice = np.s_[:num_dims]
        state_space.vel_slice = np.s_[num_dims:2*num_dims]
        state_space.pos_vel_slice = np.s_[:2*num_dims]
        state_space.accel_slice = None
        state_space.has_vel = True
        state_space.has_accel = False
        self.state_space = state_space

        self.time_delta = time_delta
        self.process_covar = process_covar
        self.update_time_step(time_delta)  # assign to local variable and update f/q matrices

    def make_transition_matrix(self, time_delta: npt.floating):
        """
        Implement the transition matrix F for a constant velocity motion model
        """
        num_dims = self.state_space.num_dims
        return np.block([[np.eye(num_dims), time_delta*np.eye(num_dims)],
                         [np.zeros((num_dims, num_dims)), np.eye(num_dims)]])

    def make_process_covariance_matrix(self, process_covar: npt.ArrayLike, time_delta: npt.floating):
        """
        Implement the process noise covariance Q for a constant velocity motion model
        """
        if np.shape(process_covar) != (self.state_space.num_dims, self.state_space.num_dims):
            raise SyntaxError('process_covar must be a square matrix of shape (num_dims, num_dims)')

        return np.block([[.25*time_delta**4*process_covar, .5*time_delta**3*process_covar],
                         [.5*time_delta**3*process_covar, time_delta**2*process_covar]])




class ConstantAccelerationMotionModel(MotionModel):
    """
    Position, Velocity, and Acceleration are tracked states.
    Acceleration is assumed to have non-zero-mean Gaussian distribution.
    """
    def __init__(self, num_dims: int, process_covar: npt.ArrayLike = None, time_delta: npt.floating = None):
        super().__init__()

        # State model is:
        # [px, py, pz, vx, vy, vz, ax, ay, az]'
        state_space = StateSpace()
        state_space.num_states = 3 * num_dims
        state_space.pos_slice = np.s_[:num_dims]
        state_space.vel_slice = np.s_[num_dims:2 * num_dims]
        state_space.pos_vel_slice = np.s_[:2 * num_dims]
        state_space.accel_slice = np.s_[2 * num_dims:3 * num_dims]
        state_space.has_vel = True
        state_space.has_accel = True
        self.state_space = state_space

        self.time_delta = time_delta
        self.process_covar = process_covar
        self.update_time_step(time_delta)  # assign to local variable and update f/q matrices

    def make_transition_matrix(self, time_delta: npt.floating):
        """
        Implement the transition matrix F for a constant acceleration motion model
        """
        num_dims = self.state_space.num_dims
        return np.block([[np.eye(num_dims), time_delta * np.eye(num_dims), .5 * time_delta ** 2 * np.eye(num_dims)],
                         [np.zeros((num_dims, num_dims)), np.eye(num_dims), time_delta * np.eye(num_dims)],
                         [np.zeros((num_dims, 2 * num_dims)), np.eye(num_dims)]])

    def make_process_covariance_matrix(self, process_covar: npt.ArrayLike, time_delta: npt.floating):
        """
        Implement the process noise covariance Q for a constant acceleration motion model
        """
        if np.shape(process_covar) != (self.state_space.num_dims, self.state_space.num_dims):
            raise SyntaxError('process_covar must be a square matrix of shape (num_dims, num_dims)')

        return np.block([[.25*time_delta**4*process_covar,
                          .5*time_delta**3*process_covar,
                          .5*time_delta**2*process_covar],
                         [.5*time_delta**3*process_covar,
                          time_delta**2*process_covar,
                          time_delta*process_covar],
                         [.5*time_delta**2*process_covar,
                          time_delta*process_covar,
                          process_covar]])

class ConstantJerkMotionModel(MotionModel):
    """

    """
    def __init__(self, num_dims: int, process_covar: npt.ArrayLike = None, time_delta: npt.floating = None):
        super().__init__()

        # State model is:
        # [px, py, pz, vx, vy, vz, ax, ay, az, jx, jy, jz]'
        state_space = StateSpace()
        state_space.num_states = 4 * num_dims
        state_space.pos_slice = np.s_[:num_dims]
        state_space.vel_slice = np.s_[num_dims:2 * num_dims]
        state_space.pos_vel_slice = np.s_[:2 * num_dims]
        state_space.accel_slice = np.s_[2 * num_dims:3 * num_dims]
        state_space.has_vel = True
        state_space.has_accel = True
        self.state_space = state_space

        self.time_delta = time_delta
        self.process_covar = process_covar
        self.update_time_step(time_delta)  # assign to local variable and update f/q matrices

    def make_transition_matrix(self, time_delta: npt.floating):
        """
        Implement the transition matrix F for a constant acceleration motion model
        """
        num_dims = self.state_space.num_dims
        return np.block([[np.eye(num_dims),
                          time_delta * np.eye(num_dims),
                          .5 * time_delta ** 2 * np.eye(num_dims),
                          1 / 6 * time_delta ** 3 * np.eye(num_dims)],
                         [np.zeros((num_dims, num_dims)),
                          np.eye(num_dims),
                          time_delta * np.eye(num_dims),
                          .5 * time_delta ** 2 * np.eye(num_dims)],
                         [np.zeros((num_dims, 2 * num_dims)), np.eye(num_dims), time_delta * np.eye(num_dims)],
                         [np.zeros((num_dims, 3 * num_dims)), np.eye(num_dims)]])

    def make_process_covariance_matrix(self, process_covar: npt.ArrayLike, time_delta: npt.floating):
        """
        Implement the process noise covariance Q for a constant acceleration motion model
        """
        if np.shape(process_covar) != (self.state_space.num_dims, self.state_space.num_dims):
            raise SyntaxError('process_covar must be a square matrix of shape (num_dims, num_dims)')

        return np.block([[time_delta ** 7 / 252 * process_covar,
                          time_delta ** 6 / 72 * process_covar,
                          time_delta ** 5 / 30 * process_covar,
                          time_delta ** 4 / 24 * process_covar],
                         [time_delta ** 6 / 72 * process_covar,
                          time_delta ** 5 / 20 * process_covar,
                          time_delta ** 4 / 8 * process_covar,
                          time_delta ** 3 / 6 * process_covar],
                         [time_delta ** 5 / 30 * process_covar,
                          time_delta ** 4 / 8 * process_covar,
                          time_delta ** 3 / 3 * process_covar,
                          time_delta ** 2 / 2 * process_covar],
                         [time_delta ** 4 / 24 * process_covar,
                          time_delta ** 3 / 6 * process_covar,
                          time_delta ** 2 / 2 * process_covar,
                          time_delta * process_covar]])

# class BallisticReentryMotionModel(MotionModel):
# class ManeuveringReentryMotionModel(MotionModel):
# class AeroMotionModel(MotionModel):
# class BallisticMotionModel(MotionModel):
