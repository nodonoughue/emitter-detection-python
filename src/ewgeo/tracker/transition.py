from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt

from ewgeo.utils.covariance import CovarianceMatrix
from . import State, StateSpace, Track


class MotionModel(ABC):
    """
    Abstract class for defining motion models.
    """
    state_space: StateSpace

    # Process Equations
    time_delta: float | None
    f: npt.ArrayLike                # Transition Matrix
    process_covar: npt.ArrayLike
    q: npt.ArrayLike                # Process Noise

    def __init__(self):
        pass

    def copy(self):
        """
        Make a new MotionModel object with the same parameters and type as this one
        """
        cls = self.__class__
        result = cls.__new__(cls)

        # The only thing worth copying is the state_space
        # Other parameters are generated as needed
        result.state_space = self.state_space.copy()
        return result

    @abstractmethod
    def make_transition_matrix(self, time_delta: float):
        pass

    @abstractmethod
    def make_process_covariance_matrix(self, process_covar: npt.ArrayLike, time_delta: float):
        pass

    @property
    def num_dims(self):
        return self.state_space.num_dims

    @property
    def num_states(self):
        return self.state_space.num_states

    def predict(self, s: State | Track, new_time: float):
        # If a track is provided as the current state, parse its current state
        if isinstance(s, Track):
            s = s.curr_state

        # Look up or compute the process noise and transition matrices
        time_delta = new_time - s.time

        if time_delta == 0:
            # We already have a state at this time; nothing to predict forward
            return s.copy()

        if time_delta is not None and time_delta != self.time_delta:
            # Generate new ones
            self.time_delta = time_delta
            self.f = self.make_transition_matrix(time_delta)
            self.q = self.make_process_covariance_matrix(self.process_covar, time_delta)

        # To predict forward, just pre-multiply the previous state with the transition matrix
        new_state = self.f @ s.state
        if s.covar is None:
            new_covar = None
        else:
            new_covar = CovarianceMatrix(self.f @ s.covar.cov @ np.transpose(self.f) + self.q)

        # Make a new State object
        return s.copy(state=new_state, covar=new_covar, time=new_time)

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
            self.time_delta = time_delta
            self.f = self.make_transition_matrix(self.time_delta)
            if self.process_covar is None:
                self.q = None
            else:
                self.q = self.make_process_covariance_matrix(self.process_covar, self.time_delta)
        return

    def validate_process_covar_input(self, process_covar: npt.ArrayLike)-> npt.NDArray:
        if process_covar is None:
            process_covar = self.process_covar

        if np.shape(process_covar) != (self.num_dims, self.num_dims):
            if np.size(process_covar) == 1:
                # It's a scalar
                process_covar = np.eye(self.num_dims) * process_covar
            elif np.size(process_covar) == self.num_dims:
                # It's a vector, convert via diag
                process_covar = np.diag(process_covar)
            else:
                # Shape is not recognized
                raise SyntaxError(f'process_covar must be a square matrix of shape ({self.num_dims}, {self.num_dims})')

        return process_covar

    @classmethod
    def make_motion_model(cls, model_type: str, num_dims: int, process_covar: npt.ArrayLike):
        """
        Constructor method for MotionModel subclasses
        """
        valid_models = {'constant_velocity': ConstantVelocityMotionModel,
                        'cv': ConstantVelocityMotionModel,
                        'constant_acceleration': ConstantAccelerationMotionModel,
                        'ca': ConstantAccelerationMotionModel,
                        'constant_jerk': ConstantJerkMotionModel,
                        'cj': ConstantJerkMotionModel
                        }
        if model_type.lower() not in valid_models:
            raise ValueError(f'Invalid model type: {model_type}. Valid options are: {valid_models.keys()}')
        else:
            return valid_models[model_type.lower()](num_dims=num_dims, process_covar=process_covar)

class ConstantVelocityMotionModel(MotionModel):
    """
    Position and Velocity are tracked states. There is no tracking of acceleration.
    Velocity is assumed to have non-zero-mean Gaussian distribution.
    """
    def __init__(self, num_dims: int, process_covar: npt.ArrayLike=None, time_delta: float=None):
        super().__init__()

        state_space = StateSpace()
        state_space.num_states = 2*num_dims
        state_space.num_dims = num_dims
        state_space.pos_slice = np.s_[:num_dims]
        state_space.vel_slice = np.s_[num_dims:2*num_dims]
        state_space.pos_vel_slice = np.s_[:2*num_dims]
        state_space.accel_slice = None
        state_space.has_pos = True
        state_space.has_vel = True
        state_space.has_accel = False
        self.state_space = state_space

        self.time_delta = time_delta
        self.process_covar = process_covar
        self.update_time_step(time_delta)

    def make_transition_matrix(self, time_delta: float=None):
        """
        Implement the transition matrix F for a constant velocity motion model
        """
        if time_delta is None:
            time_delta = self.time_delta

        return np.block([[np.eye(self.num_dims), time_delta*np.eye(self.num_dims)],
                         [np.zeros((self.num_dims, self.num_dims)), np.eye(self.num_dims)]])

    def make_process_covariance_matrix(self, process_covar: npt.ArrayLike=None, time_delta: float=None):
        """
        Implement the process noise covariance Q for a constant velocity motion model
        """
        process_covar = self.validate_process_covar_input(process_covar)
        if time_delta is None:
            time_delta = self.time_delta

        return np.block([[.25*time_delta**4*process_covar, .5*time_delta**3*process_covar],
                         [.5*time_delta**3*process_covar, time_delta**2*process_covar]])


class ConstantAccelerationMotionModel(MotionModel):
    """
    Position, Velocity, and Acceleration are tracked states.
    Acceleration is assumed to have non-zero-mean Gaussian distribution.
    """
    def __init__(self, num_dims: int, process_covar: npt.ArrayLike = None, time_delta: float = None):
        super().__init__()

        # State model is:
        # [px, py, pz, vx, vy, vz, ax, ay, az]'
        state_space = StateSpace()
        state_space.num_states = 3 * num_dims
        state_space.num_dims = num_dims
        state_space.pos_slice = np.s_[:num_dims]
        state_space.vel_slice = np.s_[num_dims:2 * num_dims]
        state_space.pos_vel_slice = np.s_[:2 * num_dims]
        state_space.accel_slice = np.s_[2 * num_dims:3 * num_dims]
        state_space.has_pos = True
        state_space.has_vel = True
        state_space.has_accel = True
        self.state_space = state_space

        self.time_delta = time_delta
        self.process_covar = process_covar
        self.update_time_step(time_delta)  # assign to local variable and update f/q matrices

    def make_transition_matrix(self, time_delta: float=None):
        """
        Implement the transition matrix F for a constant acceleration motion model
        """
        if time_delta is None:
            time_delta = self.time_delta

        return np.block([[np.eye(self.num_dims), time_delta * np.eye(self.num_dims), .5 * time_delta ** 2 * np.eye(self.num_dims)],
                         [np.zeros((self.num_dims, self.num_dims)), np.eye(self.num_dims), time_delta * np.eye(self.num_dims)],
                         [np.zeros((self.num_dims, 2 * self.num_dims)), np.eye(self.num_dims)]])

    def make_process_covariance_matrix(self, process_covar: npt.ArrayLike=None, time_delta: float=None):
        """
        Implement the process noise covariance Q for a constant acceleration motion model
        """
        process_covar = self.validate_process_covar_input(process_covar)
        if time_delta is None:
            time_delta = self.time_delta

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
    def __init__(self, num_dims: int, process_covar: npt.ArrayLike = None, time_delta: float = None):
        super().__init__()

        # State model is:
        # [px, py, pz, vx, vy, vz, ax, ay, az, jx, jy, jz]'
        state_space = StateSpace()
        state_space.num_states = 4 * num_dims
        state_space.num_dims = num_dims
        state_space.pos_slice = np.s_[:num_dims]
        state_space.vel_slice = np.s_[num_dims:2 * num_dims]
        state_space.pos_vel_slice = np.s_[:2 * num_dims]
        state_space.accel_slice = np.s_[2 * num_dims:3 * num_dims]
        state_space.has_pos = True
        state_space.has_vel = True
        state_space.has_accel = True
        self.state_space = state_space

        self.time_delta = time_delta
        self.process_covar = process_covar
        self.update_time_step(time_delta)  # assign to local variable and update f/q matrices

    def make_transition_matrix(self, time_delta: float=None):
        """
        Implement the transition matrix F for a constant acceleration motion model
        """
        if time_delta is None:
            time_delta = self.time_delta

        return np.block([[np.eye(self.num_dims),
                          time_delta * np.eye(self.num_dims),
                          .5 * time_delta ** 2 * np.eye(self.num_dims),
                          1 / 6 * time_delta ** 3 * np.eye(self.num_dims)],
                         [np.zeros((self.num_dims, self.num_dims)),
                          np.eye(self.num_dims),
                          time_delta * np.eye(self.num_dims),
                          .5 * time_delta ** 2 * np.eye(self.num_dims)],
                         [np.zeros((self.num_dims, 2 * self.num_dims)), np.eye(self.num_dims), time_delta * np.eye(self.num_dims)],
                         [np.zeros((self.num_dims, 3 * self.num_dims)), np.eye(self.num_dims)]])

    def make_process_covariance_matrix(self, process_covar: npt.ArrayLike=None, time_delta: float=None):
        """
        Implement the process noise covariance Q for a constant acceleration motion model
        """
        process_covar = self.validate_process_covar_input(process_covar)
        if time_delta is None:
            time_delta = self.time_delta

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
