from abc import ABC, abstractmethod
from collections.abc import Callable
import numpy as np
import numpy.typing as npt

from ewgeo.utils.constraints import snap_to_constraints
from ewgeo.utils.covariance import CovarianceMatrix
from . import State, StateSpace, CartesianStateSpace, Track, PolarKinematicStateSpace


class MotionModel(ABC):
    """
    Abstract class for defining motion models.
    """
    state_space: StateSpace

    # Process Equations
    time_delta: float | None
    f: npt.ArrayLike                # Transition Matrix
    process_covar: npt.ArrayLike
    q: CovarianceMatrix | None            # Process Noise

    # Optional inequality constraints applied to the position block after each predict step.
    # Each callable has the signature: (x: ndarray (num_dims, n)) -> (eps: ndarray (n,), x_valid: ndarray (num_dims, n))
    ineq_constraints: list | None = None

    def __init__(self):
        pass

    def copy(self):
        """
        Make a new MotionModel object with the same parameters and type as this one
        """
        cls = self.__class__
        result = cls.__new__(cls)

        # StateSpace is immutable; share the reference directly
        result.state_space = self.state_space
        return result

    @property
    def is_linear(self) -> bool:
        """True for linear models (KF); False for nonlinear models (EKF).
        Linear subclasses inherit this default; nonlinear subclasses must override to return False."""
        return True

    @abstractmethod
    def make_transition_matrix(self, time_delta: float):
        pass

    @abstractmethod
    def make_process_covariance_matrix(self, process_covar: npt.ArrayLike, time_delta: float)-> CovarianceMatrix:
        pass

    def transition_function(self, curr_state: State, time_delta: float) -> State:
        if time_delta != self.time_delta:
            # We need to update the transition matrix
            self.time_delta = time_delta
            self.f = self.make_transition_matrix(time_delta)
            self.q = self.make_process_covariance_matrix(self.process_covar, time_delta)

        # Default is a linear transition
        return State(state_space=curr_state.state_space, time=curr_state.time + time_delta,
                     state=self.f @ curr_state.state, covar=curr_state.covar)

    def transition_matrix(self, x: npt.ArrayLike, time_delta: float) -> npt.NDArray:
        """Default behavior, for linear models, is that the jacobian is just self.f, which
        is populated by make_transition_matrix"""
        if self.f is None or self.time_delta != time_delta:
            self.time_delta = time_delta
            self.f = self.make_transition_matrix(time_delta)
            self.q = self.make_process_covariance_matrix(self.process_covar, time_delta)

        return self.f

    @property
    def num_dims(self):
        return self.state_space.num_dims

    @property
    def num_states(self):
        return self.state_space.num_states

    def predict(self, s: State | Track, new_time: float):
        """
        Predict the state forward to new_time using this motion model.

        If s is a Track, its current state (s.curr_state) is used. The transition matrix F and
        process noise matrix Q are recomputed whenever time_delta changes.

        :param s: Current State (or Track whose curr_state will be used)
        :param new_time: Target time to predict to [seconds]
        :return: Predicted State at new_time
        """
        # If a track is provided as the current state, parse its current state
        if isinstance(s, Track):
            s = s.curr_state

        if not self.state_space.is_equal(s.state_space):
            raise ValueError(
                f"State space mismatch: motion model expects {self.state_space!r} "
                f"but received state with {s.state_space!r}."
            )

        # Look up or compute the process noise and transition matrices
        time_delta = new_time - s.time

        if time_delta == 0:
            # We already have a state at this time; nothing to predict forward
            return s

        # Predict the state forward, and grab the jacobian for use in updating the
        # state error covariance
        new_state = self.transition_function(s, time_delta)
        u = self.make_control_input(time_delta)
        if u is not None:
            new_state.state = new_state.state + u

        # Snap position to any inequality constraints (e.g. altitude bounds)
        if self.ineq_constraints is not None:
            n = self.state_space.num_dims
            pos_sl = self.state_space.pos_slice
            pos = new_state.state[pos_sl].reshape(n, 1)
            new_state.state[pos_sl] = snap_to_constraints(
                pos, ineq_constraints=self.ineq_constraints).ravel()

        f_for_cov = self.transition_matrix(s, time_delta)

        if s.covar is None:
            new_state.covar = None
        else:
            new_state.covar = CovarianceMatrix(f_for_cov @ s.covar.cov @ np.transpose(f_for_cov) + self.q.cov)

        # Make a new State object
        return new_state

    def make_control_input(self, time_delta: float) -> npt.NDArray | None:
        """
        Control input vector u for the affine state update x_{k+1} = F·x_k + u(dt).

        Returns None for models with no deterministic forcing (default). Override in
        subclasses that have a known, state-independent input (e.g. gravity in the
        ballistic model).

        :param time_delta: Prediction interval [s]
        :return: Control vector of shape (num_states,), or None
        """
        return None

    def update_time_step(self, time_delta):
        """
        Update the cached transition matrix F and process noise matrix Q for a new time_delta.
        Clears F and Q when time_delta is None.

        :param time_delta: New time step [seconds], or None to clear cached matrices
        """
        if time_delta is None:
            # Clear everything
            self.time_delta = None
            self.f = None
            self.q = None
        elif time_delta == self.time_delta and getattr(self, 'q', None) is not None and (not self.is_linear or getattr(self, 'f', None) is not None):
            pass  # nothing to do; no change
        else:
            # Update the process noise (and transition matrix for linear models)
            self.time_delta = time_delta
            if self.is_linear:
                self.f = self.make_transition_matrix(self.time_delta)
            else:
                self.f = None  # Jacobian is state-dependent; computed fresh in predict()
            if self.process_covar is None:
                self.q = None
            else:
                self.q = self.make_process_covariance_matrix(self.process_covar, self.time_delta)
        return

    def validate_process_covar_input(self, process_covar: npt.ArrayLike)-> npt.NDArray:
        """
        Normalize process_covar to a (num_dims x num_dims) array. Accepts scalar, 1-D vector (used
        as diagonal), or full matrix. Falls back to self.process_covar when input is None.

        :param process_covar: Scalar, 1-D array of length num_dims, or (num_dims x num_dims) matrix; or None
        :return: (num_dims x num_dims) ndarray
        :raises SyntaxError: if the input shape is not recognized
        """
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
                        'cj': ConstantJerkMotionModel,
                        'constant_turn': ConstantTurnMotionModel,
                        'ct': ConstantTurnMotionModel,
                        'constant_turn_rate_acceleration': ConstantTurnRateAccelerationMotionModel,
                        'ctra': ConstantTurnRateAccelerationMotionModel,
                        'ballistic': BallisticMotionModel,
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

        self.state_space = CartesianStateSpace(num_dims=num_dims, has_vel=True, has_accel=False)

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

    def make_process_covariance_matrix(self, process_covar: npt.ArrayLike=None,
                                       time_delta: float=None)-> CovarianceMatrix:
        """
        Implement the process noise covariance Q for a constant velocity motion model.

        Process noise is an acceleration disturbance.
        :param process_covar: Variance term, in units of m^2/s^3
        """
        process_covar = self.validate_process_covar_input(process_covar)
        if time_delta is None:
            time_delta = self.time_delta

        q_mat = np.block([[.25*time_delta**4*process_covar, .5*time_delta**3*process_covar],
                          [.5*time_delta**3*process_covar, time_delta**2*process_covar]])
        return CovarianceMatrix(q_mat)


class ConstantAccelerationMotionModel(MotionModel):
    """
    Position, Velocity, and Acceleration are tracked states.
    Acceleration is assumed to have non-zero-mean Gaussian distribution.
    """
    def __init__(self, num_dims: int, process_covar: npt.ArrayLike = None, time_delta: float = None):
        super().__init__()

        # State model is: [px, py, pz, vx, vy, vz, ax, ay, az]
        self.state_space = CartesianStateSpace(num_dims=num_dims, has_vel=True, has_accel=True)

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

        q_mat = np.block([[.25*time_delta**4*process_covar,
                           .5*time_delta**3*process_covar,
                           .5*time_delta**2*process_covar],
                          [.5*time_delta**3*process_covar,
                           time_delta**2*process_covar,
                           time_delta*process_covar],
                          [.5*time_delta**2*process_covar,
                           time_delta*process_covar,
                           process_covar]])
        return CovarianceMatrix(q_mat)


class ConstantJerkMotionModel(MotionModel):
    """
    Position, Velocity, Acceleration, and Jerk are tracked states.
    Jerk is assumed to have a non-zero-mean Gaussian distribution.
    """
    def __init__(self, num_dims: int, process_covar: npt.ArrayLike = None, time_delta: float = None):
        super().__init__()

        # State model is: [px, py, pz, vx, vy, vz, ax, ay, az, jx, jy, jz]
        self.state_space = CartesianStateSpace(num_dims=num_dims, has_vel=True,
                                               has_accel=True, has_jerk=True)

        self.time_delta = time_delta
        self.process_covar = process_covar
        self.update_time_step(time_delta)  # assign to local variable and update f/q matrices

    def make_transition_matrix(self, time_delta: float=None):
        """
        Implement the transition matrix F for a constant jerk motion model
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
        Implement the process noise covariance Q for a constant jerk motion model
        """
        process_covar = self.validate_process_covar_input(process_covar)
        if time_delta is None:
            time_delta = self.time_delta

        q_mat = np.block([[time_delta ** 7 / 252 * process_covar,
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
        return CovarianceMatrix(q_mat)


class ConstantTurnMotionModel(MotionModel):
    """
    Constant Turn (CT) motion model.

    State vector (via PolarKinematicStateSpace, num_turn_dims=1):
        2D: [px, py, vx, vy, ω]
        3D: [px, py, pz, vx, vy, vz, ω]

    The turn rate ω (rad/s) is a tracked state.  The velocity vector rotates in the
    horizontal (x-y) plane at rate ω.  For 3D, the z-component propagates as constant
    velocity (no vertical rotation); only yaw-only turn (num_turn_dims=1) is supported.

    Prediction uses the exact nonlinear discrete-time transition; covariance is propagated
    EKF-style via the linearised Jacobian evaluated at the current state estimate.

    :param num_dims:            Number of spatial dimensions (2 or 3).
    :param process_covar:       Kinematic process noise spectral density: scalar, 1-D array
                                of length num_dims, or (num_dims × num_dims) matrix.
    :param process_covar_omega: Turn-rate process noise spectral density [rad²/s³].
                                Discrete-time variance for ω is process_covar_omega * dt.
    :param time_delta:          Optional fixed time step [s]; pre-computes Q if provided.
    """
    state_space: PolarKinematicStateSpace

    def __init__(self, num_dims: int, process_covar: npt.ArrayLike = None,
                 process_covar_omega: float = None, time_delta: float = None):
        super().__init__()

        if num_dims not in (2, 3):
            raise ValueError("ConstantTurnMotionModel requires num_dims in {2, 3}")

        self.state_space = PolarKinematicStateSpace(num_dims=num_dims, has_vel=True,
                                                    has_accel=False, num_turn_dims=1)
        self.time_delta = time_delta
        self.process_covar = process_covar
        self.process_covar_omega = process_covar_omega if process_covar_omega is not None else 1.0
        self.update_time_step(time_delta)

    @property
    def is_linear(self) -> bool:
        return False

    def make_transition_matrix(self, time_delta: float = None):
        """Not applicable to this nonlinear model. Use make_jacobian instead."""
        raise NotImplementedError(
            "ConstantTurnMotionModel is nonlinear; the transition matrix is state-dependent. "
            "Use make_jacobian(x, time_delta) for EKF covariance propagation."
        )

    def transition_function(self, curr_state: State, time_delta: float) -> State:
        """
        Return a callable f(x) that propagates the CT state vector forward by time_delta.

        For |ω·dt| < 1e-6 the function falls back to the straight-line (CV) limit to avoid
        numerical blow-up in the sin(ω·dt)/ω and (1-cos(ω·dt))/ω terms.
        """
        if self.q is None or self.time_delta is not time_delta:
            self.time_delta = time_delta
            self.q = self.make_process_covariance_matrix(self.process_covar, time_delta)

        dt = time_delta
        n  = self.num_dims

        x = curr_state.position
        v = curr_state.velocity
        vx = v[0]
        vy = v[1]

        assert self.state_space.num_turn_dims == 1, "ConstantTurnMotionModel assumes only one turn dimension, state space defines two."

        omega = curr_state.state[self.state_space.turn_rate_slice].item()
        odt   = omega * dt  # amount of heading change

        if abs(odt) < 1e-6:           # straight-line limit (Taylor expansion)
            sow = dt
            com = 0.0
        else:
            sow = np.sin(odt) / omega
            com = (1.0 - np.cos(odt)) / omega

        new_x        = curr_state.state.copy()  # start from current state; only updated entries change
        new_x[0]     = x[0] + sow * vx - com * vy            # px'
        new_x[1]     = x[1] + com * vx + sow * vy            # py'
        new_x[n]     =  np.cos(odt) * vx - np.sin(odt) * vy  # vx'
        new_x[n + 1] =  np.sin(odt) * vx + np.cos(odt) * vy  # vy'
        if n == 3:
            new_x[2] = x[2] + dt * v[2]                  # pz' = pz + dt·vz
            # vz' = vz (unchanged), ω' = ω (unchanged)

        return State(state_space=curr_state.state_space, time=curr_state.time+time_delta,
                     state=new_x, covar=curr_state.covar)

    def transition_matrix(self, s: State, time_delta: float) -> npt.NDArray:
        """
        Return the Jacobian of the CT transition function at state x.

        Computed analytically; small-angle fallback applied when |ω·dt| < 1e-6.
        """
        x     = s.state
        dt    = time_delta
        n     = self.num_dims
        ns    = self.num_states
        vx    = float(x[n])
        vy    = float(x[n + 1])
        omega = float(x[-1])
        odt   = omega * dt
        s     = np.sin(odt)
        c     = np.cos(odt)

        if abs(odt) < 1e-6:              # Taylor limits as ω → 0
            sow          =  dt
            com          =  0.0
            d_sow_domega = -dt ** 3 / 3  # lim d(sin(ωdt)/ω)/dω
            d_com_domega =  dt ** 2 / 2  # lim d((1-cos(ωdt))/ω)/dω
        else:
            sow          =  s / omega
            com          = (1.0 - c) / omega
            d_sow_domega =  dt * c / omega - s / omega ** 2
            d_com_domega =  dt * s / omega - (1.0 - c) / omega ** 2

        F = np.eye(ns)

        # Position rows (x, y) ← velocity and turn-rate columns
        F[0, n]     =  sow
        F[0, n + 1] = -com
        F[0, -1]    =  d_sow_domega * vx - d_com_domega * vy   # ∂px'/∂ω

        F[1, n]     =  com
        F[1, n + 1] =  sow
        F[1, -1]    =  d_com_domega * vx + d_sow_domega * vy   # ∂py'/∂ω

        # Velocity rows (x, y) ← velocity and turn-rate columns
        F[n,     n]     =  c
        F[n,     n + 1] = -s
        F[n,     -1]    = -dt * s * vx - dt * c * vy           # ∂vx'/∂ω

        F[n + 1, n]     =  s
        F[n + 1, n + 1] =  c
        F[n + 1, -1]    =  dt * c * vx - dt * s * vy           # ∂vy'/∂ω

        if n == 3:
            F[2, n + 2] = dt   # ∂pz'/∂vz  (z propagates as CV; all other 3D entries = identity)

        return F

    def make_process_covariance_matrix(self, process_covar: npt.ArrayLike = None,
                                       time_delta: float = None) -> CovarianceMatrix:
        """
        Build Q = block_diag(Q_kin, Q_ω) where:
          Q_kin  is the (2·num_dims × 2·num_dims) CV-style kinematic noise block, and
          Q_ω    = process_covar_omega · dt  (random-walk / continuous-white-noise model for ω).
        """
        process_covar = self.validate_process_covar_input(process_covar)
        if time_delta is None:
            time_delta = self.time_delta
        dt = time_delta

        q_kin = np.block([[.25 * dt ** 4 * process_covar, .5 * dt ** 3 * process_covar],
                          [.5  * dt ** 3 * process_covar, dt ** 2      * process_covar]])

        q_omega = np.array([[self.process_covar_omega * dt]])
        n2      = 2 * self.num_dims
        q_mat   = np.block([[q_kin,              np.zeros((n2, 1))],
                            [np.zeros((1, n2)),  q_omega          ]])
        return CovarianceMatrix(q_mat)


class BallisticMotionModel(MotionModel):
    """
    Ballistic trajectory motion model (3-D, gravity-only forcing).

    State vector (CartesianStateSpace, num_dims=3):
        [px, py, pz, vx, vy, vz]

    The transition is the standard 3-D constant-velocity matrix F (identical to
    ConstantVelocityMotionModel with num_dims=3).  Gravity enters as a deterministic
    affine bias applied to the predicted state mean after the linear step:

        pz' += 0.5 * gravity * dt²
        vz' += gravity * dt

    Because the Jacobian is state-independent (F is constant), covariance propagates
    exactly as in a linear KF: P' = F P Fᵀ + Q.  ``is_linear`` is therefore True and
    the base-class predict() path is reused for the covariance update; only the mean
    state requires the extra gravity offset.

    :param process_covar: Kinematic process noise spectral density: scalar, 1-D array
                          of length 3, or (3 × 3) matrix.
    :param gravity:       Gravitational acceleration [m/s²] in the −z direction.
                          Pass a positive value; internally stored as negative
                          (default −9.80665 m/s²).  Alternatively supply a full
                          3-element gravity vector to model non-standard orientations.
    :param time_delta:    Optional fixed time step [s]; pre-computes F and Q if given.
    """

    def __init__(self, num_dims: int = 3, process_covar: npt.ArrayLike = None,
                 gravity: npt.ArrayLike = -9.80665,
                 time_delta: float = None):
        super().__init__()

        if num_dims != 3:
            raise ValueError("BallisticMotionModel requires num_dims=3")

        self.state_space = CartesianStateSpace(num_dims=3, has_vel=True, has_accel=False)

        gravity_arr = np.asarray(gravity, dtype=float)
        if gravity_arr.ndim == 0:
            self.gravity_vec = np.array([0., 0., float(gravity_arr)])
        elif gravity_arr.shape == (3,):
            self.gravity_vec = gravity_arr.copy()
        else:
            raise ValueError("gravity must be a scalar or a 3-element array")

        # Initialize time_delta to None so update_time_step always computes F and Q
        # on first call, regardless of what time_delta is passed.
        self.time_delta = None
        self.f = None
        self.q = None
        self.process_covar = process_covar
        self.update_time_step(time_delta)

    def make_transition_matrix(self, time_delta: float = None):
        """Standard CV transition matrix F for 3-D (identical to ConstantVelocityMotionModel)."""
        if time_delta is None:
            time_delta = self.time_delta
        n = self.num_dims   # 3
        return np.block([[np.eye(n), time_delta * np.eye(n)],
                         [np.zeros((n, n)), np.eye(n)]])

    def make_process_covariance_matrix(self, process_covar: npt.ArrayLike = None,
                                       time_delta: float = None) -> CovarianceMatrix:
        """CV-style process noise covariance Q for 3-D."""
        process_covar = self.validate_process_covar_input(process_covar)
        if time_delta is None:
            time_delta = self.time_delta
        dt = time_delta
        q_mat = np.block([[.25 * dt ** 4 * process_covar, .5 * dt ** 3 * process_covar],
                          [.5  * dt ** 3 * process_covar, dt ** 2      * process_covar]])
        return CovarianceMatrix(q_mat)

    def make_control_input(self, time_delta: float) -> npt.NDArray:
        """
        Gravity control vector u = [0,0,½g·dt², 0,0,g·dt]ᵀ.

        Applied by the base-class predict() as the affine term x_{k+1} = F·x_k + u(dt).
        Covariance propagation is unaffected (gravity is deterministic, not stochastic).

        :param time_delta: Prediction interval [s]
        :return: Control vector of shape (num_states,)
        """
        u = np.zeros(self.state_space.num_states)
        u[self.state_space.pos_slice] = 0.5 * time_delta ** 2 * self.gravity_vec
        u[self.state_space.vel_slice] = time_delta * self.gravity_vec
        return u


class ConstantTurnRateAccelerationMotionModel(MotionModel):
    """
    Constant Turn Rate and Acceleration (CTRA) motion model.

    State vector (via PolarKinematicStateSpace, has_accel=True, num_turn_dims=1):
        2D: [px, py, vx, vy, ax, ay, ω]          — 7 states
        3D: [px, py, pz, vx, vy, vz, ax, ay, az, ω]  — 10 states

    The turn rate ω (rad/s) and acceleration vector are tracked states.  The velocity
    vector rotates in the horizontal (x-y) plane at rate ω, while a world-frame
    acceleration acts as an additive forcing term on both position and velocity.
    For 3D, the z-component propagates as constant acceleration (no vertical rotation);
    only yaw-only turn (num_turn_dims=1) is supported.

    Prediction uses the exact nonlinear discrete-time transition; covariance is propagated
    EKF-style via the linearised Jacobian evaluated at the current state estimate.

    :param num_dims:            Number of spatial dimensions (2 or 3).
    :param process_covar:       Kinematic process noise spectral density: scalar, 1-D array
                                of length num_dims, or (num_dims × num_dims) matrix.
    :param process_covar_omega: Turn-rate process noise spectral density [rad²/s³].
                                Discrete-time variance for ω is process_covar_omega * dt.
    :param time_delta:          Optional fixed time step [s]; pre-computes Q if provided.
    """

    def __init__(self, num_dims: int, process_covar: npt.ArrayLike = None,
                 process_covar_omega: float = None, time_delta: float = None):
        super().__init__()

        if num_dims not in (2, 3):
            raise ValueError("ConstantTurnRateAccelerationMotionModel requires num_dims in {2, 3}")

        self.state_space = PolarKinematicStateSpace(num_dims=num_dims, has_vel=True,
                                                    has_accel=True, num_turn_dims=1)
        self.time_delta = time_delta
        self.process_covar = process_covar
        self.process_covar_omega = process_covar_omega if process_covar_omega is not None else 1.0
        self.update_time_step(time_delta)

    @property
    def is_linear(self) -> bool:
        return False

    def transition_function(self, curr_state: State, time_delta: float) -> State:
        """Propagate the CTRA state vector forward by time_delta."""
        if time_delta != self.time_delta:
            self.time_delta = time_delta
            self.q = self.make_process_covariance_matrix(self.process_covar, time_delta)

        f = self.make_transition_function(time_delta)
        new_x = f(curr_state.state)
        return State(state_space=curr_state.state_space, time=curr_state.time + time_delta,
                     state=new_x, covar=curr_state.covar)

    def make_transition_matrix(self, time_delta: float = None):
        """Not applicable to this nonlinear model. Use make_jacobian instead."""
        raise NotImplementedError(
            "ConstantTurnRateAccelerationMotionModel is nonlinear; the transition matrix is "
            "state-dependent. Use make_jacobian(x, time_delta) for EKF covariance propagation."
        )

    def make_transition_function(self, time_delta: float):
        """
        Return a callable f(x) that propagates the CTRA state vector forward by time_delta.

        CT rotation kinematics are applied to the position and velocity states; world-frame
        acceleration enters as an additive bias.  For |ω·dt| < 1e-6 the function falls back
        to the straight-line (CA) limit to avoid numerical blow-up.
        """
        dt = time_delta
        n  = self.num_dims

        def f(x):
            x     = np.asarray(x, dtype=float)
            omega = float(x[-1])
            vx, vy = x[n], x[n + 1]
            ax, ay = x[2 * n], x[2 * n + 1]
            odt    = omega * dt

            if abs(odt) < 1e-6:           # straight-line (CA) limit
                sow = dt
                com = 0.0
            else:
                sow = np.sin(odt) / omega
                com = (1.0 - np.cos(odt)) / omega

            new_x        = x.copy()
            new_x[0]     = x[0] + sow * vx - com * vy + 0.5 * dt ** 2 * ax   # px'
            new_x[1]     = x[1] + com * vx + sow * vy + 0.5 * dt ** 2 * ay   # py'
            new_x[n]     =  np.cos(odt) * vx - np.sin(odt) * vy + dt * ax    # vx'
            new_x[n + 1] =  np.sin(odt) * vx + np.cos(odt) * vy + dt * ay    # vy'
            # ax, ay, ω unchanged

            if n == 3:
                vz = x[n + 2]
                az = x[2 * n + 2]
                new_x[2]     = x[2] + dt * vz + 0.5 * dt ** 2 * az  # pz'
                new_x[n + 2] = vz + dt * az                          # vz'
                # az unchanged

            return new_x

        return f

    def transition_matrix(self, x, time_delta: float) -> npt.NDArray:
        """
        Return the Jacobian of the CTRA transition function at state x.

        Computed analytically; small-angle fallback applied when |ω·dt| < 1e-6.

        :param x: State object or ndarray of shape (num_states,)
        """
        if isinstance(x, State):
            x = x.state
        x     = np.asarray(x, dtype=float)
        dt    = time_delta
        n     = self.num_dims
        ns    = self.num_states
        vx    = float(x[n])
        vy    = float(x[n + 1])
        omega = float(x[-1])
        odt   = omega * dt
        s     = np.sin(odt)
        c     = np.cos(odt)

        if abs(odt) < 1e-6:              # Taylor limits as ω → 0
            sow          =  dt
            com          =  0.0
            d_sow_domega =  0.0          # lim_{ω→0} d(sin(ωdt)/ω)/dω = 0
            d_com_domega =  dt ** 2 / 2
        else:
            sow          =  s / omega
            com          = (1.0 - c) / omega
            d_sow_domega =  dt * c / omega - s / omega ** 2
            d_com_domega =  dt * s / omega - (1.0 - c) / omega ** 2

        F = np.eye(ns)

        # ── Position rows (px', py') ────────────────────────────────────────────
        F[0, n]         =  sow                                      # ∂px'/∂vx
        F[0, n + 1]     = -com                                      # ∂px'/∂vy
        F[0, 2 * n]     =  0.5 * dt ** 2                           # ∂px'/∂ax
        F[0, -1]        =  d_sow_domega * vx - d_com_domega * vy   # ∂px'/∂ω

        F[1, n]         =  com                                      # ∂py'/∂vx
        F[1, n + 1]     =  sow                                      # ∂py'/∂vy
        F[1, 2 * n + 1] =  0.5 * dt ** 2                           # ∂py'/∂ay
        F[1, -1]        =  d_com_domega * vx + d_sow_domega * vy   # ∂py'/∂ω

        # ── Velocity rows (vx', vy') ────────────────────────────────────────────
        F[n,     n]         =  c                                    # ∂vx'/∂vx
        F[n,     n + 1]     = -s                                    # ∂vx'/∂vy
        F[n,     2 * n]     =  dt                                   # ∂vx'/∂ax
        F[n,     -1]        = -dt * s * vx - dt * c * vy           # ∂vx'/∂ω

        F[n + 1, n]         =  s                                    # ∂vy'/∂vx
        F[n + 1, n + 1]     =  c                                    # ∂vy'/∂vy
        F[n + 1, 2 * n + 1] =  dt                                   # ∂vy'/∂ay
        F[n + 1, -1]        =  dt * c * vx - dt * s * vy           # ∂vy'/∂ω

        # ── 3D: z sub-system (pz', vz') ────────────────────────────────────────
        if n == 3:
            F[2,     n + 2]     = dt            # ∂pz'/∂vz
            F[2,     2 * n + 2] = 0.5 * dt ** 2 # ∂pz'/∂az
            F[n + 2, 2 * n + 2] = dt            # ∂vz'/∂az

        return F

    def make_process_covariance_matrix(self, process_covar: npt.ArrayLike = None,
                                       time_delta: float = None) -> CovarianceMatrix:
        """
        Build Q = block_diag(Q_kin, Q_ω) where:
          Q_kin  is the (3·num_dims × 3·num_dims) CA-style kinematic noise block, and
          Q_ω    = process_covar_omega · dt  (random-walk model for ω).
        """
        process_covar = self.validate_process_covar_input(process_covar)
        if time_delta is None:
            time_delta = self.time_delta
        dt  = time_delta
        pc  = process_covar

        q_kin = np.block([[.25 * dt ** 4 * pc, .5 * dt ** 3 * pc, .5 * dt ** 2 * pc],
                          [.5  * dt ** 3 * pc, dt ** 2      * pc, dt             * pc],
                          [.5  * dt ** 2 * pc, dt           * pc, pc                 ]])

        q_omega = np.array([[self.process_covar_omega * dt]])
        n3      = 3 * self.num_dims
        q_mat   = np.block([[q_kin,              np.zeros((n3, 1))],
                            [np.zeros((1, n3)),  q_omega          ]])
        return CovarianceMatrix(q_mat)


# =============== Elementary Kalman Filter and Extended Kalman Filter Prediction Functions ================
def kf_predict(x_est: State, q: CovarianceMatrix, f: npt.ArrayLike, time_step: float=0.0,
               u: npt.ArrayLike = None) -> State:
    """
    Conduct a Kalman Filter prediction, given the current estimated state and covariance, a transition matrix, and
    the process noise covariance.

    :param x_est: Current state estimate; includes an estimated state error covariance
    :param q: Process noise covariance, CovarianceMatrix object with size n_states
    :param f: Transition matrix, shape: (n_states, n_states)
    :param time_step: float (optional, default is zero). For advancing the time index of the new state.
    :param u: Control input vector of shape (n_states,), or None (default). When provided, added
              to the predicted mean: x_pred = F·x + u. Does not affect covariance propagation.
    :return x_pred: Predicted state (State)
    """

    # Predict the next state, applying optional deterministic control input
    xx_pred = f @ x_est.state
    if u is not None:
        xx_pred = xx_pred + u

    # Predict the next state error covariance
    pp_pred = CovarianceMatrix(f @ x_est.covar.cov @ f.T + q.cov)

    return State(state_space=x_est.state_space, state=xx_pred, covar=pp_pred, time=x_est.time + time_step)


def ekf_predict(x_est: State, q: CovarianceMatrix, transition_fun: Callable[[State, float], State],
                jacobian_fun: Callable[[State, float], npt.NDArray], time_step: float) -> State:
    """
    Conduct an Extended Kalman Filter prediction, given the current estimated state and covariance, a function handle
    to generate the predicted state, and a function handle to generate the transition matrix.

    :param x_est: Current state estimate, State object
    :param q: Process noise covariance, CovarianceMatrix object with size n_states
    :param transition_fun: Transition function handle, accepts a State and a time step; returns a State
    :param jacobian_fun: Transition Jacobian function handle, returns a matrix of shape: (n_states, n_states)
    :param time_step: float (optional, default is zero). For advancing the time index of the new state.
    :return x_pred: Predicted state estimate, State object
    """

    # Forward prediction of state
    xx_pred = transition_fun(x_est, time_step)  # State object

    # Forward prediction of state error covariance
    f = jacobian_fun(x_est, time_step)
    pp_pred = CovarianceMatrix(f @ x_est.covar.cov @ np.transpose(f) + q.cov)

    return State(state_space=x_est.state_space, state=xx_pred.state, covar=pp_pred, time=x_est.time + time_step)


def ukf_predict(x_est: State, q: CovarianceMatrix, f_fun, time_step: float=0.0,
                alpha: float = 1e-3, beta: float = 2., kappa: float = 0.) -> State:
    """
    Conduct an Unscented Kalman Filter (UKF) prediction using the scaled unscented transform.

    Unlike ekf_predict, no Jacobian is required.  The transition function is applied
    directly to 2n+1 sigma points chosen to capture the mean and covariance of the
    prior.  This makes ukf_predict drop-in compatible with any nonlinear transition
    function, including those where the Jacobian is unavailable or inaccurate.

    For the current motion models, pass::

        f_fun = motion_model.make_transition_function(time_delta)

    The same f_fun used by ekf_predict works unchanged here.

    Scaled unscented transform parameters (Julier & Uhlmann / Van der Merwe):
      λ = α²(n + κ) − n
      W_m[0] = λ/(n+λ),  W_m[i] = 1/(2(n+λ))  for i = 1 … 2n
      W_c[0] = W_m[0] + (1 − α² + β),  W_c[i] = W_m[i]  for i = 1 … 2n

    :param x_est:  Current state estimate, State object
    :param q:      Process noise covariance, CovarianceMatrix of size n_states.
    :param f_fun:  Transition function f(x) → x_new, maps (n_states,) → (n_states,).
    :param time_step: float (optional, default is zero). For advancing the time index of the new state.
    :param alpha:  Sigma-point spread (1e-4 … 1). Smaller values cluster points near
                   the mean; larger values explore more of the distribution. Default 1e-3.
    :param beta:   Prior distribution parameter. beta=2 is optimal for Gaussian priors.
                   Default 2.
    :param kappa:  Secondary scaling (0 or 3−n are common choices). Default 0.
    :return x_pred: Predicted state estimate, State object.
    """
    n   = x_est.size
    lam = alpha ** 2 * (n + kappa) - n

    # ── Weights ───────────────────────────────────────────────────────────────
    w_common = 0.5 / (n + lam)
    W_m      = np.full(2 * n + 1, w_common)
    W_c      = np.full(2 * n + 1, w_common)
    W_m[0]   = lam / (n + lam)
    W_c[0]   = lam / (n + lam) + (1. - alpha ** 2 + beta)

    # ── Matrix square root of (n+λ)·P ────────────────────────────────────────
    try:
        L = np.linalg.cholesky((n + lam) * x_est.covar.cov)
    except np.linalg.LinAlgError:
        # Near-singular P: fall back to symmetric square root via eigendecomposition
        eigvals, eigvecs = np.linalg.eigh((n + lam) * x_est.covar.cov)
        L = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 0.)))

    # ── Sigma points ──────────────────────────────────────────────────────────
    sigma = np.empty((n, 2 * n + 1))
    sigma[:, 0] = x_est.state
    for i in range(n):
        sigma[:, i + 1]     = x_est.state + L[:, i]
        sigma[:, i + 1 + n] = x_est.state - L[:, i]

    # ── Propagate sigma points through transition function ────────────────────
    sigma_pred = np.column_stack([f_fun(sigma[:, i]) for i in range(2 * n + 1)])

    # ── Predicted mean ────────────────────────────────────────────────────────
    xx_pred = sigma_pred @ W_m

    # ── Predicted covariance ──────────────────────────────────────────────────
    diff   = sigma_pred - xx_pred[:, np.newaxis]
    pp_pred = CovarianceMatrix((W_c * diff) @ diff.T + q.cov)

    return State(state_space=x_est.state_space, state=xx_pred, covar=pp_pred, time=x_est.time + time_step)