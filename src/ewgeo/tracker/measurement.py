from collections.abc import Callable
import numpy as np
import numpy.typing as npt

from ..utils.constraints import snap_to_constraints
from ..utils.covariance import CovarianceMatrix
from ..utils.system import PassiveSurveillanceSystem
from .states import StateSpace, State


class Measurement:
    time: float
    sensor: PassiveSurveillanceSystem | None
    zeta: npt.NDArray[np.float64]

    def __init__(self, time: float, sensor: PassiveSurveillanceSystem | None, zeta: npt.NDArray[np.float64]):
        """
        :param time: Timestamp of the measurement [seconds]
        :param sensor: PassiveSurveillanceSystem that produced this measurement, or None
        :param zeta: Measurement vector
        """
        self.time = time
        self.sensor = sensor
        self.zeta = zeta

    @property
    def size(self)->int:
        """Number of scalar elements in the measurement vector."""
        return self.zeta.size

    def __str__(self):
        return f"Measurement at t={self.time}: {self.zeta}"

class MeasurementModel:
    """
    Parent class to capture the measurement model for a KF/EKF-system.

    Used to update a track given new measurements
    """
    state_space: StateSpace
    pss: PassiveSurveillanceSystem

    def __init__(self, pss: PassiveSurveillanceSystem,
                 ineq_constraints: list | None = None,
                 solver_fun: Callable | None = None,
                 crlb_fun: Callable | None = None):
        """
        :param pss:              PassiveSurveillanceSystem used for EKF measurement prediction,
                                 Jacobian evaluation, and log-likelihood scoring.  Also used as the
                                 default solver/CRLB when solver_fun/crlb_fun are not provided.
        :param ineq_constraints: Optional list of inequality constraint callables applied to the
                                 position estimate produced by state_from_measurement. Each callable
                                 has signature (x: ndarray (num_dims, n)) -> (eps: ndarray (n,),
                                 x_valid: ndarray (num_dims, n)). Applied as a post-solve position
                                 snap (the LS solver operates on pos+vel and cannot apply
                                 position-only constraints mid-iteration).
        :param solver_fun:       Optional callable for track initiation.  Signature::

                                     pos_est = solver_fun(zeta, x_init)

                                 where ``zeta`` is the measurement vector (shape ``(n_meas,)``),
                                 ``x_init`` is a starting-point array of length ``num_dims`` or
                                 ``2*num_dims`` (pos or pos+vel), and the return value is a
                                 position array of length ``num_dims`` (or pos+vel of length
                                 ``2*num_dims``).  When ``None``, ``pss.least_square`` is used.
                                 Supply a custom callable to use a GD solver, a bounded LS, a
                                 centroid estimator (e.g. for DirectionFinder), or any other
                                 inversion method.
        :param crlb_fun:         Optional callable for the position CRLB used as the buffer-track
                                 and initial-track covariance.  Signature::

                                     crlb = crlb_fun(x_source)

                                 where ``x_source`` is a position array of length ``num_dims`` and
                                 the return value is either a ``CovarianceMatrix`` or an ndarray of
                                 shape ``(num_dims, num_dims)``.  When ``None``,
                                 ``pss.compute_crlb`` is used.  Supply a custom callable when the
                                 PSS CRLB is unavailable, too expensive, or the solver has a
                                 different uncertainty model (e.g. empirical bootstrap covariance
                                 from a centroid estimator).
        """
        self.pss = pss
        self.ineq_constraints = ineq_constraints
        self._solver_fun = solver_fun
        self._crlb_fun = crlb_fun

    @property
    def num_measurement_dimensions(self)->int:
        """Number of scalar elements produced by one call to the underlying PSS measurement function."""
        return self.pss.num_measurements

    def false_alarm(self, max_val: float, num: int, time: float = None)-> list[Measurement]:
        """
        Generate a list of uniformly-distributed false-alarm measurements.

        :param max_val: Measurements are drawn uniformly from [-max_val, max_val] in each dimension
        :param num: Number of false-alarm measurements to generate
        :param time: Timestamp to assign to each measurement (default: None)
        :return: List of Measurement objects
        """
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
        args = {'x_source': state.state_space.pos_component(state.state),
                'v_source': state.state_space.vel_component(state.state)}
        if noise == True:
            z = self.pss.noisy_measurement(**args)
        else:
            z = self.pss.measurement(**args)
            if isinstance(noise, np.ndarray):
                z = z + noise

        return Measurement(zeta=z, sensor=self.pss, time=state.time)

    def jacobian(self, state: State) -> npt.NDArray:
        """
        Compute the H matrix (measurement Jacobian) for the Kalman filter update at the given state.

        The PSS Jacobian is evaluated at the state's position (and velocity, if available), then mapped
        into the full tracker state vector shape via the state space slices.

        :param state: State at which to evaluate the Jacobian
        :return: H matrix of shape (num_measurements, num_states)
        """
        j = self.pss.jacobian(x_source=state.state_space.pos_component(state.state),
                              v_source=state.state_space.vel_component(state.state))

        # Jacobian may be either w.r.t. position-only (pss.num_dim rows) or pos/vel,
        # depending on which type of pss we're calling.

        # Build the H matrix
        h = np.zeros((self.num_measurement_dimensions, state.state_space.num_states))
        h[:, state.state_space.pos_slice] = np.transpose(j[:self.pss.num_dim, :])
        if state.state_space.has_vel and j.shape[0] > self.pss.num_dim:
            # The state space has velocity components, and the pss returned rows for
            # the jacobian w.r.t. velocity.
            h[:, state.state_space.vel_slice] = np.transpose(j[self.pss.num_dim:, :])

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
        """
        Compute the log-likelihood of an existing Measurement given a candidate State.

        :param state: Candidate state (the hypothesized truth)
        :param measurement: Measurement to score
        :return: Log-likelihood scalar
        """
        return self.pss.log_likelihood(x_source=state.state_space.pos_component(state.state),
                                       v_source=state.state_space.vel_component(state.state),
                                       zeta=measurement.zeta)

    def state_from_measurement(self, m: Measurement, state_space: StateSpace,
                               truth_state: bool=False,
                               x_init: npt.ArrayLike | None = None,
                               target_max_velocity: float | None = None,
                               target_max_acceleration: float | None = None)-> State:
        """
        Attempt to populate a state from a measurement.

        We will first generate an empty state, based on the state space, then attempt to populate the
        position and velocity using the least square solver from the underlying PassiveSurveillanceSystem object stored
        in self.pss.

        :param m: Input measurement from which the state will be estimated
        :param truth_state: Boolean flag (default=False) declaring whether this should be considered a
                            truth state (no error) or an estimated state (with error)
        :param x_init: Optional initial position (or pos+vel) estimate for the LS solver. If position-only
                       (num_dims elements), velocity is padded with zeros. Providing a good starting point
                       (e.g. the previous scan's position) dramatically improves convergence for distant
                       targets and eliminates mirror-image z-ambiguity when z_init > 0.
        :param target_max_velocity: Optional upper bound on target speed [m/s]. When provided, the velocity
                                    block of the initial covariance is scaled so that its largest diagonal
                                    element does not exceed target_max_velocity². Applied after the sentinel
                                    check. If None, the covariance is left as-is.
        :param target_max_acceleration: Optional upper bound on target acceleration [m/s²]. When provided,
                                        the acceleration block of the initial covariance is scaled so that
                                        its largest diagonal element does not exceed target_max_acceleration².
                                        Only applies when the state space has an acceleration component.
                                        If None, the covariance is left as-is.
        """

        # Resolve solver and CRLB callables — use explicit overrides when supplied,
        # otherwise fall back to the PSS methods.  This lets callers substitute a GD
        # solver, a bounded LS, a centroid estimator, or any other inversion method
        # without subclassing MeasurementModel.
        n = state_space.num_dims
        if self._solver_fun is not None:
            def _solve(zeta, x0):
                result = np.asarray(self._solver_fun(zeta, x0), dtype=float).ravel()
                # Pad to at least 2*n in case the custom solver returns position-only
                out = np.zeros((2*n,))
                out[:min(result.size, 2*n)] = result[:min(result.size, 2*n)]
                return out
        else:
            def _solve(zeta, x0):
                result, _ = self.pss.least_square(zeta=zeta, x_init=x0, max_num_iterations=100)
                return result

        if x_init is not None:
            x_init = np.asarray(x_init, dtype=float).ravel()
            init_pos_vel = np.zeros((2*n, ))
            init_pos_vel[:min(x_init.size, 2*n)] = x_init[:min(x_init.size, 2*n)]
        else:
            init_pos_vel = np.zeros((2*n, ))
        pos_vel_est = _solve(m.zeta, init_pos_vel)

        # If the 3D result has z < 0, retry with the z sign flipped to escape the
        # mirror-image local minimum that arises with near-coplanar sensor arrays.
        if n >= 3 and pos_vel_est[2] < 0:
            retry_init = np.zeros((2*n, ))
            retry_init[2] = -pos_vel_est[2]
            pos_vel_est_retry = _solve(m.zeta, retry_init)
            if pos_vel_est_retry[2] >= 0:
                pos_vel_est = pos_vel_est_retry

        # Sanity check: if the LS result is unreasonably far from the sensor origin
        # (> 5000 km), the solver likely diverged.  Reset to zeros so that the caller
        # can detect the failure via `norm(position) < 1` and skip the result.
        if np.linalg.norm(pos_vel_est[:n]) > 5e6:
            pos_vel_est = np.zeros((2*n, ))

        # Snap position to inequality constraints (e.g. altitude bounds), in case the LS
        # solver converged to a point that still marginally violates one.
        if self.ineq_constraints is not None:
            pos = pos_vel_est[:n].reshape(n, 1)
            pos_vel_est[:n] = snap_to_constraints(pos, ineq_constraints=self.ineq_constraints).ravel()
        # TODO: Also add support for equality constraints, once this is verified.

        # Convert to a state vector
        init_state_vec = np.zeros((state_space.num_states, ))
        init_state_vec[state_space.pos_vel_slice] = pos_vel_est

        # Initialize the state without a covariance matrix
        s = State(state_space=state_space, time=m.time, state=init_state_vec)

        if not truth_state:
            # Compute the position CRLB — use the explicit override when supplied.
            if self._crlb_fun is not None:
                crlb_raw = self._crlb_fun(s.pos_vel)
                crlb = crlb_raw if isinstance(crlb_raw, CovarianceMatrix) else CovarianceMatrix(np.atleast_2d(crlb_raw))
            else:
                crlb = self.pss.compute_crlb(x_source=s.pos_vel)
            # Note: if the PSS system has no velocity information, the CRLB calculation
            # will return a matrix of zeros for those elements.
            init_covar = 1e6*np.eye(state_space.num_states)
            init_covar[:crlb.size, :crlb.size] = crlb.cov

            # Check for a velocity component.
            # Replace the velocity block if:
            #   (a) it is near-zero (numerical noise, not a true estimate), or
            #   (b) it is extremely large (CRLB sentinel value like 1e99 for unobservable
            #       dims) — values that large cause catastrophic cancellation in F@P@F^T.
            vel_slice = state_space.vel_slice
            vel_diag = np.diag(init_covar[vel_slice, vel_slice])
            if np.all(vel_diag <= 1e-3) or np.any(vel_diag > 1e12):
                init_covar[vel_slice, vel_slice] = 1e6*np.eye(state_space.num_dims)

            if target_max_velocity is not None:
                vel_cov = init_covar[vel_slice, vel_slice]
                max_diag = np.max(np.diag(vel_cov))
                if max_diag > target_max_velocity**2:
                    init_covar[vel_slice, vel_slice] = vel_cov * (target_max_velocity**2 / max_diag)

            if target_max_acceleration is not None and state_space.has_accel:
                accel_slice = state_space.accel_slice
                accel_cov = init_covar[accel_slice, accel_slice]
                max_diag = np.max(np.diag(accel_cov))
                if max_diag > target_max_acceleration**2:
                    init_covar[accel_slice, accel_slice] = accel_cov * (target_max_acceleration**2 / max_diag)

            s.covar = CovarianceMatrix(init_covar)

        return s

    def ekf_update(self, x_prev: State, zeta: npt.ArrayLike, cov: CovarianceMatrix | None = None) -> State:
        """
        Conduct an Extended Kalman Filter update, given the previous state estimate and covariance, a measurement function,
        and the measurement matrix function.

        :param x_prev: Previous state estimate, State
        :param zeta: Measurement vector, shape: (n_meas, )
        :param cov: Measurement error covariance, Covariance Matrix with size n_meas
        :return x: Updated state estimate, State
        """

        # Grab covariance matrix from PSS, if not provided
        if cov is None:
            cov = self.pss.cov

        # Evaluate the Measurement and Jacobian at x_prev
        msmt = self.measurement(x_prev)
        jacobian = self.jacobian(x_prev)

        # Compute the Innovation (or Residual)
        y = zeta - msmt.zeta

        # Compute the Innovation Covariance
        p_prev = x_prev.covar.cov  # extract the covariance matrix
        s = jacobian @ p_prev @ jacobian.T + cov.cov

        # Compute the Kalman Gain
        k = p_prev @ jacobian.T @ np.linalg.inv(s)

        # Update the Estimate
        x = x_prev.state + k @ y
        p = CovarianceMatrix((np.eye(x_prev.size) - (k @ jacobian)) @ p_prev)

        return State(state_space=x_prev.state_space, time=x_prev.time, state=x, covar=p)

# =============== Elementary Kalman Filter and Extended Kalman Filter Update Functions ================
def kf_update(x_prev: State, zeta: npt.ArrayLike, cov: CovarianceMatrix, h: npt.ArrayLike) -> State:
    """
    Conduct a Kalman Filter update, given the previous state estimate and covariance, a measurement, and
    the measurement matrix.

    :param x_prev: Previous state estimate, State
    :param zeta: Measurement vector, shape: (n_meas, )
    :param cov: Measurement error covariance (Covariance Matrix object with size n_meas)
    :param h: Measurement matrix, shape: (n_meas, n_states)
    :return x_est: Updated state estimate, State
    """
    # Evaluate the Measurement and Jacobian at x_prev
    z = h @ x_prev.state

    # Compute the Innovation (or Residual)
    y = zeta - z

    # Compute the Innovation Covariance
    s = h @ x_prev.covar.cov @ h.T + cov.cov

    # Compute the Kalman Gain
    k = x_prev.covar.cov @ h.T/s

    # Update the Estimate
    x = x_prev.state + k @ y
    p = CovarianceMatrix((np.eye(x_prev.size) - (k @ h)) @ x_prev.covar.cov)

    return State(state_space=x_prev.state_space, time=x_prev.time, state=x, covar=p)


def ekf_update(x_prev: State, zeta: npt.ArrayLike, cov: CovarianceMatrix,
               msmt_fun: Callable[[State], Measurement],
               jacobian_fun: Callable[[State], npt.NDArray])-> State:
    """
    Conduct an Extended Kalman Filter update, given the previous state estimate and covariance, a measurement function,
    and the measurement matrix function.

    :param x_prev: Previous state estimate, State
    :param zeta: Measurement vector, shape: (n_meas, )
    :param cov: Measurement error covariance, Covariance Matrix with size n_meas
    :param msmt_fun: Function handle for measurement evaluation, accepts a State object and returns a Measurement object
    :param jacobian_fun: Function handle for measurement matrix evaluation, accepts a State object and returns a numpy array
    :return x: Updated state estimate, State
    """

    # Evaluate the Measurement and Jacobian at x_prev
    msmt = msmt_fun(x_prev)
    jacobian = jacobian_fun(x_prev)

    # Compute the Innovation (or Residual)
    y = zeta - msmt.zeta

    # Compute the Innovation Covariance
    p_prev = x_prev.covar.cov  # extract the covariance matrix
    s = jacobian @ p_prev @ jacobian.T + cov.cov

    # Compute the Kalman Gain
    k = p_prev @ jacobian.T @ np.linalg.inv(s)

    # Update the Estimate
    x = x_prev.state + k @ y
    p = CovarianceMatrix((np.eye(x_prev.size)- (k @ jacobian)) @ p_prev)

    return State(state_space=x_prev.state_space, time=x_prev.time, state=x, covar=p)
