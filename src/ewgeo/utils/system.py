from abc import ABC, abstractmethod
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from . import make_pdfs, safe_2d_shape, SearchSpace
from .covariance import CovarianceMatrix
from .perf import compute_crlb_gaussian
from .solvers import ml_solver, gd_solver, ls_solver, bestfix_solver, sensor_calibration

class PassiveSurveillanceSystem(ABC):
    _cov: CovarianceMatrix | None = None
    pos: npt.ArrayLike
    num_sensors: int
    num_dim: int
    num_measurements: int

    # Optional Fields
    bias: npt.ArrayLike | None = None              # User-defined sensor measurement biases
    _vel: npt.ArrayLike | None = None               # Sensor velocity; ignored if not defined
    _cov_pos: CovarianceMatrix | None = None    # Assumed sensor position error covariance

    # Default Values
    # --- No default sensor bias search resolution; let each PSS type overwrite this
    default_bias_search_epsilon: float = 0
    default_bias_search_size: int = 1
    # --- Default sensor position error search is to use 11 points per dimension,
    #     with a maximum offset of 25 meters and resolution of 2.5 meters.
    default_sensor_pos_search_epsilon: float = 2.5
    default_sensor_pos_search_size: int = 11
    # --- No default sensor velocity search resolution; let FDOA type overwrite this
    default_sensor_vel_search_epsilon: float = 0
    default_sensor_vel_search_size: int = 1  # By default, we can't search across sensor velocity

    def __init__(self, x: npt.ArrayLike,
                 cov: CovarianceMatrix | npt.ArrayLike | None=None,
                 bias: npt.ArrayLike | None=None,
                 cov_pos: CovarianceMatrix | npt.ArrayLike | None=None,
                 vel: npt.ArrayLike | None=None):
        if len(np.shape(x))<2: x = np.expand_dims(x, 1) # Add a second dimension if there isn't one
        self.pos = x

        # Populate optional fields
        self.cov = cov
        self.bias = bias
        self.cov_pos = cov_pos
        self.vel = vel

    @property
    def vel(self)-> npt.ArrayLike:
        return self._vel

    @vel.setter
    def vel(self, vel: npt.ArrayLike):
        if vel is not None:
            self._vel = vel
        else:
            self._vel = np.zeros_like(self.pos)

    @property
    def num_sensors(self)-> int:
        return self.pos.shape[1]

    @property
    def num_dim(self)-> int:
        return self.pos.shape[0]

    @property
    def num_measurements(self)-> int:
        if self.cov is None:
            raise ValueError("The covariance matrix is not defined; unable to compute the number of measurements that will be generated.")
        return self.cov.cov.shape[0]

    @property
    def cov(self)-> CovarianceMatrix:
        return self._cov

    @cov.setter
    def cov(self, value: CovarianceMatrix | npt.ArrayLike | None):
        if isinstance(value, CovarianceMatrix) or value is None:
            # Nothing to do; set the parameter
            self._cov = value
        else:
            self._cov = CovarianceMatrix(value)
        return

    @property
    def cov_pos(self)-> CovarianceMatrix | None:
        return self._cov_pos

    @cov_pos.setter
    def cov_pos(self, value: CovarianceMatrix | npt.ArrayLike | None):
        if isinstance(value, CovarianceMatrix) or value is None:
            self._cov_pos = value
        else:
            self._cov_pos = CovarianceMatrix(value)
        return

    # ==================== Model Methods ===================
    # These methods define the sensor measurement model, and must be implemented.
    @abstractmethod
    def measurement(self, x_source: npt.ArrayLike, 
                    x_sensor: npt.ArrayLike | None=None,
                    bias: npt.ArrayLike | None=None,
                    v_sensor: npt.ArrayLike | None=None,
                    v_source: npt.ArrayLike | None=None)-> npt.NDArray:
        pass

    def measurement_from_pos_vel(self, pos_vel: npt.ArrayLike, **kwargs)-> npt.NDArray:
        pos, vel = self.parse_source_pos_vel(pos_vel, default_vel=np.zeros_like(pos_vel))
        return self.measurement(x_source=pos, v_source=vel, **kwargs)

    def noisy_measurement(self, x_source: npt.ArrayLike, num_samples:int = 1, **kwargs)-> npt.NDArray:
        """
        Generate a set of noisy measurements. Will return a 3D matrix of shape (num_measurements, num_sources, num_samples),
        except that the latter two dimensions will be removed is their size is equal to 1.

        :param x_source: 1D or 2D array of source positions
        :param num_samples: Optional integer specifying the number of samples to generate
        All other parameters are passed directly to self.measurement() as keyword arguments
        :return: 1D or 2D array of noisy measurements, depending on the shape of x_source and num_samples.
        """

        # Generate noise-free measurement
        _, num_sources = safe_2d_shape(x_source)
        z = np.reshape(self.measurement(x_source, **kwargs), (self.num_measurements, num_sources))

        # Generate noise
        noise = np.reshape(self.cov.sample(num_samples=num_samples*num_sources),
                           (self.num_measurements, num_sources, num_samples))

        # Add the noise
        zeta = z[:, :, np.newaxis] + noise

        # Squeeze singleton-dimensions and return
        if num_samples == 1:
            zeta = zeta[:, :, 0]
        if num_sources == 1:
            zeta = zeta[:, 0]
        return zeta

    @abstractmethod
    def jacobian(self, x_source: npt.ArrayLike, 
                 v_source: npt.ArrayLike | None=None,
                 x_sensor: npt.ArrayLike | None=None,
                 v_sensor: npt.ArrayLike | None=None)-> npt.NDArray:
        pass

    def jacobian_from_posvel(self, pos_vel: npt.ArrayLike, **kwargs):
        """
        Wrapper for the jacobian function that accepts one input a pos_vel vector that is either (num_dim, num_source)
        or (2*num_dim, num_source)

        In the former case, it is taken to be solely a position vector, and the jacobian w.r.t position is returned.
        In the latter, the jacobian is returned with respect to position and velocity (and the second half of pos_vel
        is used as source velocity)
        """
        if 'v_source' in kwargs:
            default_vel = kwargs['v_source']
        else:
            default_vel = np.zeros_like(pos_vel)

        pos, vel = self.parse_source_pos_vel(pos_vel, default_vel=default_vel)
        kwargs['v_source'] = vel
        kwargs['x_source'] = pos

        j = self.jacobian(**kwargs)
        n_dim = np.shape(pos_vel)[0]
        n_dim2 = np.shape(j)[0]
        if n_dim == 2*n_dim2:
            # Calling function is asking for both pos and vel, but the native jacobian function only returned pos
            # Append with a zeros-matrix (measurements do not depend on velocity at all)
            j = np.concatenate((j, np.zeros_like(j)), axis=0)
        elif 2*n_dim == n_dim2:
            # Call function is asking for pos, but jacobian returned both. Strip the extra
            j = j[:n_dim]
        elif n_dim == n_dim2:
            # We got what we wanted
            pass
        else:
            raise ValueError(f"Unable to parse jacobian response; jacobian returned shape {j.shape}, but {n_dim} rows desired.")

        return j

    @abstractmethod
    def jacobian_uncertainty(self, x_source: npt.ArrayLike, **kwargs):
        pass

    @abstractmethod
    def log_likelihood(self, x_source: npt.ArrayLike,
                       zeta: npt.ArrayLike,
                       x_sensor: npt.ArrayLike | None=None,
                       bias: npt.ArrayLike | None=None,
                       v_sensor: npt.ArrayLike | None=None,
                       v_source: npt.ArrayLike | None=None, **kwargs):
        pass

    def log_likelihood_from_posvel(self, pos_vel: npt.ArrayLike, **kwargs):
        pos, vel = self.parse_source_pos_vel(pos_vel, default_vel=np.zeros_like(pos_vel))
        return self.log_likelihood(x_source=pos, v_source=vel, **kwargs)

    def log_likelihood_uncertainty(self, zeta: npt.ArrayLike,
                                   theta: npt.ArrayLike,
                                   do_source_vel: bool=False,
                                   do_sensor_bias: bool=False,
                                   do_sensor_pos: bool=False,
                                   do_sensor_vel: bool=False, **kwargs):
        if not do_sensor_bias and not do_sensor_pos and not do_sensor_vel:
            # None of the uncertainty parameters are being called for; theta is just x_source
            return self.log_likelihood(theta, zeta, **kwargs)

        # Parse the uncertainty vector
        indices = self.parse_uncertainty_indices(theta, do_bias=do_sensor_bias, do_sensor_pos=do_sensor_pos,
                                                 do_sensor_vel=do_sensor_vel)

        # Write a parsing function
        def _parse_uncertainty_vector(th: npt.ArrayLike):
            x = th[indices['source_pos_indices']]
            v = th[indices['source_vel_indices']] if do_source_vel else np.zeros_like(x)
            b = th[indices['bias_indices']] if do_sensor_bias else self.bias
            xs = np.reshape(th[indices['pos_indices']], self.pos.shape) if do_sensor_pos else self.pos
            vs = np.reshape(th[indices['vel_indices']], self.pos.shape) if do_sensor_vel else self.vel
            return x, v, b, xs, vs

        num_parameters, num_source_pos = safe_2d_shape(theta)
        ell = np.zeros((num_source_pos, ))

        # Make sure the source pos is a matrix, rather than simply a vector
        if num_source_pos == 1:
            theta = theta[:, np.newaxis]

        # ToDo: add a beta term for sensor velocity; and an accompanying cov_vel param.
        beta = np.ravel(self.pos) if do_sensor_pos else np.array([])

        for idx_source, th_i in enumerate(theta.T):
            x_i, v_i, b_i, xs_i, vs_i = _parse_uncertainty_vector(th_i)

            # Generate the ideal measurement matrix for this position
            zeta_i = self.measurement(x_sensor=xs_i, v_sensor=vs_i, x_source=x_i, v_source=v_i,
                                      bias=b_i)

            # Evaluate the measurement error
            err = zeta - zeta_i
            if do_sensor_pos:
                err_pos = beta - np.ravel(xs_i)

            # Compute the scaled log likelihood
            ell_x = - self.cov.solve_aca(err)
            if do_sensor_pos:
                ell_pos = - self.cov_pos.solve_aca(err_pos)
            else:
                ell_pos = 0

            ell[idx_source] = ell_x + ell_pos

        return ell

    @abstractmethod
    def grad_x(self,
               x_source: npt.ArrayLike,
               v_source: npt.ArrayLike | None=None,
               x_sensor: npt.ArrayLike | None=None,
               v_sensor: npt.ArrayLike | None=None)-> npt.NDArray:
        pass

    @abstractmethod
    def grad_bias(self,
                        x_source: npt.ArrayLike,
                        v_source: npt.ArrayLike | None=None,
                        x_sensor: npt.ArrayLike | None=None,
                        v_sensor: npt.ArrayLike | None=None)-> npt.NDArray:
        pass

    @abstractmethod
    def grad_sensor_pos(self,
                        x_source: npt.ArrayLike,
                        v_source: npt.ArrayLike | None=None,
                        x_sensor: npt.ArrayLike | None=None,
                        v_sensor: npt.ArrayLike | None=None)-> npt.NDArray:
        pass

    # ==================== Solver Methods ===================
    # These methods define the basic solver methods, and associated utilities, and must be implemented.
    def max_likelihood(self, zeta: npt.ArrayLike,
                       search_space: SearchSpace,
                       bias: npt.ArrayLike | None=None,
                       cal_data: dict | None=None,
                       **kwargs):

        if cal_data is not None:
            x_sensor, v_sensor, bias = self.sensor_calibration(**cal_data)
        else:
            x_sensor, v_sensor, bias = self.pos, self.vel, self.bias

        # Likelihood function for ML Solvers
        def ell(pos_vel: npt.ArrayLike, **ell_kwargs):
            return self.log_likelihood_from_posvel(pos_vel=pos_vel,
                                                   x_sensor=x_sensor, v_sensor=v_sensor,
                                                   zeta=zeta, bias=bias, **ell_kwargs)

        # Call the util function
        x_est, likelihood, x_grid = ml_solver(ell=ell, search_space=search_space, **kwargs)

        return x_est, likelihood, x_grid

    def max_likelihood_uncertainty(self, zeta: npt.ArrayLike,
                                   source_search: SearchSpace,
                                   do_sensor_bias: bool, do_sensor_pos: bool, do_sensor_vel:bool,
                                   bias_search: SearchSpace=None, pos_search: SearchSpace=None,
                                   vel_search: SearchSpace=None, print_progress=False,
                                   **kwargs):
        """
        To perform a Max Likelihood Search with extra uncertainty parameters (e.g., sensor bias,
        sensor position, or sensor velocity), we must encapsulate those extra variables in a broader
        SearchSpace object

        :param zeta: Measurement vector
        :param source_search: Definition of the source position (or pos/vel) search space; SearchSpace instance
        :param do_sensor_bias: flag controlling whether this function will account for unknown measurement bias
        :param do_sensor_pos: flag controlling whether this function will account for sensor position errors
        :param do_sensor_vel: flag controlling whether this function will account for sensor velocity errors
        :param bias_search: SearchSpace for measurement bias search
        :param pos_search: SearchSpace for sensor position
        :param vel_search: SearchSpace for sensor velocity
        :param print_progress: Boolean flag, if true then progress updates and elapsed/remaining time will be printed to
                           the console. [default=False]
        :return x_est:  estimated source position (or position and velocity)
        :return likelihood: array of likelihood values at each position in the combined SearchSpace
        :return th_grid: grid of search positions
        :return parameter_est: dict containing the estimated bias, sensor position, and sensor velocity terms
        """

        # ToDo: Add support for a bias covariance term, sensor position error covariance, and sensor velocity error covariance

        # Check for missing info
        if do_sensor_pos: assert pos_search is not None, 'Missing argument ''pos_search''.'
        if do_sensor_vel: assert vel_search is not None, 'Missing argument ''vel_search''.'
        if do_sensor_bias: assert bias_search is not None, 'Missing argument ''bias_search''.'

        # Make sure at least one term is true; otherwise this is just ML
        if not do_sensor_bias and not do_sensor_pos and not do_sensor_vel:
            x_est, likelihood, x_grid = self.max_likelihood(zeta, search_space=source_search, **kwargs)
            return x_est, likelihood, x_grid, None

        # Double check that we actually want to search for sensor velocities...
        do_sensor_vel = do_sensor_vel and self.vel is not None

        # Build the Search Space
        # First, we get the center, epsilon, and search_size for the uncertainty terms
        # (sensor measurement bias, sensor positions, and sensor velocities)
        unc_search_space = self.make_uncertainty_search_space(source_search,
                                                              do_bias_search=do_sensor_bias,
                                                              do_pos_search=do_sensor_pos,
                                                              do_vel_search=do_sensor_vel,
                                                              bias_search=bias_search, pos_search=pos_search,
                                                              vel_search=vel_search)

        search_space = unc_search_space['combined_search']
        indices = unc_search_space['indices']
        # dict fields:
        # source_indices   -- source parameter indices
        # bias_indices     -- bias parameter indices
        # pos_indices      -- sensor position indices
        # vel_indices      -- sensor velocity indices

        def ell(th: npt.ArrayLike, **ell_kwargs):
            # Parse the parameter vector theta
            pos_vel = th[indices['source_indices']]
            x_source, v_source = self.parse_source_pos_vel(pos_vel, default_vel=self.vel)
            bias = th[indices['bias_indices']] if do_sensor_bias else self.bias
            x_sensor = np.reshape(th[indices['pos_indices']], self.pos.shape) if do_sensor_pos else self.pos
            v_sensor = np.reshape(th[indices['vel_indices']], self.vel.shape) if do_sensor_vel else self.vel

            return self.log_likelihood(zeta=zeta, x_source=x_source, v_source=v_source,
                                       x_sensor=x_sensor, v_sensor=v_sensor,
                                       bias=bias, **ell_kwargs)

        th_est, likelihood, th_grid = ml_solver(ell=ell, search_space=search_space,
                                                print_progress=True, **kwargs)

        # Parse the estimates
        x_est = th_est[indices['source_indices']]
        th_est = {'bias': th_est[indices['bias_indices']] if do_sensor_bias else None,
                  'pos': th_est[indices['pos_indices']] if do_sensor_pos else None,
                  'vel': th_est[indices['vel_indices']] if do_sensor_vel else None}

        return x_est, likelihood, th_grid, th_est


    def gradient_descent(self, zeta: npt.ArrayLike, x_init: npt.ArrayLike, cal_data: dict=None, **kwargs):
        """

        """
        # Perform sensor calibration
        if cal_data is not None:
            x_sensor, v_sensor, bias = self.sensor_calibration(**cal_data)
        else:
            x_sensor, v_sensor, bias = self.pos, None, self.bias

        # Make a function handle for the measurement difference (y)
        def y(pos_vel: npt.ArrayLike):
            return zeta - self.measurement_from_pos_vel(pos_vel=pos_vel,
                                                        x_sensor=x_sensor, v_sensor=v_sensor, bias=bias)

        def jacobian(pos_vel: npt.ArrayLike):
            return self.jacobian_from_posvel(pos_vel=pos_vel, x_sensor=x_sensor, v_sensor=v_sensor)

        return gd_solver(x_init=x_init, y=y, jacobian=jacobian, cov=self.cov, **kwargs)


    def least_square(self, zeta: npt.ArrayLike, x_init: npt.ArrayLike=None, cal_data: dict=None, **kwargs):

        # Perform sensor calibration
        if cal_data is not None:
            x_sensor, v_sensor, bias = self.sensor_calibration(**cal_data)
        else:
            x_sensor, v_sensor, bias = self.pos, None, self.bias

        # Initialize x_init
        if x_init is None:
            # Start at origin -- anything better we can do???
            x_init = np.zeros((self.num_dim, ))

        # Make a function handle for the measurement difference (y)
        def y(pos_vel: npt.ArrayLike):
            return zeta - self.measurement_from_pos_vel(pos_vel=pos_vel,
                                                        x_sensor=x_sensor, v_sensor=v_sensor, bias=bias)

        def jacobian(pos_vel: npt.ArrayLike):
            return self.jacobian_from_posvel(pos_vel=pos_vel, x_sensor=x_sensor, v_sensor=v_sensor)

        return ls_solver(x_init=x_init, zeta=y, jacobian=jacobian, cov=self.cov, **kwargs)

    def bestfix(self, zeta: npt.ArrayLike, search_space: SearchSpace, pdf_type=None, cal_data: dict=None):
        # Perform sensor calibration
        if cal_data is not None:
            x_sensor, v_sensor, bias = self.sensor_calibration(**cal_data)
        else:
            x_sensor, v_sensor, bias = self.pos, None, self.bias

        def m(pos_vel: npt.ArrayLike):
            return self.measurement_from_pos_vel(pos_vel=pos_vel, x_sensor=x_sensor, v_sensor=v_sensor, bias=bias)

        pdfs = make_pdfs(m, zeta, pdf_type, self.cov.cov)

        return bestfix_solver(pdfs, search_space)

    def sensor_calibration(self,
                           zeta_cal: npt.ArrayLike,
                           x_cal: npt.ArrayLike,
                           v_cal: npt.ArrayLike | None=None,
                           pos_search: SearchSpace=None,
                           vel_search: SearchSpace=None,
                           bias_search: SearchSpace=None,
                           do_pos_cal: bool=True,
                           do_vel_cal: bool=False,
                           do_bias_cal: bool=False):
        """
        This function attempts to calibrate sensor uncertainties given a series of measurements (zeta_cal)
        against a set of calibration emitters. Relies on the method log_likelihood to compute a Maximum Likelihood
        estimate for bias and sensor positions.

        If pos_search is defined, then a search will be done over sensor positions (centered on the nominal positions
        in self.pos).

        If bias_search is defined, then a search will be done over sensor measurement biases (centered on zero bias).

        Either pos_search or bias_search (or both) must be defined.

        :param zeta_cal: 2D array of calibration measurements (n_msmt, n_cal)
        :param x_cal: 2D array of calibration emitter locations (n_dim, n_cal)
        :param v_cal: 2D array of calibration emitter velocities (n_dim, n_cal)
        :param pos_search: optional dictionary with parameters for the ML search for sensor positions
        :param vel_search: optional dictionary with parameters for the ML search for sensor positions
        :param bias_search: optional dictionary with parameters for the ML search for measurement bias
        :param do_pos_cal: boolean flag (default True); if True calibration data will be used to estimate sensor
                           positions
        :param do_vel_cal: boolean flag (default False); if True calibration data will be used to estimate sensor
                           velocities
        :param do_bias_cal: boolean flag (default False); if True calibration data will be used to estimate sensor
                            measurement biases
        :return x_sensor_est: Estimated sensor positions (None if ignored)
        :return bias_est: Estimated measurement biases (None if ignored)
        """

        # TODO: Test
        # ================ Parse inputs =========================
        if not do_pos_cal and not do_vel_cal and not do_bias_cal:
            # No calibration called for
            return self.pos, self.vel, self.bias

        num_dim_cal, num_cal = safe_2d_shape(x_cal)
        num_msmt, num_cal2 = safe_2d_shape(zeta_cal)
        num_cov, _ = safe_2d_shape(self.cov.cov)

        # Check dimension agreement
        assert num_dim_cal == self.num_dim, "Disagreement in number of spatial dimensions between sensor positions and calibration emitter positions."
        assert num_cal == num_cal2, "Disagreement in number of calibration emitters between x_cal and zeta_cal."

        if v_cal is not None:
            num_dim_v, num_cal_v = safe_2d_shape(v_cal)
            assert num_dim_cal == num_dim_v and num_cal == num_cal_v, "Disagreement in size of calibration emitter velocities and sensor positions."
        else:
            v_cal = np.zeros_like(x_cal)  # assume zero velocity; simplified code later on

        # ==================== Initialize Search Space ========================
        pos_search = self.initialize_sensor_pos_search(pos_search, do_pos_search=do_pos_cal)
        vel_search = self.initialize_sensor_vel_search(vel_search, do_vel_search=do_vel_cal)
        bias_search = self.initialize_bias_search(bias_search, do_bias_search=do_bias_cal)

        # ==================== Log-Likelihood Wrapper Function ================
        # Accepts a 1D vector of measurement biases and a 1D vector of sensor positions

        def ell(b: npt.ArrayLike, x: npt.ArrayLike, v: npt.ArrayLike, **ell_kwargs):
            # Reshape the sensor position and velocity
            this_x_sensor = np.reshape(x, shape=self.pos.shape)
            this_v_sensor = np.reshape(v, newshape=[self.num_dim, -1]) if v is not None else None
            res = 0
            for this_zeta, this_x, this_v in zip(zeta_cal.T, x_cal.T, v_cal.T):
                this_ell = self.log_likelihood(x_sensor=this_x_sensor, v_sensor=this_v_sensor, zeta=this_zeta,
                                               x_source=this_x, v_source=this_v, bias=b, **ell_kwargs)
                res = res + this_ell
            return res

        x_sensor_est, v_sensor_est, bias_est = sensor_calibration(ell, pos_search, vel_search, bias_search)

        # Handle response shapes
        x_sensor_est = np.reshape(x_sensor_est, shape=(self.num_dim, -1)) if x_sensor_est is not None else None
        v_sensor_est = np.reshape(v_sensor_est, shape=(self.num_dim, -1)) if v_sensor_est is not None else None

        return x_sensor_est, v_sensor_est, bias_est

    def sensor_calibration_gd(self,
                              zeta_cal: npt.ArrayLike,
                              x_cal: npt.ArrayLike,
                              v_cal: npt.ArrayLike | None=None,
                              do_pos_cal: bool=True,
                              do_vel_cal: bool=False,
                              do_bias_cal: bool=False):
        """
        This function attempts to calibrate sensor uncertainties given a series of measurements (zeta_cal)
        against a set of calibration emitters. Relies on the method log_likelihood to compute a Maximum Likelihood
        estimate for bias and sensor positions.

        If pos_search is defined, then a search will be done over sensor positions (centered on the nominal positions
        in self.pos).

        If bias_search is defined, then a search will be done over sensor measurement biases (centered on zero bias).

        Either pos_search or bias_search (or both) must be defined.

        :param zeta_cal: 2D array of calibration measurements (n_msmt, n_cal)
        :param x_cal: 2D array of calibration emitter locations (n_dim, n_cal)
        :param v_cal: 2D array of calibration emitter velocities (n_dim, n_cal)
        :param do_pos_cal: boolean flag (default True); if True calibration data will be used to estimate sensor
                           positions
        :param do_vel_cal: boolean flag (default False); if True calibration data will be used to estimate sensor
                           velocities
        :param do_bias_cal: boolean flag (default False); if True calibration data will be used to estimate sensor
                            measurement biases
        :return x_sensor_est: Estimated sensor positions (None if ignored)
        :return bias_est: Estimated measurement biases (None if ignored)
        """

        # TODO: Test
        # ================ Parse inputs =========================
        if not do_pos_cal and not do_vel_cal and not do_bias_cal:
            # No calibration called for
            return self.pos, self.vel, self.bias

        num_dim_cal, num_cal = safe_2d_shape(x_cal)
        num_msmt, num_cal2 = safe_2d_shape(zeta_cal)
        num_cov, _ = safe_2d_shape(self.cov.cov)

        # Check dimension agreement
        assert num_dim_cal == self.num_dim, "Disagreement in number of spatial dimensions between sensor positions and calibration emitter positions."
        assert num_cal == num_cal2, "Disagreement in number of calibration emitters between x_cal and zeta_cal."

        if v_cal is not None:
            num_dim_v, num_cal_v = safe_2d_shape(v_cal)
            assert num_dim_cal == num_dim_v and num_cal == num_cal_v, "Disagreement in size of calibration emitter velocities and sensor positions."
        else:
            v_cal = np.zeros_like(x_cal)  # assume zero velocity; simplified code later on

        # ==================== Initialize Search Arguments ========================
        # TODO
        if do_bias_cal:
            num_bias = self.num_measurements
            bias_slice = np.s_[:self.num_bias]  # one for each measurement
        else:
            num_bias = 0
            bias_slice = None

        if do_pos_cal:
            num_pos = self.num_dim*self.num_sensors
            pos_slice = np.s_[num_bias:num_bias+num_pos]
        else:
            num_pos = 0
            pos_slice = None

        if do_vel_cal:
            num_vel = self.num_dim*self.num_sensors
            vel_slice = np.s_[num_bias+num_pos:num_bias+num_pos+num_vel]
        else:
            num_vel = 0
            vel_slice = None

        th_init = np.zeros(num_bias + num_pos + num_vel)
        if bias_slice is not None: th_init[bias_slice] = self.bias
        if pos_slice is not None: th_init[pos_slice] = self.pos.ravel()
        if vel_slice is not None and self.vel is not None: th_init[vel_slice] = self.vel.ravel()

        # ==================== Measurement Wrapper Function ================
        def y(th: npt.ArrayLike):
            # The theta vector contains both measurement biases and sensor position/velocity errors
            bias = th[bias_slice] if bias_slice is not None else None
            x_sensor = np.reshape(th[pos_slice], shape=(self.num_dim, self.num_sensors)) if pos_slice is not None else None
            v_sensor = np.reshape(th[vel_slice], shape=(self.num_dim, self.num_sensors)) if vel_slice is not None else None
            return np.ravel(zeta_cal - self.measurement(x_sensor=x_sensor, v_sensor=v_sensor, bias=bias, x_source=x_cal, v_source=v_cal))

        def jacobian(th: npt.ArrayLike):
            x_sensor = np.reshape(th[pos_slice], shape=(self.num_dim, self.num_sensors)) if pos_slice is not None else None
            v_sensor = np.reshape(th[vel_slice], shape=(self.num_dim, self.num_sensors)) if vel_slice is not None else None

            arrs = []
            if do_bias_cal:
                j_a = np.reshape(self.grad_bias(x_sensor=x_sensor, v_sensor=v_sensor, x_source=x_cal, v_source=v_cal),
                                 shape=(self.num_measurements, self.num_measurements*num_cal))
                arrs.append(j_a)
            if do_pos_cal:
                j_b = np.reshape(self.grad_sensor_pos(x_sensor=x_sensor, v_sensor=v_sensor, x_source=x_cal, v_source=v_cal),
                                 shape=(num_pos+num_vel, self.num_measurements*num_cal))
                arrs.append(j_b)
            return np.concatenate(arrs, axis=0)

        # Make a calibration measurement covariance matrix
        cov_cal = CovarianceMatrix.block_diagonal(*[self.cov for _ in range(num_cal)])

        th_est, _ = ls_solver(zeta=y, jacobian=jacobian, x_init=th_init, cov=cov_cal)

        # Handle response shapes
        bias_est = th_est[bias_slice] if bias_slice is not None else self.bias
        x_sensor_est = np.reshape(th_est[pos_slice],
                                  shape=(self.num_dim, self.num_sensors)) if pos_slice is not None else self.pos
        v_sensor_est = np.reshape(th_est[vel_slice],
                                  shape=(self.num_dim, -1)) if pos_slice is not None else self.pos

        return x_sensor_est, v_sensor_est, bias_est


    def make_uncertainty_search_space(self,
                                      source_search: SearchSpace,
                                      do_bias_search: bool,
                                      do_pos_search: bool,
                                      do_vel_search: bool,
                                      bias_search: SearchSpace=None,
                                      pos_search: SearchSpace=None,
                                      vel_search: SearchSpace=None):
        """
        Parse an uncertainty search space across sensor measurement biases, sensor positions, and sensor
        velocities.

        This will define a bias_search, pos_search, and vel_search, for each uncertainty component, respectively,
        as well as a combined search vector for all of them.

        :param source_search: Source pos (or pos/vel) search space
        :param do_bias_search: Boolean flag; if true then a search space will be defined for measurement biases
        :param do_pos_search: Boolean flag; if true then a search space will be defined for sensor positions
        :param do_vel_search: Boolean flag; if true, then a search space will be defined for sensor velocities
        :param bias_search: Optional initial bias search space
        :param pos_search: Optional initial position search space
        :param vel_search: Optional initial velocity search space
        :return search_space: dict with the following fields:
            combined_search SearchSpace object specifying the combined parameter search
            source_search   SearchSpace object specifying the source pos (or pos/vel) search
            bias_search     SearchSpace object specifying the sensor bias search component
            pos_search`     SearchSpace object specifying the sensor position search component
            vel_search      SearchSpace object specifying the sensor velocity search component
            indices         SearchSpace object specifying the indices for x_ctr, epsilon, and search_size, with fields:
                    num_source -- number of parameters in the source search
                    num_bias   -- number of parameters in the bias search
                    num_pos    -- number of parameters in the sensor position search
                    num_vel    -- number of parameters in the sensor velocity search
                    source_indices   -- source parameter indices
                    bias_indices     -- bias parameter indices
                    pos_indices      -- sensor position indices
                    vel_indices      -- sensor velocity indices
        """

        # === Initialize Component Search Spaces ========================
        bias_search = self.initialize_bias_search(bias_search, do_bias_search=do_bias_search)
        pos_search = self.initialize_sensor_pos_search(pos_search, do_pos_search=do_pos_search)
        vel_search = self.initialize_sensor_vel_search(vel_search, do_vel_search=do_vel_search)

        # === Make sure the Search Space is Broadcasted to common size (no scalars)
        # todo: implement as a SearchSpace method
        num_params = source_search.num_parameters
        default_arr = np.ones((num_params, ))
        if np.size(source_search.x_ctr) != num_params:
            source_search.x_ctr = source_search.x_ctr * default_arr
        if np.size(source_search.epsilon) != num_params:
            source_search.epsilon = source_search.epsilon * default_arr
        if np.size(source_search.points_per_dim) != num_params:
            source_search.points_per_dim = source_search.points_per_dim * default_arr

        # === Parse the components for a combined search space =========
        field_names = ['x_ctr', 'epsilon', 'points_per_dim']
        combined_search = {}
        for field_name in field_names:
            components = [np.broadcast_to(getattr(x, field_name), x.num_parameters) for x in (source_search, bias_search, pos_search, vel_search) if x is not None]
            if len(components) > 1:
                combined_search[field_name] = np.concatenate(components, axis=None)
            else:
                combined_search[field_name] = components[0]
        search_space = SearchSpace(**combined_search)  # pass combined search terms to constructor as kwargs

        # === Compute indices =========================================
        num_source = source_search.num_parameters
        num_bias = bias_search.num_parameters if bias_search is not None else 0
        num_pos = pos_search.num_parameters if pos_search is not None else 0
        num_vel = vel_search.num_parameters if vel_search is not None else 0

        source_indices = np.arange(num_source)
        bias_indices = np.arange(num_bias) + num_source
        pos_indices = np.arange(num_pos) + num_bias + num_source
        vel_indices = np.arange(num_vel) + num_pos + num_bias + num_source
        indices = {'num_source': num_source,
                   'num_bias': num_bias,
                   'num_pos': num_pos,
                   'num_vel': num_vel,
                   'source_indices': source_indices,
                   'bias_indices': bias_indices,
                   'pos_indices': pos_indices,
                   'vel_indices': vel_indices}

        return {'combined_search': search_space,
                'source_search': source_search,
                'bias_search': bias_search,
                'pos_search': pos_search,
                'vel_search': vel_search,
                'indices': indices}

    def initialize_bias_search(self, bias_search: SearchSpace=None, do_bias_search: bool=False):
        if not do_bias_search: return None

        if bias_search.x_ctr is None:
            # If there is a bias specified in the object; use it as the center of the search window. Otherwise,
            # assume zero bias is the center
            x_ctr = self.bias if self.bias is not None else np.zeros((self.num_measurements, ))
        else:
            x_ctr = bias_search.x_ctr

        if bias_search.epsilon is None:
            epsilon = self.default_bias_search_epsilon
        else:
            epsilon = bias_search.epsilon

        if bias_search.points_per_dim is None:
            points_per_dim = self.default_bias_search_size
        else:
            points_per_dim = bias_search.points_per_dim

        return SearchSpace(x_ctr=x_ctr, epsilon=epsilon, points_per_dim=points_per_dim)

    def initialize_sensor_pos_search(self, pos_search: SearchSpace=None, do_pos_search: bool=True):
        if not do_pos_search: return None

        if pos_search is None or pos_search.x_ctr is None:
            x_ctr = self.pos
        else:
            x_ctr = pos_search.x_ctr

        if pos_search is None or pos_search.epsilon is None:
            epsilon = self.default_sensor_pos_search_epsilon
        else:
            epsilon = pos_search.epsilon

        if pos_search is None or pos_search.points_per_dim is None:
            points_per_dim = self.default_sensor_pos_search_size
        else:
            points_per_dim = pos_search.points_per_dim

        return SearchSpace(x_ctr=x_ctr, epsilon=epsilon, points_per_dim=points_per_dim)

    def initialize_sensor_vel_search(self, vel_search: SearchSpace=None, do_vel_search: bool=False):
        if not do_vel_search: return None

        if vel_search is None or vel_search.x_ctr is None:
            x_ctr = self.vel if self.vel is not None else np.zeros_like(self.pos)
        else:
            x_ctr = vel_search.x_ctr

        if vel_search is None or vel_search.epsilon is None:
            epsilon = self.default_sensor_vel_search_epsilon
        else:
            epsilon = vel_search.epsilon

        if vel_search is None or vel_search.points_per_dim is None:
            points_per_dim = self.default_sensor_vel_search_size
        else:
            points_per_dim = vel_search.points_per_dim

        return SearchSpace(x_ctr=x_ctr, epsilon=epsilon, points_per_dim=points_per_dim)

    def parse_uncertainty_indices(self, do_source_vel: bool=False, do_bias: bool=False, do_sensor_pos: bool=False,
                                  do_sensor_vel: bool=False):
        """
        Uncertainty parameters take the form:
           [x_source, v_source, bias, x_sensor.ravel(), v_sensor.ravel()]

        This function computes the indices for each term.
        """

        # First, compute the number of each component and determine is source velocity is included
        num_source_pos = self.num_dim
        num_source_vel = self.num_dim if do_source_vel else 0
        num_source = num_source_pos + num_source_vel
        num_bias = self.num_measurements if do_bias else 0
        num_pos = np.size(self.pos) if do_sensor_pos else 0
        num_vel = np.size(self.pos) if do_sensor_vel else 0

        source_pos_ind = np.arange(num_source_pos)
        source_vel_ind = np.arange(num_source_vel)
        bias_ind = num_source + np.arange(num_bias)
        pos_ind = num_source + num_bias + np.arange(num_pos)
        vel_ind = num_source + num_bias + num_pos + np.arange(num_vel)

        indices = {'num_source_pos': num_source_pos,
                   'num_source_vel': num_source_vel,
                   'num_bias': num_bias,
                   'num_pos': num_pos,
                   'num_vel': num_vel,
                   'source_pos_indices': source_pos_ind,
                   'source_vel_indices': source_vel_ind,
                   'bias_indices': bias_ind,
                   'pos_indices': pos_ind,
                   'vel_indices': vel_ind}
        return indices

    # def get_uncertainty_search_space(self, do_source_vel=False, do_sensor_bias=False, do_sensor_pos=False,
    #                                  do_sensor_vel=False):
    #     """
    #     Define and return a dict describing the uncertainty search vector
    #     """
    #
    #     # Source Position Search
    #     if do_source_vel:
    #         # The search calls for both position and velocity estimates for the source
    #         num_source_indices = 2*self.num_dim
    #     else:
    #         # The search calls for just position estimates
    #         num_source_indices = self.num_dim
    #
    #     # Sensor Bias Search
    #     num_bias_indices = self.num_measurements if do_sensor_bias else 0
    #
    #     # Sensor Position Search
    #     num_pos_indices = np.size(self.pos) if do_sensor_pos else 0
    #
    #     # Velocity doesn't matter for DirectionFinding
    #     num_vel_indices = np.size(self.vel) if do_sensor_vel and self.vel is not None else 0
    #
    #     # Build the indices
    #     source_indices = np.arange(num_source_indices)
    #     bias_indices = num_source_indices + np.arange(num_bias_indices)
    #     pos_indices = num_source_indices + num_bias_indices + np.arange(num_pos_indices)
    #     vel_indices = num_source_indices + num_bias_indices + num_pos_indices + np.arange(num_vel_indices)
    #
    #     # Assemble the dict and return
    #     return {'source_idx': source_indices,
    #             'bias_idx': bias_indices,
    #             'sensor_pos_idx': pos_indices,
    #             'sensor_vel_idx': vel_indices,
    #             'num_source_idx': num_source_indices,
    #             'num_bias_idx': num_bias_indices,
    #             'num_pos_idx': num_pos_indices,
    #             'num_vel_idx': num_vel_indices}

    # ==================== Performance Methods ================
    # These methods define basic performance predictions
    def compute_crlb(self, x_source,
                     x_sensor: npt.ArrayLike | None=None,
                     v_source: npt.ArrayLike | None=None,
                     v_sensor: npt.ArrayLike | None=None, **kwargs):
        def this_jacobian(pos_vel):
            return self.jacobian_from_posvel(pos_vel=pos_vel, x_sensor=x_sensor, v_source=v_source, v_sensor=v_sensor)

        return compute_crlb_gaussian(x_source=x_source, jacobian=this_jacobian, cov=self.cov,
                                     **kwargs)

    # ==================== Helper Methods =====================
    def plot_sensors(self, scale: float=1, ax: Axes=None, **kwargs):
        if ax is not None:
            ax.scatter(self.pos[0]/scale, self.pos[1]/scale, **kwargs)
        else:
            plt.scatter(self.pos[0]/scale, self.pos[1]/scale, **kwargs)
        return

    def parse_source_pos_vel(self, pos_vel: npt.ArrayLike, default_vel: npt.ArrayLike):
        """
        Parse a possible position/velocity setting for the source.

        Compares the input pos_vel against the number of spatial dimensions in use.

        If pos_vel has length 2*self.num_dim, it is assumed to be both spatial and velocity inputs,
        and they are separated accordingly.

        If pos_vel has length self.num_dim, then the second input default_vel is used for the source
        velocity.

        If the length is any other value, an error is raised.

        :param pos_vel: candidate array of position or position/velocity values
        :param default_vel: velocity value to return if the input is position only
        :return pos:
        :return vel:
        """
        num_dim, _ = safe_2d_shape(pos_vel)
        if num_dim==self.num_dim:
            # Position only; return zero for velocity
            pos = pos_vel
            vel = default_vel
        elif num_dim==2*self.num_dim:
            # Position/Velocity
            pos = pos_vel[:self.num_dim]
            vel = pos_vel[self.num_dim:]
        else:
            raise ValueError("Unable to parse source position/velocity; unexpected number of spatial dimensions.")

        return pos, vel

class DifferencePSS(PassiveSurveillanceSystem, ABC):
    """
    The covariance matrix can be specified one of two ways.

    1. At a sensor-level (representing errors and correlations between individual sensor measurements)
    2. At a difference-level (representing errors and correlations between each difference calculation)

    Use do_resample=True to specify case 1, and do_resample=False (default) to specify case 2.

    When accessing the covariance matrix, the difference-level covariance will be returned. To access the sensor-level
    covariance matrix directly, reference the cov_raw parameter.
    """
    _ref_idx: npt.ArrayLike | str | None = None       # specification for sensor pairs to use in difference operation
                                                      # Accepted values are:
                                                      #   int                         Index of common reference sensor
                                                      #   (2, num_pairs) ndarray      List of sensor pairs
                                                      #   'full'                      Use all possible sensor pairs
    _cov_resample: CovarianceMatrix | None           # difference-level covariance matrix
    _cov_raw: CovarianceMatrix | None                # sensor-level covariance matrix
    _do_resample: bool = True

    parent = None

    def __init__(self, x: npt.ArrayLike,
                 cov: CovarianceMatrix | npt.ArrayLike | None,
                 ref_idx: str | npt.ArrayLike=None,
                 do_resample: bool = True,
                 **kwargs):
        (super().__init__(x, cov, **kwargs))
        self.ref_idx = ref_idx
        self.update_covariance_matrix(cov, do_resample)

    @property
    def cov_raw(self):
        return self._cov_raw

    @property
    def cov(self):
        # Check if the covariance matrix needs to be re-sampled.
        # This occurs if a new reference index is provided.
        if self._do_resample:
            self.resample()
        return self._cov_resample

    @cov.setter
    def cov(self, cov: CovarianceMatrix | npt.ArrayLike | None):
        self.update_covariance_matrix(cov)

    def update_covariance_matrix(self, cov: CovarianceMatrix | npt.ArrayLike | None,
                                 do_resample: bool = True):
        # Must be input as a raw covariance matrix; one per sensor. Set the
        # do_resample flag.
        if cov is None:
            # Setting to none is the same as deleting; let's call the deleter for simplicty
            del self.cov
        else:
            if not isinstance(cov, CovarianceMatrix):
                cov = CovarianceMatrix(cov)

            # Copy the covariance matrix locally
            if do_resample:
                self._cov_raw = cov.copy()
                self._cov_resample = None
                self._do_resample = True
            else:
                self._cov_raw = None
                self._cov_resample = cov.copy()
                self._do_resample = False

            # Tell the parent (e.g., Hybrid sensor) they need to resample
            if self.parent is not None: self.parent._do_resample = True

    @cov.deleter
    def cov(self):
        self._cov_raw = None
        self._cov_resample = None
        self._do_resample = True
        if self.parent is not None: self.parent._do_resample = True

    @property
    def ref_idx(self):
        return self._ref_idx

    @ref_idx.setter
    def ref_idx(self, idx: str | npt.ArrayLike | None):
        self._ref_idx = idx
        self._do_resample = True  # Reset the do_resample flag
        if self.parent is not None: self.parent._do_resample = True

    @ref_idx.deleter
    def ref_idx(self):
        self._ref_idx = None
        self._do_resample = True
        if self.parent is not None: self.parent._do_resample = True

    def resample(self):
        if self._do_resample:
            if self._cov_raw is None:
                raise ValueError("Cannot resample covariance matrix; no raw covariance matrix is defined.")
            self._cov_resample = self._cov_raw.resample(ref_idx=self.ref_idx)
            self._do_resample = False