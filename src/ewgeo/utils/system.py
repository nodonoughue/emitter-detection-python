from abc import ABC, abstractmethod
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from . import make_pdfs, SearchSpace, parse_reference_sensor, broadcast_backwards
from .covariance import CovarianceMatrix
from .perf import compute_crlb_gaussian
from .solvers import ml_solver, gd_solver, ls_solver, bestfix_solver

class PassiveSurveillanceSystem(ABC):
    _cov: CovarianceMatrix | None = None
    _pos: npt.ArrayLike
    _num_sensors: int
    _num_dim: int
    _num_measurements: int

    # Optional Fields
    bias: npt.ArrayLike | None = None              # User-defined sensor measurement biases
    _vel: npt.ArrayLike | None = None               # Sensor velocity; ignored if not defined
    _cov_pos: CovarianceMatrix | None = None    # Assumed sensor position error covariance
    _cov_vel: CovarianceMatrix | None = None    # Assumed sensor velocity error covariance
    _cov_bias: CovarianceMatrix | None = None   # Assumed bias covariance

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
        if np.ndim(x)<2: x = np.expand_dims(x, 1) # Add a second dimension if there isn't one
        self.pos = x

        # Populate optional fields
        self.cov = cov
        self.bias = bias
        self.cov_pos = cov_pos
        self.vel = vel

    @property
    def pos(self)-> npt.NDArray[np.float64]:
        return self._pos

    @pos.setter
    def pos(self, val: npt.ArrayLike):
        val = np.array(val)
        if val.ndim < 2:
            val = val[:, np.newaxis]
        elif val.ndim > 2:
            raise ValueError(f"Sensor pos must be a 2D array (num_dim, num_sensors); received {val.shape}.")
        self._pos = val
        self._num_dim = val.shape[0]
        self._num_sensors = val.shape[1]

    @property
    def vel(self)-> npt.NDArray[np.float64] | None:
        return self._vel

    @vel.setter
    def vel(self, vel: npt.ArrayLike):
        if vel is not None:
            self._vel = vel
        else:
            self._vel = np.zeros_like(self.pos)

    @vel.deleter
    def vel(self):
        self._vel = None

    @property
    def num_sensors(self)-> int:
        return self._num_sensors

    @property
    def num_dim(self)-> int:
        return self._num_dim

    @property
    def num_measurements(self)-> int:
        return self.num_sensors # default is one measurement per sensor; subclasses should override this

    @property
    def num_measurements_raw(self)-> int:
        # raw (before difference sampling) number of measurements; used for reference index error checking
        return self.num_measurements

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
    def cov_pos(self)-> CovarianceMatrix:
        if self._cov_pos is None:
            # Make it an identity with one row for each element in self.pos
            self._cov_pos = CovarianceMatrix(np.eye(np.size(self.pos)))

        return self._cov_pos

    @cov_pos.setter
    def cov_pos(self, value: CovarianceMatrix | npt.ArrayLike | None):
        if isinstance(value, CovarianceMatrix) or value is None:
            self._cov_pos = value
        else:
            self._cov_pos = CovarianceMatrix(value)
        return

    @property
    def cov_vel(self)-> CovarianceMatrix | None:
        if self._cov_vel is None:
            # Make it an identity with one row for each element in self.vel
            self._cov_vel = CovarianceMatrix(np.eye(np.size(self.vel)))
        return self._cov_vel

    @cov_vel.setter
    def cov_vel(self, value: CovarianceMatrix | npt.ArrayLike | None):
        if isinstance(value, CovarianceMatrix) or value is None:
            self._cov_vel = value
        else:
            self._cov_vel = CovarianceMatrix(value)
        return

    @property
    def cov_bias(self)-> CovarianceMatrix | None:
        return self._cov_bias

    @cov_bias.setter
    def cov_bias(self, value: CovarianceMatrix | npt.ArrayLike | None):
        if isinstance(value, CovarianceMatrix) or value is None:
            self._cov_bias = value
        else:
            self._cov_bias = CovarianceMatrix(value)
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
        if 'v_source' in kwargs:
            default_vel = kwargs['v_source']
        else:
            default_vel = np.zeros_like(pos_vel)

        pos, vel = self.parse_source_pos_vel(pos_vel, default_vel=default_vel)
        kwargs['v_source'] = vel
        kwargs['x_source'] = pos
        return self.measurement(**kwargs)

    def noisy_measurement(self, x_source: npt.ArrayLike, num_samples:int = 1,
                          cov: CovarianceMatrix | None=None, **kwargs)-> npt.NDArray:
        """
        Generate a set of noisy measurements. Will return a 3D matrix of shape (num_measurements, num_sources, num_samples),
        except that the latter two dimensions will be removed is their size is equal to 1.

        :param x_source: 1D or 2D array of source positions
        :param num_samples: Optional integer specifying the number of samples to generate
        :param cov: Optional CovarianceMatrix to use to generate noise; if not None, this overrides self.cov
        All other parameters are passed directly to self.measurement() as keyword arguments
        :return: 1D or 2D array of noisy measurements, depending on the shape of x_source and num_samples.
        """

        # Generate noise-free measurement
        shp = np.shape(x_source)
        num_sources = shp[1] if len(shp) > 1 else 1
        z = np.reshape(self.measurement(x_source, **kwargs), (self.num_measurements, num_sources))

        # Generate noise
        if cov is None:
            cov = self.cov
        noise = np.reshape(cov.sample(num_samples=num_samples*num_sources),
                           shape=(self.num_measurements, num_sources, num_samples))

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
        n_dim = np.shape(np.atleast_1d(pos_vel))[0]
        n_dim2 = np.shape(np.atleast_1d(j))[0]
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
        if 'v_source' in kwargs:
            default_vel = kwargs['v_source']
        else:
            default_vel = np.zeros_like(pos_vel)
        pos, vel = self.parse_source_pos_vel(pos_vel, default_vel=default_vel)
        kwargs['v_source'] = vel
        kwargs['x_source'] = pos
        return self.log_likelihood(**kwargs)

    def log_likelihood_uncertainty(self, x_source: npt.ArrayLike,
                                   zeta: npt.ArrayLike,
                                   v_source: npt.ArrayLike | None=None,
                                   x_sensor: npt.ArrayLike | None=None,
                                   v_sensor: npt.ArrayLike | None=None,
                                   bias: npt.ArrayLike | None=None,
                                   do_sensor_bias: bool=False,
                                   do_sensor_pos: bool=False,
                                   do_sensor_vel: bool=False,
                                   cov_pos: CovarianceMatrix | None=None,
                                   cov_vel: CovarianceMatrix | None=None,
                                   cov_bias: CovarianceMatrix | None=None,
                                   **kwargs):

        if not do_sensor_bias and not do_sensor_pos and not do_sensor_vel:
            # None of the uncertainty parameters are being called for
            return self.log_likelihood(x_source=x_source, v_source=v_source, zeta=zeta,
                                       x_sensor=x_sensor, v_sensor=v_sensor, bias=bias, **kwargs)

        if cov_pos is None:
            cov_pos = self.cov_pos
        if cov_vel is None:
            cov_vel = self.cov_vel
        if cov_bias is None:
            cov_bias = self.cov_bias

        # Generate the ideal measurements
        z = self.measurement(x_sensor=x_sensor, v_sensor=v_sensor, x_source=x_source, v_source=v_source, bias=bias)

        # Check dimensions to ensure error is computed correctly
        arrs, _ = broadcast_backwards([z, zeta], start_dim=0, do_broadcast=True)
        z, zeta = arrs

        # Evaluate the measurement error and scaled log likelihood
        err = zeta - z
        ell_x = - self.cov.solve_aca(err.T)

        if do_sensor_pos:
            x_0 = self.pos
            arrs, _ = broadcast_backwards([x_0, x_sensor], start_dim=0, do_broadcast=True)
            x_0, x_sensor = arrs

            err_pos = x_0 - x_sensor

            # Make sure err_pos is reshaped to have the first dimension be the same size as self.cov_pos
            aa, bb, *cc = np.shape(err_pos)
            # out_shp = [aa*bb]
            # out_shp.extend(cc)
            ell_pos = - np.reshape(cov_pos.solve_aca(np.reshape(err_pos, shape=(aa*bb, -1)).T), shape=cc)
        else:
            ell_pos = 0.

        if do_sensor_vel and (self.vel is not None or v_sensor is not None):
            # Use order=F on ravel to ensure that they are done in Fortran-style; as that's the
            # way the covariance matrices are assembled.
            vel_ref = self.vel if self.vel is not None else 0.
            vel_test = v_sensor if v_sensor is not None else 0.
            arrs, _ = broadcast_backwards([vel_ref, vel_test], start_dim=0, do_broadcast=True)
            vel_ref, vel_test = arrs

            err_vel = vel_ref - vel_test
            aa, bb, *cc = np.shape(err_vel)
            # out_shp = [aa * bb]
            # out_shp.extend(cc)
            ell_vel = - np.reshape(cov_vel.solve_aca(np.reshape(err_vel, shape=(aa*bb, -1)).T), shape=cc)
        else:
            ell_vel = 0.

        if do_sensor_bias and (self.bias is not None or bias is not None) and cov_bias is not None:
            # Use order=F on ravel to ensure that they are done in Fortran-style; as that's the
            # way the covariance matrices are assembled.
            bias_ref = self.bias if self.bias is not None else 0.
            bias_test = bias if bias is not None else 0.
            arrs, _ = broadcast_backwards([bias_ref, bias_test], start_dim=0, do_broadcast=True)
            bias_ref, bias_test = arrs
            err_bias = bias_ref - bias_test
            aa, *bb = np.shape(err_bias)

            ell_bias = - np.reshape(cov_bias.solve_aca(np.reshape(err_bias, shape=(aa, -1)).T), shape=bb)
        else:
            ell_bias = 0.

        # Add the likelihood components
        ell = ell_x + ell_pos + ell_vel + ell_bias

        return ell

    @abstractmethod
    def grad_source(self,
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

    @abstractmethod
    def grad_sensor_vel(self,
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
            if 'solver_type' not in cal_data: cal_data['solver_type'] = 'ml'
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
                                   **kwargs)-> tuple[npt.NDArray[np.float64],
                                                     npt.NDArray[np.float64],
                                                     tuple[npt.NDArray[np.float64], ...],
                                                     dict]:
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

        # Future enhancement: add MAP-style prior covariances for bias, sensor position, and sensor velocity,
        # analogous to gd_ls_uncertainty, so the grid search becomes a joint MAP estimate.

        # Check for missing info
        if do_sensor_pos: assert pos_search is not None, 'Missing argument ''pos_search''.'
        if do_sensor_vel: assert vel_search is not None, 'Missing argument ''vel_search''.'
        if do_sensor_bias: assert bias_search is not None, 'Missing argument ''bias_search''.'

        # Make sure at least one term is true; otherwise this is just ML
        if not do_sensor_bias and not do_sensor_pos and not do_sensor_vel:
            x_est, likelihood, x_grid = self.max_likelihood(zeta, search_space=source_search, **kwargs)
            return x_est, likelihood, x_grid, {}

        # Double check that we actually want to search for sensor velocities...
        do_sensor_vel = do_sensor_vel and self.vel is not None

        # Build the Search Space
        # First, we get the center, epsilon, and search_size for the uncertainty terms
        # (sensor measurement bias, sensor positions, and sensor velocities)
        unc_search_space = self.make_uncertainty_search_space(source_search,
                                                              do_bias_search=do_sensor_bias,
                                                              do_pos_search=do_sensor_pos,
                                                              do_vel_search=do_sensor_vel,
                                                              bias_search=bias_search,
                                                              pos_search=pos_search,
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
            x_source, v_source = self.parse_source_pos_vel(pos_vel, default_vel=np.zeros_like(pos_vel))
            bias = th[indices['bias_indices']] if do_sensor_bias else self.bias
            x_sensor = np.reshape(th[indices['pos_indices']], self.pos.shape) if do_sensor_pos else self.pos
            v_sensor = np.reshape(th[indices['vel_indices']], self.vel.shape) if do_sensor_vel else self.vel

            return self.log_likelihood_uncertainty(zeta=zeta, x_source=x_source, v_source=v_source,
                                                   x_sensor=x_sensor, v_sensor=v_sensor,
                                                   bias=bias, do_sensor_bias=do_sensor_bias,
                                                   do_sensor_pos=do_sensor_pos, do_sensor_vel=do_sensor_vel,
                                                   **ell_kwargs)

        th_est, likelihood, th_grid = ml_solver(ell=ell, search_space=search_space, **kwargs)

        # Parse the estimates
        x_est = th_est[indices['source_indices']]
        th_est = {'bias': th_est[indices['bias_indices']] if do_sensor_bias else None,
                  'pos': th_est[indices['pos_indices']] if do_sensor_pos else None,
                  'vel': th_est[indices['vel_indices']] if do_sensor_vel else None}

        return x_est, likelihood, th_grid, th_est

    def gd_ls_solver(self, zeta: npt.ArrayLike, x_init: npt.ArrayLike, do_gd: bool, cal_data: dict=None, **kwargs)->\
        tuple[npt.NDArray, npt.NDArray] | tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:

        # Perform sensor calibration
        if cal_data is not None:
            if 'solver_type' not in cal_data:
                if do_gd: cal_data['solver_type'] = 'gd'
                else: cal_data['solver_type'] = 'ls'
            x_sensor, v_sensor, bias = self.sensor_calibration(**cal_data)
        else:
            x_sensor, v_sensor, bias = self.pos, None, self.bias

        # Make a function handle for the measurement difference (y)
        def y(pos_vel: npt.ArrayLike):
            return zeta - self.measurement_from_pos_vel(pos_vel=pos_vel,
                                                        x_sensor=x_sensor, v_sensor=v_sensor, bias=bias)

        def jacobian(pos_vel: npt.ArrayLike):
            return self.jacobian_from_posvel(pos_vel=pos_vel, x_sensor=x_sensor, v_sensor=v_sensor)

        if do_gd:
            x, x_full = gd_solver(x_init=x_init, y=y, jacobian=jacobian, cov=self.cov, **kwargs)
        else:
            x, x_full = ls_solver(x_init=x_init, y=y, jacobian=jacobian, cov=self.cov, **kwargs)

        if cal_data is not None:
            return x, x_full, x_sensor, v_sensor, bias
        else:
            return x, x_full

    def gradient_descent(self, zeta, **kwargs)-> tuple[npt.NDArray, npt.NDArray] |\
                                           tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        return self.gd_ls_solver(zeta, do_gd=True, **kwargs)

    def least_square(self, zeta, **kwargs)-> tuple[npt.NDArray, npt.NDArray] |\
                                           tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        return self.gd_ls_solver(zeta, do_gd=False, **kwargs)

    def gd_ls_uncertainty(self, zeta: npt.ArrayLike, x_init: npt.ArrayLike, do_gd: bool,
                          do_sensor_pos: bool=False, x_sensor: npt.ArrayLike=None,
                          cov_pos: CovarianceMatrix=None,
                          do_sensor_vel: bool=False, v_sensor: npt.ArrayLike=None,
                          cov_vel: CovarianceMatrix=None,
                          do_sensor_bias: bool=False, bias: npt.ArrayLike=None,
                          cov_bias: CovarianceMatrix=None, **kwargs)->\
            tuple[npt.NDArray[np.float64], dict, npt.NDArray[np.float64]]:

        x_init = np.array(x_init)
        num_source_vars = x_init.size
        source_slice = np.s_[:num_source_vars]
        th_init = x_init[:]

        if do_sensor_bias:
            if bias is None:
                bias = self.bias if self.bias is not None else np.zeros(self.num_measurements_raw, )
            th_init = np.append(th_init, bias)
            num_bias = np.size(bias)
            bias_slice = np.s_[num_source_vars:num_source_vars+num_bias]
            if cov_bias is None:
                cov_bias = self.cov_bias
            if cov_bias is None:
                raise ValueError(
                    "do_sensor_bias=True requires a bias prior covariance to regularize the joint "
                    "position+bias estimation. Pass cov_bias= to this method or assign self.cov_bias "
                    "on the PSS object."
                )
            bias_nominal = np.array(bias)  # prior mean; residual is zero when estimate equals this
        else:
            num_bias = 0
            bias_slice = None

        if do_sensor_pos:
            num_pos = self.num_dim*self.num_sensors
            pos_slice = np.s_[num_source_vars+num_bias:num_source_vars+num_bias+num_pos]
            if x_sensor is None: x_sensor = self.pos.ravel()
            th_init = np.append(th_init, x_sensor)
            if cov_pos is None:
                cov_pos = self.cov_pos  # property has a default identity fallback
            pos_nominal = self.pos.ravel()  # prior mean = nominal sensor positions
        else:
            num_pos = 0
            pos_slice = None

        if do_sensor_vel:
            num_vel = self.num_dim*self.num_sensors
            vel_slice = np.s_[num_source_vars+num_bias+num_pos:num_source_vars+num_bias+num_pos+num_vel]
            if v_sensor is None: v_sensor = self.vel.ravel()
            th_init = np.append(th_init, v_sensor)
            if cov_vel is None:
                cov_vel = self.cov_vel  # property has a default identity fallback
            vel_nominal = self.vel.ravel()  # prior mean = nominal sensor velocities
        else:
            num_vel = 0
            vel_slice = None

        # Build the augmented block-diagonal covariance: measurement block + one prior block per
        # active nuisance parameter group. The prior blocks regularise the otherwise rank-deficient
        # normal equations that arise when n_params > n_measurements.
        active_covs = [self.cov]
        if do_sensor_bias: active_covs.append(cov_bias)
        if do_sensor_pos:  active_covs.append(cov_pos)
        if do_sensor_vel:  active_covs.append(cov_vel)
        cov_aug = CovarianceMatrix.block_diagonal(*active_covs) if len(active_covs) > 1 else self.cov

        # n_params is fixed once th_init is fully assembled
        n_params = th_init.size

        # ==================== Measurement Wrapper Function ================
        pos_shape = (self.num_dim, self.num_sensors)
        def y(th: npt.ArrayLike):
            # The theta vector contains source position, and optionally: measurement biases,
            # sensor positions, and sensor velocities.
            th = np.array(th)
            x_source, v_source = self.parse_source_pos_vel(th[source_slice], default_vel=np.zeros_like(x_init))
            this_bias = th[bias_slice] if bias_slice is not None else None
            this_x_sensor = np.reshape(th[pos_slice], shape=pos_shape) if pos_slice is not None else None
            this_v_sensor = np.reshape(th[vel_slice], shape=pos_shape) if vel_slice is not None else None
            parts = [np.ravel(zeta - self.measurement(x_sensor=this_x_sensor, v_sensor=this_v_sensor,
                                                       bias=this_bias, x_source=x_source, v_source=v_source))]
            # Prior pseudo-residuals: zero when the estimate equals the nominal/prior-mean value
            if do_sensor_bias: parts.append(bias_nominal - th[bias_slice])
            if do_sensor_pos:  parts.append(pos_nominal  - th[pos_slice])
            if do_sensor_vel:  parts.append(vel_nominal  - th[vel_slice])
            return np.concatenate(parts)

        def jacobian(th: npt.ArrayLike):
            th = np.array(th)
            pos_vel = th[source_slice]
            x_source, v_source = self.parse_source_pos_vel(pos_vel, default_vel=np.zeros_like(pos_vel))
            # this_bias = th[bias_slice] if bias_slice is not None else self.bias  # not used for jacobian
            this_x_sensor = np.reshape(th[pos_slice], shape=pos_shape) if pos_slice is not None else None
            this_v_sensor = np.reshape(th[vel_slice], shape=pos_shape) if vel_slice is not None else None

            j_source = self.jacobian(x_source=x_source, v_source=v_source,
                                     x_sensor=this_x_sensor, v_sensor=this_v_sensor)
            arrs = [j_source]
            if do_sensor_bias:
                j_a = self.grad_bias(x_sensor=this_x_sensor, v_sensor=this_v_sensor,
                                     x_source=x_source, v_source=v_source)
                arrs.append(j_a)
            if do_sensor_pos:
                j_b = self.grad_sensor_pos(x_sensor=this_x_sensor, v_sensor=this_v_sensor,
                                           x_source=x_source, v_source=v_source)
                arrs.append(j_b)
            if do_sensor_vel:
                j_c = self.grad_sensor_vel(x_sensor=this_x_sensor, v_sensor=this_v_sensor,
                                           x_source=x_source, v_source=v_source)
                arrs.append(j_c)

            # Measurement block: rows = parameters, cols = measurements, shape (n_params, n_meas)
            j_meas = np.concatenate(arrs, axis=0)

            # Prior blocks: identity columns for each active nuisance group.
            # Each block has shape (n_params, n_group); the identity appears at the rows
            # corresponding to that group's slice in the parameter vector.
            prior_cols = []
            if do_sensor_bias:
                col = np.zeros((n_params, num_bias))
                col[bias_slice, :] = np.eye(num_bias)
                prior_cols.append(col)
            if do_sensor_pos:
                col = np.zeros((n_params, num_pos))
                col[pos_slice, :] = np.eye(num_pos)
                prior_cols.append(col)
            if do_sensor_vel:
                col = np.zeros((n_params, num_vel))
                col[vel_slice, :] = np.eye(num_vel)
                prior_cols.append(col)

            if prior_cols:
                return np.concatenate([j_meas] + prior_cols, axis=1)
            return j_meas

        if do_gd:
            result = gd_solver(x_init=th_init, y=y, jacobian=jacobian, cov=cov_aug, **kwargs)
        else:
            result = ls_solver(x_init=th_init, y=y, jacobian=jacobian, cov=cov_aug, **kwargs)

        th_est = result[0]
        th_est_full = result[1]

        # Parse the estimates
        x_est = th_est[source_slice]
        th_est = {'bias': th_est[bias_slice] if do_sensor_bias else None,
                  'pos': th_est[pos_slice] if do_sensor_pos else None,
                  'vel': th_est[vel_slice] if do_sensor_vel else None}

        return x_est, th_est, th_est_full

    def measurement_gradient(self, th: npt.ArrayLike):
        """
        Wrapper for the Jacobian function. Returns the transpose, which can be used in Kalman-style
        tracking equations.
        """
        return self.jacobian(th).T

    def gradient_descent_uncertainty(self, **kwargs)-> tuple[npt.NDArray[np.float64], dict, npt.NDArray[np.float64]]:
        return self.gd_ls_uncertainty(do_gd=True, **kwargs)

    def least_square_uncertainty(self, **kwargs)-> tuple[npt.NDArray[np.float64], dict, npt.NDArray[np.float64]]:
        return self.gd_ls_uncertainty(do_gd=False, **kwargs)

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
                           solver_type: str='ml',
                           do_pos_cal: bool = False,
                           do_vel_cal: bool = False,
                           do_bias_cal: bool = False,
                           **cal_data):
        """
        This function attempts to calibrate sensor uncertainties given a series of measurements (zeta_cal)
        against a set of calibration emitters. Relies on the method log_likelihood to compute a Maximum Likelihood
        estimate for bias and sensor positions.

        If pos_search is defined, then a search will be done over sensor positions (centered on the nominal positions
        in self.pos).

        If bias_search is defined, then a search will be done over sensor measurement biases (centered on zero bias).

        Either pos_search or bias_search (or both) must be defined.

        :param solver_type: String indicating which solver type to use. Options are: 'ml', 'gd', 'ls'.
        :param do_pos_cal: Boolean (default=False), set to True to force calibration to consider sensor position errors
        :param do_vel_cal: Boolean (default=False), set to True to force calibration to consider sensor velocity errors
        :param do_bias_cal: Boolean (default=False), set to True to force calibration to consider sensor bias
        :param cal_data: Dictionary containing solver-specific calibration fields.
        :return x_sensor_est: Estimated sensor positions (None if ignored)
        :return bias_est: Estimated measurement biases (None if ignored)
        """

        # ==== Parse Inputs ====
        if not do_pos_cal and not do_vel_cal and not do_bias_cal:
            # No calibration called for
            return self.pos, self.vel, self.bias


        # ==== Call the Desired Calibration Approach ====
        if solver_type.lower() == 'ml':
            # Remove any gd/ls-specific cal parameters
            gd_ls_fields = ['x_sensor', 'v_sensor', 'bias', 'epsilon']
            [cal_data.pop(f, None) for f in gd_ls_fields]
            return self.sensor_calibration_ml(do_bias_cal=do_bias_cal, do_pos_cal=do_pos_cal, do_vel_cal=do_vel_cal,
                                              **cal_data)
        elif solver_type.lower() in ['gd', 'ls']:
            # Remove any ml-specific cal parameters
            ml_fields = ['pos_search', 'vel_search', 'bias_search', 'cov_pos', 'cov_bias', 'cov_vel', 'source_search']
            [cal_data.pop(f, None) for f in ml_fields if f]
            do_gd = solver_type.lower() == 'gd'
            return self.sensor_calibration_gd_ls(do_gd=do_gd,
                                                 do_bias_cal=do_bias_cal, do_pos_cal=do_pos_cal, do_vel_cal=do_vel_cal,
                                                 **cal_data)
        else:
            raise ValueError(f"Sensor calibration type '{solver_type}' is not supported. ")

    def sensor_calibration_ml(self,
                              zeta_cal: npt.ArrayLike,
                              x_cal: npt.ArrayLike,
                              v_cal: npt.ArrayLike | None=None,
                              pos_search: SearchSpace | None = None,
                              vel_search: SearchSpace | None = None,
                              bias_search: SearchSpace | None = None,
                              cov_pos: CovarianceMatrix | None = None,
                              cov_vel: CovarianceMatrix | None = None,
                              cov_bias: CovarianceMatrix | None = None,
                              do_pos_cal: bool = True,
                              do_vel_cal: bool = False,
                              do_bias_cal: bool = False,
                              **kwargs):
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
        :param cov_pos: CovarianceMatrix defining sensor position errors
        :param cov_vel: CovarianceMatrix defining sensor velocity errors
        :param cov_bias: CovarianceMatrix defining sensor bias
        :param do_pos_cal: boolean flag (default True); if True calibration data will be used to estimate sensor
                           positions
        :param do_vel_cal: boolean flag (default False); if True calibration data will be used to estimate sensor
                           velocities
        :param do_bias_cal: boolean flag (default False); if True calibration data will be used to estimate sensor
                            measurement biases
        :return x_sensor_est: Estimated sensor positions (None if ignored)
        :return bias_est: Estimated measurement biases (None if ignored)
        """

        # ================ Parse inputs =========================


        shp = np.shape(x_cal)
        num_dim_cal = shp[0] if len(shp) > 0 else 1
        num_cal = shp[1] if len(shp) > 1 else 1
        if np.ndim(zeta_cal) < 2:
            # num_msmt = np.size(zeta_cal)
            num_cal2 = 1
            cal_batch_shape = ()
        else:
            _, num_cal2, *cal_batch_shape = np.shape(zeta_cal)
        # num_cov = self.cov.size

        # Check dimension agreement
        assert num_dim_cal == self.num_dim, "Disagreement in number of spatial dimensions between sensor positions and calibration emitter positions."
        assert num_cal == num_cal2, "Disagreement in number of calibration emitters between x_cal and zeta_cal."

        if v_cal is not None:
            num_dim_v, num_cal_v, *_ = np.shape(v_cal)
            assert num_dim_cal == num_dim_v and num_cal == num_cal_v, "Disagreement in size of calibration emitter velocities and sensor positions."
        else:
            v_cal = np.zeros_like(x_cal)  # assume zero velocity; simplified code later on

        # ==================== Initialize Search Spaces =====================
        bias_search = self.initialize_bias_search(bias_search, do_bias_search=do_bias_cal)
        pos_search = self.initialize_sensor_pos_search(pos_search, do_pos_search=do_pos_cal)
        vel_search = self.initialize_sensor_vel_search(vel_search, do_vel_search=do_vel_cal)

        # ==================== Determine Initial Sensor Pos/Vel Assumption ===================================
        if do_pos_cal and pos_search is not None:
            x_sensor = np.reshape(pos_search.x_ctr, shape=np.shape(self.pos))
        else:
            # Don't search for new sensor positions, just use what we have
            x_sensor = self.pos

        if do_vel_cal and vel_search is not None:
            v_sensor = np.reshape(vel_search.x_ctr, shape=np.shape(self.vel))
        else:
            v_sensor = np.zeros_like(x_sensor)

        x_sensor = np.array(x_sensor)
        v_sensor = np.array(v_sensor)
        zeta_cal = np.array(zeta_cal)

        bias = self.bias
        # ==================== Measurement Bias Search ========================
        if do_bias_cal:
            if bias_search is None:
                raise ValueError("Sensor calibration error. If do_bias_cal is True, then bias_search must be defined as a SearchSpace object.")

            bias_search, zoom_per_level, num_levels = bias_search.setup_mosaic()

            def ell_bias(b: npt.ArrayLike, **ell_kwargs)-> npt.NDArray:
                # shape ( num_cal, num_search_positions)
                ell = self.log_likelihood_uncertainty(x_sensor=x_sensor, v_sensor=v_sensor, bias=b,
                                                       x_source=x_cal, v_source=v_cal, zeta=zeta_cal,
                                                       cov_pos=cov_pos, cov_vel=cov_vel, cov_bias=cov_bias,
                                                       **ell_kwargs)
                return np.sum(ell, axis=0)

            bias_est, _, _ = ml_solver(ell=ell_bias, search_space=bias_search,
                                       zoom_per_level=zoom_per_level, num_levels=num_levels, **kwargs)
            bias = bias_est

        if do_pos_cal:
            if pos_search is None:
                raise ValueError("Sensor calibration error. If do_pos_cal is True, then pos_search must be defined as a SearchSpace object.")

            x_shp = np.shape(x_sensor)
            # num_pos = np.size(x_sensor)

            pos_search, zoom_per_level, num_levels = pos_search.setup_mosaic()

            def ell_pos(x: npt.ArrayLike, **ell_kwargs)-> npt.NDArray:
                x = np.array(x)
                xx = np.reshape(x, list(x_shp)+[-1])
                ell = self.log_likelihood_uncertainty(x_sensor=xx, v_sensor=v_sensor, bias=bias,
                                                       x_source=x_cal, v_source=v_cal, zeta=zeta_cal,
                                                       cov_pos=cov_pos, cov_vel=cov_vel, cov_bias=cov_bias,
                                                       **ell_kwargs)
                return np.sum(ell, axis=0)

            x_sensor_est, _, _ = ml_solver(ell=ell_pos, search_space=pos_search,
                                           zoom_per_level=zoom_per_level, num_levels=num_levels, **kwargs)
            x_sensor = np.reshape(x_sensor_est, shape=x_shp)

        if do_vel_cal:
            if vel_search is None:
                raise ValueError("Sensor calibration error. If do_vel_cal is True, then vel_search must be defined as a SearchSpace object.")

            vel_search, zoom_per_level, num_levels = vel_search.setup_mosaic()

            num_search_pts = np.prod(vel_search.points_per_dim)
            v_shp = list(np.shape(v_sensor))
            v_shp_3d = v_shp+[-1]+[1]*len(cal_batch_shape)
            # num_vel = np.size(v_sensor)
            # ell_shp = [num_cal, num_search_pts] + cal_batch_shape
            def ell_vel(v: npt.ArrayLike, **ell_kwargs) -> npt.NDArray:
                vv = np.reshape(v, v_shp_3d)
                ell = self.log_likelihood_uncertainty(x_sensor=x_sensor, v_sensor=vv, bias=bias,
                                                       x_source=x_cal, v_source=v_cal, zeta=zeta_cal[:, :, np.newaxis],
                                                       cov_pos=cov_pos, cov_vel=cov_vel, cov_bias=cov_bias,
                                                       **ell_kwargs)
                # shape: (num_cal, num_search_pts, *cal_batch_shape)
                return np.sum(np.reshape(np.moveaxis(ell, source=1, destination=0), (num_search_pts, -1)), axis=1)

            v_sensor_est, _, _ = ml_solver(ell=ell_vel, search_space=vel_search,
                                           zoom_per_level=zoom_per_level, num_levels=num_levels, **kwargs)
            v_sensor = np.reshape(v_sensor_est, shape=v_shp)

        return x_sensor, v_sensor, bias

    def sensor_calibration_gd_ls(self,
                                 zeta_cal: npt.ArrayLike, x_cal: npt.ArrayLike, v_cal: npt.ArrayLike | None=None,
                                 x_sensor: npt.ArrayLike = None, v_sensor: npt.ArrayLike = None,
                                 bias: npt.ArrayLike = None,
                                 do_pos_cal: bool=True, do_vel_cal: bool=False, do_bias_cal: bool=False,
                                 do_gd: bool=True,
                                 **gd_kwargs)-> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
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
        :param x_sensor: Optional starting sensor positions (otherwise self.pos is used)
        :param v_sensor: Optional starting sensor velocities (otherwise self.vel is used)
        :param bias: Optional starting sensor bias (otherwise self.bias is used)
        :param do_pos_cal: boolean flag (default=True); if True calibration data will be used to estimate sensor
                           positions
        :param do_vel_cal: boolean flag (default=False); if True calibration data will be used to estimate sensor
                           velocities
        :param do_bias_cal: boolean flag (default=False); if True calibration data will be used to estimate sensor
                            measurement biases
        :param do_gd: Optional boolean (default=True) specifying whether GD is to be used (true) or LS to solver for
                      calibration results.
        :return x_sensor_est: Estimated sensor positions (None if ignored)
        :return v_sensor_est: Estimated sensor velocities (None if ignored)
        :return bias_est: Estimated measurement biases (None if ignored)
        """

        # ================ Parse inputs =========================
        if bias is None: bias = self.bias
        if x_sensor is None: x_sensor = self.pos
        if v_sensor is None: v_sensor = self.vel

        if not do_pos_cal and not do_vel_cal and not do_bias_cal:
            # No calibration called for
            return x_sensor, v_sensor, bias

        num_dim_cal, num_cal, *_ = np.shape(x_cal)
        num_msmt, num_cal2, *_ = np.shape(zeta_cal)
        # num_cov = self.cov.size

        # Check dimension agreement
        assert num_dim_cal == self.num_dim, "Disagreement in number of spatial dimensions between sensor positions and calibration emitter positions."
        assert num_cal == num_cal2, "Disagreement in number of calibration emitters between x_cal and zeta_cal."

        if v_cal is not None:
            num_dim_v, num_cal_v = np.shape(v_cal)
            assert num_dim_cal == num_dim_v and num_cal == num_cal_v, "Disagreement in size of calibration emitter velocities and sensor positions."
        else:
            v_cal = np.zeros_like(x_cal)  # assume zero velocity; simplified code later on

        # Make a calibration measurement covariance matrix
        cov_cal = CovarianceMatrix.block_diagonal(*[self.cov for _ in range(num_cal)])

        # ==================== Solver Wrapper =================================
        # Both GD and LS take the same arguments, so we'll just make a wrapper that calls the desired one.
        def solver(**solver_kwargs)-> tuple[npt.NDArray, npt.NDArray]:
            if do_gd:
                return gd_solver(**solver_kwargs)
            else:
                return ls_solver(**solver_kwargs)

        # ==================== Measurement Bias Search ========================
        if do_bias_cal:
            def y_bias(b: npt.ArrayLike)-> npt.NDArray:
                return np.ravel(zeta_cal - self.measurement(x_sensor=x_sensor, v_sensor=v_sensor, bias=b,
                                                            x_source=x_cal, v_source=v_cal), order='F')

            def jacobian_bias(_)-> npt.NDArray:
                return np.reshape(self.grad_bias(x_sensor=x_sensor, v_sensor=v_sensor, x_source=x_cal, v_source=v_cal),
                                  shape=(self.num_measurements, self.num_measurements*num_cal), order='F')


            bias_est, _ = solver(y=y_bias, jacobian=jacobian_bias, x_init=bias, cov=cov_cal, **gd_kwargs)
            bias = bias_est

        # ==================== Sensor Position and Velocity Search ========================
        if do_pos_cal or do_vel_cal:
            x_shp = np.shape(x_sensor)
            x_shp_rev = np.shape(x_sensor.T)
            v_shp = np.shape(v_sensor)
            v_shp_rev = np.shape(v_sensor.T)
            num_pos = np.size(x_sensor)
            num_vel = np.size(v_sensor)

            def y_posvel(pos_vel: npt.ArrayLike)-> npt.NDArray:
                pos_vel = np.array(pos_vel)
                xx = np.reshape(pos_vel[:num_pos], shape=x_shp_rev).T
                vv = np.reshape(pos_vel[num_pos:], shape=v_shp_rev).T

                # shape: (self.num_measurements, num_cal)
                z = self.measurement(x_sensor=xx, v_sensor=vv, bias=bias, x_source=x_cal, v_source=v_cal)
                arrs, _ = broadcast_backwards([z, zeta_cal], start_dim=0, do_broadcast=True)
                z, this_zeta_cal = arrs

                # shape: (self.num_measurements * num_cal, num_cal_repetitions)
                return np.reshape(this_zeta_cal - z, shape=(num_cal*num_msmt, -1), order='F')

            def jacobian_posvel(pos_vel: npt.ArrayLike)-> npt.NDArray:
                """
                Return the jacobian (gradient with respect to sensor position and velocity) of each measurement.
                If do_pos_cal is False, then we set those rows to zero to ensure no change in sensor position is made.
                If do_vel_cal is False, then we set those columns to zero to ensure no change in sensor velocity is made.
                """
                pos_vel = np.array(pos_vel)
                xx = np.reshape(pos_vel[:num_pos], shape=x_shp_rev).T
                vv = np.reshape(pos_vel[num_pos:], shape=v_shp_rev).T

                j = np.zeros((num_pos + num_vel, self.num_measurements*num_cal))

                if do_pos_cal:
                    j_p = np.reshape(self.grad_sensor_pos(x_sensor=xx, v_sensor=vv,
                                                          x_source=x_cal, v_source=v_cal),
                                     shape=(num_pos, self.num_measurements*num_cal), order='F')
                    j[:num_pos] = j_p
                if do_vel_cal:
                    j_v = np.reshape(self.grad_sensor_vel(x_sensor=xx, v_sensor=vv,
                                                          x_source=x_cal, v_source=v_cal),
                                     shape=(num_vel, self.num_measurements * num_cal), order='F')
                    j[num_pos:] = j_v

                return j

            posvel_est, _ = solver(y=y_posvel, jacobian=jacobian_posvel,
                                     x_init=np.concatenate((x_sensor.T.ravel(), v_sensor.T.ravel()), axis=0),
                                     cov=cov_cal, **gd_kwargs)
            if do_pos_cal:
                x_sensor = np.reshape(posvel_est[:num_pos], x_shp_rev).T
            if do_vel_cal:
                v_sensor = np.reshape(posvel_est[num_pos:], v_shp_rev).T

        return x_sensor, v_sensor, bias


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

        # List of all SearchSpace objects to concatenate
        search_arr = [x for x in [source_search, bias_search, pos_search, vel_search] if x is not None]
        if len(search_arr) > 1:
            # === Parse the components for a combined search space =========
            field_names = ['x_ctr', 'epsilon', 'points_per_dim']
            combined_search = {}
            for field_name in field_names:
                components = np.concatenate([np.ravel(getattr(x, field_name)) for x in search_arr], axis=None)
                combined_search[field_name] = components
            search_space = SearchSpace(**combined_search)  # pass combined search terms to constructor as kwargs
        else:
            # the combined search space is simply the only defined one
            search_space = search_arr[0]

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

    def initialize_bias_search(self, bias_search: SearchSpace=None, do_bias_search: bool=False)-> SearchSpace | None:
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

    def initialize_sensor_pos_search(self, pos_search: SearchSpace=None, do_pos_search: bool=True)-> SearchSpace | None:
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

    def initialize_sensor_vel_search(self, vel_search: SearchSpace=None, do_vel_search: bool=False)->SearchSpace | None:
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

    # ==================== Performance Methods ================
    # These methods define basic performance predictions
    def compute_crlb(self, x_source,
                     x_sensor: npt.ArrayLike | None=None,
                     v_source: npt.ArrayLike | None=None,
                     v_sensor: npt.ArrayLike | None=None, **kwargs)-> CovarianceMatrix | list[CovarianceMatrix]:
        def this_jacobian(pos_vel):
            return self.jacobian_from_posvel(pos_vel=pos_vel, x_sensor=x_sensor, v_source=v_source, v_sensor=v_sensor)

        if 'cov' not in kwargs.keys():
            # If the user didn't manually specify a covariance matrix, use this object's current covariance matrix
            # as the default.
            kwargs['cov'] = self.cov

        return compute_crlb_gaussian(x_source=x_source, jacobian=this_jacobian,
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
        num_dim = np.shape(pos_vel)[0] if np.ndim(pos_vel) > 0 else 0
        pos_vel = np.array(pos_vel)
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
    _ref_idx_vec: npt.NDArray[np.int64] | None = None
    _test_idx_vec: npt.NDArray[np.int64] | None = None

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
    def num_measurements(self)-> int:
        return self._num_measurements

    @property
    def cov_raw(self)-> CovarianceMatrix:
        return self._cov_raw

    @property
    def cov(self)-> CovarianceMatrix:
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
    def num_measurements_raw(self)-> int:
        # for difference PSS systems, the raw (non-difference) number of measurements is one per sensor
        return self.num_sensors

    @property
    def ref_idx(self):
        return self._ref_idx

    @ref_idx.setter
    def ref_idx(self, idx: str | npt.ArrayLike | None):
        # Generate the test/ref index vectors
        self._test_idx_vec, self._ref_idx_vec = parse_reference_sensor(idx, num_sensors=self.num_measurements_raw)

        # Store their combination as the pre-processed reference index
        self._ref_idx = np.array([self._test_idx_vec, self._ref_idx_vec])
        self._num_measurements = len(self._test_idx_vec)
        self._do_resample = True  # Reset the do_resample flag
        if self.parent is not None: self.parent._do_resample = True

    @ref_idx.deleter
    def ref_idx(self):
        self._ref_idx = None
        self._test_idx_vec = None
        self._ref_idx_vec = None
        self._num_measurements = 0

        self._do_resample = True
        if self.parent is not None: self.parent._do_resample = True

    @property
    def test_idx_vec(self):
        return self._test_idx_vec

    @property
    def ref_idx_vec(self):
        return self._ref_idx_vec

    def resample(self):
        if self._do_resample:
            if self._cov_raw is None:
                raise ValueError("Cannot resample covariance matrix; no raw covariance matrix is defined.")
            self._cov_resample = self._cov_raw.resample(ref_idx=self.ref_idx)
            self._do_resample = False