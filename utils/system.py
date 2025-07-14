from abc import ABC, abstractmethod
import numpy as np
import utils
from utils.covariance import CovarianceMatrix
from utils import SearchSpace
import matplotlib.pyplot as plt


class PassiveSurveillanceSystem(ABC):
    cov: CovarianceMatrix or None = None
    pos: np.ndarray
    num_sensors: int
    num_dim: int
    num_measurements: int

    # Optional Fields
    bias: np.ndarray or None = None              # User-defined sensor measurement biases
    vel: np.ndarray or None = None               # Sensor velocity; ignored if not defined
    cov_pos: CovarianceMatrix or None = None     # Assumed sensor position error covariance

    # Default Values
    default_bias_search_epsilon = 0
    default_bias_search_size = 1
    default_sensor_pos_search_epsilon = 2.5
    default_sensor_pos_search_size = 25
    default_sensor_vel_search_epsilon = 1
    default_sensor_vel_search_size = 1  # By default, we can't search across sensor velocity

    def __init__(self, x: np.ndarray, cov: CovarianceMatrix or None, bias=None, cov_pos=None, vel=None):
        if len(np.shape(x))<2: x = np.expand_dims(x, 1) # Add a second dimension, if there isn't one
        self.pos = x
        self.cov = cov

        num_dim, num_sensors = utils.safe_2d_shape(x)
        self.num_sensors = num_sensors
        self.num_dim = num_dim
        self.num_measurements = self.cov.cov.shape[0]

        # Populate optional fields
        self.bias = bias
        self.cov_pos = cov_pos
        self.vel = vel

    # ==================== Model Methods ===================
    # These methods define the sensor measurement model, and must be implemented.
    @abstractmethod
    def measurement(self, x_source, x_sensor=None, bias=None, v_sensor=None, v_source=None):
        pass

    @abstractmethod
    def jacobian(self, x_source, v_source=None, x_sensor=None, v_sensor=None):
        pass

    @abstractmethod
    def jacobian_uncertainty(self, x_source, **kwargs):
        pass

    @abstractmethod
    def log_likelihood(self, x_source, zeta, x_sensor=None, bias=None, v_sensor=None, v_source=None):
        pass

    def log_likelihood_uncertainty(self, zeta, theta, do_source_vel=False, do_sensor_bias=False, do_sensor_pos=False,
                                   do_sensor_vel=False, **kwargs):
        if not do_sensor_bias and not do_sensor_pos and not do_sensor_vel:
            # None of the uncertainty parameters are being called for; theta is just x_source
            return self.log_likelihood(theta, zeta, **kwargs)

        # Parse the uncertainty vector
        indices = self.parse_uncertainty_indices(theta, do_bias=do_sensor_bias, do_sensor_pos=do_sensor_pos,
                                                 do_sensor_vel=do_sensor_vel)

        # Write a parsing function
        def _parse_uncertainty_vector(th):
            x = th[indices['source_pos_indices']]
            v = th[indices['source_vel_indices']] if do_source_vel else np.zeros_like(x)
            b = th[indices['bias_indices']] if do_sensor_bias else self.bias
            xs = np.reshape(th[indices['pos_indices']], self.pos.shape) if do_sensor_pos else self.pos
            vs = np.reshape(th[indices['vel_indices']], self.pos.shape) if do_sensor_vel else self.vel
            return x, v, b, xs, vs

        num_parameters, num_source_pos = utils.safe_2d_shape(theta)
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

    # ==================== Solver Methods ===================
    # These methods define the basic solver methods, and associated utilities, and must be implemented.
    @abstractmethod
    def max_likelihood(self, zeta, source_search: SearchSpace, bias, cal_data, **kwargs):
        pass

    def max_likelihood_uncertainty(self, zeta, source_search: SearchSpace,
                                   do_sensor_bias: bool, do_sensor_pos: bool, do_sensor_vel:bool,
                                   bias_search: SearchSpace, pos_search: SearchSpace, vel_search: SearchSpace,
                                   **kwargs):
        """
        To perform a Max Likelihood Search with extra uncertainty parameters (e.g., sensor bias,
        sensor position, or sensor velocity), we must encapsulate those extra variables in a broader
        SearchSpace object

        :param zeta: Measurement vector
        :param source_search: Definition of the source position (or pos/vel) search space; utils.solvers.SearchSpace instance
        :param do_sensor_bias: flag controlling whether this function will account for unknown measurement bias
        :param do_sensor_pos: flag controlling whether this function will account for sensor position errors
        :param do_sensor_vel: flag controlling whether this function will account for sensor velocity errors
        :param bias_search: utils.solvers.SearchSpace for measurement bias search
        :param pos_search: utils.solvers.SearchSpace for sensor position
        :param vel_search: utils.solvers.SearchSpace for sensor velocity
        :return x_est:  estimated source position (or position and velocity)
        :return likelihood: array of likelihood values at each position in the combined SearchSpace
        :return th_grid: grid of search positions
        :return parameter_est: dict containing the estimated bias, sensor position, and sensor velocity terms
        """

        # Make sure at least one term is true; otherwise this is just ML
        if not do_sensor_bias and not do_sensor_pos and not do_sensor_vel:
            x_est, likelihood, x_grid = self.max_likelihood(zeta, source_search, **kwargs)
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

        def ell(th):
            # Parse the parameter vector theta
            pos_vel = th[indices['source_indices']]
            x_source, v_source = self.parse_source_pos_vel(pos_vel, default_vel=self.vel)
            bias = th[indices['bias_indices']] if do_sensor_bias else self.bias
            x_sensor = np.reshape(th[indices['pos_indices']], self.pos.shape) if do_sensor_pos else self.pos
            v_sensor = np.reshape(th[indices['vel_indices']], self.vel.shape) if do_sensor_vel else self.vel

            return self.log_likelihood(zeta=zeta, x_source=x_source, v_source=v_source,
                                       x_sensor=x_sensor, v_sensor=v_sensor,
                                       bias=bias)

        th_est, likelihood, th_grid = utils.solvers.ml_solver(ell=ell, search_space=search_space,
                                                              **kwargs)

        # Parse the estimates
        x_est = th_est[indices['source_indices']]
        th_est = {'bias': th_est[indices['bias_indices']] if do_sensor_bias else None,
                  'pos': th_est[indices['pos_indices']] if do_sensor_pos else None,
                  'vel': th_est[indices['vel_indices']] if do_sensor_vel else None}

        return x_est, likelihood, th_grid, th_est

    @abstractmethod
    def gradient_descent(self, zeta, x_init, **kwargs):
        pass

    @abstractmethod
    def least_square(self, zeta, x_init, **kwargs):
        pass

    def sensor_calibration(self, zeta_cal, x_cal, v_cal=None,
                           pos_search: SearchSpace=None, vel_search: SearchSpace=None, bias_search: SearchSpace=None,
                           do_pos_cal=True, do_vel_cal=False, do_bias_cal=False):
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

        num_dim_cal, num_cal = utils.safe_2d_shape(x_cal)
        num_msmt, num_cal2 = utils.safe_2d_shape(zeta_cal)
        num_cov, _ = utils.safe_2d_shape(self.cov.cov)

        # Check dimension agreement
        assert num_dim_cal == self.num_dim, "Disagreement in number of spatial dimensions between sensor positions and calibration emitter positions."
        assert num_cal == num_cal2, "Disagreement in number of calibration emitters between x_cal and zeta_cal."

        if v_cal is not None:
            num_dim_v, num_cal_v = utils.safe_2d_shape(v_cal)
            assert num_dim_cal == num_dim_v and num_cal == num_cal_v, "Disagreement in size of calibration emitter velocities and sensor positions."
        else:
            v_cal = np.zeros_like(x_cal)  # assume zero velocity; simplified code later on

        # ==================== Initialize Search Space ========================
        pos_search = self.initialize_sensor_pos_search(pos_search, do_pos_search=do_pos_cal)
        vel_search = self.initialize_sensor_vel_search(vel_search, do_vel_search=do_vel_cal)
        bias_search = self.initialize_bias_search(bias_search, do_bias_search=do_bias_cal)

        # ==================== Log-Likelihood Wrapper Function ================
        # Accepts a 1D vector of measurement biases and a 1D vector of sensor positions

        def ell(b, x, v):
            # Reshape the sensor position
            this_x_sensor = np.reshape(x, shape=self.pos.shape)
            this_v_sensor = np.reshape(v, shape=self.vel.shape)
            res = 0
            for this_zeta, this_x, this_v in zip(zeta_cal.T, x_cal.T, v_cal.T):
                this_ell = self.log_likelihood(x_sensor=this_x_sensor, v_sensor=this_v_sensor, zeta=this_zeta,
                                               x_source=this_x, v_source=this_v, bias=b)
                res = res + this_ell
            return res

        x_sensor_est, v_sensor_est, bias_est = utils.solvers.sensor_calibration(ell, pos_search, vel_search, bias_search)

        return (np.reshape(x_sensor_est, shape=self.pos.shape),
                np.reshape(v_sensor_est, shape=self.pos.shape),
                bias_est)

    def make_uncertainty_search_space(self, source_search: SearchSpace,
                                      do_bias_search: bool, do_pos_search: bool, do_vel_search: bool,
                                      bias_search: SearchSpace=None, pos_search: SearchSpace=None,
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
        if len(source_search.x_ctr) != num_params:
            source_search.x_ctr = source_search.x_ctr * default_arr
        if len(source_search.epsilon) != num_params:
            source_search.epsilon = source_search.epsilon * default_arr
        if len(source_search.points_per_dim) != num_params:
            source_search.points_per_dim = source_search.points_per_dim * default_arr

        # === Parse the components for a combined search space =========
        field_names = ['x_ctr', 'epsilon', 'points_per_dim']
        combined_search = {}
        for field_name in field_names:
            components = [getattr(x, field_name) for x in (source_search, bias_search, pos_search, vel_search) if x is not None]
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
            x_ctr = self.pos.ravel()
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
            x_ctr = self.vel.ravel() if self.vel is not None else np.zeros_like(self.pos).ravel()
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

    def parse_uncertainty_indices(self, do_source_vel=False, do_bias=False, do_sensor_pos=False,
                                  do_sensor_vel=False):
        """
        Uncertainty parameters take the form:
           [x_source, v_source, bias, x_sensor.ravel(), v_sensor.ravel()]

        This function computes the indices for each term.
        """

        # First, compute the number of each component and determine is source velocity is included
        num_source_pos = self.num_dim
        num_source_vel = self.num_dim if do_sensor_pos else 0
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
    # These methods define basic performance predictions, and must be implemented
    @abstractmethod
    def compute_crlb(self, x_source, **kwargs):
        pass

    # ==================== Helper Methods =====================
    def plot_sensors(self, ax=None, **kwargs):
        if ax is not None:
            ax.scatter(self.pos[0], self.pos[1], **kwargs)
        else:
            plt.scatter(self.pos[0], self.pos[1], **kwargs)
        return

    def parse_source_pos_vel(self, pos_vel, default_vel):
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
        num_dim, _ = utils.safe_2d_shape(pos_vel)
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
    _ref_idx: np.floating or str or None = None      # specification for sensor pairs to use in difference operation
                                                        # Accepted values are:
                                                        #   int                         Index of common reference sensor
                                                        #   (2, num_pairs) ndarray      List of sensor pairs
                                                        #   'full'                      Use all possible sensor pairs
    _cov_resample: CovarianceMatrix or None             # difference-level covariance matrix
    _cov_raw: CovarianceMatrix or None                  # sensor-level covariance matrix
    _do_resample: bool = True

    parent = None

    def __init__(self, x: np.ndarray, cov: CovarianceMatrix or None, ref_idx, **kwargs):
        (super().__init__(x, cov, **kwargs))

        if cov is not None:
            self._cov_raw = cov.copy()
            self._do_resample = True
        else:
            self._do_resample = False

        self._ref_idx = ref_idx

        if cov is not None:
            self.resample()

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
    def cov(self, cov: CovarianceMatrix):
        # Must be input as a raw covariance matrix; one per sensor. Set the
        # do_resample flag.
        self._cov_raw = cov.copy()
        self._cov_resample = None
        self._do_resample = True
        if self.parent is not None: self.parent._do_resample = True
        self.resample()

    @cov.deleter
    def cov(self):
        self._cov_raw = None
        self._cov_resample = None
        self.num_measurements = 0
        self._do_resample = True
        if self.parent is not None: self.parent._do_resample = True

    @property
    def ref_idx(self):
        return self._ref_idx

    @ref_idx.setter
    def ref_idx(self, idx):
        # Make sure that we have a raw covariance matrix to resample
        if self._cov_raw is None:
            raise ValueError("Unable to set the ref_idx property; the raw sensor-level covariance matrix is undefined.")

        self._ref_idx = idx
        self._do_resample = True  # Reset the do_resample flag
        if self.parent is not None: self.parent._do_resample = True
        self.resample()

    @ref_idx.deleter
    def ref_idx(self):
        self._ref_idx = None
        self._do_resample = True
        if self.parent is not None: self.parent._do_resample = True

    def resample(self):
        if self._do_resample:
            self._cov_resample = self._cov_raw.resample(ref_idx=self._ref_idx)
            self.num_measurements = self._cov_resample.cov.shape[0]
        self._do_resample = False