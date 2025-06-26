from abc import ABC, abstractmethod
import numpy as np
import utils
from utils.covariance import CovarianceMatrix
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
    default_bias_search_size = 0
    default_sensor_pos_search_epsilon = 2.5
    default_sensor_pos_search_size = 25
    default_sensor_vel_search_epsilon = 1
    default_sensor_vel_search_size = 0  # By default, we can't search across sensor velocity

    def __init__(self, x: np.ndarray, cov: CovarianceMatrix, bias=None, cov_pos=None, vel=None):
        if len(np.shape(x))==0: x = np.expand_dims(x, 1) # Add a second dimension, if there isn't one
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

    @abstractmethod
    def log_likelihood_uncertainty(self, zeta, theta, **kwargs):
        pass

    # ==================== Solver Methods ===================
    # These methods define the basic solver methods, and associated utilities, and must be implemented.
    @abstractmethod
    def max_likelihood(self, zeta, x_ctr, search_size, epsilon, bias, cal_data, **kwargs):
        pass

    @abstractmethod
    def max_likelihood_uncertainty(self, zeta, x_ctr, search_size, epsilon, do_sensor_bias, **kwargs):
        pass

    @abstractmethod
    def gradient_descent(self, zeta, x_init, **kwargs):
        pass

    @abstractmethod
    def least_square(self, zeta, x_init, **kwargs):
        pass

    def sensor_calibration(self, zeta_cal, x_cal, v_cal=None, pos_search: dict=None, vel_search: dict=None,
                           bias_search: dict=None):
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
        :return x_sensor_est: Estimated sensor positions (None if ignored)
        :return bias_est: Estimated measurement biases (None if ignored)
        """

        # TODO: Test
        # ================ Parse inputs =========================
        if pos_search is None and bias_search is None:
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
        pos_search = self.initialize_sensor_pos_search(pos_search)
        vel_search = self.initialize_sensor_vel_search(vel_search)
        bias_search = self.initialize_bias_search(bias_search)

        # ==================== Log-Likelihood Wrapper Function ================
        # Accepts a 1D vector of measurement biases and a 1D vector of sensor positions

        def ell(b, x, v):
            # Reshape the sensor position
            this_x_sensor = np.reshape(x, newshape=self.pos.shape)
            this_v_sensor = np.reshape(v, newshape=self.vel.shape)
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


    def initialize_bias_search(self, bias_search: dict=None):
        if bias_search is None:
            return None

        if 'x_ctr' not in bias_search.keys() or bias_search['x_ctr'] is None:
            # If there is a bias specified in the object; use it as the center of the search window. Otherwise,
            # assume zero bias is the center
            bias_search['x_ctr'] = self.bias if self.bias is not None else np.zeros((self.num_measurements, ))

        if 'epsilon' not in bias_search.keys() or bias_search['epsilon'] is None:
            bias_search['epsilon'] = self.default_bias_search_epsilon

        if 'search_size' not in bias_search.keys() or bias_search['search_size'] is None:
            bias_search['search_size'] = self.default_bias_search_size

        return bias_search

    def initialize_sensor_pos_search(self, pos_search: dict=None):
        if pos_search is None:
            return None

        if 'x_ctr' not in pos_search.keys() or pos_search['x_ctr'] is None:
            pos_search['x_ctr'] = self.pos.ravel()

        if 'epsilon' not in pos_search.keys() or pos_search['epsilon'] is None:
            pos_search['epsilon'] = self.default_sensor_pos_search_epsilon

        if 'search_size' not in pos_search.keys() or pos_search['search_size'] is None:
            pos_search['search_size'] = self.default_sensor_pos_search_size

        return pos_search

    def initialize_sensor_vel_search(self, vel_search: dict=None):
        if vel_search is None:
            return None

        if 'x_ctr' not in vel_search.keys() or vel_search['x_ctr'] is None:
            vel_search['x_ctr'] = self.vel.ravel() if self.vel is not None else np.zeros_like(self.pos).ravel()

        if 'epsilon' not in vel_search.keys() or vel_search['epsilon'] is None:
            vel_search['epsilon'] = self.default_sensor_vel_search_epsilon

        if 'search_size' not in vel_search.keys() or vel_search['search_size'] is None:
            vel_search['search_size'] =self.default_sensor_vel_search_size

        return vel_search

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

    def __init__(self, x: np.ndarray, cov: CovarianceMatrix, ref_idx, **kwargs):
        (super().__init__(x, cov, **kwargs))

        self._cov_raw = cov.copy()
        self._ref_idx = ref_idx
        self._do_resample = True

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