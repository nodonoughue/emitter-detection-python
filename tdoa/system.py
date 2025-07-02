from . import model, perf, solvers
import utils
from utils import SearchSpace
from utils.system import DifferencePSS
from utils.covariance import CovarianceMatrix
import numpy as np

_speed_of_light = utils.constants.speed_of_light

class TDOAPassiveSurveillanceSystem(DifferencePSS):
    bias = None

    _default_tdoa_bias_search_epsilon = 1 # meters
    _default_tdoa_bias_search_size = 10 # meters

    def __init__(self,x, cov, variance_is_toa=True, **kwargs):
        # Handle empty covariance matrix inputs
        if cov is None:
            # Make a dummy; unit variance
            _, num_sensors = utils.safe_2d_shape(x)
            cov = CovarianceMatrix(np.eye(num_sensors))

        # First, we need to convert from TOA to ROA
        if variance_is_toa:
            cov = cov.multiply(_speed_of_light**2, overwrite=False)

        super().__init__(x, cov, **kwargs)

        # Overwrite uncertainty search defaults
        self.default_bias_search_epsilon = self._default_tdoa_bias_search_epsilon
        self.default_bias_search_size = self._default_tdoa_bias_search_size

    ## ============================================================================================================== ##
    ## Model Methods
    ##
    ## These methods handle the physical model for a TDOA-based PSS, and are just wrappers for the static
    ## functions defined in model.py
    ## ============================================================================================================== ##
    def measurement(self, x_source, x_sensor=None, bias=None, v_sensor=None, v_source=None):
        if x_sensor is None: x_sensor = self.pos
        if bias is None: bias = self.bias
        return model.measurement(x_sensor=x_sensor, x_source=x_source, ref_idx=self.ref_idx, bias=bias)

    def jacobian(self, x_source, v_source=None, x_sensor=None, v_sensor=None):
        if x_sensor is None: x_sensor = self.pos
        return model.jacobian(x_sensor=x_sensor, x_source=x_source, ref_idx=self.ref_idx)

    def jacobian_uncertainty(self, x_source, **kwargs):
        return model.jacobian_uncertainty(x_sensor=self.pos, x_source=x_source, ref_idx=self.ref_idx, **kwargs)

    def log_likelihood(self, zeta, x_source, x_sensor=None, bias=None, v_sensor=None, v_source=None):
        if x_sensor is None: x_sensor = self.pos
        if bias is None: bias = self.bias
        return model.log_likelihood(x_sensor=x_sensor, zeta=zeta, x_source=x_source, cov=self.cov, ref_idx=self.ref_idx,
                                    variance_is_toa=False, do_resample=False, bias=bias)

    # def log_likelihood_uncertainty(self, zeta, theta, **kwargs):
    #     return model.log_likelihood_uncertainty(x_sensor=self.pos, zeta=zeta, theta=theta, cov=self.cov,
    #                                             cov_pos=self.cov_pos, ref_idx=self.ref_idx,
    #                                             variance_is_toa=False, do_resample=False, **kwargs)

    def grad_x(self, x_source):
        return model.grad_x(x_sensor=self.pos, x_source=x_source, ref_idx=self.ref_idx)

    def grad_bias(self, x_source):
        return model.grad_bias(x_sensor=self.pos, x_source=x_source, ref_idx=self.ref_idx)

    def grad_sensor_pos(self, x_source):
        return model.grad_sensor_pos(x_sensor=self.pos, x_source=x_source, ref_idx=self.ref_idx)

    ## ============================================================================================================== ##
    ## Solver Methods
    ##
    ## These methods handle the interface to solvers
    ## ============================================================================================================== ##
    def max_likelihood(self, zeta, x_ctr, search_size, epsilon=None, cal_data: dict=None, **kwargs):
        # Perform sensor calibration
        if cal_data is not None:
            x_sensor, v_sensor, bias = self.sensor_calibration(*cal_data)
        else:
            x_sensor, v_sensor, bias = self.pos, None, self.bias

        # Call the non-calibration solver
        return solvers.max_likelihood(x_sensor=x_sensor, zeta=zeta, cov=self.cov, ref_idx=self.ref_idx, x_ctr=x_ctr,
                                      search_size=search_size, epsilon=epsilon, bias=bias,
                                      do_resample=False, variance_is_toa=False, **kwargs)

    # def max_likelihood_uncertainty(self, zeta, x_ctr, search_size, epsilon=None, do_sensor_bias=False, cov_pos=None,
    #                                **kwargs):
    #     if cov_pos is None: cov_pos = self.cov_pos
    #
    #     return solvers.max_likelihood_uncertainty(x_sensor=self.pos, zeta=zeta, cov=self.cov, cov_pos=cov_pos,
    #                                               ref_idx=self.ref_idx, x_ctr=x_ctr, search_size=search_size,
    #                                               epsilon=epsilon, do_resample=False, variance_is_toa=False,
    #                                               do_sensor_bias=do_sensor_bias, **kwargs)

    def gradient_descent(self, zeta, x_init, cal_data: dict=None, **kwargs):
        # Perform sensor calibration
        if cal_data is not None:
            x_sensor, v_sensor, bias = self.sensor_calibration(*cal_data)
        else:
            x_sensor, v_sensor, bias = self.pos, None, self.bias

        return solvers.gradient_descent(x_sensor=x_sensor, zeta=zeta, cov=self.cov, x_init=x_init, ref_idx=self.ref_idx,
                                        do_resample=False, variance_is_toa=False, **kwargs)

    def least_square(self, zeta, x_init, cal_data: dict=None, **kwargs):
        # Perform sensor calibration
        if cal_data is not None:
            x_sensor, v_sensor, bias = self.sensor_calibration(*cal_data)
        else:
            x_sensor, v_sensor, bias = self.pos, None, self.bias

        return solvers.least_square(x_sensor=x_sensor, zeta=zeta, cov=self.cov, x_init=x_init, ref_idx=self.ref_idx,
                                    do_resample=False, variance_is_toa=False, **kwargs)

    def bestfix(self, zeta, search_space: SearchSpace, pdf_type=None):
        return solvers.bestfix(x_sensor=self.pos, zeta=zeta, cov=self.cov,
                               search_space=search_space, pdf_type=pdf_type,
                               do_resample=False, variance_is_toa=False)

    ## ============================================================================================================== ##
    ## Performance Methods
    ##
    ## These methods handle predictions of system performance
    ## ============================================================================================================== ##
    def compute_crlb(self, x_source, **kwargs):
        return perf.compute_crlb(x_sensor=self.pos, x_source=x_source, cov=self.cov, ref_idx=self.ref_idx,
                                  do_resample=False, variance_is_toa=False, **kwargs)

    ## ============================================================================================================== ##
    ## Helper Methods
    ##
    ## These are generic utility functions that are unique to this class
    ## ============================================================================================================== ##
    def error(self, x_source, x_max, num_pts):
        return model.error(x_sensor=self.pos, x_source=x_source, x_max=x_max, num_pts=num_pts, cov=self.cov,
                           do_resample=False, variance_is_toa=False, ref_idx=self.ref_idx)

    def draw_isochrones(self, range_diff, num_pts, max_ortho, x_sensor=None):
        if x_sensor is None:
            x_sensor = self.pos

        test_idx_vec, ref_idx_vec = utils.parse_reference_sensor(self.ref_idx, self.num_sensors)

        isochrones = [model.draw_isochrone(x_ref=x_sensor[:,ref_idx], x_test=x_sensor[:,test_idx],
                                           range_diff=this_range_diff, num_pts=num_pts, max_ortho=max_ortho) for
                      (test_idx, ref_idx, this_range_diff) in zip(test_idx_vec, ref_idx_vec, range_diff)]
        return isochrones

    def generate_parameter_indices(self, do_bias=True):
        return model.generate_parameter_indices(x_sensor=self.pos, do_bias=do_bias)
