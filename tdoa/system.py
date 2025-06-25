from . import model, perf, solvers
import utils
from utils.system import DifferencePSS

_speed_of_light = utils.constants.speed_of_light

class TDOAPassiveSurveillanceSystem(DifferencePSS):
    bias = None

    _default_tdoa_bias_search_epsilon = 1 # meters
    _default_tdoa_bias_search_size = 10 # meters

    def __init__(self,x, cov, variance_is_toa=True, do_resample=False, **kwargs):
        # First, we need to convert from TOA to ROA
        if variance_is_toa:
            cov = cov.multiply(_speed_of_light**2, overwrite=False)

        super().__init__(x, cov, do_resample=do_resample, **kwargs)

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

    def jacobian(self, x_source, v_source=None):
        return model.jacobian(x_sensor=self.pos, x_source=x_source, ref_idx=self.ref_idx)

    def jacobian_uncertainty(self, x_source, **kwargs):
        return model.jacobian_uncertainty(x_sensor=self.pos, x_source=x_source, ref_idx=self.ref_idx, **kwargs)

    def log_likelihood(self, zeta, x_source, x_sensor=None, bias=None, v_sensor=None, v_source=None):
        if x_sensor is None: x_sensor = self.pos
        if bias is None: bias = self.bias
        return model.log_likelihood(x_sensor=x_sensor, zeta=zeta, x_source=x_source, cov=self.cov, ref_idx=self.ref_idx,
                                    variance_is_toa=False, do_resample=False, bias=bias)

    def log_likelihood_uncertainty(self, zeta, theta, **kwargs):
        return model.log_likelihood_uncertainty(x_sensor=self.pos, zeta=zeta, theta=theta, cov=self.cov,
                                                cov_pos=self.cov_pos, ref_idx=self.ref_idx,
                                                variance_is_toa=False, do_resample=False, **kwargs)

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
        x_sensor, bias = self.sensor_calibration(*cal_data)

        # Call the non-calibration solver
        return solvers.max_likelihood(x_sensor=x_sensor, psi=zeta, cov=self.cov, ref_idx=self.ref_idx, x_ctr=x_ctr,
                                      search_size=search_size, epsilon=epsilon, bias=bias,
                                      do_resample=False, variance_is_toa=False, **kwargs)

    def max_likelihood_uncertainty(self, zeta, x_ctr, search_size, epsilon=None, do_sensor_bias=False, **kwargs):
        return solvers.max_likelihood_uncertainty(x_sensor=self.pos, zeta=zeta, cov=self.cov, cov_pos=self.cov_pos,
                                                  ref_idx=self.ref_idx, x_ctr=x_ctr, search_size=search_size,
                                                  epsilon=epsilon, do_resample=False, variance_is_toa=False, **kwargs)

    def gradient_descent(self, zeta, x_init, cal_data: dict=None, **kwargs):
        # Perform sensor calibration
        x_sensor, bias = self.sensor_calibration(*cal_data)

        return solvers.gradient_descent(x_sensor=x_sensor, zeta=zeta, cov=self.cov, th_init=x_init, ref_idx=self.ref_idx,
                                        do_resample=False, variance_is_toa=False, **kwargs)

    def least_square(self, zeta, x_init, cal_data: dict=None, **kwargs):
        # Perform sensor calibration
        x_sensor, bias = self.sensor_calibration(*cal_data)

        return solvers.least_square(x_sensor=x_sensor, zeta=zeta, cov=self.cov, x_init=x_init, ref_idx=self.ref_idx,
                                    do_resample=False, variance_is_toa=False, **kwargs)

    def bestfix(self, zeta, x_ctr, search_size, epsilon, pdf_type=None):
        return solvers.bestfix(x_sensor=self.pos, zeta=zeta, cov=self.cov, x_ctr=x_ctr, search_size=search_size,
                               epsilon=epsilon, pdf_type=pdf_type,
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

    def draw_isochrones(self, range_diff, num_pts, max_ortho):
        test_idx_vec, ref_idx_vec = utils.parse_reference_sensor(self.ref_idx, self.num_sensors)

        isochrones = [model.draw_isochrone(self.pos[test_idx], self.pos[ref_idx], range_diff, num_pts, max_ortho) for
                      (test_idx, ref_idx) in zip(test_idx_vec, ref_idx_vec)]
        return isochrones

    def generate_parameter_indices(self, do_bias=True):
        return model.generate_parameter_indices(x_sensor=self.pos, do_bias=do_bias)
