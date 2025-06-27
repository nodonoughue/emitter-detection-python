from . import model, perf, solvers
import utils
from utils.system import DifferencePSS
import numpy as np

_speed_of_light = utils.constants.speed_of_light

class FDOAPassiveSurveillanceSystem(DifferencePSS):
    bias = None

    _default_fdoa_bias_search_epsilon = 1 # meters/second
    _default_fdoa_bias_search_size = 10 # meters/second
    _default_fdoa_vel_search_epsilon = 1 # meters/second
    _default_fdoa_vel_search_size = 10 # meters/second

    def __init__(self,x, cov, **kwargs):

        super().__init__(x=x, cov=cov, **kwargs)

        # Overwrite uncertainty search defaults
        self.default_bias_search_epsilon = self._default_fdoa_bias_search_epsilon
        self.default_bias_search_size = self._default_fdoa_bias_search_size
        self.default_sensor_vel_search_epsilon = self._default_fdoa_vel_search_epsilon
        self.default_sensor_vel_search_size = self._default_fdoa_vel_search_size

    ## ============================================================================================================== ##
    ## Model Methods
    ##
    ## These methods handle the physical model for a FDOA-based PSS, and are just wrappers for the static
    ## functions defined in model.py
    ## ============================================================================================================== ##
    def measurement(self, x_source, v_source=None, x_sensor=None, v_sensor=None, bias=None):
        if x_sensor is None: x_sensor = self.pos
        if v_sensor is None: v_sensor = self.vel
        if bias is None: bias = self.bias
        return model.measurement(x_sensor=x_sensor, x_source=x_source, v_sensor=v_sensor, v_source=v_source,
                                 ref_idx=self.ref_idx, bias=bias)

    def jacobian(self, x_source, v_source=None, x_sensor=None, v_sensor=None):
        if x_sensor is None: x_sensor = self.pos
        if v_sensor is None: v_sensor = self.vel
        return model.jacobian(x_sensor=x_sensor, x_source=x_source, v_sensor=v_sensor, v_source=v_source,
                              ref_idx=self.ref_idx)

    def jacobian_uncertainty(self, x_source, **kwargs):
        return model.jacobian_uncertainty(x_sensor=self.pos, x_source=x_source, v_sensor=self.vel,
                                          ref_idx=self.ref_idx, **kwargs)

    def log_likelihood(self, zeta, x_source, x_sensor=None, v_sensor=None, v_source=None, bias=None):
        if x_sensor is None: x_sensor = self.pos
        if v_sensor is None: v_sensor = self.vel
        if bias is None: bias = self.bias
        return model.log_likelihood(x_sensor=x_sensor, rho_dot=zeta, x_source=x_source, cov=self.cov,
                                    v_sensor=v_sensor, v_source=v_source, ref_idx=self.ref_idx,
                                    do_resample=False, bias=bias)

    def log_likelihood_uncertainty(self, zeta, theta, **kwargs):
        return model.log_likelihood_uncertainty(x_sensor=self.pos, rho_dot=zeta, theta=theta, cov=self.cov,
                                                cov_pos=self.cov_pos, ref_idx=self.ref_idx,
                                                v_sensor=self.vel, do_resample=False, **kwargs)

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
            x_sensor, v_sensor, bias = self.pos, self.vel, self.bias

        # Call the non-calibration solver
        return solvers.max_likelihood(x_sensor=x_sensor, v_sensor=v_sensor, psi=zeta, cov=self.cov,
                                      ref_idx=self.ref_idx, x_ctr=x_ctr, search_size=search_size, epsilon=epsilon,
                                      bias=bias, do_resample=False, **kwargs)

    def max_likelihood_uncertainty(self, zeta, x_ctr, search_size, epsilon=None, do_sensor_bias=False, **kwargs):
        return solvers.max_likelihood_uncertainty(x_sensor=self.pos, zeta=zeta, cov=self.cov, cov_pos=self.cov_pos,
                                                  ref_idx=self.ref_idx, x_ctr=x_ctr, search_size=search_size,
                                                  epsilon=epsilon, do_resample=False, v_sensor=self.vel,
                                                  do_sensor_bias=do_sensor_bias, **kwargs)

    def gradient_descent(self, zeta, x_init, cal_data: dict=None, **kwargs):
        # Perform sensor calibration
        if cal_data is not None:
            x_sensor, v_sensor, bias = self.sensor_calibration(*cal_data)
        else:
            x_sensor, v_sensor, bias = self.pos, self.vel, self.bias

        return solvers.gradient_descent(x_sensor=x_sensor, v_sensor=self.vel, zeta=zeta, cov=self.cov,
                                        x_init=x_init, ref_idx=self.ref_idx,
                                        do_resample=False, **kwargs)

    def least_square(self, zeta, x_init, cal_data: dict=None, **kwargs):
        # Perform sensor calibration
        if cal_data is not None:
            x_sensor, v_sensor, bias = self.sensor_calibration(*cal_data)
        else:
            x_sensor, v_sensor, bias = self.pos, self.vel, self.bias

        return solvers.least_square(x_sensor=x_sensor, v_sensor=self.vel, zeta=zeta, cov=self.cov, x_init=x_init,
                                    ref_idx=self.ref_idx, do_resample=False, **kwargs)

    def bestfix(self, zeta, x_ctr, search_size, epsilon, pdf_type=None):
        return solvers.bestfix(x_sensor=self.pos, v_sensor=self.vel, zeta=zeta, cov=self.cov, x_ctr=x_ctr,
                               search_size=search_size, epsilon=epsilon, pdf_type=pdf_type, ref_idx=self.ref_idx,
                               do_resample=False)

    ## ============================================================================================================== ##
    ## Performance Methods
    ##
    ## These methods handle predictions of system performance
    ## ============================================================================================================== ##
    def compute_crlb(self, x_source, **kwargs):
        return perf.compute_crlb(x_sensor=self.pos, v_sensor=self.vel, x_source=x_source, cov=self.cov,
                                 ref_idx=self.ref_idx, do_resample=False, **kwargs)

    ## ============================================================================================================== ##
    ## Helper Methods
    ##
    ## These are generic utility functions that are unique to this class
    ## ============================================================================================================== ##
    def error(self, x_source, x_max, num_pts, v_source=None):
        return model.error(x_sensor=self.pos, x_source=x_source, v_sensor=self.vel, v_source=v_source,
                           x_max=x_max, num_pts=num_pts, cov=self.cov,
                           do_resample=False, ref_idx=self.ref_idx)

    def draw_isodoppler(self, vel_diff, num_pts, max_ortho, v_source=None, x_sensor=None, v_sensor=None):
        if x_sensor is None:
            x_sensor = self.pos
        if v_sensor is None:
            v_sensor = self.vel

        test_idx_vec, ref_idx_vec = utils.parse_reference_sensor(self.ref_idx, self.num_sensors)

        test_pos = x_sensor[:,test_idx_vec]
        test_vel = v_sensor[:,test_idx_vec] if v_sensor is not None else np.zeros_like(test_pos)
        ref_pos = x_sensor[:,ref_idx_vec]
        ref_vel = v_sensor[:,ref_idx_vec] if v_sensor is not None else np.zeros_like(ref_pos)

        isodopplers = [model.draw_isodoppler(x_test=x_t, v_test=v_t, x_ref=x_r, v_ref=v_r,
                                             vdiff=v_diff, num_pts=num_pts, max_ortho=max_ortho,
                                             v_source=v_source) for
                       (x_t, v_t, x_r, v_r, v_diff) in zip(test_pos.T, test_vel.T, ref_pos.T, ref_vel.T, vel_diff)]

        return isodopplers