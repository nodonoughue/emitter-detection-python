from . import model, perf, solvers
from utils.system import PassiveSurveillanceSystem


class DirectionFinder(PassiveSurveillanceSystem):
    do_2d_aoa: bool = False

    _default_aoa_bias_search_epsilon = 0.1 # degrees
    _default_aoa_bias_search_size = 1 # degrees

    def __init__(self,x, cov, do_2d_aoa=False, **kwargs):
        super().__init__(x, cov, **kwargs)

        self.do_2d_aoa = do_2d_aoa

        # Overwrite uncertainty search defaults
        self.default_bias_search_epsilon = self._default_aoa_bias_search_epsilon
        self.default_bias_search_size = self._default_aoa_bias_search_size

    ## ============================================================================================================== ##
    ## Model Methods
    ##
    ## These methods handle the physical model for a Triangulation-based PSS, and are just wrappers for the static
    ## functions defined in model.py
    ## ============================================================================================================== ##
    def measurement(self, x_source, x_sensor=None, bias=None, v_sensor=None, v_source=None):
        if x_sensor is None: x_sensor = self.pos
        if bias is None: bias = self.bias
        return model.measurement(x_sensor=x_sensor, x_source=x_source, do_2d_aoa=self.do_2d_aoa, bias=bias)

    def jacobian(self, x_source, v_source=None, x_sensor=None, v_sensor=None):
        if x_sensor is None: x_sensor = self.pos
        return model.jacobian(x_sensor=x_sensor, x_source=x_source, do_2d_aoa=self.do_2d_aoa)

    def jacobian_uncertainty(self, x_source, **kwargs):
        return model.jacobian_uncertainty(x_sensor=self.pos, x_source=x_source, do_2d_aoa=self.do_2d_aoa, **kwargs)

    def log_likelihood(self, x_source, zeta, x_sensor=None, bias=None, v_sensor=None, v_source=None):
        if x_sensor is None: x_sensor = self.pos
        if bias is None: bias = self.bias
        return model.log_likelihood(x_sensor=x_sensor, zeta=zeta, x_source=x_source, cov=self.cov,
                                    do_2d_aoa=self.do_2d_aoa, bias=bias)

    def log_likelihood_uncertainty(self, zeta, theta, **kwargs):
        return model.log_likelihood_uncertainty(x_sensor=self.pos, zeta=zeta, theta=theta, cov=self.cov,
                                                cov_pos=self.cov_pos, do_2d_aoa=self.do_2d_aoa, **kwargs)

    ## ============================================================================================================== ##
    ## Solver Methods
    ##
    ## These methods handle the interface to solvers
    ## ============================================================================================================== ##
    def max_likelihood(self, zeta, x_ctr, search_size, epsilon, cal_data: dict=None, **kwargs):
        # Perform sensor calibration
        if cal_data is not None:
            x_sensor, v_sensor, bias = self.sensor_calibration(*cal_data)
        else:
            x_sensor, v_sensor, bias = None, None, None

        # Call the non-calibration solver
        return solvers.max_likelihood(x_sensor=x_sensor, psi=zeta, cov=self.cov, do_2d_aoa=self.do_2d_aoa, x_ctr=x_ctr,
                                      search_size=search_size, epsilon=epsilon, bias=bias, **kwargs)

    def max_likelihood_uncertainty(self, zeta, x_ctr, search_size, epsilon=None, do_sensor_bias=False, **kwargs):
        # Just a pass-through
        return solvers.max_likelihood_uncertainty(x_sensor=self.pos, psi=zeta, x_ctr=x_ctr, cov=self.cov,
                                                  cov_pos=self.cov_pos, search_size=search_size,
                                                  do_2d_aoa=self.do_2d_aoa, epsilon=epsilon,
                                                  do_sensor_bias=do_sensor_bias, **kwargs)

    def gradient_descent(self, zeta, x_init, cal_data: dict=None, **kwargs):
        # Perform sensor calibration
        if cal_data is not None:
            x_sensor, v_sensor, bias = self.sensor_calibration(*cal_data)
        else:
            x_sensor, v_sensor, bias = None, None, None

        return solvers.gradient_descent(x_sensor=x_sensor, psi=zeta, cov=self.cov, bias=bias, do_2d_aoa=self.do_2d_aoa,
                                        **kwargs)

    def least_square(self, zeta, x_init, cal_data: dict=None, **kwargs):
        # Perform sensor calibration
        if cal_data is not None:
            x_sensor, v_sensor, bias = self.sensor_calibration(*cal_data)
        else:
            x_sensor, v_sensor, bias = None, None, None

        return solvers.least_square(x_sensor=self.pos, psi=zeta, cov=self.cov, x_init=x_init, bias=bias,
                                    do_2d_aoa=self.do_2d_aoa, **kwargs)

    def bestfix(self, zeta, x_ctr, search_size, epsilon, pdf_type=None):
        return solvers.bestfix(x_sensor=self.pos, psi=zeta, cov=self.cov, x_ctr=x_ctr, search_size=search_size,
                               epsilon=epsilon, pdf_type=pdf_type)

    def angle_bisector(self, zeta):
        return solvers.angle_bisector(self.pos, zeta)

    def centroid(self, zeta):
        return solvers.centroid(self.pos, zeta)

    ## ============================================================================================================== ##
    ## Performance Methods
    ##
    ## These methods handle predictions of system performance
    ## ============================================================================================================== ##
    def compute_crlb(self, x_source, print_progress=False, **kwargs):
        return perf.compute_crlb(x_sensor=self.pos, cov=self.cov, x_source=x_source, do_2d_aoa=self.do_2d_aoa,
                                 print_progress=print_progress, **kwargs)

    ## ============================================================================================================== ##
    ## Helper Methods
    ##
    ## These are generic utility functions that are unique to this class
    ## ============================================================================================================== ##
    def error(self, x_source, x_max, num_pts):
        return model.error(x_sensor=self.pos, x_source=x_source, x_max=x_max, num_pts=num_pts, cov=self.cov,
                           do_2d_aoa=self.do_2d_aoa)

    def draw_lob(self, zeta, **kwargs):
        return model.draw_lob(x_sensor=self.pos, psi=zeta, **kwargs)
