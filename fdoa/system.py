from utils import SearchSpace
from . import model
import utils
from utils.system import DifferencePSS
from utils.covariance import CovarianceMatrix
import numpy as np

_speed_of_light = utils.constants.speed_of_light

class FDOAPassiveSurveillanceSystem(DifferencePSS):
    bias = None

    _default_fdoa_bias_search_epsilon = 1 # meters/second
    _default_fdoa_bias_search_size = 11 # num elements per search dimension
    _default_fdoa_vel_search_epsilon = 1 # meters/second
    _default_fdoa_vel_search_size = 11 # num elements per search dimensions

    def __init__(self,x, cov, **kwargs):
        # Handle empty covariance matrix inputs
        if cov is None:
            # Make a dummy; unit variance
            _, num_sensors = utils.safe_2d_shape(x)
            cov = CovarianceMatrix(np.eye(num_sensors))
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

    def log_likelihood(self, zeta, x_source, x_sensor=None, v_sensor=None, v_source=None, bias=None, **kwargs):
        if x_sensor is None: x_sensor = self.pos
        if v_sensor is None: v_sensor = self.vel
        if bias is None: bias = self.bias
        return model.log_likelihood(x_sensor=x_sensor, rho_dot=zeta, x_source=x_source, cov=self.cov,
                                    v_sensor=v_sensor, v_source=v_source, ref_idx=self.ref_idx,
                                    do_resample=False, bias=bias, **kwargs)

    # def log_likelihood_uncertainty(self, zeta, theta, **kwargs):
    #     return model.log_likelihood_uncertainty(x_sensor=self.pos, rho_dot=zeta, theta=theta, cov=self.cov,
    #                                             cov_pos=self.cov_pos, ref_idx=self.ref_idx,
    #                                             v_sensor=self.vel, do_resample=False, **kwargs)

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
    def max_likelihood(self, zeta, search_space:SearchSpace, cal_data: dict=None, **kwargs):
        # Perform sensor calibration
        if cal_data is not None:
            x_sensor, v_sensor, bias = self.sensor_calibration(**cal_data)
        else:
            x_sensor, v_sensor, bias = self.pos, self.vel, self.bias

        # Likelihood function for ML Solvers
        def ell(pos_vel, **ell_kwargs):
            # Determine if the input is position only, or position & velocity
            this_pos, this_vel = self.parse_source_pos_vel(pos_vel, np.zeros_like(pos_vel))
            return self.log_likelihood(x_sensor=x_sensor, v_sensor=v_sensor,
                                       zeta=zeta, x_source=this_pos, v_source=this_vel, **ell_kwargs)

        # Call the util function
        x_est, likelihood, x_grid = utils.solvers.ml_solver(ell=ell, search_space=search_space, **kwargs)

        return x_est, likelihood, x_grid

    # todo: delete when it's working
    # def max_likelihood_uncertainty(self, zeta, x_ctr, search_size, epsilon=None, do_sensor_bias=False, **kwargs):
    #     return solvers.max_likelihood_uncertainty(x_sensor=self.pos, zeta=zeta, cov=self.cov, cov_pos=self.cov_pos,
    #                                               ref_idx=self.ref_idx, x_ctr=x_ctr, search_size=search_size,
    #                                               epsilon=epsilon, do_resample=False, v_sensor=self.vel,
    #                                               do_sensor_bias=do_sensor_bias, **kwargs)

    def gradient_descent(self, zeta, x_init, cal_data: dict=None, **kwargs):
        # Perform sensor calibration
        if cal_data is not None:
            x_sensor, v_sensor, bias = self.sensor_calibration(**cal_data)
        else:
            x_sensor, v_sensor, bias = self.pos, self.vel, self.bias

        # Initialize measurement error and jacobian functions
        def y(pos_vel):
            this_pos, this_vel = self.parse_source_pos_vel(pos_vel, np.zeros_like(pos_vel))
            return zeta - self.measurement(x_source=this_pos, v_source=this_vel, x_sensor=x_sensor,
                                           v_sensor=v_sensor, bias=bias)

        def this_jacobian(pos_vel):
            this_pos, this_vel = self.parse_source_pos_vel(pos_vel, np.zeros_like(pos_vel))
            n_dim, _ = utils.safe_2d_shape(pos_vel) # is the calling function asking for just pos or pos/vel?
            j = self.jacobian(x_source=this_pos, v_source=this_vel, x_sensor=x_sensor, v_sensor=v_sensor)
            # Jacobian returns 2*n_dim rows; first the jacobian w.r.t. position, then velocity. Optionally
            # excise just the position portion
            return j[:n_dim]

        # Call generic Gradient Descent solver
        x_est, x_full = utils.solvers.gd_solver(y=y, jacobian=this_jacobian, cov=self.cov, x_init=x_init, **kwargs)

        return x_est, x_full

    def least_square(self, zeta, x_init, cal_data: dict=None, **kwargs):
        # Perform sensor calibration
        if cal_data is not None:
            x_sensor, v_sensor, bias = self.sensor_calibration(**cal_data)
        else:
            x_sensor, v_sensor, bias = self.pos, self.vel, self.bias

            # Initialize measurement error and jacobian functions
            def y(pos_vel):
                this_pos, this_vel = self.parse_source_pos_vel(pos_vel, np.zeros_like(pos_vel))
                return zeta - self.measurement(x_source=this_pos, v_source=this_vel, x_sensor=x_sensor,
                                               v_sensor=v_sensor, bias=bias)

            def this_jacobian(pos_vel):
                this_pos, this_vel = self.parse_source_pos_vel(pos_vel, np.zeros_like(pos_vel))
                n_dim, _ = utils.safe_2d_shape(pos_vel)  # is the calling function asking for just pos or pos/vel?
                j = self.jacobian(x_source=this_pos, v_source=this_vel, x_sensor=x_sensor, v_sensor=v_sensor)
                # Jacobian returns 2*n_dim rows; first the jacobian w.r.t. position, then velocity. Optionally
                # excise just the position portion
                return j[:n_dim]

            # Call generic Gradient Descent solver
            x_est, x_full = utils.solvers.ls_solver(zeta=y, jacobian=this_jacobian, cov=self.cov, x_init=x_init,
                                                    **kwargs)

            return x_est, x_full

    def bestfix(self, zeta, search_space: SearchSpace, pdf_type=None, cal_data: dict=None):
        # Perform sensor calibration
        if cal_data is not None:
            x_sensor, v_sensor, bias = self.sensor_calibration(**cal_data)
        else:
            x_sensor, v_sensor, bias = self.pos, self.vel, self.bias

        # Generate the PDF
        def measurement(pos_vel):
            this_pos, this_vel = self.parse_source_pos_vel(pos_vel, np.zeros_like(pos_vel))
            return self.measurement(x_source=this_pos, v_source=this_vel, x_sensor=x_sensor, v_sensor=v_sensor,
                                    bias=bias)

        pdfs = utils.make_pdfs(measurement, zeta, pdf_type, self.cov.cov)

        # Call the util function
        x_est, likelihood, x_grid = utils.solvers.bestfix(pdfs, search_space)

        return x_est, likelihood, x_grid

    ## ============================================================================================================== ##
    ## Performance Methods
    ##
    ## These methods handle predictions of system performance
    ## ============================================================================================================== ##
    def compute_crlb(self, x_source, v_source=None, **kwargs):

        def this_jacobian(pos_vel):
            this_pos, this_vel = self.parse_source_pos_vel(pos_vel, default_vel=v_source)
            n_dim, _ = utils.safe_2d_shape(pos_vel) # is the calling function asking for just pos or pos/vel?
            # Jacobian returns 2*self.n_dim rows; first the jacobian w.r.t. position, then velocity. Optionally
            # excise just the position portion
            return self.jacobian(x_source=this_pos, v_source=this_vel, x_sensor=self.pos, v_sensor=self.vel)[:n_dim]

        return utils.perf.compute_crlb_gaussian(x_source=x_source, jacobian=this_jacobian, cov=self.cov,
                                                **kwargs)

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

    def parse_source_pos_vel(self, pos_vel, default_vel):
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