import numpy as np
import numpy.typing as npt

from . import model
from ewgeo.utils import parse_reference_sensor, safe_2d_shape
from ewgeo.utils.covariance import CovarianceMatrix
from ewgeo.utils.system import DifferencePSS


class FDOAPassiveSurveillanceSystem(DifferencePSS):
    bias = None

    _default_fdoa_bias_search_epsilon: float = 1 # meters/second
    _default_fdoa_bias_search_size: int = 11 # num elements per search dimension
    _default_fdoa_vel_search_epsilon: float = 1 # meters/second
    _default_fdoa_vel_search_size: int = 11 # num elements per search dimensions

    def __init__(self,x: npt.ArrayLike,
                 cov: CovarianceMatrix | npt.ArrayLike | None=None, **kwargs):
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
    def measurement(self, x_source, v_source: npt.ArrayLike | None=None, x_sensor: npt.ArrayLike | None=None, v_sensor: npt.ArrayLike | None=None, bias: npt.ArrayLike | None=None):
        if x_sensor is None: x_sensor = self.pos
        if v_sensor is None: v_sensor = self.vel
        if bias is None: bias = self.bias
        return model.measurement(x_sensor=x_sensor, x_source=x_source, v_sensor=v_sensor, v_source=v_source,
                                 ref_idx=self.ref_idx, bias=bias)

    def jacobian(self, x_source, v_source: npt.ArrayLike | None=None, x_sensor: npt.ArrayLike | None=None, v_sensor: npt.ArrayLike | None=None):
        if x_sensor is None: x_sensor = self.pos
        if v_sensor is None: v_sensor = self.vel
        return model.jacobian(x_sensor=x_sensor, x_source=x_source, v_sensor=v_sensor, v_source=v_source,
                              ref_idx=self.ref_idx)

    def jacobian_uncertainty(self, x_source, **kwargs):
        return model.jacobian_uncertainty(x_sensor=self.pos, x_source=x_source, v_sensor=self.vel,
                                          ref_idx=self.ref_idx, **kwargs)

    def log_likelihood(self, zeta, x_source: npt.ArrayLike,
                       x_sensor: npt.ArrayLike | None=None,
                       v_sensor: npt.ArrayLike | None=None,
                       v_source: npt.ArrayLike | None=None,
                       bias: npt.ArrayLike | None=None, **kwargs):
        if x_sensor is None: x_sensor = self.pos
        if v_sensor is None: v_sensor = self.vel
        if bias is None: bias = self.bias
        return model.log_likelihood(x_sensor=x_sensor, rho_dot=zeta, x_source=x_source, cov=self.cov,
                                    v_sensor=v_sensor, v_source=v_source, ref_idx=self.ref_idx,
                                    do_resample=False, bias=bias, **kwargs)

    def grad_x(self,
               x_source: npt.ArrayLike,
               v_source: npt.ArrayLike | None=None,
               x_sensor: npt.ArrayLike | None=None,
               v_sensor: npt.ArrayLike | None=None)-> npt.NDArray:
        if x_sensor is None: x_sensor = self.pos
        if v_sensor is None: v_sensor = self.vel
        return model.grad_x(x_sensor=x_sensor, v_sensor=v_sensor,
                            x_source=x_source, v_source=v_source,
                            ref_idx=self.ref_idx)

    def grad_bias(self,
                  x_source: npt.ArrayLike,
                  v_source: npt.ArrayLike | None=None,
                  x_sensor: npt.ArrayLike | None=None,
                  v_sensor: npt.ArrayLike | None=None)-> npt.NDArray:
        if x_sensor is None: x_sensor = self.pos
        return model.grad_bias(x_sensor=x_sensor, x_source=x_source, ref_idx=self.ref_idx)

    def grad_sensor_pos(self,
                        x_source: npt.ArrayLike,
                        v_source: npt.ArrayLike | None=None,
                        x_sensor: npt.ArrayLike | None=None,
                        v_sensor: npt.ArrayLike | None=None)-> npt.NDArray:
        if x_sensor is None: x_sensor = self.pos
        if v_sensor is None: v_sensor = self.vel
        return model.grad_sensor_pos(x_sensor=x_sensor,
                                     v_sensor=v_sensor,
                                     x_source=x_source,
                                     v_source=v_source,
                                     ref_idx=self.ref_idx)

    def grad_sensor_vel(self,
                        x_source: npt.ArrayLike,
                        v_source: npt.ArrayLike | None=None,
                        x_sensor: npt.ArrayLike | None=None,
                        v_sensor: npt.ArrayLike | None=None)-> npt.NDArray:
        if x_sensor is None: x_sensor = self.pos
        if v_sensor is None: v_sensor = self.vel
        return model.grad_sensor_vel(x_sensor=x_sensor,
                                     v_sensor=v_sensor,
                                     x_source=x_source,
                                     v_source=v_source,
                                     ref_idx=self.ref_idx)

    ## ============================================================================================================== ##
    ## Solver Methods
    ##
    ## These methods handle the interface to solvers
    ## ============================================================================================================== ##
    #  No need to overload any of the super class' solver methods.

    ## ============================================================================================================== ##
    ## Performance Methods
    ##
    ## These methods handle predictions of system performance
    ## ============================================================================================================== ##

    ## ============================================================================================================== ##
    ## Helper Methods
    ##
    ## These are generic utility functions that are unique to this class
    ## ============================================================================================================== ##
    def error(self, x_source, x_max, num_pts, v_source: npt.ArrayLike | None=None):
        return model.error(x_sensor=self.pos, x_source=x_source, v_sensor=self.vel, v_source=v_source,
                           x_max=x_max, num_pts=num_pts, cov=self.cov,
                           do_resample=False, ref_idx=self.ref_idx)

    def draw_isodoppler(self, vel_diff, num_pts, max_ortho,
    v_source: npt.ArrayLike | None=None, x_sensor: npt.ArrayLike | None=None, v_sensor: npt.ArrayLike | None=None):
        if x_sensor is None:
            x_sensor = self.pos
        if v_sensor is None:
            v_sensor = self.vel

        test_idx_vec, ref_idx_vec = parse_reference_sensor(self.ref_idx, self.num_sensors)

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