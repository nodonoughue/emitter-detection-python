import numpy as np
import numpy.typing as npt
import warnings

from . import model, solvers
from ewgeo.utils import parse_reference_sensor, safe_2d_shape
from ewgeo.utils.constants import speed_of_light
from ewgeo.utils.covariance import CovarianceMatrix
from ewgeo.utils.system import DifferencePSS


class TDOAPassiveSurveillanceSystem(DifferencePSS):
    bias = None

    _default_tdoa_bias_search_epsilon: float = 1 # meters
    _default_tdoa_bias_search_size: int = 11 # num search points per dimension

    def __init__(self,x: npt.ArrayLike,
                 cov: CovarianceMatrix | npt.ArrayLike | None=None,
                 variance_is_toa=True, **kwargs):

        # First, we need to convert from TOA to ROA
        if variance_is_toa and cov is not None:
            if not isinstance(cov, CovarianceMatrix):
                # Convert it to a CovarianceMatrix object
                cov = CovarianceMatrix(cov)
            # Convert to ROA units
            cov = cov.multiply(speed_of_light ** 2, overwrite=False)

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
    def measurement(self, x_source: npt.ArrayLike,
                    x_sensor: npt.ArrayLike | None=None,
                    bias: npt.ArrayLike | None=None,
                    v_sensor: npt.ArrayLike | None=None,
                    v_source: npt.ArrayLike | None=None):
        if x_sensor is None: x_sensor = self.pos
        if bias is None: bias = self.bias
        return model.measurement(x_sensor=x_sensor, x_source=x_source, ref_idx=self.ref_idx, bias=bias)

    def jacobian(self, x_source: npt.ArrayLike,
                 v_source: npt.ArrayLike | None=None,
                 x_sensor: npt.ArrayLike | None=None,
                 v_sensor: npt.ArrayLike | None=None):
        if x_sensor is None: x_sensor = self.pos
        return model.jacobian(x_sensor=x_sensor, x_source=x_source, ref_idx=self.ref_idx)

    def jacobian_uncertainty(self, x_source: npt.ArrayLike, **kwargs):
        return model.jacobian_uncertainty(x_sensor=self.pos, x_source=x_source, ref_idx=self.ref_idx, **kwargs)

    def log_likelihood(self, zeta: npt.ArrayLike, x_source: npt.ArrayLike,
                       x_sensor: npt.ArrayLike | None=None,
                       bias: npt.ArrayLike | None=None,
                       v_sensor: npt.ArrayLike | None=None,
                       v_source: npt.ArrayLike | None=None, **kwargs):
        if x_sensor is None: x_sensor = self.pos
        if bias is None: bias = self.bias
        return model.log_likelihood(x_sensor=x_sensor, zeta=zeta, x_source=x_source, cov=self.cov, ref_idx=self.ref_idx,
                                    variance_is_toa=False, do_resample=False, bias=bias, **kwargs)

    def grad_x(self,
               x_source: npt.ArrayLike,
               v_source: npt.ArrayLike | None=None,
               x_sensor: npt.ArrayLike | None=None,
               v_sensor: npt.ArrayLike | None=None)-> npt.NDArray:
        if x_sensor is None: x_sensor = self.pos
        return model.grad_x(x_sensor=x_sensor, x_source=x_source, ref_idx=self.ref_idx)

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

        return model.grad_sensor_pos(x_sensor=x_sensor, x_source=x_source, ref_idx=self.ref_idx)

    def grad_sensor_vel(self, x_source: npt.ArrayLike, **kwargs)-> npt.NDArray:
        out_shape = [self.num_dim * self.num_sensors, self.num_measurements]
        _, num_source = safe_2d_shape(x_source)
        if num_source > 1: out_shape.append(num_source)
        return np.zeros(shape=out_shape)

    ## ============================================================================================================== ##
    ## Solver Methods
    ##
    ## These methods handle the interface to solvers
    ## ============================================================================================================== ##
    def chan_ho(self, zeta: npt.ArrayLike, cal_data: dict=None):
        # Perform sensor calibration
        if cal_data is not None:
            x_sensor, _, bias = self.sensor_calibration(**cal_data)
        else:
            x_sensor, _, bias = self.pos, None, self.bias

        if bias is not None:
            warnings.warn("Chan-Ho TDOA solver does not accept bias. Ignoring bias.")
        return solvers.chan_ho(x_sensor=x_sensor, zeta=zeta, cov=self.cov, ref_idx=self.ref_idx, do_resample=False,
                               variance_is_toa=False)

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
    def error(self, x_source: npt.ArrayLike, x_max: npt.ArrayLike, num_pts: int):
        return model.error(x_sensor=self.pos, x_source=x_source, x_max=x_max, num_pts=num_pts, cov=self.cov,
                           do_resample=False, variance_is_toa=False, ref_idx=self.ref_idx)

    def draw_isochrones(self, range_diff: npt.ArrayLike, num_pts: int, max_ortho: float,
                        x_sensor: npt.ArrayLike | None=None):
        if x_sensor is None:
            x_sensor = self.pos

        test_idx_vec, ref_idx_vec = parse_reference_sensor(self.ref_idx, self.num_sensors)

        isochrones = [model.draw_isochrone(x_ref=x_sensor[:, ref_idx], x_test=x_sensor[:, test_idx],
                                           range_diff=this_range_diff, num_pts=num_pts, max_ortho=max_ortho) for
                      (test_idx, ref_idx, this_range_diff) in zip(test_idx_vec, ref_idx_vec, range_diff)]
        return isochrones

    def generate_parameter_indices(self, do_bias: bool=True):
        return model.generate_parameter_indices(x_sensor=self.pos, do_bias=do_bias)
