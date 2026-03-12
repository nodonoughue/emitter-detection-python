from matplotlib import pyplot as plt
import numpy as np
from numpy import typing as npt

from . import model, solvers
from ewgeo.utils import SearchSpace
from ewgeo.utils.covariance import CovarianceMatrix
from ewgeo.utils.system import PassiveSurveillanceSystem


class DirectionFinder(PassiveSurveillanceSystem):
    do_2d_aoa: bool = False

    _default_aoa_bias_search_epsilon: float = 0.1 # degrees
    _default_aoa_bias_search_size: int = 11 # num search points per dimension

    def __init__(self,x: npt.ArrayLike,
                 cov: CovarianceMatrix | npt.ArrayLike | None=None,
                 do_2d_aoa: bool=False, **kwargs):

        super().__init__(x, cov, **kwargs)

        self.do_2d_aoa = do_2d_aoa

        # Overwrite uncertainty search defaults
        self.default_bias_search_epsilon = self._default_aoa_bias_search_epsilon
        self.default_bias_search_size = self._default_aoa_bias_search_size

    @property
    def num_measurements(self)-> int:
        return self.num_sensors * (2 if self.do_2d_aoa else 1)

    # For DF, if we're doing 2D AOA, then there are two bias terms per sensor, not 1.
    @property
    def cov_bias(self) -> CovarianceMatrix | None:
        if self._cov_bias is None:
            # Make it an identity matrix with one row for each sensor
            self._cov_bias = CovarianceMatrix(np.eye(self.num_sensors * (2 if self.do_2d_aoa else 1)))
        return self._cov_bias

    @cov_bias.setter
    def cov_bias(self, value: CovarianceMatrix | npt.ArrayLike | None):
        if isinstance(value, CovarianceMatrix) or value is None:
            self._cov_bias = value
        else:
            self._cov_bias = CovarianceMatrix(value)
        return

    ## ============================================================================================================== ##
    ## Model Methods
    ##
    ## These methods handle the physical model for a Triangulation-based PSS, and are just wrappers for the static
    ## functions defined in model.py
    ## ============================================================================================================== ##
    def measurement(self, x_source, x_sensor: npt.ArrayLike | None=None, bias: npt.ArrayLike | None=None,
                    v_sensor: npt.ArrayLike | None=None, v_source: npt.ArrayLike | None=None)->npt.NDArray[np.float64]:
        if x_sensor is None: x_sensor = self.pos
        if bias is None: bias = self.bias
        return model.measurement(x_sensor=x_sensor, x_source=x_source, do_2d_aoa=self.do_2d_aoa, bias=bias)

    def jacobian(self, x_source, v_source: npt.ArrayLike | None=None, x_sensor: npt.ArrayLike | None=None, v_sensor: npt.ArrayLike | None=None):
        if x_sensor is None: x_sensor = self.pos
        return model.jacobian(x_sensor=x_sensor, x_source=x_source, do_2d_aoa=self.do_2d_aoa)

    def jacobian_uncertainty(self, x_source, **kwargs):
        return model.jacobian_uncertainty(x_sensor=self.pos, x_source=x_source, do_2d_aoa=self.do_2d_aoa, **kwargs)

    def log_likelihood(self,
                       x_source: npt.ArrayLike,
                       zeta: npt.ArrayLike,
                       x_sensor: npt.ArrayLike| None=None,
                       bias: npt.ArrayLike | None=None,
                       v_sensor: npt.ArrayLike | None=None,
                       v_source: npt.ArrayLike | None=None,
                       **kwargs):
        if x_sensor is None: x_sensor = self.pos
        if bias is None: bias = self.bias
        if 'do_2d_aoa' not in kwargs: kwargs['do_2d_aoa'] = self.do_2d_aoa
        return model.log_likelihood(x_sensor=x_sensor, zeta=zeta, x_source=x_source, cov=self.cov,
                                    bias=bias, **kwargs)

    def grad_source(self,
                    x_source: npt.ArrayLike,
                    v_source: npt.ArrayLike | None=None,
                    x_sensor: npt.ArrayLike | None=None,
                    v_sensor: npt.ArrayLike | None=None)-> npt.NDArray:
        if x_sensor is None: x_sensor = self.pos
        return model.grad_source(x_sensor=x_sensor, x_source=x_source, do_2d_aoa=self.do_2d_aoa)

    def grad_bias(self,
                  x_source: npt.ArrayLike,
                  v_source: npt.ArrayLike | None=None,
                  x_sensor: npt.ArrayLike | None=None,
                  v_sensor: npt.ArrayLike | None=None)-> npt.NDArray:
        if x_sensor is None: x_sensor = self.pos
        return model.grad_bias(x_sensor=x_sensor, x_source=x_source, do_2d_aoa=self.do_2d_aoa)

    def grad_sensor_pos(self,
                        x_source: npt.ArrayLike,
                        v_source: npt.ArrayLike | None=None,
                        x_sensor: npt.ArrayLike | None=None,
                        v_sensor: npt.ArrayLike | None=None)-> npt.NDArray:
        if x_sensor is None: x_sensor = self.pos
        return model.grad_sensor_pos(x_sensor=x_sensor, x_source=x_source, do_2d_aoa=self.do_2d_aoa)

    def grad_sensor_vel(self, x_source: npt.NDArray[np.float64], **kwargs)-> npt.NDArray[np.float64]:
        out_shape = [self.num_dim * self.num_sensors, self.num_measurements]
        shp = np.shape(x_source)
        num_source = shp[1] if len(shp) > 1 else 1
        if num_source > 1: out_shape.append(num_source)
        return np.zeros(shape=out_shape)

    ## ============================================================================================================== ##
    ## Solver Methods
    ##
    ## These methods handle the interface to solvers
    ## ============================================================================================================== ##
    def max_likelihood(self, zeta, search_space: SearchSpace, cal_data: dict=None, **kwargs):
        # Specify the do_2d_aoa flag, if not already provided
        if 'do_2d_aoa' not in kwargs:
            kwargs['do_2d_aoa'] = self.do_2d_aoa

        # Call the super method
        return super().max_likelihood(zeta, search_space, cal_data=cal_data, **kwargs)

    def angle_bisector(self, zeta):
        return solvers.angle_bisector(self.pos, zeta)

    def centroid(self, zeta):
        return solvers.centroid(self.pos, zeta)

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
    def error(self, x_source, x_max, num_pts):
        return model.error(x_sensor=self.pos, x_source=x_source, x_max=x_max, num_pts=num_pts, cov=self.cov,
                           do_2d_aoa=self.do_2d_aoa)

    def draw_lobs(self, zeta, x_sensor: npt.ArrayLike | None=None, **kwargs):
        """
        Draw lines of bearing from each sensor corresponding to each measurement

        :param zeta: ndarray; size of the first dimension is the number of measurements, any other dimensions are
                              interpreted as additional cases to run (num_cases = np.prod(np.shape(zeta)[1: ]))
        :param x_sensor: optional ndarray; if empty, then self.pos will be used
        :param kwargs: other named arguments will be passed to triang.model.draw_lob().
        :return lobs: numpy array with four dimensions: num_dims (2 or 3) x 2 x num_sensors x num_cases.
        """

        # Parse the sensors
        if x_sensor is None:
            x_sensor = self.pos
            num_sensors= self.num_measurements
        else:
            shp = np.shape(x_sensor)
            # num_dim = shp[0] if len(shp) > 0 else 1
            num_sensors = shp[1] if len(shp) > 1 else 1

        x_sensor = np.array(x_sensor)  # make sure it's a numpy array
        if num_sensors == 1 & len(x_sensor.shape) == 1:
            x_sensor = x_sensor[:, np.newaxis]  # make sure it's not a 1d array; that'll mess up indexing later

        # Parse the input measurements
        shp = np.shape(zeta)
        num_zeta = shp[0] if len(shp) > 0 else 1
        num_cases = shp[1] if len(shp) > 1 else 1
        # assert num_zeta == num_measurements, "Sensor measurement dimension mismatch."
        zeta_reshape = np.reshape(zeta, shape=(num_zeta, num_cases))

        # Initialize the output
        # Dimensions are: (2 or 3) x 2 x num_sensors x num_cases
        lobs_out = np.zeros(shape=(2, 2, num_sensors, num_cases))

        # Loop over lobs
        for idx_sensor in range(num_sensors):
            this_x = x_sensor[:, idx_sensor]

            for idx_case in range(num_cases):
                if self.do_2d_aoa:
                    this_zeta = zeta_reshape[idx_sensor:num_sensors, idx_case]
                else:
                    this_zeta = zeta_reshape[idx_sensor, idx_case]

                # TODO: Test LOBs with 2D AOA measurements

                this_lob = model.draw_lob(x_sensor=this_x, psi=this_zeta, **kwargs)
                lobs_out[:, :, idx_sensor, idx_case] = np.asarray(this_lob).squeeze()

        return lobs_out

    def plot_lobs(self, ax: plt.Axes, zeta: npt.ArrayLike, x_sensor: npt.ArrayLike=None, plot_args: dict=None, **kwargs):
        # Generate a tuple of shape (2, 2, num_lobs, num_cases)
        # First dimension is physical axis (x,y)
        # Second dimensions is start/end
        # Third dimension is across sensors
        # Fourth dimension is across case
        if plot_args is None: plot_args = {} # Make an empty dict so that ax.plot doesn't throw an error

        lobs = np.reshape(self.draw_lobs(zeta=zeta, x_sensor=x_sensor, **kwargs), (2, 2, -1))
        ax.plot(lobs[0], lobs[1], **plot_args)

