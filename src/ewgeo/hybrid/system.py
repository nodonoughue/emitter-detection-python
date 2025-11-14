import numpy as np
import numpy.typing as npt
from scipy.linalg import block_diag

from ewgeo.fdoa import FDOAPassiveSurveillanceSystem
from ewgeo.tdoa import TDOAPassiveSurveillanceSystem
from ewgeo.triang import DirectionFinder
from ewgeo.utils import SearchSpace
from ewgeo.utils.covariance import CovarianceMatrix
from ewgeo.utils.system import DifferencePSS


class HybridPassiveSurveillanceSystem(DifferencePSS):
    bias = None
    aoa: DirectionFinder | None = None
    tdoa: TDOAPassiveSurveillanceSystem | None = None
    fdoa: FDOAPassiveSurveillanceSystem | None = None

    # Subordinate fields
    _aoa_sensor_idx: npt.NDArray[np.int64] | None = None
    _tdoa_sensor_idx: npt.NDArray[np.int64] | None = None
    _fdoa_sensor_idx: npt.NDArray[np.int64] | None = None
    _aoa_measurement_idx: npt.NDArray[np.int64] | None = None
    _tdoa_measurement_idx: npt.NDArray[np.int64] | None = None
    _fdoa_measurement_idx: npt.NDArray[np.int64] | None = None

    def __init__(self, cov: CovarianceMatrix | npt.ArrayLike | None=None,
                 aoa: DirectionFinder | None=None, tdoa: TDOAPassiveSurveillanceSystem | None=None,
                 fdoa: FDOAPassiveSurveillanceSystem | None=None,
                 ref_idx: str | npt.ArrayLike | None=None, **kwargs):

        assert aoa is not None or tdoa is not None or fdoa is not None, \
            'Error initializing HybridPSS system; at least one type of subordinate PSS system must be supplied.'

        # Store the subordinates
        self.aoa = aoa
        self.tdoa = tdoa
        self.fdoa = fdoa

        # Pull reference indices and covariance matrices from subordinates, if they aren't
        # explicitly provided as inputs
        if ref_idx is None:
            ref_idx = self.parse_reference_indices()

        if cov is None:
            cov = self.parse_covariance_matrix()

        # Parse the subordinates
        self.parse_subordinates()

        # Initiate the superclass
        super().__init__(self.pos, cov, ref_idx, vel=self.vel, **kwargs)

        return

    def get_attr_from_pss(self, attr: str, concat_dim=None)-> npt.ArrayLike:
        """
        Get an attribute from the subordinate PSS systems and concatenate along the specified dimension.

        :param attr: Attribute name to get from the subordinate PSS systems (str).
        :param concat_dim: Dimension along which to concatenate the attribute values (int or None). If None,
                           then the arrays are flattened before use.
        :return: Attribute values concatenated along the specified dimension.
        """
        pss_list = (self.aoa, self.tdoa, self.fdoa)

        collected = []
        for pss in pss_list:
            if pss is None: continue
            val = getattr(pss, attr, None)
            if val is not None: collected.append(np.atleast_1d(val))

        # None of the PSS objects have a defined value for this attr
        if not collected: return None

        # Scalar response; no need to concatenate
        if len(collected)==1: return np.asarray(collected[0])

        # If concat_dim is None -- flatten before concatenation
        if concat_dim is None:
            collected = [v.reshape(-1) for v in collected]
            concat_dim = 0

        return np.concatenate(collected, axis=concat_dim)

    @property
    def pos(self):
        return self.get_attr_from_pss('pos', concat_dim=1)

    @pos.setter
    def pos(self, x: npt.ArrayLike):
        x = np.array(x)
        if self.aoa and self._aoa_sensor_idx is not None:
            self.aoa.pos = x[:, self.aoa_sensor_idx]
        if self.tdoa and self._tdoa_sensor_idx is not None:
            self.tdoa.pos = x[:, self.tdoa_sensor_idx]
        if self.fdoa and self._fdoa_sensor_idx is not None:
            self.fdoa.pos = x[:, self.fdoa_sensor_idx]

    @property
    def vel_full(self):
        return self.get_attr_from_pss('vel', concat_dim=1)

    @property
    def vel(self):
        # Get just the FDOA velocities
        if self.fdoa is not None:
            return self.fdoa.vel
        else:
            return None

    @vel.setter
    def vel(self, v: npt.ArrayLike):
        if v is None:
            # Delete it from all defined subordinates
            [delattr(pss,'vel') for pss in [self.aoa, self.tdoa, self.fdoa] if pss is not None]
            return
        else:
            # Check dimensions, parse the velocity, and distribute the result
            desired_dims_full = (self.num_dim, self.num_sensors)
            desired_dims_partial = (self.num_dim, self.num_fdoa_sensors)
            v = np.array(v)
            if v.shape == desired_dims_partial:
                # Only FDOA velocity defined
                # Clear it from AOA/TDOA, for good measure
                [delattr(pss, 'vel') for pss in [self.aoa, self.tdoa] if pss is not None]
                self.fdoa.vel = v
            elif np.shape(v) == desired_dims_full:
                if self.aoa:
                    self.aoa.vel = v[:, self.aoa_sensor_idx]
                if self.tdoa:
                    self.tdoa.vel = v[:, self.tdoa_sensor_idx]
                if self.fdoa:
                    self.fdoa.vel = v[:, self.fdoa_sensor_idx]
            else:
                raise ValueError(f'Unable to parse input array shape {np.shape(v)}.')
        return

    @property
    def num_aoa_sensors(self)-> int:
        return self.aoa.num_sensors if self.aoa is not None else 0

    @property
    def num_tdoa_sensors(self)-> int:
        return self.tdoa.num_sensors if self.tdoa is not None else 0

    @property
    def num_fdoa_sensors(self)-> int:
        return self.fdoa.num_sensors if self.fdoa is not None else 0

    @property
    def num_sensors(self)->int:
        return self.num_aoa_sensors + self.num_tdoa_sensors + self.num_fdoa_sensors

    @property
    def num_aoa_measurements(self)-> int:
        return self.aoa.num_measurements if self.aoa is not None else 0

    @property
    def num_tdoa_measurements(self)-> int:
        return self.tdoa.num_measurements if self.tdoa is not None else 0

    @property
    def num_fdoa_measurements(self)-> int:
        return self.fdoa.num_measurements if self.fdoa is not None else 0

    @property
    def num_measurements(self)-> int:
        return self.num_aoa_measurements + self.num_tdoa_measurements + self.num_fdoa_measurements

    @property
    def num_measurements_raw(self)-> int:
        return np.sum(self.get_attr_from_pss("num_measurements_raw"), axis=None)

    @property
    def aoa_sensor_idx(self)-> npt.NDArray[np.int64]:
        return self._aoa_sensor_idx

    @property
    def tdoa_sensor_idx(self)-> npt.NDArray[np.int64]:
        return self._tdoa_sensor_idx

    @property
    def fdoa_sensor_idx(self)-> npt.NDArray[np.int64]:
        return self._fdoa_sensor_idx

    @property
    def aoa_measurement_idx(self)-> npt.NDArray[np.int64]:
        return self._aoa_measurement_idx

    @property
    def tdoa_measurement_idx(self)-> npt.NDArray[np.int64]:
        return self._tdoa_measurement_idx

    @property
    def fdoa_measurement_idx(self)-> npt.NDArray[np.int64]:
        return self._fdoa_measurement_idx

    @property
    def default_bias_search_epsilon(self)-> npt.NDArray[np.float64]:
        """
        The bias search term needs to be either a scalar, or have shape (num_measurements,).
        """
        bias_search_epsilon = []
        for pss in [self.aoa, self.tdoa, self.fdoa]:
            if pss is None: continue

            # Add defaults from this PSS
            bias_search_epsilon.append([pss.default_bias_search_epsilon] * pss.num_measurements)
        return np.concatenate(bias_search_epsilon, axis=None)

    @property
    def default_bias_search_size(self)-> npt.NDArray[np.int64]:
        """
        The bias search term needs to be either a scalar, or have shape (num_measurements,).
        """
        bias_search_size = []
        for pss in [self.aoa, self.tdoa, self.fdoa]:
            if pss is None: continue

            # Add defaults from this PSS
            bias_search_size.append([pss.default_bias_search_size] * pss.num_measurements)
        return np.concatenate(bias_search_size, axis=None).astype(int)

    @property
    def default_sensor_pos_search_epsilon(self) -> npt.NDArray[np.float64]:
        """
        The sensor pos search term needs to either be scalar, or have the same shape as
        self.pos
        """
        sensor_pos_search_epsilon = []
        for pss in [self.aoa, self.tdoa, self.fdoa]:
            if pss is None: continue

            # Add defaults from this PSS
            sensor_pos_search_epsilon.append(pss.default_sensor_pos_search_epsilon
                                             * np.ones([self.num_dim, pss.num_sensors]))
        return np.concatenate(sensor_pos_search_epsilon, axis=1)

    @property
    def default_sensor_pos_search_size(self) -> npt.NDArray[np.int64]:
        sensor_pos_search_size = []
        for pss in [self.aoa, self.tdoa, self.fdoa]:
            if pss is None: continue

            # Add defaults from this PSS
            sensor_pos_search_size.append(pss.default_sensor_pos_search_size * np.ones([self.num_dim, pss.num_sensors]))
        return np.concatenate(sensor_pos_search_size, axis=1).astype(int)

    @property
    def default_sensor_vel_search_epsilon(self) -> float | None:
        if self.fdoa is not None:
            return self.fdoa.default_sensor_vel_search_epsilon
        else:
            return None

    @property
    def default_sensor_vel_search_size(self) -> int | None:
        if self.fdoa is not None:
            return self.fdoa.default_sensor_vel_search_size
        else:
            return None

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
                    v_source: npt.ArrayLike | None=None)-> npt.NDArray:
        # Call the three measurement models and concatenate the results along the first axis

        # Break apart the sensor position, velocity, and bias measurement inputs into their AOA, TDOA, and FDOA
        # components
        x_aoa, x_tdoa, x_fdoa = self.parse_sensor_data(x_sensor)
        _, _, v_fdoa = self.parse_sensor_data(v_sensor, vel_input=True)
        b_aoa, b_tdoa, b_fdoa = self.parse_sensor_data(bias)

        # Parse source position and velocity
        if v_source is None:
            # It might be passed as a single input under x_source with 2*num_dim rows
            x_source, v_source = self.parse_source_pos_vel(pos_vel=x_source,
                                                           default_vel=np.zeros_like(x_source))

        # Call component models
        measurements = [
            model.measurement(x_source, x_sensor=sensor, bias=bias,
                              v_sensor=v_fdoa if model is self.fdoa else None,
                              v_source=v_source if model is self.fdoa else None)
            for model, sensor, bias in[
                (self.aoa, x_aoa, b_aoa),
                (self.tdoa, x_tdoa, b_tdoa),
                (self.fdoa, x_fdoa, b_fdoa)
            ]
            if model is not None
        ]

        return np.concatenate(measurements, axis=0)

    def jacobian(self, x_source: npt.ArrayLike,
                 v_source: npt.ArrayLike | None=None,
                 x_sensor: npt.ArrayLike | None=None,
                 v_sensor: npt.ArrayLike | None=None)-> npt.NDArray:
        # Parse sensor pos/vel overrides
        if x_sensor is None:
            x_aoa = self.aoa.pos if self.aoa is not None else None
            x_tdoa = self.tdoa.pos if self.tdoa is not None else None
            x_fdoa = self.fdoa.pos if self.fdoa is not None else None
        else:
            x_aoa, x_tdoa, x_fdoa = self.parse_sensor_data(x_sensor)
        if v_sensor is None:
            if self.fdoa is not None:
                v_fdoa = self.fdoa.vel
            else:
                v_fdoa = np.zeros_like(x_fdoa)
        else:
            _, _, v_fdoa = self.parse_sensor_data(v_sensor, vel_input=True)

        # Parse source position and velocity
        if v_source is None:
            # It might be passed as a single input under x_source with 2*num_dim rows
            x_source, v_source = self.parse_source_pos_vel(x_source, np.zeros_like(x_source))

        # Call component models
        to_concat = []
        if self.aoa is not None:
            to_concat.append(self.aoa.jacobian(x_source, x_sensor=x_aoa))

        if self.tdoa is not None:
            to_concat.append(self.tdoa.jacobian(x_source, x_sensor=x_tdoa))

        if self.fdoa is not None:
            # Replicate the AOA and TDOA jacobians to reflect jacobian w.r.t. pos and vel
            to_concat = [np.concatenate((j, np.zeros_like(j)), axis=0) for j in to_concat]
            to_concat.append(self.fdoa.jacobian(x_source=x_source, v_source=v_source, x_sensor=x_fdoa, v_sensor=v_fdoa))

        # Combine component Jacobian matrices
        return np.concatenate(to_concat, axis=1)

    def jacobian_uncertainty(self, x_source: npt.ArrayLike,
                             v_source: npt.ArrayLike | None=None, **kwargs)-> npt.NDArray:
        # Parse source position and velocity
        if v_source is None:
            # It might be passed as a single input under x_source with 2*num_dim rows
            x_source, v_source = self.parse_source_pos_vel(x_source, np.zeros_like(x_source))

        to_concat = []
        if self.aoa is not None:
            to_concat.append(self.aoa.jacobian_uncertainty(x_source=x_source, **kwargs))
        if self.tdoa is not None:
            to_concat.append(self.tdoa.jacobian_uncertainty(x_source=x_source, **kwargs))
        if self.fdoa is not None:
            # Replicate the AOA and TDOA jacobians to reflect jacobian w.r.t. pos and vel
            to_concat = [np.concatenate((j, np.zeros_like(j)), axis=0) for j in to_concat]
            to_concat.append(self.fdoa.jacobian_uncertainty(x_source=x_source, **kwargs))

        return np.concatenate(to_concat, axis=1)

    def log_likelihood(self, x_source: npt.ArrayLike, zeta: npt.ArrayLike,
                       x_sensor: npt.ArrayLike | None=None,
                       bias: npt.ArrayLike | None=None,
                       v_sensor: npt.ArrayLike | None=None,
                       v_source: npt.ArrayLike | None=None, **kwargs)-> npt.NDArray:
        # Break apart the sensor position, velocity, and bias measurement inputs into their AOA, TDOA, and FDOA
        # components
        x_aoa, x_tdoa, x_fdoa = self.parse_sensor_data(x_sensor)
        _, _, v_fdoa = self.parse_sensor_data(v_sensor, vel_input=True)
        b_aoa, b_tdoa, b_fdoa = self.parse_measurement_data(bias)
        z_aoa, z_tdoa, z_fdoa = self.parse_measurement_data(zeta)

        # Parse source position and velocity
        if v_source is None:
            # It might be passed as a single input under x_source with 2*num_dim rows
            x_source, v_source = self.parse_source_pos_vel(x_source, np.zeros_like(x_source))

        result = 0
        if self.aoa is not None:
            result = result + self.aoa.log_likelihood(x_source=x_source, zeta=z_aoa,
                                                      x_sensor=x_aoa, bias=b_aoa, **kwargs)
        if self.tdoa is not None:
            result = result + self.tdoa.log_likelihood(x_source=x_source, zeta=z_tdoa,
                                                       x_sensor=x_tdoa, bias=b_tdoa, **kwargs)
        if self.fdoa is not None:
            result = result + self.fdoa.log_likelihood(x_source=x_source, zeta=z_fdoa, x_sensor=x_fdoa,
                                                       v_sensor=v_fdoa, v_source=v_source, bias=b_fdoa, **kwargs)

        return result

    def log_likelihood_uncertainty(self, x_source: npt.ArrayLike, zeta: npt.ArrayLike,
                                   x_sensor: npt.ArrayLike | None=None,
                                   bias: npt.ArrayLike | None=None,
                                   v_sensor: npt.ArrayLike | None=None,
                                   v_source: npt.ArrayLike | None=None, **kwargs)-> npt.NDArray:

        # Break apart the sensor position, velocity, and bias measurement inputs into their AOA, TDOA, and FDOA
        # components
        x_aoa, x_tdoa, x_fdoa = self.parse_sensor_data(x_sensor)
        _, _, v_fdoa = self.parse_sensor_data(v_sensor, vel_input=True)
        b_aoa, b_tdoa, b_fdoa = self.parse_measurement_data(bias)
        z_aoa, z_tdoa, z_fdoa = self.parse_measurement_data(zeta)

        # Parse source position and velocity
        if v_source is None:
            # It might be passed as a single input under x_source with 2*num_dim rows
            x_source, v_source = self.parse_source_pos_vel(x_source, np.zeros_like(x_source))


        result = 0
        if self.aoa is not None:
            result = result + self.aoa.log_likelihood_uncertainty(x_source=x_source, zeta=z_aoa,
                                                                  x_sensor=x_aoa, bias=b_aoa, **kwargs)
        if self.tdoa is not None:
            result = result + self.tdoa.log_likelihood_uncertainty(x_source=x_source, zeta=z_tdoa,
                                                                   x_sensor=x_tdoa, bias=b_tdoa, **kwargs)
        if self.fdoa is not None:
            result = result + self.fdoa.log_likelihood_uncertainty(x_source=x_source, zeta=z_fdoa, x_sensor=x_fdoa,
                                                           v_sensor=v_fdoa, v_source=v_source, bias=b_fdoa, **kwargs)

        return result

    def grad_x(self,
               x_source: npt.ArrayLike,
               v_source: npt.ArrayLike | None=None,
               x_sensor: npt.ArrayLike | None=None,
               v_sensor: npt.ArrayLike | None=None)-> npt.NDArray:

        # Break apart the sensor position, velocity, and bias measurement inputs into their AOA, TDOA, and FDOA
        # components
        x_aoa, x_tdoa, x_fdoa = self.parse_sensor_data(x_sensor)
        _, _, v_fdoa = self.parse_sensor_data(v_sensor, vel_input=True)

        # Parse source position and velocity
        if v_source is None:
            # It might be passed as a single input under x_source with 2*num_dim rows
            x_source, v_source = self.parse_source_pos_vel(pos_vel=x_source,
                                                           default_vel=np.zeros_like(x_source))

        # Call each subordinate PSS and compute its gradient (w.r.t source position)
        grads = [
            pss.grad_x(x_source=x_source, x_sensor=sensor,
                       v_sensor=v_fdoa if pss is self.fdoa else None,
                       v_source=v_source if pss is self.fdoa else None)
            for pss, sensor in [
                (self.aoa, x_aoa),
                (self.tdoa, x_tdoa),
                (self.fdoa, x_fdoa)
            ]
            if pss is not None
        ]

        return np.concatenate(grads, axis=1)

    def grad_bias(self,
                  x_source: npt.ArrayLike,
                  v_source: npt.ArrayLike | None=None,
                  x_sensor: npt.ArrayLike | None=None,
                  v_sensor: npt.ArrayLike | None=None)-> npt.NDArray:
        # Break apart the sensor position, velocity, and bias measurement inputs into their AOA, TDOA, and FDOA
        # components
        x_aoa, x_tdoa, x_fdoa = self.parse_sensor_data(x_sensor)
        _, _, v_fdoa = self.parse_sensor_data(v_sensor, vel_input=True)

        # Parse source position and velocity
        if v_source is None:
            # It might be passed as a single input under x_source with 2*num_dim rows
            x_source, v_source = self.parse_source_pos_vel(pos_vel=x_source,
                                                           default_vel=np.zeros_like(x_source))

        # Call each subordinate PSS and compute its gradient (w.r.t source position)
        grads = [
            pss.grad_bias(x_source=x_source, x_sensor=sensor,
                          v_sensor=v_fdoa if pss is self.fdoa else None,
                          v_source=v_source if pss is self.fdoa else None)
            for pss, sensor in [
                (self.aoa, x_aoa),
                (self.tdoa, x_tdoa),
                (self.fdoa, x_fdoa)
            ]
            if pss is not None
        ]

        if len(grads) == 0:
            # No gradients; nothing to return
            return np.array([])
        elif len(grads) == 1:
            # Only one gradient was generated; return it directly
            return grads[0]

        orig_shapes = [np.shape(g) for g in grads]
        max_len = max(map(len, orig_shapes))
        if max_len > 2:
            # Move axes (0, 1) to the end so that block_diag will work on them properly
            grads = [np.moveaxis(g, (0, 1), (-2, -1)) for g in grads]

        # At this point, we know len(grads) is >0, but PyCharm doesn't, so it's presenting a static analysis warning.
        # Shift the function call to grads[0], *grads[1:] to ensure at least one positional argument is passed in.
        grad = block_diag(grads[0], *grads[1:])

        # Move the axes back
        if max_len > 2:
            grad = np.moveaxis(grad, (-2, -1), (0, 1))

        return grad

    def grad_sensor_pos(self,
                        x_source: npt.ArrayLike,
                        v_source: npt.ArrayLike | None=None,
                        x_sensor: npt.ArrayLike | None=None,
                        v_sensor: npt.ArrayLike | None=None)-> npt.NDArray:
        # Break apart the sensor position, velocity, and bias measurement inputs into their AOA, TDOA, and FDOA
        # components
        x_aoa, x_tdoa, x_fdoa = self.parse_sensor_data(x_sensor)
        _, _, v_fdoa = self.parse_sensor_data(v_sensor, vel_input=True)

        # Parse source position and velocity
        if v_source is None:
            # It might be passed as a single input under x_source with 2*num_dim rows
            x_source, v_source = self.parse_source_pos_vel(pos_vel=x_source,
                                                           default_vel=np.zeros_like(x_source))

        to_concat = []
        shp = np.shape(x_source)
        num_sources = shp[1] if len(shp) > 1 else 1
        if self.aoa is not None:
            # Response is (num_dim * num_aoa_sensors, num_aoa_measurements, num_sources)
            # Expand vertically to account for *all* sensors, then horizontally to account for *all* measurements
            j_aoa = self.aoa.grad_sensor_pos(x_source=x_source, x_sensor=x_aoa)
            j_aoa_full = np.zeros((self.num_dim, self.num_sensors, self.num_aoa_measurements, num_sources))
            j_aoa_full[:, self.aoa_sensor_idx, :, :] = np.reshape(j_aoa, (self.num_dim,
                                                                          self.num_aoa_sensors,
                                                                          self.num_aoa_measurements, -1), order='F')
            to_concat.append(np.reshape(j_aoa_full, shape=(self.num_dim*self.num_sensors,
                                                           self.num_aoa_measurements, num_sources), order='F'))
        if self.tdoa is not None:
            # Response is (num_dim * num_tdoa_sensors, num_tdoa_measurements, num_sources)
            # Expand vertically to account for
            j_tdoa = self.tdoa.grad_sensor_pos(x_source=x_source, x_sensor=x_tdoa)
            j_tdoa_full = np.zeros((self.num_dim, self.num_sensors, self.num_tdoa_measurements, num_sources))
            j_tdoa_full[:, self.tdoa_sensor_idx, :, :] = np.reshape(j_tdoa, (self.num_dim,
                                                                             self.num_tdoa_sensors,
                                                                             self.num_tdoa_measurements, -1), order='F')
            to_concat.append(np.reshape(j_tdoa_full, shape=(self.num_dim * self.num_sensors,
                                                            self.num_tdoa_measurements, num_sources), order='F'))
        if self.fdoa is not None:
            j_fdoa = self.fdoa.grad_sensor_pos(x_source=x_source, v_source=v_source, x_sensor=x_fdoa, v_sensor=v_fdoa)
            j_fdoa_full = np.zeros((self.num_dim, self.num_sensors, self.num_fdoa_measurements, num_sources))
            j_fdoa_full[:, self.fdoa_sensor_idx, :, :] = np.reshape(j_fdoa, (self.num_dim,
                                                                             self.num_fdoa_sensors,
                                                                             self.num_fdoa_measurements, -1), order='F')
            to_concat.append(np.reshape(j_fdoa_full, shape=(self.num_dim * self.num_sensors,
                                                            self.num_fdoa_measurements, num_sources), order='F'))

        return np.concatenate(to_concat, axis=1)

    def grad_sensor_vel(self,
                        x_source: npt.ArrayLike,
                        v_source: npt.ArrayLike | None=None,
                        x_sensor: npt.ArrayLike | None=None,
                        v_sensor: npt.ArrayLike | None=None)-> npt.NDArray:
        # Break apart the sensor position, velocity, and bias measurement inputs into their AOA, TDOA, and FDOA
        # components
        x_aoa, x_tdoa, x_fdoa = self.parse_sensor_data(x_sensor)
        _, _, v_fdoa = self.parse_sensor_data(v_sensor, vel_input=True)

        # Parse source position and velocity
        if v_source is None:
            # It might be passed as a single input under x_source with 2*num_dim rows
            x_source, v_source = self.parse_source_pos_vel(pos_vel=x_source,
                                                           default_vel=np.zeros_like(x_source))

        # Initialize the matrix as all zeros. We'll use the gradient w.r.t FDOA to fill in the appropriate columns
        out_shape = [self.num_dim * self.num_fdoa_sensors, self.num_measurements]
        shp = np.shape(x_source)
        num_sources = shp[1] if len(shp) > 1 else 1
        if num_sources > 1: out_shape.append(num_sources)
        res = np.zeros(shape=out_shape)

        if self.fdoa is not None:
            j_fdoa = self.fdoa.grad_sensor_vel(x_source=x_source, v_source=v_source, x_sensor=x_fdoa, v_sensor=v_fdoa)
            res[:, self.fdoa_measurement_idx] = j_fdoa

        return res

    ## ============================================================================================================== ##
    ## Solver Methods
    ##
    ## These methods handle the interface to solvers
    ## ============================================================================================================== ##
    def max_likelihood_uncertainty(self, zeta: npt.ArrayLike, search_space:SearchSpace,
                                   do_sensor_bias: bool=False, do_sensor_pos: bool=False, do_sensor_vel: bool=False,
                                   **kwargs)-> tuple[npt.NDArray[np.float64],
                                                     npt.NDArray[np.float64],
                                                     tuple[npt.NDArray[np.float64], ...],
                                                     dict]:

        # Call the super class to do the search, then reparse the results
        x_est, likelihood, th_grid, th_est = super().max_likelihood_uncertainty(zeta, search_space, do_sensor_bias,
                                                                                do_sensor_pos, do_sensor_vel, **kwargs)

        bias_aoa, bias_tdoa, bias_fdoa = self.parse_measurement_data(th_est['bias'])
        pos_aoa, pos_tdoa, pos_fdoa = self.parse_sensor_data(np.reshape(th_est['pos'], (self.num_dim, -1)))
        _, _, vel_fdoa = self.parse_sensor_data(np.reshape(th_est['vel'], (self.num_dim, -1)), vel_input=True)

        th_est['bias_aoa'] = bias_aoa
        th_est['bias_tdoa'] = bias_tdoa
        th_est['bias_fdoa'] = bias_fdoa
        th_est['pos_aoa'] = pos_aoa
        th_est['pos_tdoa'] = pos_tdoa
        th_est['pos_fdoa'] = pos_fdoa
        th_est['vel_fdoa'] = vel_fdoa

        return x_est, likelihood, th_grid, th_est

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
    def parse_subordinates(self):
        # Get dimensions and sizes from the subordinate objects, then compute sensor and measurement indices
        dims = self.get_attr_from_pss(attr='num_dim', concat_dim=0)
        if len(dims) > 1:
            if any(dims != dims[0]):
                raise ValueError("Dimension mismatch; all PassiveSurveillanceSystem objects must have the same "
                                 "number of physical dimensions.")

        self._num_dim = dims[0]
        self._num_sensors = np.sum(self.get_attr_from_pss(attr='num_sensors', concat_dim=0), axis=None)
        self._num_measurements = np.sum(self.get_attr_from_pss(attr='num_measurements', concat_dim=0), axis=None)

        self._aoa_sensor_idx = np.arange(self.num_aoa_sensors)
        self._tdoa_sensor_idx = np.arange(self.num_tdoa_sensors) + self.num_aoa_sensors
        self._fdoa_sensor_idx = np.arange(self.num_fdoa_sensors) + self.num_aoa_sensors + self.num_tdoa_sensors

        self._aoa_measurement_idx = np.arange(self.num_aoa_measurements)
        self._tdoa_measurement_idx = np.arange(self.num_tdoa_measurements) + self.num_aoa_measurements
        self._fdoa_measurement_idx = (np.arange(self.num_fdoa_measurements) + self.num_aoa_measurements
                                      + self.num_tdoa_measurements)

        return

    def parse_reference_indices(self)-> npt.NDArray[np.int64]:
        # Intuit reference indices from the components

        # First, we generate the test and reference index vectors
        test_idx_vec_aoa = np.arange(self.num_aoa_measurements)
        ref_idx_vec_aoa = np.nan * np.ones((self.num_aoa_measurements,))

        if self.tdoa is not None:
            test_idx_vec_tdoa = self.tdoa.test_idx_vec
            ref_idx_vec_tdoa = self.tdoa.ref_idx_vec
        else:
            test_idx_vec_tdoa = np.array([])
            ref_idx_vec_tdoa = np.array([])

        if self.fdoa is not None:
            test_idx_vec_fdoa = self.fdoa.test_idx_vec
            ref_idx_vec_fdoa = self.fdoa.ref_idx_vec
        else:
            test_idx_vec_fdoa = np.array([])
            ref_idx_vec_fdoa = np.array([])

        # Second, we assemble them into a single vector
        test_idx_vec = np.concatenate((test_idx_vec_aoa,
                                       self.num_aoa_measurements + test_idx_vec_tdoa,
                                       self.num_aoa_measurements + self.num_tdoa_sensors + test_idx_vec_fdoa), axis=0)
        ref_idx_vec = np.concatenate((ref_idx_vec_aoa,
                                      self.num_aoa_measurements + ref_idx_vec_tdoa,
                                      self.num_aoa_measurements + self.num_tdoa_sensors + ref_idx_vec_fdoa), axis=0)

        ref_idx = np.array([test_idx_vec, ref_idx_vec])
        return ref_idx # return

    def update_reference_indices(self):
        self.ref_idx = self.parse_reference_indices()
        # The number of measurements may have changed; update the indices
        self.parse_subordinates()
        return

    def parse_covariance_matrix(self)-> CovarianceMatrix:
        # Pull the covariance matrix from the components; we need the un-resampled version because
        # when we set it we will resample it.

        to_concat = []
        if self.aoa is not None:
            to_concat.append(self.aoa.cov)

        if self.tdoa is not None:
            to_concat.append(self.tdoa.cov_raw)

        if self.fdoa is not None:
            to_concat.append(self.fdoa.cov_raw)

        if any([t is None for t in to_concat]):
            # If any of the covariance matrices we got are None-type, then we can't
            # parse them
            new_cov = None
        else:
            # Use a block-diagonal of the provided matrices
            new_cov = CovarianceMatrix.block_diagonal(*to_concat)

        return new_cov

    def update_covariance_matrix(self, cov: CovarianceMatrix | npt.ArrayLike | None=None, do_resample: bool=True):
        if cov is None:
            # No covariance matrix specified, we'll parse the subordinate methods to make one
            super().update_covariance_matrix(cov=self.parse_covariance_matrix(), do_resample=False)
        else:
            # Call it directly, with the provided arguments
            super().update_covariance_matrix(cov=cov,do_resample=do_resample)
        return

    def parse_sensor_data(self, data: npt.ArrayLike, vel_input: bool=False):
        if data is None:
            # There's nothing to split; all three returns should be Nones
            return None, None, None

        # Possible configurations for data:
        #  2D; num_dim x num_sensors
        #      num_dim x self.fdoa.num_sensors (if vel_input=True)
        #  1D; num_sensors
        data = np.array(data)
        data_shape = data.shape

        # Initialize outputs
        data_aoa = None
        data_tdoa = None
        data_fdoa = None

        if len(data_shape) == 1: # 1D array
            if data_shape[0] == self.num_sensors:
                # Parse the AOA, TDOA, and FDOA sensor indices
                data_aoa = data[self.aoa_sensor_idx] if self.aoa is not None else None
                data_tdoa = data[self.tdoa_sensor_idx] if self.tdoa is not None else None
                data_fdoa = data[self.fdoa_sensor_idx] if self.fdoa is not None else None
            elif vel_input and data_shape[0] == self.num_fdoa_sensors:
                # It's a velocity input and is sized one-per-sensor
                data_fdoa = data
            else:
                raise ValueError('Unexpected number of entries in 1D data input.')
        elif len(data_shape) >= 2: # 2D array
            # The assumption with a 2D array is that the first dimension is spatial (x/y/z) and the second iterates
            # across the sensors
            if data_shape[1] == self.num_sensors:
                # Parse the AOA, TDOA, and FDOA sensor indices
                data_aoa = data[:,self.aoa_sensor_idx] if self.aoa is not None else None
                data_tdoa = data[:,self.tdoa_sensor_idx] if self.tdoa is not None else None
                data_fdoa = data[:,self.fdoa_sensor_idx] if self.fdoa is not None else None
            elif vel_input and data_shape[1] == self.num_fdoa_sensors:
                data_fdoa = data
            else:
                raise ValueError('Unexpected number of columns in 2D data input.')

        return data_aoa, data_tdoa, data_fdoa

    def parse_measurement_data(self, data: npt.ArrayLike):
        if data is None:
            # There's nothing to split; all three returns should be Nones
            return None, None, None

        # Assume that one of the dimensions matches the expected number of measurements; split along that dimension
        data_shape = np.shape(data)
        matching_dims = np.nonzero(np.equal(data_shape, self.num_measurements))[0]  # the comparison is 1D
        assert matching_dims.size > 0, 'Unexpected data shape; at least one dimension must match the number of measurements.'

        # Break the array into 3 sections, with breakpoints at the end of the number of AOA and AOA+TDOA measurements
        slice_index = [self.num_aoa_measurements, self.num_aoa_measurements + self.num_tdoa_measurements]
        data_split = np.split(data, slice_index, axis=matching_dims[0]) # use the first matching dimensions

        # Parse the AOA, TDOA, and FDOA sensor indices
        data_aoa = data_split[0] if self.aoa is not None else None
        data_tdoa = data_split[1] if self.tdoa is not None else None
        data_fdoa = data_split[2] if self.fdoa is not None else None

        return data_aoa, data_tdoa, data_fdoa
