import numpy as np
import numpy.typing as npt
from scipy.linalg import block_diag

from ewgeo.fdoa import FDOAPassiveSurveillanceSystem
from ewgeo.tdoa import TDOAPassiveSurveillanceSystem
from ewgeo.triang import DirectionFinder
from ewgeo.utils import parse_reference_sensor, safe_2d_shape, SearchSpace
from ewgeo.utils.covariance import CovarianceMatrix
from ewgeo.utils.perf import compute_crlb_gaussian
from ewgeo.utils.system import DifferencePSS


class HybridPassiveSurveillanceSystem(DifferencePSS):
    bias = None
    aoa: DirectionFinder | None = None
    tdoa: TDOAPassiveSurveillanceSystem | None = None
    fdoa: FDOAPassiveSurveillanceSystem | None = None

    aoa_sensor_idx = None
    tdoa_sensor_idx = None
    fdoa_sensor_idx = None

    aoa_measurement_idx = None
    tdoa_measurement_idx = None
    fdoa_measurement_idx = None

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

        if ref_idx is None:
            ref_idx = self.parse_reference_indices()

        if cov is None:
            cov = self.parse_covariance_matrix()

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
        arr = [getattr(pss, attr) for pss in [self.aoa, self.tdoa, self.fdoa]
               if pss is not None and getattr(pss, attr) is not None] # ignore any empty PSS fields,
        if not arr:
            return None
        else:
            return np.concatenate(arr, axis=concat_dim)

    @property
    def pos(self):
        return self.get_attr_from_pss('pos', concat_dim=1)

    @pos.setter
    def pos(self, x: npt.ArrayLike):
        desired_dims = (self.num_dim, self.num_sensors)
        if np.any(x.shape != desired_dims):
            raise ValueError(f'Input array shape {x.shape} does not match desired shape {desired_dims}.')
        if self.aoa:
            self.aoa.pos = x[:, self.aoa_sensor_idx]
        if self.tdoa:
            self.tdoa.pos = x[:, self.tdoa_sensor_idx]
        if self.fdoa:
            self.fdoa.pos = x[:, self.fdoa_sensor_idx]

    @property
    def vel(self):
        return self.get_attr_from_pss('vel', concat_dim=1)

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
            if np.shape(v) == desired_dims_partial:
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
    def aoa_sensor_idx(self)-> npt.NDArray[np.int64]:
        return np.arange(self.num_aoa_sensors)

    @property
    def tdoa_sensor_idx(self)-> npt.NDArray[np.int64]:
        return np.arange(self.num_tdoa_sensors) + self.num_aoa_sensors

    @property
    def fdoa_sensor_idx(self)-> npt.NDArray[np.int64]:
        return np.arange(self.num_fdoa_sensors) + self.num_aoa_sensors + self.num_tdoa_sensors

    @property
    def aoa_measurement_idx(self)-> npt.NDArray[np.int64]:
        return np.arange(self.num_aoa_measurements)

    @property
    def tdoa_measurement_idx(self)-> npt.NDArray[np.int64]:
        return np.arange(self.num_tdoa_measurements) + self.num_aoa_measurements

    @property
    def fdoa_measurement_idx(self)-> npt.NDArray[np.int64]:
        return np.arange(self.num_fdoa_measurements) + self.num_tdoa_measurements + self.num_aoa_measurements

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
            sensor_pos_search_epsilon.append(pss.default_sensor_pos_search_epsilon * np.ones([self.num_dim, pss.num_sensors]))
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
    def default_sensor_vel_search_epsilon(self) -> npt.NDArray[np.float64]:
        if self.fdoa is not None:
            return self.fdoa.default_sensor_vel_search_epsilon
        else:
            return None

    @property
    def default_sensor_vel_search_size(self) -> npt.NDArray[np.int64]:
        if self.fdoa is not None:
            return np.array(self.fdoa.default_sensor_vel_search_size, dtype=int)
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
            result = result + self.aoa.log_likelihood(x_source=x_source, zeta=z_aoa, x_sensor=x_aoa, bias=b_aoa, **kwargs)
        if self.tdoa is not None:
            result = result + self.tdoa.log_likelihood(x_source=x_source, zeta=z_tdoa, x_sensor=x_tdoa, bias=b_tdoa, **kwargs)
        if self.fdoa is not None:
            result = result + self.fdoa.log_likelihood(x_source=x_source, zeta=z_fdoa, x_sensor=x_fdoa,
                                                       v_sensor=v_fdoa, v_source=v_source, bias=b_fdoa, **kwargs)

        return result

    def log_likelihood_uncertainty(self, zeta: npt.ArrayLike, theta: npt.ArrayLike, **kwargs)-> npt.NDArray:
        zeta_aoa, zeta_tdoa, zeta_fdoa = self.parse_measurement_data(zeta)
        theta_aoa, theta_tdoa, theta_fdoa = self.parse_uncertainty_data(theta)

        result = 0
        if self.aoa is not None:
            result = result + self.aoa.log_likelihood_uncertainty(zeta=zeta_aoa, theta=theta_aoa, **kwargs)
        if self.tdoa is not None:
            result = result + self.tdoa.log_likelihood_uncertainty(zeta=zeta_tdoa, theta=theta_tdoa, **kwargs)
        if self.fdoa is not None:
            result = result + self.fdoa.log_likelihood_uncertainty(zeta=zeta_fdoa, theta=theta_fdoa, **kwargs)

        return result

    def grad_x(self, x_source: npt.ArrayLike)-> npt.NDArray:
        to_concat = []
        if self.aoa is not None:
            to_concat.append(self.aoa.grad_x(x_source=x_source))
        if self.tdoa is not None:
            to_concat.append(self.tdoa.grad_x(x_source=x_source))
        if self.fdoa is not None:
            # Replicate the AOA and TDOA gradients to reflect gradients w.r.t. pos and vel
            to_concat = [np.concatenate((g, np.zeros_like(g)), axis=0) for g in to_concat]
            to_concat.append(self.fdoa.grad_x(x_source=x_source))

        return np.concatenate(to_concat, axis=1)

    def grad_bias(self, x_source: npt.ArrayLike)-> npt.NDArray:
        to_concat = []
        if self.aoa is not None:
            to_concat.append(self.aoa.grad_bias(x_source=x_source))
        if self.tdoa is not None:
            to_concat.append(self.tdoa.grad_bias(x_source=x_source))
        if self.fdoa is not None:
            to_concat.append(self.fdoa.grad_bias(x_source=x_source))

        _, n_source = safe_2d_shape(x_source)
        if n_source <= 1:
            # There is only one source, combine the gradients with a block diagonal across axes 0 and 1
            grad = block_diag(to_concat)
        else:
            # The individual gradients are 3D, but block_diag only works on 2D, let's do some reshaping.
            # We need to move the third axis to the front
            gradients_reshape = [np.moveaxis(x, -1, 0) for x in to_concat]

            # Now we can use list comprehension to call block_diag on each in turn
            res = [block_diag(*arrays) for arrays in zip(*gradients_reshape)]

            # This is now a list of length n_source, where each entry is a block-diagonal jacobian matrix at that source
            # position. Convert back to an array and rearrange the axes
            grad = np.moveaxis(np.asarray(res), 0, -1)  # Move the first axis (n_source) back to the end.

        return grad

    def grad_sensor_pos(self, x_source: npt.ArrayLike)-> npt.NDArray:
        to_concat = []
        if self.aoa is not None:
            to_concat.append(self.aoa.grad_sensor_pos(x_source=x_source))
        if self.tdoa is not None:
            to_concat.append(self.tdoa.grad_sensor_pos(x_source=x_source))
        if self.fdoa is not None:
            # First, manipulate all the existing gradients to add in the gradient w.r.t. velocity (zeros)
            to_concat = [np.concatenate((g, np.zeros_like(g)), axis=0) for g in to_concat]

            to_concat.append(self.fdoa.grad_sensor_pos(x_source=x_source))

        return np.concatenate(to_concat, axis=1)

    ## ============================================================================================================== ##
    ## Solver Methods
    ##
    ## These methods handle the interface to solvers
    ## ============================================================================================================== ##
    def max_likelihood_uncertainty(self, zeta: npt.ArrayLike, search_space:SearchSpace,
                                   do_sensor_bias: bool=False, do_sensor_pos: bool=False, do_sensor_vel: bool=False,
                                   **kwargs)-> tuple[npt.NDArray, npt.NDArray, dict]:

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

    def get_uncertainty_search_space(self, do_source_vel: bool=False, do_sensor_bias: bool=False,
                                     do_sensor_pos: bool=False, do_sensor_vel: bool=False)-> dict:
        """
        Define and return a dict describing the uncertainty search vector
        """

        # Get Search Space for the Components
        aoa_search = self.aoa.get_uncertainty_search_space(do_source_vel=do_source_vel, do_sensor_bias=do_sensor_bias,
                                                           do_sensor_pos=do_sensor_pos) if self.aoa is not None \
            else None
        tdoa_search = self.tdoa.get_uncertainty_search_space(do_source_vel=do_source_vel, do_sensor_bias=do_sensor_bias,
                                                             do_sensor_pos=do_sensor_pos) if self.tdoa is not None \
            else None
        fdoa_search = self.fdoa.get_uncertainty_search_space(do_source_vel=do_source_vel, do_sensor_bias=do_sensor_bias,
                                                             do_sensor_pos=do_sensor_pos, do_sensor_vel=do_sensor_vel) \
            if self.fdoa is not None else None

        # Parse the component search spaces
        source_indices = aoa_search['source_idx'] if aoa_search is not None \
            else tdoa_search['source_idx'] if tdoa_search is not None else fdoa_search['source_idx']
        num_source_indices = len(source_indices)

        num_aoa_bias = aoa_search['num_bias_idx'] if aoa_search is not None else 0
        num_tdoa_bias = tdoa_search['num_bias_idx'] if tdoa_search is not None else 0
        num_fdoa_bias = fdoa_search['num_bias_idx'] if fdoa_search is not None else 0
        num_bias_indices = num_aoa_bias + num_tdoa_bias + num_fdoa_bias

        num_aoa_pos = aoa_search['num_pos_idx'] if aoa_search is not None else 0
        num_tdoa_pos = tdoa_search['num_pos_idx'] if tdoa_search is not None else 0
        num_fdoa_pos = fdoa_search['num_pos_idx'] if fdoa_search is not None else 0
        num_pos_indices = num_aoa_pos + num_tdoa_pos + num_fdoa_pos

        num_vel_indices = fdoa_search['num_vel_idx'] if fdoa_search is not None else 0

        # Construct indices
        bias_indices = num_source_indices + np.arange(num_bias_indices)
        pos_indices = num_source_indices + num_bias_indices + np.arange(num_pos_indices)
        vel_indices = num_source_indices + num_bias_indices + num_vel_indices + np.arange(num_vel_indices)

        # Assemble the dict and return
        return {'source_idx': source_indices,
                'bias_idx': bias_indices,
                'sensor_pos_idx': pos_indices,
                'sensor_vel_idx': vel_indices,
                'num_source_idx': num_source_indices,
                'num_bias_idx': num_bias_indices,
                'num_pos_idx': num_pos_indices,
                'num_vel_idx': num_vel_indices}

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
    def parse_reference_indices(self)-> npt.NDArray:
        # Intuit reference indices from the components

        # First, we generate the test and reference index vectors
        test_idx_vec_aoa = np.arange(self.num_aoa_measurements)
        ref_idx_vec_aoa = np.nan * np.ones((self.num_aoa_measurements,))

        if self.tdoa is not None:
            test_idx_vec_tdoa, ref_idx_vec_tdoa = parse_reference_sensor(self.tdoa.ref_idx, self.num_tdoa_sensors)
        else:
            test_idx_vec_tdoa = np.array([])
            ref_idx_vec_tdoa = np.array([])

        if self.fdoa is not None:
            test_idx_vec_fdoa, ref_idx_vec_fdoa = parse_reference_sensor(self.fdoa.ref_idx, self.num_fdoa_sensors)
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

    def parse_covariance_matrix(self)-> CovarianceMatrix:
        # Pull the covariance matrix from the components; we need the unresampled version because
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
        data_shape = np.shape(data)

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
        elif len(data_shape) == 2: # 2D array
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

    def parse_uncertainty_data(self, data: npt.ArrayLike):
        # Uncertainty parameters take the form:
        #   [x_source, v_source, bias, x_sensor.ravel(), v_sensor.ravel()]
        # Expanded, to show the components, they are:
        #   source pos (n_dim x 1)
        #   source vel (n_dim x 1)
        #   aoa bias
        #   tdoa bias
        #   fdoa bias
        #   aoa sensor pos
        #   tdoa sensor pos
        #   fdoa sensor pos
        #   fdoa sensor vel
        #
        # This function parses it into three separate uncertainty vectors, to be passed to
        # self.aoa, self.tdoa, and self.fdoa

        source_pos_ind = np.arange(self.num_dim)
        source_vel_ind = self.num_dim + np.arange(self.num_dim) if self.vel is not None else np.array([])

        bias_start_ind = self.num_dim if self.vel is None else 2*self.num_dim
        aoa_bias_ind = bias_start_ind + np.arange(self.num_aoa_measurements)
        tdoa_bias_ind = bias_start_ind + self.num_aoa_measurements + np.arange(self.num_tdoa_measurements)
        fdoa_bias_ind = (bias_start_ind + self.num_aoa_measurements + self.num_tdoa_measurements
                         + np.arange(self.num_fdoa_measurements))

        sensor_pos_start_ind = bias_start_ind + self.num_measurements
        aoa_pos_ind = sensor_pos_start_ind + np.arange(self.num_aoa_sensors*self.num_dim)
        tdoa_pos_ind = sensor_pos_start_ind + self.num_dim*self.num_aoa_sensors + np.arange(self.num_tdoa_sensors*self.num_dim)
        fdoa_pos_ind = (sensor_pos_start_ind + self.num_dim*(self.num_aoa_sensors+self.num_tdoa_sensors)
                        + np.arange(self.num_fdoa_sensors*self.num_dim))
        fdoa_vel_ind = (sensor_pos_start_ind
                        + self.num_dim*(self.num_aoa_sensors+self.num_tdoa_sensors+self.num_fdoa_sensors)
                        + np.arange(self.num_fdoa_sensors*self.num_dim))

        if self.aoa is None:
            theta_aoa = None
        else:
            theta_aoa = data[np.concatenate((source_pos_ind, aoa_bias_ind, aoa_pos_ind), axis=None)]

        if self.tdoa is None:
            theta_tdoa = None
        else:
            theta_tdoa = data[np.concatenate((source_pos_ind, tdoa_bias_ind, tdoa_pos_ind), axis=None)]

        if self.fdoa is None:
            theta_fdoa = None
        else:
            theta_fdoa = data[np.concatenate((source_pos_ind, source_vel_ind, fdoa_bias_ind, fdoa_pos_ind, fdoa_vel_ind), axis=None)]

        return theta_aoa, theta_tdoa, theta_fdoa

    def parse_source_pos_vel(self, pos_vel: npt.ArrayLike, default_vel: npt.ArrayLike):
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
