import numpy as np
from scipy.linalg import block_diag

import utils
from utils import SearchSpace
from utils.system import DifferencePSS
from utils.covariance import CovarianceMatrix
from triang import DirectionFinder
from tdoa import TDOAPassiveSurveillanceSystem
from fdoa import FDOAPassiveSurveillanceSystem


class HybridPassiveSurveillanceSystem(DifferencePSS):
    bias = None
    aoa: DirectionFinder or None = None
    tdoa: TDOAPassiveSurveillanceSystem or None = None
    fdoa: FDOAPassiveSurveillanceSystem or None = None

    num_aoa_sensors: int = 0
    num_tdoa_sensors: int = 0
    num_fdoa_sensors: int = 0

    num_aoa_measurements: int = 0
    num_tdoa_measurements: int = 0
    num_fdoa_measurements: int = 0

    aoa_sensor_idx = None
    tdoa_sensor_idx = None
    fdoa_sensor_idx = None

    aoa_measurement_idx = None
    tdoa_measurement_idx = None
    fdoa_measurement_idx = None

    def __init__(self, cov=None, aoa=None, tdoa=None, fdoa=None, ref_idx=None, **kwargs):
        # Parse the provided sensor types
        x_arr = []
        if aoa is not None:
            x_arr.append(aoa.pos)
            self.aoa = aoa
            self.num_aoa_sensors = self.aoa.num_sensors
            self.num_aoa_measurements = self.aoa.num_measurements

        if tdoa is not None:
            x_arr.append(tdoa.pos)
            self.tdoa = tdoa
            self.num_tdoa_sensors = self.tdoa.num_sensors
            self.num_tdoa_measurements = self.tdoa.num_measurements

        if fdoa is not None:
            x_arr.append(fdoa.pos)
            self.fdoa = fdoa
            self.num_fdoa_sensors = self.fdoa.num_sensors
            self.num_fdoa_measurements = self.fdoa.num_measurements

        assert len(x_arr)>0, 'Error initializing HybridPSS system; at least one type of subordinate PSS system must be supplied.'

        x = np.concatenate(x_arr, axis=1)

        if ref_idx is None:
            ref_idx = self.parse_reference_indices()

        if cov is None:
            cov = self.parse_covariance_matrix()

        # Initiate the superclass
        super().__init__(x, cov, ref_idx, **kwargs)

        # Overwrite the numbers of sensors/measurements and generate indices for referencing
        # from combined position, measurement, and covariance matrices.
        self.num_sensors = self.num_aoa_sensors + self.num_tdoa_sensors + self.num_fdoa_sensors
        self.num_measurements = self.num_aoa_measurements + self.num_tdoa_measurements + self.num_fdoa_measurements
        self.aoa_sensor_idx = np.arange(self.num_aoa_sensors)
        self.tdoa_sensor_idx = np.arange(self.num_tdoa_sensors) + self.num_aoa_sensors
        self.fdoa_sensor_idx = np.arange(self.num_fdoa_sensors) + self.num_aoa_sensors + self.num_tdoa_sensors
        self.aoa_measurement_idx = np.arange(self.num_aoa_measurements)
        self.tdoa_measurement_idx = np.arange(self.num_tdoa_measurements) + self.num_aoa_measurements
        self.fdoa_measurement_idx = (np.arange(self.num_fdoa_measurements) + self.num_tdoa_measurements
                                     + self.num_fdoa_measurements)

        # Overwrite the uncertainty search defaults
        bias_search_epsilon = []
        bias_search_size = []
        sensor_pos_search_epsilon = []
        sensor_pos_search_size = []
        for pss in [self.aoa, self.tdoa, self.fdoa]:
            if pss is None: continue

            # Add defaults from this PSS
            bias_search_epsilon.append([pss.default_bias_search_epsilon] * pss.num_measurements)
            bias_search_size.append([pss.default_bias_search_size] * pss.num_measurements)
            sensor_pos_search_epsilon.append([pss.default_sensor_pos_search_epsilon] * pss.num_sensors * self.num_dim)
            sensor_pos_search_size.append([pss.default_sensor_pos_search_size] * pss.num_sensors * self.num_dim)

        self._bias_search_epsilon = np.concatenate(bias_search_epsilon, axis=None)
        self._bias_search_size = np.concatenate(bias_search_size, axis=None)
        self._sensor_pos_search_epsilon = np.concatenate(sensor_pos_search_epsilon, axis=None)
        self._sensor_pos_search_size = np.concatenate(sensor_pos_search_size, axis=None)
        if self.fdoa is not None:
            self._sensor_vel_search_epsilon = self.fdoa.default_sensor_vel_search_epsilon
            self._sensor_vel_search_size = self.fdoa.default_sensor_vel_search_size

        return

    ## ============================================================================================================== ##
    ## Model Methods
    ##
    ## These methods handle the physical model for a TDOA-based PSS, and are just wrappers for the static
    ## functions defined in model.py
    ## ============================================================================================================== ##
    def measurement(self, x_source, x_sensor=None, bias=None, v_sensor=None, v_source=None):
        # Call the three measurement models and concatenate the results along the first axis

        # Break apart the sensor position, velocity, and bias measurement inputs into their AOA, TDOA, and FDOA
        # components
        x_aoa, x_tdoa, x_fdoa = self.parse_sensor_data(x_sensor)
        _, _, v_fdoa = self.parse_sensor_data(v_sensor, vel_input=True)
        b_aoa, b_tdoa, b_fdoa = self.parse_measurement_data(bias)

        # Parse source position and velocity
        if v_source is None:
            # It might be passed as a single input under x_source with 2*num_dim rows
            x_source, v_source = self.parse_source_pos_vel(x_source, np.zeros_like(x_source))

        # Call component models
        to_concat = []
        if self.aoa is not None:
            to_concat.append(self.aoa.measurement(x_source, x_sensor=x_aoa, bias=b_aoa))

        if self.tdoa is not None:
            to_concat.append(self.tdoa.measurement(x_source, x_sensor=x_tdoa, bias=b_aoa))

        if self.fdoa is not None:
            to_concat.append(self.fdoa.measurement(x_source, x_sensor=x_fdoa, v_sensor=v_fdoa, v_source=v_fdoa,
                                                   bias=b_aoa))

        z = np.concatenate(to_concat, axis=0)
        return z

    def jacobian(self, x_source, v_source=None, x_sensor=None, v_sensor=None):
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

    def jacobian_uncertainty(self, x_source, v_source=None, **kwargs):
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

    def log_likelihood(self, x_source, zeta, x_sensor=None, bias=None, v_sensor=None, v_source=None):
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
            result = result + self.aoa.log_likelihood(x_source=x_source, zeta=z_aoa, x_sensor=x_aoa, bias=b_aoa)
        if self.tdoa is not None:
            result = result + self.tdoa.log_likelihood(x_source=x_source, zeta=z_tdoa, x_sensor=x_tdoa, bias=b_tdoa)
        if self.fdoa is not None:
            result = result + self.fdoa.log_likelihood(x_source=x_source, zeta=z_fdoa, x_sensor=x_fdoa,
                                                       v_sensor=v_fdoa, v_source=v_source, bias=b_fdoa)

        return result

    def log_likelihood_uncertainty(self, zeta, theta, **kwargs):
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

    def grad_x(self, x_source):
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

    def grad_bias(self, x_source):
        to_concat = []
        if self.aoa is not None:
            to_concat.append(self.aoa.grad_bias(x_source=x_source))
        if self.tdoa is not None:
            to_concat.append(self.tdoa.grad_bias(x_source=x_source))
        if self.fdoa is not None:
            to_concat.append(self.fdoa.grad_bias(x_source=x_source))

        _, n_source = utils.safe_2d_shape(x_source)
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
            # position. Convert back to an ndarray and rearrange the axes
            grad = np.moveaxis(np.asarray(res), 0, -1)  # Move the first axis (n_source) back to the end.

        return grad

    def grad_sensor_pos(self, x_source):
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
    def max_likelihood(self, zeta, search_space: SearchSpace, cal_data:dict=None, **kwargs):
        # Perform sensor calibration
        if cal_data is not None:
            x_sensor, v_sensor, bias = self.sensor_calibration(*cal_data)
        else:
            x_sensor, v_sensor, bias = self.pos, self.vel, self.bias

        # Likelihood function for ML Solvers
        def ell(pos_vel):
            # Determine if the input is position only, or position & velocity
            this_pos, this_vel = self.parse_source_pos_vel(pos_vel, np.zeros_like(pos_vel))
            return self.log_likelihood(x_sensor=x_sensor, v_sensor=v_sensor,
                                       zeta=zeta, x_source=this_pos, v_source=this_vel)

        # Call the util function
        x_est, likelihood, x_grid = utils.solvers.ml_solver(ell=ell, search_space=search_space, **kwargs)

        return x_est, likelihood, x_grid

    def max_likelihood_uncertainty(self, zeta, search_space:SearchSpace,
                                   do_sensor_bias=False, do_sensor_pos=False, do_sensor_vel=False,
                                   **kwargs):

        # Call the super class to do the search, then re-parse the results
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

    def gradient_descent(self, zeta, x_init, cal_data: dict=None, **kwargs):
        # Perform sensor calibration
        if cal_data is not None:
            x_sensor, v_sensor, bias = self.sensor_calibration(*cal_data)
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
            x_sensor, v_sensor, bias = self.sensor_calibration(*cal_data)
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
        x_est, x_full = utils.solvers.ls_solver(zeta=y, jacobian=this_jacobian, cov=self.cov, x_init=x_init, **kwargs)

        return x_est, x_full


    def bestfix(self, zeta, search_space: SearchSpace, pdf_type=None, cal_data:dict=None):
        x_sensor, v_sensor, bias = self.sensor_calibration(*cal_data)

        # Generate the PDF
        def measurement(pos_vel):
            this_pos, this_vel = self.parse_source_pos_vel(pos_vel, np.zeros_like(pos_vel))
            return self.measurement(x_source=this_pos, v_source=this_vel, x_sensor=x_sensor, v_sensor=v_sensor,
                                    bias=bias)

        pdfs = utils.make_pdfs(measurement, zeta, pdf_type, self.cov.cov)

        # Call the util function
        x_est, likelihood, x_grid = utils.solvers.bestfix(pdfs, search_space)

        return x_est, likelihood, x_grid


    # def sensor_calibration(self, zeta_cal, x_cal, v_cal=None, pos_search: dict=None, vel_search: dict=None,
    #                        bias_search: dict=None):
        # # Calibrate each sensor component independently
        # zeta_aoa, zeta_tdoa, zeta_fdoa = self.parse_measurement_data(zeta_cal)
        #
        # x_sensor_list = []
        # b_sensor_list = []
        #
        # if self.aoa is not None:
        #     x_aoa, b_aoa = self.aoa.sensor_calibration(zeta_aoa, x_cal, pos_search, bias_search)
        #     x_sensor_list.append(x_aoa)
        #     b_sensor_list.append(b_aoa)
        # if self.tdoa is not None:
        #     x_tdoa, b_tdoa = self.tdoa.sensor_calibration(zeta_tdoa, x_cal, pos_search, bias_search)
        #     x_sensor_list.append(x_tdoa)
        #     b_sensor_list.append(b_tdoa)
        # if self.fdoa is not None:
        #     x_fdoa, v_fdoa, b_fdoa = self.fdoa.sensor_calibration(zeta_fdoa, x_cal, pos_search, vel_search, bias_search)
        #     x_sensor_list.append(x_fdoa)
        #     b_sensor_list.append(b_fdoa)
        # else:
        #     v_fdoa = None
        #
        # return np.concatenate(x_sensor_list, axis=0), v_fdoa, np.concatenate(b_sensor_list, axis=0)

    def get_uncertainty_search_space(self, do_source_vel=False, do_sensor_bias=False, do_sensor_pos=False,
                                     do_sensor_vel=False):
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
        fdoa_search = self.aoa.get_uncertainty_search_space(do_source_vel=do_source_vel, do_sensor_bias=do_sensor_bias,
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
    def compute_crlb(self, x_source, v_source=None, **kwargs):
        """
        If x_source has 2*self.num_dim rows (position and velocity), then the CRLB will be computed across both sets of
        unknowns.

        If x_source has self.num_dim rows, then it is just position, and the CRLB will be computed across only those
        uncertainties.
        """
        def this_jacobian(pos_vel):
            this_pos, this_vel = self.parse_source_pos_vel(pos_vel, default_vel=v_source)
            n_dim, _ = utils.safe_2d_shape(pos_vel) # record the number of output dimensions called for
            j = self.jacobian(x_source=this_pos, v_source=this_vel, x_sensor=self.pos, v_sensor=self.vel)
            # Jacobian returns 2*n_dim rows; first the jacobian w.r.t. position, then velocity. Optionally
            # excise just the position portion
            return j[:n_dim]

        return utils.perf.compute_crlb_gaussian(x_source=x_source, jacobian=this_jacobian, cov=self.cov,
                                                **kwargs)

    ## ============================================================================================================== ##
    ## Helper Methods
    ##
    ## These are generic utility functions that are unique to this class
    ## ============================================================================================================== ##
    def parse_reference_indices(self):
        # Intuit reference indices from the components

        # First, we generate the test and reference index vectors
        test_idx_vec_aoa = np.arange(self.num_aoa_measurements)
        ref_idx_vec_aoa = np.nan * np.ones((self.num_aoa_measurements,))

        if self.tdoa is not None:
            test_idx_vec_tdoa, ref_idx_vec_tdoa = utils.parse_reference_sensor(self.tdoa.ref_idx, self.num_tdoa_sensors)
        else:
            test_idx_vec_tdoa = np.array([])
            ref_idx_vec_tdoa = np.array([])

        if self.fdoa is not None:
            test_idx_vec_fdoa, ref_idx_vec_fdoa = utils.parse_reference_sensor(self.fdoa.ref_idx, self.num_fdoa_sensors)
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

        ref_idx = np.array([test_idx_vec, ref_idx_vec]) # store in object's field
        return ref_idx # return as well

    def update_reference_indices(self):
        self.ref_idx = self.parse_reference_indices()

    def parse_covariance_matrix(self):
        # Pull the covariance matrix from the components; we need the unresampled version because
        # when we set it we will resample it.
        # ToDo: if self.cov is not None, raise a warning that this will overwrite the current covariance matrix

        to_concat = []
        if self.aoa is not None:
            to_concat.append(self.aoa.cov)

        if self.tdoa is not None:
            to_concat.append(self.tdoa.cov_raw)

        if self.fdoa is not None:
            to_concat.append(self.fdoa.cov_raw)

        new_cov = CovarianceMatrix.block_diagonal(*to_concat)

        return new_cov

    def update_covariance_matrix(self):
        self.cov = self.parse_covariance_matrix()
        return

    def parse_sensor_data(self, data, vel_input=False):
        # todo: implement the vel_input flag; see fdoa/system.py for example
        if data is not None:
            # Split apart a vector of sensor data, must have length equal to self.num_sensors
            # assert len(data) == self.num_sensors or (vel_input and len(data)==self.num_fdoa_sensors), "Unable to parse sensor data; unexpected size."

            # todo: make this cleaner
            if len(data.shape) == 1:
                # Parse the AOA, TDOA, and FDOA sensor indices
                data_aoa = data[self.aoa_sensor_idx]
                data_tdoa = data[self.tdoa_sensor_idx]
                data_fdoa = data[self.fdoa_sensor_idx]
            else:
                # Parse the AOA, TDOA, and FDOA sensor indices
                data_aoa = data[:,self.aoa_sensor_idx]
                data_tdoa = data[:,self.tdoa_sensor_idx]
                data_fdoa = data[:,self.fdoa_sensor_idx]
        else:
            data_aoa = None
            data_tdoa = None
            data_fdoa = None

        return data_aoa, data_tdoa, data_fdoa

    def parse_measurement_data(self, data):
        if data is not None:
            # Split apart a vector of measurement data, must have length equal to self.num_measurements
            assert len(data) == self.num_measurements, "Unable to parse sensor data; unexpected size."

            # Parse the AOA, TDOA, and FDOA sensor indices
            data_aoa = data[self.aoa_measurement_idx]
            data_tdoa = data[self.tdoa_measurement_idx]
            data_fdoa = data[self.fdoa_measurement_idx]
        else:
            data_aoa = None
            data_tdoa = None
            data_fdoa = None

        return data_aoa, data_tdoa, data_fdoa

    def parse_uncertainty_data(self, data):
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
