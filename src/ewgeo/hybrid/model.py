import numpy as np
import numpy.typing as npt
from scipy.linalg import block_diag

import ewgeo.fdoa as fdoa
import ewgeo.tdoa as tdoa
import ewgeo.triang as triang
from ewgeo.utils import modulo2pi, SearchSpace, broadcast_backwards
from ewgeo.utils.covariance import CovarianceMatrix


def measurement(x_source: npt.ArrayLike, 
                x_aoa: npt.ArrayLike=None, 
                x_tdoa: npt.ArrayLike=None, 
                x_fdoa: npt.ArrayLike=None, 
                v_fdoa: npt.ArrayLike=None, 
                v_source: npt.ArrayLike | None=None,
                do_2d_aoa: bool=False, 
                tdoa_ref_idx=None, fdoa_ref_idx=None, 
                angle_bias: npt.ArrayLike | None=None,
                range_bias: npt.ArrayLike | None=None,
                range_rate_bias: npt.ArrayLike | None=None):
    """
    Computes hybrid measurements, for AOA, TDOA, and FDOA sensors.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    10 March 2021

    :param x_source: nDim x n_source array of source positions
    :param x_aoa: nDim x nAOA array of sensor positions
    :param x_tdoa: nDim x nTDOA array of TDOA sensor positions
    :param x_fdoa: nDim x nFDOA array of FDOA sensor positions
    :param v_fdoa: nDim x nFDOA array of FDOA sensor velocities
    :param v_source: nDim x 1 source velocity; assumed stationary if left blank
    :param do_2d_aoa: Optional boolean parameter specifying whether 1D (az-only) or 2D (az/el) AOA is being performed
    :param tdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for TDOA
    :param fdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for FDOA
    :param angle_bias: (Optional) nAOA x 1 (or nAOA x 2, if do_2d_aoa=True) vector of angle bias terms [default=None]
    :param range_bias: (Optional) nTDOA x 1 vector of range bias terms (time bias * c) [default=None]
    :param range_rate_bias: (Optional) nFDOA x 1 vector of range-rate bias terms (freq bias * c/f0) [default=None]
    :return zeta: nAoa + nTDOA + nFDOA - 2 x nSource array of measurements
    """

    # Construct component measurements
    to_concat = []
    if x_aoa is not None:
        z_a = triang.model.measurement(x_sensor=x_aoa, x_source=x_source, do_2d_aoa=do_2d_aoa, bias=angle_bias)
        to_concat.append(z_a)
    if x_tdoa is not None:
        z_t = tdoa.model.measurement(x_sensor=x_tdoa, x_source=x_source, ref_idx=tdoa_ref_idx, bias=range_bias)
        to_concat.append(z_t)
    if x_fdoa is not None:
        z_f = fdoa.model.measurement(x_sensor=x_fdoa, v_sensor=v_fdoa, 
                                     x_source=x_source, v_source=v_source,
                                     ref_idx=fdoa_ref_idx, bias=range_rate_bias)
        to_concat.append(z_f)

    # Combine into a single data vector
    z = np.concatenate(to_concat, axis=0)

    return z


def jacobian(x_source: npt.ArrayLike, 
             x_aoa: npt.ArrayLike=None,
             x_tdoa: npt.ArrayLike=None,
             x_fdoa: npt.ArrayLike=None,
             v_fdoa: npt.ArrayLike=None,
             v_source: npt.ArrayLike | None=None,
             do_2d_aoa: bool=False,
             tdoa_ref_idx=None,
             fdoa_ref_idx=None):
    """
    # Returns the Jacobian matrix for hybrid set of AOA, TDOA, and FDOA
    # measurements.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    10 March 2021

    :param x_source: nDim x n_source array of source positions
    :param x_aoa: nDim x nAOA array of sensor positions
    :param x_tdoa: nDim x nTDOA array of TDOA sensor positions
    :param x_fdoa: nDim x nFDOA array of FDOA sensor positions
    :param v_fdoa: nDim x nFDOA array of FDOA sensor velocities
    :param v_source: nDim x 1 source velocity; assumed stationary if left blank
    :param do_2d_aoa: Optional boolean parameter specifying whether 1D (az-only) or 2D (az/el) AOA is being performed
    :param tdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for TDOA
    :param fdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for FDOA
    :return: n_dim x nMeasurement x n_source Jacobian matrices, one for each candidate source position
    """

    # Find out how many source points are requested
    shp = np.shape(x_source)
    n_dim = shp[0] if len(shp) > 0 else 1
    n_source = shp[1] if len(shp) > 1 else 1
    if n_source > 1:
        empty_dims = (n_dim, 0, n_source)
    else:
        empty_dims = (n_dim, 0)

    # Compute Jacobian for AOA measurements
    if x_aoa is not None:
        j_aoa = triang.model.jacobian(x_aoa, x_source, do_2d_aoa=do_2d_aoa)
    else:
        j_aoa = np.zeros(shape=empty_dims)

    # Compute Jacobian for TDOA measurements
    if x_tdoa is not None:
        j_tdoa = tdoa.model.jacobian(x_tdoa, x_source, tdoa_ref_idx)
    else:
        j_tdoa = np.zeros(shape=empty_dims)

    # Compute Jacobian for FDOA measurements
    if x_fdoa is not None and v_fdoa is not None:
        j_aoa = np.concatenate((j_aoa, np.zeros_like(j_aoa)), axis=0)
        j_tdoa = np.concatenate((j_tdoa, np.zeros_like(j_tdoa)), axis=0)
        j_fdoa = fdoa.model.jacobian(x_sensor=x_fdoa, v_sensor=v_fdoa, x_source=x_source, v_source=v_source,
                                     ref_idx=fdoa_ref_idx)
    else:
        j_fdoa = np.zeros(shape=empty_dims)

    # Combine component Jacobian matrices
    return np.concatenate((j_aoa, j_tdoa, j_fdoa), axis=1)


def jacobian_uncertainty(x_source: npt.ArrayLike,
                         x_aoa: npt.ArrayLike=None,
                         x_tdoa: npt.ArrayLike=None,
                         x_fdoa: npt.ArrayLike=None,
                         v_fdoa: npt.ArrayLike=None,
                         v_source: npt.ArrayLike | None=None,
                         do_2d_aoa: bool=False,
                         tdoa_ref_idx=None,
                         fdoa_ref_idx=None,
                         do_bias: bool=False,
                         do_pos_error: bool=False):
    """
    Returns the Jacobian matrix for a set of TDOA measurements in the presence of sensor
    uncertainty, in the form of measurement bias and/or sensor position errors.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    30 April 2025

    :param x_source: nDim x n_source array of source positions
    :param x_aoa: nDim x nAOA array of sensor positions
    :param x_tdoa: nDim x nTDOA array of TDOA sensor positions
    :param x_fdoa: nDim x nFDOA array of FDOA sensor positions
    :param v_fdoa: nDim x nFDOA array of FDOA sensor velocities
    :param v_source: nDim x 1 source velocity; assumed stationary if left blank
    :param do_2d_aoa: Optional boolean parameter specifying whether 1D (az-only) or 2D (az/el) AOA is being performed
    :param tdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for TDOA
    :param fdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for FDOA
    :param do_bias: if True, jacobian includes gradient w.r.t. measurement biases
    :param do_pos_error: if True, jacobian includes gradient w.r.t. sensor pos/vel errors
    :return: n_dim x nMeasurement x n_source matrix of Jacobians, one for each candidate source position
    """

    # Make a dict with the args for all three gradient function calls
    jacob_args = {'x_source': x_source,
                  'v_source': v_source,
                  'x_aoa': x_aoa,
                  'x_tdoa': x_tdoa,
                  'x_fdoa': x_fdoa,
                  'v_fdoa': v_fdoa,
                  'do_2d_aoa': do_2d_aoa,
                  'tdoa_ref_idx': tdoa_ref_idx,
                  'fdoa_ref_idx': fdoa_ref_idx}

    # Gradient w.r.t source position
    j_source = grad_source(**jacob_args)
    j_list = [j_source]

    # Gradient w.r.t measurement biases
    if do_bias:
        j_bias = grad_bias(**jacob_args)
        j_list.append(j_bias)

    # Gradient w.r.t sensor position
    if do_pos_error:
        j_sensor_pos = grad_sensor_pos(**jacob_args)
        j_list.append(j_sensor_pos)

    # Combine component Jacobians
    j = np.concatenate(j_list, axis=0)

    return j


def log_likelihood(x_source: npt.NDArray[np.float64],
                   zeta: npt.NDArray[np.float64],
                   cov: CovarianceMatrix,
                   x_aoa: npt.NDArray[np.float64] | None=None,
                   x_tdoa: npt.NDArray[np.float64] | None=None,
                   x_fdoa: npt.NDArray[np.float64] | None=None,
                   v_fdoa: npt.NDArray[np.float64] | None=None,
                   v_source: npt.NDArray[np.float64] | None=None,
                   do_2d_aoa: bool=False,
                   tdoa_ref_idx=None,
                   fdoa_ref_idx=None,
                   do_resample: bool=False,
                   angle_bias: npt.NDArray[np.float64] | None=None,
                   range_bias: npt.NDArray[np.float64] | None=None,
                   range_rate_bias: npt.NDArray[np.float64] | None=None):
    """
    Computes the Log Likelihood for Hybrid sensor measurement (AOA, TDOA, and
    FDOA), given the received measurement vector zeta, covariance matrix C,
    and set of candidate source positions x_source.

    Ported from MATLAB Code.

    Nicholas O'Donoughue
    10 March 2021

    :param x_source: Candidate source positions
    :param zeta: Combined AOA/TDOA/FDOA measurement vector
    :param cov: Measurement covariance matrix (or its precomputed inverse)
    :param x_aoa: nDim x nAOA array of sensor positions
    :param x_tdoa: nDim x nTDOA array of TDOA sensor positions
    :param x_fdoa: nDim x nFDOA array of FDOA sensor positions
    :param v_fdoa: nDim x nFDOA array of FDOA sensor velocities
    :param v_source: nDim x 1 source velocity; assumed stationary if left blank
    :param do_2d_aoa: Optional boolean parameter specifying whether 1D (az-only) or 2D (az/el) AOA is being performed
    :param tdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for TDOA
    :param fdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for FDOA
    :param do_resample: Boolean flag; if true the covariance matrix will be resampled, using ref_idx
    :param angle_bias: AOA measurement bias
    :param range_bias: TDOA measurement bias
    :param range_rate_bias: FDOA measurement bias
    :return ell: Log-likelihood evaluated at each position x_source.
    """

    shp = np.shape(x_source)
    # n_dim = shp[0] if len(shp) > 0 else 1
    n_source_pos = shp[1] if len(shp) > 1 else 1
    shp = np.shape(v_source)
    n_source_pos2 = shp[1] if len(shp) > 1 else 1

    # Make the source position and velocity 2D, if they're not already
    if n_source_pos == 1:
        x_source = x_source[:, np.newaxis]
    if v_source is not None and n_source_pos2 == 1:
        v_source = v_source[:, np.newaxis]

    # Pre-compute covariance matrix inverses
    if do_resample:
        shp = np.shape(x_aoa)
        num_aoa = shp[1] if len(shp) > 1 else 1
        shp = np.shape(x_tdoa)
        num_tdoa = shp[1] if len(shp) > 1 else 1
        shp = np.shape(x_fdoa)
        num_fdoa = shp[1] if len(shp) > 1 else 1
        if do_2d_aoa: num_aoa *= 2
        # Use the hybrid-specific covariance matrix resampler, which handles the assumed structure.
        cov = cov.resample_hybrid(num_aoa=num_aoa, num_tdoa=num_tdoa, num_fdoa=num_fdoa,
                                  tdoa_ref_idx=tdoa_ref_idx, fdoa_ref_idx=fdoa_ref_idx)

    # Generate the ideal measurement matrix for this position
    zeta_dot = measurement(x_aoa=x_aoa, x_tdoa=x_tdoa, x_fdoa=x_fdoa, v_fdoa=v_fdoa, x_source=x_source,
                           v_source=v_source, do_2d_aoa=do_2d_aoa, tdoa_ref_idx=tdoa_ref_idx, fdoa_ref_idx=fdoa_ref_idx,
                            angle_bias=angle_bias, range_bias=range_bias, range_rate_bias=range_rate_bias)
    arrs, _ = broadcast_backwards([zeta, zeta_dot], start_dim=0, do_broadcast=True)
    zeta, zeta_dot = arrs

    # Evaluate the measurement error
    err = zeta_dot - zeta

    # Compute the scaled log likelihood
    ell = - cov.solve_aca(np.moveaxis(err, source=0, destination=-1))

    return ell


def error(x_source: npt.ArrayLike,
          cov: CovarianceMatrix,
          x_aoa: npt.ArrayLike=None,
          x_tdoa: npt.ArrayLike=None,
          x_fdoa: npt.ArrayLike=None,
          v_fdoa: npt.ArrayLike=None,
          v_source: npt.ArrayLike | None=None,
          x_max: npt.ArrayLike=1,
          num_pts: int=11,
          do_2d_aoa: bool=False,
          tdoa_ref_idx=None,
          fdoa_ref_idx=None,
          do_resample: bool=False):
    """
    Construct a 2-D field from -x_max to +x_max, using numPts in each
    dimension.  For each point, compute the hybrid solution for each sensor
    against the reference (the first sensor), and compare to the FDOA
    solution from the true emitter position.

    Ported from MATLAB Code.

    Nicholas O'Donoughue
    21 January 2021

    :param x_source: nDim x 1 matrix of true emitter position
    :param cov: N x N covariance matrix
    :param x_aoa: nDim x nAOA array of sensor positions
    :param x_tdoa: nDim x nTDOA array of TDOA sensor positions
    :param x_fdoa: nDim x nFDOA array of FDOA sensor positions
    :param v_fdoa: nDim x N matrix of sensor velocities
    :param v_source: nDim x 1 matrix of true emitter velocity
    :param x_max: nDim x 1 (or scalar) vector of maximum offset from origin for plotting
    :param num_pts: Number of test points along each dimension
    :param do_2d_aoa: Optional boolean parameter specifying whether 1D (az-only) or 2D (az/el) AOA is being performed
    :param tdoa_ref_idx: Scalar index of reference sensor, or n_dim x n_pair matrix of sensor pairings for TDOA
    :param fdoa_ref_idx: Scalar index of reference sensor, or n_dim x n_pair matrix of sensor pairings for FDOA
    :param do_resample: Boolean flag; if true the covariance matrix will be resampled, using ref_idx
    :return epsilon: 2-D plot of FDOA error
    :return x_vec:
    :return y_vec:
    """

    # Compute the true measurement
    zeta = measurement(x_aoa=x_aoa, x_tdoa=x_tdoa, x_fdoa=x_fdoa, v_fdoa=v_fdoa, x_source=x_source, v_source=v_source,
                       do_2d_aoa=do_2d_aoa, tdoa_ref_idx=tdoa_ref_idx, fdoa_ref_idx=fdoa_ref_idx)

    # Pre-process the covariance matrix
    if do_resample:
        shp = np.shape(x_aoa)
        num_aoa = shp[1] if len(shp) > 1 else 1
        shp = np.shape(x_tdoa)
        num_tdoa = shp[1] if len(shp) > 1 else 1
        shp = np.shape(x_fdoa)
        num_fdoa = shp[1] if len(shp) > 1 else 1
        if do_2d_aoa: num_aoa *= 2
        cov = cov.resample_hybrid(num_aoa=num_aoa, num_tdoa=num_tdoa, num_fdoa=num_fdoa,
                                  tdoa_ref_idx=tdoa_ref_idx, fdoa_ref_idx=fdoa_ref_idx)

    # Set up test points
    grid_res = 2*x_max / (num_pts-1)
    search_space = SearchSpace(x_ctr=x_source,
                               max_offset=x_max,
                               epsilon=grid_res)
    x_set, x_grid = search_space.x_set, search_space.x_grid
    x_vec = x_grid[0][0, :]
    y_vec = x_grid[1][:, 0]

    zeta_list = measurement(x_aoa=x_aoa, x_tdoa=x_tdoa, x_fdoa=x_fdoa, v_fdoa=v_fdoa,
                            x_source=x_set, v_source=v_source, tdoa_ref_idx=tdoa_ref_idx, fdoa_ref_idx=fdoa_ref_idx)

    err = zeta[:, np.newaxis] - zeta_list

    # Angle measurements require a modulo 2*pi operation to avoid edge effects
    if x_aoa is not None:
        shp = np.shape(x_aoa)
        num_aoa = shp[1] if len(shp) > 1 else 1
        for idx_aoa in range(num_aoa):
            err[idx_aoa, :] = modulo2pi(err[idx_aoa, :])

    # epsilon_list = [cov.solve_aca(this_err) for this_err in err.T]
    epsilon_list = cov.solve_aca(err.T)

    return np.reshape(epsilon_list, search_space.grid_shape), x_vec, y_vec


def grad_source(x_source: npt.ArrayLike,
           x_aoa: npt.ArrayLike=None,
           x_tdoa: npt.ArrayLike=None,
           x_fdoa: npt.ArrayLike=None,
           v_fdoa: npt.ArrayLike=None,
           v_source: npt.ArrayLike | None=None,
           do_2d_aoa: bool=False,
           tdoa_ref_idx=None,
           fdoa_ref_idx=None):
    """
    Return the gradient of Hybrid measurements, with sensor uncertainties, with respect to target position, x.
    Equation 6.43. This function calls grad_source for each sensor type in succession, then concatenates the results.

    Ported from MATLAB code.

    Nicholas O'Donoughue
    14 April 2025

    :param x_source:
    :param x_aoa: nDim x nAOA array of sensor positions
    :param x_tdoa: nDim x nTDOA array of TDOA sensor positions
    :param x_fdoa: nDim x nFDOA array of FDOA sensor positions
    :param v_fdoa: nDim x nFDOA array of FDOA sensor velocities
    :param v_source: nDim x 1 source velocity; assumed stationary if left blank
    :param do_2d_aoa: Optional boolean parameter specifying whether 1D (az-only) or 2D (az/el) AOA is being performed
    :param tdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for TDOA
    :param fdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for FDOA
    :return jacobian:   Jacobian matrix representing the desired gradient
    """
    # TODO: Debug

    gradients = []
    if x_aoa is not None:
        gradients.append(triang.model.grad_source(x_source=x_source, x_sensor=x_aoa, do_2d_aoa=do_2d_aoa))

    if x_tdoa is not None:
        gradients.append(tdoa.model.grad_source(x_source=x_source, x_sensor=x_tdoa, ref_idx=tdoa_ref_idx))

    if x_fdoa is not None:
        # FDOA grad_source wraps fdoa.model.jacobian which stacks position and velocity rows
        # (shape 2*n_dim x n_meas).  Pad any already-collected AOA/TDOA gradients with zero
        # velocity rows so all blocks share the same row count before concatenating.
        gradients = [np.concatenate((g, np.zeros_like(g)), axis=0) for g in gradients]
        gradients.append(fdoa.model.grad_source(x_source=x_source, x_sensor=x_fdoa, v_sensor=v_fdoa,
                                                v_source=v_source, ref_idx=fdoa_ref_idx))

    grad = np.concatenate(gradients, axis=1)

    return grad


def grad_bias(x_source: npt.ArrayLike,
              x_aoa: npt.ArrayLike=None,
              x_tdoa: npt.ArrayLike=None,
              x_fdoa: npt.ArrayLike=None,
              v_fdoa: npt.ArrayLike=None,
              v_source: npt.ArrayLike | None=None,
              do_2d_aoa: bool=False,
              tdoa_ref_idx=None,
              fdoa_ref_idx=None):
    """
    Return the gradient of FDOA measurements, with sensor uncertainties, with respect to the unknown measurement bias
    terms.

    Ported from MATLAB code.

    Nicholas O'Donoughue
    14 April 2025

    :param x_source:
    :param x_aoa: nDim x nAOA array of sensor positions
    :param x_tdoa: nDim x nTDOA array of TDOA sensor positions
    :param x_fdoa: nDim x nFDOA array of FDOA sensor positions
    :param v_fdoa: nDim x nFDOA array of FDOA sensor velocities
    :param v_source: nDim x 1 source velocity; assumed stationary if left blank
    :param do_2d_aoa: Optional boolean parameter specifying whether 1D (az-only) or 2D (az/el) AOA is being performed
    :param tdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for TDOA
    :param fdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for FDOA
    :return jacobian:   Jacobian matrix representing the desired gradient
    """
    # TODO: Debug

    gradients = []
    if x_aoa is not None:
        gradients.append(triang.model.grad_bias(x_source=x_source, x_sensor=x_aoa, do_2d_aoa=do_2d_aoa))

    if x_tdoa is not None:
        gradients.append(tdoa.model.grad_bias(x_source=x_source, x_sensor=x_tdoa, ref_idx=tdoa_ref_idx))

    if x_fdoa is not None:
        gradients.append(fdoa.model.grad_bias(x_source=x_source, x_sensor=x_fdoa,
                                              ref_idx=fdoa_ref_idx))

    if len(gradients) == 0:
        # No gradients; nothing to return
        return np.array([])
    elif len(gradients) == 1:
        # Only one gradient was generated; return it directly
        return gradients[0]

    orig_shapes = [np.shape(g) for g in gradients]
    max_len = max(map(len, orig_shapes))
    if max_len > 2:
        # Move axes (0, 1) to the end so that block_diag will work on them properly
        gradients = [np.moveaxis(g, (0, 1), (-2, -1)) for g in gradients]

    # At this point, we know len(grads) is >0, but PyCharm doesn't, so it's presenting a static analysis warning.
    # Shift the function call to grads[0], *grads[1:] to ensure at least one positional argument is passed in.
    gradients = block_diag(gradients[0], *gradients[1:])

    # Move the axes back
    if max_len > 2:
        gradients = np.moveaxis(gradients, (-2, -1), (0, 1))

    return gradients


def grad_sensor_pos(x_source: npt.ArrayLike,
                    x_aoa: npt.ArrayLike=None,
                    x_tdoa: npt.ArrayLike=None,
                    x_fdoa: npt.ArrayLike=None,
                    v_fdoa: npt.ArrayLike=None,
                    v_source: npt.ArrayLike | None=None,
                    do_2d_aoa: bool=False,
                    tdoa_ref_idx=None,
                    fdoa_ref_idx=None):
    """
    Compute the gradient of hybrid measurements, with sensor uncertainties, with respect to sensor position and
    velocity, according to equation 6.43.

    Ported from MATLAB code.

    Nicholas O'Donoughue
    14 April 2025

    :param x_source:
    :param x_aoa: nDim x nAOA array of sensor positions
    :param x_tdoa: nDim x nTDOA array of TDOA sensor positions
    :param x_fdoa: nDim x nFDOA array of FDOA sensor positions
    :param v_fdoa: nDim x nFDOA array of FDOA sensor velocities
    :param v_source: nDim x 1 source velocity; assumed stationary if left blank
    :param do_2d_aoa: Optional boolean parameter specifying whether 1D (az-only) or 2D (az/el) AOA is being performed
    :param tdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for TDOA
    :param fdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for FDOA
    :return jacobian:   Jacobian matrix representing the desired gradient
    """
    # TODO: Debug

    # Each sub-gradient block is assembled as (2*n_dim*n_k, n_meas_k):
    #   rows 0..n_dim*n_k-1     → gradient w.r.t. sensor position
    #   rows n_dim*n_k..end     → gradient w.r.t. sensor velocity (zeros for AOA/TDOA)
    # The final result is block-diagonal across sub-systems.
    blocks = []

    if x_aoa is not None:
        grad_a = triang.model.grad_sensor_pos(x_source=x_source, x_sensor=x_aoa, do_2d_aoa=do_2d_aoa)
        # AOA measurements have no velocity dependence; append a zero block for velocity rows
        blocks.append(np.concatenate((grad_a, np.zeros_like(grad_a)), axis=0))

    if x_tdoa is not None:
        grad_t = tdoa.model.grad_sensor_pos(x_source=x_source, x_sensor=x_tdoa, ref_idx=tdoa_ref_idx)
        # TDOA measurements have no velocity dependence; append a zero block for velocity rows
        blocks.append(np.concatenate((grad_t, np.zeros_like(grad_t)), axis=0))

    if x_fdoa is not None:
        # FDOA measurements depend on both position and velocity
        grad_fp = fdoa.model.grad_sensor_pos(x_source=x_source, x_sensor=x_fdoa, v_sensor=v_fdoa,
                                             v_source=v_source, ref_idx=fdoa_ref_idx)
        grad_fv = fdoa.model.grad_sensor_vel(x_source=x_source, x_sensor=x_fdoa, v_sensor=v_fdoa,
                                             v_source=v_source, ref_idx=fdoa_ref_idx)
        blocks.append(np.concatenate((grad_fp, grad_fv), axis=0))

    # Assemble as block-diagonal: shape (sum of rows, sum of cols)
    total_rows = sum(b.shape[0] for b in blocks)
    total_cols = sum(b.shape[1] for b in blocks)
    grad = np.zeros((total_rows, total_cols))
    row_off = 0
    col_off = 0
    for b in blocks:
        nr, nc = b.shape
        grad[row_off:row_off + nr, col_off:col_off + nc] = b
        row_off += nr
        col_off += nc

    return grad
