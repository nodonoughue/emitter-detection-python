import numpy as np
from numpy import typing as npt
from scipy.special import erfcinv
from scipy import stats

import utils
from .unit_conversions import lin_to_db
from itertools import combinations
from collections.abc import Iterable
import seaborn as sns
import matplotlib.pyplot as plt
import os
import time


def init_plot_style(dpi=400):
    """
    Initialize plotting styles, including output resolution
    """

    # Specify dpi for figure saving; matplotlib default is 200. It's pretty low-res, so we're using a default of 400
    # unless a different value is specified when calling this function.
    plt.rcParams['figure.dpi'] = dpi

    # Initialize seaborn for pretty plots; use a colorblind palette for accessibility
    sns.set_theme(context='paper', palette='colorblind')


def init_output_dir(subdir=''):
    """
    Create the output directory for figures, if needed, and return address as a prefix string

    :return: path to output directory
    """

    # Set up directory and filename for figures
    dir_nm = 'figures'
    if not os.path.exists(dir_nm):
        os.makedirs(dir_nm)

    # Make the requested subfolder
    dir_nm = os.path.join(dir_nm, subdir)
    if not os.path.exists(dir_nm):
        os.makedirs(dir_nm)

    return dir_nm + os.sep


def sinc_derivative(x):
    """
    Returns the derivative of sinc(x), which is given
          y= (x * cos(x) - sin(x)) / x^2
    for x ~= 0.  When x=0, y=0.  The input is in radians.

    NOTE: The MATLAB sinc function is defined sin(pi*x)/(pi*x).  Its usage
    will be different.  For example, if calling
              y = sinc(x)
    then the corresponding derivative will be
              z = sinc_derivative(pi*x);

    Ported from MATLAB code.

    Nicholas O'Donoughue
    9 January 2021

    :param x: input, radians
    :return x_dot: derivative of sinc(x), in radians
    """

    # Apply the sinc derivative where the mask is valid, and a zero where it is not
    return np.piecewise(x,
                        [x == 0],
                        [0, lambda z: (z * np.cos(z) - np.sin(z)) / (z ** 2)])


def make_taper(taper_len: int, taper_type: str):
    """
    Generate an amplitude taper of length N, according to the desired taperType, and optional set of parameters

    For discussion of these, and many other windows, see the Wikipedia page:
        https://en.wikipedia.org/wiki/Window_function/

    Ported from MATLAB Code.

    Nicholas O'Donoughue
    16 January 2021

    :param taper_len: Length of the taper
    :param taper_type: String describing the type of taper desired.  Supported options are: "uniform", "cosine",
                      "hanning", "hamming", "bartlett", and "blackman-harris"
    :return w: Set of amplitude weights [0-1]
    :return snr_loss: SNR Loss of peak return, w.r.t. uniform taper
    """

    # Some constants/utilities for the window functions
    def idx_centered(x):
        return np.arange(x) - (x - 1) / 2

    switcher = {'uniform': lambda x: np.ones(shape=(x,)),
                'cosine': lambda x: np.sin(np.pi / (2 * x)) * np.cos(np.pi * (np.arange(x) - (x - 1) / 2) / x),
                'hann': lambda x: np.cos(np.pi * idx_centered(x) / x) ** 2,
                'hamming': lambda x: .54 + .46 * np.cos(2 * np.pi * idx_centered(x) / x),
                'blackman-harris': lambda x: .42 + .5 * np.cos(2 * np.pi * idx_centered(x) / x)
                                                 + .08 * np.cos(4 * np.pi * idx_centered(x) / x)
                }

    # Generate the window
    taper_type = taper_type.lower()
    if taper_type in switcher:
        w = switcher[taper_type](taper_len)
    else:
        raise KeyError('Unrecognized taper type ''{}''.'.format(taper_type))

    # Set peak to 1
    w = w / np.max(np.fabs(w))

    # Compute SNR Loss, rounded to the nearest hundredth of a dB
    snr_loss = np.around(lin_to_db(np.sum(np.fabs(w) / taper_len)), decimals=2)

    return w, snr_loss


def parse_reference_sensor(ref_idx, num_sensors=0):
    """
    Accepts a reference index setting (either None, a scalar integer, or a 2 x N array of sensor pairs),
    and returns matching vectors for test and reference indices.

    :param ref_idx: reference index setting, acceptable formats are:
            None        -- use default (last sensor) as a common reference, generates a non-redundant set
            Integer     -- use the specified sensor number as a common reference
            'Full'      -- generate all possible sensor pairs
            2 x N array -- specifies N separate measurement pairs; the first vector is taken as the test indices, and
                           the second as reference.
    :param num_sensors: Number of available sensors
    :return test_idx_vec: Indices of sensors used for each measurement; sensors can be used more than once. Every
        element should be an integer.
    :return ref_idx_vec: Indices of sensors used for each measurement as a reference; sensors can be used more than
        once. Every element should be an integer, although in general nan is used for measurements that don't use a
        reference (e.g., AoA).
    """

    if ref_idx is None:
        # Default behavior is to use the last sensor as a common reference
        test_idx_vec = np.arange(num_sensors-1)
        ref_idx_vec = (num_sensors - 1) * np.ones_like(test_idx_vec)

    elif isinstance(ref_idx, str) and ref_idx.lower() == 'full':
        # Generate all possible sensor pairs
        perm = list(combinations(np.arange(num_sensors), 2))
        test_idx_vec = np.asarray([x[0] for x in perm])
        ref_idx_vec = np.asarray([x[1] for x in perm])

    elif np.isscalar(ref_idx):
        # Check for error condition
        assert ref_idx < num_sensors, 'Bad reference index; unable to parse.'

        # Scalar reference index, use all other sensors as test sensors
        test_idx_vec = np.asarray([i for i in np.arange(num_sensors) if i != ref_idx])
        ref_idx_vec = ref_idx * np.ones_like(test_idx_vec)

    else:
        # Pair of vectors; first row is test sensors, second is reference
        test_idx_vec = ref_idx[0, :]
        ref_idx_vec = ref_idx[1, :]

    return test_idx_vec, ref_idx_vec


def resample_covariance_matrix(cov: npt.ArrayLike, test_idx: npt.ArrayLike, ref_idx: npt.ArrayLike,
                               test_weights=None, ref_weights=None) -> npt.ArrayLike:
    """
    Resample a 2D covariance matrix to generate the covariance matrix that would result from a series of difference
    operations on the underlying random variables. See Section 3.3.1 of the 2022 text for derivation of the covariance
    matrix that results from sensor pair difference operations.

    :param cov: two-dimensional numpy array, representing a covariance matrix, to be resampled
    :param test_idx: numpy 1D array of indices for the test sensor for each measurement
    :param ref_idx: None, or numpy 1D array of indices for the reference sensor for each measurement. Any NaN
            entries are treated as test-only measurements (e.g., angle of arrival) that don't require a reference
            measurement. Those entries in the covariance matrix are not resampled.
    :param test_weights: Optional weights to apply to each measurement when resampling.
    :param ref_weights: Optional weights to apply to each measurement when resampling.
    :return: two-dimensional  numpy array, representing the re-sampled covariance matrix.
    """

    # Parse Inputs
    n_sensor = np.size(cov, axis=0)

    # Determine output size
    n_test = np.size(test_idx)
    n_ref = np.size(ref_idx)
    n_pair_out = np.fmax(n_test, n_ref)

    # Error Checking
    if 1 < n_test != n_ref > 1:
        raise TypeError("Error calling covariance matrix resample.  "
                        "Reference and test vectors must have the same shape.")

    if np.any(test_idx >= n_sensor) or np.any(ref_idx >= n_sensor):
        raise TypeError("Error calling covariance matrix resample.  "
                        "Indices exceed the dimensions of the covariance matrix.")

    # Parse sensor weights
    shp_test_wt = 1
    if test_weights:
        shp_test_wt = np.size(test_weights)

    shp_ref_wt = 1
    if ref_weights:
        shp_ref_wt = np.size(ref_weights)

    def _populate_2d_entries(arr_in, idx_i, idx_j):
        """
        Use the indices idx_i and idx_j to reference the two dimensions of the input
        arr_in.

        Any nans in idx_i and idx_j should be ignored
        """
        arr_out = np.zeros_like(idx_i)

        mask_i = np.isnan(idx_i)
        mask_j = np.isnan(idx_j)
        mask = ~np.logical_or(mask_i, mask_j)

        if not np.all(mask):
            arr_out[mask] = arr_in[idx_i[mask].astype(int), idx_j[mask].astype(int)]
        else:
            arr_out = arr_in[idx_i.astype(int), idx_j.astype(int)]
        return arr_out

    # Function to execute at each entry of output covariance matrix
    def element_func(idx_row, idx_col):
        idx_row = idx_row.astype(int)
        idx_col = idx_col.astype(int)
        a_i = test_idx[idx_row % n_test]
        b_i = ref_idx[idx_row % n_ref]
        a_j = test_idx[idx_col % n_test]
        b_j = ref_idx[idx_col % n_ref]
        if test_weights:
            a_i_wt = test_weights[idx_row % shp_test_wt]
            a_j_wt = test_weights[idx_col % shp_test_wt]
        else:
            a_i_wt = a_j_wt = 1.
        if ref_weights:
            b_i_wt = ref_weights[idx_row % shp_ref_wt]
            b_j_wt = ref_weights[idx_col % shp_ref_wt]
        else:
            b_i_wt = b_j_wt = 1.

        cov_ai_aj = _populate_2d_entries(cov, a_i, a_j)
        cov_bi_bj = _populate_2d_entries(cov, b_i, b_j)
        cov_ai_bj = _populate_2d_entries(cov, a_i, b_j)
        cov_bi_aj = _populate_2d_entries(cov, b_i, a_j)

        res = b_i_wt * b_j_wt * cov_bi_bj + \
            a_i_wt * a_j_wt * cov_ai_aj - \
            a_i_wt * b_j_wt * cov_ai_bj - \
            b_i_wt * a_j_wt * cov_bi_aj

        return res
    cov_out = np.fromfunction(element_func, (n_pair_out, n_pair_out), dtype=float)
    return cov_out


def resample_noise(noise: npt.ArrayLike, test_idx: npt.ArrayLike = None, ref_idx=None, test_weights=None, ref_weights=None):
    """
    Generate resampled noise according to the set of test and reference sensors provided. See Section 3.3.1 of the 2022
    text for a discussion of sensor pairs and noise statistics. If the input noise is distributed according to a
    covariance matrix cov_in, then the result will be distributed according to
    resample_covariance_matrix(cov_in, test_idx, ref_idx).

    :param noise: numpy ndarray of noise samples; the first dimension represents individual sensor measurements
    :param test_idx: numpy 1D array of indices for the test sensor for each measurement, or None
    :param ref_idx: numpy 1D array of indices for the reference sensor for each measurement. Any NaN
            entries are treated as test-only measurements (e.g., angle of arrival) that don't require a reference
            measurement. Those entries in the covariance matrix are not resampled.
            If test_idx is None, then ref_idx is passed to utils.parse_reference_sensor, and may be any valid input
            to that function.
    :param test_weights: Optional weights to apply to each measurement when resampling.
    :param ref_weights: Optional weights to apply to each measurement when resampling.
    :return: numpy ndarray of resampled noise; the first dimension has the same length as test_idx and ref_idx.
    """
    # Parse Inputs
    n_sensor, n_sample = utils.safe_2d_shape(noise)

    if test_idx is None:
        # We need to use the ref_idx
        test_idx_vec, ref_idx_vec = utils.parse_reference_sensor(ref_idx, n_sensor)
    else:
        test_idx_vec = test_idx
        ref_idx_vec = ref_idx

    # Determine output size
    n_test = np.size(test_idx_vec)
    n_ref = np.size(ref_idx_vec)
    n_pair_out = np.fmax(n_test, n_ref)

    # Error Checking
    if 1 < n_test != n_ref > 1:
        raise TypeError("Error calling covariance matrix resample.  "
                        "Reference and test vectors must have the same shape.")

    if np.any(test_idx_vec >= n_sensor) or np.any(ref_idx_vec >= n_sensor):
        raise TypeError("Error calling covariance matrix resample.  "
                        "Indices exceed the dimensions of the covariance matrix.")

    # Parse sensor weights
    shp_test_wt = 1
    if test_weights:
        shp_test_wt = np.size(test_weights)

    shp_ref_wt = 1
    if ref_weights:
        shp_ref_wt = np.size(ref_weights)

    # Function to execute at each entry of output covariance matrix
    def element_func(idx_row):
        idx_row = idx_row.astype(int)
        a_i = test_idx_vec[idx_row % n_test]
        b_i = ref_idx_vec[idx_row % n_ref]

        if test_weights:
            a_i_wt = test_weights[idx_row % shp_test_wt]
        else:
            a_i_wt = 1.
        if ref_weights:
            b_i_wt = ref_weights[idx_row % shp_ref_wt]
        else:
            b_i_wt = 1.

        noise_ai = np.zeros((len(a_i), utils.safe_2d_shape(noise)[1]))
        noise_bi = np.zeros_like(noise_ai)

        mask_ai = ~np.isnan(a_i)
        mask_bi = ~np.isnan(b_i)

        if not np.all(mask_ai):
            noise_ai[mask_ai] = noise[a_i[mask_ai].astype(int)]
        else:
            noise_ai = noise[a_i.astype(int)]

        if not np.all(mask_bi):
            noise_bi[mask_bi] = noise[b_i[mask_bi].astype(int)]
        else:
            noise_bi = noise[b_i.astype(int)]

        res = b_i_wt * noise_bi - a_i_wt * noise_ai
        # raise ValueError('mo')
        return res

    noise_out = np.fromfunction(element_func, (n_pair_out, ), dtype=float)
    return noise_out


def ensure_invertible(covariance, epsilon=1e-10):
    """
    Check the input matrix by finding the eigenvalues and checking that they are all >= a small value
    (epsilon), to ensure that it can be inverted.

    If any of the eigenvalues are too small, then a diagonal loading term is applied to ensure that the matrix is
    positive definite (all eigenvalues are >= epsilon).

    Ported from MATLAB code.

    Nicholas O'Donoughue
    5 Sept 2021

    :param covariance: 2D (nDim x nDim) covariance matrix.  If >2 dimensions, the process is repeated for each.
    :param epsilon: numerical precision term (the smallest eigenvalue must be >= epsilon) [Default = 1e-10]
    :return covariance_out: Modified covariance matrix that is guaranteed invertible
    """

    # Check input dimensions
    if np.isscalar(covariance):
        return covariance  # Input is a scalar; it is invertible by definition

    sz = np.shape(covariance)
    assert len(sz) > 1, 'Input must have at least two dimensions.'
    assert sz[0] == sz[1], 'First two dimensions of input matrix must be equal.'
    dim = sz[0]
    if len(sz) > 2:
        n_matrices = np.prod(sz[2:])
    else:
        n_matrices = 1

    # Iterate across matrices (dimensions >2)
    cov_out = np.zeros(shape=sz)

    for idx_matrix in np.arange(n_matrices):
        # Isolate the current covariance matrix
        if len(sz) > 2:
            this_cov = np.squeeze(covariance[:, :, idx_matrix])
        else:
            this_cov = covariance

        # Eigen-decomposition
        lam, v = np.linalg.eigh(this_cov)

        # Initialize the diagonal loading term
        d = epsilon * np.eye(N=dim)

        # Repeat until the smallest eigenvalue is larger than epsilon
        while np.amin(lam) < epsilon:
            # Add the diagonal loading term
            this_cov += d

            # Re-examine the eigenvalue
            lam, v = np.linalg.eigh(this_cov)

            # Increase the magnitude of diagonal loading (for the next iteration)
            d *= 10.0

        # Store the modified covariance matrix in the output
        if len(sz) > 2:
            cov_out[:, :, idx_matrix] = this_cov
        else:
            cov_out = this_cov

    return cov_out


def make_pdfs(measurement_function, measurements, pdf_type='MVN', covariance: npt.ArrayLike = 1):
    """
    Generate a joint PDF or set of unitary PDFs representing the measurements, given the measurement_function,
    covariance matrix and pdf_type

    The only currently supported pdf types are:
        'mvn'       multivariate normal
        'normal'    normal (each measurement is independent)

    Ported from MATLAB Code

    Nicholas O'Donoughue
    16 January 2021

    :param measurement_function: A single function handle that will accept an nDim x nSource array of candidate emitter
                                 positions and return a num_measurement x num_source array of measurements that those
                                 emitters are expected to have generated.
    :param measurements: The received measurements
    :param pdf_type: The type of distribution to assume.
    :param covariance: Array of covariances (num_measurement x 1 for normal, num_measurement x num_measurement for
                       multivariate normal)
    :return pdfs: List of function handles, each of which accepts an nDim x nSource array of candidate source
                  positions, and returns a 1 x nSource array of probabilities.
    """

    if pdf_type is None:
        pdf_type = 'mvn'

    if pdf_type.lower() == 'mvn' or pdf_type.lower() == 'normal':
        rv = stats.multivariate_normal(mean=measurements, cov=covariance)  # frozen MVN representation
        pdfs = [lambda x: rv.pdf(measurement_function(x))]
    else:
        raise KeyError('Unrecognized PDF type setting: ''{}'''.format(pdf_type))

    return pdfs


def print_elapsed(t_elapsed):
    """
    Print the elapsed time, provided in seconds.

    Nicholas O'Donoughue
    6 May 2021

    :param t_elapsed: elapsed time, in seconds
    """

    hrs_elapsed = np.floor(t_elapsed / 3600)
    minutes_elapsed = np.floor((t_elapsed - 3600 * hrs_elapsed) / 60)
    secs_elapsed = t_elapsed - hrs_elapsed * 3600 - minutes_elapsed * 60

    print('Elapsed Time: {} hrs, {} min, {:.2f} sec'.format(hrs_elapsed, minutes_elapsed, secs_elapsed))


def print_predicted(t_elapsed, pct_elapsed, do_elapsed=False):
    """
    Print the elapsed and predicted time, provided in seconds.

    Nicholas O'Donoughue
    6 May 2021

    :param t_elapsed: elapsed time, in seconds
    :param pct_elapsed:
    :param do_elapsed:
    """

    if do_elapsed:
        hrs_elapsed = np.floor(t_elapsed / 3600)
        minutes_elapsed = (t_elapsed - 3600 * hrs_elapsed) / 60

        print('Elapsed Time: {:.0f} hrs, {:.2f} min. '.format(hrs_elapsed, minutes_elapsed), end='')

    t_remaining = t_elapsed * (1 - pct_elapsed) / pct_elapsed

    hrs_remaining = np.floor(t_remaining / 3600)
    minutes_remaining = (t_remaining - 3600 * hrs_remaining) / 60

    print('Estimated Time Remaining: {} hrs, {:.2f} min'.format(hrs_remaining, minutes_remaining))


def print_progress(num_total, curr_idx, iterations_per_marker, iterations_per_row, t_start):
    """
    Print progress for a long-duration simulation, including progress markers (dots),
    and predicted time remaining.

    Nicholas O'Donoughue
    28 July 2024

    :param num_total: Total number of iterations in for loop
    :param curr_idx: Current iteration
    :param iterations_per_marker: Number of iterations per marker
    :param iterations_per_row: Number of iterations per row
    :param t_start: Time at which the for loop began
    """
    if np.mod(curr_idx + 1, iterations_per_marker) == 0:
        print('.', end='')  # Use end='' to prevent the newline

    if np.mod(curr_idx + 1, iterations_per_row) == 0:
        print(' ({}/{}) '.format(curr_idx + 1, num_total), end='')
        pct_elapsed = curr_idx / num_total
        t_elapsed = time.perf_counter() - t_start
        print_predicted(t_elapsed, pct_elapsed, do_elapsed=True)


def safe_2d_shape(x: np.array) -> np.array:
    """
    Compute the 2D shape of the input, x, safely. Avoids errors when the input is a 1D array (in which case, the
    second output is 1).  Any dimensions higher than the second are ignored.

    Nicholas O'Donoughue
    19 May 2021

    :param x: ND array to determine the size of.
    :return dim1: length of first dimension
    :return dim2: length of second dimension
    """

    if x is None:
        return 0, 0
        
    # Wrap x in an array, in case it's a scalar or list
    x = np.asarray(x)

    # Initialize output dimensions
    dims_out = [1, 1]  # The default, for empty dimensions, is 1

    # Attempt to overwrite default with the actual size, for defined dimensions
    dims = np.shape(x)
    for idx_dim in np.arange(np.amin([2, len(dims)])):
        dims_out[idx_dim] = dims[idx_dim]

    return dims_out


def make_nd_grid(x_ctr, max_offset, grid_spacing):
    """
    Create and return an ND search grid, based on the specified center of the search space, extent, and grid spacing.

    28 December 2021
    Nicholas O'Donoughue

    :param x_ctr: ND array of search grid center, for each dimension.  The size of x_ctr dictates how many dimensions
                  there are
    :param max_offset: scalar or ND array of the extent of the search grid in each dimension, taken as the one-sided
                        maximum offset from x_ctr
    :param grid_spacing: scalar or ND array of grid spacing in each dimension
    :return x_set: n_dim x N numpy array of positions
    :return x_grid: n_dim-tuple of n_dim-dimensional numpy arrays containing the coordinates for each dimension.
    :return out_shape:  tuple with the size of the generated grid
    """

    n_dim = np.size(x_ctr)

    if np.size(max_offset) == 1:
        max_offset = max_offset * np.ones((n_dim, ))

    if np.size(grid_spacing) == 1:
        grid_spacing = grid_spacing * np.ones((n_dim, ))

    assert n_dim == np.size(max_offset) and n_dim == np.size(grid_spacing), \
           'Search space dimensions do not match across specification of the center, search_size, and epsilon.'

    n_elements = np.fix(1 + 2 * max_offset / grid_spacing).astype(int)

    # Check Search Size
    max_elements = 1e8  # Set a conservative limit
    assert np.prod(n_elements) < max_elements, \
           'Search size is too large; python is likely to crash or become unresponsive. Reduce your search size, or' \
           + ' increase the max allowed.'

    # Make a set of axes, one for each dimension, that are centered on x_ctr
    dims = [x + np.linspace(start=-x_max, stop=x_max*(1+n)/n, num=n) for (x, x_max, n)
            in zip(x_ctr, max_offset, n_elements)]

    # Use meshgrid expansion; each element of x_grid is now a full n_dim dimensioned grid
    x_grid = np.meshgrid(*dims)

    # Rearrange to a single 2D array of grid locations (n_dim x N)
    x_set = np.asarray([x.flatten() for x in x_grid])

    return x_set, x_grid, n_elements


def is_broadcastable(a, b):
    """
    Determine if two inputs are broadcastable. In other words, check all common dimensions, and
    ensure that they are either equal or of length 1.

    14 November 2022
    Nicholas O'Donoughue

    :param a: first input array
    :param b: second input array
    :return result:  Boolean (true is a and b are broadcastable)
    """

    while len(np.shape(a)) > len(np.shape(b)):
        b = b[:, np.newaxis]

    while len(np.shape(b)) > len(np.shape(a)):
        a = a[:, np.newaxis]

    try:
        np.broadcast_arrays(a, b)
        return True
    except ValueError:
        return False


def modulo2pi(x):
    """
    Perform a 2*pi modulo operation, but with the result centered on zero, spanning
    from -pi to pi, rather than on the interval 0 to 2*pi.

    8 December 2022
    Nicholas O'Donoughue

    :param x: Numpy array-like or scalar
    :return: Modulo, centered on 0 (on the interval -pi to pi)
    """

    # Shift the input so that zero is now pi
    x_shift = x + np.pi

    # Perform a modulo operation; result is on the interval [0, 2*pi)
    x_modulo = x_shift % (2*np.pi)

    # Undo the shift, so that a zero input is now a zero output.
    # Result is on the interval [-pi, pi]
    result = x_modulo - np.pi

    return result


def remove_outliers(data, axis=0, remove_nan=False):
    """
    Remove outliers from a dataset.  If it is a vector, the outliers are individual datapoints. If it is an array,
    then outlier detection is run across the specified dimension (default=0), and any sub-arrays containing an outlier
    are excised.
    
    Behavior is built to reflect MATLAB's rmoutliers, with the default (median) processing type.
    
    Outliers are defined as those that are more than three scaled MAD from the median. MAD is defined:
    
    c * median(abs(data - median(data))), where c = -1/(sqrt(2)*erfcinv(1.5))
    
    Nicholas O'Donoughue
    19 February 2025
    """

    # Compute the median
    median = np.nanmedian(data, axis=axis, keepdims=True)  # ignore nans
    median_distance = np.abs(data - median)

    # Compute the scale factor
    c = -1 / (np.sqrt(2) * erfcinv(1.5))

    # Compute the MAD
    mad = c * np.nanmedian(median_distance, axis=axis, keepdims=True)
    mad[mad == 0] = 1.  # if there's no variation along the text axis, scaled_distance will all equal zero; avoid that
    scaled_distance = median_distance / mad

    # Threshold
    outlier_mask = np.abs(scaled_distance) >= 3
    if remove_nan:
        outlier_mask = outlier_mask or np.isnan(data)

    # Flag any subarrays with at least one outlier for deletion
    deletion_mask = np.any(outlier_mask, axis=tuple(x for x in range(data.ndim) if x != axis))  # should be 1D

    # Remove outlier subarrays
    data_out = np.delete(data, deletion_mask, axis=axis)

    return data_out

def ensure_iterable(var, flatten=False)->Iterable:
    """
    Ensure that the input is an iterable. If it is not, wrap it in a list.

    Optionally searched for nested iterables and flatten them, so that all entries
    can be iterated over in a single for loop.

    Nicholas O'Donoughue
    3 April 2025

    :param var: variable to be tested
    :param flatten: whether to flatten the variable (default=False)
    :return var_out: Iterable containing the elements of var
    """
    # Make sure it's iterable
    if not isinstance(var, Iterable):  # accepts list, tuple, array, ...
        var = [var]  # wrap it in a list

    # Check for nested iterables
    if flatten:
        # Repeat until none of the elements are iterable
        while any(isinstance(element, Iterable) for element in var):
            var_out = []
            for element in var:
                if isinstance(element, Iterable):
                    # Use list comprehension
                    var_out.append(*element)
                else:
                    var_out.append(element)
            var = var_out  # overwrite the variable

    return var


def make_uncertainty_indices(num_dim=2, num_aoa=0, num_tdoa=0, num_fdoa=0, do_2d_aoa=False,
                             do_aoa_bias=False, do_tdoa_bias=False, do_fdoa_bias=False,
                             do_aoa_pos_error=False, do_tdoa_pos_error=False, do_fdoa_pos_error=False):
    """
    Build the search space for source localization problem with uncertainty. Parameter vector takes the form:
                Parameter                           Number of Unknowns
        theta = [x_source,                          num_dim
                 v_source,                          num_dim (if num_fdoa > 0)
                 aoa_bias,                          num_aoa (2*num_aoa if do_2d_aoa=True)
                 tdoa_bias,                         num_tdoa
                 fdoa_bias,                         num_fdoa
                 aoa_sensor_position,               num_aoa * num_dim
                 tdoa_sensor_position,              num_tdoa * num_dim
                 fdoa_sensor_position,              num_fdoa * num_dim
                 fdoa_sensor_velocity]              num_fdoa * num_dim

    This follows the description in Table 6.1.

    Nicholas O'Donoughue
    22 April 2025

    :param num_dim: Number of spatial dimensions in the problem (default=2)
    :param num_aoa: Number of AOA sensors (default=0)
    :param num_tdoa: Number of TDOA sensors (default=0)
    :param num_fdoa: Number of FDOA sensors (default=0)
    :param do_2d_aoa: Boolean flag (azimuth angle only if False, azimuth/elevation if True; default=False)
    :param do_aoa_bias: Boolean flag indicating whether to include AOA bias (default=False)
    :param do_tdoa_bias: Boolean flag indicating whether to include TDOA bias (default=False)
    :param do_fdoa_bias: Boolean flag indicating whether to include FDOA bias (default=False)
    :param do_aoa_pos_error: Boolean flag indicating whether to include AOA sensor position errors (default=False)
    :param do_tdoa_pos_error: Boolean flag indicating whether to include TDOA sensor position errors (default=False)
    :param do_fdoa_pos_error: Boolean flag indicating whether to include FDOA sensor position errors (default=False)
    :return indices: dictionary with the following fields, indicating the indices for rows corresponding to each
        'source_pos'
        'source_vel'
        'aoa_bias'
        'tdoa_bias'
        'fdoa_bias'
        'aoa_pos'
        'tdoa_pos'
        'fdoa_pos'
        'fdoa_vel'
    """

    # Source Position and Velocity
    source_pos = np.arange(num_dim)
    source_vel = (num_dim + np.arange(num_dim) if num_fdoa > 0 else None)
    bias_start_ind = (2*num_dim if num_fdoa > 0 else num_dim)

    # Sensor Measurement Biases
    num_aoa_bias = num_aoa * (2 if do_2d_aoa else 1) if do_aoa_bias and num_aoa > 0 else 0
    num_tdoa_bias = num_tdoa if do_tdoa_bias and num_tdoa > 0 else 0
    num_fdoa_bias = num_fdoa if do_fdoa_bias and num_fdoa > 0 else 0
    aoa_bias_ind = np.arange(num_aoa_bias) + bias_start_ind if num_aoa_bias > 0 else None
    tdoa_bias_ind = np.arange(num_tdoa_bias) + num_aoa_bias + bias_start_ind if num_tdoa_bias > 0 else None
    fdoa_bias_ind = np.arange(num_fdoa_bias) + num_tdoa_bias + num_aoa_bias + bias_start_ind if num_tdoa_bias > 0 else None
    pos_start_ind = bias_start_ind + num_aoa_bias + num_tdoa_bias + num_fdoa_bias

    # Sensor Position Errors
    num_aoa_pos = num_dim * num_aoa if do_aoa_pos_error else 0
    num_tdoa_pos = num_dim * num_tdoa if do_tdoa_pos_error else 0
    num_fdoa_pos = num_dim * num_fdoa if do_fdoa_pos_error else 0
    aoa_pos_ind = np.arange(num_aoa_pos) + pos_start_ind if do_aoa_pos_error else None
    tdoa_pos_ind = np.arange(num_tdoa_pos) + pos_start_ind + num_aoa_pos if do_tdoa_pos_error else None
    fdoa_pos_ind = np.arange(num_fdoa_pos) + pos_start_ind + num_aoa_pos + num_tdoa_pos if do_fdoa_pos_error else None
    fdoa_vel_ind = np.arange(num_fdoa_pos) + pos_start_ind + num_aoa_pos + num_tdoa_pos + num_fdoa_pos \
        if do_fdoa_pos_error else None
    total_num_params = pos_start_ind + num_aoa_pos + num_tdoa_pos + 2 * num_fdoa_pos

    indices = {'source_pos': source_pos,
               'source_vel': source_vel,
               'aoa_bias': aoa_bias_ind,
               'tdoa_bias': tdoa_bias_ind,
               'fdoa_bias': fdoa_bias_ind,
               'aoa_pos': aoa_pos_ind,
               'tdoa_pos': tdoa_pos_ind,
               'fdoa_pos': fdoa_pos_ind,
               'fdoa_vel': fdoa_vel_ind,
               'source': np.arange(bias_start_ind),
               'bias': np.arange(start=bias_start_ind, stop=pos_start_ind) if pos_start_ind > bias_start_ind else None,
               'sensor': np.arange(start=pos_start_ind, stop=total_num_params) if total_num_params > pos_start_ind else None,
               'num_parameters': total_num_params}
    return indices


def make_uncertainty_search_space(th_center, search_size, search_resolution,
                                  do_2d_aoa=False, do_aoa_bias=False, do_tdoa_bias=False, do_fdoa_bias=False,
                                  aoa_bias_search=1, tdoa_bias_search=10, fdoa_bias_search=10,
                                  sensor_pos_search=25, sensor_vel_search=10, source_pos_search=10e3,
                                  source_vel_search=10,
                                  source_search_dim = 101, bias_search_dim=11, sensor_search_dim=11,
                                  x_aoa=None, x_tdoa=None, x_fdoa=None, v_fdoa=None):
    """
    Build the search space for source localization problem with uncertainty. Parameter vector takes the form:
        theta = [x_source,
                 v_source,
                 aoa_bias,
                 tdoa_bias,
                 fdoa_bias,
                 aoa_sensor_position,
                 tdoa_sensor_position,
                 fdoa_sensor_position,
                 fdoa_sensor_velocity]

    This follows the description in Table 6.1.

    Nicholas O'Donoughue
    22 April 2025

    :param


    """
    # Look up parameter indices
    sensor_dims = parse_sensor_coords(x_aoa=x_aoa, x_tdoa=x_tdoa, x_fdoa=x_fdoa, v_fdoa=v_fdoa)
    param_indices = make_uncertainty_indices(**sensor_dims,  # num_dim, num_aoa, num_tdoa, num_fdoa
                                             do_2d_aoa=do_2d_aoa,
                                             do_aoa_bias=do_aoa_bias,
                                             do_tdoa_bias=do_tdoa_bias,
                                             do_fdoa_bias=do_fdoa_bias,
                                             do_aoa_pos_error=x_aoa is not None,
                                             do_tdoa_pos_error=x_tdoa is not None,
                                             do_fdoa_pos_error=x_fdoa is not None)


    # Check search space center
    num_params_in, _ = utils.safe_2d_shape(th_center)
    if num_params_in == param_indices['num_parameters']:
        # Do nothing, x_center is already properly defined
        pass
    else:
        # Redefine it; need to specify centers for the bias terms (0) and the sensor position terms (nominal positions)
        tmp = th_center  # copy the input to a temp variable
        th_center = np.zeros((param_indices['num_parameters'], ))

        # Source Position/Velocity
        if tmp is None:
            pass # leave source pos and vel center points at default (0)
        elif np.size(tmp) == len(param_indices['source_pos']) or np.size(tmp) == 1: # input matches source_pos
            th_center[param_indices['source_pos']] = tmp
            # leave source vel center points at default (0)
        elif np.size(tmp) == len(param_indices['source']): # input matches source_pos and vel
            th_center[param_indices['source']] = tmp
        else:
            print('Unable to parse input search space center; using defaults.')

        # Bias terms
        # The default is zero, no reason to do anything here.

        # Sensor Position Terms
        if param_indices['aoa_pos'] is not None:
            th_center[param_indices['aoa_pos']] = x_aoa.ravel()
        if param_indices['tdoa_pos'] is not None:
            th_center[param_indices['tdoa_pos']] = x_tdoa.ravel()
        if param_indices['fdoa_pos'] is not None:
            th_center[param_indices['fdoa_pos']] = x_fdoa.ravel()
        if param_indices['fdoa_vel'] is not None:
            th_center[param_indices['fdoa_vel']] = v_fdoa.ravel()

    # Check search size
    num_params_in, _ = utils.safe_2d_shape(search_size)
    if num_params_in == param_indices['num_parameters']:
        # Do nothing, search_center is already properly defined
        pass
    else:
        # Redefine it; need to search size for the bias terms and the sensor position terms
        tmp = search_size
        search_size = np.zeros((param_indices['num_parameters'], ))

        # Source Position/Velocity
        if tmp is None:
            search_size[param_indices['source_pos']] = source_pos_search  # use default position search
            search_size[param_indices['source_vel']] = source_vel_search  # use default velocity search
        elif np.size(tmp) == len(param_indices['source_pos']) or np.size(tmp) == 1: # input matches source_pos
            search_size[param_indices['source_pos']] = tmp
            search_size[param_indices['source_vel']] = source_vel_search  # use default velocity search
        elif np.size(tmp) == len(param_indices['source']): # input matches source_pos and vel
            search_size[param_indices['source']] = tmp
        else:
            print('Unable to parse input search space center; using defaults.')
            search_size[param_indices['source_vel']] = source_pos_search  # use default position search
            search_size[param_indices['source_vel']] = source_vel_search  # use default velocity search

        # Bias terms
        if param_indices['aoa_bias'] is not None:
            search_size[param_indices['aoa_bias']] = aoa_bias_search
        if param_indices['tdoa_bias'] is not None:
            search_size[param_indices['tdoa_bias']] = tdoa_bias_search
        if param_indices['fdoa_bias'] is not None:
            search_size[param_indices['fdoa_bias']] = fdoa_bias_search

        # Sensor Position Terms
        if param_indices['aoa_pos'] is not None:
            search_size[param_indices['aoa_pos']] = sensor_pos_search
        if param_indices['tdoa_pos'] is not None:
            search_size[param_indices['tdoa_pos']] = sensor_pos_search
        if param_indices['fdoa_pos'] is not None:
            search_size[param_indices['fdoa_pos']] = sensor_pos_search
        if param_indices['fdoa_vel'] is not None:
            search_size[param_indices['fdoa_vel']] = sensor_vel_search

    # Check search resolution
    num_params_in, _ = utils.safe_2d_shape(search_resolution)
    if num_params_in == param_indices['num_parameters']:
        # Do nothing, search_resolution is already properly defined
        pass
    else:
        # Redefine it; need to search size for the bias terms and the sensor position terms
        tmp = search_resolution
        search_resolution = np.zeros((param_indices['num_parameters'], ))

        # Source Position/Velocity
        if tmp is None:
            search_resolution[param_indices['source_pos']] = (2 * search_size[param_indices['source']] /
                                                              (source_search_dim - 1))  # use default search dimension size
        elif np.size(tmp) == len(param_indices['source_pos']) or np.size(tmp) == 1: # input matches source_pos
            search_resolution[param_indices['source_pos']] = tmp
            search_resolution[param_indices['source_vel']] = (2 * search_size[param_indices['source_vel']] /
                                                              (source_search_dim - 1))  # use default search dimension
        elif np.size(tmp) == len(param_indices['source']): # input matches source_pos and vel
            search_resolution[param_indices['source']] = tmp
        else:
            print('Unable to parse input search space center; using defaults.')
            search_resolution[param_indices['source']] = (2 * search_size[param_indices['source']] /
                                                          (source_search_dim - 1))  # use default search dimension size

        # Bias terms
        if param_indices['bias'] is not None:
            # use default search dimension size
            search_resolution[param_indices['bias']] = 2 * search_size[param_indices['bias']] / (bias_search_dim - 1)

        # Sensor Position Terms
        if param_indices['pos'] is not None:
            # use default search dimension size
            search_resolution[param_indices['pos']] = 2 * search_size[param_indices['pos']] / (sensor_search_dim - 1)

    return th_center, search_size, search_resolution, param_indices


def parse_sensor_coords(x_aoa=None, x_tdoa=None, x_fdoa=None, v_fdoa=None):
    n_dim1, n_aoa = utils.safe_2d_shape(x_aoa)  # returns 0, 0 if x_aoa is None
    n_dim2, n_tdoa = utils.safe_2d_shape(x_tdoa)
    n_dim3, n_fdoa = utils.safe_2d_shape(x_fdoa)
    n_dim4, _ = utils.safe_2d_shape(v_fdoa)

    # Check number of dimensions
    n_dim_set = np.array([n_dim1, n_dim2, n_dim3, n_dim4])
    n_dim_nonzero = n_dim_set[n_dim_set != 0]
    if len(n_dim_nonzero) <= 0:
        raise TypeError('At least one sensor position must be specified (they can\'t all be none).')

    if len(set(n_dim_nonzero)) > 1:
        raise TypeError('Not all defined sensor positions have the same number of dimensions.')

    if x_fdoa is not None and v_fdoa is not None and not utils.is_broadcastable(x_fdoa, v_fdoa):
        raise TypeError('FDOA sensor position and velocity inputs must have matching shapes.')

    # Package Response
    sensor_coords = {'num_dim': n_dim_nonzero[0],
                     'num_aoa': n_aoa,
                     'num_tdoa': n_tdoa,
                     'num_fdoa': n_fdoa}

    return sensor_coords