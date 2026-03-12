from collections.abc import Iterable
import itertools
import matplotlib.pyplot as plt
import numpy as np
from numpy import typing as npt
import os
from scipy import stats
from scipy.special import erfcinv
from scipy.linalg import pinvh
import seaborn as sns
import time
from typing import Callable

from .unit_conversions import lin_to_db


def init_plot_style(dpi: int=400):
    """
    Initialize plotting styles, including output resolution
    """

    # Specify dpi for figure saving; matplotlib default is 200. It's pretty low-res, so we're using a default of 400
    # unless a different value is specified when calling this function.
    plt.rcParams['figure.dpi'] = dpi

    # Initialize seaborn for pretty plots; use a colorblind palette for accessibility
    sns.set_theme(context='paper', palette='colorblind')


def init_output_dir(subdir: str='')-> str:
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


def sinc_derivative(x: npt.ArrayLike)-> npt.NDArray:
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


def make_taper(taper_len: int, taper_type: str)-> tuple[npt.NDArray, float]:
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


def parse_reference_sensor(ref_idx: str | npt.NDArray[np.int64] | None,
                           num_sensors:int=0)-> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """
    Accepts a reference index setting (either None, a string 'full', a scalar integer, or an array of sensor pairs)
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

    # Basic sanity checks
    if num_sensors <= 0:
        raise ValueError("num_sensors must be a positive integer.")

    # Case 1: Default common reference (last sensor)
    if ref_idx is None:
        # The default behavior is to use the last sensor as a common reference
        test_idx_vec = np.arange(num_sensors-1, dtype=np.int64)
        ref_idx_vec = np.full(num_sensors - 1, num_sensors-1, dtype=np.int64)

    # Case 2: String keyword
    elif isinstance(ref_idx, str):
        if ref_idx.lower() == 'full':
            # Generate all unique sensor pairs (i < j)
            test_idx_vec, ref_idx_vec = np.triu_indices(num_sensors, k=1)
        else:
            raise ValueError(f"Unrecognized reference index setting, {ref_idx}.")

    # Case 3: Scalar index
    elif np.isscalar(ref_idx):
        ref_idx = int(ref_idx)  # convert to base int type
        # Check for error condition
        if not (0 <= ref_idx < num_sensors):
            raise ValueError('Bad reference index; unable to parse.')

        # All sensors except the reference one
        test_idx_vec = np.r_[np.arange(ref_idx), np.arange(ref_idx + 1, num_sensors)].astype(np.int64)
        ref_idx_vec = np.full(num_sensors - 1, ref_idx, dtype=np.int64)

    # Case 4: Explicit array of pairs
    else:
        # Pair of vectors; first row is test sensors, second is reference
        arr = np.asarray(ref_idx)

        if arr.ndim != 2 or arr.shape[0] != 2:
            raise ValueError(f"ref_idx must be a (2, N) array of sensor pairs, got shape {arr.shape}.")
        if np.any((arr < 0) | (arr >= num_sensors)):
            raise ValueError("All sensor indices must be within the valid range [0, num_sensors-1].")

        test_idx_vec, ref_idx_vec = arr

        # Ensure contiguous arrays for faster downstream use
        test_idx_vec = np.ascontiguousarray(test_idx_vec)
        ref_idx_vec = np.ascontiguousarray(ref_idx_vec)

    return test_idx_vec, ref_idx_vec


def resample_covariance_matrix(cov: npt.ArrayLike,
                               test_idx: npt.ArrayLike,
                               ref_idx: npt.ArrayLike,
                               test_weights: npt.ArrayLike=None,
                               ref_weights: npt.ArrayLike=None) -> npt.NDArray:
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
    :return: two-dimensional numpy array, representing the re-sampled covariance matrix.
    """

    # Parse Inputs
    n_sensor = np.size(cov, axis=0)

    # Determine output size
    n_test, n_ref, n_pair_out = parse_ref_vec_output_size(test_idx, ref_idx, n_sensor)

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


def resample_noise(noise: npt.NDArray[np.float64],
                   test_idx: npt.NDArray[np.int64] = None,
                   ref_idx: str | npt.NDArray[np.int64] | None=None,
                   test_weights: npt.ArrayLike=None,
                   ref_weights: npt.ArrayLike=None) -> npt.NDArray:
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
            If test_idx is None, then ref_idx is passed to parse_reference_sensor and may be any valid input
            to that function.
    :param test_weights: Optional weights to apply to each measurement when resampling.
    :param ref_weights: Optional weights to apply to each measurement when resampling.
    :return: numpy ndarray of resampled noise; the first dimension has the same length as test_idx and ref_idx.
    """
    # Parse Inputs
    shp = np.shape(noise)
    n_sensor = shp[0] if len(shp) > 0 else 1
    n_sample = shp[1] if len(shp) > 1 else 1

    if test_idx is None:
        # We need to use the ref_idx
        test_idx_vec, ref_idx_vec = parse_reference_sensor(ref_idx, n_sensor)
        # Make sure test/ref indices are ints
        test_idx_vec = np.array(test_idx_vec, dtype=int)
        ref_idx_vec = np.array(ref_idx_vec, dtype=int)
    else:
        # Cast the inputs as dtype=int, they'll be used as indices later
        test_idx_vec = np.array(test_idx, dtype=int)
        ref_idx_vec = np.array(ref_idx, dtype=int)

    # Determine output size
    n_test, n_ref, n_pair_out = parse_ref_vec_output_size(test_idx_vec, ref_idx_vec, n_sensor)

    # Parse sensor weights
    shp_test_wt = int(1)
    if test_weights:
        shp_test_wt = np.size(test_weights)

    shp_ref_wt = int(1)
    if ref_weights:
        shp_ref_wt = np.size(ref_weights)

    # Function to execute at each entry of output covariance matrix
    def element_func(idx_row: npt.NDArray[np.int64]):
        idx_row = np.asarray(idx_row, dtype=np.int64)
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

        noise_ai = np.zeros((len(a_i), n_sample))
        noise_bi = np.zeros_like(noise_ai)

        mask_ai = ~np.isnan(a_i)
        mask_bi = ~np.isnan(b_i)

        if not np.all(mask_ai):
            noise_ai[mask_ai] = noise[a_i[mask_ai]]
        else:
            noise_ai = noise[a_i]

        if not np.all(mask_bi):
            noise_bi[mask_bi] = noise[b_i[mask_bi]]
        else:
            noise_bi = noise[b_i]

        res = b_i_wt * noise_bi - a_i_wt * noise_ai
        return res

    noise_out = np.fromfunction(element_func, (n_pair_out, ), dtype=float)
    return noise_out


def parse_ref_vec_output_size(test_idx_vec: npt.NDArray[np.int64],
                              ref_idx_vec: npt.NDArray[np.int64],
                              n_sensor: int)-> tuple[int, int, int]:
    """
    Validate test/reference index vectors and return their sizes.

    :param test_idx_vec: 1-D integer array of test sensor indices
    :param ref_idx_vec: 1-D integer array of reference sensor indices (same length as test_idx_vec)
    :param n_sensor: Total number of sensors (used for bounds checking)
    :return: Tuple of (n_test, n_ref, n_pair_out) where n_pair_out = max(n_test, n_ref)
    :raises TypeError: if the vectors have mismatched lengths or indices are out of range
    """
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

    return n_test, n_ref, n_pair_out

def make_pdfs(measurement_function,
              measurements: npt.ArrayLike,
              pdf_type: str='MVN',
              covariance: npt.ArrayLike = 1):
    """
    Generate a joint PDF or set of unitary PDFs representing the measurements, given the measurement_function,
    covariance matrix, and pdf_type

    The only currently supported probability distribution types are:
        'mvn': multivariate normal
        'normal': normal (each measurement is independent)

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


def make_prior(pdf_type: str, mean: npt.NDArray[np.float64], covariance: npt.NDArray[np.float64])\
        -> tuple[Callable, Callable]:
    """
    Return a prior distribution, and its Fisher Information Matrix. Currently
    only supports Gaussian (multivariate normal) as a valid type, for which
    the usage is:
    
    :param pdf_type: String dictating type of statistical prior; currently only 'gaussian' is implemented
    :param mean: numpy array (n_dim, ) with the expected value of the prior
    :param covariance: CovarianceMatrix object (with size n_dim) of the prior
    :return prior: function handle for the statistical prior; accepts (n_dim, ) or (num_batch, n_dim) inputs and returns
                   a (num_batch, ) array of probabilities.
    :return fim_prior: the Fisher Information Matrix of the prior as a constant (n_dim x n_dim) array; for a Gaussian
                   prior this equals the inverse of the covariance matrix and is independent of the input position.
    """
    if pdf_type.lower() == 'gaussian':
        rv = stats.multivariate_normal(mean=mean, cov=covariance)
        fim = rv.pdf

        # For a multivariate Gaussian, the FIM is the inverse of the covariance matrix, regardless of the input
        inv = pinvh(covariance)
        fim_prior = lambda x: inv
    else:
        raise KeyError('Unrecognized PDF type setting: ''{}'''.format(pdf_type))

    return fim, fim_prior


def print_elapsed(t_elapsed: float):
    """
    Print the elapsed time, provided in seconds.

    Nicholas O'Donoughue
    6 May 2021

    :param t_elapsed: elapsed time in seconds
    """

    hrs_elapsed = np.floor(t_elapsed / 3600)
    minutes_elapsed = np.floor((t_elapsed - 3600 * hrs_elapsed) / 60)
    secs_elapsed = t_elapsed - hrs_elapsed * 3600 - minutes_elapsed * 60

    if hrs_elapsed > 0:
        print('Elapsed Time: {} hrs, {:.2f} min'.format(hrs_elapsed, minutes_elapsed + secs_elapsed/60))
    else:
        print('Elapsed Time: {} min, {:.2f} sec'.format(minutes_elapsed, secs_elapsed))


def print_predicted(t_elapsed: float, pct_elapsed: float, do_elapsed: bool=False):
    """
    Print the elapsed and predicted time, provided in seconds.

    Nicholas O'Donoughue
    6 May 2021

    :param t_elapsed: elapsed time in seconds
    :param pct_elapsed:
    :param do_elapsed:
    """

    if do_elapsed:
        hrs_elapsed = np.floor(t_elapsed / 3600)
        minutes_elapsed = (t_elapsed - 3600 * hrs_elapsed) / 60
        if hrs_elapsed > 0:
            print('Elapsed Time: {} hrs, {:.2f} min. '.format(hrs_elapsed, minutes_elapsed), end='')
        else:
            minutes_elapsed = np.floor(minutes_elapsed)
            secs_elapsed = t_elapsed - minutes_elapsed * 60
            print('Elapsed Time: {} min, {:.2f} sec. '.format(minutes_elapsed, secs_elapsed), end='')

    t_remaining = t_elapsed * (1 - pct_elapsed) / pct_elapsed

    hrs_remaining = np.floor(t_remaining / 3600)
    minutes_remaining = (t_remaining - 3600 * hrs_remaining) / 60

    if hrs_remaining > 0:
        print('Estimated Time Remaining: {} hrs, {:.2f} min'.format(hrs_remaining, minutes_remaining))
    else:
        minutes_remaining = np.floor(minutes_remaining)
        secs_remaining = t_remaining - minutes_remaining * 60
        print('Estimated Time Remaining: {} min, {:.2f} sec'.format(minutes_remaining, secs_remaining))

def print_progress(num_total: int,
                   curr_idx: int,
                   iterations_per_marker: int,
                   iterations_per_row: int,
                   t_start: float):
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
        pct_elapsed = curr_idx / num_total
        print(f' ({pct_elapsed*100:.1f}%) ', end='')
        t_elapsed = time.perf_counter() - t_start
        print_predicted(t_elapsed, pct_elapsed, do_elapsed=True)


def broadcast_backwards(arrs: list[npt.NDArray, ], start_dim: int=0, do_broadcast: bool=False)\
        -> tuple[list[npt.NDArray, ], tuple]:
    """
    Ensure all inputs have compatible shapes for trailing-dimension broadcasting (MATLAB-style).
    New singleton dimensions are appended to the end rather than the front.

    :param arrs: List of arrays to align
    :param start_dim: Dimensions before this index are left untouched; broadcasting is computed
                      only from ``start_dim`` onward
    :param do_broadcast: If True, call ``np.broadcast_to`` on each array to fully expand them;
                         if False (default), only append singleton dimensions
    :return: Tuple of (list of reshaped/broadcast arrays, broadcast output shape from start_dim onward)
    """

    # Parse the input shapes
    orig_shapes = [np.shape(a) for a in arrs]
    max_len = max(map(len, orig_shapes), default=start_dim) # find out how long to make the new array shapes
    max_len = max(max_len, start_dim)  # ensure they're at least as long as start_dim

    # Pad arrays with singleton trailing dimensions
    output_arrs = []
    for arr, shp in zip(arrs, orig_shapes):
        ndim_missing = max_len - len(shp)
        if ndim_missing > 0:
            # Do a reshape to the correct size
            arr = np.reshape(arr, shp + (1, ) * ndim_missing)

        # Append the array to our list
        output_arrs.append(arr)

    # Compute broadcast shape after start_dim
    if max_len <= start_dim:
        out_shp = ()
    else:
        out_shp = np.broadcast_shapes(*[np.shape(a)[start_dim:] for a in output_arrs])

    if do_broadcast:
        # Perform the broadcasting; arrays will align along any dimensions after start_dim
        output_arrs = [np.broadcast_to(a, np.shape(a)[:start_dim]+out_shp) for a in output_arrs]

    # Return
    return output_arrs, out_shp


def modulo2pi(x: npt.ArrayLike)-> npt.NDArray:
    """
    Perform a 2*pi modulo operation, but with the result centered on zero, on
    the interval (-pi, pi), rather than (0, 2*pi).

    8 December 2022
    Nicholas O'Donoughue

    :param x: Numpy array-like or scalar
    :return: Modulo, centered on 0
    """

    # Shift the input so that zero is now pi
    x_shift = x + np.pi

    # Perform a modulo operation. The result is on the interval [0, 2*pi)
    x_modulo = x_shift % (2*np.pi)

    # Undo the shift, so that a zero input is now a zero output.
    # The result is on the interval [-pi, pi]
    result = x_modulo - np.pi

    return result


def remove_outliers(data: npt.ArrayLike, axis: int=0, remove_nan: bool=False)->npt.NDArray:
    """
    Remove outliers from a dataset.  If it is a vector, the outliers are individual datapoints. If it is an array,
    then outlier detection is run across the specified dimension (default=0), and any subarrays containing an outlier
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

def ensure_iterable(var, flatten: bool=False)->Iterable:
    """
    Ensure that the input is an iterable. If it is not, wrap it in a list.

    Optionally search for nested iterables and flatten them, so that all entries
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

def atleast_nd_trailing(x: npt.NDArray, n: int)-> npt.NDArray:
    """
    Replicates the functionality of numpy.atleast_2d with two changes:
    1) New dimensions are added to the end (trailing) rather than the front
    2) Accepts arbitrary number of desired dimensions
    """
    # Make sure the input is a numpy array
    x = np.array(x, copy=False, subok=True)

    # If the number of dimensions is sufficient, do nothing
    if x.ndim >= n:
        return x

    # Otherwise, use reshape to add extra dimensions
    return np.reshape(x, x.shape + (1,) * (n - x.ndim))

def print_matrix(x: npt.NDArray)->None:
    """
    Print a numpy array using MATLAB-style matrices.
    If the array has more than two dimensions, any dimension
    before the last two will be used as a batch dimension.
    """

    if np.ndim(x) > 2:
        # Batching
        for indices in itertools.product(*[range(s) for s in np.shape(x)[:-2]]):
            # indices is a tuple like (0, 0), (0, 1), etc.
            print('[',end='')
            [print('{:d},'.format(i), end='') for i in indices]
            print(':,:]')

            print_matrix(x[indices])

    # === Print the matrix pretty ===

    # 1. Find the scale
    scale = 10**np.fix(np.log10(np.amax(np.abs(x),axis=None)))

    # 2. Print the scale
    print(f"   {scale:.2g} *")

    # 3. Print the array
    with np.printoptions(formatter={'all': lambda x: (
            f"{x:8.4f}" if x >= 1e-4 else
            "  0.    ")}):
        print(np.matrix(x/scale))

def compute_sample_mean(zeta: npt.ArrayLike)-> tuple[npt.NDArray, npt.NDArray]:
    """
    Computes the sample mean across the second dimension of the input (zeta).

    Any additional dimensions (beyond the second) are assumed to be parallel
    trials, and are ignored.

    Ported from MATLAB.

    :param zeta: set of samples
    :return zeta_mean: mean value of samples
    :return zeta_mean_full: iterative mean value of samples
    """

    # Parse Input
    dims = np.shape(zeta)
    num_samples = dims[1]

    # Compute the cumulative sum
    cumsum = np.cumsum(zeta, axis=1)
    zeta_mean_full = cumsum / np.reshape(np.arange(num_samples)+1, shape=(1, num_samples))

    zeta_mean = np.squeeze(zeta_mean_full[:, -1]) # final sample mean is the result
    return zeta_mean, zeta_mean_full

def compute_sample_mean_update(zeta: npt.ArrayLike, zeta_mean_previous: npt.ArrayLike, num_samples_previous: int)->\
    tuple[npt.NDArray, int]:
    """
    Compute the sample mean iteratively, across the second dimension of zeta.

    Ported from MATLAB.

    :param zeta: new sample
    :param zeta_mean_previous: previously computed sample mean
    :param num_samples_previous: number of samples used for previous mean
    :return zeta_mean_update: new sample mean
    :return num_samples_new: total number of samples used for new mean
    """

    # Parse inputs
    dims = np.shape(zeta)
    num_samples_curr = dims[1]
    num_samples = num_samples_previous + num_samples_curr

    # Sample Mean Update
    zeta_innovation = np.sum(zeta, axis=1)/num_samples
    return zeta_mean_previous * num_samples_previous/num_samples + zeta_innovation, num_samples
