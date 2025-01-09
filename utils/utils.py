import numpy as np
from scipy import stats
from .unit_conversions import lin_to_db
from itertools import permutations
import seaborn as sns
import matplotlib.pyplot as plt
import os
import time


def init_plot_style(dpi=400):
    """
    Initialize plotting styles, including output resolution
    """

    # Specify dpi for figure saving; matplotlib default is 200. It's pretty low-res, so we're using a default of 300
    plt.rcParams['figure.dpi'] = dpi

    # Initialize seaborn for pretty plots
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


def parse_reference_sensor(ref_idx, num_sensors):
    """
    Accepts a reference index setting (either None, a scalar integer, or a 2 x N array of sensor pairs),
    and returns matching vectors for test and reference indices.

    :param ref_idx: reference index setting
    :param num_sensors: Number of available sensors
    :return test_idx_vec:
    :return ref_idx_vec:
    """

    # Debug info
    # print('Parsing reference sensor...')
    # print('\tref_idx: {}'.format(ref_idx))
    # print('\tnum_sensors: {}'.format(num_sensors))

    if ref_idx is None:
        # Default behavior is to use the last sensor as a common reference
        # test_idx_vec = np.arange(num_sensors)#np.asarray([i for i in np.arange(num_sensors - 1)])

        # do num_sensors=1 b/c we don't want to compare ref sensor to ref sensor
        test_idx_vec = np.arange(num_sensors-1)  # np.asarray([i for i in np.arange(num_sensors - 1)])
        ref_idx_vec = (num_sensors - 1) * np.ones_like(test_idx_vec)

    elif ref_idx == 'full':
        # Generate all possible sensor pairs
        perm = permutations(np.arange(num_sensors), 2)
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


def resample_covariance_matrix(cov, test_idx_vec, ref_idx_vec=None, test_weights=None, ref_weights=None):

    # Parse Inputs
    n_sensor = np.size(cov, axis=0)
    if test_idx_vec is None:
        # Default behavior, if not specified, is to use the final sensor as the reference
        test_idx_vec = n_sensor - 1

    # Parse reference and test index vector
    if ref_idx_vec is None:
        # Only one was provided; it must be fed to parse_reference_sensor to generate the matched pair of vectors
        test_idx_vec, ref_idx_vec = parse_reference_sensor(test_idx_vec, n_sensor)

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
    def element_func(idx_row, idx_col):
        idx_row = idx_row.astype(int)
        idx_col = idx_col.astype(int)
        a_i = test_idx_vec[idx_row % n_test]
        b_i = ref_idx_vec[idx_row % n_ref]
        a_j = test_idx_vec[idx_col % n_test]
        b_j = ref_idx_vec[idx_col % n_ref]
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

        cov_aiaj = cov[a_i, a_j]
        cov_aibj = np.zeros_like(cov_aiaj)
        cov_biaj = cov_aibj.copy()
        cov_bibj = cov_aibj.copy() 
        
        mask_ai = np.isnan(a_i)
        mask_aj = np.isnan(a_j)
        mask_bi = np.isnan(b_i)
        mask_bj = np.isnan(b_j)

        mask_bibj = ~np.logical_or(mask_bi, mask_bj)
        if not np.all(mask_bibj):
            cov_bibj[mask_bibj] = cov[b_i[mask_bibj].astype(int), b_j[mask_bibj].astype(int)]
        else: 
            cov_bibj = cov[b_i.astype(int), b_j.astype(int)]
            
        mask_aibj = ~np.logical_or(mask_ai, mask_bj)
        if not np.all(mask_aibj):
            cov_aibj[mask_aibj] = cov[a_i[mask_aibj].astype(int), b_j[mask_aibj].astype(int)]
        else:
            cov_aibj = cov[a_i.astype(int), b_j.astype(int)]

        mask_biaj = ~np.logical_or(mask_bi, mask_aj)
        if not np.all(mask_biaj):
            cov_biaj[mask_biaj] = cov[b_i[mask_biaj].astype(int), a_j[mask_biaj].astype(int)]
        else:
            cov_biaj = cov[b_i.astype(int), a_j.astype(int)]

        res = b_i_wt * b_j_wt * cov_bibj + \
            a_i_wt * a_j_wt * cov_aiaj - \
            a_i_wt * b_j_wt * cov_aibj - \
            b_i_wt * a_j_wt * cov_biaj
        # raise ValueError('mo')
        return res
    cov_out = np.fromfunction(element_func, (n_pair_out, n_pair_out), dtype=float)
    return cov_out


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
        this_cov = np.squeeze(covariance[:, :, idx_matrix])

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
        cov_out[:, :, idx_matrix] = this_cov

    return cov_out


def make_pdfs(measurement_function, measurements, pdf_type='MVN', covariance=1):
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

        print('Elapsed Time: {} hrs, {:.2f} min. '.format(hrs_elapsed, minutes_elapsed), end='')

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
        return [0, 0]
        
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
    :return x_set: n-tuple of 1D axes for each dimension
    :return x_grid: n-tuple of ND coordinates for each dimension.
    :return out_shape:  tuple with the size of the generated grid
    """

    n_dim = np.size(x_ctr)

    if n_dim < 1 or n_dim > 3:
        raise AttributeError('Number of spatial dimensions must be between 1 and 3')

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
    x_set = np.asarray([x.flatten() for x in x_grid]).T

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
    # Result is on the interval [-pi, pi)
    result = x_modulo - np.pi

    return result
