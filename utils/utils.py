import numpy as np
import scipy.stats as stats
from .unit_conversions import lin_to_db
from itertools import permutations
import os


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
        test_idx_vec = np.asarray([i for i in np.arange(num_sensors - 1)])
        ref_idx_vec = np.array([num_sensors - 1])

    elif ref_idx == 'full':
        # Generate all possible sensor pairs
        perm = permutations(np.arange(num_sensors), 2)
        test_idx_vec = np.asarray([x[0] for x in perm])
        ref_idx_vec = np.asarray([x[1] for x in perm])

    elif np.isscalar(ref_idx):
        # Scalar reference index, use all other sensors as test sensors
        test_idx_vec = np.asarray([i for i in np.arange(num_sensors) if i != ref_idx])
        ref_idx_vec = ref_idx
    else:
        # Pair of vectors; first row is test sensors, second is reference
        test_idx_vec = ref_idx[0, :]
        ref_idx_vec = ref_idx[1, :]

    return test_idx_vec, ref_idx_vec


def resample_covariance_matrix(cov, test_idx_vec, ref_idx_vec=None, test_weights=None, ref_weights=None):
    """
    Resample a covariance matrix based on a set of reference and test indices.  This assumes a linear combination
    of the test and reference vectors.  The output is an n_pair x n_pair covariance matrix for the n_pair linear
    combinations.

    In the resampled covariance matrix, the i,j-th entry is given
    [cov_out]_ij = [cov]_bi,bj + [cov]_ai,aj - [cov]_ai,bj - [cov]_bi,aj
       where:  a_i, a_j are the i-th and j-th reference indices
               b_i, b_j are the i-th and j-th test indices
               C is the input covariance matrix

    If any elements of the test_idx_vec or ref_idx_vec are set to nan, then those elements will be ignored for
    covariance matrix resampling.  This is used to correspond either to perfect (noise-free) measurements, or to single
    sensor measurements, such as AoA, that do not require comparison with a second sensor measurement.

    Nicholas O'Donoughue
    21 February 2021

    :param cov: n_sensor x n_sensor array representing the covariance matrix for input data.  Optional: if input is a
                1D array, it is assumed to be a diagonal matrix.
    :param test_idx_vec: n_pair x 1 array of indices for the 'test' sensor in each pair -- or -- a valid reference
                         index input to parse_reference_sensor.
    :param ref_idx_vec: n_pair x 1 array of indices for the 'reference' sensor in each pair.  Set to None (or do not
                        provide) if test_idx_vec is to be passed to parse_reference_sensor.
    :param test_weights: Optional, applies a scale factor to the test measurements
    :param ref_weights: Optional, applies a scale factor to the reference measurements
    :return:
    """

    # Determine the sizes
    n_sensor = np.size(cov, axis=0)
    n_test = np.size(test_idx_vec)
    n_ref = np.size(ref_idx_vec)
    n_pair_out = np.fmax(n_test, n_ref)

    if 1 < n_test != n_ref > 1:
        raise TypeError("Error calling covariance matrix resample.  "
                        "Reference and test vectors must have the same shape.")

    if np.any(test_idx_vec > n_sensor) or np.any(ref_idx_vec > n_sensor):
        raise TypeError("Error calling covariance matrix resample.  "
                        "Indices exceed the dimensions of the covariance matrix.")

    # Parse reference and test index vector
    if ref_idx_vec is None:
        # Only one was provided; it must be fed to parse_reference_sensor to generate the matched pair of vectors
        test_idx_vec, ref_idx_vec = parse_reference_sensor(test_idx_vec, n_sensor)

    # Parse sensor weights
    shp_test_wt = 1
    if test_weights:
        shp_test_wt = np.size(test_weights)

    shp_ref_wt = 1
    if ref_weights:
        shp_ref_wt = np.size(ref_weights)

    # Initialize output
    cov_out = np.zeros((n_pair_out, n_pair_out))

    a_i_wt = 1.
    a_j_wt = 1.
    b_i_wt = 1.
    b_j_wt = 1.

    # Step through reference sensors
    for idx_row in np.arange(n_pair_out):
        a_i = test_idx_vec[idx_row % n_test]
        b_i = ref_idx_vec[idx_row % n_ref]

        # print('\tRow: {}, a_i: {}, b_i: {}'.format(idx_row, a_i, b_i))

        if test_weights:
            a_i_wt = test_weights[idx_row % shp_test_wt]

        if ref_weights:
            b_i_wt = ref_weights[idx_row % shp_ref_wt]

        for idx_col in np.arange(n_pair_out):
            a_j = test_idx_vec[idx_col % n_test]
            b_j = ref_idx_vec[idx_col % n_ref]

            # print('\tCol: {}, a_j: {}, b_j: {}'.format(idx_col, a_j, b_j))

            if test_weights:
                a_j_wt = test_weights[idx_col % shp_test_wt]

            if ref_weights:
                b_j_wt = ref_weights[idx_col % shp_ref_wt]

            # Parse input covariances
            if np.isnan(b_i) or np.isnan(b_j):
                cov_bibj = 0.
            else:
                cov_bibj = cov[b_i, b_j]

            if np.isnan(a_i) or np.isnan(a_j):
                cov_aiaj = 0.
            else:
                cov_aiaj = cov[a_i, a_j]

            if np.isnan(a_i) or np.isnan(b_j):
                cov_aibj = 0.
            else:
                cov_aibj = cov[a_i, b_j]

            if np.isnan(b_i) or np.isnan(a_j):
                cov_biaj = 0.
            else:
                cov_biaj = cov[b_i, a_j]

            #  [cov_out]_ij = [cov]_bi,bj + [cov]_ai,aj - [cov]_ai,bj - [cov]_bi,aj
            # Put it together with the weights
            cov_out[idx_row, idx_col] = b_i_wt * b_j_wt * cov_bibj + \
                                        a_i_wt * a_j_wt * cov_aiaj - \
                                        a_i_wt * b_j_wt * cov_aibj - \
                                        b_i_wt * a_j_wt * cov_biaj

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
        this_cov = np.squeze(covariance[:, :, idx_matrix])

        # Eigen-decomposition
        lam, v = np.linalg.eig(this_cov)

        # Initialize the diagonal loading term
        d = epsilon * np.eye(N=dim)

        # Repeat until the smallest eigenvalue is larger than epsilon
        while np.amin(lam) < epsilon:
            # Add the diagonal loading term
            this_cov += d

            # Re-examine the eigenvalue
            lam, v = np.linalg.eig(this_cov)

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
        pdfs = [lambda x: stats.multivariate_normal.pdf(measurement_function(x), mean=measurements, cov=covariance)]
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

    print('Elapsed Time: {} hrs, {} min, {} sec'.format(hrs_elapsed, minutes_elapsed, secs_elapsed))


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

    if x.ndim > 2:
        # 3D array, drop anything after the second dimension
        dim1, dim2, _ = np.shape(x)

    elif x.ndim > 1:
        # 2D array
        dim1, dim2 = np.shape(x)

    else:
        # 1D array
        dim1 = np.size(x)
        dim2 = 1

    return dim1, dim2


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
    dims = [x + x_max * np.linspace(start=-x_max, stop=x_max*(1+n)/n, num=n) for (x, x_max, n)
            in zip(x_ctr, max_offset, n_elements)]

    # Use meshgrid expansion; each element of x_grid is now a full n_dim dimensioned grid
    x_grid = np.meshgrid(*dims)

    # Rearrange to a single 2D array of grid locations (n_dim x N)
    x_set = np.asarray([x.flatten() for x in x_grid]).T

    return x_set, x_grid, n_elements
