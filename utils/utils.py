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


def sinc_deriv(x):
    """
    Returns the derivative of sinc(x), which is given
          y= (x * cos(x) - sin(x)) / x^2
    for x ~= 0.  When x=0, y=0.  The input is in radians.

    NOTE: The MATLAB sinc function is defined sin(pi*x)/(pi*x).  Its usage
    will be different.  For example, if calling
              y = sinc(x)
    then the corresponding derivative will be
              z = sinc_deriv(pi*x);

    Ported from MATLAB code.

    Nicholas O'Donoughue
    9 January 2021

    :param x: input, radians
    :return x_dot: derivative of sinc(x), in radians
    """

    # Apply the sinc derivative where the mask is valid, and a zero where it is not
    return np.piecewise(x,
                        [x == 0],
                        [0, lambda z: (z*np.cos(z) - np.sin(z))/(z**2)])


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
    idx_centered = lambda x: np.arange(x) - (x-1)/2
    switcher = {'uniform': lambda x: np.ones(shape=(x, )),
                'cosine': lambda x: np.sin(np.pi/(2*x)) * np.cos(np.pi * (np.arange(x)-(x-1)/2)/x),
                'hann': lambda x: np.cos(np.pi*idx_centered(x)/x)**2,
                'hamming': lambda x: .54 + .46*np.cos(2*np.pi*idx_centered(x)/x),
                'blackman-harris': lambda x: .42 + .5*np.cos(2*np.pi*idx_centered(x)/x)
                                                 + .08*np.cos(4*np.pi*idx_centered(x)/x)
                }

    # Generate the window
    taper_type = taper_type.lower()
    if taper_type in switcher:
        w = switcher[taper_type](taper_len)
    else:
        raise KeyError('Unrecognized taper type ''{}''.'.format(taper_type))

    # Set peak to 1
    w = w/np.max(np.fabs(w))

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

    if ref_idx is None:
        # Default behavior is to generate all possible sensor pairs
        perm = permutations(np.arange(num_sensors), 2)
        test_idx_vec = np.asarray([x[0] for x in perm])
        ref_idx_vec = np.asarray([x[1] for x in perm])

    elif np.isscalar(ref_idx):
        test_idx_vec = np.asarray([i for i in np.arange(num_sensors) if i != ref_idx])
        ref_idx_vec = ref_idx
    else:
        test_idx_vec = ref_idx[0, :]
        ref_idx_vec = ref_idx[1, :]

    return test_idx_vec, ref_idx_vec


def resample_covariance_matrix(cov, test_idx_vec, ref_idx_vec, test_weights=None, ref_weights=None):
    """
    Resample a covariance matrix based on a set of reference and test indices.  This assumes a linear combination
    of the test and reference vectors.  The output is an n_pair x n_pair covariance matrix for the n_pair linear
    combinations.

    In the resampled covariance matrix, the i,j-th entry is given
    [Cout]_ij = [C]_bibj + [C]_aiaj - [C]_aibj - [C]_biaj
       where:  ai, aj are the i-th and j-th reference indices
               bi, bj are the i-th and j-th test indices
               C is the input covariance matrix

    Nicholas O'Donoughue
    21 February 2021

    :param cov: n_sensor x n_sensor array representing the covariance matrix for input data.  Optional: if input is a
                1D array, it is assumed to be a diagonal matrix.
    :param test_idx_vec: n_pair x 1 array of indices for the 'test' sensor in each pair
    :param ref_idx_vec: n_pair x 1 array of indices for the 'reference' sensor in each pair
    :param test_weights: Optional, applies a scale factor to the test measurements
    :param ref_weights: Optional, applies a scale factor to the reference measurements
    :return:
    """

    # Determine the sizes
    n_sensor = np.shape(cov, axis=0)
    shp_test = np.size(test_idx_vec)
    shp_ref = np.size(ref_idx_vec)
    n_pair_out = np.fmax(shp_test, shp_ref)

    if 1 < shp_test != shp_ref > 1:
        raise TypeError("Error calling covariance matrix resample.  "
                        "Reference and test vectors must have the same shape.")

    if np.any(test_idx_vec > n_sensor) or np.any(ref_idx_vec > n_sensor):
        raise TypeError("Error calling covariance matrix resample.  "
                        "Indices exceed the dimensions of the covariance matrix.")

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
        a_i = test_idx_vec[idx_row % shp_test]
        b_i = ref_idx_vec[idx_row % shp_ref]

        if test_weights:
            a_i_wt = ref_weights[idx_row % shp_test_wt]

        if ref_weights:
            b_i_wt = ref_weights[idx_row % shp_ref_wt]

        for idx_col in np.arange(n_pair_out):
            a_j = test_idx_vec[idx_col % shp_test]
            b_j = ref_idx_vec[idx_col % shp_ref]

            if test_weights:
                a_j_wt = ref_weights[idx_col % shp_test_wt]

            if ref_weights:
                b_j_wt = ref_weights[idx_col % shp_ref_wt]

            #  [Cout]_ij = [C]_bibj + [C]_aiaj - [C]_aibj - [C]_biaj
            cov_out[idx_row, idx_col] = b_i_wt * b_j_wt * cov[b_i, b_j] + \
                                        a_i_wt * a_j_wt * cov[a_i, a_j] - \
                                        a_i_wt * b_j_wt * cov[a_i, b_j] - \
                                        b_i_wt * a_j_wt * cov[b_i, a_j]

    return cov_out


def make_pdfs(msmt_function, msmts, pdftype='MVN', covariance=1):
    """
    Generate a joint PDF or set of unitary PDFs representing the measurements
    'msmts', given the measurement function handle 'msmt_function',
    covariance matrix 'C' and pdftype

    The only currently supported pdf types are:
        'MVN'       multivariate normal
        'normal'    normal (each measurement is independent)

    Ported from MATLAB Code

    Nicholas O'Donoughue
    16 January 2021

    :param msmt_function: A single function handle that will accept an nDim x nSource array of candidate emitter
                          positions and return an nMsmt x nSource array of measurements that those emitters are
                          expected to have generated.
    :param msmts: The received measurements
    :param pdftype: The type of distribution to assume.
    :param covariance: Array of covariances (nMsmt x 1 for normal, nMxmt x nMsmt for multivariate normal)
    :return pdfs: List of function handles, each of which accepts an nDim x nSource array of candidate source
                  positions, and returns a 1 x nSource array of probabilities.
    """

    if pdftype.lower() == 'mvn' or pdftype.lower() == 'normal':
        pdfs = [lambda x: stats.multivariate_normal.pdf(msmt_function(x), mean=msmts, cov=covariance)]
    else:
        raise KeyError('Unrecognized PDF type setting: ''{}'''.format(pdftype))

    return pdfs


def print_elapsed(t_elapsed):
    """
    Print the elapsed time, provided in seconds.

    Nicholas O'Donoughue
    6 May 2021

    :param t_elapsed: elapsed time, in seconds
    """

    hrs_elapsed = np.floor(t_elapsed/3600)
    mins_elapsed = np.floor((t_elapsed - 3600 * hrs_elapsed) / 60)
    secs_elapsed = t_elapsed - hrs_elapsed*3600 - mins_elapsed * 60

    print('Elapsed Time: {} hrs, {} min, {} sec'.format(hrs_elapsed, mins_elapsed, secs_elapsed))