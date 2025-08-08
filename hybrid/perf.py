import numpy as np
import utils
from . import model
from utils.covariance import CovarianceMatrix


def compute_crlb(x_source, cov: CovarianceMatrix, x_aoa=None, x_tdoa=None, x_fdoa=None, v_fdoa=None,
                 do_2d_aoa=False, tdoa_ref_idx=None, fdoa_ref_idx=None, do_resample=False,
                 print_progress=False, **kwargs):
    """
    Computes the CRLB on position accuracy for source at location xs and
    a combined set of AOA, TDOA, and FDOA measurements.  The covariance
    matrix C dictates the combined variances across the three measurement
    types.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    10 March 2021

    :param x_source: Candidate source positions
    :param cov: Measurement error covariance matrix
    :param x_aoa: nDim x nAOA array of sensor positions
    :param x_tdoa: nDim x nTDOA array of TDOA sensor positions
    :param x_fdoa: nDim x nFDOA array of FDOA sensor positions
    :param v_fdoa: nDim x nFDOA array of FDOA sensor velocities
    :param do_2d_aoa: Optional boolean parameter specifying whether 1D (az-only) or 2D (az/el) AOA is being performed
    :param tdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for TDOA
    :param fdoa_ref_idx: Scalar index of reference sensor, or nDim x nPair matrix of sensor pairings for FDOA
    :param do_resample: Boolean flag; if true the covariance matrix will be resampled, using ref_idx [default=False]
    :param print_progress: Boolean flag, if true then progress updates and elapsed/remaining time will be printed to
                           the console. [default=False]
    :return crlb: Lower bound on the error covariance matrix for an unbiased AOA/TDOA/FDOA estimator (Ndim x Ndim)
    """

    n_dim, n_source = utils.safe_2d_shape(x_source)

    if n_source == 1:
        # Make sure it's got a second dimension, so that it doesn't fail when we iterate over source positions
        x_source = x_source[:, np.newaxis]

    if do_resample:
        cov = cov.resample_hybrid(x_aoa=x_aoa, x_tdoa=x_tdoa, x_fdoa=x_fdoa, do_2d_aoa=do_2d_aoa,
                                  tdoa_ref_idx=tdoa_ref_idx, fdoa_ref_idx=fdoa_ref_idx)

    # Define a wrapper for the jacobian matrix that accepts only the position 'x'
    def jacobian(x):
        j = model.jacobian(x_aoa=x_aoa, x_tdoa=x_tdoa, x_fdoa=x_fdoa, v_fdoa=v_fdoa,
                           x_source=x, do_2d_aoa=do_2d_aoa,
                           tdoa_ref_idx=tdoa_ref_idx, fdoa_ref_idx=fdoa_ref_idx)
        return j[:n_dim] # just return the jacobian w.r.t. source position

    crlb = utils.perf.compute_crlb_gaussian(x_source=x_source, jacobian=jacobian, cov=cov,
                                            print_progress=print_progress, **kwargs)

    return crlb
