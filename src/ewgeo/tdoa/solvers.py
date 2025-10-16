import numpy as np
import numpy.typing as npt
from scipy.linalg import pinvh

from ewgeo.utils.constants import speed_of_light
from ewgeo.utils.covariance import CovarianceMatrix


def chan_ho(x_sensor: npt.ArrayLike,
            zeta: npt.ArrayLike,
            cov: CovarianceMatrix,
            ref_idx=None,
            do_resample: bool=False,
            variance_is_toa: bool=False):
    """
    Computes the Chan-Ho solution for TDOA processing.

    Ref:  Y.-T. Chan and K. Ho, “A simple and efficient estimator for 
          hyperbolic location,” IEEE Transactions on signal processing, 
          vol. 42, no. 8, pp. 1905–1915, 1994.

    Ported from MATLAB Code
    
    Nicholas O'Donoughue
    11 March 2021
    
    :param x_sensor: sensor positions [m]
    :param zeta: range-difference measurements [m]
    :param cov: error covariance matrix for range-difference [m]
    :param ref_idx: index of the reference sensor for TDOA processing. Default is the last sensor. Must be scalar.
    :param do_resample: Boolean flag; if true the covariance matrix will be resampled, using ref_idx
    :param variance_is_toa: Boolean flag; if true then the input covariance matrix is in units of s^2; if false, then
    it is in m^2
    :return: estimated source position [m]
    """

    # TODO: Debug

    # Resample the covariance matrix, if needed
    cov = preprocess_cov(cov=cov, do_resample=do_resample, variance_is_toa=variance_is_toa, ref_idx=ref_idx)

    # Accept an arbitrary reference position
    n_dims, n_sensor = np.shape(x_sensor)
    if ref_idx is not None and ref_idx != n_sensor-1:
        # Throw an error if there are multiple reference sensors
        assert np.size(ref_idx) == 1, 'The Chan-Ho solver currently requires a single reference sensor.'

        # Re-arrange the sensors
        sort_idx = [i for i in range(n_sensor) if i != ref_idx]
        sort_idx.append(int(ref_idx))
        x_sensor = x_sensor[:, sort_idx]

        # Note: We don't need to rearrange cov, since
        # the following code block will handle its resampling to account
        # for the test and reference indices.

    # Stage 1: Initial Position Estimate
    # Compute system matrix overline(A) according to 11.23

    # Compute shifted measurement vector overline(y) according to 11.24
    # NOTE: In python, indexing with [-1] is the final entry, while [:-1] is all
    # entries *except* the last entry in a list or array.
    r = np.sqrt(np.sum(np.fabs(x_sensor) ** 2, axis=0))
    y1 = (zeta ** 2 - r[:-1] ** 2 + r[-1] ** 2)
    last_sensor = x_sensor[:, -1]
    dx = x_sensor[:, :-1] - last_sensor[:, np.newaxis]
    g1 = -2*np.concatenate((np.transpose(dx), zeta[:, np.newaxis]), axis=1)

    # Compute initial position estimate overline(theta) according to 11.25
    b = np.eye(n_sensor-1)
    th1, cov_mod = _chan_ho_theta(cov, b, g1, y1)

    # w1 = b.dot(cov).dot(np.transpose(np.conjugate(b)))
    # w1_inv = np.linalg.pinv(w1)
    # g1w = np.transpose(np.conjugate(g1)).dot(w1_inv)
    # g1wg = g1w.dot(g1)
    # g1wy = g1w.dot(y1)
    # th1 = np.linalg.lstsq(g1wg, g1wy, rcond=None)[0]

    # Refine sensor estimate
    for _ in np.arange(3):
        ri_hat = np.sqrt(np.sum((x_sensor - th1[:-1, np.newaxis]) ** 2, axis=0))

        # Re-compute initial position estimate overline(theta) according to 11.25
        b = 2*np.diag(ri_hat[0:-1])
        th1, cov_mod = _chan_ho_theta(cov, b, g1, y1)
        # w1 = b.dot(cov).dot(np.transpose(np.conjugate(b)))
        # w1_inv = np.linalg.pinv(w1)
        # g1w = np.transpose(np.conjugate(g1)).dot(w1_inv)
        # g1wg = g1w.dot(g1)
        # g1wy = g1w.dot(y1)
        # th1 = np.linalg.lstsq(g1wg, g1wy, rcond=None)[0]

    th1p = np.subtract(th1, np.concatenate((x_sensor[:, -1], [0]), axis=0))

    # Stage 2: Account for Parameter Dependence
    y2 = th1**2
    g2 = np.concatenate((np.eye(n_dims), np.ones(shape=(1, n_dims))), axis=0)
    b2 = 2*np.diag(th1p)

    # Compute final parameter estimate overline(theta)' according to 13.32
    th2 = _chan_ho_theta_hat(cov_mod, b2, g1, g2, y2)

    # g1wg1 = np.transpose(np.conjugate(g1)).dot(cov_mod).dot(g1)
    # w2 = np.linalg.pinv(np.conjugate(np.transpose(b2))).dot(g1wg1).dot(np.linalg.pinv(b2))
    # g2w = np.transpose(np.conjugate(g2)).dot(w2)
    # g2wg2 = g2w.dot(g2)
    # g2wy = g2w.dot(y2)
    # th2 = np.linalg.lstsq(g2wg2, g2wy, rcond=None)[0]

    # Compute position estimate overline(x)' according to 13.33
    # x_prime1 = x0(:,end)+sqrt(th_prime);
    # x_prime2 = x0(:,end)-sqrt(th_prime);
    #
    # offset1 = norm(x_prime1-th(1:end-1));
    # offset2 = norm(x_prime2-th(1:end-1));
    #
    # if offset1 <= offset2
    #     x = x_prime1;
    # else
    #     x = x_prime2;
    # end

    x = np.sign(np.diag(th1[:-1])).dot(np.sqrt(th2)) + x_sensor[:, -1]

    return x


def _chan_ho_theta(cov: CovarianceMatrix, b, g, y):
    """
    Compute position estimate overline(theta) according to 11.25.
    This is an internal function called by chan_ho.

    :param cov: measurement-level covariance matrix
    :param b:
    :param g: system matrix
    :param y: shifted measurement vector
    :return theta: parameter vector (eq 11.26)
    :return cov_mod: modified covariance matrix (eq 11.28)
    """

    cov_mod = b @ cov.cov @ b.T  # BCB', eq 11.28
    cov_mod_inv = pinvh(cov_mod)

    # Assemble matrix products g^H*w_inv*g, and g^H*w_inv*y
    gw = g.T @ cov_mod_inv
    gwg = gw @ g
    gwy = gw @ y

    theta = np.linalg.lstsq(gwg, gwy, rcond=None)[0]

    return theta, cov_mod


def _chan_ho_theta_hat(cov_mod, b2, g1, g2, y2):
    """

    :param cov_mod: measurement-level covariance matrix
    :param b2:
    :param g1:
    :param g2:
    :param y2: shifted measurement vector
    :return theta: parameter vector (eq 11.36)
    """

    # TODO: Debug and verify
    g1wg1 = g1.T @ cov_mod @ g1
    w2 = pinvh(b2.T @ pinvh(g1wg1) @ b2)
    g2w = g2.T @ w2
    g2wg2 = g2w @ g2
    g2wy = g2w @ y2

    theta = np.linalg.lstsq(g2wg2, g2wy, rcond=None)[0]

    return theta


def preprocess_cov(cov: CovarianceMatrix, ref_idx=None, do_resample=False, variance_is_toa=False):
    if variance_is_toa:
        # Convert from TOA/TDOA to ROA/RDOA -- copy to a new object for sanity's sake
        cov_out = cov.multiply(speed_of_light ** 2, overwrite=False)
    else:
        cov_out = cov.copy()

    if do_resample:
        cov_out = cov_out.resample(ref_idx=ref_idx)

    return cov_out