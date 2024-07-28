import numpy as np


def crlb_det(covariance, noise_power, psi_vec, num_snapshots, v, v_dot):
    """
    Computes the deterministic CRLB for array-based DOA, according to section 8.4.

    Ported from MATLAB Code.

    Nicholas O'Donoughue
    18 January 2021

    :param covariance: Source signal covariance matrix
    :param noise_power: Noise power
    :param psi_vec: Array of D steering angles (in radians) for each source
    :param num_snapshots: Number of temporal snapshots taken
    :param v: Function handle to steering vector v(psi)
    :param v_dot: Function handle to steering vector gradient dv(psi)/dpsi
    :return crlb: CRLB matrix (DxD) for angle of arrival estimation of each source, in radians^2
                  Note: multiply by (180/pi)^2 to convert to degrees.
    """

    # Apply source steering angles to steering vector and steering vector gradient function handles.
    steer = v(psi_vec)                     # N x D
    steer_gradient = v_dot(psi_vec)                 # N x D, equation 8.78

    # Construct the QR Decomposition of the vector subspace V, and use it to form the projection matrix orthogonal
    # to the subspace spanned by V
    q, r = np.linalg.qr(steer)
    proj = q.dot(np.conjugate(q).T)                    # N x N
    proj_ortho = np.eye(N=np.shape(proj)[0]) - proj      # N x N
    h = np.conjugate(steer_gradient).T.dot(proj_ortho).dot(steer_gradient)       # D x D, equation 8.77

    # Build spectral matrix from linear SNR
    xi = covariance / noise_power                     # D x D

    # Scaled CRLB, use an if/else to handle scalar cases separately
    if np.size(covariance) > 1:
        c = np.linalg.pinv(np.real(xi*h.T))
    else:
        c = 1/np.real(xi*h)

    return c / (2 * num_snapshots)


def crlb_stochastic(covariance, noise_power, psi_vec, num_snapshots, v, v_dot):
    """
    Computes the stochastic CRLB for array-based DOA, according to section 8.4.

    Ported from MATLAB Code.

    Nicholas O'Donoughue
    18 January 2021

    :param covariance: Source signal covariance matrix
    :param noise_power: Noise power
    :param psi_vec: Array of D steering angles (in radians) for each source
    :param num_snapshots: Number of temporal snapshots taken
    :param v: Function handle to steering vector v(psi)
    :param v_dot: Function handle to steering vector gradient dv(psi)/dpsi
    :return crlb: CRLB matrix (DxD) for angle of arrival estimation of each source, in radians^2
                  Note: multiply by (180/pi)^2 to convert to degrees.
    """

    # Apply source steering angles to steering vector and steering vector gradient function handles.
    steer = v(psi_vec)  # N x D
    steer_gradient = v_dot(psi_vec)  # N x D, equation 8.78
    num_elements, num_sources = np.shape(steer)

    # Construct the QR Decomposition of the vector subspace V, and use it to form the projection matrix orthogonal
    # to the subspace spanned by V
    q, r = np.linalg.qr(steer)
    proj = q.dot(np.conjugate(q).T)  # N x N
    proj_ortho = np.eye(N=num_elements) - proj  # N x N
    h = np.conjugate(steer_gradient).T.dot(proj_ortho).dot(steer_gradient)  # D x D, equation 8.77

    # Build the spectral matrix from linear SNR
    a = np.conjugate(r).T.dot(r).dot(covariance) / noise_power
    if num_sources > 1:
        b = np.dot(np.linalg.lstsq(np.eye(N=num_sources) + a, covariance), a)
    else:
        b = covariance * a / (1+a)

    # CRLB, ex 8.75
    return (noise_power / (2 * num_snapshots)) * np.linalg.pinv(np.real(b*h.T))        # D x D
