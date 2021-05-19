import numpy as np
import matplotlib.pyplot as plt

def beamscan(x, v, psi_max=np.pi/2, num_points=101):
    """
    # Generates a beamscan image for N_pts equally spaced angular coordinates
    # from -psi_max to psi_max (in radians), given the input data x, and array
    # steering vector v

    Ported from MATLAB Code

    Nicholas O'Donoughue
    18 January 2021

    :param x: N x M data vector
    :param v: Steering vector function that returns N point steering vector for each input (in radians).
    :param psi_max: Maximum steering angle (radians)
    :param num_points: Number of steering angles to compute
    :return p: Power image (1 x N_pimts) in linear units
    :return psi_vec: Vector of scan angles computed (in radians)
    """

    # Generate scan vector
    psi_vec = np.linspace(start=-1, stop=1, num=num_points) * psi_max

    # Parse inputs
    num_array_elements, num_samples = np.shape(x)

    # Generate steering vectors
    steering_vectors = v(psi_vec)/np.sqrt(num_samples)  # num_array_elements x num_points

    # Steer each of the num_samples data samples
    # - take the magnitude squared
    # - compute the mean across M snapshots
    p = np.ravel(np.sum(np.abs(np.conjugate(x).T.dot(steering_vectors))**2, axis=0)/num_array_elements)

    return p, psi_vec


def beamscan_mvdr(x, v, psi_max=np.pi/2, num_points=101):
    """
    Generates a beamscan image for N_pts equally spaced angular coordinates
    from -psi_max to psi_max (in radians), given the input data x, and array
    steering vector v

    Ported from MATLAB Code

    Nicholas O'Donoughue
    18 January 2021

    :param x: N x M data vector
    :param v: Steering vector function that returns N point steering vector for each input (in radians).
    :param psi_max: Maximum steering angle (radians)
    :param num_points: Number of steering angles to compute
    :return p: Power image (1 x N_pts) in linear units
    :return psi_vec: Vector of scan angles computed (in radians)
    """

    # Generate scan vector
    psi_vec = np.linspace(start=-1, stop=1, num=num_points) * psi_max

    # Compute the sample covariance matrix
    n, m = np.shape(x)
    covariance = np.cov(x)

    # Pre-compute covariance matrix inverses
    c_pinv = np.linalg.pinv(covariance)

    # Steer each of the M data samples
    p = np.zeros(shape=(num_points, ))
    for idx_psi in np.arange(num_points):
        this_v = v(psi_vec[idx_psi])/np.sqrt(n)  # N x 1

        p[idx_psi] = 1/np.abs(np.conjugate(this_v).T.dot(c_pinv).dot(this_v))

    return p, psi_vec


def music(x, steer, num_sig_dims=0, max_psi=np.pi / 2, num_points=101):
    """
    Generates a MUSIC-based image for N_pts equally spaced angular
    coordinates from -psi_max to psi_max (in radians), given the input
    data x, array steering vector v, and optional number of signals D.

    If left blank, or set to zero, the value D will be estimated using a
    simple algorithm that counts the number of eigenvalues greater than twice
    the minimum eigenvalue.  This will break down in low SNR scenarios.

    Ported from MATLAB Code.

    Nicholas O'Donoughue
    18 January 2021

    :param x: N x M data vector
    :param steer: Steering vector function that returns N point steering vector for each input (in radians).
    :param num_sig_dims: Number of signals [optional, set to zero to automatically estimate D based on a threshold
                         eigenvalue twice the minimum eigenvalue]
    :param max_psi: Maximum steering angle (radians)
    :param num_points: Number of steering angles to compute
    :return p: Power image (1 x N_pts) in linear units
    :return psi_vec: Vector of scan angles computed (in radians)
    """

    # TODO: Double check.  The second MC simulation showed a loss w.r.t. CRB and beamscan.

    # Compute the sample covariance matrix
    n, m = np.shape(x)
    # covariance = np.cov(x, bias=True)
    covariance = np.zeros((n, n), dtype=complex)
    for idx in np.arange(m):
        this_x = np.expand_dims(x[:, idx], axis=1)
        tmp = this_x @ np.conjugate(this_x.T)
        covariance += tmp/m

    # Perform Eigendecomposition of the covariance matrix
    lam, eig_vec = np.linalg.eig(covariance)

    # Sort the eigenvectors and eigenvectors
    idx_sort = np.flip(np.argsort(np.abs(lam)))  # np.argsort operates in ascending order, reverse it
    # lam_sort = np.take_along_axis(lam, idx_sort, axis=0)
    eig_vec_sort = np.take_along_axis(eig_vec, np.expand_dims(idx_sort, axis=0), axis=1)

    # Isolate Noise Subspace
    if num_sig_dims != 0:
        eig_vec_noise = eig_vec_sort[:, num_sig_dims:]
    else:
        # We need to estimate D first

        # Assume that the noise power is given by the smallest eigenvalue
        noise = lam[idx_sort[-1]]

        # Set a threshold of 2x the noise level; and find the eigenvalue that
        # first cuts above it
        num_sig_dims = np.argwhere(lam >= 2 * noise)[-1]

        eig_vec_noise = eig_vec_sort[:, num_sig_dims:]

    # Noise Subspace Projection
    proj = eig_vec_noise.dot(np.conjugate(eig_vec_noise).T)

    # Generate steering vectors
    psi_vec = np.linspace(start=-1, stop=1, num=num_points) * max_psi

    p = np.zeros(shape=(num_points, ))
    for idx_pt in np.arange(num_points):
        # Project the steering vector onto the noise subspace
        vv = steer(psi_vec[idx_pt]) / np.sqrt(n)
        q = np.conjugate(vv).T.dot(proj).dot(vv)

        # Invert the power
        p[idx_pt] = 1/np.abs(q)

    return p, psi_vec
