import numpy as np

from ewgeo.utils.unit_conversions import db_to_lin


def crlb(snr, num_samples, g, g_dot, psi_samples, psi_true):
    """
    Computes the CRLB for a directional antenna with amplitude measurements taken at a series of angles.  Supports M
    measurements from each of N different angles.

    If there are multiple true angles of arrival provided (psi_true), then the CRLB is computed independently for each
    one.
    
    Ported from MATLAB Code.
    
    Nicholas O'Donoughue
    14 January 2021
    
    :param snr: Signal-to-Noise ratio [dB]
    :param num_samples: Number of samples for each antenna position
    :param g: Function handle to g(psi)
    :param g_dot: Function handle to g_dot(psi)
    :param psi_samples: The sampled steering angles (radians)
    :param psi_true: The true angle of arrival (radians)
    :return crlb: Lower bound on the Mean Squared Error of an unbiased estimation of psi (radians)
    """
    
    # Convert SNR from dB to linear units
    snr_lin = db_to_lin(snr)
    
    # Evaluate the antenna pattern and antenna gradient at each of the steering angles sampled.
    g_vec = np.array([g(psi-psi_true) for psi in psi_samples])
    g_dot_vec = np.array([g_dot(psi-psi_true) for psi in psi_samples])
    
    # Pre-compute steering vector inner products
    g_g = np.sum(g_vec * g_vec, axis=0)
    g_dot_g = np.sum(g_vec * g_dot_vec, axis=0)
    g_dot_g_dot = np.sum(g_dot_vec * g_dot_vec, axis=0)
    
    # Compute CRLB for each true angle theta
    jacobian = 2 * num_samples * snr_lin * (g_dot_g_dot - g_dot_g ** 2 / g_g)  # Eq 7.25
    return 1./jacobian  # 1 x num_angles


def compute_df(s, psi_samples, g, psi_res=0.1, min_psi=-np.pi, max_psi=np.pi):
    """
    Computes an estimate of the angle of arrival psi (in radians) for a set of amplitude measurements s, taken at
    various steering angles psi_samples

    Ported from MATLAB Code.

    Nicholas O'Donoughue
    14 January 2021

    :param s: Set of num_samples measurements taken at each of num_steering steering angles.
    :param psi_samples: Steering angles at which measurements were taken [radians]
    :param g: Function handle to gain equation, g(psi)
    :param psi_res: Desired resolution for output estimate [default = .1]
    :param min_psi: Lower bound on valid region for psi [default = -pi]
    :param max_psi: Upper bound on valid region for psi [default = pi]
    :return: Estimated angle of arrival [radians]
    """

    # Initialize the outer loop to .1 radians
    this_psi_res = .1
    psi = 0.  # Initialize output

    # Ensure at least one loop
    psi_res = min(psi_res, this_psi_res)

    while this_psi_res >= psi_res:
        psi_vec = np.arange(start=min_psi, stop=max_psi + this_psi_res, step=this_psi_res)  # Set up search vector

        # Find the difference between each possible AoA (psi_vec) and the measurement steering angles
        psi_diff = psi_samples[:, np.newaxis] - psi_vec[np.newaxis, :]

        # Compute the expected gain pattern for each candidate AoA value
        g_vec = g(psi_diff)
    
        # Find the candidate AoA value that minimizes the MSE between the
        # expected and received gain patterns.
        sse = np.sum(np.absolute(s[:, :, np.newaxis]-g_vec[:, np.newaxis, :])**2, axis=(0, 1))
        idx_opt = np.argmin(sse)
        psi = psi_vec[idx_opt]
        
        # Set up the bounds and resolution for the next iteration
        this_psi_res /= 10
        idx_min = max(0, idx_opt-4)
        idx_max = min(len(psi_vec)-1, idx_opt+4)
        min_psi = psi_vec[idx_min]
        max_psi = psi_vec[idx_max]

    return psi


