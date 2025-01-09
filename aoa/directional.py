import numpy as np
import matplotlib.pyplot as plt
from utils.unit_conversions import db_to_lin
from .aoa import make_gain_functions


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

    # Determine how many samples exist
    num_steering, num_samples = np.shape(s)
    
    # Initialize the outer loop to .1 radians
    this_psi_res = .1
    psi_vec = np.arange(start=min_psi, stop=max_psi + this_psi_res, step=this_psi_res)  # Set up search vector
    psi = 0.  # Initialize output

    # Iteratively search for the optimal index, until the sample spacing is less than the desired resolution
    if psi_res > this_psi_res:
        psi_res = this_psi_res

    while this_psi_res >= psi_res:
        # Find the difference between each possible AoA (psi_vec) and the measurement steering angles
        psi_diff = np.expand_dims(psi_samples, axis=1) - np.expand_dims(psi_vec, axis=0)

        # Compute the expected gain pattern for each candidate AoA value
        g_vec = np.reshape(np.asarray([g(psi) for psi in psi_diff]), newshape=psi_diff.shape)
    
        # Find the candidate AoA value that minimizes the MSE between the
        # expected and received gain patterns.

        sse = np.sum(np.sum((np.reshape(s, [num_steering, 1, num_samples])-np.expand_dims(g_vec, axis=2))**2,
                            axis=2), axis=0)
        idx_opt = np.argmin(sse)
        psi = psi_vec[idx_opt]
        
        # Set up the bounds and resolution for the next iteration
        this_psi_res = this_psi_res / 10
        min_psi = psi_vec[np.maximum(0, idx_opt - 2)]
        max_psi = psi_vec[np.minimum(np.size(psi_vec)-1, idx_opt + 2)]
        psi_vec = np.arange(start=min_psi, stop=max_psi + this_psi_res, step=this_psi_res)

    return psi


def run_example():
    """
    Example evaluation of an Adcock and Rectangular-aperture DF receiver

    Ported from MATLAB code.

    Nicholas O'Donoughue
    14 January 2021

    :return: None
    """

    # ================================ Adcock Test Script ================================
    # Create the antenna pattern generating function
    # --- NOTE --- g,g_dot take radian inputs (psi, not theta)
    d_lam = .25
    [g, g_dot] = make_gain_functions(aperture_type='adcock', d_lam=d_lam, psi_0=0.)
        
    # Generate the angular samples and true gain values
    th_true = 5.                    # degrees
    psi_true = np.deg2rad(th_true)  # radians
    psi_res = .001  # desired resolution from multi-stage search directional_df() for details.

    num_angles = 10                        # Number of angular samples
    th = np.linspace(start=-180., stop=180., num=num_angles, endpoint=False)  # evenly spaced across unit circle
    psi = np.deg2rad(th)
    x = g(psi-psi_true)  # Actual gain values
    
    # Set up the parameter sweep
    num_samples_vec = np.array([1, 10, 100])         # Number of temporal samples at each antenna test point
    snr_db_vec = np.arange(start=-20, step=2, stop=20+2)  # signal to noise ratio
    num_mc = 1000              # number of monte carlo trials at each parameter setting
    
    # Set up output variables
    out_shp = [np.size(num_samples_vec), np.size(snr_db_vec)]
    rmse_psi = np.zeros(shape=out_shp)
    crlb_psi = np.zeros(shape=out_shp)
    
    # Loop over parameters
    print('Executing Adcock Monte Carlo sweep...')
    for idx_num_samples, num_samples in enumerate(num_samples_vec.tolist()):
        this_num_mc = num_mc / num_samples
        print('\tnum_samples={}'.format(num_samples))
    
        # Generate Monte Carlo Noise with unit power
        noise_base = [np.random.normal(size=(num_angles, num_samples)) for _ in np.arange(this_num_mc)]
        
        # Loop over SNR levels
        for idx_snr, snr_db in enumerate(snr_db_vec.tolist()):
            print('.')
            
            # Compute noise power, scale base noise
            noise_amp = db_to_lin(-snr_db/2)
            
            # Generate noisy measurement
            y = [x+noise_amp*this_noise for this_noise in noise_base]
            
            # Estimate angle of arrival for each Monte Carlo trial
            psi_est = np.array([compute_df(this_y, psi, g, psi_res, min(psi), max(psi)) for this_y in y])

            # Compute RMS Error
            rmse_psi[idx_num_samples, idx_snr] = np.sqrt(np.mean((psi_est - psi_true) ** 2))

            # Compute CRLB for RMS Error
            crlb_psi[idx_num_samples, idx_snr] = crlb(snr_db, num_samples, g, g_dot, psi, psi_true)

        print('done.')

    _, _ = plt.subplots()

    for idx_num_samples, this_num_samples in enumerate(num_samples_vec):
        crlb_label = 'CRLB, M={}'.format(this_num_samples)
        mc_label = 'Simulation Result, M={}'.format(this_num_samples)

        # Plot the MC and CRLB results for this number of samples
        handle1 = plt.semilogy(snr_db_vec, np.rad2deg(np.sqrt(crlb_psi[idx_num_samples, :])), label=crlb_label)
        plt.semilogy(snr_db_vec, np.rad2deg(rmse_psi[idx_num_samples, :]), color=handle1[0].get_color(),
                     style='--', label=mc_label)

    plt.xlabel(r'$\xi$ [dB]')
    plt.ylabel('RMSE [deg]')
    plt.title('Adcock DF Performance')
    plt.legend(loc='lower left')
    
    # ============================= Reflector/Array Test Script =============================
    # Create the antenna pattern generating function
    aperture_size_wavelengths = 5
    [g, g_dot] = make_gain_functions(aperture_type='rectangular', d_lam=aperture_size_wavelengths, psi_0=0.)
        
    # Generate the angular samples and true gain values
    th_true = 5.
    psi_true = np.deg2rad(th_true)
    num_angles = 36  # number of samples
    th = np.linspace(start=-180, stop=180, num=num_angles, endpoint=False)  # evenly spaced across unit circle
    psi = np.deg2rad(th)
    x = g(psi - psi_true)  # Actual gain values
    psi_res = .001  # desired resolution from multi-stage search, see directional_df for details.

    # Set up the parameter sweep
    num_samples_vec = np.array([1, 10, 100])  # Number of temporal samples at each antenna test point
    snr_db_vec = np.arange(start=-20, step=2, stop=20 + 2)  # signal to noise ratio
    num_mc = 1000  # number of monte carlo trials at each parameter setting

    # Set up output variables
    out_shp = [np.size(num_samples_vec), np.size(snr_db_vec)]
    rmse_psi = np.zeros(shape=out_shp)
    crlb_psi = np.zeros(shape=out_shp)

    # Loop over parameters
    print('Executing Adcock Monte Carlo sweep...')
    for idx_num_samples, num_samples in enumerate(num_samples_vec.tolist()):
        this_num_mc = num_mc / num_samples
        print('\tnum_samples={}'.format(num_samples))

        # Generate Monte Carlo Noise with unit power
        noise_base = [np.random.normal(size=(num_angles, num_samples)) for _ in np.arange(this_num_mc)]

        # Loop over SNR levels
        for idx_snr, snr_db in enumerate(snr_db_vec.tolist()):
            print('.')

            # Compute noise power, scale base noise
            noise_amp = db_to_lin(-snr_db/2)

            # Generate noisy measurement
            y = [x + noise_amp * this_noise for this_noise in noise_base]

            # Estimate angle of arrival for each Monte Carlo trial
            psi_est = np.array([compute_df(this_y, psi, g, psi_res, min(psi), max(psi)) for this_y in y])

            # Compute RMS Error
            rmse_psi[idx_num_samples, idx_snr] = np.sqrt(np.mean((psi_est - psi_true) ** 2))

            # Compute CRLB for RMS Error
            crlb_psi[idx_num_samples, idx_snr] = crlb(snr_db, num_samples, g, g_dot, psi, psi_true)

        print('done.')

    _, _ = plt.subplots()

    for idx_num_samples, this_num_samples in enumerate(num_samples_vec):
        crlb_label = 'CRLB, M={}'.format(this_num_samples)
        mc_label = 'Simulation Result, M={}'.format(this_num_samples)

        # Plot the MC and CRLB results for this number of samples
        handle1 = plt.semilogy(snr_db_vec, np.rad2deg(np.sqrt(crlb_psi[idx_num_samples, :])), label=crlb_label)
        plt.semilogy(snr_db_vec, np.rad2deg(rmse_psi[idx_num_samples, :]), color=handle1[0].get_color(),
                     style='--', label=mc_label)

    plt.xlabel(r'$\xi$ [dB]')
    plt.ylabel('RMSE [deg]')
    plt.title('Rectangular Array DF Performance')
    plt.legend(loc='lower left')
