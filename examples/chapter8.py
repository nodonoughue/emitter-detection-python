import numpy as np
import matplotlib.pyplot as plt
from utils import constants
from utils.unit_conversions import lin_to_db, db_to_lin
from utils import print_elapsed, print_progress
import prop
import array_df
import time
import os
from scipy.io import loadmat, savemat
from scipy.signal import find_peaks


def run_all_examples(generate_data=True):
    """
    Run all chapter 8 examples and return a list of figure handles

    :return figs: list of figure handles
    """

    # Random Number Generator
    rng = np.random.default_rng(0)

    if generate_data:
        generate_ex1_data(rng)

    fig1 = example1(rng)
    fig2 = example2(rng)

    return [fig1, fig2]


def generate_ex1_data(rng=np.random.default_rng()):
    """
    Generate data for Example 8.1, which will be stored in examples/chapter8_data.csv, and for Problems 8.5 and 8.6,
    which will be stored in hw/problem8_5.csv and hw/problem8_6.csv.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    5 May 2021

    :param rng: random number generator
    """
    # ===== Generate Data for Example 8.1 =====
    # Initialize PTT Radio Parameters
    tx_pwr = 10
    tx_gain = 0
    tx_loss = 3
    # tx_bw_hz = 50e3 -- not used
    tx_ht_m = 6
    freq = 750e6

    # Initialize receiver parameters
    rx_gain = 0
    rx_loss = 2
    noise_figure = 4
    rx_bw_hz = 1e6
    rx_ht_m = 1e3
    num_elements = 25   # Array elements
    num_samples = 100  # Time samples
    d_lam = .5
    v, _ = array_df.model.make_steering_vector(d_lam, num_elements)

    # Initialize source positions
    num_sources = 3  # Sources
    range_vec = np.array([45, 50, 55])*1e3
    theta = np.array([-30, 20, 24])
    psi = theta*np.pi/180

    # Compute Received Power at each element
    prop_loss = prop.model.get_path_loss(range_vec, freq, tx_ht_m, rx_ht_m, False, None)
    rx_pwr_dbw = (lin_to_db(tx_pwr) + tx_gain - tx_loss) - prop_loss + rx_gain - rx_loss
    rx_noise_dbw = lin_to_db(constants.kT*rx_bw_hz)+noise_figure
    # snr_db = rx_pwr_dbw - rx_noise_dbw -- not used
    # snr_lin = db_to_lin(snr_db)  # num_sources x 1 -- not used
    rx_pwr_w = db_to_lin(rx_pwr_dbw)
    rx_noise_w = db_to_lin(rx_noise_dbw)

    # Set up received data vector
    v_steer_mtx = v(psi)   # num_elements x num_sources
    print('v_steer_mtx: {}'.format(np.shape(v_steer_mtx)))
    print('rx_pwr_w: {}'.format(np.shape(rx_pwr_w)))
    print('num_source: {}'.format(num_sources))
    print('num_samples: {}'.format(num_samples))

    random_signal = (rng.standard_normal(size=(num_sources, num_samples))
                     + 1j * rng.standard_normal(size=(num_sources, num_samples)))
    rx_signal = (v_steer_mtx * np.sqrt(np.expand_dims(rx_pwr_w/2, axis=0))).dot(random_signal)

    # Add noise
    noise = np.sqrt(rx_noise_w/2)*(rng.standard_normal(size=(num_elements, num_samples))
                                   + 1j * rng.standard_normal(size=(num_elements, num_samples)))
    noisy_signal = (rx_signal+noise)  # num_elements x num_samples

    savemat('./ex8_1.mat',
            {'x': noisy_signal,
             'num_sources': num_sources,
             'num_elements': num_elements,
             'num_samples': num_samples,
             'd_lam': d_lam})

    # ===== Problem 8.5 =====

    # Initialize PTT Radio Parameters
    tx_pwr = 10
    tx_gain = 0
    tx_loss = 3
    # tx_bw_hz = 50e3 -- not used
    tx_ht_m = 6
    freq = 750e6

    # Initialize receiver parameters
    rx_gain = 0
    rx_loss = 2
    noise_figure = 6
    rx_bw_hz = 1e6
    rx_ht_m = 1e3
    num_elements = 25   # Array elements
    num_samples = 100  # Time samples
    d_lam = .5
    v, v_dot = array_df.model.make_steering_vector(d_lam, num_elements)

    # Initialize source positions
    num_sources = 3  # Sources
    range_vec = np.array([85, 95, 100])*1e3
    theta = np.array([-10, 15, -5])
    psi = theta*np.pi/180

    # Compute Received Power at each element
    prop_loss = prop.model.get_path_loss(range_vec, freq, tx_ht_m, rx_ht_m, False, None)
    rx_pwr_dbw = (lin_to_db(tx_pwr) + tx_gain - tx_loss) - prop_loss + rx_gain - rx_loss
    rx_noise_dbw = lin_to_db(constants.kT*rx_bw_hz)+noise_figure
    # snr_db = rx_pwr_dbw - rx_noise_dbw -- not used
    # snr_lin = db_to_lin(snr_db)  # num_sources x 1 -- not used
    rx_pwr_w = db_to_lin(rx_pwr_dbw)
    rx_noise_w = db_to_lin(rx_noise_dbw)

    # Set up received data vector
    v_steer_mtx = v(psi)  # num_elements x num_sources
    rx_signal = np.matmul(np.multiply(v_steer_mtx, np.sqrt(rx_pwr_w[:]/2)),
                          (rng.standard_normal(size=(num_sources, num_samples))
                          + 1j * (rng.standard_normal(size=(num_sources, num_samples)))))

    # Add noise
    noise = np.sqrt(rx_noise_w/2)*(rng.standard_normal(size=(num_elements, num_samples))
                                   + 1j * rng.standard_normal(size=(num_elements, num_samples)))
    noisy_signal = (rx_signal+noise)  # num_elements x num_samples

    savemat('./problem8_5.mat',
            {'x': noisy_signal,
             'num_sources': num_sources,
             'num_elements': num_elements,
             'num_samples': num_samples,
             'd_lam': d_lam})

    # ===== Problem 8.6 =====

    # Initialize PTT Radio Parameters
    tx_pwr = 10
    tx_gain = 0
    tx_loss = 3
    # tx_bw_hz = 50e3 --- not used
    tx_ht_m = 6
    freq = 750e6

    # Initialize receiver parameters
    rx_gain = 0
    rx_loss = 5
    noise_figure = 6
    rx_bw_hz = 1e6
    rx_ht_m = 1e3
    num_elements = 25  # Array elements
    num_samples = 100  # Time samples
    d_lam = .5
    v, v_dot = array_df.model.make_steering_vector(d_lam, num_elements)

    # Initialize source positions
    num_sources = 3  # Sources
    range_vec = np.array([250, 260, 300])*1e3
    theta = np.array([-10, 15, -5])
    psi = theta*np.pi/180

    # Compute Received Power at each element
    prop_loss = prop.model.get_path_loss(range_vec, freq, tx_ht_m, rx_ht_m, False, None)
    rx_pwr_dbw = (lin_to_db(tx_pwr) + tx_gain - tx_loss) - prop_loss + rx_gain - rx_loss
    rx_noise_dbw = lin_to_db(constants.kT*rx_bw_hz)+noise_figure
    # snr_db = rx_pwr_dbw - rx_noise_dbw --- not used
    # snr_lin = lin_to_db(snr_db)  # num_sources x 1 --- not used
    rx_pwr_w = db_to_lin(rx_pwr_dbw)
    rx_noise_w = db_to_lin(rx_noise_dbw)

    # Set up received data vector
    v_steer_mtx = v(psi)  # num_elements x num_sources
    rx_signal = np.matmul(np.multiply(v_steer_mtx, np.sqrt(rx_pwr_w[:]/2)),
                          (rng.standard_normal(size=(num_sources, num_samples))
                          + 1j * (rng.standard_normal(size=(num_sources, num_samples)))))

    # Add noise
    noise = np.sqrt(rx_noise_w/2)*(rng.standard_normal(size=(num_elements, num_samples))
                                   + 1j * rng.standard_normal(size=(num_elements, num_samples)))
    noisy_signal = (rx_signal+noise)  # num_elements x num_samples

    savemat('./problem8_6.mat',
            {'x': noisy_signal,
             'num_sources': num_sources,
             'num_elements': num_elements,
             'num_samples': num_samples,
             'd_lam': d_lam})


def example1(rng=np.random.default_rng()):
    """
    Executes Example 8.1, relying on the sample data in examples/ex8_1.mat, and generates one figure.

    Ported from MATLAB Code
    
    Nicholas O'Donoughue
    5 May 2021

    :param rng: random number generator, used if we need to re-generate the example data
    :return fig: figure handle to generated graphic
    """

    # If the data is missing, make it
    data_fnm = './ex8_1.mat'
    if not os.path.exists(data_fnm):
        generate_ex1_data(rng)

    # Load sample data
    data = loadmat(data_fnm)
    #   x                noisy data vector (M x N)
    #   num_source       number of sources
    #   num_elements     number of array elements
    #   num_samples      number of time snapshots
    #   d_lam            array spacing
    num_elements = data['num_elements']
    d_lam = data['d_lam']
    x = data['x']

    # Construct array steering vector
    v, _ = array_df.model.make_steering_vector(d_lam, num_elements)

    # Call Beamformer
    pwr_vec, psi_vec = array_df.solvers.beamscan(x, v, np.pi/2, 1001)
    peaks, _ = find_peaks(pwr_vec, prominence=.1*np.max(pwr_vec))
    # print(peaks)
    psi_peaks = psi_vec[peaks]
    peak_vals = pwr_vec[peaks]
    th_peaks = 180*psi_peaks/np.pi

    # Call MVDR Beamformer
    pwr_vec_mvdr, psi_vec = array_df.solvers.beamscan_mvdr(x, v, np.pi/2, 1001)
    peaks_mvdr, _ = find_peaks(pwr_vec_mvdr, prominence=.2*np.max(pwr_vec_mvdr))
    psi_peaks_mvdr = psi_vec[peaks_mvdr]
    peak_vals_mvdr = pwr_vec_mvdr[peaks_mvdr]
    th_peaks_mvdr = 180*psi_peaks_mvdr/np.pi

    # Plot
    th_vec = 180*psi_vec/np.pi
    fig = plt.figure()
    plt.plot(th_vec, lin_to_db(np.abs(pwr_vec)), linewidth=1.5, label='Beamscan')
    plt.plot(th_vec, lin_to_db(np.abs(pwr_vec_mvdr)), label='MVDR')
    plt.scatter(th_peaks, lin_to_db(peak_vals), marker='v', label='Beamscan Soln.')  # , markersize=6)
    plt.scatter(th_peaks_mvdr, lin_to_db(peak_vals_mvdr), marker='^', label='MVDR Soln.')  # , markersize=6)
    plt.xlabel(r'$\theta$ [deg]')
    plt.ylabel('P [dB]')
    plt.ylim([-145, -100])
    plt.legend(loc='upper left')

    return fig


def example2(rng=np.random.default_rng(), mc_params=None):
    """
    Executes Example 8.2 and generates one figure

    Ported from MATLAB Code

    Nicholas O'Donoughue
    5 May 2021

    :param rng: random number generator
    :param mc_params: Optional struct to control Monte Carlo trial size
    :return: figure handle to generated graphic
    """
    
    # Define transmitter values
    tx_pwr = 100
    tx_gain = 0
    # tx_bw_hz = 500e3 -- not used
    tx_freq = 950e6
    tx_loss = 3
    tx_ht_m = 5e3
    theta = 5
    psi = theta*np.pi/180
    
    # Define receiver values
    d_lam = .5
    num_elements = 5
    rx_gain = 0
    rx_loss = 2
    noise_figure = 4
    rx_bw_hz = 100e6
    rx_alt_m = 10
    t_integration = 1e-6
    
    # Compute the number of snapshots
    t_samp = 1/(2*tx_freq)
    num_samples = int(np.floor(t_integration/t_samp))
    
    # Generate the array factor
    v, v_dot = array_df.model.make_steering_vector(d_lam, num_elements)
    
    # Generate received signal power and SNR at multiple ranges
    range_vec_m = np.arange(start=100e3, step=20e3, stop=500e3)
    prop_loss = prop.model.get_path_loss(range_vec_m, tx_freq, tx_ht_m, rx_alt_m, False)
    rx_pwr_dbw = lin_to_db(tx_pwr) + tx_gain + rx_gain - tx_loss - rx_loss - prop_loss
    rx_noise_dbw = lin_to_db(constants.kT*rx_bw_hz) + noise_figure
    snr_db = rx_pwr_dbw-rx_noise_dbw
    snr_lin = db_to_lin(snr_db)
    
    # Compute CRLB
    crlb_psi = np.zeros(shape=(np.size(snr_db), 1))
    crlb_psi_stoch = np.zeros_like(crlb_psi)
    
    for idx_xi, this_xi_lin in enumerate(snr_lin):
        crlb_psi[idx_xi] = array_df.perf.crlb_det(this_xi_lin, 1, psi, num_samples, v, v_dot)
        crlb_psi_stoch[idx_xi] = array_df.perf.crlb_stochastic(this_xi_lin, 1, psi, num_samples, v, v_dot)
    
    crlb_rmse_deg = np.sqrt(crlb_psi) * 180/np.pi
    crlb_rmse_deg_stoch = np.sqrt(crlb_psi_stoch) * 180/np.pi
    
    # Compute MC Experiment
    num_monte_carlo = 1000
    if mc_params is not None:
        num_monte_carlo = max(int(num_monte_carlo/mc_params['monte_carlo_decimation']),mc_params['min_num_monte_carlo'])
    sig = np.sqrt(1/2)*(rng.standard_normal(size=(num_samples, num_monte_carlo))
                        + 1j * rng.standard_normal(size=(num_samples, num_monte_carlo)))
    noise = np.sqrt(1/2)*(rng.standard_normal(size=(num_elements, num_samples, num_monte_carlo))
                          + 1j * rng.standard_normal(size=(num_elements, num_samples, num_monte_carlo)))
    out_shp = (np.size(range_vec_m), )
    rmse_deg_beam = np.zeros(shape=out_shp)
    rmse_deg_mvdr = np.zeros(shape=out_shp)
    rmse_deg_music = np.zeros(shape=out_shp)

    print('Executing array DF monte carlo trial...')
    iterations_per_marker = 10
    markers_per_row = 40
    iterations_per_row = markers_per_row * iterations_per_marker
    total_iterations = num_monte_carlo * len(range_vec_m)
    t_start = time.perf_counter()
    for idx_r, this_range in enumerate(range_vec_m):
        this_signal = np.reshape(sig*np.sqrt(db_to_lin(snr_db[idx_r])), (1, num_samples, num_monte_carlo))
        this_rx_signal = np.expand_dims(v(psi), axis=2) * this_signal + noise
    
        this_err_beamscan = np.zeros(shape=(num_monte_carlo, ))
        this_err_mvdr = np.zeros(shape=(num_monte_carlo, ))
        this_err_music = np.zeros(shape=(num_monte_carlo, ))

        for idx_mc in np.arange(num_monte_carlo):
            curr_idx = idx_mc + idx_r * num_monte_carlo
            print_progress(total_iterations, curr_idx, iterations_per_marker, iterations_per_row, t_start)

            # Compute beamscan image
            pwr_vec, psi_vec = array_df.solvers.beamscan(this_rx_signal[:, :, idx_mc], v, np.pi/2, 2001)
            idx_pk = np.argmax(np.abs(pwr_vec))
            this_err_beamscan[idx_mc] = np.abs(psi_vec[idx_pk]-psi)
    
            # Compute beamscan MVDR image
            # ToDo: plot doesn't look right; diagnose and fix.
            pwr_vec_mvdr, psi_vec = array_df.solvers.beamscan_mvdr(this_rx_signal[:, :, idx_mc], v, np.pi/2, 2001)
            idx_pk = np.argmax(np.abs(pwr_vec_mvdr))
            this_err_mvdr[idx_mc] = np.abs(psi_vec[idx_pk]-psi)
    
            # Compute MUSIC image
            pwr_vec_music, psi_vec = array_df.solvers.music(this_rx_signal[:, :, idx_mc], v, 1, np.pi/2, 2001)
            idx_pk = np.argmax(np.abs(pwr_vec_music))
            this_err_music[idx_mc] = np.abs(psi_vec[idx_pk]-psi)
    
            # Average Results
            rmse_deg_beam[idx_r] = (180/np.pi)*np.sqrt(np.sum(this_err_beamscan**2)/num_monte_carlo)
            rmse_deg_mvdr[idx_r] = (180/np.pi)*np.sqrt(np.sum(this_err_mvdr**2)/num_monte_carlo)
            rmse_deg_music[idx_r] = (180/np.pi)*np.sqrt(np.sum(this_err_music**2)/num_monte_carlo)
    
    print('done.')
    t_elapsed = time.perf_counter() - t_start
    print_elapsed(t_elapsed)

    # Plot results
    fig = plt.figure()
    plt.loglog(range_vec_m/1e3, crlb_rmse_deg, linestyle='--', color='black', label='CRLB (det.)')
    plt.loglog(range_vec_m/1e3, crlb_rmse_deg_stoch, linestyle='-.', color='black', label='CRLB (stoch.)')
    plt.plot(range_vec_m/1e3, rmse_deg_beam, linewidth=2, label='Beamscan')
    plt.plot(range_vec_m/1e3, rmse_deg_mvdr, linewidth=1.5, label='MVDR')
    plt.plot(range_vec_m/1e3, rmse_deg_music, linewidth=1, label='MUSIC')
    plt.xlabel('Range [km]')
    plt.ylabel('RMSE [deg]')
    plt.legend(loc='upper left')

    return fig


if __name__ == '__main__':
    run_all_examples()
    plt.show()
