import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import prop
import detector
import utils
# from utils.unit_conversions import lin_to_db, db_to_lin
lin_to_db = utils.unit_conversions.lin_to_db
db_to_lin = utils.unit_conversions.db_to_lin


def run_all_examples(rng=None, colors=None):
    """
    Run all chapter 3 examples and return a list of figure handles

    :param rng: random number generator
    :param colors: colormap
    :return figs: list of figure handles
    """

    return [example1(rng, colors), example2(rng, colors)]


def example1(rng=None, colors=None):
    """
    Executes Example 3.1.

    Ported from MATLAB Code
    
    Nicholas O'Donoughue
    23 March 2021
    
    :param rng: random number generator
    :param colors: colormap
    :return: figure handle to generated graphic
    """

    # Initialize random number generator and colormap
    if rng is None:
        rng = np.random.default_rng()

    if colors is None:
        colors = plt.get_cmap('tab10')
            
    #  At each range, compute the received  power level.  Conduct a monte carlo trial at each power level to compute 
    #  PD and compare to the desired threshold.
    
    ht = 100
    hr = 2
    range_vec = np.arange(start=100e3, step=1e3, stop=1000e3)
    range_vec_coarse = np.arange(100e3, step=50e3, stop=1000e3)
    
    f0 = 100e6
    
    # Compute Losses and Fresnel Zone
    loss_prop = prop.model.get_path_loss(range_m=range_vec, freq_hz=f0, tx_ht_m=ht, rx_ht_m=hr, include_atm_loss=False)
    loss_prop_coarse = prop.model.get_path_loss(range_m=range_vec_coarse, freq_hz=f0, tx_ht_m=ht, rx_ht_m=hr,
                                                include_atm_loss=False)
    
    # Noise Power
    bandwidth = 2e5         # channel bandwidth [Hz]
    noise_figure = 5        # noise figure [dB]
    noise_pwr = lin_to_db(utils.constants.kT*bandwidth)+noise_figure
    
    # Signal Power
    eirp = 47    # dBW
    rx_gain = 0  # Receive antenna gain
    rx_loss = 0
    
    # Received Power and SNR
    signal_pwr_vec = eirp-loss_prop+rx_gain-rx_loss
    signal_pwr_vec_coarse = eirp-loss_prop_coarse+rx_gain-rx_loss
    
    # Desired Detection Performance
    num_samples = 10    # Number of samples
    prob_fa = 1e-6
    prob_det_desired = .5
    
    # Find expected max det range, for plotting
    max_range_soln = detector.squareLaw.max_range(prob_fa=prob_fa, prob_d=prob_det_desired, num_samples=num_samples,
                                                  f0=f0, ht=ht, hr=hr, snr0=eirp+rx_gain-rx_loss-noise_pwr,
                                                  include_atm_loss=False)
    
    # Monte Carlo Trial
    num_monte_carlo = int(1e4)
    
    # Convert noise and signal power to linear units
    noise_pwr_lin = db_to_lin(noise_pwr)
    signal_pwr_vec_coarse_lin = db_to_lin(signal_pwr_vec_coarse)
    xi = signal_pwr_vec-noise_pwr
    xi_lin = db_to_lin(xi)
    
    # Theoretical Result
    eta = stats.chi2.ppf(q=1-prob_fa, df=2*num_samples)
    prob_det_theo = 1-stats.ncx2.cdf(eta, df=2*num_samples, nc=2*num_samples*xi_lin)
    
    # Generate noise and signal vectors
    noise = np.sqrt(noise_pwr_lin/2) * (rng.standard_normal(size=(num_samples, num_monte_carlo)) +
                                        1j*rng.standard_normal(size=(num_samples, num_monte_carlo)))
    start_phase = rng.uniform(low=0.0, high=2*np.pi, size=(1, num_monte_carlo))
    pulse = np.expand_dims(np.exp(1j*2*np.pi*np.arange(num_samples)/10), axis=1)  # Unit Power
    signal = pulse*np.exp(1j*start_phase)
    
    # Compute Sufficient Statistic
    prob_det_vec_coarse = np.zeros_like(signal_pwr_vec_coarse)

    for idx_rng, signal_pwr in enumerate(signal_pwr_vec_coarse_lin):
        # Scale signal power
        this_signal = np.sqrt(signal_pwr)*signal
    
        # Run Energy Detector
        detection_result = detector.squareLaw.det_test(z=noise+this_signal, noise_var=noise_pwr_lin/2, prob_fa=prob_fa)
        
        # Count detections
        prob_det_vec_coarse[idx_rng] = np.sum(detection_result)/num_monte_carlo
    
    fig = plt.figure()
    plt.plot(range_vec/1e3, prob_det_theo, color=colors(1), label='Theoretical')
    plt.scatter(range_vec_coarse/1e3, prob_det_vec_coarse, marker='^', color=colors(1), label='Monte Carlo')
    plt.plot(max_range_soln/1e3*[1, 1], [0, 1], linestyle=':', color=colors(1), label='Max Range')
    
    plt.plot(range_vec_coarse/1e3, prob_det_desired*np.ones_like(range_vec_coarse), linestyle='--', color=colors(3),
             label='$P_D$=0.5')
    
    plt.xlabel('Range [km]')
    plt.ylabel('$P_D$')
    plt.legend(loc='lower left')
    
    return fig

    
def example2(rng=None, colors=None):
    """
    Executes Example 3.2.
    
    Ported from MATLAB Code
    
    Nicholas O'Donoughue
    23 March 2021

    :param rng: random number generator
    :param colors: colormap
    :return: figure handle to generated graphic
    """

    # Initialize random number generator and colormap
    if rng is None:
        rng = np.random.default_rng()

    if colors is None:
        colors = plt.get_cmap('tab10')

    #  At each range, compute the received power level.  Conduct a monte carlo trial at each power level to compute
    #  PD and compare to the desired threshold.

    # Transmit Parameters
    f0 = 35e9
    tx_pwr = lin_to_db(.035)  # 35 mW / -14.56 dBW
    tx_gain = 34  # dBi
    rx_gain = 10
    tx_loss = 0
    rx_loss = 2
    sidelobe_level = 25  # Sidelobe level (below tx_gain)
    bandwidth = 40e6
    noise_figure = 4
    tx_height_m = 500
    rx_height_m = 500

    # Compute Noise Level and MDS
    noise_power = noise_figure + lin_to_db(utils.constants.kT * bandwidth)

    # SNR with no propagation loss
    snr0 = tx_pwr + tx_gain - tx_loss + rx_gain - rx_loss - noise_power
    snr0_sll = snr0 - sidelobe_level

    # Compute Path Loss
    range_vec = np.logspace(start=2, stop=5, num=5000)
    range_vec_coarse = np.logspace(start=2, stop=5, num=25)

    alt_m = 1000

    loss_prop_vec = prop.model.get_path_loss(range_m=range_vec, freq_hz=f0, tx_ht_m=alt_m, rx_ht_m=alt_m,
                                             include_atm_loss=True)
    loss_prop_vec_coarse = prop.model.get_path_loss(range_m=range_vec_coarse, freq_hz=f0, tx_ht_m=alt_m, rx_ht_m=alt_m,
                                                    include_atm_loss=True)

    # Compute Received Power
    rx_pwr_vec = tx_pwr + tx_gain - tx_loss+rx_gain-rx_loss-loss_prop_vec
    rx_pwr_sll_vec = rx_pwr_vec - sidelobe_level
    rx_pwr_vec_coarse = tx_pwr + tx_gain - tx_loss+rx_gain-rx_loss-loss_prop_vec_coarse
    rx_pwr_sll_vec_coarse = rx_pwr_vec_coarse - sidelobe_level

    # Desired Detection Performance
    num_samples = 10   # Number of samples
    prob_fa = 1e-6
    prob_det_desired = .5
    # snr_min = detector.squareLaw.min_sinr(prob_fa=prob_fa, prob_d=prob_det_desired, num_samples=num_samples)
    # rx_pwr_min = noise_power + snr_min

    # Find expected max det range, for plotting
    range_soln = detector.squareLaw.max_range(prob_fa=prob_fa, prob_d=prob_det_desired, num_samples=num_samples,
                                              f0=f0, ht=tx_height_m, hr=rx_height_m, snr0=snr0,
                                              include_atm_loss=True)
    range_soln_sll = detector.squareLaw.max_range(prob_fa=prob_fa, prob_d=prob_det_desired, num_samples=num_samples,
                                                  f0=f0, ht=tx_height_m, hr=rx_height_m, snr0=snr0_sll,
                                                  include_atm_loss=True)

    # Monte Carlo Trial
    num_monte_carlo = int(1e4)

    # Convert noise and signal power to linear units
    noise_pwr_lin = db_to_lin(noise_power)
    rx_pwr_vec_coarse_lin = db_to_lin(rx_pwr_vec_coarse)
    rx_pwr_sll_vec_coarse_lin = db_to_lin(rx_pwr_sll_vec_coarse)

    xi = rx_pwr_vec-noise_power
    xi_lin = db_to_lin(xi)
    xi_sl = rx_pwr_sll_vec-noise_power
    xi_sl_lin = db_to_lin(xi_sl)

    # Theoretical Result
    eta = stats.chi2.ppf(q=1-prob_fa, df=2*num_samples)
    prob_det_theo = 1-stats.ncx2.cdf(x=eta, df=2*num_samples, nc=2*num_samples*xi_lin)
    prob_det_sll_theo = 1-stats.ncx2.cdf(x=eta, df=2*num_samples, nc=2*num_samples*xi_sl_lin)

    # Generate noise and signal vectors
    noise = np.sqrt(noise_pwr_lin/2) * (rng.standard_normal(size=(num_samples, num_monte_carlo)) +
                                        1j*rng.standard_normal(size=(num_samples, num_monte_carlo)))
    start_phase = rng.uniform(low=0, high=2*np.pi, size=(1, num_monte_carlo))
    pulse = np.expand_dims(np.exp(1j*2*np.pi*np.arange(num_samples)/10), axis=1)  # Unit Power
    signal = pulse*np.exp(1j*start_phase)

    # Compute Sufficient Statistic
    prob_det_vec = np.zeros_like(rx_pwr_vec_coarse)
    prob_det_sll_vec = np.zeros_like(rx_pwr_vec_coarse)
    for idx in np.arange(np.size(rx_pwr_vec_coarse)):
        # Scale signal power
        this_signal = np.sqrt(rx_pwr_vec_coarse_lin[idx])*signal
        this_signal_sll = np.sqrt(rx_pwr_sll_vec_coarse_lin[idx])*signal

        # Run Energy Detector
        detection_result = detector.squareLaw.det_test(z=noise+this_signal, noise_var=noise_pwr_lin/2, prob_fa=prob_fa)
        detection_result_sll = detector.squareLaw.det_test(z=noise+this_signal_sll, noise_var=noise_pwr_lin/2,
                                                           prob_fa=prob_fa)

        # Count detections
        prob_det_vec[idx] = np.sum(detection_result)/num_monte_carlo
        prob_det_sll_vec[idx] = np.sum(detection_result_sll)/num_monte_carlo
    
    fig = plt.figure()
    plt.semilogx(range_vec/1e3, prob_det_theo, color=colors(1), label='Theoretical (ML)')
    plt.scatter(range_vec_coarse/1e3, prob_det_vec, marker='^', color=colors(1), label='Monte Carlo (ML)')
    plt.plot(range_soln/1e3*[1, 1], [0, 1], linestyle=':', color=colors(1), label='Max Range (ML)')

    plt.plot(range_vec/1e3, prob_det_sll_theo, color=colors(2), label='Theoretical (SL)')
    plt.scatter(range_vec_coarse/1e3, prob_det_sll_vec, marker='^', color=colors(2), label='Monte Carlo (SL)')
    plt.plot(range_soln_sll/1e3*[1, 1], [0, 1], linestyle=':', color=colors(2), label='Max Range (SL)')

    plt.plot(range_vec_coarse/1e3, prob_det_desired*np.ones_like(range_vec_coarse), linestyle='--', color='k',
             label='$P_D=0.5$')

    plt.xlabel('Range [km]')
    plt.ylabel('$P_D$')
    plt.legend(loc='lower left')

    return fig
