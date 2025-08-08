import numpy as np
import matplotlib.pyplot as plt
import prop
import detector
import utils
from utils.unit_conversions import lin_to_db


def run_all_examples(colors=None):
    """
    Run all chapter 4 examples and return a list of figure handles

    :param colors: colormap
    :return figs: list of figure handles
    """

    return [example1(), example2(colors)]


def example1():
    """
    Executes Example 4.1.

    Ported from MATLAB Code
    
    Nicholas O'Donoughue
    24 March 2021
    
    :return: figure handle to generated graphic
    """
            
    # Transmit Side
    erp = np.array([10, -7])  # tower / user
    freq_hz = np.array([850e6, 1900e6])
    tx_ht_m = np.array([60, 2])
    tx_loss = 0
    bw_signal = 5e6
    pulse_duration = 20e-3
    
    # Receive Side
    rx_ht_m = 1e3
    rx_gain = 0
    rx_loss = 3
    noise_figure = 5
    bw_noise = 100e6
    corr_time = 1e-4
        
    # Compute xi0
    noise_pwr = noise_figure + lin_to_db(utils.constants.kT*bw_noise)
    xi0 = erp + rx_gain - tx_loss - rx_loss - noise_pwr
    
    # Compute Prop Loss
    range_vec_m = np.logspace(start=3, stop=6, num=100)
    prop_loss_twr = prop.model.get_path_loss(range_m=range_vec_m, freq_hz=freq_hz[0], tx_ht_m=tx_ht_m[0],
                                             rx_ht_m=rx_ht_m, include_atm_loss=False)
    prop_loss_usr = prop.model.get_path_loss(range_m=range_vec_m, freq_hz=freq_hz[1], tx_ht_m=tx_ht_m[1],
                                             rx_ht_m=rx_ht_m, include_atm_loss=False)
    
    xi_twr = xi0[0] - prop_loss_twr
    xi_usr = xi0[1] - prop_loss_usr
    
    # Compute thresholds
    prob_d = .8
    prob_fa = 1e-6
    num_samples = int(np.fix(corr_time*bw_noise))
    xi_ed = detector.squareLaw.min_sinr(prob_fa=prob_fa, prob_d=prob_d, num_samples=num_samples)
    xi_xc = detector.xcorr.min_sinr(prob_fa=prob_fa, prob_d=prob_d, corr_time=corr_time, pulse_duration=pulse_duration,
                                    bw_noise=bw_noise, bw_signal=bw_signal)
    
    # Compute Max Range
    max_range_twr_ed = detector.squareLaw.max_range(prob_fa=prob_fa, prob_d=prob_d, num_samples=num_samples,
                                                    f0=freq_hz[0], ht=tx_ht_m[0], hr=rx_ht_m, snr0=xi0[0],
                                                    include_atm_loss=False)[0]
    max_range_usr_ed = detector.squareLaw.max_range(prob_fa=prob_fa, prob_d=prob_d, num_samples=num_samples,
                                                    f0=freq_hz[1], ht=tx_ht_m[1], hr=rx_ht_m, snr0=xi0[1],
                                                    include_atm_loss=False)[0]
    max_range_twr_xc = detector.xcorr.max_range(prob_fa=prob_fa, prob_d=prob_d, corr_time=corr_time,
                                                pulse_duration=pulse_duration, bw_noise=bw_noise, bw_signal=bw_signal,
                                                f0=freq_hz[0], ht=tx_ht_m[0], hr=rx_ht_m, snr0=xi0[0],
                                                include_atm_loss=False)[0]
    max_range_usr_xc = detector.xcorr.max_range(prob_fa=prob_fa, prob_d=prob_d, corr_time=corr_time,
                                                pulse_duration=pulse_duration, bw_noise=bw_noise, bw_signal=bw_signal,
                                                f0=freq_hz[1], ht=tx_ht_m[1], hr=rx_ht_m, snr0=xi0[1],
                                                include_atm_loss=False)[0]
    
    print('Detection range of tower')
    print('\tusing Energy Detector: {:.2f} km'.format(max_range_twr_ed/1e3))
    print('\tusing Cross Correlator: {:.2f} km'.format(max_range_twr_xc/1e3))
    print('Detection range of handset user')
    print('\tusing Energy Detector: {:.2f} km'.format(max_range_usr_ed/1e3))
    print('\tusing Cross Correlator: {:.2f} km'.format(max_range_usr_xc/1e3))
    
    # Plot results
    fig = plt.figure()
    plt.semilogx(range_vec_m/1e3, xi_twr, label=r'$\xi$ (Tower)')
    plt.plot(range_vec_m/1e3, xi_usr, label=r'$\xi$ (Handset)')
    plt.plot(range_vec_m/1e3, xi_ed*np.ones_like(range_vec_m), color='k', linestyle='--', label=r'$\xi_{min}$ (ED)')
    plt.plot(range_vec_m/1e3, xi_xc*np.ones_like(range_vec_m), color='k', linestyle='-.', label=r'$\xi_{min}$ (XC)')
    plt.legend(loc='upper right')
    plt.xlabel('Range [km]')
    plt.ylabel(r'$\xi$ [dB]')
    
    return fig

    
def example2(colors=None):
    """
    Executes Example 4.2.
    
    Ported from MATLAB Code
    
    Nicholas O'Donoughue
    23 March 2021

    :param colors: colormap
    :return: figure handle to generated graphic
    """

    # Initialize colormap
    if colors is None:
        colors = plt.get_cmap('tab10')

    # Transmit Side
    tx_pwr = lin_to_db(500)          # dBW
    freq_hz = 16e9                   # Hz
    tx_ht_m = 6096                   # m
    tx_loss = 3                      # dB
    bw_signal = 200e6                # Hz
    pulse_duration = 20e-6           # s
    tx_gain = np.array([30, 10, 0])  # dBi
    
    # Receive Side
    rx_ht_m = 20                     # m
    rx_gain = 0                      # dBi
    rx_loss = 3                      # dB
    noise_figure = 4                 # dB
    bw_noise = 500e6                 # Hz
    corr_time = 1e-4                 # s
        
    # Compute xi0
    noise_pwr = noise_figure + lin_to_db(utils.constants.kT*bw_noise)
    xi0 = tx_pwr + tx_gain + rx_gain - tx_loss - rx_loss - noise_pwr
    
    # Adjust xi0 to account for partial pulse
    xi0 = xi0 + lin_to_db(np.minimum(1, pulse_duration/corr_time))
    
    # Compute Prop Loss
    range_vec_m = np.logspace(start=4, stop=6, num=201)
    prop_loss = prop.model.get_path_loss(range_m=range_vec_m, freq_hz=freq_hz, tx_ht_m=tx_ht_m, rx_ht_m=rx_ht_m,
                                         include_atm_loss=False)
    
    xi = np.expand_dims(xi0, axis=0) - np.expand_dims(prop_loss, axis=1)
    
    # Compute thresholds
    prob_d = .8
    prob_fa = 1e-6
    num_samples = int(np.fix(corr_time*bw_noise))
    xi_ed = detector.squareLaw.min_sinr(prob_fa=prob_fa, prob_d=prob_d, num_samples=num_samples)
    xi_xc = detector.xcorr.min_sinr(prob_fa=prob_fa, prob_d=prob_d, corr_time=corr_time, pulse_duration=pulse_duration,
                                    bw_noise=bw_noise, bw_signal=bw_signal)
    
    # Compute Max Range
    max_range_ml_ed = detector.squareLaw.max_range(prob_fa=prob_fa, prob_d=prob_d, num_samples=num_samples, f0=freq_hz,
                                                   ht=tx_ht_m, hr=rx_ht_m, snr0=xi0[0], include_atm_loss=False)[0]
    max_range_near_sl_ed = detector.squareLaw.max_range(prob_fa=prob_fa, prob_d=prob_d, num_samples=num_samples,
                                                        f0=freq_hz, ht=tx_ht_m, hr=rx_ht_m, snr0=xi0[1],
                                                        include_atm_loss=False)[0]
    max_range_far_sl_ed = detector.squareLaw.max_range(prob_fa=prob_fa, prob_d=prob_d, num_samples=num_samples,
                                                       f0=freq_hz, ht=tx_ht_m, hr=rx_ht_m, snr0=xi0[2],
                                                       include_atm_loss=False)[0]
    max_range_ml_xc = detector.xcorr.max_range(prob_fa=prob_fa, prob_d=prob_d, corr_time=corr_time,
                                               pulse_duration=pulse_duration, bw_noise=bw_noise, bw_signal=bw_signal,
                                               f0=freq_hz, ht=tx_ht_m, hr=rx_ht_m, snr0=xi0[0],
                                               include_atm_loss=False)[0]
    max_range_near_sl_xc = detector.xcorr.max_range(prob_fa=prob_fa, prob_d=prob_d, corr_time=corr_time,
                                                    pulse_duration=pulse_duration, bw_noise=bw_noise,
                                                    bw_signal=bw_signal, f0=freq_hz, ht=tx_ht_m, hr=rx_ht_m,
                                                    snr0=xi0[1], include_atm_loss=False)[0]
    max_range_far_sl_xc = detector.xcorr.max_range(prob_fa=prob_fa, prob_d=prob_d, corr_time=corr_time,
                                                   pulse_duration=pulse_duration, bw_noise=bw_noise,
                                                   bw_signal=bw_signal, f0=freq_hz, ht=tx_ht_m, hr=rx_ht_m,
                                                   snr0=xi0[2], include_atm_loss=False)[0]
    
    print('Mainlobe detection')
    print('\tusing Energy Detector: {:.2f} km'.format(max_range_ml_ed/1e3))
    print('\tusing Cross Correlator: {:.2f} km'.format(max_range_ml_xc/1e3))
    print('Near sidelobe detection')
    print('\tusing Energy Detector: {:.2f} km'.format(max_range_near_sl_ed/1e3))
    print('\tusing Cross Correlator: {:.2f} km'.format(max_range_near_sl_xc/1e3))
    print('Far sidelobe detection')
    print('\tusing Energy Detector: {:.2f} km'.format(max_range_far_sl_ed/1e3))
    print('\tusing Cross Correlator: {:.2f} km'.format(max_range_far_sl_xc/1e3))
    
    # Plot results
    fig = plt.figure()
    for idx, suffix in enumerate(['ML', 'Near SL', 'Far SL']):
        plt.semilogx(range_vec_m/1e3, xi[:, idx], color=colors(idx), label=r'$\xi$ (' + suffix + ')')

    plt.plot(range_vec_m/1e3, xi_ed*np.ones_like(range_vec_m), color='k', linestyle='--', label=r'$\xi_{min}$ (ED)')
    plt.plot(range_vec_m/1e3, xi_xc*np.ones_like(range_vec_m), color='k', linestyle='-.', label=r'$\xi_{min}$ (XC)')
    plt.legend(loc='upper right')
    plt.xlabel('Range [km]')
    plt.ylabel(r'$\xi$ [dB]')

    return fig


if __name__ == '__main__':
    run_all_examples()
    plt.show()
