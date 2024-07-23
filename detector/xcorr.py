import numpy as np
import scipy.stats as stats
from . import squareLaw
from utils.unit_conversions import lin_to_db, db_to_lin
import prop


def det_test(y1, y2, noise_var, num_samples, prob_fa):
    """
    Apply cross-correlation to determine whether a signal (y2) is present or
    absent in the provided received data vector (y1).
    
    Ported from MATLAB Code
    
    Nicholas O'Donoughue
    18 January 2021
    
    :param y1: Data vector (MxN)
    :param y2: Desired signal (Mx1)
    :param noise_var: Variance of the noise in y1
    :param num_samples: Number of samples in y1
    :param prob_fa: Acceptable probability of false alarm
    :return: Array of N binary detection results
    """

    # Compute the sufficient statistic
    sigma_0_sq = num_samples * noise_var ** 2 / 2
    suff_stat = np.absolute(np.sum(np.conjugate(y1)*y2, axis=0))**2 / sigma_0_sq

    # Compute the threshold
    eta = stats.chi2.ppf(q=1-prob_fa, df=2)

    # Compare T to eta
    det_result = np.array(suff_stat > eta)

    # In the rare event that T==eta, flip a weighted coin
    coin_flip_mask = suff_stat == eta
    num_flips = np.sum(coin_flip_mask, axis=None)

    if num_flips > 0:
        coin_flip_result = np.random.uniform(low=0., high=1., size=(num_flips,)) > (1 - prob_fa)

        if np.isscalar(det_result):
            det_result = coin_flip_result[0]
        else:
            det_result[coin_flip_mask] = coin_flip_result

    return det_result


def min_sinr(prob_fa, prob_d, corr_time, pulse_duration, bw_noise, bw_signal):
    """
    Compute the required SNR to achieve the desired probability of detection,
    given the maximum acceptable probability of false alarm, and the number
    of complex samples M.

    The returned SNR is the ratio of signal power to complex noise power.

    Ported from MATLAB Code.

    Nicholas O'Donoughue
    18 January 2021

    :param prob_fa: Probability of False Alarm [0-1]
    :param prob_d: Probability of Detection [0-1]
    :param corr_time: Correlation time [sec]
    :param pulse_duration: Pulse Duration [sec]
    :param bw_noise: Noise bandwidth [Hz]
    :param bw_signal: Signal Bandwidth [Hz]
    :return: Signal-to-Noise ratio [dB]
    """

    # Make sure the signal bandwidth and time are observable
    bw_signal = np.minimum(bw_signal, bw_noise)
    pulse_duration = np.minimum(pulse_duration, corr_time)
    M = np.fix(corr_time * bw_noise)
    
    # Find the min SNR after cross-correlation processing
    xi_min_out = squareLaw.min_sinr(prob_fa, prob_d, M)
    
    # Invert the SNR Gain equation
    xi_out_lin = db_to_lin(xi_min_out)
    xi_in_lin = (xi_out_lin + np.sqrt(xi_out_lin * (xi_out_lin + bw_signal * corr_time))) / (pulse_duration * bw_signal)
    xi = lin_to_db(xi_in_lin)

    return xi


def max_range(prob_fa, prob_d, corr_time, pulse_duration, bw_noise, bw_signal, f0, ht, hr, snr0, include_atm_loss=False,
              atmosphere=None):
    """
    Compute the maximum range for a square law detector, as specified by the
    PD, PFA, and number of samples (M).  The link is described by the carrier
    frequency (f0), and transmit/receive antenna heights (ht and hr), and the
    transmitter and receiver are specified by the SNR in the absence of path
    loss (SNR0).  If specified, the atmospheric struct is passed onto the
    path loss model.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    18 January 2021

    :param prob_fa: Probability of False Alarm [0-1]
    :param prob_d: Probability of Detection [0-1]
    :param corr_time: Correlation time [sec]
    :param pulse_duration: Pulse Duration [sec]
    :param bw_noise: Noise bandwidth [Hz]
    :param bw_signal: Signal Bandwidth [Hz]
    :param f0: Carrier Frequency [Hz]
    :param ht: Transmitter height [m]
    :param hr: Receiver height [m]
    :param snr0: SNR in the absence of path loss [dB]
    :param include_atm_loss: Boolean flag indicating whether atmospheric loss should be considered in path loss
                           [Default = false]
    :param atmosphere: (Optional) struct containing atmospheric parameters; can be called from atm.standardAtmosphere().
    :return: Maximum range at which the specified PD/PFA condition can be met [m]
    """

    # Find the required SNR Threshold
    snr_min = min_sinr(prob_fa, prob_d, corr_time, pulse_duration, bw_noise, bw_signal)
    
    max_range = np.zeros(np.shape(snr_min))
    
    for idx_snr_min, this_snr_min in enumerate(snr_min):
        if np.size(snr0) > 1:
            this_snr0 = snr0[idx_snr_min]
        else:
            this_snr0 = snr0
        
        # Find the acceptable propagation loss
        prop_loss_max = this_snr0 - this_snr_min
        
        # Set up error function
        def prop_loss(r):
            return prop.model.get_path_loss(range_m=r, freq_hz=f0, tx_ht_m=ht, rx_ht_m=hr,
                                            include_atm_loss=include_atm_loss, atmosphere=atmosphere)

        def err_fun(r):
            return prop_loss(r) - prop_loss_max
        
        # Set up initial search point
        this_r = 1e3
        err = err_fun(this_r)
        
        # Optimization Parameters
        err_tol = .01       # SNR error tolerance [dB]
        max_iter = 1000     # Maximum number of iterations
        iter_num = 0        # Iteration counter

        # Perform the optimization
        while iter_num < max_iter and abs(err) > err_tol:
            # Compute derivative
            d_r = 1  # 1 meter
            y1 = err_fun(this_r+d_r)
            y0 = err_fun(this_r-d_r)
            df = (y1-y0)/(2*d_r)

            # Error Checking for Flat Derivative to avoid divide by zero
            # when calculating step size
            if df == 0:
                df = 1

            # Newton Method step
            this_r = this_r - err/df
            err = err_fun(this_r)
            
            # Iteration count
            iter_num += 1

        max_range[idx_snr_min] = this_r

    return max_range

