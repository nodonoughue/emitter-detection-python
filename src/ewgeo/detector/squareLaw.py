import numpy as np
import scipy.stats as stats
import warnings

import ewgeo.prop as prop
from ewgeo.utils.unit_conversions import db_to_lin


def det_test(z, noise_var, prob_fa):
    """
    Compute detection via the square law and return binary detection events
    
    Ported from MATLAB Code
    
    Nicholas O'Donoughue
    18 January 2021
    
    :param z: Input signal, (MxN) for M samples per detection event, and N separate test
    :param noise_var: Noise variance on input signal
    :param prob_fa: Acceptable probability of false alarm
    :return detResult: Array of N binary detection results 
    """
    
    # Compute the sufficient statistic
    suff_stat = np.sum(np.absolute(z)**2, axis=0)/noise_var
    
    # Compute the threshold
    eta = stats.chi2.ppf(q=1-prob_fa, df=2*np.shape(z)[0])

    # Compare T to eta
    det_result = np.greater(suff_stat, eta)
    
    # In the rare event that T==eta, flip a weighted coin
    coin_flip_mask = suff_stat == eta
    coin_flip_result = np.random.uniform(low=0., high=1.) > (1-prob_fa)

    np.putmask(det_result, mask=coin_flip_mask, values=coin_flip_result)

    return det_result


def min_sinr(prob_fa, prob_d, num_samples):
    """
    Compute the required SNR to achieve the desired probability of detection,
    given the maximum acceptable probability of false alarm, and the number
    of complex samples M.

    The returned SNR is the ratio of signal power to complex noise power.

    Ported from MATLAB Code.

    Nicholas O'Donoughue
    18 January 2021

    :param prob_fa: Probability of False Alarm
    :param prob_d: Probability of Detection
    :param num_samples: Number of samples collected
    :return xi: Required input signal-to-noise ratio [dB]
    """

    eta = stats.chi2.ppf(q=1 - prob_fa, df=2 * num_samples)

    if np.isscalar(eta):
        eta = np.array([eta])

    xi = np.zeros(np.shape(eta))
    for ii, this_eta in enumerate(eta):
        if np.size(num_samples) > 1:
            this_m = num_samples[ii]
        else:
            this_m = num_samples
        
        if np.size(prob_d) > 1:
            this_pd = prob_d[ii]
        else:
            this_pd = prob_d

        # Set up function for probability of detection and error calculation
        def pd_fun(x):
            return stats.ncx2.sf(x=this_eta, df=2*this_m, nc=2*this_m*db_to_lin(x))  # Xi is in dB

        def err_fun(x):
            return pd_fun(x) - this_pd
        
        # Initial Search Value
        this_xi = 0.0           # Start at 0 dB
        err = err_fun(this_xi)  # Compute the difference between PD and desired PD
        
        # Initialize Search Parameters
        err_tol = .0001    # Desired PD error tolerance
        max_iter = 1000    # Maximum number of iterations
        idx = 0            # Current iteration number
        max_step = .5      # Maximum Step Size [dB] - to prevent badly scaled results when PD is near 0 or 1

        # Perform optimization
        while abs(err) > err_tol and idx < max_iter:
            # Compute derivative at the current test point
            dxi = .01  # dB
            y0 = err_fun(this_xi-dxi)
            y1 = err_fun(this_xi+dxi)
            df = (y1-y0)/(2*dxi)

            # Error Checking for Flat Derivative to avoid divide by zero
            # when calculating step size
            if df == 0:
                df = 1

            # Newton-Rhapson Step Size
            step = -err/df
            
            # Ensure that the step size is not overly large
            if abs(step) > max_step:
                step = np.sign(step)*max_step
            
            # Iterate the Newton Approximation
            this_xi = this_xi + step
            err = err_fun(this_xi)
            
            # Increment the iteration counter
            idx += 1

        if idx >= max_iter:
            warnings.warn('Computation finished before suitable tolerance achieved.'
                          '  Error = {:.6f}'.format(np.fabs(err)))

        xi[ii] = this_xi

    return xi

    
def max_range(prob_fa, prob_d, num_samples, f0, ht, hr, snr0, include_atm_loss=False, atm_struct=None):
    """
    Compute the maximum range for a square law detector, as specified by the
    PD, PFA, and number of samples (M).  The link is described by the carrier
    frequency (f0), and transmit/receive antenna heights (ht and hr), and the
    transmitter and receiver are specified by the SNR in the absence of path
    loss (SNR0).  If specified, the atmospheric struct is passed onto the
    path loss model.

    :param prob_fa: Probability of False Alarm
    :param prob_d: Probability of Detection
    :param num_samples: Number of samples collected
    :param f0: Carrier frequency [Hz]
    :param ht: Transmitter height [m]
    :param hr: Receiver height [m]
    :param snr0: Signal-to-Noise ratio [dB] without path loss
    :param include_atm_loss: Binary flag determining whether atmospheric loss is to be included.  [Default=False]
    :param atm_struct: (Optional) struct containing fields that specify atmospherics parameters.  See
                      atm.standardAtmosphere().
    :return range: Maximum range at which square law detector can achieve the PD/PFA test point.
    """

    # Find the required SNR Threshold
    snr_min = min_sinr(prob_fa, prob_d, num_samples)
    
    max_range_vec = np.zeros_like(snr_min)
    
    for idx_snr, this_snr_min in enumerate(snr_min):
        if np.size(snr0) > 1:
            this_snr0 = snr0[idx_snr]
        else:
            this_snr0 = snr0
        
        # Find the acceptable propagation loss
        prop_loss_max = this_snr0 - this_snr_min
        
        # Set up error function
        def prop_loss_fun(r):
            return prop.model.get_path_loss(r, f0, ht, hr, include_atm_loss, atm_struct)

        def err_fun(r):
            return prop_loss_fun(r) - prop_loss_max
        
        # Set up initial search point
        this_range = 1e3
        err = err_fun(this_range)
        
        # Optimization Parameters
        err_tol = .01      # SNR error tolerance [dB]
        max_iter = 1000     # Maximum number of iterations
        iter_num = 0           # Iteration counter
        
        # Perform the optimization
        while iter_num < max_iter and np.fabs(err) > err_tol:
            # Compute derivative
            range_deriv = 1  # 1 meter
            y1 = err_fun(this_range+range_deriv)
            y0 = err_fun(this_range-range_deriv)
            df = (y1-y0)/(2*range_deriv)
            
            # Newton Method step
            this_range = this_range - err/df
            err = err_fun(this_range)
            
            # Iteration count
            iter_num += 1

        max_range_vec[idx_snr] = this_range

    return max_range_vec
