import numpy as np
import matplotlib.pyplot as plt
from utils.unit_conversions import db_to_lin
from scipy import stats


def run_all_examples():
    """
    Run all chapter 4 examples and return a list of figure handles

    :return figs: list of figure handles
    """

    return [example1(), example2(), example3()]


def example1():
    """
    Executes Example 5.1.

    Ported from MATLAB Code
    
    Nicholas O'Donoughue
    25 March 2021

    :return: figure handle to generated graphic
    """

    # Scan Variables
    t_hop = np.array([10e-3, 1e-3, 1e-4])  # Target signal hopping period
    bw_hop = 200e6                         # Target signal hopping bandwidth
    # bw_signal = 5e6                        # Target signal transmit bandwidth
    bw_receiver = 5e6                      # Target signal receive bandwidth
    
    det_time = t_hop*bw_receiver/bw_hop
    
    sample_freq = bw_receiver
    num_samples = np.expand_dims(np.floor(det_time*sample_freq), axis=0)
    
    # Detection Curve
    snr_db_vec = np.arange(start=-15.0, step=.1, stop=10.0)
    snr_lin_vec = np.expand_dims(db_to_lin(snr_db_vec), axis=1)
    prob_fa = 1e-6
    
    threshold = stats.chi2.ppf(q=1-prob_fa, df=2*num_samples)
    prob_det = stats.ncx2.sf(x=threshold, df=2*num_samples, nc=2*num_samples*snr_lin_vec)

    fig = plt.figure()
    for idx, this_thop in enumerate(t_hop):
        plt.plot(snr_db_vec, prob_det[:, idx], label=r'$T_{{hop}}$ = ' + '{:.1f} ms'.format(this_thop*1e3))

    plt.xlabel('SNR [dB]')
    plt.ylabel('$P_D$')
    plt.legend(loc='lower right')

    return fig

    
def example2():
    """
    Executes Example 5.2.
    
    Ported from MATLAB Code
    
    Nicholas O'Donoughue
    25 March 2021

    :return: figure handle to generated graphic
    """

    t_hop = np.array([1e-3, 1e-4, 1e-5])
    bw_signal = np.arange(start=1e5, step=1e5, stop=1e7)
    bw_hop = 4e9
    
    t_dwell = 1/bw_signal
    
    num_scans = np.expand_dims(t_hop, axis=0)/np.expand_dims(t_dwell, axis=1)
    bw_receiver = np.maximum(np.expand_dims(bw_signal, axis=1), bw_hop/num_scans)
    
    fig = plt.figure()
    for idx, this_thop in enumerate(t_hop):
        plt.loglog(bw_signal/1e6, bw_receiver[:, idx]/1e6, label=r'$T_{hop}$' + '={:.2f} ms'.format(this_thop*1e3))

    plt.xlabel(r'Frequency Resolution ($\delta_f$) [MHz]')
    plt.ylabel(r'Receiver Bandwidth ($B_r$) [MHz]')
    plt.legend(loc='upper right')

    return fig


def example3():
    """
    Executes Example 5.3.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    25 March 2021

    :return: figure handle to generated graphic
    """

    # Define input parameters
    bw_signal = np.arange(start=1e5, step=1e5, stop=1e7)
    bw_hop = 4e9
    t_hop = np.array([1e-3, 1e-4, 1e-5])
    duty = .2
    pulse_duration = duty*t_hop
    
    t_dwell = 1/bw_signal
    num_scans = np.expand_dims(pulse_duration, axis=0)/np.expand_dims(t_dwell, axis=1)
    bw_receive = np.maximum(np.expand_dims(bw_signal, axis=1), bw_hop/num_scans)
    
    fig = plt.figure()
    for idx, this_tp in enumerate(pulse_duration):
        plt.loglog(bw_signal/1e6, bw_receive[:, idx]/1e6, label='$t_p$={:.0f}'.format(this_tp*1e6) + r'$\mu$s')

    plt.xlabel(r'Frequency Resolution ($\delta_f$) [MHz]')
    plt.ylabel(r'Receiver Bandwidth ($B_r$) [MHz]')
    plt.legend(loc='upper right')
    
    return fig
