"""
Draw Figures - Chapter 1

This script generates all the figures that appear in Chapter 1 of the textbook.

Ported from MATLAB Code

Nicholas O'Donoughue
21 March 2021
"""

from .. import utils
from ..utils.unit_conversions import lin_to_db, db_to_lin
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def make_all_figures(close_figs=False):
    """
    Call all the figure generators for this chapter

    :close_figs: Boolean flag.  If true, will close all figures after generating them; for batch scripting.
                 Default=False
    :return: List of figure handles
    """

    # Reset the random number generator, to ensure reproducibility
    rng = np.random.default_rng()

    # Find the output directory
    prefix = utils.init_output_dir('chapter1')

    # Activate seaborn for prettier plots
    sns.set()

    # Generate all figures
    fig1 = make_figure_1(prefix, rng)
    fig2 = make_figure_2(prefix)
    fig3 = make_figure_3(prefix)

    figs = [fig1, fig2, fig3]
    if close_figs:
        for fig in figs:
            plt.close(fig)

        return None
    else:
        plt.show()

        return figs


def make_figure_1(prefix=None, rng=None):
    """
    Figure 1, Detection Threshold

    :param prefix: output directory to place generated figure
    :param rng: random number generator
    :return: figure handle
    """

    # Initialize variables
    noise_pwr_db = 0  # noise power, dB
    num_samples = 512  # number of points
    n = np.sqrt(10**(noise_pwr_db/10)/2) * (rng.standard_normal((num_samples, 1))
                                            + 1j*rng.standard_normal((num_samples, 1)))

    # Compute Threshold
    prob_false_alarm = 1e-12
    threshold = np.sqrt(-np.log10(prob_false_alarm))

    # Manually spike one noise sample
    index_spike = rng.integers(low=0, high=num_samples-1, size=(1, ))
    n[index_spike] = threshold+1

    # Target Positions
    index_1 = int(np.fix(num_samples/3))
    amplitude_1 = threshold + 3
    index_2 = int(np.fix(5*num_samples/8))
    amplitude_2 = threshold - 1

    # Target signal - use the auto-correlation of a window of length N to generate the lobing structure
    t = 10*np.pi*np.linspace(start=-1, stop=1, num=num_samples)
    p = np.sinc(t)/np.sqrt(np.sum(np.sinc(t)**2, axis=0))
    s = np.zeros((1, num_samples))
    s = s + np.roll(p, index_1, axis=0) * db_to_lin(amplitude_1)
    s = s + np.roll(p, index_2, axis=0) * db_to_lin(amplitude_2)
    
    # Apply a matched filter with the signal p
    s = np.reshape(np.fft.ifft(np.fft.fft(s, num_samples)
                               * np.conj(np.fft.fft(p, num_samples)), num_samples), (num_samples, 1))

    # Plot Noise and Threshold
    fig1 = plt.figure()
    sample_vec = np.reshape(np.arange(num_samples), (num_samples, 1))
    plt.plot(sample_vec, lin_to_db(np.abs(s)), label='Signals', linewidth=2)
    plt.plot(sample_vec, lin_to_db(np.abs(n)), label='Noise', linewidth=.5)
    plt.plot(np.array([1, num_samples]), threshold*np.array([1, 1]), linestyle='--', label='Threshold')

    # Set axes limits
    plt.ylim([-5, threshold+5])
    plt.xlim([1, num_samples])

    # Add overlay text
    plt.text(index_1+(num_samples/50), amplitude_1, 'Detection', fontsize=12)
    plt.text(index_2, amplitude_2+.5, 'Missed Detection', fontsize=12)
    plt.text(index_spike-10, threshold+4, 'False Alarm', fontsize=12)

    # Add legend
    plt.legend(loc='upper right')
    
    # Display the plot
    plt.draw()

    # Output to file
    if prefix is not None:
        fig1.savefig(prefix + 'fig1.svg')
        fig1.savefig(prefix + 'fig1.png')

    return fig1


def make_figure_2(prefix=None):
    """
    Figure 2, AOA Geometry

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    # Compute an AOA slice from sensor 1

    # Initialize Detector/Source Locations
    x1 = np.array([0, 0])
    xs = np.array([.1, .9])

    # Compute Ranges
    r1 = utils.geo.calc_range(x1, xs)

    # Error Values
    angle_error = 5*np.pi/180

    # Find AOA
    lob = xs - x1
    aoa1 = np.arctan2(lob[1], lob[0])
    x_aoa_1 = x1 + np.array([[0, np.cos(aoa1)], [0, np.sin(aoa1)]])*5*r1
    x_aoa_1_plus = x1 + np.array([[0, np.cos(aoa1+angle_error)], [0, np.sin(aoa1+angle_error)]])*5*r1
    x_aoa_1_minus = x1 + np.array([[0, np.cos(aoa1-angle_error)], [0, np.sin(aoa1-angle_error)]])*5*r1
    lob_fill1 = np.concatenate((x_aoa_1_plus, np.fliplr(x_aoa_1_minus),
                                np.expand_dims(x_aoa_1_plus[:, 0], axis=1)), axis=1)

    # Draw Figure
    fig2 = plt.figure()

    # LOBs
    plt.plot(x_aoa_1[0, :], x_aoa_1[1, :], linestyle='-', color='k', label='AOA Solution')

    # Uncertainty Intervals
    plt.fill(lob_fill1[0, :], lob_fill1[1, :], linestyle='--', alpha=.1, edgecolor='k', label='Uncertainty Interval')

    # Position Markers
    plt.scatter(x1[0], x1[1], marker='o', label='Sensor')
    plt.scatter(xs[0], xs[1], marker='^', label='Transmitter')

    # Adjust Axes
    plt.legend(loc='lower right')
    plt.ylim([-.5, 1.5])

    # Draw the figure
    plt.draw()

    # Save figure
    if prefix is not None:
        fig2.savefig(prefix + 'fig2.svg')
        fig2.savefig(prefix + 'fig2.png')

    return fig2


def make_figure_3(prefix=None):
    """
    Figure 3, Geolocation Geometry
    Compute an isochrone between sensors 1 and 2, then draw an AOA slice from sensor 3

    Ported from MATLAB Code

    Nicholas O'Donoughue
    22 March 2021

    :param prefix: output directory to place generated figure
    :return: figure handle
    """

    # Initialize Detector/Source Locations
    x1 = np.array([0, 0])
    x2 = np.array([1, 1])
    xs = np.array([.1, .9])
    
    # Compute Ranges
    r1 = utils.geo.calc_range(x1, xs)
    r2 = utils.geo.calc_range(x2, xs)
    
    # Error Values
    angle_error = 5*np.pi/180
    
    # Find AOA 
    lob1 = xs - x1
    aoa1 = np.arctan2(lob1[1], lob1[0])
    x_aoa_1 = x1 + np.array([[0, np.cos(aoa1)], [0, np.sin(aoa1)]])*5*r1
    x_aoa_1_plus = x1 + np.array([[0, np.cos(aoa1+angle_error)], [0, np.sin(aoa1+angle_error)]])*5*r1
    x_aoa_1_minus = x1 + np.array([[0, np.cos(aoa1-angle_error)], [0, np.sin(aoa1-angle_error)]])*5*r1
    lob_fill1 = np.concatenate((x_aoa_1_plus, np.fliplr(x_aoa_1_minus),
                                np.expand_dims(x_aoa_1_plus[:, 0], axis=1)), axis=1)
    
    lob2 = xs - x2
    aoa2 = np.arctan2(lob2[1], lob2[0])
    x_aoa_2 = x2 + np.array([[0, np.cos(aoa2)], [0, np.sin(aoa2)]])*5*r2
    x_aoa_2_plus = x2 + np.array([[0, np.cos(aoa2+angle_error)], [0, np.sin(aoa2+angle_error)]])*5*r2
    x_aoa_2_minus = x2 + np.array([[0, np.cos(aoa2-angle_error)], [0, np.sin(aoa2-angle_error)]])*5*r2
    lob_fill2 = np.concatenate((x_aoa_2_plus, np.fliplr(x_aoa_2_minus),
                                np.expand_dims(x_aoa_2_plus[:, 0], axis=1)), axis=1)

    # Draw Figure
    fig3 = plt.figure()
    
    # LOBs
    plt.plot(x_aoa_1[0, :], x_aoa_1[1, :], linestyle='-', color='b', label='AOA Solution')
    plt.plot(x_aoa_2[0, :], x_aoa_2[1, :], linestyle='-', color='b', label=None)

    # Uncertainty Intervals
    plt.fill(lob_fill1[0, :], lob_fill1[1, :], facecolor='k', alpha=.1, linestyle='--',
             label='Uncertainty Interval')
    plt.fill(lob_fill2[0, :], lob_fill2[1, :], facecolor='k', alpha=.1, linestyle='--', label=None)
    
    # Position Markers
    plt.scatter(np.array([x1[0], x2[0]]), np.array([x1[1], x2[1]]), marker='o', label='Sensors')
    plt.scatter(xs[0], xs[1], marker='^', label='Transmitter')
    
    # Position Labels
    plt.text(x1[0]+.05, x1[1]-.1, '$S_1$')
    plt.text(x2[0]+.05, x2[1]-.1, '$S_2$')
    
    # Adjust Axes
    plt.legend(loc='lower right')
    plt.ylim([-.5, 1.5])
    plt.xlim([-1, 2])
    
    # Draw the figure
    plt.draw()

    # Save figure
    if prefix is not None:
        fig3.savefig(prefix + 'fig3.svg')
        fig3.savefig(prefix + 'fig3.png')

    return fig3
