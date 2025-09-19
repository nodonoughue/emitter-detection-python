import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf, erfinv


def run_all_examples():
    """
    Run all chapter 2 examples and return a list of figure handles

    :return figs: list of figure handles
    """

    return [example2()]


def example2():
    """
    Executes Example 2.2.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    23 March 2021

    :return: figure handle to generated graphic
    """

    # Set up PFA and SNR vectors
    prob_fa = np.expand_dims(np.linspace(start=1.0e-9, stop=1, num=1001), axis=1)
    d2_vec = np.expand_dims(np.array([1e-6, 1, 5, 10]), axis=0)  # Use 1e-6 instead of 0, to avoid NaN

    # Compute the threshold eta using MATLAB's built-in error function erf(x)
    # and inverse error function erfinv(x).
    eta = np.sqrt(2*d2_vec) * erfinv(1-2*prob_fa)-d2_vec/2

    # Compute the probability of detection
    prob_det = .5*(1-erf((eta-d2_vec/2)/np.sqrt(2*d2_vec)))

    # Plot the ROC curve
    fig = plt.figure()
    for idx, d2 in enumerate(d2_vec[0, :]):
        plt.plot(prob_fa, prob_det[:, idx], label='$d^2$ = {:.0f}'.format(d2))

    # Axes Labels
    plt.ylabel('$P_D$')
    plt.xlabel('$P_{FA}$')

    # Legend
    plt.legend(loc='lower right')

    # Align the axes
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    return fig


if __name__ == '__main__':
    run_all_examples()
    plt.show()
