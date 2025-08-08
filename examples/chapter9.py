import numpy as np
import matplotlib.pyplot as plt
from utils import errors


def run_all_examples():
    """
    Run all chapter 9 examples and return a list of figure handles

    :return figs: list of figure handles
    """

    fig1 = example1()
    fig2 = example2()

    return [fig1, fig2]


def example1():
    """
    Executes Example 9.1 and generates one figure

    Ported from MATLAB Code
    
    Nicholas O'Donoughue
    17 May 2021

    :return fig: figure handle to generated graphic
    """
    
    # Set up error covariance matrix
    covariance = np.array([[10, -3], [-3, 5]])
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    sort_index = np.argsort(eigenvalues)

    v_max = eigenvectors[:, sort_index[-1]]
    # v_min = eigenvectors[:, sort_index[0]]  -- not used
    lam_max = eigenvalues[sort_index[-1]]
    lam_min = eigenvalues[sort_index[0]]

    gamma = 4.601  # 90% confidence interval
    a = np.sqrt(gamma*lam_max)
    b = np.sqrt(gamma*lam_min)
    
    num_plot_points = 101
    th = np.linspace(0, 2*np.pi, num_plot_points)
    x1 = a * np.cos(th)
    x2 = b * np.sin(th)
    
    alpha = np.arctan2(v_max[1], v_max[0])
    x = x1 * np.cos(alpha) - x2 * np.sin(alpha)
    y = x1 * np.sin(alpha) + x2 * np.cos(alpha)
    
    fig = plt.figure()
    plt.scatter(0, 0, marker='+', label='Bias Point')
    plt.text(-2.25, .25, 'Bias Point')
    plt.text(3.5, 3, '90% Error Ellipse')
    plt.plot(x, y, linestyle='-', label='Error Ellipse')

    # Draw the semi-minor and semi-major axes
    plt.plot([0, -a*np.cos(alpha)], [0, -a*np.sin(alpha)], color='k', linestyle='--')
    plt.plot([0, b*np.sin(alpha)], [0, -b*np.cos(alpha)], color='k', linestyle='--')
    plt.text(4.5, -2, '$r_1=7.24$', fontsize=12)
    plt.text(1.1, 2, '$r_2=4.07$', fontsize=12)
    
    plt.plot([0, 3], [0, 0], color='k', linestyle='--')
    th_vec = np.pi / 180.0 * np.arange(start=0, stop=-25, step=-0.1)
    plt.plot(2*np.cos(th_vec), 2*np.sin(th_vec), color='k', linestyle='-', linewidth=.5)
    plt.text(2.1, -.75, r'$\alpha = -25^\circ$', fontsize=12)

    return fig


def example2():
    """
    Executes Example 9.2 and generates one figure

    Ported from MATLAB Code

    Nicholas O'Donoughue
    17 May 2021

    :return: figure handle to generated graphic
    """
       
    # Set up error covariance matrix
    covariance = np.array([[10.0, -3.0], [-3.0, 5.0]])
    
    cep50 = errors.compute_cep50(covariance)
    print('CEP50: {:0.2f}'.format(cep50))
    
    x_ell, y_ell = errors.draw_error_ellipse(np.array([0, 0]), covariance, num_pts=101, conf_interval=50)
    x_cep, y_cep = errors.draw_cep50(np.array([0, 0]), covariance, num_pts=101)

    # Draw the Ellipse and CEP
    fig = plt.figure()
    plt.scatter(0, 0, marker='^', label='Bias Point')
    plt.plot(x_ell, y_ell, linestyle='--', label='Error Ellipse')
    plt.plot(x_cep, y_cep, linestyle='-', label='$CEP_{50}$')

    # Annotation
    plt.text(-1.3, .1, 'Bias Point', fontsize=12)
    plt.text(-.4, 1.4, '50% Error Ellipse', fontsize=12)
    plt.text(2.2, 2.3, r'$CEP_{50}$', fontsize=12)

    return fig


if __name__ == '__main__':
    run_all_examples()
    plt.show()
