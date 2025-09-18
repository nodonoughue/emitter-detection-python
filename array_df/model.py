import numpy as np
import warnings


def compute_array_factor(v_fun, h, psi):
    """
    Computes the array factor given a beamformer h, and array
    steering vector v (function handle) evaluated at a set of
    angles psi (in radians).

    The return is the response of the specified beamformer (h) to
    a plane wave (defined by v_fun) at various possible source angles.
    The outputs is in linear units of amplitude.

    Ported from MATLAB code

    Nicholas O'Donoughue
    18 January 2021

    :param v_fun: Function handle that returns N-element vector of complex values
    :param h: Beamformer factor (length N); will be normalized to peak amplitude of 1
    :param psi: Angles (in radians) over which to evaluate v_fun
    :return af: Array output at each steering angle
    """

    # Normalize the beamformer and make it a column vector
    h = np.reshape(h, shape=(np.size(h), 1))/np.max(np.abs(h))

    # Generate the steering vectors
    vv = v_fun(psi)  # should be N x numel(psi)

    # Compute the inner product
    return np.conjugate(vv.T).dot(h)


def compute_array_factor_ula(d_lam, num_elements, psi, psi_0=np.pi / 2, el_pattern=lambda x: 1):
    """
    Computes the array factor for a uniform linear array with specified
    parameters.

    Ported from MATLAB code.
    
    Nicholas O'Donoughue
    18 January 2021
    
    :param d_lam: Inter-element spacing (in wavelengths) 
    :param num_elements: Number of array elements
    :param psi: Incoming signal angle [radians]
    :param psi_0: Steering angle [radians]
    :param el_pattern: Optional element pattern (function handle that accepts psi and returns the individual element
                       amplitude).
    :return af: Array output at each steering angle
    """
    
    # Build the Array Pattern -- ignore runtime warnings, we're going to handle divide by zero cases in a few lines
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        af = np.fabs(np.sin(num_elements * np.pi * d_lam * (np.sin(psi) - np.sin(psi_0))) /
                     (num_elements * (np.sin(np.pi * d_lam * (np.sin(psi) - np.sin(psi_0))))))

    # Look for grating lobes
    epsilon = 1e-6
    mask = np.less(np.fabs(np.mod(d_lam*(np.sin(psi)-np.sin(psi_0)) + .5, 1) - .5), epsilon)
    np.putmask(af, mask=mask, values=1)

    # Apply the element pattern
    el = el_pattern(psi)
    return af * el


def make_steering_vector(d_lam, num_elements):
    """
    Returns an array manifold for a uniform linear array with N elements
    and inter-element spacing d_lam.

    Ported from MATLAB code.

    Nicholas O'Donoughue
    18 January 2021

    :param d_lam: Inter-element spacing, in units of wavelengths
    :param num_elements: Number of elements in array
    :return v: Function handle that accepts an angle psi (in radians) and returns an N-element vector of complex phase
               shifts for each element.  If multiple angles are supplied, the output is a matrix of size N x numel(psi).
    :return v_dot: Function handle that computes the gradient of v(psi) with respect to psi.  Returns a matrix of
                   size N x numel(psi)
    """

    element_idx_vec = np.expand_dims(np.arange(num_elements), axis=1)  # Make it 2D, elements along first dim

    def steer(psi):
        return np.exp(1j * 2 * np.pi * d_lam * element_idx_vec * np.sin(np.atleast_1d(psi)[np.newaxis, :]))

    def steer_grad(psi):
        return (-1j * 2 * np.pi * d_lam * element_idx_vec * np.cos(np.atleast_1d(psi)[np.newaxis, :])) * steer(psi)

    return steer, steer_grad
