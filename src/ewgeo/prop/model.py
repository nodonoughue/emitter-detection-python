import numpy as np

import ewgeo.atm as atm
from ewgeo.utils.constants import speed_of_light, radius_earth_eff, radius_earth_true


def get_path_loss(range_m, freq_hz, tx_ht_m, rx_ht_m, include_atm_loss=True, atmosphere=None):
    """
    Computes the propagation loss according to a piece-wise model where free space is used at close range, and
    two-ray is used at long range.  The cross-over range between the two is the Fresnel Zone.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    21 March 2021

    :param range_m: Range of link [m]
    :param freq_hz: Carrier frequency [Hz]
    :param tx_ht_m: Transmitter height [m]
    :param rx_ht_m: Receiver height [m]
    :param include_atm_loss: Boolean flag.  If true (default) then atmospheric absorption is modeled [Default=True]
    :param atmosphere: Atmospheric loss parameter struct, must match the format expected by calcAtmLoss. If blank, then
                      a standard atmosphere will be used.
    :return: Path Loss [dB]
    """

    # Find the fresnel zone distance
    fz = get_fresnel_zone(freq_hz, tx_ht_m, rx_ht_m)
    
    # Compute free space path loss - w/out atmospherics
    loss_free_space = get_free_space_path_loss(range_m, freq_hz, False)
    loss_two_ray = get_two_ray_path_loss(range_m, freq_hz, tx_ht_m, rx_ht_m, False)
    broadcast_out = np.broadcast(loss_free_space, loss_two_ray)

    # Combine the free space and two ray path loss calculations, using binary singleton expansion to handle non-uniform
    # parameter sizes, so long as all non-singleton dimension match, this will succeed.
    free_space_mask = range_m < fz
    two_ray_mask = np.logical_not(free_space_mask)

    loss_path = np.zeros(shape=broadcast_out.shape)
    loss_path[free_space_mask] = loss_free_space[free_space_mask]
    loss_path[two_ray_mask] = loss_two_ray[two_ray_mask]

    if include_atm_loss:
        if atmosphere is None:
            atmosphere = atm.reference.get_standard_atmosphere(np.sort(np.unique((tx_ht_m, rx_ht_m))))

        loss_atmosphere = atm.model.calc_atm_loss(freq_hz, gas_path_len_m=range_m, atmosphere=atmosphere)
    else:
        loss_atmosphere = 0

    return loss_path + loss_atmosphere


def get_two_ray_path_loss(range_m, freq_hz, height_tx_m, height_rx_m=None, include_atm_loss=True, atmosphere=None):
    """
    Computes the two-ray path loss according to
         L = 10*log10(R^4/(h_t^2*h_r^2))
    
    This model is generally deemed appropriate for low altitude transmitters and receivers, with a flat Earth model.

    If the input includeAtmLoss is True (default), then a call is also made to the loss_atmosphere function, and the 
    total loss is returned.
    
    Ported from MATLAB Code
    
    Nicholas O'Donoughue
    21 March 2021
    
    :param range_m: Range [m]
    :param freq_hz: Carrier frequency [Hz]
    :param height_tx_m: Height of the transmitter [m]
    :param height_rx_m: Height of the receiver [m]
    :param include_atm_loss: Boolean flag indicating whether atmospheric loss should be included (Default = True)
    :param atmosphere: Optional atmosphere, to be used for atmospheric loss.
    :return: Path Loss [dB]
    """

    if height_rx_m is None:
        height_rx_m = height_tx_m

    # Two-Ray Path Loss    
    loss_two_ray = 10*np.log10(range_m ** 4 / (height_tx_m ** 2 * height_rx_m ** 2))
    
    if include_atm_loss:
        if atmosphere is None:
            atmosphere = atm.reference.get_standard_atmosphere(np.sort(np.unique(height_tx_m, height_rx_m)))
            
        loss_atmosphere = atm.model.calc_atm_loss(freq_hz, gas_path_len_m=range_m, atmosphere=atmosphere)
    else:
        loss_atmosphere = 0
    
    return loss_two_ray + loss_atmosphere


def get_knife_edge_path_loss(dist_tx_m, dist_rx_m, ht_above_los):
    """
    Knife Edge path loss.
    
    Ported from MATLAB Code
    
    Nicholas O'Donoughue
    21 March 2021
    
    :param dist_tx_m: Distance from the transmitter to the obstruction [m]
    :param dist_rx_m: Distance from the obstruction to the receiver [m]
    :param ht_above_los: Vertical distance between the top of the obstruction and the line of sight between the
                         transmitter and receiver [m]
    :return: Path loss [dB]
    """
    
    # Equation (B.5)
    nu = ht_above_los * np.sqrt(2) / (1 + dist_tx_m / dist_rx_m)
    
    # Initialize the output loss matrix
    loss_knife = np.zeros_like(nu)
    
    # First piece-wise component of (B.6)
    np.place(loss_knife, nu <= 0, 0)
    
    # Second piece-wise component of (B.6)
    mask = nu > 0 & nu <= 2.4
    loss_knife[mask] = 6+9*nu[mask]-1.27*nu[mask]**2
    
    # Third piece-wise component of (B.6)
    mask = nu > 2.4
    loss_knife[mask] = 13 + 20*np.log10(nu[mask])

    return loss_knife


def get_free_space_path_loss(range_m, freq_hz, include_atm_loss=True, atmosphere=None, height_tx_m=None,
                             height_rx_m=None):
    """
    Computes the free space path loss according to:
        L = 20*log10(4*pi*R/lambda)

    If the field includeAtmLoss is set to true (default), then atmospheric loss is computed for the path and returned,
    in addition to the path loss.
        
    Ported from MATLAB Code
    
    Nicholas O'Donoughue
    21 March 2021
    
    :param range_m: Range [m]
    :param freq_hz: Carrier frequency [Hz]
    :param include_atm_loss: Boolean flag indicating whether atmospheric loss should be included (Default = True)
    :param atmosphere: Atmospheric loss parameter struct, must match the format expected by calcAtmLoss.
    :param height_tx_m: Optional transmitter altitude (used for atmospheric loss); default=0 [m]
    :param height_rx_m: Optional receiver altitude (used for atmospheric loss); default=0 [m]
    :return: Path loss [dB]
    """
        
    # Convert from frequency to wavelength [m]
    lam = speed_of_light / freq_hz

    # Equation (B.1)
    loss_free_space = 20*np.log10(4 * np.pi * range_m / lam)

    # Add atmospheric loss, if called for
    if include_atm_loss:
        if atmosphere is None:
            atmosphere = atm.reference.get_standard_atmosphere(np.sort(np.unique((height_tx_m, height_rx_m))))
            
        loss_atmosphere = atm.model.calc_atm_loss(freq_hz, gas_path_len_m=range_m, atmosphere=atmosphere)
    else:
        loss_atmosphere = 0

    return loss_free_space+loss_atmosphere


def get_fresnel_zone(f0, ht, hr):
    """
    Computes the Fresnel Zone for a given transmission, given by the equation:
       FZ = 4*pi*h_t*h_r / lambda
    
    Ported from MATLAB Code
    
    Nicholas O'Donoughue
    21 March 2021
    
    :param f0: Carrier frequency [Hz]
    :param ht: Transmitter altitude [m]
    :param hr: Receiver altitude [m]
    :return: Fresnel Zone range [m]
    """
    
    # Convert the carrier frequency to wavelength [m]
    lam = speed_of_light/f0

    # Equation (B.3)
    return 4*np.pi*ht*hr/lam


def compute_radar_horizon(h1, h2, use_four_thirds_radius=True):
    """
    Computes the radar horizon for transmitter and receiver at the given distances above a smooth, round Earth. It does
    not take into consideration terrain, or the Earth's ellipticity.

    Uses the approximation: R = (2 * h * Re + h^2)^0.5, where h is the observer height above the Earth's surface,
    and Re is the Earth's radius.
    
    Leverages the (4/3) Earth radius approximation common for electromagnetic propagation by default.

    Ref: Doerry, Armin Walter. Earth curvature and atmospheric refraction effects on radar signal propagation. 
    doi:10.2172/1088060.
     
    Ported from MATLAB Code

    Nicholas O'Donoughue
    21 March 2021

    :param h1: Height of transmitter [m]
    :param h2: Height of receiver [m]
    :param use_four_thirds_radius: Boolean flag.  If True, will use 4/3 Earth radius approximation
    :return: Radar horizon [m]
    """

    if use_four_thirds_radius:
        radius_earth = radius_earth_eff
    else:
        radius_earth = radius_earth_true
    
    range_1 = np.sqrt(2*h1*radius_earth + h1**2)
    range_2 = np.sqrt(2*h2*radius_earth + h2**2)

    return range_1 + range_2
