import numpy as np
from utils.unit_conversions import db_to_lin, lin_to_db
from utils import constants
import atm


def get_thermal_noise(bandwidth_hz, noise_figure_db=0, temp_ext_k=0):
    """
    N = thermal_noise(bw,nf,t_ext)

    Compute the total noise power, given the receiver's noise bandwidth, noise figure, and external noise temperature.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    15 March 2021

    :param bandwidth_hz: Receiver noise bandwidth [Hz]
    :param noise_figure_db: Receiver noise figure [dB] (DEFAULT = 0 dB)
    :param temp_ext_k: External noise temp [K] (DEFAULT = 0 K)
    :return: Thermal noise power [dBW]
    """

    # Add the external noise temp to the reference temp (270 K)
    temp = constants.ref_temp + temp_ext_k

    # Boltzmann's Constant
    k = constants.boltzmann

    # Equation (D.6)
    return lin_to_db(k * temp * bandwidth_hz) + noise_figure_db
    
    
def get_atmospheric_noise_temp(freq_hz, alt_start_m=0, el_angle_deg=90):
    """
    Computes the noise temperature contribution from the reradaition of
    energy absorbed by the atmosphere in the direction of the antenna's
    mainlobe.

    Ported from MATLAB code.

    Nicholas O'Donoughue
    15 March 2021

    :param freq_hz: Frequency [Hz]
    :param alt_start_m: Altitude of receiver [m]
    :param el_angle_deg: Elevation angle of receive mainbeam [degrees above local ground plane]
    :return: Atmospheric noise temperature [K]
    """

    # Assume integrated antenna gain is unity
    alpha_a = 1

    # Compute zenith loss along main propagation path
    zenith_angle_deg = (90-el_angle_deg)
    loss_db, _, _ = atm.model.calc_zenith_loss(freq_hz=freq_hz, alt_start_m=alt_start_m,
                                               zenith_angle_deg=zenith_angle_deg)
    loss_lin = db_to_lin(loss_db)

    # Compute average atmospheric temp
    alt_bands = np.arange(start=alt_start_m, stop=100.0e3+100, step=100)
    atmosphere = atm.reference.get_standard_atmosphere(alt_bands)
    t_atmos = np.mean(atmosphere.temp)
    # t_atmos = utils.constants.T0;

    # Equation D.12
    return alpha_a * t_atmos * (1-1/loss_lin)


def get_sun_noise_temp(freq_hz):
    """
    Returns the noise temp (in Kelvin) for the sun at the specified
    frequency f (in Hertz). f can be a scalar, or N-dimensional matrix.

    Assumes a quiet sun, and represents a rough approximation from ITU
    documentation on radio noise.  Sun noise can be several orders of
    magnitude larger during solar disturbances.

    Ref: Rec. ITU-R P.372-14

    Ported from MATLAB Code

    Nicholas O'Donoughue
    15 March 2021

    :param freq_hz: Carrier frequency [Hz]
    :return: Sun noise temp [K]
    """

    # Based on a visual reading on Figure 12 and the corresponding text
    f_ghz = np.hstack((np.array([.05, .2]), np.arange(start=1, stop=10, step=1),
                       np.arange(start=10, step=10, stop=110)))
    t_ref = np.asarray([1e6, 1e6, 2e5, 9e4, 4.5e4, 2.9e4, 2e4, 1.6e4, 1.4e4, 1.3e4, 1.2e4, 1e4, 7e3, 6.3e3,
                        6.2e3, 6e3, 6e3, 6e3, 6e3, 6e3, 6e3])

    # Perform linear interpolation
    return np.interp(xp=f_ghz, fp=t_ref, x=freq_hz/1e9, left=0, right=0)


def get_moon_noise_temp():
    """
    Returns the noise temp (in Kelvin) for the moon.

    The moon noise temp is fairly constant across spectrum, with ~140 K during new moon phase and ~280 K during at full 
    moon.  Using the arithmatic mean here as an approximate value.
    
    Ported from MATLAB Code

    Ref: Rec. ITU-R P.372-8
        
    Nicholas O'Donoughue
    15 March 2021
        
    :return: Moon noise temp [K]
    """

    return (140 + 280)/2


def get_cosmic_noise_temp(freq_hz, rx_alt_m=0, alpha_c=0.95, gain_sun_dbi=-np.inf, gain_moon_dbi=-np.inf):
    """
    Computes the combined cosmic noise temperature, including contributions from the sun, the moon, and the galactic
    background.  Includes approximate effect of atmospheric loss (sun and moon are treated as as coming from zenith;
    rather than their true angles.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    15 March 2021

    :param freq_hz: Carrier frequency [Hz]
    :param rx_alt_m: Receiver altitude [m]
    :param alpha_c: Fraction of the antenna's receive pattern that is above the horizon [0-1]
    :param gain_sun_dbi: Antenna gain directed at the sun [dBi]
    :param gain_moon_dbi: Antenna gain directed at the moon [dBi]
    :return: Combined cosmic noise temperature [K]
    """

    # Compute Raw Noise Temp
    temp_100_mhz = 3050  # Geometric mean of 100 MHz noise spectrum samples

    temp_cosmic = temp_100_mhz * (100e6 / freq_hz) ** 2.5 + 2.7
    temp_sun = get_sun_noise_temp(freq_hz)
    temp_moon = get_moon_noise_temp()

    # Above 2 GHz, the only contribution is from cosmic background radiation
    # (2.7 K), which is essentially negligible.
    high_freq_mask = freq_hz >= 2e9
    np.place(temp_cosmic, high_freq_mask, 2.7)

    # Apply Antenna Patterns
    gain_sun_lin = db_to_lin(gain_sun_dbi)
    gain_moon_lin = db_to_lin(gain_moon_dbi)

    init_temp = (temp_cosmic * alpha_c) + (temp_sun * 4.75e-6 * gain_sun_lin) + (temp_moon * 4.75e-6 * gain_moon_lin)

    # Compute Atmospheric Losses for Zenith Path at pi/4 (45 deg from zenith)
    zenith_loss_db, _, _ = atm.model.calc_zenith_loss(freq_hz, rx_alt_m, np.pi / 4)
    zenith_loss_lin = db_to_lin(np.reshape(zenith_loss_db, np.shape(freq_hz)))

    # Apply Atmospheric Loss to combined galactic noise temp
    return init_temp / zenith_loss_lin


def get_ground_noise_temp(ant_gain_ground_dbi=-5, ground_emissivity=1, angular_area=np.pi):
    """
    Compute the combined noise temperature from ground effects; predominantly caused by reradiation of thermal energy 
    from the sun.
    
    Ported from MATLAB Code
    
    Nicholas O'Donoughue
    15 March 2021
        
    :param ant_gain_ground_dbi: Average antenna gain in direction of the ground [dBi] (DEFAULT = -5 dBi)
    :param ground_emissivity: Emissivity of ground (Default = 1)
    :param angular_area: Area (in steradians) of ground as visible from antenna (DEFAULT = pi)
    :return: Ground noise temperature [K]
    """

    # Convert average ground antenna gain to linear units
    gain_lin = db_to_lin(ant_gain_ground_dbi)
    
    # Assume ground temp is 290 K (ref temp)
    thermal_temp_ground = constants.ref_temp
    
    # Compute ground noise temp according to (D.13)
    return angular_area * gain_lin * ground_emissivity * thermal_temp_ground / (4*np.pi)
