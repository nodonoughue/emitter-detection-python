from atm import reference
import numpy as np
from utils import geo


def calc_atm_loss(freq_hz, gas_path_len_m=0, rain_path_len_m=0, cloud_path_len_m=0, atmosphere=None, pol_angle=0,
                  el_angle=0):
    """
    Ref:
       ITU-R P.676-11(09/2016) Attenuation by atmospheric gases
       ITU-R P.840-6 (09/2013) Attenuation due to clouds and fog
       ITU-R P.838-3 (03/2005) Specific attenuation model for rain for use in
       prediction methods

    Ported from MATLAB Code

    Nicholas O'Donoughue
    16 March 2021

    :param freq_hz: Frequency [Hz]
    :param gas_path_len_m: Path length for gas loss [m] [default = 0]
    :param rain_path_len_m: Path length for rain loss [m] [default = 0]
    :param cloud_path_len_m: Path length for cloud loss [m] [default = 0]
    :param atmosphere: atm.reference.Atmosphere object (if not provided, standard atmosphere will be generated)
    :param pol_angle: Polarization angle [radians], 0 for Horizontal, pi/2 for Vertical, between 0 and pi for slant.
                      [default = 0]
    :param el_angle: Elevation angle of the path under test [default = 0]
    :return: loss along the path due to atmospheric absorption [dB, one-way]
    """

    if atmosphere is None:
        # Default atmosphere is the standard atmosphere at sea level, with no
        # fog/clouds or rain.
        atmosphere = reference.get_standard_atmosphere(0)

    # Compute loss coefficients
    if np.any(gas_path_len_m > 0):
        coeff_ox, coeff_water = get_gas_loss_coeff(freq_hz, atmosphere.press, atmosphere.water_vapor_press,
                                                   atmosphere.temp)
        coeff_gas = coeff_ox + coeff_water
    else:
        coeff_gas = 0
    
    if np.any(rain_path_len_m > 0) and np.any(atmosphere.rainfall) > 0:
        coeff_rain = get_rain_loss_coeff(freq_hz, pol_angle, el_angle, atmosphere.rainfall)
    else:
        coeff_rain = 0

    if np.any(cloud_path_len_m > 0) and np.any(atmosphere.cloud_dens) > 0:
        coeff_cloud = get_fog_loss_coeff(freq_hz, atmosphere.cloud_dens, atmosphere.temp)
    else:
        coeff_cloud = 0
    
    # Compute loss components
    loss_gass_db = coeff_gas * gas_path_len_m / 1.0e3
    loss_rain_db = coeff_rain * rain_path_len_m / 1.0e3
    loss_cloud_db = coeff_cloud * cloud_path_len_m / 1.0e3
    
    return loss_gass_db + loss_rain_db + loss_cloud_db


def calc_zenith_loss(freq_hz, alt_start_m=0, zenith_angle_deg=0):
    """
    # Computes the cumulative loss from alt_start [m] to zenith (100 km
    # altitude), for the given frequencies (freq) in Hz and angle from zenith
    # zenith_angle, in degrees.
    #
    # Does not account for refraction of the signal as it travels through the
    # atmosphere; assumes a straight line propagation at the given zenith
    # angle.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    17 March 2021

    :param freq_hz: Carrier frequency [Hz]
    :param alt_start_m: Starting altitude [m]
    :param zenith_angle_deg: Angle between line of sight and zenith (straight up) [deg]
    :return zenith_loss: Cumulative loss to the edge of the atmosphere [dB]
    :return zenith_loss_o: Cumulative loss due to dry air [dB]
    :return zenith_loss_w: Cumulative loss due to water vapor [dB]
    """

    # Add a new first dimension to all the inputs (if they're not scalar)
    if np.size(freq_hz) > 1:
        freq_hz = np.expand_dims(freq_hz, axis=0)

    if np.size(alt_start_m) > 1:
        alt_start_m = np.expand_dims(alt_start_m, axis=0)

    if np.size(zenith_angle_deg) > 1:
        zenith_angle_deg = np.expand_dims(zenith_angle_deg, axis=0)

    # Make Altitude Layers
    # From ITU-R P.676-11(12/2017), layers should be set at exponential intervals
    num_layers = 922  # Used for ceiling of 100 km
    layer_delta = .0001*np.exp(np.arange(num_layers)/100)  # Layer thicknesses [km], eq 21
    layer_delta = np.reshape(layer_delta, (num_layers, 1))
    layer_top = np.cumsum(layer_delta)  # [km]
    layer_bottom = layer_top - layer_delta  # [km]
    layer_mid = (layer_top+layer_bottom)/2
    
    # Drop layers below alt_start
    alt_start_km = alt_start_m / 1e3
    layer_mask = layer_top >= min(alt_start_km)
    layer_bottom = layer_bottom[layer_mask]
    layer_mid = layer_mid[layer_mask]
    layer_top = layer_top[layer_mask]
    
    # Lookup standard atmosphere for each band
    atmosphere = reference.get_standard_atmosphere(layer_mid*1e3)
    
    # Compute loss coefficient for each band
    ao, aw = get_gas_loss_coeff(freq_hz, atmosphere.P, atmosphere.e, atmosphere.T)
    
    # Account for off-nadir paths and partial layers
    el_angle_deg = 90 - zenith_angle_deg
    layer_delta_eff = geo.compute_slant_range(max(layer_bottom, alt_start_km), layer_top, el_angle_deg, True)
    np.place(layer_delta_eff, layer_top <= alt_start_km, 0)  # Set all layers below alt_start_km to zero
    
    # Zenith Loss by Layer (loss to pass through each layer)
    zenith_loss_by_layer_oxygen = ao*layer_delta_eff
    zenith_loss_by_layer_water = aw*layer_delta_eff
    
    # Cumulative Zenith Loss
    # Loss from ground to the bottom of each layer
    zenith_loss_o = np.squeeze(np.sum(zenith_loss_by_layer_oxygen, axis=0))
    zenith_loss_w = np.squeeze(np.sum(zenith_loss_by_layer_water, axis=0))
    zenith_loss = zenith_loss_o + zenith_loss_w

    return zenith_loss, zenith_loss_o, zenith_loss_w


def get_rain_loss_coeff(freq_hz, pol_angle_rad, el_angle_rad, rainfall_rate):
    """
    Computes the rain loss coefficient given a frequency, polarization,
    elevation angle, and rainfall rate, according to ITU-R P.838-3, 2005.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    16 March 2021

    :param freq_hz: Propagation Frequency [Hz]
    :param pol_angle_rad: Polarization angle [radians], 0 = Horizontal and pi/2 is Vertical.  Slanted polarizations will
                     have a value 0 and pi.
    :param el_angle_rad: Propagation path elevation angle [radians]
    :param rainfall_rate: Rainfall rate [mm/hr]
    :return: Loss coefficient [dB/km] caused by rain.
    """

    # Add a new first dimension to all the inputs (if they're not scalar)
    if np.size(freq_hz) > 1:
        freq_hz = np.expand_dims(freq_hz, axis=0)

    if np.size(pol_angle_rad) > 1:
        pol_angle_rad = np.expand_dims(pol_angle_rad, axis=0)

    if np.size(el_angle_rad) > 1:
        el_angle_rad = np.expand_dims(el_angle_rad, axis=0)

    if np.size(rainfall_rate) > 1:
        rainfall_rate = np.expand_dims(rainfall_rate, axis=0)

    # Coeffs for kh
    a = np.array([-5.3398, -0.35351, -0.23789, -0.94158])
    b = np.array([-0.10008, 1.26970, 0.86036, 0.64552])
    c = np.array([1.13098, 0.454, 0.15354, 0.16817])
    m = -0.18961
    ck = 0.71147
    
    log_kh = np.squeeze(np.sum(a * np.exp(-((np.log10(freq_hz / 1e9) - b) / c) ** 2), axis=0)
                        + m * np.log10(freq_hz / 1e9) + ck)
    kh = 10**log_kh
    
    # Coeffs for kv
    a = np.array([-3.80595, -3.44965, -0.39902, 0.50167])
    b = np.array([0.56934, -0.22911, 0.73042, 1.07319])
    c = np.array([0.81061, 0.51059, 0.11899, 0.27195])
    m = -0.16398
    ck = 0.63297
    
    log_kv = np.squeeze(np.sum(a * np.exp(-((np.log10(freq_hz / 1e9) - b) / c) ** 2), axis=0)
                        + m * np.log10(freq_hz / 1e9) + ck)
    kv = 10**log_kv
    
    # Coeffs for ah
    a = np.array([-0.14318, 0.29591, 0.32177, -5.37610, 16.1721])
    b = np.array([1.82442, 0.77564, 0.63773, -0.96230, -3.29980])
    c = np.array([-0.55187, 0.19822, 0.13164, 1.47828, 3.43990])
    m = 0.67849
    ca = -1.95537
    
    ah = np.squeeze(np.sum(a * np.exp(-((np.log10(freq_hz / 1e9) - b) / c) ** 2), axis=0)
                    + m * np.log10(freq_hz / 1e9) + ca)
    
    # Coeffs for av
    a = np.array([-0.07771, 0.56727, -0.20238, -48.2991, 48.5833])
    b = np.array([2.33840, 0.95545, 1.14520, 0.791669, 0.791459])
    c = np.array([-0.76284, 0.54039, 0.26809, 0.116226, 0.116479])
    m = -0.053739
    ca = 0.83433
    
    av = np.squeeze(np.sum(a * np.exp(-((np.log10(freq_hz / 1e9) - b) / c) ** 2), axis=0)
                    + m * np.log10(freq_hz / 1e9) + ca)
    
    # Account for Polarization and Elevation Angles
    k = .5*(kh + kv + (kh-kv) * np.cos(el_angle_rad) ** 2 * np.cos(2 * pol_angle_rad))
    a = (kh * ah + kv * av + (kh*ah-kv*av) * np.cos(el_angle_rad) ** 2 * np.cos(2 * pol_angle_rad)) / (2 * k)
    
    return k*rainfall_rate**a

    
def get_fog_loss_coeff(f, cloud_dens, temp_k=None):
    """
    Implement the absorption loss coefficient due to clouds and fog, as a function of the frequency, cloud density,
    and temperature, according to ITU-R P.840-7 (2017).

    Ported from MATLAB Code

    Nicholas O'Donoughue
    16 March 2021

    :param f: Propagation Frequencies [Hz]
    :param cloud_dens: Cloud/fog density [g/m^3]
    :param temp_k: Atmospheric temperature [K]
    :return: Loss coefficient [dB/km]
    """

    if temp_k is None:
        atmosphere = reference.get_standard_atmosphere()
        temp_k = atmosphere.temp

    # Cloud Liquid Water Specific Attenuation Coefficient
    theta = 300 / temp_k
    e0 = 77.66+103.3*(theta-1)
    e1 = 0.0671*e0
    e2 = 3.52
    
    fp = 20.20-146*(theta-1)+316*(theta-1)**2
    fs = 39.8*fp
    
    e_prime = (e0-e1)/(1+((f/1e9)/fp)**2)+(e1-e2)/(1+((f/1e9)/fs)**2)+e2
    e_prime_prime = (f/1e9)*(e0-e1)/(fp*(1+(f/1e9/fp)**2))+((f/1e9)*(e1-e2)/(fs*(1+((f/1e9)/fs)**2)))
    
    eta = (2+e_prime)/e_prime_prime
    kl = .819*(f/1e9)/(e_prime_prime*(1+eta**2))
    
    # Cloud attenuation
    return kl * cloud_dens


def get_gas_loss_coeff(freq_hz, press, water_vapor_press, temp):
    """
    Implement the atmospheric loss coefficients from Annex 1 of ITU-R P.676-11 (12/2017)

    If array inputs are specified, then array results are given for alphaO and alphaW.

    :param freq_hz: Propagation Frequencies [Hz]
    :param press: Dry Air Pressure [hPa]
    :param water_vapor_press: Water Vapor Partial Pressure [hPa]
    :param temp: Temperature [K]
    :return coeff_ox: Gas loss coefficient due to oxygen [dB/km]
    :return coeff_water: Gas loss coefficient due to water vapor [dB/km]
    """
    
    # Determine largest dimension in use
    if np.size(freq_hz) > 1:
        freq_hz = np.expand_dims(freq_hz, axis=0)

    if np.size(press) > 1:
        press = np.expand_dims(press, axis=0)

    if np.size(water_vapor_press) > 1:
        water_vapor_press = np.expand_dims(water_vapor_press, axis=0)

    if np.size(temp) > 1:
        temp = np.expand_dims(temp, axis=0)
    
    # Read in the spectroscopic tables (Tables 1 and 2 of Annex 1)
    # All table data will be Nx1 for the N spectroscopic lines of
    # each table.
    ref_ox = reference.get_spectroscopic_table_oxygen()
    ref_water = reference.get_spectroscopic_table_water()

    # Compute the dry continuum due to pressure-induced Nitrogen absorption and the Debye spectrum (eq 8)
    f0 = freq_hz / 1e9  # Convert freq from Hz to GHz
    th = 300 / temp
    d = 5.6e-4 * (press + water_vapor_press) * th ** .8
    nd = f0 * press * th ** 2 * (6.14e-5 / (d * (1 + (f0 / d) ** 2))
                                 + (1.4e-12 * press * th ** 1.5) / (1 + 1.9e-5 * f0 ** 1.5))
    
    # Compute the strength of the i-th water/o2 vapor line (eq 3)
    line_strength_oxygen = np.expand_dims((ref_ox[:, 1] * 1e-7 * press * th ** 3) * np.exp(ref_ox[:, 2] * (1 - th)),
                                          axis=1)
    line_strength_water = np.expand_dims((ref_water[:, 1] * 1e-1 * water_vapor_press * th ** 3.5)
                                         * np.exp(ref_water[:, 2] * (1 - th)), axis=1)
    
    # Compute the line shape factor for each
    #  Correction factor due to interference effects in oxygen lines (eq 7)
    dox = np.expand_dims((ref_ox[:, 5]+ref_ox[:, 6]*th) * 1e-4 * (press + water_vapor_press) * th ** .8, axis=1)
    dw = 0
    
    # spectroscopic line width (eq 6a)
    dfox = np.expand_dims((ref_ox[:, 3]*1e-4)*(press * th ** (.8 - ref_ox[:, 4]) + 1.1 * water_vapor_press * th),
                          axis=1)
    dfw = np.expand_dims((ref_water[:, 3]*1e-4)*(press * th ** (ref_water[:, 4])
                                                 + ref_water[:, 5] * water_vapor_press * th ** ref_water[:, 6]), axis=1)
    
    # modify spectroscopic line width to account for Zeeman splitting of oxygen
    # lines and Doppler broadening of water vapour lines (eq 6b)
    dfox_sq = dfox**2+2.25e-6
    dfox = np.sqrt(dfox_sq)
    dfw = .535*dfw+np.sqrt(.217*dfw**2+(2.1316e-12*np.expand_dims(ref_water[:, 0], axis=1)**2)/th)
    
    # Compute line shape factor
    delta_fox = np.expand_dims(ref_ox[:, 0], axis=1)-f0
    sum_fox = np.expand_dims(ref_ox[:, 0], axis=1)+f0
    delta_fw = np.expand_dims(ref_water[:, 0], axis=1)-f0
    sum_fw = np.expand_dims(ref_water[:, 0], axis=1)+f0
    line_shape_oxygen = f0*(((dfox-dox*delta_fox)/(delta_fox**2+dfox**2))
                            + ((dfox-dox*sum_fox)/(sum_fox**2+dfox_sq))) / np.expand_dims(ref_ox[:, 0], axis=1)
    line_shape_water = f0*(((dfw-dw*delta_fw)/(delta_fw**2+dfw**2))
                           + ((dfw-dw*sum_fw)/(sum_fw**2+dfw**2))) / np.expand_dims(ref_water[:, 0], axis=1)
    
    # Compute complex refractivities
    refractivity_oxygen = np.squeeze(np.sum(line_strength_oxygen*line_shape_oxygen, axis=0)+nd)
    refractivity_water = np.squeeze(np.sum(line_strength_water*line_shape_water, axis=0))
    
    coeff_ox = .1820*f0*refractivity_oxygen
    coeff_water = .1820*f0*refractivity_water
    
    # Handle all freqs < 1 GHz
    if np.any(f0 < 1):
        low_freq_mask = f0 < 1
        if np.isscalar(coeff_ox):
            coeff_ox = 0
        else:
            np.place(coeff_ox, low_freq_mask, 0)

        if np.isscalar(coeff_water):
            coeff_water = 0
        else:
            np.place(coeff_water, low_freq_mask, 0)

    return coeff_ox, coeff_water
