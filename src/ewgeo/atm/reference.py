import numpy as np


class Atmosphere:
    def __init__(self, alt, temp, press, water_vapor_dens, water_vapor_press, alt_geopotential=None, rainfall=None,
                 cloud_dens=None):
        self.alt = alt
        self.alt_geopotential = alt_geopotential
        self.temp = temp
        self.press = press
        self.water_vapor_dens = water_vapor_dens
        self.water_vapor_press = water_vapor_press
        self.rainfall = rainfall
        self.cloud_dens = cloud_dens


def get_standard_atmosphere(alt_bands=np.array([0])):
    """
    Implement the standard atmosphere recommended by ITU-R pressure.835-6 (12/2017).

    Requires an input altitude.  If an array of altitudes are defined, the function is run once per input and the
    array of outputs are returned.

    Currently using the mean annual standard temperature, as opposed to the seasonal and latitude-specific models also
    provided in ITU-R pressure.835-6 (12/2017).

    :param alt_bands: Altitude [m], can be scalar or N-dimensional matrix
    :return atmosphere: An Atmosphere object representing the pressure, temperature and water vapor content at each
                        of the specified altitudes.
    """

    # Use recursion to handle array inputs
    if np.size(alt_bands) > 1:
        # Use python list comprehension to generate a list of atmosphere objects
        atms = [get_standard_atmosphere(alt) for alt in alt_bands]

        # Convert to a single atmosphere object for easier handling
        pressure = np.array([atm.press for atm in atms])
        temp = np.array([atm.temp for atm in atms])
        water_dens = np.array([atm.water_vapor_dens for atm in atms])
        water_press = np.array([atm.water_vapor_press for atm in atms])
        rainfall = np.array([atm.rainfall for atm in atms])
        cloud_dens = np.array([atm.cloud_dens for atm in atms])

        atmosphere = Atmosphere(alt_bands, temp, pressure, water_dens, water_press, rainfall, cloud_dens)

        return atmosphere

    # At this point, alt is a scalar
    if np.isscalar(alt_bands):
        alt = alt_bands
    else:
        alt = alt_bands[0]

    # Compute Geopotential Height from Geometric Height
    # Equation 1a
    hp = 6356.766 * (alt / 1e3) / (6356.766 + (alt / 1e3))  # [km']
    h = 6356.766 * hp / (6356.766 - hp)

    # Temperature (K)
    # Equation (2a)-(2g)
    if hp <= 11:
        temp = 288.15 - 6.5 * hp
    elif hp <= 20:
        temp = 216.65
    elif hp <= 32:
        temp = 216.65 + (hp - 20)
    elif hp <= 47:
        temp = 228.65 + 2.8 * (hp - 32)
    elif hp <= 51:
        temp = 270.65
    elif hp <= 71:
        temp = 270.65 - 2.8 * (hp - 51)
    elif hp <= 84.852:
        temp = 214.65 - 2 * (hp - 71)
    elif h <= 91:  # Region 2
        temp = 186.8673
    elif h <= 100:
        temp = 263.1905 - 76.3232 * (1 - ((h - 91) / 19.9429) ** 2) ** .5
    else:
        raise ValueError('Standard temperature not defined above 100 km altitude.')

    # Pressure (hPa)
    if hp <= 11:
        pressure = 1013.25 * (288.15 / (288.15 - 6.5 * hp)) ** (-34.1623 / 6.5)
    elif hp <= 20:
        pressure = 226.3226 * np.exp(-34.1634 * (hp - 11) / 216.65)
    elif hp <= 32:
        pressure = 54.74980 * (216.65 / (216.65 + (hp - 20))) ** 34.1632
    elif hp <= 47:
        pressure = 8.680422 * (228.65 / (228.65 + 2.8 * (hp - 32))) ** (34.1632 / 2.8)
    elif hp <= 51:
        pressure = 1.109106 * np.exp(-34.1632 * (hp - 47) / 170.65)
    elif hp <= 71:
        pressure = 0.6694167 * (270.65 / (270.65 - 2.8 * (hp - 51))) ** (-34.1632 / 2.8)
    elif hp <= 84.852:
        pressure = 0.03956649 * (214.65 / (214.65 - 2 * (hp - 71))) ** (-34.1632 / 2)
    elif h <= 100:
        a0 = 95.571899
        a1 = -4.011801
        a2 = 6.424731e-2
        a3 = -4.789660e-4
        a4 = 1.340543e-6
        pressure = np.exp(a0 + a1 * h + a2 * h ** 2 + a3 * h ** 3 + a4 * h ** 4)
    else:
        raise ValueError('Standard pressure not defined above 100 km altitude.')

    # Water Vapor Pressure/Density
    h0 = 2  # Scale height [km]
    rho0 = 7.5  # Standard ground-level density [g/m^3]
    rho = rho0 * np.exp(-h / h0)  # [g/m^3]

    e = rho * temp / 216.67  # [hPa]

    # Adjust water vapor pressure for constant mixing above a certain altitude
    mixing_ratio = np.maximum(2.0e-6, e / pressure)  # See text following equation 8
    e = pressure * mixing_ratio

    # Wrap it in a struct
    return Atmosphere(alt=alt, alt_geopotential=h, press=pressure, temp=temp, water_vapor_press=e, water_vapor_dens=rho,
                      rainfall=0, cloud_dens=0)


def get_spectroscopic_table_oxygen():
    """
    Returns the spectroscopic table for oxygen, according to ITU-R P.676-11
    "Attenuation by atmospheric gases (09/2016)"

    Ported from MATLAB Code

    Nicholas O'Donoughue
    16 March 2021

    :return ref_tbl: numpy 2D array containing the freq [MHz] index and six reference variables as an (num_rows x 7)
    """

    fox = np.array([50.474214, 50.987745, 51.50336, 52.021429, 52.542418, 53.066934, 53.595775, 54.130025, 54.67118,
                    55.221384, 55.783815, 56.264774, 56.363399, 56.968211, 57.612486, 58.323877, 58.446588, 59.164204,
                    59.590983, 60.306056, 60.434778, 61.150562, 61.800158, 62.41122, 62.486253, 62.997984, 63.568526,
                    64.127775, 64.67891, 65.224078, 65.764779, 66.302096, 66.836834, 67.369601, 67.900868, 68.431006,
                    68.960312, 118.750334, 368.498246, 424.76302, 487.249273, 715.392902, 773.83949, 834.145546])
    a1 = np.array([0.975, 2.529, 6.193, 14.32, 31.24, 64.29, 124.6, 227.3, 389.7, 627.1, 945.3, 543.4, 1331.8, 1746.6,
                   2120.1, 2363.7, 1442.1, 2379.9, 2090.7, 2103.4, 2438, 2479.5, 2275.9, 1915.4, 1503, 1490.2, 1078,
                   728.7, 461.3, 274, 153, 80.4, 39.8, 18.56, 8.172, 3.397, 1.334, 940.3, 67.4, 637.7, 237.4, 98.1,
                   572.3, 183.1])
    a2 = np.array([9.651, 8.653, 7.709, 6.819, 5.983, 5.201, 4.474, 3.8, 3.182, 2.618, 2.109, 0.014, 1.654, 1.255,
                   0.91, 0.621, 0.083, 0.387, 0.207, 0.207, 0.386, 0.621, 0.91, 1.255, 0.083, 1.654, 2.108, 2.617,
                   3.181, 3.8, 4.473, 5.2, 5.982, 6.818, 7.708, 8.652, 9.65, 0.01, 0.048, 0.044, 0.049, 0.145, 0.141,
                   0.145])
    a3 = np.array([6.69, 7.17, 7.64, 8.11, 8.58, 9.06, 9.55, 9.96, 10.37, 10.89, 11.34, 17.03, 11.89, 12.23, 12.62,
                   12.95, 14.91, 13.53, 14.08, 14.15, 13.39, 12.92, 12.63, 12.17, 15.13, 11.74, 11.34, 10.88, 10.38,
                   9.96, 9.55, 9.06, 8.58, 8.11, 7.64, 7.17, 6.69, 16.64, 16.4, 16.4, 16, 16, 16.2, 14.7])
    a4 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    a5 = np.array([2.566, 2.246, 1.947, 1.667, 1.388, 1.349, 2.227, 3.17, 3.558, 2.56, -1.172, 3.525, -2.378, -3.545,
                   -5.416, -1.932, 6.768, -6.561, 6.957, -6.395, 6.342, 1.014, 5.014, 3.029, -4.499, 1.856, 0.658,
                   -3.036, -3.968, -3.528, -2.548, -1.66, -1.68, -1.956, -2.216, -2.492, -2.773, -0.439, 0, 0, 0, 0,
                   0, 0])
    a6 = np.array([6.85, 6.8, 6.729, 6.64, 6.526, 6.206, 5.085, 3.75, 2.654, 2.952, 6.135, -0.978, 6.547, 6.451, 6.056,
                   0.436, -1.273, 2.309, -0.776, 0.699, -2.825, -0.584, -6.619, -6.759, 0.844, -6.675, -6.139, -2.895,
                   -2.59, -3.68, -5.002, -6.091, -6.393, -6.475, -6.545, -6.6, -6.65, 0.079, 0, 0, 0, 0, 0, 0])

    return np.stack((fox, a1, a2, a3, a4, a5, a6)).T


def get_spectroscopic_table_water():
    """
    Returns the spectroscopic table for water, according to ITU-R P.676-11
    "Attenuation by atmospheric gases (09/2016)"

    Ported from MATLAB Code

    Nicholas O'Donoughue
    16 March 2021

    :return ref_tbl: numpy 2D array containing the freq [MHz] index and six reference variables as an (num_rows x 7)
    """

    fw = np.array([22.23508, 67.80396, 119.99594, 183.310087, 321.22563, 325.152888, 336.227764, 380.197353, 390.134508,
                   437.346667, 439.150807, 443.018343, 448.001085, 470.888999, 474.689092, 488.490108, 503.568532,
                   504.482692, 547.67644, 552.02096, 556.935985, 620.700807, 645.766085, 658.00528, 752.033113,
                   841.051732, 859.965698, 899.303175, 902.611085, 906.205957, 916.171582, 923.112692, 970.315022,
                   987.926764, 1780])
    b1 = np.array([0.1079, 0.0011, 0.0007, 2.273, 0.047, 1.514, 0.001, 11.67, 0.0045, 0.0632, 0.9098, 0.192, 10.41,
                   0.3254, 1.26, 0.2529, 0.0372, 0.0124, 0.9785, 0.184, 497, 5.015, 0.0067, 0.2732, 243.4, 0.0134,
                   0.1325, 0.0547, 0.0386, 0.1836, 8.4, 0.0079, 9.009, 134.6, 17506])
    b2 = np.array([2.144, 8.732, 8.353, 0.668, 6.179, 1.541, 9.825, 1.048, 7.347, 5.048, 3.595, 5.048, 1.405, 3.597,
                   2.379, 2.852, 6.731, 6.731, 0.158, 0.158, 0.159, 2.391, 8.633, 7.816, 0.396, 8.177, 8.055, 7.914,
                   8.429, 5.11, 1.441, 10.293, 1.919, 0.257, 0.952])
    b3 = np.array([26.38, 28.58, 29.48, 29.06, 24.04, 28.23, 26.93, 28.11, 21.52, 18.45, 20.07, 15.55, 25.64, 21.34,
                   23.2, 25.86, 16.12, 16.12, 26, 26, 30.86, 24.38, 18, 32.1, 30.86, 15.9, 30.6, 29.85, 28.65, 24.08,
                   26.73, 29, 25.5, 29.85, 196.3])
    b4 = np.array([0.76, 0.69, 0.7, 0.77, 0.67, 0.64, 0.69, 0.54, 0.63, 0.6, 0.63, 0.6, 0.66, 0.66, 0.65, 0.69, 0.61,
                   0.61, 0.7, 0.7, 0.69, 0.71, 0.6, 0.69, 0.68, 0.33, 0.68, 0.68, 0.7, 0.7, 0.7, 0.7, 0.64, 0.68, 2])
    b5 = np.array([5.087, 4.93, 4.78, 5.022, 4.398, 4.893, 4.74, 5.063, 4.81, 4.23, 4.483, 5.083, 5.028, 4.506, 4.804,
                   5.201, 3.98, 4.01, 4.5, 4.5, 4.552, 4.856, 4, 4.14, 4.352, 5.76, 4.09, 4.53, 5.1, 4.7, 5.15, 5,
                   4.94, 4.55, 24.15])
    b6 = np.array([1, 0.82, 0.79, 0.85, 0.54, 0.74, 0.61, 0.89, 0.55, 0.48, 0.52, 0.5, 0.67, 0.65, 0.64, 0.72, 0.43,
                   0.45, 1, 1, 1, 0.68, 0.5, 1, 0.84, 0.45, 0.84, 0.9, 0.95, 0.53, 0.78, 0.8, 0.67, 0.9, 5])

    return np.stack((fw, b1, b2, b3, b4, b5, b6)).T


def get_spectral_lines(min_freq_hz=None, max_freq_hz=None):
    """
    Returns the frequencies for the spectral absorption lines of oxygen and water( in Hz).

    Ported from MATLAB Code

    Nicholas O'Donoughue
    16 March 2021

    :param min_freq_hz: (Optional) minimum frequency to return
    :param max_freq_hz: (Optional) maximum frequency to return
    :return f_ctr_o: numpy array of oxygen spectral lines
    :return f_ctr_w: numpy array of water spectral lines
    """

    f_ctr_o = np.array([50.474214, 50.987745, 51.50336, 52.021429, 52.542418, 53.066934, 53.595775, 54.130025, 54.67118,
                        55.221384, 55.783815, 56.264774, 56.363399, 56.968211, 57.612486, 58.323877, 58.446588,
                        59.164204, 59.590983, 60.306056, 60.434778, 61.150562, 61.800158, 62.41122, 62.486253,
                        62.997984, 63.568526, 64.127775, 64.67891, 65.224078, 65.764779, 66.302096, 66.836834,
                        67.369601, 67.900868, 68.431006, 68.960312, 118.750334, 368.498246, 424.76302, 487.249273,
                        715.392902, 773.83949, 834.145546]) * 1.0e9
    f_ctr_w = np.array([22.23508, 67.80396, 119.99594, 183.310087, 321.22563, 325.152888, 336.227764, 380.197353,
                        390.134508, 437.346667, 439.150807, 443.018343, 448.001085, 470.888999, 474.689092, 488.490108,
                        503.568532, 504.482692, 547.67644, 552.02096, 556.935985, 620.700807, 645.766085, 658.00528,
                        752.033113, 841.051732, 859.965698, 899.303175, 902.611085, 906.205957, 916.171582, 923.112692,
                        970.315022, 987.926764, 1780]) * 1.0e9

    if min_freq_hz is not None:
        f_ctr_o = f_ctr_o[f_ctr_o >= min_freq_hz]
        f_ctr_w = f_ctr_w[f_ctr_w >= min_freq_hz]

    if max_freq_hz is not None:
        f_ctr_o = f_ctr_o[f_ctr_o <= max_freq_hz]
        f_ctr_w = f_ctr_w[f_ctr_w <= max_freq_hz]

    return f_ctr_o, f_ctr_w
