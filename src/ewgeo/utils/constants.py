"""
Define a set of universal constants that may be useful
"""

# Boltzmann's Constant
boltzmann = 1.3806e-23
        
# Reference Temperature (290 K), used for background noise level
ref_temp = 290.
kT = 4.00374e-21  # boltzmann * T0
        
# Speed of Light
speed_of_light = 299792458.
        
# Radius of the earth; with a (4/3) constant applied to account for
# electromagnetic propagation around the Earth
radius_earth_true = 6371000.
radius_earth_eff = radius_earth_true*4./3.

# Elliptical Earth parameters
semimajor_axis_km = 6378.137
semiminor_axis_km = 6356.752314245
first_ecc_sq = 6.69438002290e-3
second_ecc_sq = 6.73949677548e-3