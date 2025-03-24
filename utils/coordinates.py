"""
Define a set of coordinate transforms to convert between several local and global
coordinate systems, including:

- ENU (East, North, Up)
- AER (Azimuth, Elevation, Range)
- LLA (Latitude, Longitude, Altitude)
- ECEF (Earth-Centered, Earth-Fixed)
"""
from . import unit_conversions
from . import constants
import numpy as np


def aer_to_ecef(az, el, rng, lat_ref, lon_ref, alt_ref, angle_units='deg', dist_units='m'):
    """
    Convert from local AER (azimuth, elevation, range) to ECEF (Earth-Centered, Earth-Fixed),
    where azimuth in degrees East from North (as opposed to the typical degrees +y from +x)
    and elevation is degrees above the horizontal.

    Nicholas O'Donoughue
    7 February 2025

    :param az: azimuth
    :param el: elevation
    :param rng: range
    :param lat_ref: latitude reference
    :param lon_ref: longitude reference
    :param alt_ref: altitude reference
    :param angle_units: 'deg' or 'rad'; for a, e, lat_ref, and lon_ref
    :param dist_units:
    :return x: ECEF x-coordinate, in meters
    :return y: ECEF y-coordinate, in meters
    :return z: ECEF z-coordinate, in meters
    """

    # Convert to ENU -- response expressed as distance in 'dist_units'
    east, north, up = aer_to_enu(az, el, rng, angle_units)

    # Convert from ENU to ECEF -- response expressed as distance in 'dist_units'
    x, y, z = enu_to_ecef(east, north, up, lat_ref, lon_ref, alt_ref, angle_units, dist_units)

    return x, y, z


def aer_to_enu(az, el, rng, angle_units='deg'):
    """
    Convert spherical AER coordinates to cartesian ENU, where azimuth is measured in degrees East from North (as
    opposed to the typical degrees +y from +x) and elevation is degrees above the horizon.

    Nicholas O'Donoughue
    7 February 2025

    :param az: azimuth
    :param el: elevation
    :param rng: range
    :param angle_units: 'deg' or 'rad'; for az, el, and outputs
    :return east: East
    :return north: North
    :return up: Up
    """

    # Convert input angles to radians
    az_rad = unit_conversions.convert(az, from_unit=angle_units, to_unit='rad')
    el_rad = unit_conversions.convert(el, from_unit=angle_units, to_unit='rad')

    # Compute Ground Range
    ground_range = rng * np.cos(el_rad)

    # Compute ENU
    east = ground_range * np.sin(az_rad)
    north = ground_range * np.cos(az_rad)
    up = rng * np.sin(el_rad)

    return east, north, up


def aer_to_lla(az, el, rng, lat_ref, lon_ref, alt_ref, angle_units='deg', dist_units='m'):
    """
    Convert spherical AER coordinates to global LLA (lat/lon/alt), where azimuth is measured in degrees East from North
    (as opposed to the typical degrees +y from +x) and elevation is degrees above the horizon. The AER coordinates are
    expressed relative to a global coordinate at lat_ref, lon_ref, alt_ref.

    Nicholas O'Donoughue
    14 February 2025

    :param az: azimuth
    :param el: elevation
    :param rng: range
    :param lat_ref: latitude reference
    :param lon_ref: longitude reference
    :param alt_ref: altitude reference
    :param angle_units: 'deg' or 'rad'; for a, e, lat_ref, lon_ref, and output lat/lon
    :param dist_units: distance units for range, alt_ref, and output alt
    :return lat: latitude
    :return lon: longitude
    :return alt: altitude
    """

    # Convert AER to ECEF -- response
    x, y, z = aer_to_ecef(az, el, rng, lat_ref, lon_ref, alt_ref, angle_units, dist_units)

    # Convert ECEF to LLA -- response expressed as angles in angle_units and distance in dist_units
    lat, lon, alt = ecef_to_lla(x, y, z, angle_units, dist_units)

    return lat, lon, alt


def ecef_to_aer(x, y, z, lat_ref, lon_ref, alt_ref, angle_units='deg', dist_units='m'):
    """
    Convert az set of ECEF coordinates to AER (azimuth, elevation, range),
    relative to az reference Lat/Lon point.  ECEF inputs must be
    broadcastable to az common size (all non-singleton dimensions must match).

    The reference lat/lon must be scalar.

    The optional inputs angle_units and dist_units are used to specify the
    units for lat/lon (either radians or degrees), and alt (any valid length
    unit).

    Nicholas O'Donoughue
    14 Feb 2025

    :param x: ECEF x-coordinate
    :param y: ECEF y-coordinate
    :param z: ECEF z-coordinate
    :param lat_ref: latitude reference
    :param lon_ref: longitude reference
    :param alt_ref: altitude reference
    :param angle_units: 'deg' or 'rad'; for lat_ref, lon_ref and outputs az, and e
    :param dist_units: distance units for input x, y, z, alt_ref, and output range
    :return az: azimuth
    :return el: elevation
    :return rng: range
    """

    # Convert ECEF to ENU, using lat/lon/alt ref point -- response expressed as distance in dist_units
    east, north, up = ecef_to_enu(x, y, z, lat_ref, lon_ref, alt_ref, angle_units, dist_units)

    # Convert ENU to AER
    az, el, rng = enu_to_aer(east, north, up, angle_units)

    return az, el, rng


def ecef_to_enu(x, y, z, lat_ref, lon_ref, alt_ref, angle_units='deg', dist_units='m'):
    """
    Convert a set of ECEF coordinates to ENU (east, north, up),
    relative to a reference Lat/Lon point.  ECEF inputs must be
    broadcastable to a common size (all non-singleton dimensions must match).

    The reference lat/lon must be scalar.

    The optional inputs angle_units and dist_units are used to specify the
    units for lat/lon (either radians or degrees), and alt (any valid length
    unit).

    Nicholas O'Donoughue
    14 Feb 2025

    :param x: Vector or matrix of ECEF x-coordinates
    :param y: Vector or matrix of ECEF y-coordinates
    :param z: Vector or matrix of ECEF z-coordinates
    :param lat_ref: Reference latitude
    :param lon_ref: Reference longitude
    :param alt_ref: Reference altitude
    :param angle_units: Units for lat and lon [Default='deg']
    :param dist_units: Units for ECEF coordinates, alt_ref, and outputs [Default='m']
    :return east: east
    :return north: north
    :return up: up
    """
    # Convert the reference point to ECEF
    [x_ref, y_ref, z_ref] = lla_to_ecef(lat_ref, lon_ref, alt_ref, angle_units, dist_units)

    # Take the difference
    dx = x - x_ref
    dy = y - y_ref
    dz = z - z_ref

    # Rotate the position vector using the lat/lon reference point
    # This code is the same as that for converting a velocity vector from ECEF to ENU, so just call that function
    east, north, up = ecef_to_enu_vel(dx, dy, dz, lat_ref, lon_ref, angle_units)

    # Compute some sin/cos terms; we need lat_ref/lon_ref in radians first
    # lat_ref_rad = unit_conversions.convert(lat_ref, from_unit=angle_units, to_unit='rad')
    # lon_ref_rad = unit_conversions.convert(lon_ref, from_unit=angle_units, to_unit='rad')
    #
    # cos_lat = np.cos(lat_ref_rad)
    # cos_lon = np.cos(lon_ref_rad)
    # sin_lat = np.sin(lat_ref_rad)
    # sin_lon = np.sin(lon_ref_rad)
    #
    # # Compensation term (t)
    # t = (cos_lon * dx) + (sin_lon * dy)
    #
    # # Compute east, north, up
    # east = -(sin_lon * dx) + (cos_lon * dy)
    # north = -(sin_lat * t) + (cos_lat * dz)
    # up = (cos_lat * t) + (sin_lat * dz)

    return east, north, up


def ecef_to_enu_vel(vel_x, vel_y, vel_z, lat_ref, lon_ref, angle_units='deg'):
    """
    Convert a set of ECEF velocities to ENU, relative to a reference lat/lon point. ECEF inputs must be broadcastable
    to a common size.

    Similar to ecef_to_enu, except that the origin is not translated; only rotation occurs, since the velocity vector
    is not a pointing vector from the ENU origin to the target, rather it is a pointing vector for how the target is
    moving.

    Nicholas O'Donoughue
    14 Feb 2025

    :param vel_x: Vector or matrix of ECEF velocity x-components
    :param vel_y: Vector or matrix of ECEF velocity y-components
    :param vel_z: Vector or matrix of ECEF velocity z-components
    :param lat_ref: Reference latitude
    :param lon_ref: Reference longitude
    :param angle_units: Units for lat and lon [Default='deg']
    :return vel_e: East component of velocity (m)
    :return vel_n: North component of velocity(m)
    :return vel_u: Up component of velocity (m)

    """

    # Compute some sin/cos terms; we need lat_ref/lon_ref in radians first
    lat_ref_rad = unit_conversions.convert(lat_ref, from_unit=angle_units, to_unit='rad')
    lon_ref_rad = unit_conversions.convert(lon_ref, from_unit=angle_units, to_unit='rad')

    cos_lat = np.cos(lat_ref_rad)
    cos_lon = np.cos(lon_ref_rad)
    sin_lat = np.sin(lat_ref_rad)
    sin_lon = np.sin(lon_ref_rad)

    # Compensation term (t)
    t = (cos_lon * vel_x) + (sin_lon * vel_y)

    # Compute e, n, u
    vel_e = -(sin_lon * vel_x) + (cos_lon * vel_y)
    vel_n = -(sin_lat * t) + (cos_lat * vel_z)
    vel_u = (cos_lat * t) + (sin_lat * vel_z)

    return vel_e, vel_n, vel_u


def ecef_to_lla(x, y, z, angle_units, dist_units):
    """
    Conversion from cartesian ECEF to geodetic LLA coordinates, using a
    direct computation methods.  Note that iterative solutions also exist,
    and can be made to be more accurate (by iterating until a desired
    accuracy is achieved).  For the purposes of this package, we have
    implemented the faster direct computation method, understanding that
    its accuracy may be limited in certain edge cases.

    Source: https://microem.ru/files/2012/08/GPS.G1-X-00006.pdf

    Nicholas O'Donoughue
    14 Feb 2025

    :param x: ECEF x-coordinates
    :param y: ECEF y-coordinates
    :param z: ECEF z-coordinates
    :param angle_units: Units for lat and lon [Default='deg']
    :param dist_units: Units for ECEF coordinates and output altitude [Default='m']
    :return lat: Latitude
    :return lon: Longitude
    :return alt: Altitude
    """

    # Compute the longitude as the arc-tan of y / x
    lon_rad = np.atan2(y, x)

    # Read in constants
    a = constants.semimajor_axis_km * 1e3
    b = constants.semiminor_axis_km * 1e3
    e1_sq = constants.first_ecc_sq
    e2_sq = constants.second_ecc_sq

    # Compute Auxiliary Values
    p = np.sqrt(x**2 + y**2)
    th = np.atan2(z * a, p * b)

    # Compute Latitude estimate
    lat_rad = np.atan2(z + e2_sq * b * np.sin(th)**3, p - e1_sq * a * np.cos(th)**3)

    # Compute Height
    n = a / np.sqrt(1 - e1_sq * np.sin(lat_rad)**2)
    alt_m = p / np.cos(lat_rad) - n

    # Format Outputs
    lat = unit_conversions.convert(lat_rad, from_unit="rad", to_unit=angle_units)
    lon = unit_conversions.convert(lon_rad, from_unit="rad", to_unit=angle_units)
    alt = unit_conversions.convert(alt_m, from_unit="m", to_unit=dist_units)

    return lat, lon, alt


def enu_to_aer(east, north, up, angle_units='deg'):
    """
    Convert cartesian ENU coordinates to spherical AER, where azimuth is
    in degrees East from North (as opposed to the typical degrees +y from +x)
    and elevation is degrees above the horizontal.

    Nicholas O'Donoughue
    14 Feb 2025

    :param east: east
    :param north: north
    :param up: up
    :param angle_units: Units for output azimuth and elevation [Default='deg']
    :return az: azimuth
    :return el: elevation
    :return rng: range (will be in the same units as el, n, u)
    """

    # Compute Ground and Slant (output) Ranges
    ground_range = np.sqrt(east ** 2 + north ** 2)
    rng = np.sqrt(ground_range ** 2 + up ** 2)  # slant range

    # Az/El Angles, in radians
    az_rad = np.atan2(east, north)
    el_rad = np.atan2(up, ground_range)

    # Convert output units
    az = unit_conversions.convert(az_rad, from_unit="rad", to_unit=angle_units)
    el = unit_conversions.convert(el_rad, from_unit="rad", to_unit=angle_units)

    return az, el, rng


def enu_to_ecef(east, north, up, lat_ref, lon_ref, alt_ref, angle_units='deg', dist_units='m'):
    """
    Convert local ENU coordinates to ECEF, given the reference LLA position
    for the local ENU coordinates.

    Nicholas O'Donoughue
    14 Feb 2025

    :param east: east
    :param north: north
    :param up: up
    :param lat_ref: Reference latitude
    :param lon_ref: Reference longitude
    :param alt_ref: Reference altitude
    :param angle_units: Units for input lat/lon [Default='deg']
    :param dist_units: Units for inputs e, n, u, alt_ref, and output ECEF coordinates [Default='m']
    :return x: ECEF x-coordinates
    :return y: ECEF y-coordinates
    :return z: ECEF z-coordinates
    """

    # TODO: Verify that converting an ENU velocity to ECEF and back to ENU is the same

    # Rotate the ENU vector to XYZ, assuming it starts at the origin
    dx, dy, dz = enu_to_ecef_vel(east, north, up, lat_ref, lon_ref, angle_units)
    # # Precompute Trig Functions of Reference Lat/Lon
    # lat_rad = unit_conversions.convert(lat_ref, from_unit=angle_units, to_unit='rad')
    # lon_rad = unit_conversions.convert(lon_ref, from_unit=angle_units, to_unit='rad')
    #
    # sin_lat = np.sin(lat_rad)
    # sin_lon = np.sin(lon_rad)
    # cos_lat = np.cos(lat_rad)
    # cos_lon = np.cos(lon_rad)
    #
    # # Convert from ENU to dx/dy/dz, using reference coordinates
    # t = -sin_lat*n + cos_lat*u
    #
    # dx = ((-sin_lon * e) + (cos_lon * t))
    # dy = ((cos_lon * e) + (sin_lon * t))
    # dz = ((cos_lat * n) + (sin_lat * u))

    # Convert Ref LLA to ECEF; output in 'dist_units'
    x_ref, y_ref, z_ref = lla_to_ecef(lat_ref, lon_ref, alt_ref, angle_units, dist_units)

    # Translate the rotated ECEF velocity vector so that its origin is at ENU
    x = x_ref + dx
    y = y_ref + dy
    z = z_ref + dz

    return x, y, z


def enu_to_ecef_vel(vel_e, vel_n, vel_u, lat_ref, lon_ref, angle_units='deg'):
    """
    Convert local ENU velocity vector to ECEF, given the reference LLA position
    for the local ENU coordinates.

    Similar to enu_to_ecef, except that this function does not translate to
    account for migration of the origin from lat0, lon0 to the center of the
    Earth, since the input vectors are taken as referenced from target
    position.  Instead, only the rotation operation is conducted.

    Nicholas O'Donoughue
    14 Feb 2025

    :param vel_e: east velocity
    :param vel_n: north velocity
    :param vel_u: up velocity
    :param lat_ref: Reference Latitude
    :param lon_ref: Reference Longitude
    :param angle_units: Units for input lat/lon [Default='deg']
    :return vel_x: ECEF x-velocity
    :return vel_y: ECEF y-velocity
    :return vel_z: ECEF z_velocity
    """

    # Precompute Trig Functions of Reference Lat/Lon
    lat_rad = unit_conversions.convert(lat_ref, from_unit=angle_units, to_unit='rad')
    lon_rad = unit_conversions.convert(lon_ref, from_unit=angle_units, to_unit='rad')

    sin_lat = np.sin(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lat = np.cos(lat_rad)
    cos_lon = np.cos(lon_rad)

    # Convert from ENU to dx/dy/dz, using reference coordinates
    t = -sin_lat * vel_n + cos_lat * vel_u

    vel_x = ((-sin_lon * vel_e) + (cos_lon * t))
    vel_y = ((cos_lon * vel_e) + (sin_lon * t))
    vel_z = ((cos_lat * vel_n) + (sin_lat * vel_u))

    return vel_x, vel_y, vel_z


def enu_to_lla(east, north, up, lat_ref, lon_ref, alt_ref, angle_units='deg', dist_units='m'):
    """
    Convert local ENU coordinates to LLA, given the reference LLA position
    for the local ENU coordinates.

    Nicholas O'Donoughue
    14 Feb 2025

    :param east: east
    :param north: north
    :param up: up
    :param lat_ref: Reference Latitude
    :param lon_ref: Reference Longitude
    :param alt_ref: Reference Altitude
    :param angle_units:
    :param dist_units:
    :return lat: Latitude
    :return lon: Longitude
    :return alt: Altitude
    """

    # Convert ENU to global ECEF
    x, y, z = enu_to_ecef(east, north, up, lat_ref, lon_ref, alt_ref, angle_units, dist_units)

    # Convert ECEF to LLA
    lat, lon, alt = ecef_to_lla(x, y, z, angle_units, dist_units)

    return lat, lon, alt


def lla_to_aer(lat, lon, alt, lat_ref, lon_ref, alt_ref, angle_units, dist_units):
    """
    Convert a set of Lat, Lon, Alt coordinates to AER (az, el, range), as
    seen by a reference sensor at lat0, lon0, alt0.  Any non-scalar LLA
    inputs must be broadcastable to a common shape.

    The optional inputs angle_units and dist_units are used to specify the
    units for lat/lon (either radians or degrees), and alt (any valid length
    unit).

    Nicholas O'Donoughue
    14 Feb 2025

    :param lat: latitude
    :param lon: longitude
    :param alt: altitude
    :param lat_ref: reference Latitude
    :param lon_ref: reference Longitude
    :param alt_ref: reference altitude
    :param angle_units: Units for input lat/lon [Default='deg']
    :param dist_units: Units for input alt, alt_ref, and output range
    :return az: Azimuth
    :return el: Elevation
    :return rng: Range
    """

    # Convert from LLA to ENU
    east, north, up = lla_to_enu(lat, lon, alt, lat_ref, lon_ref, alt_ref, angle_units, dist_units)

    # Convert from ENU to AER
    az, el, rng = enu_to_aer(east, north, up, angle_units)

    return az, el, rng


def lla_to_ecef(lat, lon, alt, angle_units='deg', dist_units='m'):
    """
    Convert a set of Lat, Lon, Alt coordinates to ECEF (x, y, z).  Lat,
    Lon, and Alt inputs must be broadcastable to a common size (all
    non-singleton dimensions must match).

    The optional inputs angle_units and dist_units are used to specify the
    units for lat/lon (either radians or degrees), and alt (any valid length
    unit).

    Nicholas O'Donoughue
    14 Feb 2025

    :param lat: latitude
    :param lon: longitude
    :param alt: altitude
    :param angle_units: Units for input lat/lon [Default='deg']
    :param dist_units: Units for input alt, and output ECEF coordinates [Default='m']
    :return x: ECEF x-coordinate
    :return y: ECEF y-coordinate
    :return z: ECEF z-coordinate
    """

    # Lookup Earth Constants
    semi_major_axis_m = constants.semimajor_axis_km*1e3
    ecc_sq = constants.first_ecc_sq

    # Make sure Lat/Lon are in rad, alt in meters
    lat_rad = unit_conversions.convert(lat, from_unit=angle_units, to_unit='rad')
    lon_rad = unit_conversions.convert(lon, from_unit=angle_units, to_unit='rad')
    alt_m = unit_conversions.convert(alt, from_unit=dist_units, to_unit='m')

    # Compute Sin and Cos of Lat/Lon Inputs
    sin_lat = np.sin(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lat = np.cos(lat_rad)
    cos_lon = np.cos(lon_rad)

    # Compute effective radius
    eff_rad = semi_major_axis_m / np.sqrt(1 - ecc_sq * sin_lat**2)

    # Compute ECEF Coordinates
    x = (eff_rad + alt_m) * cos_lat * cos_lon
    y = (eff_rad + alt_m) * cos_lat * sin_lon
    z = ((1 - ecc_sq) * eff_rad + alt_m) * sin_lat

    return x, y, z


def lla_to_enu(lat, lon, alt, lat_ref, lon_ref, alt_ref, angle_units, dist_units):
    """
    Convert a set of Lat, Lon, Alt coordinates to ENU (east, north, up),
    relative to a reference Lat/Lon point.  Lat, Lon, and Alt inputs must be
    broadcastable to a common size (all non-singleton dimensions must match).

    The optional inputs angle_units and dist_units are used to specify the
    units for lat/lon (either radians or degrees), and alt (any valid length
    unit).

    Nicholas O'Donoughue
    14 Feb 2025

    :param lat: latitude
    :param lon: longitude
    :param alt: altitude
    :param lat_ref: reference latitude
    :param lon_ref: reference longitude
    :param alt_ref: reference altitude
    :param angle_units: Units for input lat/lon and reference lat/lon [Default = 'deg']
    :param dist_units: Units for input alt and reference alt, and outputs [Default = 'm']
    :return east: east
    :return north: north
    :return up: up
    """

    # Convert input LLA to ECEF
    x, y, z = lla_to_ecef(lat, lon, alt, angle_units, dist_units)

    # Convert from ECEF to ENU
    east, north, up = ecef_to_enu(x, y, z, lat_ref, lon_ref, alt_ref, angle_units, dist_units)

    return east, north, up
