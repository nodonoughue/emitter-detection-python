import numpy as np
from numpy import typing as npt
import warnings

from . import broadcast_backwards
from .constants import radius_earth_eff, radius_earth_true, speed_of_light


def calc_range(x1: npt.ArrayLike, x2: npt.ArrayLike)-> npt.NDArray:
    """
    Computes the range between two N-dimensional position vectors, using
    the Euclidean (L2) norm.

    Any dimensions beyond the second must be broadcastable between x1 and x2.
    Ported from MATLAB Code.

    Nicholas O'Donoughue
    16 January 2021

    :param x1: (N, M1, *out_shp) matrix of N-dimensional positions
    :param x2: (N, M2, *out_shp) matrix of N-dimensional positions
    :return r: M1xM2 matrix of pair-wise ranges
    """
    arrs, in_shp = broadcast_backwards([x1, x2], start_dim=2)
    x1, x2 = arrs

    # Find the input dimensions and test for compatibility
    shp = np.shape(x1)
    num_dim_1 = shp[0] if len(shp) > 0 else 1
    num_x1 = shp[1] if len(shp) > 1 else 1
    shp = np.shape(x2)
    num_dim_2 = shp[0] if len(shp) > 0 else 1
    num_x2 = shp[1] if len(shp) > 1 else 1
    if num_dim_1 != num_dim_2:
        raise TypeError('First dimension of both inputs must match.')

    out_shp = [num_x1, num_x2]
    out_shp.extend(in_shp)
    # Check for scalar output
    if np.prod(out_shp)==1:
        out_shp = []

    # Manually extend the dimensions
    x1 = x1[:, :, np.newaxis]
    # while len(x1.shape) < len(out_shp)+1: x1 = np.expand_dims(x1, axis=-1)
    x2 = x2[:, np.newaxis, :]
    # while len(x2.shape) < len(out_shp)+1: x2 = np.expand_dims(x2, axis=-1)

    # Compute the Euclidean Norm
    return np.reshape(np.linalg.norm(x1-x2, axis=0), shape=out_shp)


def calc_range_diff(x0: npt.ArrayLike, x1: npt.ArrayLike, x2: npt.ArrayLike)-> npt.NDArray:
    """
    Computes the difference in range from the reference input (x0) to each
    of the input vectors x1 and x2.  Difference is taken as the range from
    x0 to each column in x2 minus the range from x0 to each column in x1,
    pair-wise.

    The first dimension of all three inputs must match.  The dimensions of
    x1 and x2 determine the dimensions of the output dr. Any axes beyond the
    second must be broadcastable between x1 and x2.

    Ported from MATLAB Code.

    Nicholas O'Donoughue
    16 January 2021

    :param x0: (N, 1, *out_shp) reference position
    :param x1: (N, M1, *out_shp) vector of test positions
    :param x2: (N, M2, *out_shp) vector of test positions
    :return dr: (M1, M2, *out_shp) matrix of range differences
    """
    # Make sure all three inputs can broadcast together
    arrs, in_shp = broadcast_backwards([x0, x1, x2], start_dim=2)
    x0, x1, x2 = arrs

    out_shp = [x1.shape[1], x2.shape[1]]
    out_shp.extend(in_shp)
    if np.prod(out_shp)==1:
        out_shp = []

    # Compute the range from the reference position to each of the set of
    # test positions
    r1 = calc_range(x0, x1)  # (1, num_x1, *out_shp)
    r2 = calc_range(x0, x2)  # (1, num_x2, *out_shp)

    # Take the difference, with appropriate dimension reshaping
    if not out_shp:
        # M1 and M2 are both one; we expect both r1 and r2 to be single-element arrays, just take the difference
        # directly
        pass
    else:
        # Let's reshape r2 and r1 appropriately
        r2 = np.reshape(r2, shape=(1, x2.shape[1], *in_shp))
        r1 = np.reshape(r1, shape=(x1.shape[1], 1, *in_shp))

    return np.reshape(r2  - r1, shape=out_shp)


def calc_doppler(x_source: npt.ArrayLike, v_source: npt.ArrayLike, x_sensor: npt.ArrayLike, v_sensor: npt.ArrayLike,
                 f: float)-> npt.NDArray:
    """
    Given source and sensor at position x1 and x2 with velocity v1 and v2,
    compute the Doppler velocity shift

    Ported from MATLAB Code

    Nicholas O'Donoughue
    16 January 2021

    :param x_source: Position vector of num_sources sources (num_dims x num_sources x *out_shp), in m
    :param v_source: Velocity vector of num_sources sources (num_dims x num_sources x *out_shp), in m/s
    :param x_sensor: Position vector of num_sensors sensors (num_dims x num_sensors x *out_shp), in m
    :param v_sensor: Velocity vector of num_sensors sensors (num_dims x num_sensors x *out_shp), in m/s
    :param f: Carrier frequency, in Hertz
    :return fd: Doppler shifts for each source, sensor pair (num_sources x num_sensors x *out_shp), in Hertz
    """

    src_arrs, _ = broadcast_backwards([x_source, v_source], start_dim=0)
    x_source, v_source = src_arrs

    sensor_arrs, _ = broadcast_backwards([x_sensor, v_sensor], start_dim=0)
    x_sensor, v_sensor = sensor_arrs

    arrs, out_shp2 = broadcast_backwards([x_source, v_source, x_sensor, v_sensor], start_dim=2)
    x_source, v_source, x_sensor, v_sensor = arrs

    # # Make sure all the inputs are at least 2d
    # if len(np.shape(x_source)) < 2: x_source = x_source[:, np.newaxis]
    # if len(np.shape(v_source)) < 2: v_source = v_source[:, np.newaxis]
    # if len(np.shape(x_sensor)) < 2: x_sensor = x_sensor[:, np.newaxis]
    # if len(np.shape(v_sensor)) < 2: v_sensor = v_sensor[:, np.newaxis]
    #
    # # Check for broadcastability
    # if ((not is_broadcastable(x_source, v_source)) or (not is_broadcastable(x_sensor, v_sensor)) or
    #         (not is_broadcastable(x_source, x_sensor))):
    #     raise TypeError('Input dimensions do not match.')

    # Parse input dimensions
    shp = np.shape(x_source)
    num_sources = shp[1] if len(shp) > 1 else 1
    shp = np.shape(x_sensor)
    num_sensors = shp[1] if len(shp) > 1 else 1

    # in_shp = np.broadcast_shapes(out_shp_1, out_shp_2, out_shp_3, out_shp_4)
    out_shp = [num_sources, num_sensors]
    out_shp.extend(out_shp2)
    if np.prod(out_shp)==1:
        out_shp = []

    # Add a new dimension to separate num_sources/num_sensors
    x_source = x_source[:, :, np.newaxis]
    v_source = v_source[:, :, np.newaxis]
    x_sensor = x_sensor[:, np.newaxis, :]
    v_sensor = v_sensor[:, np.newaxis, :]

    # Unit vector from x1 to x2
    # Note: Using the np.newaxis ensures that dist has the same shape as dx.
    dx = x_sensor - x_source
    dist = np.linalg.norm(dx, axis=0, keepdims=True)
    u12 = np.divide(dx, dist, out=np.zeros_like(dx), where=dist!=0)
    u21 = -u12

    # x1 velocity towards x2
    vv1 = np.sum(v_source * u12, axis=0, keepdims=False) # use false to remove the spatial axis
    vv2 = np.sum(v_sensor * u21, axis=0, keepdims=False) # use false to remove the spatial axis

    # Sum of combined velocity
    v = vv1 + vv2

    # Convert to Doppler
    c = speed_of_light
    return np.reshape(f * v/c, shape=out_shp)  # remove the 1+; it's doppler, not absolute frequency


def calc_doppler_diff(x_source: npt.ArrayLike, v_source: npt.ArrayLike, x_ref:npt.ArrayLike, v_ref: npt.ArrayLike,
                      x_test: npt.ArrayLike, v_test: npt.ArrayLike, f: float)-> npt.NDArray:
    """
    Computes the difference in Doppler shift between reference and test
    sensor pairs and a source.  The two sets of sensor positions (x1 and x2)
    must have the same size.  Corresponding pairs will be compared.

    Ported from MATLAB Code.

    Nicholas O'Donoughue
    16 January 2021

    :param x_source: Position vector for N sources (nDim x N), in m
    :param v_source: Velocity vector for N sources (nDim x N), in m/s
    :param x_ref: Position vector for M reference sensors (nDim x M), in m
    :param v_ref: Velocity vector for M reference sensors (nDim x M), in m/s
    :param x_test: Position vector for M test sensors (nDim x M), in m
    :param v_test: Velocity vector for M test sensors (nDim x M), in m/s
    :param f: Carrier frequency, in Hertz
    :return fd_diff: Differential Doppler shift (N x M), in Hertz
    """

    test_arrs, _ = broadcast_backwards([x_test, v_test], start_dim=0)
    x_test, v_test = test_arrs

    ref_arrs, _ = broadcast_backwards([x_ref, v_ref], start_dim=0)
    x_ref, v_ref = ref_arrs

    source_arrs, _ = broadcast_backwards([x_source, v_source], start_dim=0)
    x_source, v_source = source_arrs

    arrs, out_shp2 = broadcast_backwards([x_source, v_source, x_test, v_test, x_ref, v_ref], start_dim=2)
    x_source, v_source, x_test, v_test, x_ref, v_ref = arrs

    # Make sure all the inputs are at least 2d
    # if len(np.shape(x_source)) < 2: x_source = x_source[:, np.newaxis]
    # if len(np.shape(v_source)) < 2: v_source = v_source[:, np.newaxis]
    # if len(np.shape(x_ref)) < 2: x_ref = x_ref[:, np.newaxis]
    # if len(np.shape(v_ref)) < 2: v_ref = v_ref[:, np.newaxis]
    # if len(np.shape(x_test)) < 2: x_test = x_test[:, np.newaxis]
    # if len(np.shape(v_test)) < 2: v_test = v_test[:, np.newaxis]

    # Check for broadcastability
    # if ((not is_broadcastable(x_source, v_source)) or (not is_broadcastable(x_ref, v_ref))
    #         or (not is_broadcastable(x_test, v_test)) or (not is_broadcastable(x_ref, x_test))
    #         or (not is_broadcastable(x_source, x_ref))):
    #     raise TypeError('Input dimensions do not match.')

    # Compute Doppler velocity from reference to each set of test positions
    dop_ref = calc_doppler(x_source, v_source, x_ref, v_ref, f)  # N x M
    dop_test = calc_doppler(x_source, v_source, x_test, v_test, f)  # N x M

    # Doppler difference
    return dop_test - dop_ref


def compute_slant_range(alt1: npt.ArrayLike, alt2: npt.ArrayLike, el_angle_deg: npt.ArrayLike,
                        use_effective_earth: bool=False)-> npt.NDArray:
    """
    Computes the slant range between two points given the altitude above the
    Earth, and the elevation angle relative to ground plane horizontal.

     R = sqrt((r1*sin(th))^2 + r2^2 - r1^2) - r1*sin(th)

    where
       r1 = alt1 + radius_earth
       r2 = alt2 + radius_earth
       th = elevation angle (above horizon) in degrees at point 1

    If the fourth argument is specified and set to true, then radius_earth is the 4/3 Earth Radius used for
    RF propagation paths otherwise the true earth radius is used.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    16 January 2021

    :param alt1: Altitude at the start of the path, in meters
    :param alt2: Altitude at the end of the path, in meters
    :param el_angle_deg: Elevation angle, at the start of the path, in degrees above the local horizontal plane
    :param use_effective_earth: Binary flag [Default=False], specifying whether to use the 4/3 Earth Radius
                                approximation common to RF propagation
    :return: Straight line slant range between the start and end point specified, in meters
    """

    # Parse earth radius setting
    if use_effective_earth:
        # True radius
        radius_earth = radius_earth_true
    else:
        # 4/3 approximation -- to account for refraction
        radius_earth = radius_earth_eff

    # Compute the two radii
    r1 = radius_earth + alt1
    r2 = radius_earth + alt2
    r1c = r1 * np.sin(np.deg2rad(el_angle_deg))

    # Compute the slant range
    return np.sqrt(r1c**2 + r2**2 - r1**2) - r1c


def find_intersect(x0: npt.NDArray[np.float64], psi0: float,
                   x1: npt.NDArray[np.float64], psi1: float)-> npt.NDArray[np.float64]:
    """
    # Find the intersection of two lines of bearing with origins at x0 and x1, and bearing given by psi0 and psi1.

    Ported from MATLAB Code

    Nicholas O'Donoughue
    16 January 2021

    :param x0: 2-D position of the first sensor
    :param psi0: Bearing of the first line, which begins at x0
    :param x1: 2-D position of the second sensor
    :param psi1: Bearing of the second line, which begins at x0
    :return x_int: 2-D position of the intersection point
    """

    # Find slope and intercept for each line of bearing
    m0 = np.sin(psi0)/np.cos(psi0)
    m1 = np.sin(psi1)/np.cos(psi1)

    b0 = x0[1] - m0*x0[0]
    b1 = x1[1] - m1*x1[0]

    if (np.isinf(m0) and np.isinf(m1)) or np.fabs(m0-m1) < 1e-12:
        # They're parallel, effectively
        warnings.warn('Slopes are almost parallel; the solution is ill-conditioned.')
        return x0

    # Check Boundary Cases
    x = np.zeros(shape=(2,), dtype=float)
    if np.abs(np.cos(psi0)) < 1e-10:
        # First LOB is parallel to y-axis; x is fixed
        x[0] = x0[0]

        # Use slope/intercept definition of second LOB to solve for y
        x[1] = m1 * x[0] + b1
    elif np.abs(np.cos(psi1)) < 1e-10:
        # Same issue, but with the second LOB being parallel to y-axis
        x[0] = x1[0]
        x[1] = m0 * x[0] + b0
    else:
        # Find the point of intersection
        x[0] = (b0-b1)/(m1-m0)
        x[1] = m1*x[0] + b1

    return x
