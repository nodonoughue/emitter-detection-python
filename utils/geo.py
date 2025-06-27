import warnings
import numpy as np
from . import constants
from . import utils


def calc_range(x1, x2):
    """
    Computes the range between two N-dimensional position vectors, using
    the Euclidean (L2) norm.

    Ported from MATLAB Code.

    Nicholas O'Donoughue
    16 January 2021

    :param x1: NxM1 matrix of N-dimensional positions
    :param x2: NxM2 matrix of N-dimensional positions
    :return r: M1xM2 matrix of pair-wise ranges
    """

    # Find the input dimensions and test for compatibility
    sz_1 = np.shape(x1)
    sz_2 = np.shape(x2)
    if sz_1[0] != sz_2[0]:
        raise TypeError('First dimension of both inputs must match.')

    num_dims = sz_1[0]
    num_x1 = int(np.prod(sz_1[1:]))
    num_x2 = int(np.prod(sz_2[1:]))
    # out_shp = (num_x1, num_x2)

    # Reshape the inputs
    x1 = np.reshape(x1, (num_dims, num_x1, 1))
    x2 = np.reshape(x2, (num_dims, 1, num_x2))

    # Compute the Euclidean Norm
    return np.squeeze(np.linalg.norm(x1-x2, axis=0))


def calc_range_diff(x0, x1, x2):
    """
    Computes the difference in range from the reference input (x0) to each
    of the input vectors x1 and x2.  Difference is taken as the range from
    x0 to each column in x2 minus the range from x0 to each column in x1,
    pair-wise.

    The first dimension of all three inputs must match.  The dimensions of
    x1 and x2 determine the dimensions of the output dr.

    Ported from MATLAB Code.

    Nicholas O'Donoughue
    16 January 2021

    :param x0: Nx1 reference position
    :param x1: NxM1 vector of test positions
    :param x2: NxM2 vector of test positions
    :return dr: M1xM2 matrix of range differences
    """

    # Compute the range from the reference position to each of the set of
    # test positions
    r1 = calc_range(x0, x1)  # 1xM1
    r2 = calc_range(x0, x2)  # 1xM2

    # Take the difference, with appropriate dimension reshaping
    return r1 - r2.T


def calc_doppler(x1, v1, x2, v2, f):
    """
    Given source and sensor at position x1 and x2 with velocity v1 and v2,
    compute the Doppler velocity shift

    Ported from MATLAB Code

    Nicholas O'Donoughue
    16 January 2021

    :param x1: Position vector of num_sources sources (num_dims x num_sources), in m
    :param v1: Velocity vector of num_sources sources (num_dims x num_sources), in m/s
    :param x2: Position vector of num_sensors sensors (num_dims x num_sensors), in m
    :param v2: Velocity vector of num_sensors sensors (num_dims x num_sensors), in m/s
    :param f: Carrier frequency, in Hertz
    :return fd: Doppler shifts for each source, sensor pair (num_sources x num_sensors), in Hertz
    """

    # Reshape inputs
    num_dims, num_sources = utils.safe_2d_shape(x1)
    _, num_sources2 = utils.safe_2d_shape(v1)
    num_dims2, num_sensors = utils.safe_2d_shape(x2)
    _, num_sensors2 = utils.safe_2d_shape(v2)

    if num_dims != num_dims2 or \
            (not utils.is_broadcastable(x1, v1)) or \
            (not utils.is_broadcastable(x2, v2)):
        raise TypeError('Input dimensions do not match.')

    x1 = np.reshape(x1.T, (num_sources, 1, num_dims))
    v1 = np.reshape(v1.T, (num_sources2, 1, num_dims))
    x2 = np.reshape(x2.T, (1, num_sensors, num_dims))
    v2 = np.reshape(v2.T, (1, num_sensors2, num_dims))

    # Unit vector from x1 to x2
    # Note: I'm not sure why, but just dividing dx/dist broadcasts the dimensions weirdly.  Using the np.newaxis ensures
    # that dist has the same shape as dx, to avoid the broadcasting bug.
    dx = x2-x1
    dist = np.linalg.norm(dx, axis=2)
    u12 = dx / dist[:, :, np.newaxis]
    u21 = -u12

    # x1 velocity towards x2
    vv1 = np.sum(v1*u12, axis=2)
    vv2 = np.sum(v2*u21, axis=2)

    # Sum of combined velocity
    v = vv1 + vv2

    # Convert to Doppler
    c = constants.speed_of_light
    return f * (1 + v/c)


def calc_doppler_diff(x_source, v_source, x_ref, v_ref, x_test, v_test, f):
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

    # Compute Doppler velocity from reference to each set of test positions
    dop_ref = calc_doppler(x_source, v_source, x_ref, v_ref, f)  # N x M
    dop_test = calc_doppler(x_source, v_source, x_test, v_test, f)  # N x M

    # Doppler difference
    return dop_test - dop_ref


def compute_slant_range(alt1, alt2, el_angle_deg, use_effective_earth=False):
    """
    Computes the slant range between two points given the altitude above the
    Earth, and the elevation angle relative to ground plane horizontal.

     R = sqrt((r1*sin(th))^2 + r2^2 - r1^2) - r1*sin(th)

    where
       r1 = alt1 + radius_earth
       r2 = alt2 + radius_earth
       th = elevation angle (above horizon) in degrees at point 1

    If the fourth argument is specified and set to true,
    then radius_earth is the 4/3 Earth Radius used for RF propagation paths
    otherwise the true earth radius is used.

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
        radius_earth = constants.radius_earth_true
    else:
        # 4/3 approximation -- to account for refraction
        radius_earth = constants.radius_earth_eff

    # Compute the two radii
    r1 = radius_earth + alt1
    r2 = radius_earth + alt2
    r1c = r1 * np.sin(np.deg2rad(el_angle_deg))

    # Compute the slant range
    return np.sqrt(r1c**2 + r2**2 - r1**2) - r1c


def find_intersect(x0, psi0, x1, psi1):
    """
    # Find the intersection of two lines of bearing with origins at x0 and x1,
    # and bearing given by psi0 and psi1.

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
