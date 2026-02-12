import numpy as np

from ewgeo.utils import constraints, coordinates, constants


def test_fixed_alt():
    """
    Generate a fixed altitude constraint and test it.
    """
    alt_bound = 1000
    num_test_points = 1000
    alt_vec = alt_bound + 100 * np.random.rand(num_test_points)
    cost = alt_vec - alt_bound

    # Flat Earth
    geo_type = 'flat'
    a, a_grad = constraints.fixed_alt(alt_bound, geo_type)

    # Assumes local ENU-style coordinates.
    # Make random East/North vectors
    x_vec = np.array([np.random.randn(num_test_points),
                      np.random.randn(num_test_points),
                      alt_vec])
    eps_test, _ = a(x_vec)
    assert equal_to_tolerance(cost, eps_test)

    # Spherical Earth
    # The cost for spherical Earth (equation 5.5) is the difference of the magnitude squared of the input and the
    # square of the desired radius.
    geo_type = 'sphere'
    a, a_grad = constraints.fixed_alt(alt_bound, geo_type)

    # Inputs are ECEF-style, but for a spherical Earth.
    # Generate some random 3D data, then scale it to the desired altitude
    x_vec = np.random.randn(3, num_test_points)
    radius = constants.radius_earth_true + alt_vec
    cost_sphere = radius**2 - (constants.radius_earth_true + alt_bound)**2
    # Divide by the magnitude of each sample, then multiply by the desired radius for each test point
    x_vec = x_vec * radius[np.newaxis, :] / np.sqrt(np.sum(np.abs(x_vec)**2, axis=0, keepdims=True))
    eps_test, _ = a(x_vec)
    assert equal_to_tolerance(cost_sphere, eps_test, tol=1)

    # Elliptical Earth
    geo_type = 'ellipse'
    a, a_grad = constraints.fixed_alt(alt_bound, geo_type)

    # Inputs are ECEF-style, generate some random LLA and
    # convert
    lat_vec = np.random.rand(num_test_points) * 80
    lon_vec = np.random.rand(num_test_points) * 170
    xx, yy, zz = coordinates.lla_to_ecef(lat_vec, lon_vec, alt_vec)
    x_vec = np.array([xx, yy, zz])

    eps_test, _ = a(x_vec)
    assert equal_to_tolerance(cost, eps_test)

def test_fixed_cartesian():
    """
    Generate a fixed cartesian constraint and test it
    """
    bound_val = 1000
    num_test_points = 1000
    test_vec = np.random.rand(num_test_points) * 100 + bound_val
    cost = test_vec - bound_val

    # X-constraint
    bound_type = 'x'
    a, a_grad = constraints.fixed_cartesian(bound_type, bound_val)
    x_vec = np.array([test_vec,
                      np.random.randn(num_test_points),
                      np.random.randn(num_test_points)])
    eps_test, _ = a(x_vec)
    assert equal_to_tolerance(cost, eps_test)

    # Y-constraint
    bound_type = 'y'
    a, a_grad = constraints.fixed_cartesian(bound_type, bound_val)
    x_vec = np.array([np.random.randn(num_test_points),
                      test_vec,
                      np.random.randn(num_test_points)])
    eps_test, _ = a(x_vec)
    assert equal_to_tolerance(cost, eps_test)

    # Z-constraint
    bound_type = 'z'
    a, a_grad = constraints.fixed_cartesian(bound_type, bound_val)
    x_vec = np.array([np.random.randn(num_test_points),
                      np.random.randn(num_test_points),
                      test_vec])
    eps_test, _ = a(x_vec)
    assert equal_to_tolerance(cost, eps_test)

    # ToDo: Test case for 'linear' type cartesian constraints

def equal_to_tolerance(x, y, tol=1e-6)->bool:
    """
    Shorthand to compare two lists and ensure each entry has an error less than the specified tolerance
    """
    if len(x) != len(y): return False
    return all([abs(xx-yy)<tol for xx, yy in zip(x,y)])