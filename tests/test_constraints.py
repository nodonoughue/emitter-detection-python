import numpy as np
import pytest

from ewgeo.utils import constraints, coordinates, constants
from ewgeo.utils.constraints import (
    bounded_alt,
    snap_to_equality_constraints,
    snap_to_inequality_constraints,
    constrain_likelihood,
)


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

# ---------------------------------------------------------------------------
# bounded_alt
# ---------------------------------------------------------------------------

def test_bounded_alt_max_only():
    """Only alt_max → list with one upper-bound constraint."""
    bounds = bounded_alt('flat', alt_max=1000.)
    assert len(bounds) == 1


def test_bounded_alt_min_only():
    """Only alt_min → list with one lower-bound constraint."""
    bounds = bounded_alt('flat', alt_min=0.)
    assert len(bounds) == 1


def test_bounded_alt_both():
    """Both alt_min and alt_max → list with two constraints."""
    bounds = bounded_alt('flat', alt_min=0., alt_max=1000.)
    assert len(bounds) == 2


def test_bounded_alt_neither():
    """No bounds → empty list."""
    bounds = bounded_alt('flat')
    assert bounds == []


def test_bounded_alt_returns_callables():
    """Each element in the returned list is callable."""
    bounds = bounded_alt('flat', alt_min=0., alt_max=500.)
    for b in bounds:
        assert callable(b)


# ---------------------------------------------------------------------------
# snap_to_equality_constraints  (tests for already-satisfied path only)
# ---------------------------------------------------------------------------

def test_snap_to_equality_already_satisfied():
    """Point that already satisfies the constraint is returned unchanged."""
    # Constraint: eps = 0 for any input (always satisfied)
    def always_ok(x):
        return np.zeros(x.shape[1]), x.copy()

    x = np.array([[3.], [4.], [5.]])
    x_out = snap_to_equality_constraints(x, [always_ok])
    assert np.array_equal(x_out, x)


def test_snap_to_equality_does_not_mutate_input():
    """Input array is not modified in-place."""
    def always_ok(x):
        return np.zeros(x.shape[1]), x.copy()

    x = np.array([[1.], [2.], [3.]])
    x_orig = x.copy()
    snap_to_equality_constraints(x, [always_ok])
    assert np.array_equal(x, x_orig)


# ---------------------------------------------------------------------------
# snap_to_inequality_constraints  (tests for already-satisfied path only)
# ---------------------------------------------------------------------------

def test_snap_to_inequality_already_satisfied():
    """Point that already satisfies the constraint is returned unchanged."""
    # Constraint: eps = -1 (always satisfied, since -1 <= 0)
    def always_ok(x):
        return -np.ones(x.shape[1]), x.copy()

    x = np.array([[1.], [2.]])
    x_out = snap_to_inequality_constraints(x, [always_ok])
    assert np.array_equal(x_out, x)


def test_snap_to_inequality_does_not_mutate_input():
    """Input array is not modified in-place."""
    def always_ok(x):
        return -np.ones(x.shape[1]), x.copy()

    x = np.array([[5.], [6.]])
    x_orig = x.copy()
    snap_to_inequality_constraints(x, [always_ok])
    assert np.array_equal(x, x_orig)


# ---------------------------------------------------------------------------
# constrain_likelihood
# ---------------------------------------------------------------------------

def test_constrain_likelihood_returns_callable():
    """constrain_likelihood always returns a callable."""
    ell = lambda x: np.ones(x.shape[1])
    eq  = lambda x: (np.zeros(x.shape[1]), x)
    f = constrain_likelihood(ell, eq_constraints=[eq])
    assert callable(f)


def test_constrain_likelihood_satisfied_eq():
    """Point satisfying the equality constraint receives ell's value."""
    ell = lambda x: np.full(x.shape[1], 7.0)
    eq  = lambda x: (np.zeros(x.shape[1]), x)   # eps = 0 ≤ tol
    f = constrain_likelihood(ell, eq_constraints=[eq])
    x = np.array([[1.], [2.]])
    result = f(x)
    assert result[0] == pytest.approx(7.0)


def test_constrain_likelihood_violated_eq():
    """Point violating the equality constraint gets -inf."""
    ell = lambda x: np.ones(x.shape[1])
    eq  = lambda x: (np.full(x.shape[1], 1e6), x)   # eps huge → violated
    f = constrain_likelihood(ell, eq_constraints=[eq])
    x = np.array([[1.], [2.]])
    result = f(x)
    assert np.isinf(result[0]) and result[0] < 0


def test_constrain_likelihood_satisfied_ineq():
    """Point satisfying the inequality constraint receives ell's value."""
    ell = lambda x: np.full(x.shape[1], 3.0)
    ineq = lambda x: (-np.ones(x.shape[1]), x)   # eps = -1 ≤ 0
    f = constrain_likelihood(ell, ineq_constraints=[ineq])
    x = np.array([[1.], [2.]])
    result = f(x)
    assert result[0] == pytest.approx(3.0)


def test_constrain_likelihood_violated_ineq():
    """Point violating the inequality constraint gets -inf."""
    ell = lambda x: np.ones(x.shape[1])
    ineq = lambda x: (np.full(x.shape[1], 5.0), x)   # eps = 5 > 0
    f = constrain_likelihood(ell, ineq_constraints=[ineq])
    x = np.array([[1.], [2.]])
    result = f(x)
    assert np.isinf(result[0]) and result[0] < 0


def equal_to_tolerance(x, y, tol=1e-6)->bool:
    """
    Shorthand to compare two lists and ensure each entry has an error less than the specified tolerance
    """
    if len(x) != len(y): return False
    return all([abs(xx-yy)<tol for xx, yy in zip(x,y)])