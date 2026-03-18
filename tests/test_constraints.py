import numpy as np
import pytest

from ewgeo.utils import constraints, coordinates, constants
from ewgeo.utils.constraints import (
    bounded_alt,
    snap_to_constraints,
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


def test_fixed_cartesian_linear():
    """Linear constraint snaps points onto the line through x0 in direction u_vec."""
    # Line through origin along the x-axis
    x0 = np.array([0., 0., 0.])
    u_vec = np.array([1., 0., 0.])
    a, a_gradient = constraints.fixed_cartesian('linear', x0=x0, u_vec=u_vec)

    # Points already on the line: epsilon should be zero, x_valid unchanged
    x_on = np.array([[1., 2., -3.], [0., 0., 0.], [0., 0., 0.]])
    eps_on, x_valid_on = a(x_on)
    assert equal_to_tolerance(eps_on, np.zeros(3))
    assert np.allclose(x_valid_on, x_on)

    # Point off the line: offset 3 in y, 0 in z → distance from line = 3
    x_off = np.array([[5.], [3.], [0.]])
    eps_off, x_valid_off = a(x_off)
    assert equal_to_tolerance(eps_off, [3.])
    assert np.allclose(x_valid_off, [[5.], [0.], [0.]])  # snapped onto the line

    # is_upper_bound=False flips the sign of epsilon
    a_lower, _ = constraints.fixed_cartesian('linear', x0=x0, u_vec=u_vec, is_upper_bound=False)
    eps_lower, _ = a_lower(x_off)
    assert equal_to_tolerance(eps_lower, [-3.])

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
# snap_to_equality_constraints  (violation path)
# ---------------------------------------------------------------------------

def test_snap_to_equality_snaps_violated_point():
    """A point that violates the constraint is projected onto the constraint surface."""
    # Constraint: x[0] must be zero; snap by setting x[0] = 0
    def snap_x0_zero(x):
        eps = x[0].copy()
        x_v = x.copy()
        x_v[0] = 0.
        return eps, x_v

    x = np.array([[5.], [3.]])   # x[0] = 5 violates the constraint
    x_out = snap_to_equality_constraints(x, [snap_x0_zero])
    assert x_out[0, 0] == pytest.approx(0.0)
    assert x_out[1, 0] == pytest.approx(3.0)   # unchanged


def test_snap_to_equality_1d_input_shape_preserved():
    """1-D input (n_dim,) is handled correctly and the output shape is preserved."""
    # Constraint: x[0] must be zero
    def snap_x0_zero(x):
        eps = x[0].copy()
        x_v = x.copy()
        x_v[0] = 0.
        return eps, x_v

    x = np.array([7., 4.])       # 1-D input
    x_out = snap_to_equality_constraints(x, [snap_x0_zero])
    assert x_out.shape == (2,)
    assert x_out[0] == pytest.approx(0.0)
    assert x_out[1] == pytest.approx(4.0)


def test_snap_to_equality_recheck_loop():
    """
    Snapping constraint B can re-violate constraint A; the re-check loop
    must iterate until both are satisfied.

    Constraint A: x[0] = x[1]  (snap by setting x[0] <- x[1])
    Constraint B: x[1] = 0     (snap by setting x[1] <- 0)

    Without the re-check loop a single pass would leave x = [[3.], [0.]],
    which violates A.  With the re-check loop the result is [[0.], [0.]].
    """
    def equalize_x0_x1(x):
        eps = x[0] - x[1]
        x_v = x.copy()
        x_v[0] = x_v[1]
        return eps, x_v

    def zero_x1(x):
        eps = x[1].copy()
        x_v = x.copy()
        x_v[1] = 0.
        return eps, x_v

    x = np.array([[5.], [3.]])
    x_out = snap_to_equality_constraints(x, [equalize_x0_x1, zero_x1])
    assert x_out[0, 0] == pytest.approx(0.0)
    assert x_out[1, 0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# snap_to_inequality_constraints  (violation path)
# ---------------------------------------------------------------------------

def test_snap_to_inequality_snaps_violated_point():
    """A point that violates the inequality is projected to the boundary."""
    # Constraint: x[0] <= 2; snap by clamping to 2
    def clip_x0_le_2(x):
        max_val = 2.
        eps = x[0] - max_val
        x_v = x.copy()
        x_v[0] = np.minimum(x_v[0], max_val)
        return eps, x_v

    x = np.array([[9.], [3.]])   # x[0] = 9 violates x[0] <= 2
    x_out = snap_to_inequality_constraints(x, [clip_x0_le_2])
    assert x_out[0, 0] == pytest.approx(2.0)
    assert x_out[1, 0] == pytest.approx(3.0)   # unchanged


def test_snap_to_inequality_1d_input_shape_preserved():
    """1-D input is handled correctly and the output shape is preserved."""
    def clip_x0_le_2(x):
        max_val = 2.
        eps = x[0] - max_val
        x_v = x.copy()
        x_v[0] = np.minimum(x_v[0], max_val)
        return eps, x_v

    x = np.array([9., 3.])       # 1-D input
    x_out = snap_to_inequality_constraints(x, [clip_x0_le_2])
    assert x_out.shape == (2,)
    assert x_out[0] == pytest.approx(2.0)
    assert x_out[1] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# snap_to_constraints  (combined eq + ineq)
# ---------------------------------------------------------------------------

def test_snap_to_constraints_eq_only():
    """snap_to_constraints with only equality constraints matches snap_to_equality_constraints."""
    def snap_x0_zero(x):
        eps = x[0].copy()
        x_v = x.copy()
        x_v[0] = 0.
        return eps, x_v

    x = np.array([[5.], [3.]])
    x_out_combined = snap_to_constraints(x, eq_constraints=[snap_x0_zero])
    x_out_eq       = snap_to_equality_constraints(x, [snap_x0_zero])
    assert np.allclose(x_out_combined, x_out_eq)


def test_snap_to_constraints_ineq_only():
    """snap_to_constraints with only inequality constraints matches snap_to_inequality_constraints."""
    def clip_x0_le_2(x):
        max_val = 2.
        eps = x[0] - max_val
        x_v = x.copy()
        x_v[0] = np.minimum(x_v[0], max_val)
        return eps, x_v

    x = np.array([[9.], [3.]])
    x_out_combined  = snap_to_constraints(x, ineq_constraints=[clip_x0_le_2])
    x_out_ineq      = snap_to_inequality_constraints(x, [clip_x0_le_2])
    assert np.allclose(x_out_combined, x_out_ineq)


def test_snap_to_constraints_both_types():
    """Both eq and ineq constraints are satisfied simultaneously in the output."""
    # Equality: x[0] = 0
    def snap_x0_zero(x):
        eps = x[0].copy()
        x_v = x.copy()
        x_v[0] = 0.
        return eps, x_v

    # Inequality: x[1] <= 5
    def clip_x1_le_5(x):
        max_val = 5.
        eps = x[1] - max_val
        x_v = x.copy()
        x_v[1] = np.minimum(x_v[1], max_val)
        return eps, x_v

    x = np.array([[3.], [8.]])   # both violated
    x_out = snap_to_constraints(x, eq_constraints=[snap_x0_zero], ineq_constraints=[clip_x1_le_5])
    assert x_out[0, 0] == pytest.approx(0.0)   # equality satisfied
    assert x_out[1, 0] == pytest.approx(5.0)   # inequality satisfied


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