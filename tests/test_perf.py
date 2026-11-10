import numpy as np
import pytest

from ewgeo.utils.covariance import CovarianceMatrix
from ewgeo.utils.perf import compute_crlb_gaussian


def equal_to_tolerance(x, y, tol=1e-6):
    return np.all(np.fabs(np.array(x, dtype=float) - np.array(y, dtype=float)) < tol)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# 2D source; identity measurement model: h(x) = x, J = I, cov = sigma^2 * I
SIGMA = 2.0
COV_2D = CovarianceMatrix(SIGMA**2 * np.eye(2))


def _jac_identity(x):
    """Jacobian for h(x) = x in 2D: shape (2, 2)."""
    return np.eye(2)


# 2D source; measurement only sensitive to x[0]: J = [[1], [0]], cov = sigma^2
COV_1D = CovarianceMatrix(SIGMA**2 * np.eye(1))


def _jac_x0_only(x):
    """Jacobian sensitive only to x[0]: shape (2, 1)."""
    return np.array([[1.0], [0.0]])


# ---------------------------------------------------------------------------
# Fully observable problem
# ---------------------------------------------------------------------------

def test_crlb_observable_returns_covariance_matrix():
    x = np.array([1.0, 2.0])
    result = compute_crlb_gaussian(x, _jac_identity, COV_2D)
    assert isinstance(result, CovarianceMatrix)


def test_crlb_observable_value():
    """For h(x)=x with cov=sigma^2*I, CRLB should equal sigma^2*I."""
    x = np.array([1.0, 2.0])
    crlb = compute_crlb_gaussian(x, _jac_identity, COV_2D)
    assert equal_to_tolerance(crlb.cov, SIGMA**2 * np.eye(2), tol=1e-8)


# ---------------------------------------------------------------------------
# Unobservable dimension — default unobservable_dim_err
# ---------------------------------------------------------------------------

def test_crlb_unobservable_dim_observable_entry():
    """Observable dimension (x[0]) should still yield the correct CRLB."""
    x = np.array([1.0, 2.0])
    with pytest.warns(RuntimeWarning, match='unobservable'):
        crlb = compute_crlb_gaussian(x, _jac_x0_only, COV_1D)
    assert equal_to_tolerance(crlb.cov[0, 0], SIGMA**2, tol=1e-8)


def test_crlb_unobservable_dim_replaced_with_sentinel():
    """Unobservable dimension (x[1]) diagonal must be replaced with unobservable_dim_err."""
    x = np.array([1.0, 2.0])
    with pytest.warns(RuntimeWarning, match='unobservable'):
        crlb = compute_crlb_gaussian(x, _jac_x0_only, COV_1D)
    assert crlb.cov[1, 1] == 1e99


def test_crlb_unobservable_dim_off_diagonal_zero():
    """Off-diagonal entries coupling observable and unobservable dims should be zero."""
    x = np.array([1.0, 2.0])
    with pytest.warns(RuntimeWarning, match='unobservable'):
        crlb = compute_crlb_gaussian(x, _jac_x0_only, COV_1D)
    assert crlb.cov[0, 1] == 0.0
    assert crlb.cov[1, 0] == 0.0


# ---------------------------------------------------------------------------
# Custom unobservable_dim_err value
# ---------------------------------------------------------------------------

def test_crlb_custom_unobservable_dim_err():
    """Custom unobservable_dim_err should appear on the unobservable diagonal."""
    x = np.array([1.0, 2.0])
    sentinel = 42.0
    with pytest.warns(RuntimeWarning, match='unobservable'):
        crlb = compute_crlb_gaussian(x, _jac_x0_only, COV_1D, unobservable_dim_err=sentinel)
    assert crlb.cov[1, 1] == sentinel


# ---------------------------------------------------------------------------
# Ill-defined Fisher matrix (NaN/Inf) — existing NaN path
# ---------------------------------------------------------------------------

def _jac_nan(x):
    """Jacobian that returns NaN, making the Fisher matrix ill-defined."""
    return np.array([[np.nan, 0.0], [0.0, 1.0]])


def test_crlb_nan_fisher_returns_nan_diagonal():
    """When the Fisher matrix contains NaN the result should have NaN on the diagonal."""
    x = np.array([1.0, 2.0])
    cov = CovarianceMatrix(np.eye(2))
    crlb = compute_crlb_gaussian(x, _jac_nan, cov)
    assert np.all(np.isnan(np.diag(crlb.cov)))


# ---------------------------------------------------------------------------
# Multi-source input
# ---------------------------------------------------------------------------

def test_crlb_multi_source_returns_list():
    x_sources = np.array([[1.0, 3.0], [2.0, 4.0]])  # shape (2, 2) — two sources
    result = compute_crlb_gaussian(x_sources, _jac_identity, COV_2D)
    assert isinstance(result, list)
    assert len(result) == 2


def test_crlb_multi_source_each_entry_is_covariance_matrix():
    x_sources = np.array([[1.0, 3.0], [2.0, 4.0]])
    result = compute_crlb_gaussian(x_sources, _jac_identity, COV_2D)
    for entry in result:
        assert isinstance(entry, CovarianceMatrix)


def test_crlb_multi_source_unobservable_dim_each_source():
    """Unobservable-dim sentinel must appear in every source's CRLB."""
    x_sources = np.array([[1.0, 3.0], [2.0, 4.0]])
    with pytest.warns(RuntimeWarning, match='unobservable'):
        result = compute_crlb_gaussian(x_sources, _jac_x0_only, COV_1D)
    for crlb in result:
        assert crlb.cov[1, 1] == 1e99


# ---------------------------------------------------------------------------
# Warning on unobservable directions
# ---------------------------------------------------------------------------

def test_crlb_unobservable_warns_once_for_multiple_sources():
    x_sources = np.array([[1.0, 3.0], [2.0, 4.0]])
    with pytest.warns(RuntimeWarning, match='2 of 2 source') as record:
        compute_crlb_gaussian(x_sources, _jac_x0_only, COV_1D)
    assert len(record) == 1


def test_crlb_observable_does_not_warn(recwarn):
    compute_crlb_gaussian(np.array([1.0, 2.0]), _jac_identity, COV_2D)
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def _jac_diagonal(x):
    """Measurement sensitive only to x[0] + x[1]: unobservable direction is (1, -1)/sqrt(2)."""
    return np.array([[1.0], [1.0]])


def test_crlb_unobservable_direction_not_axis_aligned():
    """The sentinel belongs on the unobservable direction, not on whichever axes look zero."""
    err = 1e6
    with pytest.warns(RuntimeWarning, match='unobservable'):
        crlb = compute_crlb_gaussian(np.array([1.0, 2.0]), _jac_diagonal, COV_1D, unobservable_dim_err=err)
    # Observable part: pinv of F = (1/sigma^2) [[1,1],[1,1]] is sigma^2/4 [[1,1],[1,1]]
    expected = SIGMA**2 / 4 * np.ones((2, 2)) + err / 2 * np.array([[1.0, -1.0], [-1.0, 1.0]])
    assert np.allclose(crlb.cov, expected, rtol=1e-9)


# ---------------------------------------------------------------------------
# Equality constraints
# ---------------------------------------------------------------------------

def _grad_fix_x1(x):
    """Constraint gradient for x[1] = const: shape (n_dim, n_source)."""
    return np.array([[0.0], [1.0]])


def _grad_sum(x):
    """Constraint gradient for x[0] + x[1] = const."""
    return np.array([[1.0], [1.0]])


def test_crlb_constrained_dim_has_zero_variance(recwarn):
    """A dimension fixed exactly by a constraint has zero variance, not the unobservable sentinel."""
    crlb = compute_crlb_gaussian(np.array([1.0, 2.0]), _jac_identity, COV_2D, eq_constraints_grad=[_grad_fix_x1])
    assert np.allclose(crlb.cov, np.diag([SIGMA**2, 0.0]), atol=1e-12)
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def test_crlb_constrained_matches_projection_formula():
    """For a well-conditioned problem, C = F^-1 - F^-1 g (g' F^-1 g)^-1 g' F^-1 = sigma^2 (I - g g'/2)."""
    crlb = compute_crlb_gaussian(np.array([1.0, 2.0]), _jac_identity, COV_2D, eq_constraints_grad=[_grad_sum])
    assert np.allclose(crlb.cov, SIGMA**2 * (np.eye(2) - 0.5 * np.ones((2, 2))), atol=1e-12)


def _jac_zero_row(x):
    """Measurements carry no information about x[1] (e.g., a coplanar sensor/target geometry)."""
    return np.array([[1.0, 0.0], [0.0, 0.0]])


def test_crlb_constraint_on_unobservable_dim_restores_finite_bound(recwarn):
    """Constraining an otherwise-unobservable dimension gives a finite bound (no LinAlgError, no warning)."""
    crlb = compute_crlb_gaussian(np.array([1.0, 2.0]), _jac_zero_row, COV_2D, eq_constraints_grad=[_grad_fix_x1])
    assert np.allclose(crlb.cov, np.diag([SIGMA**2, 0.0]), atol=1e-12)
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def test_crlb_unobservable_dim_not_fixed_by_constraint_still_warns():
    """A constraint on a different dimension does not make the unobservable one observable."""
    def grad_fix_x0(x):
        return np.array([[1.0], [0.0]])
    with pytest.warns(RuntimeWarning, match='unobservable'):
        crlb = compute_crlb_gaussian(np.array([1.0, 2.0]), _jac_zero_row, COV_2D, eq_constraints_grad=[grad_fix_x0])
    assert np.allclose(crlb.cov, np.diag([0.0, 1e99]))
