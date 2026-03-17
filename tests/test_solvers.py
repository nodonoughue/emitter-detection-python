import numpy as np

from ewgeo.utils.covariance import CovarianceMatrix
from ewgeo.utils import SearchSpace
from ewgeo.utils.solvers import ls_solver, gd_solver, ml_solver, bestfix_solver


def equal_to_tolerance(x, y, tol=1e-3):
    if np.size(x) != np.size(y):
        return False
    return np.all(np.fabs(np.array(x) - np.array(y)) < tol)


# ---------------------------------------------------------------------------
# Shared test problem: identity measurement model, y = zeta − x
# True solution is X_TRUE; init is offset by [2, -1]
# ---------------------------------------------------------------------------

X_TRUE = np.array([5.0, 3.0])
COV_ID = CovarianceMatrix(np.eye(2))


def _residual(x):
    """Measurement residual: zeta - h(x) with h(x) = x."""
    return X_TRUE - x


def _jacobian(x):
    """Jacobian dh/dx = I for h(x) = x; shape (n_dim, n_meas) = (2, 2)."""
    return np.eye(2)


X_INIT = X_TRUE + np.array([2.0, -1.0])


# ===========================================================================
# ls_solver
# ===========================================================================

def test_ls_solver_returns_two_outputs():
    x_est, x_full = ls_solver(_residual, _jacobian, COV_ID, X_INIT)
    assert x_est is not None
    assert x_full is not None


def test_ls_solver_converges_to_true():
    x_est, _ = ls_solver(_residual, _jacobian, COV_ID, X_INIT)
    assert equal_to_tolerance(x_est, X_TRUE, tol=1e-4), \
        f"LS did not converge: {x_est} vs {X_TRUE}"


def test_ls_solver_x_full_shape():
    x_est, x_full = ls_solver(_residual, _jacobian, COV_ID, X_INIT)
    assert x_full.ndim == 2
    assert x_full.shape[0] == len(X_TRUE)


def test_ls_solver_different_init():
    """LS should converge from a different starting point."""
    x_init_far = X_TRUE + np.array([-5.0, 8.0])
    x_est, _ = ls_solver(_residual, _jacobian, COV_ID, x_init_far)
    assert equal_to_tolerance(x_est, X_TRUE, tol=1e-4), \
        f"LS did not converge from far init: {x_est} vs {X_TRUE}"


def test_ls_solver_1d_problem():
    """1-D test: h(x) = x, zeta = 7. Previously failed due to 1×1 squeeze→scalar."""
    x_true_1d = np.array([7.0])
    cov_1d = CovarianceMatrix(np.eye(1))
    def res_1d(x): return x_true_1d - x
    def jac_1d(x): return np.eye(1)
    x_est, _ = ls_solver(res_1d, jac_1d, cov_1d, np.array([0.0]))
    assert equal_to_tolerance(x_est, x_true_1d, tol=1e-4)


# ===========================================================================
# gd_solver
# ===========================================================================

def test_gd_solver_returns_two_outputs():
    x_est, x_full = gd_solver(_residual, _jacobian, COV_ID, X_INIT)
    assert x_est is not None
    assert x_full is not None


def test_gd_solver_converges_to_true():
    x_est, _ = gd_solver(_residual, _jacobian, COV_ID, X_INIT)
    assert equal_to_tolerance(x_est, X_TRUE, tol=1e-3), \
        f"GD did not converge: {x_est} vs {X_TRUE}"


def test_gd_solver_x_full_shape():
    x_est, x_full = gd_solver(_residual, _jacobian, COV_ID, X_INIT)
    assert x_full.ndim == 2
    assert x_full.shape[0] == len(X_TRUE)


# ===========================================================================
# ls_solver / gd_solver — constraint tests
# ===========================================================================

# Constraint helpers (designed for 2-D input, which is what snap_to_constraints
# always passes to constraint callables)

def _eq_x0_zero(x):
    """Equality constraint: pin x[0] to zero."""
    eps = x[0].copy()
    x_v = x.copy()
    x_v[0] = 0.
    return eps, x_v


def _ineq_x1_le_10(x):
    """Inequality constraint: x[1] must not exceed 10."""
    eps = x[1] - 10.
    x_v = x.copy()
    x_v[1] = np.minimum(x_v[1], 10.)
    return eps, x_v


def test_ls_solver_with_eq_constraint():
    """LS solver satisfies an equality constraint at convergence."""
    x_est, _ = ls_solver(_residual, _jacobian, COV_ID, X_INIT,
                          eq_constraints=[_eq_x0_zero])
    assert abs(x_est[0]) < 1e-3, f"x[0]={x_est[0]:.4f} should be ~0 (eq constraint)"


def test_ls_solver_with_both_constraints():
    """LS solver handles simultaneous eq + ineq constraints without error."""
    # Equality pins x[0]=0; inequality x[1]<=10 is inactive (X_TRUE[1]=3)
    x_est, _ = ls_solver(_residual, _jacobian, COV_ID, X_INIT,
                          eq_constraints=[_eq_x0_zero],
                          ineq_constraints=[_ineq_x1_le_10])
    assert abs(x_est[0]) < 1e-3, f"x[0]={x_est[0]:.4f} should be ~0 (eq constraint)"
    assert abs(x_est[1] - X_TRUE[1]) < 1e-3, \
        f"x[1]={x_est[1]:.4f} should converge to {X_TRUE[1]} (ineq inactive)"


def test_gd_solver_with_eq_constraint():
    """GD solver satisfies an equality constraint at convergence."""
    x_est, _ = gd_solver(_residual, _jacobian, COV_ID, X_INIT,
                          eq_constraints=[_eq_x0_zero])
    assert abs(x_est[0]) < 1e-3, f"x[0]={x_est[0]:.4f} should be ~0 (eq constraint)"


def test_gd_solver_with_both_constraints():
    """GD solver handles simultaneous eq + ineq constraints without error."""
    x_est, _ = gd_solver(_residual, _jacobian, COV_ID, X_INIT,
                          eq_constraints=[_eq_x0_zero],
                          ineq_constraints=[_ineq_x1_le_10])
    assert abs(x_est[0]) < 1e-3, f"x[0]={x_est[0]:.4f} should be ~0 (eq constraint)"
    assert abs(x_est[1] - X_TRUE[1]) < 1e-3, \
        f"x[1]={x_est[1]:.4f} should converge to {X_TRUE[1]} (ineq inactive)"


# ===========================================================================
# ml_solver
# ===========================================================================

def _log_likelihood(x_set):
    """Gaussian log-likelihood centred on X_TRUE; x_set shape (2, N)."""
    diff = x_set - X_TRUE[:, np.newaxis]
    return -0.5 * np.sum(diff ** 2, axis=0)


def test_ml_solver_returns_three_outputs():
    ss = SearchSpace(x_ctr=X_TRUE, epsilon=0.5, max_offset=5.0)
    result = ml_solver(_log_likelihood, ss)
    assert len(result) == 3


def test_ml_solver_locates_peak():
    ss = SearchSpace(x_ctr=X_TRUE, epsilon=0.5, max_offset=5.0)
    x_est, likelihood, _ = ml_solver(_log_likelihood, ss)
    assert np.linalg.norm(x_est - X_TRUE) < 1.0, \
        f"ML peak too far from truth: {np.linalg.norm(x_est - X_TRUE):.3f}"


def test_ml_solver_likelihood_shape():
    ss = SearchSpace(x_ctr=X_TRUE, epsilon=1.0, max_offset=5.0)
    _, likelihood, x_grid = ml_solver(_log_likelihood, ss)
    assert likelihood is not None
    assert x_grid is not None


# ===========================================================================
# bestfix_solver
# ===========================================================================

def test_bestfix_solver_returns_three_outputs():
    from scipy.stats import multivariate_normal
    rv = multivariate_normal(mean=X_TRUE, cov=np.eye(2))
    pdfs = [rv.pdf]
    ss = SearchSpace(x_ctr=X_TRUE, epsilon=0.5, max_offset=5.0)
    result = bestfix_solver(pdfs, ss)
    assert len(result) == 3


def test_bestfix_solver_locates_peak():
    from scipy.stats import multivariate_normal
    rv = multivariate_normal(mean=X_TRUE, cov=np.eye(2))
    pdfs = [rv.pdf]
    ss = SearchSpace(x_ctr=X_TRUE, epsilon=0.5, max_offset=5.0)
    x_est, _, _ = bestfix_solver(pdfs, ss)
    assert np.linalg.norm(x_est - X_TRUE) < 1.0, \
        f"BESTFIX peak too far from truth: {np.linalg.norm(x_est - X_TRUE):.3f}"


def test_bestfix_solver_multiple_pdfs():
    """With two Gaussians at the same centre the product still peaks there."""
    from scipy.stats import multivariate_normal
    rv1 = multivariate_normal(mean=X_TRUE, cov=2 * np.eye(2))
    rv2 = multivariate_normal(mean=X_TRUE, cov=0.5 * np.eye(2))
    pdfs = [rv1.pdf, rv2.pdf]
    ss = SearchSpace(x_ctr=X_TRUE, epsilon=0.5, max_offset=5.0)
    x_est, _, _ = bestfix_solver(pdfs, ss)
    assert np.linalg.norm(x_est - X_TRUE) < 1.0
