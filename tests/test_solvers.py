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
