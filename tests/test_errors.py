import numpy as np
import pytest

from ewgeo.utils import errors
from ewgeo.utils.covariance import CovarianceMatrix


def equal_to_tolerance(x, y, tol=1e-10):
    return np.all(np.fabs(np.array(x, dtype=float) - np.array(y, dtype=float)) < tol)

def test_cep50():

    res = errors.compute_cep50(CovarianceMatrix(np.array([[1, 0],[0, 1]])))
    assert res == 1.18

def test_cep50_3d():

    res = errors.compute_cep50(CovarianceMatrix(np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])))
    assert res == 1.18

def test_rmse_scaling():
    inputs = [0.01, .1, .2, .3, .4, .5, .6, .7, .8, .9, .99]
    outputs = [0.012533469508069276,
               0.12566134685507416,
               0.2533471031357997,
               0.38532046640756773,
               0.5244005127080407,
               0.6744897501960817,
               0.8416212335729143,
               1.0364333894937898,
               1.2815515655446004,
               1.6448536269514722,
               2.5758293035489004]
    assert all(errors.compute_rmse_scaling(x)==y for x, y in zip(inputs, outputs))

def test_rmse_confidence_interval():
    inputs = np.arange(20) - 10
    outputs = [-1.0, -1.0,
               -0.9999999999999987,
               -0.9999999999974403,
               -0.9999999980268246,
               -0.9999994266968562,
               -0.9999366575163338,
               -0.9973002039367398,
               -0.9544997361036416,
               -0.6826894921370859,
               0.0,
               0.6826894921370859,
               0.9544997361036416,
               0.9973002039367398,
               0.9999366575163338,
               0.9999994266968562,
               0.9999999980268246,
               0.9999999999974403,
               0.9999999999999987,
               1.0]
    assert all(errors.compute_rmse_confidence_interval(x) == y for x, y in zip(inputs, outputs))

def test_draw_cep50():
    """draw_cep50 returns a circle: (a) constant radius, (b) centered on x, (c) radius = compute_cep50."""
    x_target = np.array([3., -5.])
    cov = CovarianceMatrix(np.eye(2))
    num_pts = 100
    xx, yy = errors.draw_cep50(x_target, cov, num_pts=num_pts)

    expected_radius = errors.compute_cep50(cov)  # 1.18

    # (a) all points are equidistant from the center → it is a circle
    radii = np.sqrt((xx - x_target[0])**2 + (yy - x_target[1])**2)
    assert np.all(np.fabs(radii - expected_radius) < 1e-10)

    # (b) centered on the target: midpoint of bounding box equals x_target
    assert equal_to_tolerance((xx.max() + xx.min()) / 2, x_target[0], tol=1e-3)
    assert equal_to_tolerance((yy.max() + yy.min()) / 2, x_target[1], tol=1e-3)

    # (c) radius matches compute_cep50
    assert equal_to_tolerance(radii[0], expected_radius)


def test_draw_error_ellipse():
    """draw_error_ellipse returns correct shape and is centered on x; isotropic cov gives a circle."""
    x_target = np.array([3., -5.])
    cov = CovarianceMatrix(np.eye(2))
    num_pts = 100
    result = errors.draw_error_ellipse(x_target, cov, num_pts=num_pts)

    # Output shape is (2, num_pts)
    assert result.shape == (2, num_pts)

    # For isotropic covariance all radii are equal → ellipse degenerates to a circle
    radii = np.linalg.norm(result - x_target[:, np.newaxis], axis=0)
    expected_r = np.sqrt(1.386)  # sqrt(gamma * lam) with gamma=1.386, lam=1 for 50% CI
    assert np.all(np.fabs(radii - expected_r) < 1e-10)

    # Centered on the target
    assert equal_to_tolerance((result[0].max() + result[0].min()) / 2, x_target[0], tol=1e-3)
    assert equal_to_tolerance((result[1].max() + result[1].min()) / 2, x_target[1], tol=1e-3)


# ---------------------------------------------------------------------------
# compute_cep50_fast
# ---------------------------------------------------------------------------

def test_cep50_fast_matches_scalar():
    """compute_cep50_fast on a single-element list matches compute_cep50 scalar."""
    cov = CovarianceMatrix(np.eye(2))
    result = errors.compute_cep50_fast([cov])
    expected = errors.compute_cep50(cov)   # 0.59*(1+1) = 1.18
    assert equal_to_tolerance(result[0], expected)


def test_cep50_fast_batch_shape():
    """Returns one value per input matrix."""
    covs = [CovarianceMatrix(np.eye(2))] * 5
    result = errors.compute_cep50_fast(covs)
    assert result.shape == (5,)


def test_cep50_fast_batch_all_equal():
    """Identical matrices in the list produce identical CEP values."""
    cov = CovarianceMatrix(np.diag([4., 1.]))
    result = errors.compute_cep50_fast([cov, cov, cov])
    assert np.all(result == result[0])


def test_cep50_fast_3d_input():
    """3D covariance is accepted and returns a finite value."""
    cov = CovarianceMatrix(np.eye(3))
    result = errors.compute_cep50_fast([cov])
    assert result.shape == (1,)
    assert np.isfinite(result[0])


def test_cep50_fast_raises_dim_too_small():
    """1×1 covariance raises ValueError."""
    with pytest.raises(ValueError):
        errors.compute_cep50_fast([CovarianceMatrix(np.eye(1))])


def test_cep50_fast_raises_dim_too_large():
    """4×4 covariance raises ValueError."""
    with pytest.raises(ValueError):
        errors.compute_cep50_fast([CovarianceMatrix(np.eye(4))])


def test_cep50_fast_raises_mixed_sizes():
    """List with mixed-size matrices raises ValueError."""
    with pytest.raises(ValueError):
        errors.compute_cep50_fast([CovarianceMatrix(np.eye(2)), CovarianceMatrix(np.eye(3))])


# ---------------------------------------------------------------------------
# compute_rmse
# ---------------------------------------------------------------------------

def test_rmse_single_identity_2d():
    """sqrt(trace(I_2)) = sqrt(2)."""
    cov = CovarianceMatrix(np.eye(2))
    assert equal_to_tolerance(errors.compute_rmse(cov), np.sqrt(2))


def test_rmse_single_diagonal():
    """sqrt(4 + 9 + 16) = sqrt(29) for diagonal [4, 9, 16]."""
    cov = CovarianceMatrix(np.diag([4., 9., 16.]))
    assert equal_to_tolerance(errors.compute_rmse(cov), np.sqrt(29.))


def test_rmse_list_shape():
    """List of N matrices returns array of length N."""
    covs = [CovarianceMatrix(np.eye(2))] * 4
    result = errors.compute_rmse(covs)
    assert result.shape == (4,)


def test_rmse_list_matches_scalar():
    """Each element of the list result matches the corresponding scalar result."""
    cov = CovarianceMatrix(np.diag([1., 4.]))
    expected = errors.compute_rmse(cov)
    result = errors.compute_rmse([cov, cov])
    assert np.all(equal_to_tolerance(result, expected))


def test_rmse_ndarray_input():
    """Raw ndarray of shape (N, 2, 2) is accepted."""
    covs = np.stack([np.eye(2)] * 3, axis=0)
    result = errors.compute_rmse(covs)
    assert result.shape == (3,)
    assert np.all(equal_to_tolerance(result, np.sqrt(2)))


def test_rmse_raises_bad_type():
    """Non-array, non-list, non-CovarianceMatrix input raises TypeError."""
    with pytest.raises(TypeError):
        errors.compute_rmse("not a covariance")