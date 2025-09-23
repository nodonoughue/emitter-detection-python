import numpy as np
import scipy as sp

from ewgeo.utils.covariance import CovarianceMatrix


def test_init():
    """
    Test that we can initialize a covariance matrix, and that the fields
    are properly initialized

    1. No flags
    2. do_chol=True, do_inverse=False
    3. do_cholesky=False, do_inverse=True
    4. Use block_diagonal to initialize
    """

    c = np.array([[1, .1, .2],[.1, 3, .4],[.2, .4, 3]])

    # 1 - No flags
    cov = CovarianceMatrix(c)
    assert equal_to_tolerance(cov.cov,c)
    assert cov.do_cholesky == True
    assert cov.do_inverse == True
    assert equal_to_tolerance(cov.lower @ cov.lower.T, c)
    assert equal_to_tolerance(cov.inv @ c, np.eye(3))

    # 2 - No Inverse
    cov = CovarianceMatrix(c, do_inverse=False)
    assert equal_to_tolerance(cov.cov,c)
    assert cov.do_cholesky == True
    assert cov.do_inverse == False
    assert equal_to_tolerance(cov.lower @ cov.lower.T, c)
    assert cov.inv is None

    # 3 - No Cholesky Decomp
    cov = CovarianceMatrix(c, do_cholesky=False)
    assert equal_to_tolerance(cov.cov,c)
    assert cov.do_cholesky == False
    assert cov.do_inverse == True
    assert cov.lower is None
    assert equal_to_tolerance(cov.inv @ c, np.eye(3))

    # 4 - Block diagonal
    c2 = np.array([[2, 1],[1, 3]])
    c_full = sp.linalg.block_diag(c, c2)
    cov2 = CovarianceMatrix(c2)
    cov_full = CovarianceMatrix.block_diagonal(cov, cov2)
    assert equal_to_tolerance(cov_full.cov,c_full)
    assert cov_full.do_cholesky == True
    assert cov_full.do_inverse == True
    assert equal_to_tolerance(cov_full.lower @ cov_full.lower.T, c_full)
    assert equal_to_tolerance(cov_full.inv @ c_full, np.eye(5))

def test_manual():
    """
    Test manually setting various parameters

    1. cov
    2. lower
    3. inv
    """

def test_copy():
    """
    Make a copy, edit it. Test that the original is unchanged.
    """

def test_solve_aca():
    """
    Solve the matrix problem res = A @ C^{-1} @ A.T

    Do it with do_inverse=True and =False
    """

def test_solve_acb():
    """
    Solve the matrix problem res = A @ C^{-1} @ B

    Do it with do_inverse=True and =False
    """

def test_solve_lstsq():
    """
    Solve a linear equation of the form for x:
            J@C^{-1}@y = J@C^{-1}@J.T @ x

    Do it with do_inverse=True and =False
    """

def test_resample():
    """
    Define multiple reference index types, check that outputs are of the
    expected shape and composition.

    1. ref_idx=None
    2. ref_idx is an integer
    3. ref_idx='Full'
    4. ref_idx='full'
    5. ref_idx_vec and test_idx_vec are both defined
    """

def test_resample_hybrid():
    """
    Resample a covariance matrix from a hybrid PSS sytem

    1. AOA/TDOA
    2. AOA/FDOA
    3. TDOA/FDOA
    4. AOA/TDOA/FDOA
    """

def test_multiply():
    """
    Multiply a covariance matrix by a constant.

    Do it with overwrite=True and =False, check the proper behavior of
    the original CovarianceMatrix.
    """

def test_sample():
    """
    Generate a random sample; check dimensions.
    """

def test_sample_shape():
    """
    Generate 1,000,000 random samples; compute covariance matrix estimate.

    Check that the estimated covariance matrix is close to the true covariance.
    """

def equal_to_tolerance(x, y, tol=1e-6)->bool:
    """
    Shorthand to compare two lists and ensure each entry has an error less than the specified tolerance
    """
    if np.any(np.shape(x) != np.shape(y)): return False
    return all([abs(xx-yy)<tol for xx, yy in zip(np.ravel(x),np.ravel(y))])