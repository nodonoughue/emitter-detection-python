import numpy as np
import scipy as sp

from ewgeo.utils.covariance import CovarianceMatrix
from ewgeo.utils import resample_covariance_matrix

def test_init():
    """
    Test that we can initialize a covariance matrix and that the fields
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
    c = np.array([[1, .1, .2], [.1, 3, .4], [.2, .4, 3]])
    c2 = 2*c

    cov_test = CovarianceMatrix(c2)

    # Step 1: manually set cov
    cov = CovarianceMatrix(c)  # define with c
    cov.cov = c2               # overwrite with c2
    assert equal_to_tolerance(cov.cov, cov_test.cov)
    assert equal_to_tolerance(cov.inv, cov_test.inv)
    assert equal_to_tolerance(cov.lower, cov_test.lower)

    # Step 2: manually set lower
    # In this case, the do_inverse flag gets set to False
    cov = CovarianceMatrix(c)    # define with c
    cov.lower = cov_test.lower  # overwrite with lower from c2
    assert equal_to_tolerance(cov.cov, cov_test.cov)
    assert equal_to_tolerance(cov.lower, cov_test.lower)
    assert cov.do_inverse == False
    assert cov.inv is None

    # Step 3: manually set inv
    # In this case, the do_cholesky flag gets set to False and
    # the original .cov is cleared
    cov = CovarianceMatrix(c)  # define with c
    cov.inv = cov_test.inv  # overwrite with inv from c2
    assert equal_to_tolerance(cov.inv, cov_test.inv)
    assert cov.cov is None
    assert cov.do_cholesky == False
    assert cov.lower is None

    return

def test_copy():
    """
    Make a copy, edit it. Test that the original is unchanged.
    """
    c = np.array([[1, .1, .2], [.1, 3, .4], [.2, .4, 3]])
    c2 = 2*c

    cov = CovarianceMatrix(c)
    cov2 = cov.copy()
    cov2.cov = c2

    assert ~equal_to_tolerance(cov.cov, cov2.cov)
    assert ~equal_to_tolerance(cov.inv, cov2.inv)
    assert ~equal_to_tolerance(cov.lower, cov2.lower)

def test_solve_aca():
    """
    Solve the matrix problem res = A @ C^{-1} @ A.T

    Do it with do_inverse=True and =False

    These commands should be equivalent:
      res = cov.solve_aca(a)
      res = np.sum(scipy.linalg.solve_triangular(cov.lower, err, lower=True)**2)
    """
    c = np.array([[1, .1, .2], [.1, 3, .4], [.2, .4, 3]])
    a = np.array([.3, .5, .2])

    cov = CovarianceMatrix(c, do_inverse=True)
    res1 = cov.solve_aca(a)
    res2 = np.sum(sp.linalg.solve_triangular(cov.lower, a, lower=True)**2)
    assert equal_to_tolerance(res1, res2)

    cov = CovarianceMatrix(c, do_inverse=False)
    res1 = cov.solve_aca(a)
    res2 = np.sum(sp.linalg.solve_triangular(cov.lower, a, lower=True)**2)
    assert equal_to_tolerance(res1, res2)

    # Multidimensional inputs
    a = np.tile(np.reshape(a, (1, 1, 1, 3)), (2, 3, 4, 1))
    res1 = cov.solve_aca(a)
    assert equal_to_tolerance(res1, res2*np.ones_like(res1))
    assert res1.shape == (2, 3, 4)

def test_solve_acb():
    """
    Solve the matrix problem res = A @ C^{-1} @ B

    Do it with do_inverse=True and =False

    The following commands should be equivalent:
      res = cov.solve_acb(jacobian,y)

      r1 = scipy.linalg.solve_triangular(cov.lower, jacobian.T, lower=True)
      r2 = scipy.linalg.solve_triangular(cov.lower, y, lower=True)
      res = r1.T @ r2
    """
    c = np.array([[1, .1, .2], [.1, 3, .4], [.2, .4, 3]])
    jacobian = np.array([[1, 4, 3],[2, 1, 5]])
    y = np.array([.3, .5, .2])

    cov = CovarianceMatrix(c, do_inverse=True)
    res1 = cov.solve_acb(jacobian,y)
    r1 = sp.linalg.solve_triangular(cov.lower, jacobian.T, lower=True)
    r2 = sp.linalg.solve_triangular(cov.lower, y, lower=True)
    res2 = r1.T @ r2
    assert equal_to_tolerance(res1, res2)

    cov = CovarianceMatrix(c, do_inverse=False)
    res1 = cov.solve_acb(jacobian,y)
    r1 = sp.linalg.solve_triangular(cov.lower, jacobian.T, lower=True)
    r2 = sp.linalg.solve_triangular(cov.lower, y, lower=True)
    res2 = r1.T @ r2
    assert equal_to_tolerance(res1, res2)

def test_solve_lstsq():
    """
    Solve a linear equation of the form for x:
            J@C^{-1}@y = J@C^{-1}@J.T @ x

    Do it with do_inverse=True and =False

    The following commands should be equivalent:
       res = cov.solve_lstsq(y, jacobian)

       a = scipy.linalg.solve_triangular(cov.lower, jacobian.T, lower=True)
       b = scipy.linalg.solve_triangular(cov.lower, y, lower=True)
       res, _, _, _ = np.linalg.lstsq(a.T @ a, a.T @ b, rcond=None)
    """
    c = np.array([[1, .1, .2], [.1, 3, .4], [.2, .4, 3]])
    jacobian = np.array([[1, 4, 3],[2, 1, 5]])
    y = np.array([.3, .5, .2])

    cov = CovarianceMatrix(c, do_inverse=True)
    res = cov.solve_lstsq(y, jacobian)
    a = sp.linalg.solve_triangular(cov.lower, jacobian.T, lower=True)
    b = sp.linalg.solve_triangular(cov.lower, y, lower=True)
    res2, _, _, _ = np.linalg.lstsq(a.T @ a, a.T @ b, rcond=None)
    assert equal_to_tolerance(res, res2)

    cov = CovarianceMatrix(c, do_inverse=False)
    res = cov.solve_lstsq(y, jacobian)
    a = sp.linalg.solve_triangular(cov.lower, jacobian.T, lower=True)
    b = sp.linalg.solve_triangular(cov.lower, y, lower=True)
    res2, _, _, _ = np.linalg.lstsq(a.T @ a, a.T @ b, rcond=None)
    assert equal_to_tolerance(res, res2)

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
    num_sensors = 10
    c = np.diag(range(num_sensors))
    cov = CovarianceMatrix(c)

    # 1. ref_idx=None
    # Behavior should be to use the last as a reference
    # Result should be 2 + eye(num_sensors-1)
    c_out = c[:-1, :-1] + c[-1, -1]
    cov_out = cov.resample(ref_idx=None)
    assert equal_to_tolerance(cov_out.cov, c_out)
    assert ~equal_to_tolerance(cov_out.cov, cov.cov)  # make sure the original wasn't resampled

    # 2. ref_idx=integer
    ref_idx = 5
    c_out = sp.linalg.block_diag(c[:ref_idx, :ref_idx], c[ref_idx+1:, ref_idx+1:]) + c[ref_idx, ref_idx]
    cov_out = cov.resample(ref_idx=5)
    assert equal_to_tolerance(cov_out.cov, c_out)

    # 3. ref_idx='Full'
    num_sensors = 4
    # The "full" for 3 sensors should be:
    #  ref_idx = [0, 0, 0, 1, 1, 2]
    #  test_idx = [1, 2, 3, 2, 3, 3]
    a = np.random.rand(num_sensors, num_sensors)
    c = a + a.T
    c_out = resample_covariance_matrix(c,
                                       ref_idx=np.array([0, 0, 0, 1, 1, 2]),
                                       test_idx=np.array([1, 2, 3, 2, 3, 3]))
    cov = CovarianceMatrix(c)
    cov_out = cov.resample(ref_idx='Full')
    assert equal_to_tolerance(cov_out.cov, c_out)

    # 4. ref_idx='full'
    cov_out = cov.resample(ref_idx='full')
    assert equal_to_tolerance(cov_out.cov, c_out)

    # 5. ref_idx_vec and test_idx_vec manually defined
    ref_idx = np.array([0, 0, 1, 2, 2])
    test_idx = np.array([1, 3, 2, 0, 3])
    c_out = resample_covariance_matrix(c, ref_idx=ref_idx, test_idx=test_idx)
    cov_out = cov.resample(ref_idx_vec=ref_idx, test_idx_vec=test_idx)
    assert equal_to_tolerance(cov_out.cov, c_out)


def test_resample_hybrid():
    """
    Resample a covariance matrix from a hybrid PSS sytem

    1. AOA/TDOA
    2. AOA/FDOA
    3. TDOA/FDOA
    4. AOA/TDOA/FDOA
    """
    # TODO: Implement test

def test_multiply():
    """
    Multiply a covariance matrix by a constant.

    Do it with overwrite=True and =False, check the proper behavior of
    the original CovarianceMatrix.
    """
    c = np.array([[1, .1, .2], [.1, 3, .4], [.2, .4, 3]])
    c2 = 2*c

    # Overwrite=True
    cov = CovarianceMatrix(c)
    cov2 = cov.multiply(2, overwrite=True)
    assert equal_to_tolerance(cov.cov, c2)  # make sure the multiply worked
    assert cov2 is None    # make sure no new CovarianceMatrix was returned

    # Overwrite=False
    cov = CovarianceMatrix(c)
    cov2 = cov.multiply(2, overwrite=False)
    assert equal_to_tolerance(cov2.cov, c2)  # make sure the multiply worked
    assert equal_to_tolerance(cov.cov, c)    # make sure the original is untouched

def test_sample():
    """
    Generate a random sample; check dimensions.
    """
    c = np.array([[1, .1, .2], [.1, 3, .4], [.2, .4, 3]])
    cov = CovarianceMatrix(c)

    # Single sample
    # Should have size (num_measurements, ) if no size specified,
    # but (num_measurements, 1) if a single sample is explicitly asked for
    res = cov.sample()
    assert res.shape == (3,)

    res = cov.sample(1)
    assert res.shape == (3, 1)

    # Multiple samples at once
    # Should have size (num_measurements, num_samples)
    for num_samples in np.random.random_integers(low=5, high=100, size=(10, )):
        res = cov.sample(num_samples)
        assert res.shape == (3, num_samples)

def test_sample_shape():
    """
    Generate 1,000,000 random samples; compute covariance matrix estimate.

    Check that the estimated covariance matrix is close to the true covariance.
    """
    num_samples = int(1e6)
    c = np.array([[1, .1, .2], [.1, 3, .4], [.2, .4, 3]])
    cov = CovarianceMatrix(c)
    res = cov.sample(num_samples)

    c_est = np.cov(res)
    assert equal_to_tolerance(c_est, c, tol=5e-2)

def equal_to_tolerance(x, y, tol=1e-6)->bool:
    """
    Shorthand to compare two lists and ensure each entry has an error less than the specified tolerance
    """
    if np.any(np.shape(x) != np.shape(y)): return False
    return all([abs(xx-yy)<tol for xx, yy in zip(np.ravel(x),np.ravel(y))])