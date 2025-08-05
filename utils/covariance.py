import numpy as np
from numpy.linalg import cholesky, lstsq
from scipy.linalg import pinvh, solve_triangular, block_diag
import utils
import warnings
import copy
from numpy import typing as npt


class CovarianceMatrix:
    def __init__(self, cov: npt.ArrayLike, do_cholesky: bool=True, do_inverse: bool=True):
        self._cov = np.asarray(cov)  # Store the covariance matrix; use copy to make sure it's a fresh copy
        self._inv = None
        self._lower = None
        self._do_parse = True
        self._do_cholesky = do_cholesky
        self._do_inverse = do_inverse

    """
    =========================================================
    Properties
    =========================================================
    """
    @property
    def cov(self):
        return self._cov

    @cov.setter
    def cov(self, cov: npt.ArrayLike):
        self._cov = cov.copy()
        self._do_parse = True

    @cov.deleter
    def cov(self):
        self._cov = None
        self._do_parse = True

    @property
    def lower(self):
        self._parse()
        return self._lower

    @lower.setter
    def lower(self, val: npt.ArrayLike):
        self._lower = val.copy()

        # Use the lower to also define the covariance matrix
        self._cov = val @ val.T
        self._inv = None

        # Turn off the matrix inverse, and clear the do_parse flag
        self._do_inverse = False
        self._do_cholesky = True
        self._do_parse = False

    @lower.deleter
    def lower(self):
        self._lower = None
        self._do_parse = True

    @property
    def inv(self):
        self._parse()
        return self._inv

    @inv.setter
    def inv(self, val: npt.ArrayLike):
        self._inv = val.copy()

        # Clear the basic covariance and Cholesky decomposition
        self._cov = None
        self._lower = None

        # Turn off the cholesky flag, and clear the do_parse flag
        self._do_inverse = True
        self._do_cholesky = False
        self._do_parse = False

    @inv.deleter
    def inv(self):
        self._inv = None
        self._do_parse = True

    @property
    def do_cholesky(self):
        return self._do_cholesky

    @do_cholesky.setter
    def do_cholesky(self, do_cholesky: bool):
        if do_cholesky != self._do_cholesky:
            # It's a change, clear some flags to force recalculation
            self._do_cholesky = do_cholesky
            self._lower = None
            self._do_parse = True

    @property
    def do_inverse(self):
        return self._do_inverse

    @do_inverse.setter
    def do_inverse(self, do_inverse: bool):
        if do_inverse != self._do_inverse:
            # It's a change, clear some flags to force recalculation
            self._do_inverse = do_inverse
            self._inv = None
            self._do_parse = True  # Clear the do_parse flag, to make sure it gets recomputed

    """
    =========================================================
    Administrative Functions
    
    copy, parsing the object (for inverse and Cholesky), etc.
    =========================================================
    """
    def copy(self) -> 'CovarianceMatrix':
        """
        Make a copy of self and return it.
        """

        # We must typically be careful here, but CovarianceMatrix doesn't store any object references or large datasets
        # that must be shared, so deepcopy should be safe.
        return copy.deepcopy(self)

    def _parse(self):
        """
        Parse a covariance matrix to generate the lower cholesky decomposition and matrix inverse
        """
        if not self._do_parse:
            # Already parsed, don't waste time
            return

        assert self._cov is not None, "Covariance matrix not initialized."

        # Check for scalar inputs
        if np.isscalar(self._cov) or self._cov.size <= 1:
            # Cholesky decomposition doesn't make sense, but the matrix inverse does (and it's trivial)
            self._do_cholesky = False
            self._do_inverse = True

            self._lower = None
            self._inv = 1/self._cov

            # Clear the do_parse flag
            self._do_parse = False
            return

        # Check for bad inputs
        if not self._do_cholesky and not self._do_inverse:
            warnings.warn("Covariance matrix flags are preventing both Cholesky decomposition and matrix inversion.")

        # Make sure the input is invertible
        self._cov = utils.ensure_invertible(self._cov)

        # Perform a cholesky decomposition and matrix inversion
        if self._do_cholesky:
            self._lower = cholesky(self._cov)
        else:
            self._lower = None

        if self._do_inverse:
            self._inv = np.real(pinvh(self._cov))
        else:
            self._inv = None

        # Clear the do_parse flag
        self._do_parse = False

        return

    def solve_aca(self, a: npt.ArrayLike):
        """
        Solve the matrix problem res = A @ C^{-1} @ A.T

        If self._inv is defined, this will be computed directly. If it is not defined, this will be
        computed via the Cholesky decomposition.

        """

        # Make sure we've parsed the covariance
        self._parse()

        # Check for the matrix inverse
        if self._do_inverse:
            if np.isscalar(self._inv) or np.size(self._inv) == 1:
                val = self._inv * a @ a.T
            else:
                val = a @ self._inv @ a.T

        # Check for Cholesky decomposition
        elif self._do_cholesky:
            c = solve_triangular(self._lower, a.T, lower=True)
            if c.ndim == 1:
                # It's a 1D vector, just take the sum of the square of each element
                val = np.sum(c**2)
            else:
                val = c.T @ c

        else:
            # If we've gotten here, something is wrong
            raise RuntimeError

        return val

    def solve_acb(self, a: npt.ArrayLike, b: npt.ArrayLike):
        """
        Solve the matrix problem res = A @ C^{-1} @ B

        If self._inv is defined, this will be computed directly. If it is not defined, this will be
        computed via the Cholesky decomposition.

        """

        # Make sure we've parsed the covariance
        self._parse()

        # Check for the matrix inverse
        if self._do_inverse:
            if np.isscalar(self._inv) or np.size(self._inv) == 1:
                val = self._inv * a @ b
            else:
                val = a @ self._inv @ b

        # Check for Cholesky decomposition
        elif self._do_cholesky:
            x1 = solve_triangular(self._lower, a.T, lower=True)
            x2 = solve_triangular(self._lower, b, lower=True)

            val = x1.T @ x2

        else:
            # If we've gotten here, something is wrong
            raise RuntimeError

        return val

    def solve_lstsq(self, y: npt.ArrayLike, jacobian: npt.ArrayLike):
        """
        Solve a linear equation of the form for x:
            J@C^{-1}@y = J@C^{-1}@J.T @ x

        If the matrix is pre_inverted, we'll solve it directly. Otherwise, if the matrix has been decomposed, we'll
        use forward substitution to precompute the two components before solving.

        Nicholas O'Donoughue
        28 February 2025
        """

        self._parse()  # Make sure the matrix has been parsed

        if self._do_inverse:
            # Direct computation approach
            jc = jacobian @ self._inv
            jcj = jc @ jacobian.T
            jcy = jc @ y
            val, _, _, _ = lstsq(jcj, jcy, rcond=None)
        elif self._do_cholesky:
            # Using Cholesky decomposition:
            # [J @ C^{-1} @ J.T]^{-1} @ J @ C^{-1} @ y is
            # rewritten
            # [ a.T @ a ] ^{-1} @ a.T @ b
            # where a and b are solved via forward substitution
            # from the lower triangular matrix L.
            #   L @ a = J.T
            #   L @ b = y
            a = solve_triangular(self._lower, jacobian.T, lower=True)
            b = solve_triangular(self._lower, y, lower=True)
            # Then, we solve the system
            #  (a.T @ a) @ delta_x = a.T @ b
            val, _, _, _ = lstsq(a.T @ a, a.T @ b, rcond=None)
        else:
            # We shouldn't get here
            raise RuntimeError

        return val

    def resample(self, ref_idx=None, ref_idx_vec: npt.ArrayLike = None, test_idx_vec: npt.ArrayLike = None) \
            -> 'CovarianceMatrix':
        """
        Resample the covariance matrix. Users may specify the reference index in one of three ways:

        resample(self, ref_idx=None)
            The function utils.parse_reference_sensor will be used to generate paired reference and test indices using
            the default common reference sensor (the N-1th sensor).

        resample(self, ref_idx: int)
            The function utils.parse_reference_sensor will be used to generate paired reference and test indices using
            the specified sensor number as a common reference.

        resample(self, 'Full')
            The function utils.parse_reference_sensor will be used to generate paired reference and test indices using
            the full set of possible sensor pairs.

        resample(self, ref_idx_vec: numpy.ndarray, test_idx_vec: numpy.ndarray)
            The specified reference and test index vectors will be used directly to resample the covariance matrix.
            Both must be 1D numpy.ndarray vectors populated with integers.

        """
        assert self._cov is not None, f"Covariance matrix not initialized."

        # Parse the inputs
        if ref_idx_vec is not None and test_idx_vec is not None:
            # Make sure they're both 1D arrays
            num_ref, dim1 = utils.safe_2d_shape(ref_idx_vec)
            num_test, dim2 = utils.safe_2d_shape(test_idx_vec)

            assert num_ref == num_test, 'Inputs ref_idx_vec and test_idx_vec must match shape.'
            assert dim1 == dim2 == 1, 'Inputs ref_idx_vec and test_idx_vec must be vectors.'
        else:
            # Only the reference index was provided; use parse_reference sensor to generate paired test and reference
            # indices. That function will handle testing for valid inputs.
            test_idx_vec, ref_idx_vec = utils.parse_reference_sensor(ref_idx=ref_idx, num_sensors=self._cov.shape[0])

        new_cov = utils.resample_covariance_matrix(self._cov, test_idx=test_idx_vec, ref_idx=ref_idx_vec)

        return CovarianceMatrix(new_cov)

    def resample_hybrid(self, x_aoa=None, x_tdoa=None, x_fdoa=None, do_2d_aoa=False,
                        tdoa_ref_idx=None, fdoa_ref_idx=None) -> 'CovarianceMatrix':
        """
        Resample a block-diagonal covariance matrix representing AOA, TDOA, and FDOA measurements errors. Original
        matrix size is square with (num_aoa*aoa_dim + num_tdoa + num_fdoa) rows/columns. Output matrix size will be
        square with (num_aoa*aoa_dim + num_tdoa_sensor_pairs + num_fdoa_sensor_pairs) rows/columns.

        The covariance matrix is assumed to have AOA measurements first, then TDOA, then FDOA.

        Ported from MATLAB Code.

        Nicholas O'Donoughue
        21 January 2021

        :param x_aoa: AOA sensor positions
        :param x_tdoa: TDOA sensor positions
        :param x_fdoa: FDOA sensor positions
        :param do_2d_aoa: Boolean flag; if true the number of AOA sensors is doubled (for elevation measurements)
        :param tdoa_ref_idx: Scalar index of reference sensor, or n_dim x n_pair matrix of sensor pairings for TDOA
        :param fdoa_ref_idx: Scalar index of reference sensor, or n_dim x n_pair matrix of sensor pairings for FDOA
        :return cov_out:
        """

        # Parse sensor counts
        _, num_aoa = utils.safe_2d_shape(x_aoa)
        _, num_tdoa = utils.safe_2d_shape(x_tdoa)
        _, num_fdoa = utils.safe_2d_shape(x_fdoa)
        if do_2d_aoa:
            num_aoa = 2 * num_aoa

        # First, we generate the test and reference index vectors
        test_idx_vec_aoa = np.arange(num_aoa)
        ref_idx_vec_aoa = np.nan * np.ones((num_aoa,))
        test_idx_vec_tdoa, ref_idx_vec_tdoa = utils.parse_reference_sensor(tdoa_ref_idx, num_tdoa)
        test_idx_vec_fdoa, ref_idx_vec_fdoa = utils.parse_reference_sensor(fdoa_ref_idx, num_fdoa)

        # Second, we assemble them into a single vector
        test_idx_vec = np.concatenate((test_idx_vec_aoa, num_aoa + test_idx_vec_tdoa,
                                       num_aoa + num_tdoa + test_idx_vec_fdoa), axis=0)
        ref_idx_vec = np.concatenate((ref_idx_vec_aoa, num_aoa + ref_idx_vec_tdoa,
                                      num_aoa + num_tdoa + ref_idx_vec_fdoa), axis=0)

        # Finally, call the generic resampler and return the result
        return self.resample(test_idx_vec=test_idx_vec, ref_idx_vec=ref_idx_vec)

    def multiply(self, val, overwrite=True):
        """
        Multiply the covariance matrix by a given value. val must be a finite number.

        Nicholas O'Donoughue
        28 February 2025
        """

        assert np.isfinite(val) and (np.isscalar(val) or np.size(val) <= 1), \
            'Input to the CovarianceMatrix multiply command must be a finite scalar.'

        if not overwrite:
            # Make a new instance of self
            obj = self.copy()

            # Perform the multiplication on the new instance
            obj.multiply(val)

            # Return an object handle
            return obj

        if self._cov is not None:
            self._cov = self._cov * val

        if self._lower is not None:
            # The Cholesky decomposition is C = L @ L.T, so we apply the square root of the value to L
            self._lower = self._lower * np.sqrt(val)

        if self._inv is not None:
            self._inv = self._inv / val

    @classmethod
    def block_diagonal(cls, *args: 'CovarianceMatrix') -> 'CovarianceMatrix':
        c = block_diag(*[x.cov for x in args])
        return CovarianceMatrix(c)
