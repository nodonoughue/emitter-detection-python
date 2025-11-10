import copy
import numpy as np
from numpy import typing as npt
from numpy.linalg import cholesky, lstsq
from scipy.linalg import pinvh, solve_triangular, block_diag
import warnings

from . import ensure_invertible, parse_reference_sensor, resample_covariance_matrix, safe_2d_shape


class CovarianceMatrix:
    # Covariance Matrix and it's Decompositions
    _cov: npt.ArrayLike | None = None
    _inv: npt.ArrayLike | None = None
    _lower: npt.ArrayLike | None = None
    _eigenvalues: npt.ArrayLike | None = None
    _eigenvectors: npt.ArrayLike | None = None

    # Flags
    _do_parse: bool = True          # A parse is needed
    _do_cholesky: bool = True       # Control whether _lower will be filled
    _do_inverse: bool = True        # Control whether _inv will be filled

    def __init__(self, cov: npt.ArrayLike, do_cholesky: bool=True, do_inverse: bool=True):
        if isinstance(cov, CovarianceMatrix):
            # Copy it instead (this is a deepcopy), then set all the
            # attributes of the current object to point to those of the copy.
            new_cov = cov.copy()
            for key, value in new_cov.__dict__.items():
                setattr(self, key, value)
        else:
            self._cov = np.asarray(cov)  # Store the covariance matrix; use copy to make sure it's a fresh copy
            self._inv = None
            self._lower = None
            self._do_parse = True
            self._do_cholesky = do_cholesky
            self._do_inverse = do_inverse

    def __str__(self):
        np.set_printoptions(precision=4, suppress=False)
        return f"CovarianceMatrix: {self.cov}"

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

    @property
    def size(self)-> int:
        if self._cov is None:
            return 0
        else:
            return np.shape(self._cov)[0]

    @property
    def eigenvalues(self)-> npt.NDArray:
        self._parse_eig()
        return self._eigenvalues

    @property
    def eigenvectors(self)-> npt.NDArray:
        self._parse_eig()
        return self._eigenvectors

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
        self._cov = ensure_invertible(self._cov)

        # Perform a cholesky decomposition and matrix inversion
        if self._do_cholesky:
            self._lower = cholesky(self._cov)
        else:
            self._lower = None

        if self._do_inverse:
            self._inv = pinvh(self._cov)
        else:
            self._inv = None

        # Clear the eigenvectors and eigenvalues
        self._eigenvalues = None
        self._eigenvectors = None

        # Clear the do_parse flag
        self._do_parse = False

        return

    def _parse_eig(self):
        """
        Compute the eigenvectors and eigenvalues of the covariance matrix; store them to speed up repeated calls.
        """
        self._parse() # If a change happened to ._cov, ._inv, or ._lower, then this flag is set. Let's resolve it

        if self._eigenvalues is not None and self._eigenvectors is not None:
            # They were already computed
            return

        if self._cov is None:
            # There is no covariance matrix; it can't have eigenvectors/eigenvalues
            self._eigenvalues = None
            self._eigenvectors = None
            return

        # Parse the covariance matrix
        lam, v = np.linalg.eigh(self.cov)
        self._eigenvalues = lam
        self._eigenvectors = v
        return

    def solve_aca(self, a: npt.ArrayLike, do_2d: bool = False):
        """
        Solve the matrix problem res = A @ C^{-1} @ A.T

        If self._inv is defined, this will be computed directly. If it is not defined, this will be
        computed via the Cholesky decomposition.

        Inputs
        ------
        a : array_like
            May be:
              - 1D vector of shape (n, )
              - 2D array of shape (m, n) where each row is a separate vector
              - ND array of shape (*, n) for batch processing
            If the optional argument do_2d is True, then valid sizes are:
              - 2D array of shape (m, n) where the matrix is processed as one input
              - ND array of shape (*, m, n) for batch processing
        do_2d : bool
            If True, then a will be treated as a 2D array. Default = False

        Returns
        -------
        res : ndarray
            Scalar if input is 1D.
            If input is 2D, then res is an array of shape (m, ) if do_2d is False, or (m, m) if do_2d is True.
            If input is ND, then res is an array of shape (*, ) if do_2d is False, or (m, m, *) if do_2d is True.
        """

        self._parse()
        n = self.size
        a = np.asarray(a)
        if a.shape[-1] != n:
            raise ValueError(f"Input last dimension {a.shape[-1]} must match covariance size.")

        # Collapse all non-last dimensions into a batch axis
        if do_2d:
            batch_shape = list(a.shape[:-2])
            batch_shape.extend([a.shape[-2], a.shape[-2]])
            a_flat = a.reshape(-1, a.shape[-2], n)
        else:
            batch_shape = a.shape[:-1]
            a_flat = a.reshape(-1, n)

        if self.do_inverse:
            inv = self._inv
            if np.isscalar(inv) or np.size(inv) == 1:
                # a_flat should have shape (*, 1) if do_2d=False, or (*, m, 1) if do_2d=True
                if do_2d:
                    # We need to do a matrix multiplication of the last two dimensions, then multiply by the inverse
                    val = inv * np.expand_dims(a_flat, -1) @ np.conj(np.expand_dims(a_flat, -2)) # shape (*, m, m)
                else:
                    # Result should be
                    val = np.sum(np.abs(a_flat)**2, axis=1) * inv # shape (*, )
            else:
                # a_flat should have shape (*, n) if do_2d=False or (*, m, n) if do_2d=True
                if do_2d:
                    # (num_cases, m, n) x (n, n) -> (num_cases, m, n)
                    tmp = a_flat @ inv
                    # (num_cases, m, n) x (num_cases, n, m) -> (num_cases, m, m)
                    # use np.conj to be complex-safe
                    val = np.conj(tmp) @ np.swapaxes(a_flat, axis1=1, axis2=2)
                else:
                    # (num_cases, n) x (n, n) -> (num_cases, n)
                    tmp = a_flat @ inv
                    # each row dot with itself (complex‑safe)
                    val = np.sum(np.conj(tmp) * a_flat, axis=1)
        elif self.do_cholesky:
            L = self._lower
            # Using vectorized triangular solve for all columns at once

            # if do_2d = True, then the operation is (n, n) and (*, n, m) -> (*, n, m)

            if do_2d:
                # shapes are (n, n) and (*, n, m) -> (*, n, m)
                c = solve_triangular(L, np.transpose(a_flat, (0, 2, 1)), lower=True)
                # (*, m, n) x (*, n, m) = (*, m, m)
                val = np.conj(np.transpose(c, (0, 2, 1))) @ c
            else:
                # shapes are (n, n) and (n, *) -> (n, *)
                c = solve_triangular(L, a_flat.T, lower=True)
                # Energy term = sum(|c|^2)
                val = np.sum(np.abs(c)**2, axis=0)
        else:
            raise RuntimeError("Unable to solve; covariance matrix not parse with Cholesky or inverse.")

        val = np.reshape(val, batch_shape)
        return val

        # # Check for multi-dimensional inputs
        # if np.ndim(a) > 2 or (np.ndim(a)==2 and np.shape(a)[1] != self.size):
        #     # Either there are 3+ dimensions, or it's 2D but the second dimension does not match the expected size
        #     out_shp = np.shape(a)[1:]
        #     if np.prod(out_shp) == 1: out_shp = []
        #
        #     a = np.reshape(a, (self.size, -1)).T
        #
        #     # Assume the first dimension follows the size of self, and any remaining dimensions are
        #     # parallel cases
        #     val = np.array([self.solve_aca(aa) for aa in a]).reshape(out_shp)
        #     return val
        #
        # # Make sure we trim any dimensions after the second; we're doing matrix math here.
        # if np.ndim(a) > 2:
        #     a = np.squeeze(a, axis=tuple(range(2, np.ndim(a))))
        #
        # # Check the size of a
        # if np.size(a)==self.size and (np.shape(a)[0]==self.size or np.shape(a)[-1]==self.size):
        #     # It's a 1D array, but may be stored as a 2D array. Let's squeeze it, for convenience
        #     a = np.squeeze(a)
        # elif np.ndim(a)==1:
        #     # Make sure a is a vector of the right size
        #     assert np.shape(a)[0] == self.size, f"Input vector a must be the same size as the covariance matrix."
        # else:
        #     # Make sure the second dimension of a matches
        #     assert np.shape(a)[1] == self.size, f"Input matrix a must have second dimension the same size as the covariance matrix."
        #
        # # Make sure we've parsed the covariance
        # self._parse()
        #
        # # Check for the matrix inverse
        # if self._do_inverse:
        #     if np.isscalar(self._inv) or np.size(self._inv) == 1:
        #         val = self._inv * a @ np.conj(a.T)
        #     else:
        #         val = a @ self._inv @ np.conj(a.T)
        #
        # # Check for Cholesky decomposition
        # elif self._do_cholesky:
        #     c = solve_triangular(self._lower, a.T, lower=True)
        #     if c.ndim == 1:
        #         # It's a 1D vector, just take the sum of the square of each element
        #         val = np.sum(np.abs(c)**2)
        #     else:
        #         val = np.conj(c.T) @ c
        #
        # else:
        #     # If we've gotten here, something is wrong
        #     raise RuntimeError
        #
        # return val

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
            The function parse_reference_sensor will be used to generate paired reference and test indices using
            the default common reference sensor (the N-1th sensor).

        resample(self, ref_idx: int)
            The function parse_reference_sensor will be used to generate paired reference and test indices using
            the specified sensor number as a common reference.

        resample(self, 'Full')
            The function parse_reference_sensor will be used to generate paired reference and test indices using
            the full set of possible sensor pairs.

        resample(self, ref_idx_vec: numpy.ndarray, test_idx_vec: numpy.ndarray)
            The specified reference and test index vectors will be used directly to resample the covariance matrix.
            Both must be 1D numpy.ndarray vectors populated with integers.

        """
        assert self._cov is not None, f"Covariance matrix not initialized."

        # Parse the inputs
        if ref_idx_vec is not None and test_idx_vec is not None:
            # Make sure they're both 1D arrays
            num_ref, dim1 = safe_2d_shape(ref_idx_vec)
            num_test, dim2 = safe_2d_shape(test_idx_vec)

            assert num_ref == num_test, 'Inputs ref_idx_vec and test_idx_vec must match shape.'
            assert dim1 == dim2 == 1, 'Inputs ref_idx_vec and test_idx_vec must be vectors.'
        else:
            # Only the reference index was provided; use parse_reference sensor to generate paired test and reference
            # indices. That function will handle testing for valid inputs.
            test_idx_vec, ref_idx_vec = parse_reference_sensor(ref_idx=ref_idx, num_sensors=self._cov.shape[0])

        new_cov = resample_covariance_matrix(self._cov, test_idx=test_idx_vec, ref_idx=ref_idx_vec)

        return CovarianceMatrix(new_cov)

    def resample_hybrid(self, num_aoa: int=0, num_tdoa: int=None, num_fdoa: int=None,
                        tdoa_ref_idx=None, fdoa_ref_idx=None) -> 'CovarianceMatrix':
        """
        Resample a block-diagonal covariance matrix representing AOA, TDOA, and FDOA measurements errors. Original
        matrix size is square with (num_aoa*aoa_dim + num_tdoa + num_fdoa) rows/columns. Output matrix size will be
        square with (num_aoa*aoa_dim + num_tdoa_sensor_pairs + num_fdoa_sensor_pairs) rows/columns.

        The covariance matrix is assumed to have AOA measurements first, then TDOA, then FDOA.

        Ported from MATLAB Code.

        Nicholas O'Donoughue
        21 January 2021

        :param num_aoa: Number of AOA measurements
        :param num_tdoa: Number of TDOA sensors
        :param num_fdoa: Number of FDOA sensors
        :param tdoa_ref_idx: Scalar index of reference sensor, or n_dim x n_pair matrix of sensor pairings for TDOA
        :param fdoa_ref_idx: Scalar index of reference sensor, or n_dim x n_pair matrix of sensor pairings for FDOA
        :return cov_out:
        """

        # First, we generate the test and reference index vectors
        test_idx_vec_aoa = np.arange(num_aoa)
        ref_idx_vec_aoa = np.nan * np.ones((num_aoa,))
        test_idx_vec_tdoa, ref_idx_vec_tdoa = parse_reference_sensor(tdoa_ref_idx, num_tdoa)
        test_idx_vec_fdoa, ref_idx_vec_fdoa = parse_reference_sensor(fdoa_ref_idx, num_fdoa)

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

        return None

    def sample(self, num_samples: int = None, mean_vec:npt.ArrayLike = None) -> npt.ArrayLike:
        """
        Generate a random sample from the covariance matrix.

        Nicholas O'Donoughue
        16 September 2025

        :param num_samples: The number of independent samples to generate (default=1)
        :param mean_vec: A numpy array of shape (num_measurements, ) containing the mean vector to apply
        :return: A numpy array of shape (num_measurements, num_samples) containing independent samples of the
                 (num_measurements, ) random vector defined by this covariance matrix. If num_samples is not provided,
                 then the response is a 1D vector with shape (num_measurements, )
        """

        self._parse()  # Make sure the matrix has been parsed
        num_measurements = self._cov.shape[0]
        if num_samples is None:
            num_samples = 1
            do_squeeze = True
        else:
            do_squeeze = False

        if self._cov.size == 1:
            # The covariance matrix is scalar; let's do it manually
            x = np.random.randn(1, num_samples) * np.sqrt(self._cov)
        elif self.do_cholesky:
            # We already did Cholesky decomposition; use it directly to generate colored noise
            x = self._lower @ np.random.randn(num_measurements, num_samples) # shape (num_measurements, num_samples)
        else:
            # Use the builtin multivariate_normal generator with self._cov
            # This one will return the result with shape (num_samples, num_measurements), so we need to transpose
            x = np.transpose(np.random.multivariate_normal(mean=np.zeros(num_measurements),
                                                           cov=self._cov,
                                                           size=num_samples)) # shape (num_measurements, num_samples)

        if mean_vec is not None:
            # Make sure that mean_vec is a 1d array
            mean_vec_dims = safe_2d_shape(mean_vec)
            if mean_vec_dims[1] > 1:
                warnings.warn("Input mean_vec is not a 1D array; it will be ignored.")
            elif num_measurements != mean_vec_dims[0]:
                warnings.warn("Input mean_vec size does not match the covariance matrix; it will be ignored.")
            else:
                # Add a new axis to the end of mean_vec, to ensure proper broadcasting to y
                x = x + mean_vec[:, np.newaxis] # shape (num_measurements, num_samples) -- or -- (num_samples, )

        if do_squeeze:
            # The user didn't specify a num_samples variable, so let's remove the second dimension
            x = x[:, 0]

        return x

    @classmethod
    def block_diagonal(cls, *args: 'CovarianceMatrix') -> 'CovarianceMatrix':
        c = block_diag(*[x.cov for x in args])
        return CovarianceMatrix(c)
