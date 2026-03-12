import copy
import numpy as np
from numpy import typing as npt
from numpy.linalg import cholesky, lstsq
from scipy.linalg import pinvh, solve_triangular, block_diag
from typing import Self
import warnings

from . import parse_reference_sensor, resample_covariance_matrix, print_matrix

class CovarianceMatrix:
    # Covariance Matrix and it's Decompositions
    _cov: npt.ArrayLike | None = None
    _inv: npt.ArrayLike | None = None
    _lower: npt.ArrayLike | None = None
    _eigenvalues: npt.ArrayLike | None = None
    _eigenvectors: npt.ArrayLike | None = None
    _size: int = 0

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
            if np.ndim(cov) == 1:
                cov = np.diag(cov)
            self.cov = np.asarray(cov, dtype=float)  # Store the covariance matrix; use copy to make sure it's a fresh copy
            self._inv = None
            self._lower = None
            self._do_cholesky = do_cholesky
            self._do_inverse = do_inverse

    def __str__(self):
        return f"CovarianceMatrix: {np.matrix(self.cov)}"

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
        self._size = cov.shape[0]
        self._do_parse = True

    @cov.deleter
    def cov(self):
        self._cov = None
        self._size = 0
        self._do_parse = True

    @property
    def size(self)-> int:
        return self._size

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
    def eigenvalues(self)-> npt.NDArray:
        if self._eigenvalues is None:
            # We need to parse the covariance matrix
            self._do_parse = True
            self._parse()
        return self._eigenvalues

    @property
    def eigenvectors(self)-> npt.NDArray:
        if self._eigenvalues is None:
            # We need to parse the covariance matrix
            self._do_parse = True
            self._parse()
        return self._eigenvectors

    """
    =========================================================
    Administrative Functions
    
    copy, parsing the object (for inverse and Cholesky), etc.
    =========================================================
    """
    def copy(self) -> Self:
        """
        Make a copy of self and return it.
        """

        # We must typically be careful here, but CovarianceMatrix doesn't store any object references or large datasets
        # that must be shared, so deepcopy should be safe.
        return copy.deepcopy(self)

    def _ensure_invertible(self, tolerance: float=1e-10):
        """
        Check the eigenvalues and ensure that they are all >= a small value (tolerance), to ensure that it can be
        inverted.

        If any of the eigenvalues are too small, then a diagonal loading term is applied to ensure that the matrix is
        positive definite.

        Ported from MATLAB code.

        Nicholas O'Donoughue
        5 Sept 2021

        :param tolerance: numerical precision term (the smallest eigenvalue must be >= tolerance) [Default = 1e-10]
        """

        # Error-prevention -- this loop will fail if self._cov is an integer type, let's force it to be a float
        self._cov = np.asarray(self._cov, dtype=float)

        # Eigen-decomposition
        lam, v = np.linalg.eigh(self.cov)

        # Initialize the diagonal loading term
        d = tolerance * np.eye(self.size)

        # Repeat until the smallest eigenvalue is larger than tolerance
        while np.amin(lam) < tolerance:
            # Add the diagonal loading term
            this_cov = self.cov + d

            # Reexamine the eigenvalue
            lam, v = np.linalg.eigh(this_cov)

            # Increase the amount of diagonal loading (for the next iteration) by an order of magnitude
            d *= 2.0

        # When we're done, store the result
        self._eigenvalues = lam
        self._eigenvectors = v
        self._cov += d # add the diagonal loading term to the stored matrix

        return

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

            # Set the eigenvalues and eigenvectors
            self._eigenvalues = np.abs(self._cov)
            self._eigenvectors = self._cov / self._eigenvalues

        # Check for bad inputs
        elif not self._do_cholesky and not self._do_inverse:
            warnings.warn("Covariance matrix flags are preventing both Cholesky decomposition and matrix inversion.")
        else:
            # Make sure the input is invertible
            # This also generates and stores the eigenvalues and eigenvectors
            self._ensure_invertible()

            # Perform a cholesky decomposition and matrix inversion
            if self._do_cholesky:
                self._lower = cholesky(self._cov)
            else:
                self._lower = None

            if self._do_inverse:
                self._inv = pinvh(self._cov)
            else:
                self._inv = None

        # Clear the do_parse flag
        self._do_parse = False

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
            lower = self._lower
            # Using vectorized triangular solve for all columns at once
            # if do_2d = True, then the operation is (n, n) and (*, n, m) -> (*, n, m)

            if do_2d:
                # shapes are (n, n) and (*, n, m) -> (*, n, m)
                c = solve_triangular(lower, np.transpose(a_flat, (0, 2, 1)), lower=True)
                # (*, m, n) x (*, n, m) = (*, m, m)
                val = np.conj(np.transpose(c, (0, 2, 1))) @ c
            else:
                # shapes are (n, n) and (n, *) -> (n, *)
                c = solve_triangular(lower, a_flat.T, lower=True)
                # Energy term = sum(|c|^2)
                val = np.sum(np.abs(c)**2, axis=0)
        else:
            raise RuntimeError("Unable to solve; covariance matrix not parse with Cholesky or inverse.")

        val = np.reshape(val, batch_shape)
        return val

    def solve_acb(self, a: npt.NDArray[np.float64], b: npt.NDArray[np.float64])-> npt.NDArray[np.float64]:
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

    def resample(self, ref_idx=None, ref_idx_vec: npt.ArrayLike = None, test_idx_vec: npt.ArrayLike = None)-> Self:
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
            shp = np.shape(ref_idx_vec)
            num_ref = shp[0] if len(shp) > 0 else 1
            dim1 = shp[1] if len(shp) > 1 else 1
            shp = np.shape(test_idx_vec)
            num_test = shp[0] if len(shp) > 0 else 1
            dim2 = shp[1] if len(shp) > 1 else 1
            assert num_ref == num_test, 'Inputs ref_idx_vec and test_idx_vec must match shape.'
            assert dim1 == dim2 == 1, 'Inputs ref_idx_vec and test_idx_vec must be vectors.'
        else:
            # Only the reference index was provided; use parse_reference sensor to generate paired test and reference
            # indices. That function will handle testing for valid inputs.
            test_idx_vec, ref_idx_vec = parse_reference_sensor(ref_idx=ref_idx, num_sensors=self._cov.shape[0])

        new_cov = resample_covariance_matrix(self._cov, test_idx=test_idx_vec, ref_idx=ref_idx_vec)

        return CovarianceMatrix(new_cov)

    def resample_hybrid(self, num_aoa: int=0, num_tdoa: int=0, num_fdoa: int=0,
                        tdoa_ref_idx=None, fdoa_ref_idx=None) -> Self:
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
        if num_tdoa > 0:
            test_idx_vec_tdoa, ref_idx_vec_tdoa = parse_reference_sensor(tdoa_ref_idx, num_tdoa)
        else:
            test_idx_vec_tdoa = np.array([])
            ref_idx_vec_tdoa = np.array([])
        if num_fdoa > 0:
            test_idx_vec_fdoa, ref_idx_vec_fdoa = parse_reference_sensor(fdoa_ref_idx, num_fdoa)
        else:
            test_idx_vec_fdoa = np.array([])
            ref_idx_vec_fdoa = np.array([])

        # Second, we assemble them into a single vector
        test_idx_vec = np.concatenate((test_idx_vec_aoa, num_aoa + test_idx_vec_tdoa,
                                       num_aoa + num_tdoa + test_idx_vec_fdoa), axis=0)
        ref_idx_vec = np.concatenate((ref_idx_vec_aoa, num_aoa + ref_idx_vec_tdoa,
                                      num_aoa + num_tdoa + ref_idx_vec_fdoa), axis=0)

        # Finally, call the generic resampler and return the result
        return self.resample(test_idx_vec=test_idx_vec, ref_idx_vec=ref_idx_vec)

    def multiply(self, val, overwrite=True)-> Self | None:
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

    def sample(self, num_samples: int = None, mean_vec:npt.NDArray[np.float64] | None = None) -> npt.ArrayLike:
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
            mean_vec_dims = np.shape(mean_vec)
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
    def block_diagonal(cls, *args: Self | npt.ArrayLike) -> Self:
        if len(args) == 1:
            return CovarianceMatrix(args[0])

        arrs = []
        for arg in args:
            if isinstance(arg, CovarianceMatrix): arrs.append(arg.cov)
            else: arrs.append(np.array(arg))

        c = block_diag(arrs[0], *arrs[1:])
        return CovarianceMatrix(c)
