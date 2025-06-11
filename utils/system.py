import numpy

import utils
from utils.covariance import CovarianceMatrix


class PassiveSurveillanceSystem:
    cov: CovarianceMatrix
    pos: numpy.ndarray
    num_sensors: numpy.floating
    num_dim: numpy.floating
    num_measurements: numpy.floating

    def __init__(self, x: numpy.ndarray, cov: CovarianceMatrix):
        self.pos = x
        self.cov = cov

        num_dim, num_sensors = utils.safe_2d_shape(x)
        self.num_sensors = num_sensors
        self.num_dim = num_dim


class DifferencePSS(PassiveSurveillanceSystem):
    _ref_idx: numpy.floating or None = None  # need a type
    _cov_resample: CovarianceMatrix or None # resampled covariance matrix
    _cov_raw: CovarianceMatrix or None
    _do_resample: bool = True

    def __init__(self, x: numpy.ndarray, cov: CovarianceMatrix, ref_idx):
        (super().__init__(x, cov))
        self._cov_raw = cov
        self._ref_idx = ref_idx

        self.resample()

    @property
    def cov(self):
        return self._cov_raw

    @cov.setter
    def cov(self, cov: CovarianceMatrix):
        self._cov_raw = cov.copy()
        self._do_resample = True

    @cov.deleter
    def cov(self):
        self._cov_raw = None
        self._do_resample = True

    @property
    def ref_idx(self):
        return self._ref_idx

    @ref_idx.setter
    def ref_idx(self, idx):
        self._ref_idx = idx
        self._do_resample = True

    @ref_idx.deleter
    def ref_idx(self):
        self._ref_idx = None
        self._do_resample = True

    @property
    def cov_r(self):
        self.resample()
        return self._cov_resample

    @cov_r.setter
    def cov_r(self, _):
        raise TypeError('Setting the resampled covariance matrix directly is not permitted.')

    @cov_r.deleter
    def cov_r(self):
        self._cov_resample = None
        self._do_resample = True

    def resample(self):
        if self._do_resample:
            self._cov_resample = self._cov_raw.resample(ref_idx=self._ref_idx)
        self._do_resample = False