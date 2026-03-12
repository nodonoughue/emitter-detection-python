"""
Tests for tracker/association.py.

Note: NNAssociator.associate() contains an active print() statement (line 475)
that will produce output during tests involving that class.
"""
import numpy as np
import pytest
from scipy.stats import chi2

from ewgeo.tracker.association import (
    Hypothesis,
    MissedDetectionHypothesis,
    GMMHypothesis,
    NNAssociator,
    GNNAssociator,
    PDAAssociator,
)
from ewgeo.tracker.measurement import Measurement
from ewgeo.tracker.states import State
from ewgeo.tracker.track import Track
from ewgeo.tracker.transition import ConstantVelocityMotionModel
from ewgeo.utils.covariance import CovarianceMatrix


def equal_to_tolerance(x, y, tol=1e-6):
    return np.all(np.fabs(np.array(x) - np.array(y)) < tol)


# ---------------------------------------------------------------------------
# Minimal sensor stub
# ---------------------------------------------------------------------------

class _MockSensor:
    """1D direct-position sensor: z = x."""
    num_dim = 1
    num_measurements = 1
    cov = CovarianceMatrix(np.array([[1.0]]))

    def measurement(self, x_source, v_source=None, **kwargs):
        return np.atleast_1d(x_source).astype(float)

    def jacobian(self, x_source, v_source=None, **kwargs):
        # shape (num_dim, num_measurements) = (1, 1)
        return np.array([[1.0]])


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_cv_model():
    return ConstantVelocityMotionModel(num_dims=1, process_covar=1.0)


def make_track(x, vx, t=0.0, track_id='T0'):
    model = make_cv_model()
    state = State(model.state_space, time=t,
                  state=np.array([x, vx], dtype=float),
                  covar=CovarianceMatrix(np.eye(2)))
    return Track(initial_state=state, track_id=track_id)


def make_measurement(x, t=1.0):
    return Measurement(time=t, sensor=_MockSensor(), zeta=np.array([x], dtype=float))


# ---------------------------------------------------------------------------
# Reference values for the standard test case
# (track: x=10, vx=5, P=I; CV 1D dt=1, process_covar=1; sensor R=[[1]])
#
# F = [[1,1],[0,1]]; Q = [[0.25,0.5],[0.5,1.0]]
# P_pred = F @ I @ F^T + Q = [[2.25, 1.5], [1.5, 2.0]]
# predicted state at t=1: [15, 5]
# H = [[1, 0]]
# S = H @ P_pred @ H^T + R = 2.25 + 1.0 = 3.25
# For measurement at x=14: innovation = [-1], distance = 1/3.25 ≈ 0.30769
# For measurement at x=40: innovation = [25], distance = 625/3.25 ≈ 192.3
# ---------------------------------------------------------------------------

PREDICTED_X = 15.0
INNOV_COVAR  = 3.25   # scalar S
DIST_NEAR    = 1.0 / INNOV_COVAR          # measurement at x=14
DIST_FAR     = (40.0 - PREDICTED_X)**2 / INNOV_COVAR   # measurement at x=40


# ---------------------------------------------------------------------------
# Hypothesis — basic construction
# ---------------------------------------------------------------------------

def test_hypothesis_stores_track_and_measurement():
    track = make_track(10, 5)
    msmt = make_measurement(14)
    model = make_cv_model()
    h = Hypothesis(track=track, measurement=msmt, motion_model=model)
    assert h.track is track
    assert h.measurement is msmt
    assert h.motion_model is model


def test_hypothesis_is_valid_initially():
    h = Hypothesis(track=make_track(0, 0), measurement=make_measurement(0))
    assert h.is_valid


# ---------------------------------------------------------------------------
# Hypothesis — invalidate
# ---------------------------------------------------------------------------

def test_hypothesis_invalidate_sets_flags():
    h = Hypothesis(track=make_track(0, 0), measurement=make_measurement(0))
    h.invalidate()
    assert not h.is_valid
    assert h._distance == np.inf
    assert h._likelihood == 0.0
    assert h._log_likelihood == -np.inf


# ---------------------------------------------------------------------------
# Hypothesis — compute_gate_size
# ---------------------------------------------------------------------------

def test_hypothesis_gate_size_1d_measurement():
    """Gate size for 1 measurement at p=0.99 equals chi2.ppf(0.99, 1)."""
    h = Hypothesis(track=make_track(0, 0), measurement=make_measurement(0))
    expected = chi2.ppf(0.99, 1)
    assert equal_to_tolerance(h.compute_gate_size(0.99), expected)


def test_hypothesis_gate_size_null_measurement_is_inf():
    h = Hypothesis(track=make_track(0, 0), measurement=None)
    assert h.compute_gate_size(0.99) == np.inf


# ---------------------------------------------------------------------------
# Hypothesis — innovation, innovation_covar, distance
# ---------------------------------------------------------------------------

def test_hypothesis_innovation():
    """innovation = zeta - predicted_measurement."""
    track = make_track(10, 5)
    msmt = make_measurement(14)
    h = Hypothesis(track=track, measurement=msmt, motion_model=make_cv_model())
    # predicted position = 15; innovation = 14 - 15 = -1
    assert equal_to_tolerance(h.innovation, [-1.0])


def test_hypothesis_innovation_covar():
    """S = H P_pred H^T + R = 2.25 + 1.0 = 3.25."""
    track = make_track(10, 5)
    msmt = make_measurement(14)
    h = Hypothesis(track=track, measurement=msmt, motion_model=make_cv_model())
    assert equal_to_tolerance(h.innovation_covar.cov, [[INNOV_COVAR]])


def test_hypothesis_distance_near_measurement():
    """Distance = innovation^T S^{-1} innovation / n_meas."""
    track = make_track(10, 5)
    h = Hypothesis(track=track, measurement=make_measurement(14), motion_model=make_cv_model())
    assert equal_to_tolerance(h.distance, DIST_NEAR)


def test_hypothesis_distance_far_measurement():
    track = make_track(10, 5)
    h = Hypothesis(track=track, measurement=make_measurement(40), motion_model=make_cv_model())
    assert equal_to_tolerance(h.distance, DIST_FAR)


def test_hypothesis_distance_null_measurement_is_inf():
    h = Hypothesis(track=make_track(0, 0), measurement=None)
    assert h.distance == np.inf


# ---------------------------------------------------------------------------
# Hypothesis — apply_distance_gate
# ---------------------------------------------------------------------------

def test_apply_gate_passes_for_near_measurement():
    h = Hypothesis(track=make_track(10, 5), measurement=make_measurement(14),
                   motion_model=make_cv_model())
    h.apply_distance_gate(0.99)
    assert h.is_valid


def test_apply_gate_fails_for_far_measurement():
    h = Hypothesis(track=make_track(10, 5), measurement=make_measurement(40),
                   motion_model=make_cv_model())
    h.apply_distance_gate(0.99)
    assert not h.is_valid


# ---------------------------------------------------------------------------
# Hypothesis — likelihood
# ---------------------------------------------------------------------------

def test_hypothesis_likelihood_is_positive():
    h = Hypothesis(track=make_track(10, 5), measurement=make_measurement(14),
                   motion_model=make_cv_model())
    assert 0 < h.likelihood < 1


def test_hypothesis_likelihood_decreases_with_distance():
    """Closer measurement → higher likelihood."""
    track_near = make_track(10, 5)
    track_far  = make_track(10, 5)
    h_near = Hypothesis(track=track_near, measurement=make_measurement(14),
                        motion_model=make_cv_model())
    h_far  = Hypothesis(track=track_far,  measurement=make_measurement(40),
                        motion_model=make_cv_model())
    assert h_near.likelihood > h_far.likelihood


def test_hypothesis_log_likelihood_consistent():
    h = Hypothesis(track=make_track(10, 5), measurement=make_measurement(14),
                   motion_model=make_cv_model())
    assert equal_to_tolerance(np.exp(h.log_likelihood), h.likelihood)


# ---------------------------------------------------------------------------
# Hypothesis — override_likelihood
# ---------------------------------------------------------------------------

def test_override_likelihood():
    h = Hypothesis(track=make_track(0, 0), measurement=make_measurement(0))
    h.override_likelihood(0.42)
    assert equal_to_tolerance(h.likelihood, 0.42)
    assert equal_to_tolerance(h.log_likelihood, np.log(0.42))


# ---------------------------------------------------------------------------
# Hypothesis — clear_dependent_parameters
# ---------------------------------------------------------------------------

def test_clear_dependent_parameters_resets_cache():
    h = Hypothesis(track=make_track(10, 5), measurement=make_measurement(14),
                   motion_model=make_cv_model())
    _ = h.distance   # populate cache
    h.clear_dependent_parameters()
    assert h._distance is None
    assert h._innov is None
    assert h._innov_covar is None
    assert h.is_valid


# ---------------------------------------------------------------------------
# MissedDetectionHypothesis
# ---------------------------------------------------------------------------

def test_missed_detection_distance_is_fixed():
    track = make_track(10, 5)
    model = make_cv_model()
    h = MissedDetectionHypothesis(track=track, motion_model=model,
                                  sensor=_MockSensor(), distance=0.05, time=1.0)
    assert h.distance == 0.05


def test_missed_detection_likelihood_is_fixed():
    track = make_track(10, 5)
    h = MissedDetectionHypothesis(track=track, motion_model=make_cv_model(),
                                  sensor=_MockSensor(), distance=0.05, time=1.0)
    assert h.likelihood == 0.05


def test_missed_detection_innovation_is_zeros():
    track = make_track(10, 5)
    h = MissedDetectionHypothesis(track=track, motion_model=make_cv_model(),
                                  sensor=_MockSensor(), distance=0.05, time=1.0)
    assert np.all(h.innovation == 0)


# ---------------------------------------------------------------------------
# NNAssociator
# ---------------------------------------------------------------------------

def test_nn_assigns_nearest_measurement():
    """One track, two measurements: NN picks the closer one."""
    track = make_track(10, 5, track_id='T1')
    model = make_cv_model()
    m_near = make_measurement(14)  # closer to predicted x=15
    m_far  = make_measurement(40)  # farther

    assoc = NNAssociator(motion_model=model, gate_probability=0.99)
    hypotheses, unassoc = assoc.associate([track], [m_near, m_far])

    assert track in hypotheses
    assert hypotheses[track].measurement is m_near
    assert m_far in unassoc


def test_nn_empty_measurements_returns_empty_hypotheses():
    track = make_track(10, 5, track_id='T1')
    assoc = NNAssociator(motion_model=make_cv_model(), gate_probability=0.99)
    hypotheses, unassoc = assoc.associate([track], [])
    assert hypotheses == {}
    assert unassoc == []


def test_nn_all_outside_gate_uses_null_hypothesis():
    """When all measurements fail the gate, track gets a MissedDetectionHypothesis."""
    track = make_track(10, 5, track_id='T1')
    m_far = make_measurement(40)   # distance ≈ 192 >> gate ≈ 6.6

    assoc = NNAssociator(motion_model=make_cv_model(), gate_probability=0.99)
    hypotheses, unassoc = assoc.associate([track], [m_far])

    assert isinstance(hypotheses[track], MissedDetectionHypothesis)


# ---------------------------------------------------------------------------
# GNNAssociator
# ---------------------------------------------------------------------------

def test_gnn_empty_inputs_returns_empty():
    assoc = GNNAssociator(motion_model=make_cv_model(), gate_probability=0.99)
    hypotheses, unassoc = assoc.associate([], [])
    assert hypotheses == {}
    assert unassoc == []


def test_gnn_empty_tracks_returns_empty():
    assoc = GNNAssociator(motion_model=make_cv_model(), gate_probability=0.99)
    hypotheses, unassoc = assoc.associate([], [make_measurement(5)])
    assert hypotheses == {}


def test_gnn_assigns_globally_optimal():
    """
    Two tracks near x=0 and x=20; two measurements near x=1 and x=21.
    Optimal assignment: T1→M1, T2→M2 (not cross-assigned).

    gate_probability=0.5 is used so the null-hypothesis cost (1-0.5=0.5) exceeds
    the real-measurement distance (~0.308), ensuring Munkres prefers real assignments.
    The cross-assignments fail the gate (distance ~111-136 >> gate ~0.455).
    """
    t1 = make_track(0,  0, track_id='T1')
    t2 = make_track(20, 0, track_id='T2')
    m1 = make_measurement(1)
    m2 = make_measurement(21)

    assoc = GNNAssociator(motion_model=make_cv_model(), gate_probability=0.5)
    hypotheses, unassoc = assoc.associate([t1, t2], [m1, m2])

    assert hypotheses[t1].measurement is m1
    assert hypotheses[t2].measurement is m2
    assert unassoc == []


# ---------------------------------------------------------------------------
# PDAAssociator
# ---------------------------------------------------------------------------

def test_pda_returns_gmm_hypothesis():
    """PDAAssociator wraps per-track hypotheses in GMMHypothesis."""
    track = make_track(10, 5, track_id='T1')
    m_near = make_measurement(14)

    assoc = PDAAssociator(motion_model=make_cv_model(), gate_probability=0.99,
                          detection_probability=0.9)
    hypotheses, _ = assoc.associate([track], [m_near])

    assert isinstance(hypotheses[track], GMMHypothesis)


def test_pda_likelihoods_sum_to_one():
    """Weights inside the GMMHypothesis are normalized to sum to 1."""
    track = make_track(10, 5, track_id='T1')
    m_near = make_measurement(14)

    assoc = PDAAssociator(motion_model=make_cv_model(), gate_probability=0.99,
                          detection_probability=0.9)
    hypotheses, _ = assoc.associate([track], [m_near])

    gmm = hypotheses[track]
    assert equal_to_tolerance(np.sum(gmm._weights), 1.0)


def test_pda_measurement_outside_gate_only_null_hypothesis():
    """When measurement fails gate, only the missed-detection hypothesis survives."""
    track = make_track(10, 5, track_id='T1')
    m_far = make_measurement(40)   # fails gate

    assoc = PDAAssociator(motion_model=make_cv_model(), gate_probability=0.99,
                          detection_probability=0.9)
    hypotheses, _ = assoc.associate([track], [m_far])

    gmm = hypotheses[track]
    assert len(gmm._hypotheses) == 1
    assert isinstance(gmm._hypotheses[0], MissedDetectionHypothesis)
