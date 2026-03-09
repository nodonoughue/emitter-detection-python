from abc import ABC, abstractmethod
import numpy as np

from .association import Associator, MissedDetectionHypothesis
from .measurement import Measurement, MeasurementModel
from .track import Track, State, StateSpace
from ..utils.covariance import CovarianceMatrix

class Initiator(ABC):
    next_track_id: int = 0

    @abstractmethod
    def initiate(self, measurements: list[Measurement]) -> list[Track]:
        pass

class SinglePointMeasurementInitiator(Initiator):
    """
    This initiator type accepts a list of measurements and instantiates a track for each one.

    A significant limitation of this type of initiator is that, unless the PSS system includes
    FDOA elements that can directly estimate velocity, the velocity term will initialize to zero.

    The covariance is initially very large, but it can still be difficult for the tracker to
    properly estimate and update velocity indirectly from subsequent measurements, as they may not
    properly associate with the track.
    """
    msmt_model: MeasurementModel

    def __init__(self, msmt_model: MeasurementModel):
        self.msmt_model = msmt_model

    def initiate(self, measurements: list[Measurement], next_track_id: int=None) -> tuple[list[Track], int]:
        tracks = []
        if next_track_id is None:
            next_track_id = self.next_track_id

        for m in measurements:
            # Determine a position and/or velocity for this measurement
            s = self.msmt_model.state_from_measurement(m)

            # Initialize a track object
            t = Track(initial_state=s, track_id=next_track_id)
            next_track_id += 1
            tracks.append(t)

        self.next_track_id = next_track_id
        return tracks, next_track_id

class TwoPointInitiator(Initiator):
    """
    Buffers single-point observations and creates tracks from pairs of
    measurements across two consecutive time steps, giving a direct
    velocity estimate.
    """
    msmt_model: MeasurementModel  # Used to convert measurements to states
    associator: Associator        # Used to associate measurements for track initiation
    _buffered_measurements: dict  # keyed by some tentative track id
    _buffer_tracks: list          # single-point tentative tracks for association

    def __init__(self, msmt_model: MeasurementModel, associator: Associator):
        self.msmt_model = msmt_model
        self.associator = associator
        self._buffered_measurements = {}
        self._buffer_tracks = []

    def initiate(self, measurements: list[Measurement], next_track_id: int=None) -> tuple[list[Track], int]:
        confirmed_tracks = []
        if next_track_id is None:
            next_track_id = self.next_track_id

        # Step 1: Associate new measurements with buffered single-point tracks
        if self._buffer_tracks:
            hypothesis_dict, unmatched = self.associator.associate(
                tracks=self._buffer_tracks,
                measurements=measurements
            )
            for track, hyp in hypothesis_dict.items():
                if not isinstance(hyp, MissedDetectionHypothesis):
                    # We have two points — estimate velocity
                    s1 = track.curr_state            # first point
                    s2 = hyp.state_from_measurement  # second point

                    dt = s2.time - s1.time
                    if dt > 0:
                        vel_est = (s2.position - s1.position) / dt

                        # Build a proper initial state with velocity estimate
                        init_state = s2.copy(
                            state=self._build_state_with_velocity(s2, vel_est),
                            covar=self._build_initial_covariance(s2, dt)
                        )
                        confirmed_tracks.append(
                            Track(initial_state=init_state, track_id=track.track_id)
                        )
                # Either way, remove from buffer
                self._buffer_tracks.remove(track)

            # Buffer any unmatched new measurements as new single-point tracks
            for m in unmatched:
                s = self.msmt_model.state_from_measurement(m)
                self._buffer_tracks.append(Track(initial_state=s, track_id=next_track_id))
                next_track_id += 1
        else:
            # Nothing buffered yet — buffer all measurements
            for m in measurements:
                s = self.msmt_model.state_from_measurement(m)
                self._buffer_tracks.append(Track(initial_state=s, track_id=next_track_id))
                next_track_id += 1

        self.next_track_id = next_track_id
        return confirmed_tracks, next_track_id

    @staticmethod
    def _build_state_with_velocity(state: State, vel_est) -> np.ndarray:
        new_state_vec = state.state.copy()
        new_state_vec[state.state_space.vel_slice] = vel_est
        return new_state_vec

    @staticmethod
    def _build_initial_covariance(state: State, dt: float) -> CovarianceMatrix:
        # Start from the CRLB-based covariance but use a
        # conservative (larger) value
        n = state.state_space.num_states
        num_dim = state.state_space.num_dims
        init_covar = np.eye(n)
        pos_covar_multiplier = 10.0

        # Grab the state's covariance matrix as a starting point and take the max between that and our hard-coded
        # initial variance
        crlb = state.covar.cov  # should have the same size as n
        pos_slice = state.state_space.pos_slice
        crlb_pos = pos_covar_multiplier * crlb[:num_dim, :num_dim] # keep only the position error covariance
        init_covar[pos_slice, pos_slice] = crlb_pos

        # Use crlb for position to estimate velocity error
        if state.state_space.has_vel:
            vel_slice = state.state_space.vel_slice
            crlb_vel = crlb_pos / (dt**2)
            init_covar[vel_slice, vel_slice] = crlb_vel

        # There isn't enough information to initialize acceleration; so just give it a large value and let
        # the tracker bring it down
        if state.state_space.has_accel:
            accel_slice = state.state_space.accel_slice
            crlb_accel = crlb_pos * 1e6
            init_covar[accel_slice, accel_slice] = crlb_accel

        return CovarianceMatrix(init_covar)

class ThreePointInitiator(Initiator):
    """
    Buffers geolocated positions across three consecutive time steps,
    then initializes a track with position, velocity, and acceleration
    estimates using finite differences. Suitable for constant acceleration
    motion models.
    """
    msmt_model: MeasurementModel  # Used to convert measurements to states
    associator: Associator        # Used to associate measurements for track initiation
    _buffered_measurements: dict  # keyed by some tentative track id
    _stage1_tracks: list[Track]   # single-point buffer (t1)
    _stage2_tracks: list[Track]   # two-point buffer (t1, t2)

    def __init__(self, msmt_model: MeasurementModel, associator: Associator):
        self.msmt_model = msmt_model
        self.associator = associator
        self._stage1_tracks = []
        self._stage2_tracks = []

    def initiate(self, measurements: list[Measurement], next_track_id: int=None) -> tuple[list[Track], int]:
        confirmed_tracks = []
        if next_track_id is None:
            next_track_id = self.next_track_id

        # Step 1: Try to advance stage-2 tracks (t1, t2) to full tracks
        # Any measurements not used in step 1 are contained in the list unmatched_2
        if self._stage2_tracks:
            hyp_dict, unmatched_2 = self.associator.associate(
                tracks=self._stage2_tracks,
                measurements=measurements
            )
            matched_stage2 = []
            for track, hyp in hyp_dict.items():
                if not isinstance(hyp, MissedDetectionHypothesis):
                    # We have all three points; build a full track
                    s1 = track.states[0]    # geolocated position at t1
                    s2 = track.curr_state   # geolocated position at t2
                    s3 = hyp.state_from_measurement # t3

                    full_track = self._build_track(s1, s2, s3)
                    if full_track is not None:
                        confirmed_tracks.append(full_track)
                    matched_stage2.append(track)
                # Unconditionally retire stage-2 tracks (matched or not)
                # A missed detection at stage 2 means the track is ambiguous;
                # drop it and let stage 1 restart
                self._stage2_tracks = [t for t in self._stage2_tracks
                                       if t not in hyp_dict]
        else:
            unmatched_2 = measurements[:]

        # Step 2: Try to advance stage-1 tracks (t1) to stage-2 tracks
        # Any unused measurements will be stored in the list unmatched_2
        if self._stage1_tracks:
            hyp_dict_1, unmatched_1 = self.associator.associate(
                tracks=self._stage1_tracks,
                measurements=unmatched_2
            )
            for track, hyp in hyp_dict_1.items():
                if not isinstance(hyp, MissedDetectionHypothesis):
                    # Associate the second point and promote to stage 2
                    s2 = hyp.state_from_measurement
                    track.append(s2)
                    self._stage2_tracks.append(track)
                # Drop stage-1 tracks regardless of match — they either
                # advanced to stage 2 or failed
            self._stage1_tracks = [t for t in self._stage1_tracks
                                   if t not in hyp_dict_1]
        else:
            unmatched_1 = unmatched_2[:]

        # Step 3: Buffer all remaining measurements as stage_1 tracks
        for m in unmatched_1:
            s = self.msmt_model.state_from_measurement(m)
            self._stage1_tracks.append(Track(initial_state=s, track_id=next_track_id))
            next_track_id += 1

        self.next_track_id = next_track_id
        return confirmed_tracks, next_track_id

    def _build_track(self, s1: State, s2: State, s3: State) -> Track | None:
        dt1 = s2.time - s1.time
        dt2 = s3.time - s2.time

        # Require consistent time steps (within 10%)
        if abs(dt1 - dt2) / max(dt1, dt2) > 0.1:
            return None

        dt = (dt1 + dt2) / 2.0   # average dt

        # ---- Finite difference estimates ----
        x1 = s1.position
        x2 = s2.position
        x3 = s3.position

        vel_est = (x3 - x1) / (2.0 * dt)           # central difference
        acc_est = (x3 - 2.0 * x2 + x1) / (dt ** 2) # second difference

        # ---- Build state vector ----
        init_state_vec = np.zeros((s3.state_space.num_states,))
        init_state_vec[s3.state_space.pos_slice] = x3
        if s3.state_space.has_vel:
            init_state_vec[s3.state_space.vel_slice] = vel_est
        if s3.state_space.has_accel:
            init_state_vec[s3.state_space.accel_slice] = acc_est

        # ---- Build covariance from propagated position uncertainty ----

        # Use the average CRLB trace as the scalar position variance,
        # but apply a conservative floor to avoid collapse
        crlb = s3.covar.cov
        pos_covar_multiplier = 10.0
        num_dim = s3.state_space.num_dims
        pos_var = np.mean([pos_covar_multiplier * s.covar.cov[:num_dim, :num_dim] for s in [s1, s2, s3]], axis=0)

        init_covar = self._propagate_covariance(pos_var, dt, s3.state_space)
        init_state = s3.copy(state=init_state_vec, covar=CovarianceMatrix(init_covar))

        return Track(initial_state=init_state)

    def _propagate_covariance(self, pos_var: float, dt: float,
                              state_space: StateSpace) -> np.ndarray:
        """
        Propagate scalar position variance through finite difference
        formulas to get velocity and acceleration variances.

        Var(v) = pos_var / (2 * dt²)      [central difference]
        Var(a) = 6 * pos_var / dt⁴        [second difference]
        """
        n = state_space.num_states
        covar = np.eye(n)

        vel_var = pos_var / (2.0 * dt ** 2)
        acc_var = 6.0 * pos_var / (dt ** 4)

        covar[state_space.pos_slice, state_space.pos_slice] *= pos_var
        covar[state_space.vel_slice, state_space.vel_slice] *= vel_var
        covar[state_space.accel_slice, state_space.accel_slice] *= acc_var

        return covar