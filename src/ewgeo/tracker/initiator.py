from abc import ABC, abstractmethod
import numpy as np

from .association import Associator, MissedDetectionHypothesis
from .measurement import Measurement, MeasurementModel
from .track import Track, State, StateSpace
from .states import adapt_cartesian_state
from ..utils.covariance import CovarianceMatrix

class Initiator(ABC):
    """Abstract base class for track initiation logic."""
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

    def __init__(self, msmt_model: MeasurementModel, target_state_space: StateSpace = None):
        """
        :param msmt_model:         MeasurementModel used to convert each measurement into an initial State
        :param target_state_space: If provided, each newly created State is adapted to this StateSpace
                                   via adapt_cartesian_state before the Track is created.
        """
        self.msmt_model = msmt_model
        self.target_state_space = target_state_space

    def initiate(self, measurements: list[Measurement], next_track_id: int=None) -> tuple[list[Track], int]:
        """
        Create one tentative track per measurement.

        :param measurements: New unassociated measurements from the current scan
        :param next_track_id: Starting track ID counter; uses and updates self.next_track_id if None
        :return: Tuple of (list of newly created Track objects, updated next_track_id)
        """
        tracks = []
        if next_track_id is None:
            next_track_id = self.next_track_id

        for m in measurements:
            # Determine a position and/or velocity for this measurement
            s = self.msmt_model.state_from_measurement(m)
            if self.target_state_space is not None:
                s = adapt_cartesian_state(s, self.target_state_space)

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

    def __init__(self, msmt_model: MeasurementModel, associator: Associator,
                 target_state_space: StateSpace = None,
                 target_max_velocity: float | None = None,
                 target_max_acceleration: float | None = None):
        """
        :param msmt_model:              MeasurementModel used to convert each measurement into a State
        :param associator:              Associator used to pair new measurements with buffered single-point tracks
        :param target_state_space:      If provided, each confirmed Track's initial State is adapted to this
                                        StateSpace via adapt_cartesian_state before the Track is created.
        :param target_max_velocity:     Optional upper bound on target speed [m/s].  When provided, the
                                        velocity block of the initial covariance is scaled so that its
                                        largest diagonal element does not exceed target_max_velocity².
                                        If None, the CRLB-derived estimate (pos_var / dt²) is used as-is.
        :param target_max_acceleration: Optional upper bound on target acceleration [m/s²].  When provided,
                                        the acceleration block of buffer-track covariances is scaled so
                                        that its largest diagonal element does not exceed
                                        target_max_acceleration².  Only has effect when the state space
                                        includes an acceleration component.  If None, the default 1e6 m²/s⁴
                                        sentinel is used as-is.
        """
        self.msmt_model = msmt_model
        self.associator = associator
        self.target_state_space = target_state_space
        self.target_max_velocity = target_max_velocity
        self.target_max_acceleration = target_max_acceleration
        self._buffered_measurements = {}
        self._buffer_tracks = []

    def initiate(self, measurements: list[Measurement], next_track_id: int=None) -> tuple[list[Track], int]:
        """
        Process one scan of measurements. Pairs new measurements with buffered single-point tracks to
        produce velocity estimates; unmatched new measurements are buffered for the next scan.

        :param measurements: New unassociated measurements from the current scan
        :param next_track_id: Starting track ID counter; uses and updates self.next_track_id if None
        :return: Tuple of (list of newly confirmed Track objects with velocity estimates, updated next_track_id)
        """
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
                    s1 = track.curr_state  # first point
                    # If the buffer LS failed (position at origin), skip this pair —
                    # a zero first-point would produce a wild velocity estimate.
                    if np.linalg.norm(s1.position) < 1.0:
                        continue
                    # Seed the second-point LS from s1's position so the solver starts
                    # close to the true location rather than at the origin.  This is
                    # especially important for z (altitude): if the sensor array is
                    # near-coplanar with z≈0, an LS started from the origin often
                    # converges to the mirror-image z<0 solution.  Starting from
                    # s1.position (which already has z>0 after the z-retry fix in
                    # state_from_measurement) eliminates that ambiguity for s2.
                    s2 = self.msmt_model.state_from_measurement(hyp.measurement,
                                                                x_init=s1.position)
                    # Skip if the second-point LS failed (position at or near origin)
                    if np.linalg.norm(s2.position) < 1.0:
                        continue

                    dt = s2.time - s1.time
                    if dt > 0:
                        vel_est = (s2.position - s1.position) / dt

                        # Build a proper initial state with velocity estimate
                        init_state = State(s2.state_space, s2.time,
                                           self._build_state_with_velocity(s2, vel_est),
                                           self._build_initial_covariance(s2, dt,
                                               self.target_max_velocity))
                        if self.target_state_space is not None:
                            init_state = adapt_cartesian_state(init_state, self.target_state_space)
                        confirmed_tracks.append(
                            Track(initial_state=init_state, track_id=track.track_id)
                        )
                # Either way, remove from buffer
                self._buffer_tracks.remove(track)

            # Buffer any unmatched new measurements as new single-point tracks
            for m in unmatched:
                s = self.msmt_model.state_from_measurement(
                    m,
                    target_max_velocity=self.target_max_velocity,
                    target_max_acceleration=self.target_max_acceleration)
                self._buffer_tracks.append(Track(initial_state=s, track_id=next_track_id))
                next_track_id += 1
        else:
            # Nothing buffered yet — buffer all measurements
            for m in measurements:
                s = self.msmt_model.state_from_measurement(
                    m,
                    target_max_velocity=self.target_max_velocity,
                    target_max_acceleration=self.target_max_acceleration)
                self._buffer_tracks.append(Track(initial_state=s, track_id=next_track_id))
                next_track_id += 1

        self.next_track_id = next_track_id
        return confirmed_tracks, next_track_id

    @staticmethod
    def _build_state_with_velocity(state: State, vel_est) -> np.ndarray:
        """Return a copy of ``state``'s state vector with the velocity sub-vector replaced by ``vel_est``."""
        new_state_vec = state.state.copy()
        new_state_vec[state.state_space.vel_slice] = vel_est
        return new_state_vec

    @staticmethod
    def _build_initial_covariance(state: State, dt: float,
                                  target_max_velocity: float | None = None) -> CovarianceMatrix:
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
            if target_max_velocity is not None:
                # Scale down the velocity covariance so that no diagonal element exceeds
                # target_max_velocity², while preserving the correlation structure.
                max_diag = np.max(np.diag(crlb_vel))
                if max_diag > target_max_velocity**2:
                    crlb_vel = crlb_vel * (target_max_velocity**2 / max_diag)
            init_covar[vel_slice, vel_slice] = crlb_vel

        # There isn't enough information to initialize acceleration; scale by another 1/dt²
        # (consistent with the velocity scaling above) to get a physically reasonable estimate.
        if state.state_space.has_accel:
            accel_slice = state.state_space.accel_slice
            crlb_accel = crlb_vel / (dt**2)
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

    def __init__(self, msmt_model: MeasurementModel, associator: Associator,
                 target_state_space: StateSpace = None,
                 target_max_velocity: float | None = None,
                 target_max_acceleration: float | None = None):
        """
        :param msmt_model:               MeasurementModel used to convert measurements into States
        :param associator:               Associator used to link measurements across the three buffered stages
        :param target_state_space:       If provided, each confirmed Track's initial State is adapted to this
                                         StateSpace via adapt_cartesian_state before the Track is created.
        :param target_max_velocity:      Optional upper bound on target speed [m/s].  When provided, the
                                         velocity block of the initial covariance is scaled so that its
                                         largest diagonal element does not exceed target_max_velocity².
                                         If None, the finite-difference propagation (pos_var / (2·dt²))
                                         is used as-is.
        :param target_max_acceleration:  Optional upper bound on target acceleration magnitude [m/s²].
                                         When provided, the acceleration block is scaled so that its
                                         largest diagonal element does not exceed target_max_acceleration².
                                         If None, the finite-difference propagation (6·pos_var / dt⁴)
                                         is used as-is.
        """
        self.msmt_model = msmt_model
        self.associator = associator
        self.target_state_space = target_state_space
        self.target_max_velocity = target_max_velocity
        self.target_max_acceleration = target_max_acceleration
        self._stage1_tracks = []
        self._stage2_tracks = []

    def initiate(self, measurements: list[Measurement], next_track_id: int=None) -> tuple[list[Track], int]:
        """
        Process one scan. Attempts to advance stage-2 tracks to full confirmed tracks,
        then advances stage-1 tracks to stage-2, then buffers any remaining measurements.

        :param measurements: New unassociated measurements from the current scan
        :param next_track_id: Starting track ID counter; uses and updates self.next_track_id if None
        :return: Tuple of (list of newly confirmed Track objects, updated next_track_id)
        """
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
                    s3 = self.msmt_model.state_from_measurement(hyp.measurement,
                                                                x_init=s2.position)  # t3
                    if np.linalg.norm(s3.position) < 1.0:
                        matched_stage2.append(track)
                        continue

                    full_track = self._build_track(s1, s2, s3)
                    if full_track is not None:
                        if self.target_state_space is not None:
                            adapted = adapt_cartesian_state(
                                full_track.curr_state, self.target_state_space)
                            full_track = Track(initial_state=adapted,
                                               track_id=full_track.track_id)
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
                    # Associate the second point and promote to stage 2; warm-start
                    # the LS from the stage-1 position to avoid mirror-image ambiguity.
                    s2 = self.msmt_model.state_from_measurement(hyp.measurement,
                                                                x_init=track.curr_state.position)
                    if np.linalg.norm(s2.position) < 1.0:
                        continue
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
            s = self.msmt_model.state_from_measurement(m,
                target_max_velocity=self.target_max_velocity,
                target_max_acceleration=self.target_max_acceleration)
            self._stage1_tracks.append(Track(initial_state=s, track_id=next_track_id))
            next_track_id += 1

        self.next_track_id = next_track_id
        return confirmed_tracks, next_track_id

    def _build_track(self, s1: State, s2: State, s3: State) -> Track | None:
        """
        Construct a confirmed Track from three geolocated States using central finite differences.

        Returns None if the time steps between the three states differ by more than 10%.

        :param s1: State at the first observation time
        :param s2: State at the second observation time
        :param s3: State at the third observation time (used as the initial state of the new track)
        :return: Initialized Track with position, velocity, and (if available) acceleration estimates,
                 or None if the time steps are inconsistent
        """
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
        init_state = State(s3.state_space, s3.time, init_state_vec, CovarianceMatrix(init_covar))

        return Track(initial_state=init_state)

    def _propagate_covariance(self, pos_var: float, dt: float,
                              state_space: StateSpace) -> np.ndarray:
        """
        Propagate scalar position variance through finite difference
        formulas to get velocity and acceleration variances.

        Var(v) = pos_var / (2 * dt²)      [central difference]
        Var(a) = 6 * pos_var / dt⁴        [second difference]

        If target_max_velocity or target_max_acceleration were supplied at construction,
        each block is scaled down so that its largest diagonal element does not exceed
        the corresponding squared bound, while preserving the correlation structure.
        """
        n = state_space.num_states
        covar = np.eye(n)

        vel_var = pos_var / (2.0 * dt ** 2)
        acc_var = 6.0 * pos_var / (dt ** 4)

        if self.target_max_velocity is not None:
            max_diag = np.max(np.diag(vel_var))
            if max_diag > self.target_max_velocity ** 2:
                vel_var = vel_var * (self.target_max_velocity ** 2 / max_diag)

        if self.target_max_acceleration is not None:
            max_diag = np.max(np.diag(acc_var))
            if max_diag > self.target_max_acceleration ** 2:
                acc_var = acc_var * (self.target_max_acceleration ** 2 / max_diag)

        covar[state_space.pos_slice, state_space.pos_slice] *= pos_var
        if state_space.has_vel:
            covar[state_space.vel_slice, state_space.vel_slice] *= vel_var
        if state_space.has_accel:
            covar[state_space.accel_slice, state_space.accel_slice] *= acc_var

        return covar