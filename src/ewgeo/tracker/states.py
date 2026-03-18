from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.lines import Line2D
from matplotlib.quiver import Quiver

from ..utils.covariance import CovarianceMatrix
from ..utils.errors import draw_error_ellipse


class StateSpace(ABC):
    """
    Abstract base class describing the layout of a tracker state vector.

    Subclasses define which physical quantities are tracked and how to extract
    them from a flat state vector.  All properties are read-only after construction
    so that a single StateSpace instance can safely be shared across many State
    objects without risk of mutation.
    """

    # ------------------------------------------------------------------ #
    # Abstract scalar properties                                           #
    # ------------------------------------------------------------------ #

    @property
    @abstractmethod
    def num_dims(self) -> int:
        """Number of spatial dimensions (e.g. 2 for 2-D, 3 for 3-D)."""

    @property
    @abstractmethod
    def num_states(self) -> int:
        """Total length of the state vector."""

    @property
    @abstractmethod
    def has_pos(self) -> bool:
        """True if the state vector contains position components."""

    @property
    @abstractmethod
    def has_vel(self) -> bool:
        """True if the state vector contains velocity components."""

    @property
    @abstractmethod
    def has_accel(self) -> bool:
        """True if the state vector contains acceleration components."""

    # ------------------------------------------------------------------ #
    # Abstract slice properties                                            #
    # ------------------------------------------------------------------ #

    @property
    @abstractmethod
    def pos_slice(self) -> slice:
        """Index slice selecting the position components of the state vector."""

    @property
    @abstractmethod
    def vel_slice(self) -> slice | None:
        """Index slice selecting the velocity components, or None when absent."""

    @property
    @abstractmethod
    def pos_vel_slice(self) -> slice | None:
        """Index slice selecting the combined position+velocity block, or None when absent."""

    @property
    @abstractmethod
    def accel_slice(self) -> slice | None:
        """Index slice selecting the acceleration components, or None when absent."""

    # ------------------------------------------------------------------ #
    # Concrete accessor methods (use the abstract slice properties above)  #
    # Subclasses may override these for non-Cartesian state representations #
    # ------------------------------------------------------------------ #

    def pos_component(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return the position sub-vector of state vector x."""
        return x[self.pos_slice]

    def vel_component(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64] | None:
        """Return the velocity sub-vector of state vector x, or None if absent."""
        return x[self.vel_slice] if self.has_vel else None

    def pos_vel_component(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return the combined position+velocity sub-vector, falling back to position-only."""
        return x[self.pos_vel_slice] if self.has_vel else self.pos_component(x)

    def accel_component(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64] | None:
        """Return the acceleration sub-vector of state vector x, or None if absent."""
        return x[self.accel_slice] if self.has_accel else None


class CartesianStateSpace(StateSpace):
    """
    Concrete StateSpace for models whose state vector has the block layout:

        [pos (n_dims) | vel (n_dims) | accel (n_dims) | jerk (n_dims)]

    where trailing blocks are optional.  All properties are computed once at
    construction and exposed as read-only via @property.

    :param num_dims:  Number of spatial dimensions.
    :param has_vel:   Include a velocity block (default True).
    :param has_accel: Include an acceleration block (default False).
    :param has_jerk:  Include a jerk block (default False); requires has_accel=True.
    """

    def __init__(self, num_dims: int, has_vel: bool = True,
                 has_accel: bool = False, has_jerk: bool = False):
        if has_jerk and not has_accel:
            raise ValueError("has_jerk=True requires has_accel=True")

        n = num_dims
        num_blocks = 1 + int(has_vel) + int(has_accel) + int(has_jerk)

        self._num_dims   = n
        self._num_states = num_blocks * n
        self._has_vel    = has_vel
        self._has_accel  = has_accel

        self._pos_slice     = np.s_[:n]
        self._vel_slice     = np.s_[n:2*n]       if has_vel   else None
        self._pos_vel_slice = np.s_[:2*n]         if has_vel   else np.s_[:n]
        self._accel_slice   = np.s_[2*n:3*n]     if has_accel else None

    @property
    def num_dims(self) -> int:
        return self._num_dims

    @property
    def num_states(self) -> int:
        return self._num_states

    @property
    def has_pos(self) -> bool:
        return True

    @property
    def has_vel(self) -> bool:
        return self._has_vel

    @property
    def has_accel(self) -> bool:
        return self._has_accel

    @property
    def pos_slice(self) -> slice:
        return self._pos_slice

    @property
    def vel_slice(self) -> slice | None:
        return self._vel_slice

    @property
    def pos_vel_slice(self) -> slice | None:
        return self._pos_vel_slice

    @property
    def accel_slice(self) -> slice | None:
        return self._accel_slice

class State:
    """
    Representation of the position, and optionally the velocity and acceleration, of a target at
    a given point in time.
    """
    state_space: StateSpace
    time: float
    state: npt.NDArray[np.float64]
    covar: CovarianceMatrix

    def __init__(self, state_space: StateSpace, time: float, state: npt.ArrayLike, covar: CovarianceMatrix=None):
        """
        :param state_space: StateSpace object describing the layout of the state vector
        :param time: Timestamp of this state [seconds]
        :param state: State vector array of shape (num_states,); zeros used if None
        :param covar: Optional CovarianceMatrix (or array-like) for the state estimate error
        """
        self.state_space = state_space
        self.time = time
        if state is None:
            self.state = np.zeros((state_space.num_states,))
        else:
            self.state = np.array(state, dtype=np.float64)
        self.covar = CovarianceMatrix(covar) if covar is not None else None

    def __str__(self):
        return f"State at t={self.time}: {self.state}"

    @property
    def size(self) -> int:
        return self.state_space.num_states

    @property
    def position(self) -> npt.NDArray[np.float64]:
        return self.state_space.pos_component(self.state)

    @position.setter
    def position(self, value: npt.ArrayLike):
        """Set the position sub-vector of the state vector."""
        self.state[self.state_space.pos_slice] = np.array(value)

    @property
    def velocity(self) -> npt.NDArray[np.float64] | None:
        return self.state_space.vel_component(self.state) if self.has_vel else None

    @velocity.setter
    def velocity(self, value: npt.ArrayLike):
        """Set the velocity sub-vector of the state vector. Raises ValueError if state space has no velocity."""
        if self.has_vel:
            self.state[self.state_space.vel_slice] = np.array(value)
        else:
            raise ValueError("Unable to set velocity component of state that has not velocity component.")

    @property
    def pos_vel(self) -> npt.NDArray[np.float64] | None:
        return self.state_space.pos_vel_component(self.state) if self.has_vel else None

    @pos_vel.setter
    def pos_vel(self, value: npt.ArrayLike):
        """Set the position+velocity sub-vector of the state vector. Falls back to setting position-only
        if this state space has no velocity."""
        if self.has_vel:
            self.state[self.state_space.pos_vel_slice] = np.array(value)
        else:
            # Try to just set the position
            self.position = np.array(value)

    @property
    def acceleration(self) -> npt.NDArray[np.float64] | None:
        return self.state_space.accel_component(self.state) if self.has_accel else None

    @acceleration.setter
    def acceleration(self, value: npt.ArrayLike):
        """Set the acceleration sub-vector of the state vector. Raises ValueError if state space has no acceleration."""
        if self.has_accel:
            self.state[self.state_space.accel_slice] = np.array(value)
        else:
            raise ValueError("Unable to set acceleration component of state that has not acceleration component.")

    @property
    def has_vel(self) -> bool:
        return self.state_space.has_vel

    @property
    def has_accel(self) -> bool:
        return self.state_space.has_accel

    @ property
    def position_covar(self) -> CovarianceMatrix | None:
        if self.covar is None:
            return None
        else:
            pos_slice = self.state_space.pos_slice
            return CovarianceMatrix(self.covar.cov[pos_slice, pos_slice])

    @property
    def velocity_covar(self) -> CovarianceMatrix | None:
        if self.covar is None or not self.has_vel:
            return None
        else:
            vel_slice = self.state_space.vel_slice
            return CovarianceMatrix(self.covar.cov[vel_slice, vel_slice])

    @property
    def acceleration_covar(self) -> CovarianceMatrix | None:
        if self.covar is None or not self.has_accel:
            return None
        else:
            accel_slice = self.state_space.accel_slice
            return CovarianceMatrix(self.covar.cov[accel_slice, accel_slice])

    @property
    def pos_vel_covar(self) -> CovarianceMatrix | None:
        if self.covar is None:
            return None
        else:
            pos_vel_slice = self.state_space.pos_vel_slice
            return CovarianceMatrix(self.covar.cov[pos_vel_slice, pos_vel_slice])

    def plot(self, ax: plt.Axes, plot_dims: slice=np.s_[:],
             do_pos: bool=True, do_vel: bool=False, do_cov: bool=False,
             scale: float=1, cov_ellipse_confidence: float=0.75,
             **kwargs)-> tuple[Line2D, Line2D, Quiver] | None:
        """
        Plot this state on the provided axes.

        :param ax: Matplotlib Axes to plot on
        :param plot_dims: Slice selecting which spatial dimensions to include (default: all)
        :param do_pos: If True, plot the position as a point (default: True)
        :param do_vel: If True, draw a velocity arrow at the position (default: False)
        :param do_cov: If True, draw a covariance error ellipse around the position (default: False)
        :param scale: Divide all coordinates by this factor before plotting (e.g., 1e3 for km) (default: 1)
        :param cov_ellipse_confidence: Confidence interval for the covariance ellipse (default: 0.75)
        :param kwargs: Additional keyword arguments forwarded to ax.plot()
        :return: Tuple of (position handle, error ellipse handle, velocity quiver handle); any may be None
        """

        coords = self.position[plot_dims]/scale

        # Plot Position
        if do_pos:
            trk_hdl = ax.plot(*coords, **kwargs)[0]
            if 'color' not in kwargs.keys():
                kwargs['color'] = trk_hdl.get_color()
        else:
            trk_hdl = None

        # Current State Covariance
        if do_cov and self.covar is not None:
            xy_ellipse = draw_error_ellipse(x=coords,
                                            covariance=self.position_covar,
                                            conf_interval=cov_ellipse_confidence)
            trk_err_hdl = plt.plot(*xy_ellipse, **kwargs)[0]
        else:
            trk_err_hdl = None

        # Plot Velocity
        if do_vel and self.has_vel:
            trk_vel_hdl = ax.quiver(*coords, *self.velocity[plot_dims]/scale, **kwargs)
        else:
            trk_vel_hdl = None

        return trk_hdl, trk_err_hdl, trk_vel_hdl

    def update_plot(self,
                    trk_hdl: Line2D | None, trk_err_hdl: Line2D | None, trk_vel_hdl: Quiver | None,
                    plot_dims: slice = np.s_[:], do_pos: bool=True, do_vel: bool=False, do_cov: bool=False,
                    scale: float=1, cov_ellipse_confidence: float=0.75):
        """
        Update existing plot handles with the current state data (for animation).

        :param trk_hdl: Existing position Line2D handle to update, or None
        :param trk_err_hdl: Existing covariance ellipse Line2D handle to update, or None
        :param trk_vel_hdl: Existing velocity Quiver handle to update, or None
        :param plot_dims: Slice selecting which spatial dimensions to include (default: all)
        :param do_pos: If True, update the position handle (default: True)
        :param do_vel: If True, update the velocity arrow (default: False)
        :param do_cov: If True, update the covariance ellipse (default: False)
        :param scale: Divide all coordinates by this factor before plotting (default: 1)
        :param cov_ellipse_confidence: Confidence interval for the covariance ellipse (default: 0.75)
        """

        # Plot Position
        coords = self.position[plot_dims]/scale
        if do_pos and trk_hdl is not None:
            trk_hdl.set_data(*coords)

        # Plot Velocity
        if do_vel and self.has_vel and trk_vel_hdl is not None:
            trk_vel_hdl.set_offsets(np.column_stack(*coords))
            trk_vel_hdl.set_UVC(*self.velocity[plot_dims]/scale)

        # Current State Covariance
        if do_cov and self.covar is not None and trk_err_hdl is not None:
            xy_ellipse = draw_error_ellipse(x=coords,
                                            covariance=self.position_covar,
                                            conf_interval=cov_ellipse_confidence)
            trk_err_hdl.set_data(*xy_ellipse)

        return