import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.lines import Line2D
from matplotlib.quiver import Quiver

from ..utils.covariance import CovarianceMatrix
from ..utils.errors import draw_error_ellipse

class StateSpace:
    """
    Class to represent various state types and parameters to assist in easily accessing them.
    """
    num_dims: int
    num_states: int
    has_pos: bool
    has_vel: bool
    has_accel: bool
    pos_slice: slice | None
    vel_slice: slice | None
    pos_vel_slice: slice | None
    accel_slice: slice | None

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def pos_component(self, x: npt.ArrayLike) -> npt.NDArray:
        return x[self.pos_slice]

    def vel_component(self, x: npt.ArrayLike) -> npt.NDArray:
        return x[self.vel_slice] if self.has_vel else None

    def pos_vel_component(self, x: npt.ArrayLike) -> npt.NDArray:
        return x[self.pos_vel_slice] if self.has_vel else self.pos_component(x)

    def accel_component(self, x: npt.ArrayLike) -> npt.NDArray:
        return x[self.accel_slice] if self.has_accel else None

    def copy(self, **kwargs):
        new_state_space = StateSpace(**self.__dict__)
        for key, value in kwargs.items():
            new_state_space.__setattr__(key, value)
        return new_state_space

class State:
    """
    Representation of the position, and optionally the velocity and acceleration, of a target at
    a given point in time.
    """
    state_space: StateSpace
    time: float
    state: npt.ArrayLike
    covar: CovarianceMatrix

    def __init__(self, state_space: StateSpace, time: float, state: npt.ArrayLike, covar: CovarianceMatrix=None):
        self.state_space = state_space
        self.time = time
        if state is None:
            self.state = np.zeros((state_space.num_states,))
        else:
            self.state = np.asarray(state)
        if covar is not None:
            self.covar = CovarianceMatrix(covar)

    def __str__(self):
        return f"State at t={self.time}: {self.state}"

    def copy(self, **kwargs):
        new_state=State(self.state_space, self.time, self.state, self.covar)
        for key, value in kwargs.items():
            new_state.__setattr__(key, value)
        return new_state

    @property
    def size(self) -> int:
        return self.state_space.num_states

    @property
    def position(self) -> npt.NDArray:
        return self.state_space.pos_component(self.state)

    @position.setter
    def position(self, value: npt.ArrayLike):
        self.state[self.state_space.pos_slice] = value

    @property
    def velocity(self) -> npt.NDArray:
        return self.state_space.vel_component(self.state) if self.has_vel else None

    @velocity.setter
    def velocity(self, value: npt.ArrayLike):
        if self.has_vel:
            self.state[self.state_space.vel_slice] = value
        else:
            raise ValueError("Unable to set velocity component of state that has not velocity component.")

    @property
    def pos_vel(self) -> npt.NDArray:
        return self.state_space.pos_vel_component(self.state) if self.has_vel else None

    @pos_vel.setter
    def pos_vel(self, value: npt.ArrayLike):
        if self.has_vel:
            self.state[self.state_space.pos_vel_slice] = value
        else:
            # Try to just set the position
            self.position = value

    @property
    def acceleration(self) -> npt.NDArray:
        return self.state_space.accel_component(self.state) if self.has_accel else None

    @acceleration.setter
    def acceleration(self, value: npt.ArrayLike):
        if self.has_accel:
            self.state[self.state_space.accel_slice] = value
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
             **kwargs)-> tuple[Line2D, Line2D, Quiver] or None:

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
                    trk_hdl: Line2D or None, trk_err_hdl: Line2D or None, trk_vel_hdl: Quiver or None,
                    plot_dims: slice = np.s_[:], do_pos: bool=True, do_vel: bool=False, do_cov: bool=False,
                    scale: float=1, cov_ellipse_confidence: float=0.75):

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