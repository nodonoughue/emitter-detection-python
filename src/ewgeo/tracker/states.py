import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

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
    pos_slice: slice
    vel_slice: slice
    pos_vel_slice: slice
    accel_slice: slice

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def pos_component(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return x[self.pos_slice]

    def vel_component(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return x[self.vel_slice] if self.has_vel else None

    def accel_component(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return x[self.accel_slice] if self.has_accel else None

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
        self.state = state
        self.covar = covar

    def copy(self, **kwargs):
        new_state=State(self.state_space, self.time, self.state, self.covar)
        for key, value in kwargs.items():
            new_state.__setattr__(key, value)
        return new_state

    @property
    def position(self):
        return self.state_space.pos_component(self.state)

    @property
    def velocity(self):
        return self.state_space.vel_component(self.state)

    @property
    def acceleration(self):
        return self.state_space.accel_component(self.state)

    @property
    def has_vel(self):
        return self.state_space.has_vel

    @property
    def has_accel(self):
        return self.state_space.has_accel

    @ property
    def position_covar(self):
        if self.covar is None:
            return None
        else:
            pos_slice = self.state_space.pos_slice
            return CovarianceMatrix(self.covar.cov[pos_slice, pos_slice])

    @property
    def velocity_covar(self):
        if self.covar is None:
            return None
        else:
            vel_slice = self.state_space.vel_slice
            return CovarianceMatrix(self.covar.cov[vel_slice, vel_slice])

    @property
    def acceleration_covar(self):
        if self.covar is None:
            return None
        else:
            accel_slice = self.state_space.accel_slice
            return CovarianceMatrix(self.covar.cov[accel_slice, accel_slice])

    @property
    def pos_vel_covar(self):
        if self.covar is None:
            return None
        else:
            pos_vel_slice = self.state_space.pos_vel_slice
            return CovarianceMatrix(self.covar.cov[pos_vel_slice, pos_vel_slice])

    def plot(self, ax=plt.Axes, plot_dims: slice=np.s_[:], do_pos: bool=True, do_vel: bool=False, do_cov: bool=False, **kwargs):

        # Plot Position
        coords = self.position[plot_dims]
        if do_pos:
            ax.scatter(*coords, **kwargs)

        # Plot Velocity
        if do_vel and self.has_vel:
            ax.quiver(*coords, *self.velocity[plot_dims], **kwargs)

        # Current State Covariance
        if do_cov and self.covar is not None:
            xy_ellipse = draw_error_ellipse(x=coords,
                                            covariance=self.pos_vel_covar.cov,
                                            conf_interval=50)
            plt.plot(*xy_ellipse, **kwargs)
