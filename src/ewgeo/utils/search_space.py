import numpy as np
import numpy.typing as npt
from typing import Self

class SearchSpace:
    _x_ctr: npt.NDArray[np.float64] | None = None
    _epsilon: npt.NDArray[np.float64] | None = None
    _points_per_dim: npt.NDArray[np.int64] | None = None
    _max_offset: npt.NDArray[np.float64] | None = None

    # Inferred grid
    _x_set: npt.NDArray[np.float64] | None = None                   # (n_dim, N) array of all positions in grid
    _x_grid: tuple[npt.NDArray[np.float64], ...] | None = None      # each element is an n-dimensional grid with N elements
    _x_vec: tuple[npt.NDArray[np.float64], ...] | None = None       # each element is a 1d numpy array
    _extent: tuple[float, ...] | None = None

    def __init__(self,
                 x_ctr:npt.NDArray[np.float64] | float,
                 epsilon:npt.NDArray[np.float64] | float | None=None,
                 points_per_dim:npt.NDArray[np.int64] | int | None=None,
                 max_offset:npt.NDArray[np.float64] | float | None=None):
        self._x_ctr = np.array(x_ctr, dtype=np.float64)
        if epsilon is not None: self._epsilon = np.array(epsilon, dtype=np.float64)
        if points_per_dim is not None: self._points_per_dim = np.array(points_per_dim, dtype=np.int64)
        if max_offset is not None: self._max_offset = np.array(max_offset, dtype=np.float64)

        # Verify sizing and consistency
        self.check_consistency()

    @property
    def num_parameters(self)-> int:
        self.broadcast()
        return np.prod(np.shape(self.x_ctr)).astype(np.int64).item()

    @property
    def x_ctr(self)-> npt.NDArray[np.float64]:
        self.broadcast()
        return self._x_ctr

    @x_ctr.setter
    def x_ctr(self, x_ctr: npt.NDArray[np.float64]):
        self.reset()  # clear the dependent fields
        self._x_ctr = x_ctr

    @property
    def epsilon(self)-> npt.NDArray[np.float64]:
        self.broadcast()
        if self._epsilon is None:
            # Build epsilon from max_offset and points_per_dim
            out_shape = np.amax(np.shape(self.points_per_dim), np.shape(self.max_offset))
            self._epsilon = np.divide(self.max_offset, self.points_per_dim - 1,
                                      out=np.ones(out_shape), where=self.points_per_dim > 1)
        return self._epsilon

    @epsilon.setter
    def epsilon(self, epsilon: npt.NDArray[np.float64]):
        self.reset()  # clear the dependent fields
        self._epsilon = epsilon
        if self._epsilon is not None:
            self.check_consistency()

    @property
    def max_offset(self)-> npt.NDArray[np.float64]:
        self.broadcast()
        if self._max_offset is None:
            # Build max_offset from epsilon and points_per_dim
            self._max_offset = self.epsilon * (self.points_per_dim - 1) / 2
        return self._max_offset

    @max_offset.setter
    def max_offset(self, max_offset: npt.NDArray[np.float64]):
        self.reset()  # clear the dependent fields
        self._max_offset = max_offset
        if self._max_offset is not None:
            self.check_consistency()

    @property
    def points_per_dim(self)-> npt.NDArray[np.int64]:
        if self._points_per_dim is None:
            # Build points_per_dim from max_offset and epsilon
            self._points_per_dim = np.where(self.epsilon != 0, np.floor(1 + 2 * self.max_offset / self.epsilon), 1).astype(int)
        return self._points_per_dim

    @points_per_dim.setter
    def points_per_dim(self, points_per_dim: npt.NDArray[np.int64]):
        self.reset()  # clear the dependent fields
        self._points_per_dim = points_per_dim
        if self._points_per_dim is not None:
            self.check_consistency()

    @property
    def x_vec(self)-> tuple[npt.NDArray[np.float64], ...]:
        if self._x_vec is None:
            self.make_nd_grid()
        return self._x_vec

    @property
    def x_set(self)-> npt.NDArray[np.float64]:
        if self._x_set is None:
            self.make_nd_grid()
        return np.array(self._x_set)

    @property
    def x_grid(self)-> tuple[npt.NDArray[np.float64], ...]:
        if self._x_grid is None:
            self.make_nd_grid()
        return self._x_grid

    @property
    def grid_shape(self)-> tuple[int, ...]:
        self.broadcast()
        return tuple([i for i in self.points_per_dim if i > 1])

    def reset(self):
        """
        Clear dependent fields
        """
        self._x_set = None
        self._x_grid = None

    def check_consistency(self):
        """
        Check that max_offset, points_per_dim, and epsilon are consistent.  If not, raise an error.
        """
        if self._points_per_dim is None or self._epsilon is None or self._max_offset is None:
            # Nothing to do; they're consistent because one is missing
            return True
        else:
            # Compute epsilon from max_offset and points_per_dim
            out_shape = np.maximum(np.shape(self.points_per_dim), np.shape(self.max_offset))
            epsilon_local = np.divide(self.max_offset, self.points_per_dim - 1,
                                      out=np.ones(out_shape), where=self.points_per_dim>1)

            # Compare to epsilon and throw an assertion error if it's more than 0.1% off
            err = self.epsilon - epsilon_local
            return np.sqrt(np.sum(np.abs(err)**2, axis=None)) < .001 * np.sqrt(np.sum(np.abs(self.epsilon)**2, axis=None))

    def broadcast(self)-> bool:
        # Verify that all variable sizes are compatible
        attrs = ['_x_ctr', '_epsilon', '_points_per_dim', '_max_offset']
        try:
            b = np.broadcast(*[getattr(self, attr) for attr in attrs if getattr(self, attr) is not None])
            for attr in attrs:
                if getattr(self, attr) is not None:
                    setattr(self, attr, np.broadcast_to(getattr(self, attr), shape=b.shape))
            return True
        except ValueError:
            return False

    def make_nd_grid(self):
        """
        Create and return an ND search grid, based on the specified center of the search space, extent, and grid spacing.

        28 December 2021
        Nicholas O'Donoughue

        :return x_set: n_dim x N numpy array of positions
        :return x_grid: n_dim-tuple of n_dim-dimensional numpy arrays containing the coordinates for each dimension.
        :return out_shape:  tuple with the size of the generated grid
        """

        n_dim = self.num_parameters

        if np.size(self.x_ctr) == 1:
            x_ctr = self.x_ctr * np.ones((n_dim, ))
        else:
            x_ctr = self.x_ctr.ravel()

        if np.size(self.max_offset) == 1:
            max_offset = self.max_offset * np.ones((n_dim, ))
        else:
            max_offset = self.max_offset.ravel()

        if np.size(self.points_per_dim) == 1:
            points_per_dim = self.points_per_dim * np.ones((n_dim, ))
        else:
            points_per_dim = self.points_per_dim.ravel()

        assert n_dim == np.size(max_offset) and n_dim == np.size(points_per_dim), \
               'Search space dimensions do not match across specification of the center, search_size, and epsilon.'

        # Check Search Size
        max_elements = 1e8  # Set a conservative limit
        assert np.prod(points_per_dim) < max_elements, \
               'Search size is too large; python is likely to crash or become unresponsive. Reduce your search size, or' \
               + ' increase the max allowed.'

        # Make a set of axes, one for each dimension, that are centered on x_ctr
        dims = [x + np.linspace(start=-x_max, stop=x_max, num=n) if n > 1 else x for (x, x_max, n)
                in zip(x_ctr, max_offset, points_per_dim)]

        # Use meshgrid expansion; each element of x_grid is now a full n_dim dimensioned grid
        x_grid = np.meshgrid(*dims)

        # Rearrange to a single 2D array of grid locations (n_dim x N)
        x_set = np.asarray([x.flatten() for x in x_grid])

        self._x_vec = tuple(dims)
        self._x_set = x_set
        self._x_grid = x_grid


    def get_extent(self, axes: tuple[int, int] | list[int] | npt.NDArray[np.int64] | None=None,
                   multiplier: float=1)-> tuple[float, float, float, float]:
        """
        For the specified axes, generate and return a tuple to be used with plotting commands.
        Optionally accepts a multiplier to scale the extent (e.g., from meters to kilometers).

        :param axes: list[int], list of axes indices over which to generate an extent. If empty, all axes are returned.
        :param multiplier: float, multiplier to scale the extent (e.g., .001 for scale from meters to kilometers).
        :return extent: tuple[float, ...] grid extent suitable for use with matplotlib plotting commands
        """
        if self._x_set is None:
            self.make_nd_grid()

        if axes is None:
            ax0 = 0
            ax1 = 1
        elif len(axes) == 2:
            ax0 = axes[0]
            ax1 = axes[1]
        else:
            raise ValueError(f'axes must have 2 entries; received {axes}')

        x0 = self.x_ctr[ax0].item()*multiplier
        x1 = self.x_ctr[ax1].item()*multiplier
        o0 = self.max_offset[ax0].item()*multiplier
        o1 = self.max_offset[ax1].item()*multiplier

        return x0-o0, x0+o0, x1-o1, x1+o1

    def zoom_in(self, new_ctr: npt.ArrayLike, zoom: float=2.0, overwrite: bool=False)-> Self | None:
        if np.shape(new_ctr) != np.shape(self.x_ctr):
            raise ValueError('New center must have the same dimensionality as the existing center.')

        # Keep the number of grid points the same, but cut the grid resolution by the zoom factor
        if overwrite:
            self.epsilon = self.epsilon/zoom
            self.max_offset = None
            return None
        else:
            return SearchSpace(x_ctr=new_ctr,
                               epsilon=self.epsilon/zoom,
                               points_per_dim=self.points_per_dim)