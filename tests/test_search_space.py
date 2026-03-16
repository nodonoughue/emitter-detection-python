import numpy as np
import pytest

from ewgeo.utils import SearchSpace


# ===========================================================================
# Construction modes — each pair of params derives the third
# ===========================================================================

def test_construct_epsilon_and_max_offset():
    """points_per_dim should be floor(1 + 2*max_offset/epsilon)."""
    ss = SearchSpace(x_ctr=np.array([0.0, 0.0]), epsilon=10.0, max_offset=100.0)
    expected = int(np.floor(1 + 2 * 100.0 / 10.0))  # 21
    assert np.all(ss.points_per_dim == expected)


def test_construct_epsilon_and_points_per_dim():
    """max_offset should be epsilon * (N-1) / 2."""
    ss = SearchSpace(x_ctr=np.array([0.0, 0.0]), epsilon=5.0, points_per_dim=11)
    expected = 5.0 * (11 - 1) / 2  # 25.0
    assert np.allclose(ss.max_offset, expected)


def test_construct_max_offset_and_points_per_dim():
    """Grid should span the full max_offset range with the correct number of points."""
    max_offset = 50.0
    ppd = 11
    ss = SearchSpace(x_ctr=np.array([0.0, 0.0]), max_offset=max_offset, points_per_dim=ppd)
    # Shape: 2 dims × ppd² total points
    assert ss.x_set.shape == (2, ppd * ppd)
    # Range in first dimension: should go from -max_offset to +max_offset
    x0 = ss.x_set[0]
    assert np.isclose(x0.min(), -max_offset)
    assert np.isclose(x0.max(),  max_offset)


# ===========================================================================
# x_set shape and content
# ===========================================================================

def test_x_set_shape_1d():
    # points_per_dim = floor(1 + 2*2/1) = 5
    ss = SearchSpace(x_ctr=np.array([5.0]), epsilon=1.0, max_offset=2.0)
    assert ss.x_set.shape == (1, 5)


def test_x_set_shape_2d():
    # total points = 5 * 5 = 25
    ss = SearchSpace(x_ctr=np.array([0.0, 0.0]), epsilon=1.0, points_per_dim=5)
    assert ss.x_set.shape == (2, 25)


def test_x_set_contains_center():
    """The center point must appear somewhere in x_set."""
    x_ctr = np.array([3.0, 7.0])
    ss = SearchSpace(x_ctr=x_ctr, epsilon=1.0, points_per_dim=5)
    diffs = np.linalg.norm(ss.x_set - x_ctr[:, np.newaxis], axis=0)
    assert np.any(diffs < 1e-10), "Center not found in x_set"


def test_x_set_bounds():
    """All x_set values must lie within [x_ctr - max_offset, x_ctr + max_offset]."""
    x_ctr = np.array([10.0, -5.0])
    max_offset = 3.0
    ss = SearchSpace(x_ctr=x_ctr, epsilon=1.0, max_offset=max_offset)
    for dim in range(2):
        assert np.all(ss.x_set[dim] >= x_ctr[dim] - max_offset - 1e-10)
        assert np.all(ss.x_set[dim] <= x_ctr[dim] + max_offset + 1e-10)


def test_x_set_spacing_1d():
    """Consecutive x_set values in a 1-D grid should be spaced exactly epsilon apart."""
    ss = SearchSpace(x_ctr=np.array([0.0]), epsilon=2.5, points_per_dim=7)
    x = np.sort(ss.x_set.ravel())
    diffs = np.diff(x)
    assert np.allclose(diffs, 2.5)


# ===========================================================================
# grid_shape
# ===========================================================================

def test_grid_shape_2d():
    ss = SearchSpace(x_ctr=np.array([0.0, 0.0]), epsilon=1.0, points_per_dim=7)
    assert ss.grid_shape == (7, 7)


def test_grid_shape_excludes_singletons():
    """A dimension with points_per_dim == 1 must be excluded from grid_shape."""
    ss = SearchSpace(x_ctr=np.array([0.0, 0.0]),
                     epsilon=np.array([1.0, 1.0]),
                     points_per_dim=np.array([5, 1]))
    assert ss.grid_shape == (5,)


# ===========================================================================
# num_parameters
# ===========================================================================

def test_num_parameters_1d():
    ss = SearchSpace(x_ctr=0.0, epsilon=1.0, max_offset=5.0)
    assert ss.num_parameters == 1


def test_num_parameters_2d():
    ss = SearchSpace(x_ctr=np.array([0.0, 0.0]), epsilon=1.0, max_offset=5.0)
    assert ss.num_parameters == 2


def test_num_parameters_3d():
    ss = SearchSpace(x_ctr=np.array([0.0, 0.0, 0.0]), epsilon=1.0, points_per_dim=5)
    assert ss.num_parameters == 3


# ===========================================================================
# zoom_in
# ===========================================================================

def test_zoom_in_halves_epsilon():
    ss = SearchSpace(x_ctr=np.array([0.0, 0.0]), epsilon=2.0, points_per_dim=11)
    ss2 = ss.zoom_in(np.array([1.0, 0.5]), zoom=2.0)
    assert np.allclose(ss2.epsilon, ss.epsilon / 2.0)


def test_zoom_in_preserves_points_per_dim():
    ss = SearchSpace(x_ctr=np.array([0.0, 0.0]), epsilon=2.0, points_per_dim=11)
    ss2 = ss.zoom_in(np.array([1.0, 0.5]), zoom=2.0)
    assert np.all(ss2.points_per_dim == ss.points_per_dim)


def test_zoom_in_new_center():
    ss = SearchSpace(x_ctr=np.array([0.0, 0.0]), epsilon=2.0, points_per_dim=11)
    new_ctr = np.array([3.0, -1.0])
    ss2 = ss.zoom_in(new_ctr, zoom=2.0)
    assert np.allclose(ss2.x_ctr, new_ctr)


def test_zoom_in_returns_new_object():
    ss = SearchSpace(x_ctr=np.array([0.0, 0.0]), epsilon=2.0, points_per_dim=11)
    ss2 = ss.zoom_in(np.array([0.0, 0.0]))
    assert ss2 is not ss


def test_zoom_in_custom_zoom_factor():
    ss = SearchSpace(x_ctr=np.array([0.0, 0.0]), epsilon=4.0, points_per_dim=9)
    ss4 = ss.zoom_in(np.array([0.0, 0.0]), zoom=4.0)
    assert np.allclose(ss4.epsilon, ss.epsilon / 4.0)


# ===========================================================================
# get_extent
# ===========================================================================

def test_get_extent_default_axes():
    x_ctr = np.array([10.0, 20.0])
    ss = SearchSpace(x_ctr=x_ctr, epsilon=1.0, max_offset=5.0)
    ext = ss.get_extent()
    # (x0-o0, x0+o0, x1-o1, x1+o1)
    assert np.allclose(ext, (5.0, 15.0, 15.0, 25.0))


def test_get_extent_multiplier():
    x_ctr = np.array([1000.0, 2000.0])
    ss = SearchSpace(x_ctr=x_ctr, epsilon=100.0, max_offset=500.0)
    ext = ss.get_extent(multiplier=0.001)  # meters to km
    assert np.allclose(ext, (0.5, 1.5, 1.5, 2.5))


def test_get_extent_explicit_axes():
    x_ctr = np.array([1.0, 2.0, 3.0])
    ss = SearchSpace(x_ctr=x_ctr, epsilon=1.0, max_offset=np.array([10.0, 20.0, 30.0]))
    ext = ss.get_extent(axes=[0, 2])
    assert np.allclose(ext, (-9.0, 11.0, -27.0, 33.0))
