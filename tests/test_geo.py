import numpy as np

from ewgeo.utils.constants import speed_of_light
from ewgeo.utils.geo import calc_range, calc_range_diff, calc_doppler, calc_doppler_diff

def test_calc_range():
    """
    Test behavior of calc_range function.
    """

    # Unit test -- one axis changed
    res = calc_range([0., 0.], [1., 0.])
    assert res==1.0

    # Unit test -- two axes changed
    res = calc_range([0., 0., .0], [1., 1., 0.])
    assert res==np.sqrt(2)

    # Unit test -- multidimensional input, check proper broadcasting and calculation
    a = np.zeros((3, 3, 1, 2))
    b = np.ones((3, 1, 2, 1, 4))
    res_val = np.sqrt(3)
    res_shape = (3, 1, 2, 2, 4)
    res = calc_range(a, b)
    # check shape
    assert res.shape == res_shape
    # check value
    assert np.all(res==res_val)

def test_calc_range_diff():

    # Single position
    x_source = np.array([1., 1., 1.])
    x_ref = np.array([0., 0., 0.])
    x_test = np.array([1., 1., 0.])
    res_val = 1 - np.sqrt(3)
    res = calc_range_diff(x_source, x_ref, x_test)
    assert equal_to_tolerance(res, res_val)

    # Arbitrary dimensions
    x_source = np.array([[[1., 2., 3., 4.]], [[1., 2., 3., 4.]]])  # (2, 4)
    x_ref = np.array([[0., 0.], [0., 0.]])  # (2, 2)
    x_test = np.array([[1., -1.], [0.5, 0.5]])  # (2, 2)
    res_shape = (2, 2, 4)
    res_val = np.array([[[-.91421356,-1.02565149,-1.04107857,-1.04708202],
                         [0.64733925, 0.52567484, 0.47434988, 0.44642356]],
                        [[-.91421356,-1.02565149,-1.04107857,-1.04708202],
                         [0.64733925, 0.52567484, 0.47434988, 0.44642356]]])
    res = calc_range_diff(x_source, x_ref, x_test)
    assert res.shape == res_shape
    print(res-res_val)
    assert equal_to_tolerance(res, res_val, tol=1e-4)

def test_calc_doppler():
    # Zero doppler cases
    f0 = 1e9
    src_pos = np.array([0., 0., 0.])
    dst_pos = np.array([1000., 0., 0.])
    src_vel = np.zeros(3)
    dst_vel = np.zeros(3)
    fd = calc_doppler(src_pos, src_vel, dst_pos, dst_vel, f0)
    assert equal_to_tolerance(fd, 0.)

    src_vel = np.random.rand(3)
    dst_vel = src_vel
    fd = calc_doppler(src_pos, src_vel, dst_pos, dst_vel, f0)
    assert equal_to_tolerance(fd, 0.)

    src_pos = np.array([0., 0., 0.])
    dst_pos = np.array([0., 0., 1000.])
    src_vel = np.array([500., 0., 0.])  # sideways motion
    dst_vel = np.zeros(3)
    fd = calc_doppler(src_pos, src_vel, dst_pos, dst_vel, f0)
    assert equal_to_tolerance(fd, 0.)

    # Positive Doppler cases
    src_pos = np.array([0., 0., 0.])
    dst_pos = np.array([1000., 0., 0.])
    src_vel = np.zeros(3)
    dst_vel = np.array([-1000., 0., 0.])  # toward source at 1000 m/s

    fd = calc_doppler(src_pos, src_vel, dst_pos, dst_vel, f0)
    expected_shift = f0 * 1000.0 / speed_of_light
    assert equal_to_tolerance(fd, expected_shift, tol=1e-12)

    src_pos = np.array([0., 0., 0.])
    dst_pos = np.array([10000., 0., 0.])
    src_vel = np.array([1000., 0., 0.])  # toward destination
    dst_vel = np.zeros(3)
    fd = calc_doppler(src_pos, src_vel, dst_pos, dst_vel, f0)
    expected_shift = f0 * 1000.0 / speed_of_light
    assert equal_to_tolerance(fd, expected_shift, tol=1e-12)

    # Negative Doppler Cases
    src_pos = np.array([0., 0., 0.])
    dst_pos = np.array([0., 0., 1000.])
    src_vel = np.zeros(3)
    dst_vel = np.array([0., 0., 1000.])  # away from source
    fd = calc_doppler(src_pos, src_vel, dst_pos, dst_vel, f0)
    expected_shift = -f0 * 1000.0 / speed_of_light
    assert equal_to_tolerance(fd, expected_shift, tol=1e-12)

    # TODO: multi-dimensional case to test array shapes

def test_calc_doppler_diff():
    pass


def equal_to_tolerance(x, y, tol=1e-6) -> bool:
    """
    Shorthand to compare two lists and ensure each entry has an error less than the specified tolerance
    """
    if np.size(x) != np.size(y): return False
    return np.all(np.fabs(x - y) < tol)