import numpy as np

import ewgeo.prop.model as prop
from ewgeo.utils.constants import speed_of_light


def equal_to_tolerance(x, y, tol=1e-6):
    if np.size(x) != np.size(y):
        return False
    return np.all(np.fabs(np.array(x) - np.array(y)) < tol)


# ===========================================================================
# get_free_space_path_loss
# ===========================================================================

def test_fspl_known_value():
    """FSPL = 20*log10(4π*R/λ) = 20*log10(4π*R*f/c)."""
    R = 1000.0   # m
    f = 1e9      # Hz
    lam = speed_of_light / f
    expected = 20 * np.log10(4 * np.pi * R / lam)
    result = prop.get_free_space_path_loss(R, f, include_atm_loss=False)
    assert equal_to_tolerance(result, expected, tol=1e-6)


def test_fspl_increases_with_range():
    f = 1e9
    l1 = prop.get_free_space_path_loss(1e3, f, include_atm_loss=False)
    l2 = prop.get_free_space_path_loss(1e4, f, include_atm_loss=False)
    assert l2 > l1


def test_fspl_doubles_range_adds_6dB():
    """Doubling range adds exactly 6 dB in free space."""
    f = 1e9
    l1 = prop.get_free_space_path_loss(1e3, f, include_atm_loss=False)
    l2 = prop.get_free_space_path_loss(2e3, f, include_atm_loss=False)
    assert equal_to_tolerance(l2 - l1, 20 * np.log10(2), tol=1e-6)


def test_fspl_increases_with_frequency():
    R = 1000.0
    l_low  = prop.get_free_space_path_loss(R, 1e8, include_atm_loss=False)
    l_high = prop.get_free_space_path_loss(R, 1e9, include_atm_loss=False)
    assert l_high > l_low


def test_fspl_positive_dB():
    result = prop.get_free_space_path_loss(1000.0, 1e9, include_atm_loss=False)
    assert result > 0


# ===========================================================================
# get_two_ray_path_loss
# ===========================================================================

def test_two_ray_known_value():
    """L = 10*log10(R^4 / (ht^2 * hr^2))."""
    R = 1e4
    ht = 10.0
    hr = 10.0
    expected = 10 * np.log10(R ** 4 / (ht ** 2 * hr ** 2))
    result = prop.get_two_ray_path_loss(R, 1e9, ht, hr, include_atm_loss=False)
    assert equal_to_tolerance(result, expected, tol=1e-6)


def test_two_ray_increases_with_range():
    ht = hr = 10.0
    l1 = prop.get_two_ray_path_loss(1e3, 1e9, ht, hr, include_atm_loss=False)
    l2 = prop.get_two_ray_path_loss(1e4, 1e9, ht, hr, include_atm_loss=False)
    assert l2 > l1


def test_two_ray_symmetric_heights():
    """Swapping tx/rx heights should not change path loss."""
    R, f = 5e3, 1e9
    l1 = prop.get_two_ray_path_loss(R, f, 10.0, 20.0, include_atm_loss=False)
    l2 = prop.get_two_ray_path_loss(R, f, 20.0, 10.0, include_atm_loss=False)
    assert equal_to_tolerance(l1, l2)


# ===========================================================================
# get_fresnel_zone
# ===========================================================================

def test_fresnel_zone_known_formula():
    """FZ = 4π*ht*hr / λ = 4π*ht*hr*f / c."""
    f, ht, hr = 1e9, 10.0, 10.0
    expected = 4 * np.pi * ht * hr * f / speed_of_light
    result = prop.get_fresnel_zone(f, ht, hr)
    assert equal_to_tolerance(result, expected)


def test_fresnel_zone_positive():
    assert prop.get_fresnel_zone(1e9, 10.0, 20.0) > 0


def test_fresnel_zone_increases_with_height():
    fz_low  = prop.get_fresnel_zone(1e9, 5.0, 5.0)
    fz_high = prop.get_fresnel_zone(1e9, 20.0, 20.0)
    assert fz_high > fz_low


# ===========================================================================
# compute_radar_horizon
# ===========================================================================

def test_radar_horizon_positive():
    result = prop.compute_radar_horizon(100.0, 100.0)
    assert result > 0


def test_radar_horizon_increases_with_height():
    rh_low  = prop.compute_radar_horizon(10.0, 10.0)
    rh_high = prop.compute_radar_horizon(100.0, 100.0)
    assert rh_high > rh_low


def test_radar_horizon_symmetric():
    """Horizon distance should be the same if tx/rx heights are swapped."""
    rh1 = prop.compute_radar_horizon(50.0, 200.0)
    rh2 = prop.compute_radar_horizon(200.0, 50.0)
    assert equal_to_tolerance(rh1, rh2)


def test_radar_horizon_greater_with_effective_earth():
    """4/3 Earth radius model gives longer horizon than true radius."""
    rh_eff  = prop.compute_radar_horizon(100.0, 100.0, use_four_thirds_radius=True)
    rh_true = prop.compute_radar_horizon(100.0, 100.0, use_four_thirds_radius=False)
    assert rh_eff > rh_true


def test_radar_horizon_zero_height_is_zero():
    assert prop.compute_radar_horizon(0.0, 0.0) == 0.0


# ===========================================================================
# get_knife_edge_path_loss
# ===========================================================================

def test_knife_edge_below_los_is_zero():
    """Obstruction below line-of-sight (nu <= 0) produces zero loss."""
    # ht_above_los negative → nu < 0
    result = prop.get_knife_edge_path_loss(1000.0, 1000.0, -5.0)
    assert result[0] == 0.0, f"Expected 0 dB loss, got {result[0]}"


def test_knife_edge_mid_range_formula():
    """For 0 < nu <= 2.4, loss = 6 + 9*nu - 1.27*nu^2."""
    # Choose dist_tx = dist_rx = 1000 → nu = ht*sqrt(2)/2
    # Set ht = sqrt(2) → nu = 1.0
    ht = np.sqrt(2)
    result = prop.get_knife_edge_path_loss(1000.0, 1000.0, ht)
    nu = 1.0
    expected = 6 + 9 * nu - 1.27 * nu ** 2
    assert equal_to_tolerance(result[0], expected, tol=1e-6), \
        f"Expected {expected:.4f} dB, got {result[0]:.4f} dB"


def test_knife_edge_large_nu_formula():
    """For nu > 2.4, loss = 13 + 20*log10(nu)."""
    # Set ht = 3*sqrt(2) → nu = 3.0
    ht = 3.0 * np.sqrt(2)
    result = prop.get_knife_edge_path_loss(1000.0, 1000.0, ht)
    nu = 3.0
    expected = 13 + 20 * np.log10(nu)
    assert equal_to_tolerance(result[0], expected, tol=1e-6), \
        f"Expected {expected:.4f} dB, got {result[0]:.4f} dB"


def test_knife_edge_positive_obstruction_raises_loss():
    """Obstruction above LOS (nu > 0) must produce positive loss."""
    result = prop.get_knife_edge_path_loss(1000.0, 1000.0, 5.0)
    assert result[0] > 0, f"Expected positive loss, got {result[0]}"


def test_knife_edge_increases_with_obstruction():
    """Higher obstruction → greater knife-edge loss."""
    r1 = prop.get_knife_edge_path_loss(1000.0, 1000.0, 1.0)
    r2 = prop.get_knife_edge_path_loss(1000.0, 1000.0, 5.0)
    assert r2[0] > r1[0], f"Expected loss to increase with obstruction height"


def test_knife_edge_vectorized():
    """Vectorized ht_above_los input returns array with correct shape."""
    hts = np.array([-1.0, np.sqrt(2), 3.0 * np.sqrt(2)])
    result = prop.get_knife_edge_path_loss(1000.0, 1000.0, hts)
    assert result.shape == (3,), f"Expected shape (3,), got {result.shape}"
    assert result[0] == 0.0, "Below-LOS region should be 0 dB"
    assert result[1] > 0, "Mid-range region should be positive"
    assert result[2] > 0, "Large-nu region should be positive"


# ===========================================================================
# get_path_loss
# ===========================================================================

def test_path_loss_free_space_below_fresnel():
    """Below the Fresnel zone, get_path_loss should equal free-space loss."""
    f, ht, hr = 1e9, 10.0, 10.0
    # fz = 4π*10*10*1e9/c ≈ 4189 m; use 1000 m (well below fz)
    r = np.array([1000.0])
    result = prop.get_path_loss(r, f, ht, hr, include_atm_loss=False)
    expected = prop.get_free_space_path_loss(r, f, include_atm_loss=False)
    assert equal_to_tolerance(result[0], expected[0], tol=1e-6), \
        f"Expected free-space loss {expected[0]:.2f} dB, got {result[0]:.2f} dB"


def test_path_loss_two_ray_above_fresnel():
    """Above the Fresnel zone, get_path_loss should equal two-ray loss."""
    f, ht, hr = 1e9, 10.0, 10.0
    # fz ≈ 4189 m; use 10000 m (well above fz)
    r = np.array([10000.0])
    result = prop.get_path_loss(r, f, ht, hr, include_atm_loss=False)
    expected = prop.get_two_ray_path_loss(r, f, ht, hr, include_atm_loss=False)
    assert equal_to_tolerance(result[0], expected[0], tol=1e-6), \
        f"Expected two-ray loss {expected[0]:.2f} dB, got {result[0]:.2f} dB"


def test_path_loss_positive():
    """Path loss must be positive."""
    r = np.array([5000.0])
    result = prop.get_path_loss(r, 1e9, 10.0, 10.0, include_atm_loss=False)
    assert result[0] > 0, f"Expected positive path loss, got {result[0]}"


def test_path_loss_increases_with_range():
    """Path loss should increase monotonically with range."""
    f, ht, hr = 1e9, 10.0, 10.0
    r = np.array([500.0, 1000.0, 5000.0, 10000.0])
    result = prop.get_path_loss(r, f, ht, hr, include_atm_loss=False)
    assert np.all(np.diff(result) > 0), "Path loss should increase with range"
