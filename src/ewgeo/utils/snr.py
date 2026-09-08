import numpy as np
import numpy.typing as npt

from .geo import calc_range
from .coordinates import ecef_to_lla, enu_to_lla
from ewgeo.prop.model import get_path_loss, get_free_space_path_loss


def _extract_heights(x_sensor: np.ndarray,
                     x_source: np.ndarray,
                     coord_system: str | None,
                     enu_ref_lla: tuple | None) -> tuple[np.ndarray, float]:
    """
    Return (rx_heights_m, tx_height_m) given sensor and source arrays.

    x_sensor: (n_dim, n_sensor)
    x_source: (n_dim, 1)
    Returns rx_heights as (n_sensor,) array, tx_height as a scalar.
    """
    cs = coord_system.lower() if coord_system is not None else None

    if cs == 'enu':
        if enu_ref_lla is not None:
            lat_ref, lon_ref, alt_ref = enu_ref_lla
            # Source
            _, _, tx_ht = enu_to_lla(
                x_source[0, 0], x_source[1, 0], x_source[2, 0],
                lat_ref, lon_ref, alt_ref,
            )
            # Sensors
            _, _, rx_hts = enu_to_lla(
                x_sensor[0, :], x_sensor[1, :], x_sensor[2, :],
                lat_ref, lon_ref, alt_ref,
            )
        else:
            # Up component is a good approximation of height above local origin
            tx_ht = float(x_source[2, 0])
            rx_hts = x_sensor[2, :]
    elif cs == 'ecef':
        _, _, tx_ht = ecef_to_lla(x_source[0, 0], x_source[1, 0], x_source[2, 0])
        _, _, rx_hts = ecef_to_lla(x_sensor[0, :], x_sensor[1, :], x_sensor[2, :])
    else:
        raise ValueError(f"Unrecognised coord_system '{coord_system}'.")

    return np.atleast_1d(np.asarray(rx_hts, dtype=float)), float(tx_ht)


def compute_snr_per_sensor(x_sensor: npt.ArrayLike,
                           x_source: npt.ArrayLike,
                           erp_dbw: float,
                           mds_dbw: float,
                           freq_hz: float,
                           coord_system: str | None = None,
                           enu_ref_lla: tuple | None = None,
                           include_atm_loss: bool = True,
                           atmosphere=None) -> npt.NDArray[np.float64]:
    """
    Compute the received SNR [dB] at each sensor for a given source position.

    Propagation model selected based on coord_system and position dimensionality:

    - coord_system=None or 2-D positions: free-space path loss only, no atmospheric
      correction.  Heights are unavailable so get_path_loss cannot be used.
    - coord_system='enu', enu_ref_lla=None: the Up (3rd-row) component of each
      position is treated as height above the local ENU origin.
    - coord_system='enu', enu_ref_lla=(lat, lon, alt): enu_to_lla() converts each
      position to MSL altitude for accurate height-above-ground values.
    - coord_system='ecef': ecef_to_lla() extracts MSL altitude from ECEF coordinates.

    In all 3-D cases, get_path_loss() selects free-space or two-ray propagation
    based on the Fresnel zone, and optionally applies atmospheric absorption.

    :param x_sensor: (n_dim, n_sensor) sensor positions [m]
    :param x_source: (n_dim,) or (n_dim, 1) source position [m]
    :param erp_dbw: Effective radiated power [dBW]
    :param mds_dbw: Minimum detectable signal / noise floor [dBW]
    :param freq_hz: Carrier frequency [Hz]
    :param coord_system: None | 'enu' | 'ecef'.  When None (or when positions
        are 2-D), free-space path loss without atmospheric correction is used.
    :param enu_ref_lla: (lat_deg, lon_deg, alt_m) of the ENU frame origin.
        Only used when coord_system='enu'; enables accurate MSL altitude via
        enu_to_lla().  If omitted, the Up component is used directly as height.
    :param include_atm_loss: Passed to get_path_loss when heights are available.
        Ignored for the 2-D / no-coord-system path.
    :param atmosphere: Optional atmosphere struct for get_path_loss.
    :return: (n_sensor,) array of SNR values [dB]
    """
    x_sensor = np.asarray(x_sensor, dtype=float)
    x_source = np.asarray(x_source, dtype=float)
    if x_source.ndim == 1:
        x_source = x_source[:, np.newaxis]  # (n_dim, 1)

    r = calc_range(x_sensor, x_source)
    r = np.atleast_1d(np.squeeze(r))  # (n_sensor,)

    n_dim = x_sensor.shape[0]
    has_3d = n_dim >= 3 and coord_system is not None

    if has_3d:
        rx_hts, tx_ht = _extract_heights(x_sensor, x_source, coord_system, enu_ref_lla)
        path_loss_db = get_path_loss(r, freq_hz, tx_ht, rx_hts,
                                     include_atm_loss=include_atm_loss,
                                     atmosphere=atmosphere)
    else:
        # No height information — use free-space only, no atmospheric correction
        path_loss_db = get_free_space_path_loss(r, freq_hz, include_atm_loss=False)

    # SNR = ERP - path_loss + MDS  (MDS is the noise floor reference)
    return erp_dbw - path_loss_db + mds_dbw
