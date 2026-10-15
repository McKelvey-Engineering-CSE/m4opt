import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord, concatenate

from ._geodesic import geodesic
from ._healpix import healpix
from ._sinusoidal import sinusoidal
from ._spiral import golden_angle_spiral


def extend_skygrid(skygrid: SkyCoord, fov: u.Quantity) -> SkyCoord:
    """Extend a skygrid beyond 360 degrees.

    Parameters
    ----------
    skygrid : SkyGrid
        The skygrid to extend.
    fov : `~astropy.units.Quantity`
        The field of view of the instrument.

    Returns
    -------
    extended_skygrid : SkyGrid
        The extended skygrid.
    """

    full_circle = 360 * u.deg
    if fov <= full_circle:
        return skygrid

    n_extended = int((fov / full_circle).to_value(u.dimensionless_unscaled))

    extended_skygrid = (
        skygrid if n_extended == 1 else concatenate([skygrid] * n_extended)
    )
    remainder = fov - (n_extended * full_circle)

    if remainder > 0 * u.deg:
        longitude = np.mod(skygrid.spherical.lon.to_value(u.deg), 360)
        extended_skygrid = concatenate(
            (extended_skygrid, skygrid[longitude < remainder.to_value(u.deg)])
        )

    return extended_skygrid


__all__ = ("extend_skygrid", "geodesic", "golden_angle_spiral", "healpix", "sinusoidal")
