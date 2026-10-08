import astropy.units as u
import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from regions import RectangleSkyRegion
from synphot import Gaussian1D, SpectralElement

from ...observer import EarthFixedObserverLocation
from .. import skygrid
from ..dynamics import TwoAxisSlew
from ..synphot import Detector
from ..synphot.background import GalacticBackground
from ._core import Mission

FIELD_OF_VIEW = 4.6
ROLLA = EarthLocation.from_geodetic(lon=-91.774237, lat=37.953132, height=330 * u.m)

s30pro = Mission(
    name="s30pro",
    fov=RectangleSkyRegion(
        center=SkyCoord(0 * u.deg, 0 * u.deg), width=FIELD_OF_VIEW * u.deg, height=FIELD_OF_VIEW * u.deg
    ),
    constraints=(
        # EarthLimbConstraint(25 * u.deg)
        # & SunSeparationConstraint(46 * u.deg)
        # & MoonSeparationConstraint(25 * u.deg)
    ),
    detector=Detector(
        npix=4 * np.pi,
        # "This is Nyquist sampled by the 1 arcsec pixels."
        plate_scale=1 * u.arcsec**2,
        # "...an effective aperture of 75cm."
        area=np.pi * np.square(0.5 * 75 * u.cm),
        bandpasses={
            "FUV": SpectralElement(
                Gaussian1D,
                amplitude=0.15,
                mean=1600 * u.angstrom,
                stddev=100 * u.angstrom,
            ),
            "NUV": SpectralElement(
                Gaussian1D,
                amplitude=0.2,
                mean=2300 * u.angstrom,
                stddev=180 * u.angstrom,
            ),
        },
        background=GalacticBackground(),
        # Made up to match plot
        read_noise=2,
        dark_noise=1e-3 * u.Hz,
        gain=0.85,
    ),
    observer_location=EarthFixedObserverLocation(ROLLA),
    # Sky grid optimized for full coverage of the sky by circles circumscribed
    # within the square field of view (so that each field is fully covered
    # at all roll angles).
    skygrid=skygrid.geodesic(7.7 * u.deg**2, class_="III", base="icosahedron"),
    slew=TwoAxisSlew(
        max_angular_velocity=6 * u.deg / u.s,
        max_angular_acceleration=100 * u.deg / u.s**2,
        settling_time=0 * u.s,
        simultaneous_axis_motion=False,
    ),
)
s30pro.__doc__ = r"""SeeStar S30 Pro mission, copied from UVEX configuration."""

