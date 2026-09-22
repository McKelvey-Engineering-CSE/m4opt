import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from regions import RectangleSkyRegion
from synphot import Gaussian1D, SpectralElement

from .. import skygrid
from ..constraints import (
    EarthLimbConstraint,
    MoonSeparationConstraint,
    SunSeparationConstraint,
)
from ..dynamics import EigenAxisSlew
from ..observer import TleObserverLocation
from ..synphot import Detector
from ..synphot.background import GalacticBackground, ZodiacalBackground
from ._core import Mission

adapt = Mission(
    name="uvex",
    fov=RectangleSkyRegion(
        center=SkyCoord(0 * u.deg, 0 * u.deg), width=2.5 * u.deg, height=2.5 * u.deg
    ),
    constraints=(
        EarthLimbConstraint(25 * u.deg)
        & SunSeparationConstraint(46 * u.deg)
        & MoonSeparationConstraint(25 * u.deg)
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
    # UVEX will be in a highly elliptical TESS-like orbit.
    # This is the TESS TLE downloaded from Celestrak at 2024-09-10T00:43:57Z.
    observer_location=TleObserverLocation(
        "1 43435U 18038A   24262.33225493 -.00001052  00000+0  00000+0 0  9993",
        "2 43435  51.7454  60.8303 4593193 124.3403   0.2501  0.07594463  1386",
    ),
    # Sky grid optimized for full coverage of the sky by circles circumscribed
    # within the square field of view (so that each field is fully covered
    # at all roll angles).
    skygrid=skygrid.geodesic(7.7 * u.deg**2, class_="III", base="icosahedron"),
    # Made up slew model.
    slew=EigenAxisSlew(
        max_angular_velocity=10 * u.deg / u.s,
        max_angular_acceleration=100 * u.deg / u.s**2,
        settling_time=0 * u.s,
    ),
)
adapt.__doc__ = r"""Sample ADAPT mission, copied from UVEX configuration.
"""
