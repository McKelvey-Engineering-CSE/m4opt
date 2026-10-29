import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation, SkyCoord
from regions import RectangleSkyRegion
from synphot import Empirical1D, SpectralElement

from ... import skygrid
from ...constraints import AltitudeConstraint
from ...dynamics import AngularMotionProfile, TwoAxisSlew
from ...observer import EarthFixedObserverLocation
from ...synphot import Detector
from ...synphot.background import GalacticBackground, ZodiacalBackground
from .._core import Mission

ROLLA = EarthLocation.from_geodetic(lon=-91.774237, lat=37.953132, height=330 * u.m)

s30pro = Mission(
    name="s30pro",
    fov=RectangleSkyRegion(
        center=SkyCoord(0 * u.deg, 0 * u.deg), width=2.4 * u.deg, height=4.3 * u.deg
    ),
    constraints=AltitudeConstraint(-90 * u.deg, 87.5 * u.deg),
    detector=Detector(
        npix=4 * np.pi,
        # IMX585 2.9 um pixels and the S30 Pro's approximately 150 mm focal
        # length give about 4.0 arcsec per pixel.
        plate_scale=((2.9 * u.um / (150 * u.mm)) * u.rad).to(u.arcsec) ** 2,
        area=np.pi * np.square(0.5 * 30 * u.mm),
        bandpasses={
            # Approximate visible throughput; replace with measured camera,
            # filter, and optical throughput when available.
            "VIS": SpectralElement(
                Empirical1D,
                points=[3500, 4500, 6000, 8000, 9500, 10500] * u.angstrom,
                lookup_table=[0, 0.4, 0.8, 0.7, 0.2, 0],
            ),
        },
        background=GalacticBackground() + ZodiacalBackground(),
        # Approximate uncooled-camera values; calibrate from dark and bias
        # frames for the actual S30 Pro electronics and operating mode.
        read_noise=8,
        dark_noise=3 / u.hr,
        gain=1/6.2,
    ),
    observer_location=EarthFixedObserverLocation(ROLLA),
    # Sky grid optimized for full coverage of the sky by circles circumscribed
    # within the square field of view (so that each field is fully covered
    # at all roll angles).
    skygrid=skygrid.extend_skygrid(skygrid.geodesic(1.5 * u.deg**2, class_="III", base="icosahedron"), 500 * u.deg),
    slew=TwoAxisSlew(
        longitudinal=AngularMotionProfile(
            max_angular_velocity=6 * u.deg / u.s,
            max_angular_acceleration=100 * u.deg / u.s**2,
            settling_time=0 * u.s,
        ),
        latitudinal=AngularMotionProfile(
            max_angular_velocity=6 * u.deg / u.s,
            max_angular_acceleration=100 * u.deg / u.s**2,
            settling_time=0 * u.s,
        ),
        simultaneous_axis_motion=False,
    ),
)
s30pro.__doc__ = r"""SeeStar S30 Pro mission, copied from UVEX configuration."""

