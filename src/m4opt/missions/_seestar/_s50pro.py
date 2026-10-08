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

s50pro = {}