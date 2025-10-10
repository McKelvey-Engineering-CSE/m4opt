"""This module contains builtin settings for supported missions."""

from ._core import Mission
from ._rubin import rubin
from ._ultrasat import ultrasat
from ._adapt import adapt
from ._uvex import uvex
from ._ztf import ztf

__all__ = ("Mission", "rubin", "ultrasat", "uvex", "adapt", "ztf")
