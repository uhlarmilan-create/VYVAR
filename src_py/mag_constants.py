"""Shared magnitude-domain scale for relative flux error (2.5 / ln 10)."""

from __future__ import annotations

import math

MAG_ERR_SCALE = 2.5 / math.log(10.0)

#: Gaia BP-RP of the Sun (G2V, Teff 5770 K): Pecaut & Mamajek (2013, ApJS 208, 9), table version 2022.04.16
#: (G10-17: the one solar colour of VYVAR - HRD Teff scale and the field zero point "for a star of solar colour").
SOLAR_BP_RP = 0.823
