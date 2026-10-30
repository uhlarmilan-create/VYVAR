# -*- coding: ascii -*-
"""Gaia DR3 G + BP-RP -> Johnson/Cousins comp magnitudes for OSC band exports.

Coefficient source (single source of truth for pinned a_i and sigma):
  Gaia Data Release 3 documentation release 1.3, CU5 photometric systems,
  section 5.5.1 "Photometric relationships with other photometric systems",
  Table 5.9 "Johnson-Cousins relationships" (Gaia DR3 sources in common with
  Johnson-Cousins). Independent variable X = GBP-GRP; dependent Y = G-X where
  X is Johnson V, B, or Cousins R (table rows G-V, G-B, G-R). Polynomial form:

      Y = G - X = sum_i a_i * (GBP-GRP)^i

  Residual scatter sigmas pinned from the same Table 5.9 sigma column (mag):
  G-V 0.03017, G-B 0.0633, G-R 0.03167.

Validation reference (Landolt-standard independent fit; NOT the coefficient source):
  Ruelas-Mayorga et al. 2025, RAS Techniques & Instruments 4,
  doi:10.1093/rasti/rzaf037 (Crossref-verified authors: Ruelas-Mayorga,
  Macias-Estrada, Sanchez, Paez-Amador, Segura Montero, Nigoche-Netro).
  Abstract reports Landolt residuals smaller than ~0.05 mag (V, R, I) and
  ~0.1 mag (B) for 8 < Mag < 16 - consistent with Table 5.9 sigmas.

Riello et al. 2021 EDR3 Table C.2-style coefficients are exposed as
``RIELLO_EDR3_FALLBACK`` for unit tests only (E3); production OSC exports use
``GDR3_TABLE59_COEFFS`` only.
"""
from __future__ import annotations

import logging
from typing import Final

LOGGER = logging.getLogger(__name__)

JohnsonBand = str  # "V", "B", "RC"

# Table 5.10 applicability of G-V = f(GBP-GRP): -0.5 < GBP-GRP < 5.0 (Gaia DR3 docs 5.5.1; was 5.1 until
# 2026-10-29, Milan). Table 5.10 gives colour ranges only - no G magnitude limit (the 8 < Mag < 16 of
# Ruelas-Mayorga et al. 2025 is their validation sample, not a validity range).
BPRP_MAX: Final[float] = 5.0

# Table 5.9 sigma column (mag), same source as GDR3_TABLE59_COEFFS.
SCATTER_SIGMA: Final[dict[JohnsonBand, float]] = {
    "V": 0.03017,
    "B": 0.0633,
    "RC": 0.03167,
}

# Table 5.9 Johnson-Cousins rows, X = GBP-GRP, Y = G-X (a0..an).
GDR3_TABLE59_COEFFS: Final[dict[JohnsonBand, tuple[float, ...]]] = {
    "V": (-0.02704, 0.01424, -0.2156, 0.01426),
    "B": (0.01448, -0.6874, -0.3604, 0.06718, -0.006061),
    "RC": (-0.02275, 0.3961, -0.1243, -0.01396, 0.003775),
}



COEFFICIENT_CITATION: Final[str] = (
    "Gaia DR3 CU5 Table 5.9 (GBP-GRP -> G-V/G-B/G-R polynomials + sigma column)"
)
VALIDATION_CITATION: Final[str] = (
    "Ruelas-Mayorga et al. 2025 RASTI 4:37 doi:10.1093/rasti/rzaf037 (Landolt validation)"
)
TRANSFORM_CITATION: Final[str] = f"{COEFFICIENT_CITATION}; {VALIDATION_CITATION}"


