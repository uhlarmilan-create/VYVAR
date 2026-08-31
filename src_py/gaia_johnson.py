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
import math
from dataclasses import dataclass
from typing import Final

LOGGER = logging.getLogger(__name__)

JohnsonBand = str  # "V", "B", "RC"

# Table 5.10 applicability for Johnson-Cousins GBP-GRP polynomials (Gaia DR3 docs).
BPRP_MIN: Final[float] = -0.5
BPRP_MAX: Final[float] = 5.1
G_MAG_MIN: Final[float] = 8.0
G_MAG_MAX: Final[float] = 16.0

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

# Back-compat alias (pre-push name; coefficients are GDR3 Table 5.9, not a separate fit).
RODRIGUEZ2025_LANDOLT = GDR3_TABLE59_COEFFS

# Riello et al. 2021 EDR3 Table C.2 / DR2 documentation (tests only).
RIELLO_EDR3_FALLBACK: Final[dict[JohnsonBand, tuple[float, ...]]] = {
    "V": (-0.0176, -0.00686, -0.1732),
    "B": (0.01448, -0.6874, -0.3604, 0.06718, -0.006061),
    "RC": (-0.02126, 0.4077, -0.1772, -0.02347, 0.00571),
}

OSC_BAND_TO_JOHNSON: Final[dict[str, JohnsonBand]] = {
    "TG": "V",
    "TB": "B",
    "TR": "RC",
}

COEFFICIENT_CITATION: Final[str] = (
    "Gaia DR3 CU5 Table 5.9 (GBP-GRP -> G-V/G-B/G-R polynomials + sigma column)"
)
VALIDATION_CITATION: Final[str] = (
    "Ruelas-Mayorga et al. 2025 RASTI 4:37 doi:10.1093/rasti/rzaf037 (Landolt validation)"
)
TRANSFORM_CITATION: Final[str] = f"{COEFFICIENT_CITATION}; {VALIDATION_CITATION}"


@dataclass(frozen=True)
class JohnsonTransformResult:
    ok: bool
    johnson_mag: float
    johnson_mag_err: float
    johnson_band: str
    reason: str = ""
    source: str = "gdr3_table59"


def _eval_poly1d(coeffs: tuple[float, ...], x: float) -> float:
    """Horner-style 1-D polynomial. Not the 2-D astrometry evaluator."""
    return float(sum(a * (x**i) for i, a in enumerate(coeffs)))


def _combine_mag_err(g_err: float | None, scatter: float) -> float:
    parts: list[float] = [float(scatter)]
    if g_err is not None and math.isfinite(g_err) and g_err > 0:
        parts.append(float(g_err))
    return float(math.hypot(*parts))


def transform_gaia_to_johnson(
    g_mag: float,
    bp_rp: float,
    band: JohnsonBand,
    *,
    g_mag_err: float | None = None,
    coeff_set: dict[JohnsonBand, tuple[float, ...]] | None = None,
    bprp_min: float = BPRP_MIN,
    bprp_max: float = BPRP_MAX,
    g_min: float = G_MAG_MIN,
    g_max: float = G_MAG_MAX,
) -> JohnsonTransformResult:
    """Transform Gaia G + BP-RP to Johnson V, B, or Cousins R_C."""
    coeffs_map = coeff_set if coeff_set is not None else GDR3_TABLE59_COEFFS
    jb = str(band or "").strip().upper()
    if jb == "R":
        jb = "RC"
    coeffs = coeffs_map.get(jb)
    if coeffs is None:
        return JohnsonTransformResult(
            ok=False,
            johnson_mag=float("nan"),
            johnson_mag_err=float("nan"),
            johnson_band=jb,
            reason=f"unsupported band {band!r}",
        )
    g = float(g_mag)
    c = float(bp_rp)
    if not (math.isfinite(g) and math.isfinite(c)):
        return JohnsonTransformResult(
            ok=False,
            johnson_mag=float("nan"),
            johnson_mag_err=float("nan"),
            johnson_band=jb,
            reason="non-finite G or BP-RP",
        )
    if g < g_min or g > g_max:
        return JohnsonTransformResult(
            ok=False,
            johnson_mag=float("nan"),
            johnson_mag_err=float("nan"),
            johnson_band=jb,
            reason=f"G={g:.3f} outside [{g_min},{g_max}]",
        )
    if c < bprp_min or c > bprp_max:
        return JohnsonTransformResult(
            ok=False,
            johnson_mag=float("nan"),
            johnson_mag_err=float("nan"),
            johnson_band=jb,
            reason=f"BP-RP={c:.3f} outside [{bprp_min},{bprp_max}]",
        )
    g_minus_x = _eval_poly1d(coeffs, c)
    x_mag = g - g_minus_x
    scatter = float(SCATTER_SIGMA.get(jb, 0.05))
    err = _combine_mag_err(g_mag_err, scatter)
    return JohnsonTransformResult(
        ok=True,
        johnson_mag=float(x_mag),
        johnson_mag_err=err,
        johnson_band=jb,
        source="gdr3_table59" if coeff_set is None else "custom",
    )


def johnson_band_for_osc_aavso_token(aavso_token: str) -> JohnsonBand | None:
    return OSC_BAND_TO_JOHNSON.get(str(aavso_token or "").strip().upper())


def transform_comp_row_for_osc_band(
    row: dict[str, object] | object,
    aavso_band_token: str,
    *,
    log_exclusions: bool = True,
) -> JohnsonTransformResult:
    """Map one comp/check row (Gaia G catalog mag) to the Johnson band for an OSC export."""
    jb = johnson_band_for_osc_aavso_token(aavso_band_token)
    if jb is None:
        return JohnsonTransformResult(
            ok=False,
            johnson_mag=float("nan"),
            johnson_mag_err=float("nan"),
            johnson_band="",
            reason=f"no Johnson mapping for band token {aavso_band_token!r}",
        )

    def _get(key: str) -> object:
        if isinstance(row, dict):
            return row.get(key)
        try:
            return row[key]  # type: ignore[index]
        except (KeyError, TypeError, IndexError):
            return None

    g = _get("mag")
    if g is None:
        g = _get("phot_g_mean_mag")
    bp_rp = _get("bp_rp")
    g_err = _get("mag_err")
    if g_err is None:
        g_err = _get("phot_g_mean_mag_error")
    res = transform_gaia_to_johnson(
        float(pd_numeric(g)),
        float(pd_numeric(bp_rp)),
        jb,
        g_mag_err=float(pd_numeric(g_err)) if pd_numeric(g_err) is not None else None,
    )
    if log_exclusions and not res.ok:
        cid = str(_get("catalog_id") or _get("name") or "?")
        LOGGER.warning(
            "[OSC-EXPORT] comp %s excluded from %s ensemble: %s",
            cid,
            jb,
            res.reason,
        )
    return res


def pd_numeric(v: object) -> float | None:
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None
