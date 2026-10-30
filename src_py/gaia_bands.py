# -*- coding: ascii -*-
"""Gaia DR3 G + (BP-RP) -> magnitude in the observed band (BAND-2026-10).

Coefficients and validity ranges: Gaia DR3 documentation release 1.3, CU5,
section 5.5.1 "Photometric relationships with other photometric systems":

* Table 5.8 SDSS12 rows, X = G_BP - G_RP, Y = G - x:
    G - g = 0.2199 - 0.6365 X - 0.1548 X^2 + 0.0064 X^3           sigma 0.0745
    G - r = -0.09837 + 0.08592 X + 0.1907 X^2 - 0.1701 X^3 + 0.02263 X^4   sigma 0.03776
    G - i = -0.293 + 0.6404 X - 0.09609 X^2 - 0.002104 X^3        sigma 0.04092
  Table 5.10 ranges: g 0.3 < X < 3.0, r 0.0 < X < 3.0, i 0.5 < X < 2.0;
  notes f, g: above X = 2.0 the g and r relations are valid only for M giants,
  so VYVAR uses X <= 2.0 for g and r.
* Table 5.9 Johnson-Cousins rows (V, B, R_C) are in :mod:`gaia_johnson`.

A magnitude outside the validity range is returned as NaN (no extrapolation).
"""
from __future__ import annotations

import math
from typing import Final

from band_classify import (
    PhotometricBand,
    classify_photometric_band,
    obs_group_first_token,
    standard_band_of_filter,
)
from gaia_johnson import BPRP_MAX as _JC_XMAX, GDR3_TABLE59_COEFFS, SCATTER_SIGMA as _JC_SIGMA

# band -> (coefficients a0..an of G - band in X = BP-RP, X_min, X_max, sigma)
_SDSS: Final[dict[str, tuple[tuple[float, ...], float, float, float]]] = {
    "g": ((0.2199, -0.6365, -0.1548, 0.0064), 0.3, 2.0, 0.0745),
    "r": ((-0.09837, 0.08592, 0.1907, -0.1701, 0.02263), 0.0, 2.0, 0.03776),
    "i": ((-0.293, 0.6404, -0.09609, -0.002104), 0.5, 2.0, 0.04092),
}
# Johnson-Cousins (Table 5.9). Ranges: V as gaia_johnson (Table 5.10); B and R_C are
# valid beyond 1.75 / 2.0 only for M giants (Table 5.10 notes k, j).
_JC: Final[dict[str, tuple[tuple[float, ...], float, float, float]]] = {
    "V": (GDR3_TABLE59_COEFFS["V"], -0.5, _JC_XMAX, _JC_SIGMA["V"]),
    "B": (GDR3_TABLE59_COEFFS["B"], -0.5, 1.75, _JC_SIGMA["B"]),
    "R": (GDR3_TABLE59_COEFFS["RC"], 0.0, 2.0, _JC_SIGMA["RC"]),
}

# G08-13: filter name -> standard band comes from the ONE table (band_classify.standard_band_of_filter);
# this maps a standard band to the zero-point band of the Gaia relations above. Bands without a
# relation here (Sloan u/z, Johnson U, I_C, R_J, I_J) -> "G" (no transform).
_ZP_OF_STANDARD_BAND: Final[dict[str, str]] = {"g": "g", "r": "r", "i": "i", "V": "V", "B": "B", "Rc": "R"}


def band_for_filter(filter_name: str | None) -> str:
    """Band key ('g','r','i','V','B','R') for a FITS FILTER value; 'G' when unknown/clear."""
    return _ZP_OF_STANDARD_BAND.get(standard_band_of_filter(filter_name) or "", "G")


def band_mag_from_gaia(g_mag: float, bp_rp: float, band: str) -> tuple[float, float]:
    """(magnitude in ``band``, sigma of the relation). NaN when outside the valid colour range.

    ``band == 'G'`` returns G itself with sigma 0.
    """
    try:
        g = float(g_mag)
    except (TypeError, ValueError):
        return float("nan"), float("nan")
    if band == "G":
        return (g, 0.0) if math.isfinite(g) else (float("nan"), float("nan"))
    row = _SDSS.get(band) or _JC.get(band)
    if row is None:
        return float("nan"), float("nan")
    coeffs, xmin, xmax, sigma = row
    try:
        x = float(bp_rp)
    except (TypeError, ValueError):
        return float("nan"), float("nan")
    if not (math.isfinite(g) and math.isfinite(x)) or not (xmin < x <= xmax):
        return float("nan"), float("nan")
    g_minus = sum(a * x**i for i, a in enumerate(coeffs))
    return g - g_minus, sigma


def band_for_obs_group(obs_group: str | None) -> str:
    """Band key from an observation group name such as ``g_120_4`` or ``g|120|4``."""
    return band_for_filter(obs_group_first_token(str(obs_group or "")))


# ---------------------------------------------------------------------------
# BANDAUTH-2026-10: one authority for filter -> zero-point band -> AAVSO code.
# ---------------------------------------------------------------------------
from dataclasses import dataclass  # noqa: E402


@dataclass(frozen=True)
class FilterBand:
    """How a FITS filter is calibrated and reported.

    zp_band        band of the comparison catalogue magnitudes ('g','r','i','V','B','R');
                   'G' = no supported band (Gaia G scale, not reportable to AAVSO).
    aavso_code     AAVSO filter code; '' when the filter cannot be reported.
    transformable  True when a colour term to the standard band is meaningful
                   (real standard filter); False for clear / tri-colour / unknown.
    label          human readable band name for reports.
    """

    filter_token: str
    zp_band: str
    aavso_code: str
    transformable: bool
    label: str

    @property
    def blend_band(self) -> str:
        """Band of the neighbour-light (blend) flux ratios. BLEND-G-2026-10 (Milan 2026-10-21): the
        catalogue reference is Gaia, so a clear / tri-colour / unknown filter (no transform) uses
        Gaia G; a standard filter its own band."""
        return self.zp_band if self.transformable else "G"


_ZP_LABEL: Final[dict[str, str]] = {
    "g": "Sloan g", "r": "Sloan r", "i": "Sloan i",
    "V": "Johnson V", "B": "Johnson B", "R": "Cousins R", "G": "Gaia G",
}
_ZP_CODE: Final[dict[str, str]] = {"g": "SG", "r": "SR", "i": "SI", "V": "V", "B": "B", "R": "R"}
# Tri-colour (DSLR / OSC) channels: reported with their own AAVSO code on the
# corresponding Johnson zero point, never transformed.
_TRICOLOUR: Final[dict[str, tuple[str, str]]] = {"tg": ("V", "TG"), "tb": ("B", "TB"), "tr": ("R", "TR")}


def filter_band(filter_name: str | None) -> FilterBand:
    """Single source of truth: FITS FILTER token -> :class:`FilterBand`."""
    raw = str(filter_name or "").strip().replace(" ", "")
    low = raw.lower()
    if low in _TRICOLOUR:
        zp, code = _TRICOLOUR[low]
        return FilterBand(raw, zp, code, False, f"{_ZP_LABEL[zp]} ({code}, untransformed)")
    band = band_for_filter(raw)
    if band != "G":
        return FilterBand(raw, band, _ZP_CODE[band], True, _ZP_LABEL[band])
    cls = classify_photometric_band("", fits_filter=raw)
    if cls in (PhotometricBand.CLEAR_UNFILTERED, PhotometricBand.LUMINANCE):
        # AAVSO 'CV': unfiltered / luminance with V comparison magnitudes, untransformed.
        return FilterBand(raw, "V", "CV", False, "Clear on Johnson V zero point (CV)")
    return FilterBand(raw, "G", "", False, "Gaia G (filter not supported for AAVSO)")


def filter_band_for_obs_group(obs_group: str | None) -> FilterBand:
    """:func:`filter_band` for an observation group name such as ``g_120_4``.

    OSC channel groups resolve to their tri-colour token (TR/TG/TB) via :mod:`osc_align`.
    """
    try:
        from osc_align import obs_group_band_token, parse_osc_channel  # noqa: PLC0415

        if parse_osc_channel(str(obs_group or "")):
            tok = obs_group_band_token(str(obs_group or ""))
            if tok in ("TR", "TG", "TB"):
                return filter_band(tok)
    except Exception:  # noqa: BLE001
        pass
    return filter_band(obs_group_first_token(str(obs_group or "")))


def zp_mag_from_gaia(g_mag: float, bp_rp: float, fb: FilterBand) -> float:
    """Catalogue magnitude of a star in the zero-point band of ``fb`` (NaN outside range)."""
    return band_mag_from_gaia(g_mag, bp_rp, fb.zp_band)[0]


def lc_uncalibrated(lc_dir, target_cid: str, summary_band: str | None = None) -> bool:
    """UNCAL-LC-2026-10: True when the target's light curve has no band zero point
    (band_system NONE: no comparison star with a valid band magnitude). Source: the
    target's comp_quality json (zero_point.band / band_system), else ``summary_band``.
    Such a light curve is the differential delta_mag - shown, never exported."""
    import json as _json  # noqa: PLC0415
    from pathlib import Path as _Path  # noqa: PLC0415

    band = ""
    try:
        raw = _json.loads((_Path(lc_dir) / f"comp_quality_{target_cid}.json").read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            zp = raw.get("zero_point") if isinstance(raw.get("zero_point"), dict) else {}
            band = str(zp.get("band") or raw.get("band_system") or "")
    except (OSError, ValueError):
        band = ""
    if not band and summary_band is not None:
        band = str(summary_band or "")
    return band.strip().upper() == "NONE"



def relation_bp_rp_range(band: str) -> tuple[float, float]:
    """(BP-RP min, max) where the Gaia DR3 -> ``band`` relation is used (NaN, NaN if none)."""
    row = _SDSS.get(band) or _JC.get(band)
    if row is None:
        return float("nan"), float("nan")
    return float(row[1]), float(row[2])


def zero_point_note_text(note: str, band: str, gz: dict | None, *, short: bool = False) -> str:
    """GSPC-ZP-2026-10 (texts approved by Milan 2026-10-11): calibration note for PDF / exports.

    note "A": no standard zero point; "B": GSPC zero point differs by >= 0.02 mag;
    "B-heuristic": GSPC not available, red comps in the Galactic plane. ``gz`` is the
    ``gspc_zero_point`` record of comp_quality json. ``short`` = AAVSO NOTES sub-field."""
    gz = dict(gz or {})
    lo, hi = relation_bp_rp_range(band)
    med, cmin, cmax = gz.get("comp_bp_rp_median"), gz.get("comp_bp_rp_min"), gz.get("comp_bp_rp_max")
    b = gz.get("galactic_b_deg")
    diff, n = gz.get("diff_vs_std_mag"), int(gz.get("n_comp") or 0)
    sig = gz.get("sigma_data") if gz.get("sigma_data") is not None else gz.get("sigma_model")
    has_g = gz.get("C") is not None
    if note == "A" and short:
        t = "ZPNOTE=not calibrated (differential)"
        if cmin is not None and cmax is not None:
            t += f"; comps BP-RP {cmin:.2f}-{cmax:.2f} outside Gaia->{band} range"
        if _gspc_unavailable_text(gz):
            t += "; GSPC unavailable"
        return t
    if note == "A":
        from gspc_zp import MIN_COMPS as _gspc_min  # noqa: PLC0415

        t = (f"No comparison star in the ensemble has a valid {band} magnitude from Gaia DR3 G, BP-RP "
             f"(relation valid for BP-RP <= {hi:.1f}; Riello et al. 2021).")
        if cmin is not None and cmax is not None:
            t += f" Comparison stars BP-RP {cmin:.2f}-{cmax:.2f}."
        if has_g and n >= _gspc_min:
            # GSPC-STD-2026-10 (Milan 2026-10-11, option a): standard magnitudes on the GSPC scale.
            t += (f" Zero point from Gaia DR3 synthetic photometry (GSPC; Gaia Collaboration, Montegriffo et al. 2023),"
                  f" {n} comparison stars" + (f", sigma_ZP {1000 * float(sig):.0f} mmag" if sig is not None else "")
                  + f"; exported as standard {band} magnitudes (AAVSO NOTES ZP=GaiaDR3-GSPC)."
                  f" Plot: delta_mag (left), {band} GSPC (right).")
            return t
        # UNCAL-EXPORT-2026-10 (Milan 2026-10-11): without GSPC exported as differential magnitudes.
        # REPORT-AUDIT-2026-10 (Milan 2026-10-21): differential light curves go to VarAstro only
        t += (" Light curve not calibrated: differential magnitudes (delta_mag = target - ensemble), exported to"
              " VarAstro only (dif mag); no AAVSO file.")
        _un = _gspc_unavailable_text(gz)
        if _un:
            t += _un
        else:
            t += (f" GSPC zero point not available: {n} comparison star{'' if n == 1 else 's'} with an accepted"
                  f" Gaia DR3 synthetic {band} magnitude (at least {_gspc_min} needed; see the GSPC column).")
        return t
    if note not in ("B", "B-heuristic"):
        return ""
    red = med is not None and float(med) > 1.5
    if short:
        t = "ZPNOTE=red comps" if red else "ZPNOTE=comps"
        if med is not None:
            t += f" BP-RP {med:.2f}"
        if b is not None:
            t += f" at b={b:+.1f}deg"
        t += f"; Gaia->{band} zero point uncertain by a few 0.01 mag"
        if diff is not None:
            t += f"; GSPC dZP={float(diff):+.3f}"
        elif _gspc_unavailable_text(gz):
            t += "; GSPC unavailable"
        return t
    t = "Zero-point caution: comparison stars are red" if red else "Zero-point caution: comparison stars"
    if med is not None:
        t += f" (median BP-RP {med:.2f})"
    if b is not None:
        t += f" in a field at Galactic latitude b = {b:+.1f} deg"
    t += (f". The Gaia -> {band} relation is calibrated on low-extinction stars; for reddened stars it can"
          " be off by a few hundredths of a magnitude.")
    if diff is not None:
        t += (f" Gaia DR3 synthetic photometry (GSPC, {n} comparison stars) gives a zero point "
              f"{float(diff):+.3f} mag from the standard one (second scale).")
    t += " The differential light curve is not affected."
    t += _gspc_unavailable_text(gz)
    return t


def _gspc_unavailable_text(gz: dict) -> str:
    """GSPC-OFFLINE-2026-10: the second zero point needs the Gaia archive; when it could not be
    reached the module is skipped and the note says why (Milan 2026-10-11)."""
    st = str((gz or {}).get("status") or "")
    if st.startswith("unavailable") or st.startswith("error"):
        why = st.split("(", 1)[1].rsplit(")", 1)[0] if "(" in st else st
        return f" GSPC calibration not available (Gaia archive not reachable: {why[:80]})."
    return ""
