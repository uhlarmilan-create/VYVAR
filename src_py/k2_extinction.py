"""Second-order extinction (k'') - literature defaults and per-frame correction.

Native literature coefficients (Smith et al. 2002 Sloan, Henden 1982 Johnson B) convert to
BP-RP units via Jordi et al. 2010 (A&A 523, A48) colour slopes:

    k2_bprp = k''_native * d(C_native)/d(C_bprp)

C_native is the colour the literature k'' multiplies (Smith 2002 eqs. 4-8: u' -> u'-g', g' -> g'-r',
r' -> r'-i', i' and z' -> i'-z'; AAVSO PEP Guide 3.5.4: B -> B-V). K2-BAND-2026-10 (G09-05 / G08-47 / G08-48):
every resolution carries a reason; bands without a citable k'' get source NONE (no k2_delta_mag column).
K2-RELIABLE-2026-11: only bands with a cited k'' AND a cited BP-RP slope are corrected (g, r, B); u / U are NONE;
a target is corrected only when its colour and the colour of every weighted comp are known
(:func:`k2_colour_reference`).

The night fit of k'' (NIGHT_FIT v2, never enabled) is removed (CFG-REVIEW-2026-10).

See ``dev/results/specs/VYVAR_K2_DESIGN_SPEC.md`` v1.1.
"""
from __future__ import annotations

import enum
import logging
import math
from typing import Any

import numpy as np

from band_classify import (
    band_failsafe_clear,
    classify_photometric_band,
    standard_band_of_filter,
)

LOGGER = logging.getLogger(__name__)

# Jordi et al. 2010 A&A 523 A48 colour-colour polynomials (Sect. 5.2, Eq. 1):
#   C_Gaia = a + b*C_native + c*C_native^2 + d*C_native^3
# Converter uses d(C_native)/d(BP-RP) = 1 / (b + 2*c*C0 + 3*d*C0^2) at an FGK anchor C0.
# Validity: Table 5 SDSS fits include reddened SEDs (A = 0, 1, 3, 5 mag); B-V fits have large scatter for
# Teff < 4500 K (Sect. 5.2). Coefficients re-read from the A&A / arXiv text 2026-11-04 (K2-RELIABLE-2026-11).


def jordi_native_per_bprp(coeffs: tuple[float, float, float, float], anchor: float) -> float:
    """d(C_native)/d(BP-RP) of a Jordi 2010 GBP-GRP(C_native) cubic at ``anchor`` (the one converter)."""
    _a, b, c, d = coeffs
    return 1.0 / (b + 2.0 * c * anchor + 3.0 * d * anchor ** 2)


# K2-RELIABLE-2026-11 (b): Table 5 (g-r) -> GBP-GRP row: a=0.3482, b=1.3463, c=-0.0310, d=0.0067 (sigma 0.35).
# The former a=0.3523 b=1.1876 c=-0.5370 d=0.4003 ("Table 6") is not in Jordi 2010 (Table 6 has V-I / g-i only).
# Anchor g-r=0.48 (FGK): d(g-r)/d(BP-RP) ~ 0.757 (was 1.054). Smith g k2 = -0.016 x 0.757 ~ -0.0121.
_JORDI_T5_GR = (0.3482, 1.3463, -0.0310, 0.0067)
_GR_ANCHOR = 0.48
SLOPE_GR_PER_BPRP = jordi_native_per_bprp(_JORDI_T5_GR, _GR_ANCHOR)  # d(g'-r')/d(BP-RP) ~ 0.757

# K2-BAND-2026-10 (G09-05): Smith 2002 eq. 6 multiplies the r' k'' by (r'-i'), not (g'-r').
# Table 5 (r-i) -> GBP-GRP: a=0.4649, b=2.3969, c=-0.2976, d=0.0207 (sigma 0.12).
# Anchor r-i=0.15 (FGK): d(r-i)/d(BP-RP) ~ +0.433. Smith r k2 = -0.004 x 0.433 ~ -0.0017.
_JORDI_T5_RI = (0.4649, 2.3969, -0.2976, 0.0207)
_RI_ANCHOR = 0.15
SLOPE_RI_PER_BPRP = jordi_native_per_bprp(_JORDI_T5_RI, _RI_ANCHOR)  # d(r'-i')/d(BP-RP) ~ 0.433

# Table 3 Johnson-Cousins (B-V) -> GBP-GRP: a=0.0981, b=1.4290, c=-0.0269, d=0.0061 (sigma 0.43).
# Anchor B-V=0.58 (FGK): d(B-V)/d(BP-RP) ~ 0.712 (the rounded 0.713 before). k''_B -0.03 per (B-V) -> k2_B ~ -0.021.
_JORDI_T3_BV = (0.0981, 1.4290, -0.0269, 0.0061)
_BV_ANCHOR = 0.58
SLOPE_BV_PER_BPRP = jordi_native_per_bprp(_JORDI_T3_BV, _BV_ANCHOR)  # d(B-V)/d(BP-RP) ~ 0.712

# K2-RELIABLE-2026-11 (v10 item 8): Jordi 2010 has no u-g relation (Tables 3-6) and no source gives
# d(u'-g')/d(BP-RP); the uncited slope 1.091 and the Johnson U borrowing of the Sloan u' k'' are removed -
# u / U -> source none with K2_REASON_NO_SLOPE_U.

# Smith et al. 2002 (AJ 123, 2121) Table 5 - k'' per native Sloan colour (mag/airmass/mag colour) of the bands
# that have a cited BP-RP slope (g' per g'-r', r' per r'-i'). u' (-0.021 per u'-g'), i' (0.006) and z' (0.003
# per i'-z') have no cited slope and are not used.
SMITH_K2_NATIVE: dict[str, float] = {"G": -0.016, "R": -0.004}

# AAVSO PEP Guide 3.0 Sect. 3.5.4: k''_B typically -0.02 ... -0.04 per (B-V) (Henden & Kaitchuck 1982;
# Hall & Genet -0.03); the midpoint -0.03 is used.
HENDEN_K2_B_PER_BV = -0.03

# Tokens with STANDARD_FILTER band class but no citable k'' (OSC Bayer RGB).
K2_NONE_TOKENS: frozenset[str] = frozenset(
    {
        "BLUE",
        "GREEN",
        "RED",
        "TG",
        "TB",
        "TR",
    }
)

class K2Source(str, enum.Enum):
    LITERATURE_DEFAULT = "literature_default"
    NONE = "none"


def filter_token_from_obs_group(obs_group: str) -> str:
    """Canonical filter token from obs_group (OSC channel suffix -> TR/TG/TB/CLEAR)."""
    from osc_align import obs_group_band_token

    tok = obs_group_band_token(obs_group)
    if tok:
        return str(tok).strip().replace("''", "'")
    return ""


def k2_native_to_bprp(k_native: float, *, native_slope: float) -> float:
    return float(k_native) * float(native_slope)


# G08-13: k'' per standard band of the ONE filter table (band_classify.standard_band_of_filter).
# K2-RELIABLE-2026-11 audit - every applied band has a cited k'' AND a cited BP-RP slope:
#   g: Smith 2002 k''_g' per g'-r' x Jordi 2010 Table 5 g-r row; r: Smith 2002 k''_r' per r'-i' x Jordi 2010
#   Table 5 r-i row; B: AAVSO PEP Guide 3.5.4 k''_B per B-V x Jordi 2010 Table 3 B-V row.
#   u / U, i, z: no cited slope -> none; V / Rc / Ic / Rj / Ij: negligible -> none; others: not standard -> none.
# K2-BAND-2026-10 (G08-47 / G08-48): Johnson-Cousins V, R, I have no k'' correction (negligible, AAVSO PEP
# Guide 3.5.4) - source NONE, not a column of zeros; Sloan i' / z' (Smith: per i'-z') have no citable
# BP-RP slope (no Jordi 2010 i-z relation) - source NONE until one is sourced.
_K2_SLOAN_OF_BAND: dict[str, str] = {"g": "G", "r": "R"}
_K2_SLOPE_OF_SLOAN: dict[str, float] = {"G": SLOPE_GR_PER_BPRP, "R": SLOPE_RI_PER_BPRP}
_K2_ZERO_BANDS: frozenset[str] = frozenset({"V", "Rc", "Ic", "Rj", "Ij"})
_K2_NO_SLOPE_BANDS: frozenset[str] = frozenset({"i", "z"})
_K2_NO_SLOPE_U_BANDS: frozenset[str] = frozenset({"u", "U"})

K2_REASON_NEGLIGIBLE = "none - negligible per AAVSO PEP Guide 3.5.4"
K2_REASON_UNCORRECTED = "none - uncorrected (no literature k'' for unfiltered / tri-colour); k'' X dC remains"
K2_REASON_NO_SLOPE = "none - no citable colour slope (Smith 2002 k'' per i'-z'; no Jordi 2010 i-z relation)"
K2_REASON_NO_SLOPE_U = ("none - no citable colour slope (Smith 2002 k'' per u'-g'; no Jordi 2010 u-g relation; "
                        "no Johnson U k'' in the cited sources)")
K2_REASON_NOT_STANDARD = "none - filter is not a standard band"
K2_REASON_OFF = "none - k2_mode off"
K2_REASON_OVERRIDE = "config k2_defaults_bprp"


def k2_for_token(token: str) -> tuple[float | None, str]:
    """(literature k'' per BP-RP or None, reason) for a filter token (K2-BAND-2026-10)."""
    raw = str(token or "").strip().replace("''", "'")
    if not raw or raw.upper() in K2_NONE_TOKENS:
        return None, K2_REASON_UNCORRECTED
    band = standard_band_of_filter(raw)
    if band is None:
        cls = classify_photometric_band("", fits_filter=raw)
        if cls.value == "standard_filter":
            return None, "none - no literature k'' for this band"
        return None, K2_REASON_UNCORRECTED if band_failsafe_clear(cls) and cls.value != "unknown" \
            else K2_REASON_NOT_STANDARD
    if band in _K2_ZERO_BANDS:
        return None, K2_REASON_NEGLIGIBLE
    if band in _K2_NO_SLOPE_BANDS:
        return None, K2_REASON_NO_SLOPE
    if band in _K2_NO_SLOPE_U_BANDS:
        return None, K2_REASON_NO_SLOPE_U
    if band == "B":
        return (k2_native_to_bprp(HENDEN_K2_B_PER_BV, native_slope=SLOPE_BV_PER_BPRP),
                "AAVSO PEP Guide 3.5.4 (Henden & Kaitchuck 1982) k''_B per B-V x Jordi 2010 Table 3 d(B-V)/d(BP-RP)")
    if band not in _K2_SLOAN_OF_BAND:
        return None, "none - no literature k'' for this band"
    slo = _K2_SLOAN_OF_BAND[band]
    colour = {"G": "g'-r'", "R": "r'-i'"}[slo]
    return (k2_native_to_bprp(SMITH_K2_NATIVE[slo], native_slope=_K2_SLOPE_OF_SLOAN[slo]),
            f"Smith et al. 2002 k'' per {colour} x Jordi 2010 Table 5 d({colour})/d(BP-RP)")


def computed_k2_bprp_for_token(token: str) -> float | None:
    """Literature k'' per BP-RP for a filter token, or None if not applicable (reason: :func:`k2_for_token`)."""
    return k2_for_token(token)[0]


def resolve_k2_mode(cfg: Any | None) -> str:
    mode = str(getattr(cfg, "k2_mode", "literature") or "literature").strip().lower()
    if mode in ("0", "false", "no", "off", "none"):
        return "off"
    return "literature"


def _literature_k2_bprp_for_obs_group(
    cfg: Any | None,
    obs_group: str,
    *,
    fits_filter: str | None = None,
    aavso_code: str | None = None,
) -> tuple[float, K2Source, str]:
    """Literature / none path: (k'' per BP-RP, source, reason)."""
    from band_classify import ambiguous_filter_note  # noqa: PLC0415

    band = classify_photometric_band(obs_group, fits_filter=fits_filter, aavso_code=aavso_code)
    token = filter_token_from_obs_group(obs_group)
    if band_failsafe_clear(band):
        if band.value == "unknown":
            _amb = ambiguous_filter_note(token)
            return float("nan"), K2Source.NONE, (f"none - {_amb}" if _amb else K2_REASON_NOT_STANDARD)
        return float("nan"), K2Source.NONE, K2_REASON_UNCORRECTED

    if token in K2_NONE_TOKENS:
        return float("nan"), K2Source.NONE, K2_REASON_UNCORRECTED

    overrides = getattr(cfg, "k2_defaults_bprp", None) or {}
    if isinstance(overrides, dict):
        for lookup in (token, token.lower(), token.replace("'", "")):
            if lookup in overrides:
                try:
                    v = float(overrides[lookup])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(v):
                    return v, K2Source.LITERATURE_DEFAULT, K2_REASON_OVERRIDE

    computed, reason = k2_for_token(token)
    if computed is None:
        return float("nan"), K2Source.NONE, reason
    return float(computed), K2Source.LITERATURE_DEFAULT, reason


def resolve_k2(
    cfg: Any | None,
    obs_group: str,
    *,
    fits_filter: str | None = None,
    aavso_code: str | None = None,
) -> tuple[float, K2Source, str]:
    """(k2 per BP-RP, source, reason) for an obs_group: literature default, or none with the reason
    (k2_mode off, negligible V/R/I, unfiltered / tri-colour, no citable slope, not a standard band)."""
    if resolve_k2_mode(cfg) == "off":
        return float("nan"), K2Source.NONE, K2_REASON_OFF
    return _literature_k2_bprp_for_obs_group(cfg, obs_group, fits_filter=fits_filter, aavso_code=aavso_code)


def resolve_k2_bprp_value(
    cfg: Any | None,
    obs_group: str,
    *,
    fits_filter: str | None = None,
    aavso_code: str | None = None,
) -> tuple[float, K2Source]:
    """Return (k2 per BP-RP, source) for an obs_group (:func:`resolve_k2` without the reason)."""
    v, src, _ = resolve_k2(cfg, obs_group, fits_filter=fits_filter, aavso_code=aavso_code)
    return v, src


def colourless_weight_fraction(comp_bp_rp: dict[str, float], weights: dict[str, float]) -> float:
    """K2-BAND-2026-10 (G09-05 b): Broeg-weight fraction of the ensemble comps without a BP-RP colour.
    :func:`ensemble_bp_rp` is the weighted colour of the comps WITH a colour; this is the share it misses
    (NaN without weights)."""
    tot = miss = 0.0
    for cid, w in (weights or {}).items():
        w = float(w)
        if not (math.isfinite(w) and w > 0):
            continue
        tot += w
        if not math.isfinite(float(comp_bp_rp.get(cid, float("nan")))):
            miss += w
    return miss / tot if tot > 0 else float("nan")


K2_REASON_TARGET_COLOUR = "none - target BP-RP unknown: dC = C_target - C_comparison unknown, k'' X dC not applied"
K2_REASON_NO_WEIGHTS = "none - no ensemble weights: comparison colour unknown, k'' X dC not applied"


def k2_colour_reference(target_bp_rp: float, comp_bp_rp: dict[str, float],
                        weights: dict[str, float]) -> tuple[float, float, str]:
    """K2-RELIABLE-2026-11 (v10 item 8): (comparison colour, colourless weight share, reason) of one target's
    k'' X X x dC term. The reason is "" when the correction applies; otherwise the colour reference is NaN and the
    reason says why (no assumed colour is ever used):

    - target BP-RP unknown -> dC unknown;
    - no positive Broeg weights -> the comparison colour is undefined;
    - any Broeg weight on comps without BP-RP (share f > 0): the comparison colour is
      (1 - f) C_measured + f C_unknown, so dC carries the unmeasured term f (C_unknown - C_measured), unbounded by the
      data. The colour ladder of the comparison selection gives every colourless candidate |dBP-RP| = inf when the
      target has a colour (``photometry_comp._sel_colour_ladder``), so f > 0 here means the k'' colour source and
      the selection disagree - shown, not covered by an assumed colour.
    """
    wfrac = colourless_weight_fraction(comp_bp_rp, weights)
    if not math.isfinite(float(target_bp_rp)):
        return float("nan"), wfrac, K2_REASON_TARGET_COLOUR
    if not math.isfinite(wfrac):
        return float("nan"), wfrac, K2_REASON_NO_WEIGHTS
    if wfrac > 0.0:
        return float("nan"), wfrac, (f"none - {100.0 * wfrac:.1f} % of the ensemble weight on comps without BP-RP: "
                                     "comparison colour unknown, k'' X dC not applied")
    return ensemble_bp_rp(comp_bp_rp, weights), wfrac, ""


def bp_rp_comp_median(
    comp_bp_rp: dict[str, float],
    comp_quality: dict[str, dict] | None = None,
) -> float:
    """Median BP-RP of usable comps (same definition as ``apply_color_term``)."""
    usable = (
        [cid for cid, q in comp_quality.items() if q.get("quality") in ("good", "suspect")]
        if comp_quality
        else list(comp_bp_rp.keys())
    )
    vals = [
        float(comp_bp_rp[cid])
        for cid in usable
        if cid in comp_bp_rp and math.isfinite(float(comp_bp_rp[cid]))
    ]
    if not vals:
        vals = [float(v) for v in comp_bp_rp.values() if math.isfinite(float(v))]
    if not vals:
        return float("nan")
    return float(np.median(np.asarray(vals, dtype=np.float64)))


def ensemble_bp_rp(comp_bp_rp: dict[str, float], weights: dict[str, float]) -> float:
    """Weighted mean BP-RP of the ensemble comps (weights = the ensemble's Broeg weights)."""
    num = den = 0.0
    for cid, w in (weights or {}).items():
        c = float(comp_bp_rp.get(cid, float("nan")))
        w = float(w)
        if math.isfinite(c) and math.isfinite(w) and w > 0:
            num += w * c
            den += w
    return num / den if den > 0 else float("nan")


def apply_k2_per_frame(
    mag: np.ndarray,
    airmass: np.ndarray,
    *,
    object_bp_rp: float,
    bp_rp_comp_med: float,
    k2_value: float,
    k2_source: K2Source,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Apply k'' per frame. Returns (corrected_mag, k2_delta, k2_source per row)."""
    out = np.asarray(mag, dtype=np.float64).copy()
    n = len(out)
    delta = np.full(n, float("nan"), dtype=np.float64)
    sources: list[str] = [K2Source.NONE.value] * n
    src_ok = k2_source is K2Source.LITERATURE_DEFAULT
    if (not src_ok) or not math.isfinite(float(k2_value)):
        return out, delta, sources
    src_tag = str(k2_source.value)
    if float(k2_value) == 0.0:
        if math.isfinite(float(object_bp_rp)) and math.isfinite(float(bp_rp_comp_med)):
            return out, np.zeros(n, dtype=np.float64), [src_tag] * n
        return out, delta, sources
    if not math.isfinite(float(object_bp_rp)) or not math.isfinite(float(bp_rp_comp_med)):
        return out, delta, sources

    am = np.asarray(airmass, dtype=np.float64)
    d_c = float(object_bp_rp) - float(bp_rp_comp_med)
    for i in range(n):
        if not math.isfinite(float(out[i])):
            continue
        if i >= len(am) or not math.isfinite(float(am[i])):
            continue
        d = float(k2_value) * d_c * float(am[i])
        out[i] = float(out[i]) - d
        delta[i] = d
        sources[i] = src_tag
    return out, delta, sources


def apply_k2_to_comp_mag_inst(
    comp_mag_inst: dict[str, np.ndarray],
    comp_bp_rp: dict[str, float],
    comp_quality: dict[str, dict],
    airmass: np.ndarray,
    k2_value: float,
    bp_rp_comp_med: float,
    *,
    k2_source: K2Source = K2Source.LITERATURE_DEFAULT,
) -> dict[str, np.ndarray]:
    """Apply per-frame k'' to each comp instrumental-magnitude series (group CT fit path)."""
    if not math.isfinite(float(k2_value)):
        return comp_mag_inst
    if k2_source is not K2Source.LITERATURE_DEFAULT:
        return comp_mag_inst
    out: dict[str, np.ndarray] = {}
    for cid, series in comp_mag_inst.items():
        q = comp_quality.get(cid, {})
        if q.get("quality") == "excluded":
            out[cid] = np.asarray(series, dtype=np.float64).copy()
            continue
        bp = float(comp_bp_rp.get(cid, float("nan")))
        corr, _, _ = apply_k2_per_frame(
            np.asarray(series, dtype=np.float64),
            airmass,
            object_bp_rp=bp,
            bp_rp_comp_med=bp_rp_comp_med,
            k2_value=float(k2_value),
            k2_source=k2_source,
        )
        out[cid] = corr
    return out


def airmass_from_proc_csvs(csv_files: list[Any]) -> np.ndarray:
    """Per-frame airmass aligned with ``csv_files`` order."""
    from pathlib import Path

    am = np.full(len(csv_files), float("nan"), dtype=np.float64)
    for i, csv_path in enumerate(csv_files):
        p = Path(csv_path)
        try:
            import pandas as pd

            from proc_frame_store import read_proc_csv  # noqa: PLC0415

            df = read_proc_csv(p, usecols=lambda c: c in ("airmass", "AIRMASS"), nrows=8)
        except Exception as exc:  # noqa: BLE001
            from except_fix_counters import get_except_fix_counters

            get_except_fix_counters().k2_airmass_read_fail += 1
            LOGGER.error(
                "[K2] airmass read failed for proc CSV %s: %s",
                p,
                exc,
            )
            continue
        col = "airmass" if "airmass" in df.columns else ("AIRMASS" if "AIRMASS" in df.columns else None)
        if col is None:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(vals):
            am[i] = float(vals.iloc[0])
    return am

