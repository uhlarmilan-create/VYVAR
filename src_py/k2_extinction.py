"""Second-order extinction (k'') - literature defaults and per-frame correction.

Native literature coefficients (Smith 2002 Sloan, Henden 1982 Johnson B) convert to
BP-RP units via Jordi et al. 2010 colour slopes:

    k2_bprp = k''_native * d(C_native)/d(C_bprp)

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

# Jordi et al. 2010 A&A 515 A16 colour-colour polynomials (Sect. 5.2, Eq. 1):
#   C_Gaia = a + b*C_native + c*C_native^2 + d*C_native^3
# Converter uses d(C_native)/d(BP-RP) = 1 / (b + 2*c*C0 + 3*d*C0^2) at FGK anchor C0.
# Validity: BaSeL3.1 FGK dwarfs; B-V fits have large scatter for Teff < 4500 K (Sect. 5.2).

# Table 6 unreddened SDSS (g-r) -> GBP-GRP: a=0.3523, b=1.1876, c=-0.5370, d=0.4003.
# Anchor g-r=0.48 (FGK): d(g-r)/d(BP-RP) = 1/(1.1876 - 2*0.5370*0.48 + 3*0.4003*0.48^2) ~ +1.054.
# Smith g k2 = -0.016 * 1.054 ~ -0.0169; r k2 = -0.004 * 1.054 ~ -0.0042.
SLOPE_GR_PER_BPRP = 1.054  # d(g'-r')/d(BP-RP); Jordi 2010 Table 6, FGK anchor g-r=0.48

# Table 3 Johnson-Cousins (B-V) -> GBP-GRP, all Av: a=0.0981, b=1.4290, c=-0.0269, d=0.0061.
# Anchor B-V=0.58 (FGK): d(B-V)/d(BP-RP) = 1/(1.429 - 2*0.0269*0.58 + 3*0.0061*0.58^2) ~ +0.713.
# Range B-V 0.45-0.75: slope ~0.71-0.72. Henden k''_B=-0.03 per (B-V) -> k2_B ~ -0.021.
SLOPE_BV_PER_BPRP = 0.713  # d(B-V)/d(BP-RP); Jordi 2010 Table 3, FGK anchor

# No Jordi u-g transformation in Tables 3-6; spec-anchored value retained pending citable source.
# Smith u k2 = -0.021 * 1.091 = -0.0229 (ledger K2-SLOPE-UG, FUTURE).
SLOPE_UG_PER_BPRP = 1.091  # d(u'-g')/d(BP-RP); documented exception (no Jordi row)

# Smith et al. 2002 - k'' per native Sloan colour (mag/airmass/mag colour).
SMITH_K2_NATIVE: dict[str, float] = {
    "U": -0.021,
    "SU": -0.021,
    "UP": -0.021,
    "G": -0.016,
    "SG": -0.016,
    "GP": -0.016,
    "G'": -0.016,
    "R": -0.004,
    "SR": -0.004,
    "RP": -0.004,
    "R'": -0.004,
    "I": 0.006,
    "SI": 0.006,
    "IP": 0.006,
    "I'": 0.006,
    "Z": 0.003,
    "SZ": 0.003,
    "ZP": 0.003,
    "Z'": 0.003,
}

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


# Johnson-Cousins V, R, I: negligible k'' (AAVSO practice).
# G08-13: k'' per standard band of the ONE filter table (band_classify.standard_band_of_filter).
# Johnson U uses the Sloan u' value of Smith et al. 2002 (no Johnson U k'' in the cited sources).
_K2_SLOAN_OF_BAND: dict[str, str] = {"u": "U", "g": "G", "r": "R", "i": "I", "z": "Z", "U": "U"}
_K2_ZERO_BANDS: frozenset[str] = frozenset({"V", "Rc", "Ic", "Rj", "Ij"})


def computed_k2_bprp_for_token(token: str) -> float | None:
    """Literature k'' per BP-RP for a filter token, or None if not applicable."""
    raw = str(token or "").strip().replace("''", "'")
    if not raw or raw.upper() in K2_NONE_TOKENS:
        return None
    band = standard_band_of_filter(raw)
    if band is None:
        return None
    if band in _K2_ZERO_BANDS:
        return 0.0
    if band == "B":
        return k2_native_to_bprp(HENDEN_K2_B_PER_BV, native_slope=SLOPE_BV_PER_BPRP)
    slo = _K2_SLOAN_OF_BAND[band]
    slope = SLOPE_UG_PER_BPRP if slo == "U" else SLOPE_GR_PER_BPRP
    return k2_native_to_bprp(SMITH_K2_NATIVE[slo], native_slope=slope)


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
) -> tuple[float, K2Source]:
    """Literature / none path."""
    band = classify_photometric_band(obs_group, fits_filter=fits_filter, aavso_code=aavso_code)
    if band_failsafe_clear(band):
        return float("nan"), K2Source.NONE

    token = filter_token_from_obs_group(obs_group)
    if token in K2_NONE_TOKENS:
        return float("nan"), K2Source.NONE

    overrides = getattr(cfg, "k2_defaults_bprp", None) or {}
    if isinstance(overrides, dict):
        for lookup in (token, token.lower(), token.replace("'", "")):
            if lookup in overrides:
                try:
                    v = float(overrides[lookup])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(v):
                    return v, K2Source.LITERATURE_DEFAULT

    computed = computed_k2_bprp_for_token(token)
    if computed is None:
        return float("nan"), K2Source.NONE
    return float(computed), K2Source.LITERATURE_DEFAULT


def resolve_k2_bprp_value(
    cfg: Any | None,
    obs_group: str,
    *,
    fits_filter: str | None = None,
    aavso_code: str | None = None,
) -> tuple[float, K2Source]:
    """Return (k2 per BP-RP, source) for an obs_group: literature default, or none (k2_mode off / no k'')."""
    mode = resolve_k2_mode(cfg)
    if mode == "off":
        return float("nan"), K2Source.NONE

    return _literature_k2_bprp_for_obs_group(cfg, obs_group, fits_filter=fits_filter, aavso_code=aavso_code)


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

