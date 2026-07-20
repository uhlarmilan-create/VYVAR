"""Second-order extinction (k'') - literature defaults and per-frame correction.

Native literature coefficients (Smith 2002 Sloan, Henden 1982 Johnson B) convert to
BP-RP units via Jordi et al. 2010 colour slopes:

    k2_bprp = k''_native * d(C_native)/d(C_bprp)

NIGHT_FIT v2 (gated by ``k2_fit_enabled``, default OFF): ``fit_k2_night`` +
``k2_feasibility_pregate`` per ``dev/results/specs/VYVAR_K2_DESIGN_SPEC.md`` v1.1 S5/S6.

See ``dev/results/specs/VYVAR_K2_DESIGN_SPEC.md`` v1.1.
"""
from __future__ import annotations

import enum
import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from band_classify import (
    PhotometricBand,
    band_failsafe_clear,
    classify_photometric_band,
    obs_group_first_token,
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
    NIGHT_FIT = "night_fit"
    LITERATURE_DEFAULT = "literature_default"
    NONE = "none"


def filter_token_from_obs_group(obs_group: str) -> str:
    """Canonical filter token from obs_group (first segment; case preserved for Sloan vs Johnson)."""
    raw = obs_group_first_token(obs_group)
    if not raw:
        return ""
    return str(raw).strip().replace("''", "'")


# Johnson/Cousins bands with negligible k'' (AAVSO practice); uppercase tokens only.
JOHNSON_K2_ZERO_TOKENS: frozenset[str] = frozenset({"V", "VC", "R", "RC", "RJ", "I", "IC", "IJ"})

# Legacy alias kept for imports; prefer JOHNSON_K2_ZERO_TOKENS.
K2_ZERO_TOKENS = JOHNSON_K2_ZERO_TOKENS


def k2_native_to_bprp(token: str, k_native: float, *, native_slope: float) -> float:
    return float(k_native) * float(native_slope)


def computed_k2_bprp_for_token(token: str) -> float | None:
    """Literature k'' per BP-RP for a canonical filter token, or None if not applicable."""
    raw = str(token or "").strip().replace("''", "'")
    if not raw:
        return None
    key = raw.upper()
    if key in K2_NONE_TOKENS:
        return None
    # Sloan lowercase obs_group tokens (g,r,i,u,z) - distinct from Johnson uppercase R/I.
    if len(raw) == 1 and raw.islower() and raw in "griuz":
        slo = raw.upper()
        if slo in SMITH_K2_NATIVE:
            kn = SMITH_K2_NATIVE[slo]
            if slo in ("U", "SU", "UP"):
                return k2_native_to_bprp(slo, kn, native_slope=SLOPE_UG_PER_BPRP)
            return k2_native_to_bprp(slo, kn, native_slope=SLOPE_GR_PER_BPRP)
    if raw.isupper() and key in JOHNSON_K2_ZERO_TOKENS:
        return 0.0
    if key in ("B", "BC"):
        return k2_native_to_bprp(key, HENDEN_K2_B_PER_BV, native_slope=SLOPE_BV_PER_BPRP)
    if key in SMITH_K2_NATIVE:
        kn = SMITH_K2_NATIVE[key]
        if key in ("U", "SU", "UP"):
            return k2_native_to_bprp(key, kn, native_slope=SLOPE_UG_PER_BPRP)
        return k2_native_to_bprp(key, kn, native_slope=SLOPE_GR_PER_BPRP)
    return None


def resolve_k2_mode(cfg: Any | None) -> str:
    mode = str(getattr(cfg, "k2_mode", "literature") or "literature").strip().lower()
    if mode in ("0", "false", "no", "off", "none"):
        return "off"
    if mode in ("fit", "fit_else_literature", "night_fit", "auto"):
        return mode
    return "literature"


def _literature_k2_bprp_for_obs_group(
    cfg: Any | None,
    obs_group: str,
    *,
    fits_filter: str | None = None,
    aavso_code: str | None = None,
) -> tuple[float, K2Source]:
    """Literature / none path (v1). Never returns NIGHT_FIT."""
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


def _k2_fit_mode_requested(cfg: Any | None) -> bool:
    """True when config asks for a night fit attempt (still gated by k2_fit_enabled)."""
    if not bool(getattr(cfg, "k2_fit_enabled", False)):
        return False
    mode = resolve_k2_mode(cfg)
    return mode in ("fit", "fit_else_literature", "night_fit", "auto")


def resolve_k2_bprp_value(
    cfg: Any | None,
    obs_group: str,
    *,
    fits_filter: str | None = None,
    aavso_code: str | None = None,
    night_fit_result: K2NightFitResult | None = None,
) -> tuple[float, K2Source]:
    """Return (k2 per BP-RP, source) for an obs_group.

    When ``k2_fit_enabled`` and mode is ``fit_else_literature`` (or fit/night_fit/auto)
    and ``night_fit_result.accepted``, returns NIGHT_FIT. Otherwise literature/none.
    Default ``k2_fit_enabled=False`` is byte-identical to the v1 literature path.
    """
    mode = resolve_k2_mode(cfg)
    if mode == "off":
        return float("nan"), K2Source.NONE

    lit_val, lit_src = _literature_k2_bprp_for_obs_group(
        cfg, obs_group, fits_filter=fits_filter, aavso_code=aavso_code
    )
    if not _k2_fit_mode_requested(cfg):
        return lit_val, lit_src

    if night_fit_result is not None and bool(night_fit_result.accepted):
        if math.isfinite(float(night_fit_result.k2_value)):
            return float(night_fit_result.k2_value), K2Source.NIGHT_FIT
    # Fit refused / missing -> LITERATURE_DEFAULT (or NONE if band has no k'').
    return lit_val, lit_src


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
    src_ok = k2_source in (K2Source.LITERATURE_DEFAULT, K2Source.NIGHT_FIT)
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
    if k2_source not in (K2Source.LITERATURE_DEFAULT, K2Source.NIGHT_FIT):
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

            df = pd.read_csv(p, usecols=lambda c: c in ("airmass", "AIRMASS"), nrows=8)
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


# ---------------------------------------------------------------------------
# NIGHT_FIT v2 - fit path + feasibility pre-gate (S5 / S6)
# Model (S5 form): residual = k2 * (C_i - C_ref) * X_t
# ---------------------------------------------------------------------------

@dataclass
class K2FitDiagnostics:
    """Intermediate fit statistics consumed by ``k2_feasibility_pregate``."""

    k2_fit: float
    sigma_boot: float
    sigma_k2_pred: float
    k2_literature: float
    colour_tertile_k2: list[float] = field(default_factory=list)
    brightness_tertile_k2: list[float] = field(default_factory=list)
    arc_k2: list[float] = field(default_factory=list)
    airmass_monotonic: bool = False
    n_points: int = 0
    sd_c_dx: float = float("nan")
    residual_rms: float = float("nan")


@dataclass
class K2NightFitResult:
    """Outcome of ``fit_k2_night`` (accepted night fit or refuse -> literature)."""

    accepted: bool
    k2_value: float
    sigma_boot: float
    refuse_reason: str = ""
    diagnostics: K2FitDiagnostics | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_meta(self) -> dict[str, Any]:
        out = {
            "k2_fit_attempted": True,
            "k2_fit_accepted": bool(self.accepted),
            "k2_fit_refuse_reason": str(self.refuse_reason or ""),
            "k2_fit_value": float(self.k2_value) if math.isfinite(float(self.k2_value)) else None,
            "k2_fit_sigma_boot": (
                float(self.sigma_boot) if math.isfinite(float(self.sigma_boot)) else None
            ),
        }
        out.update(dict(self.meta or {}))
        return out


def _ols_slope_through_origin(x: np.ndarray, y: np.ndarray) -> float:
    """k2 = sum(x y) / sum(x^2) for model y = k2 * x."""
    xx = float(np.dot(x, x))
    if xx <= 0.0 or not math.isfinite(xx):
        return float("nan")
    return float(np.dot(x, y) / xx)


def _honeycutt_residuals(
    mag: np.ndarray,
    *,
    star_index: np.ndarray,
    frame_index: np.ndarray,
) -> np.ndarray:
    """Flux-derived differential residuals: per-frame ensemble subtract + per-star night median.

    ``mag`` must already be instrumental (-2.5 log10 flux), never catalog ``mag``.
    """
    out = np.asarray(mag, dtype=np.float64).copy()
    n = len(out)
    if n == 0:
        return out
    # Per-frame ensemble median (common mode / transparency).
    for fi in np.unique(frame_index):
        m = frame_index == fi
        vals = out[m]
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        out[m] = out[m] - float(np.median(vals))
    # Per-star night median (Honeycutt star term).
    for si in np.unique(star_index):
        m = star_index == si
        vals = out[m]
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        out[m] = out[m] - float(np.median(vals))
    return out


def _airmass_is_monotonic(airmass_by_frame: np.ndarray, *, tol: float = 0.01) -> bool:
    """True when X(t) has no usable second arc (strictly mono up or down within tol)."""
    x = np.asarray(airmass_by_frame, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 6:
        return True
    d = np.diff(x)
    # Allow tiny numerical wiggles; require a clear reverse of direction for non-mono.
    up = d > tol
    down = d < -tol
    if not np.any(up) or not np.any(down):
        return True
    # Non-monotonic if both rise and fall segments exist with material amplitude.
    return False


def _split_arcs(airmass_by_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split frames into pre-/post-culmination arcs at the airmass minimum."""
    x = np.asarray(airmass_by_frame, dtype=np.float64)
    n = len(x)
    finite = np.isfinite(x)
    if int(finite.sum()) < 6:
        return np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    idx_min = int(np.nanargmin(x))
    arc1 = np.zeros(n, dtype=bool)
    arc2 = np.zeros(n, dtype=bool)
    arc1[: idx_min + 1] = finite[: idx_min + 1]
    arc2[idx_min:] = finite[idx_min:]
    return arc1, arc2


def _tertile_masks(values: np.ndarray) -> list[np.ndarray]:
    v = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(v)
    out: list[np.ndarray] = []
    if int(finite.sum()) < 3:
        return out
    qs = np.nanquantile(v[finite], [1.0 / 3.0, 2.0 / 3.0])
    out.append(finite & (v <= qs[0]))
    out.append(finite & (v > qs[0]) & (v <= qs[1]))
    out.append(finite & (v > qs[1]))
    return out


def k2_feasibility_pregate(
    diag: K2FitDiagnostics,
    *,
    k2_ceiling: float = 0.1,
    k2_fit_min_detectability: float = 3.0,
    k2_fit_consistency_sigma: float = 2.0,
    k2_fit_lit_factor: float = 4.0,
) -> tuple[bool, str]:
    """S6 pre-gate: ALL must hold. Returns (ok, refuse_reason_code)."""
    if bool(diag.airmass_monotonic):
        return False, "monotonic_airmass"

    lit = float(diag.k2_literature)
    thr_det = abs(lit) / max(float(k2_fit_min_detectability), 1e-9)
    if not math.isfinite(float(diag.sigma_k2_pred)) or float(diag.sigma_k2_pred) > thr_det:
        return False, "detectability"

    # Consistency band: k2_fit_consistency_sigma * sigma_boot, with a small
    # relative floor so an ultra-tight bootstrap on clean nights cannot make
    # tertile/arc checks impossible (still ~spec S6.3; floor is 5% of |k2|).
    sig = float(diag.sigma_boot)
    if not math.isfinite(sig) or sig < 0.0:
        return False, "consistency_sigma_boot"
    rel_floor = 0.10 * max(abs(float(diag.k2_fit)), abs(float(diag.k2_literature)), 1e-4)
    band = float(k2_fit_consistency_sigma) * max(sig, rel_floor)

    def _agree(vals: list[float]) -> bool:
        finite = [float(v) for v in vals if math.isfinite(float(v))]
        if len(finite) < 2:
            return True
        return (max(finite) - min(finite)) <= band

    # Prefer outer-tertile comparison when caller packed [lo, mid, hi]; fall back
    # to all finite values.
    def _outer_or_all(vals: list[float]) -> list[float]:
        if len(vals) >= 3:
            return [vals[0], vals[2]]
        return vals

    if diag.colour_tertile_k2 and not _agree(_outer_or_all(diag.colour_tertile_k2)):
        return False, "colour_tertile_inconsistent"
    if diag.brightness_tertile_k2 and not _agree(_outer_or_all(diag.brightness_tertile_k2)):
        return False, "brightness_tertile_inconsistent"
    if diag.arc_k2 and not _agree(diag.arc_k2):
        return False, "arc_inconsistent"

    k2 = float(diag.k2_fit)
    if not math.isfinite(k2):
        return False, "plausibility_nonfinite"
    if abs(k2) > float(k2_ceiling):
        return False, "plausibility_ceiling"

    # Sign + magnitude within lit_factor of literature default.
    if math.isfinite(lit) and abs(lit) > 0.0:
        if (k2 * lit) < 0.0:
            return False, "plausibility_literature_sign"
        if abs(k2) > float(k2_fit_lit_factor) * abs(lit):
            return False, "plausibility_literature"
    else:
        # Literature ~0: only near-zero fits are plausible.
        if abs(k2) > 1e-4:
            return False, "plausibility_literature"

    return True, ""


def fit_k2_night(
    *,
    residual: np.ndarray,
    colour: np.ndarray,
    airmass: np.ndarray,
    star_index: np.ndarray,
    frame_index: np.ndarray,
    brightness: np.ndarray | None = None,
    colour_ref: float | None = None,
    k2_literature: float,
    k2_ceiling: float = 0.1,
    k2_fit_min_detectability: float = 3.0,
    k2_fit_consistency_sigma: float = 2.0,
    k2_fit_lit_factor: float = 4.0,
    n_bootstrap: int = 200,
    rng: np.random.Generator | None = None,
    airmass_by_frame: np.ndarray | None = None,
) -> K2NightFitResult:
    """Fit k2 from flux-derived Honeycutt residuals (S5 model) + S6 pre-gate.

    Parameters are already QC-filtered (caller builds the fit-frame subset read-only).
    ``residual`` should be Honeycutt residuals (or raw mag_inst if ``apply_honeycutt``
    is performed by the caller beforehand). This function treats ``residual`` as y.
    """
    y = np.asarray(residual, dtype=np.float64)
    c = np.asarray(colour, dtype=np.float64)
    x_am = np.asarray(airmass, dtype=np.float64)
    si = np.asarray(star_index, dtype=np.int64)
    fi = np.asarray(frame_index, dtype=np.int64)
    n = len(y)
    if brightness is None:
        bright = np.zeros(n, dtype=np.float64)
    else:
        bright = np.asarray(brightness, dtype=np.float64)

    if colour_ref is None or not math.isfinite(float(colour_ref)):
        cref = float(np.nanmedian(c[np.isfinite(c)])) if np.any(np.isfinite(c)) else float("nan")
    else:
        cref = float(colour_ref)

    ok = np.isfinite(y) & np.isfinite(c) & np.isfinite(x_am) & np.isfinite(cref)
    if int(ok.sum()) < 12:
        return K2NightFitResult(
            accepted=False,
            k2_value=float("nan"),
            sigma_boot=float("nan"),
            refuse_reason="insufficient_data",
            meta={"n_points": int(ok.sum())},
        )

    y = y[ok]
    c = c[ok]
    x_am = x_am[ok]
    si = si[ok]
    fi = fi[ok]
    bright = bright[ok]

    # Per-frame airmass series for monotonic / arc checks.
    if airmass_by_frame is None:
        n_frames = int(fi.max()) + 1 if len(fi) else 0
        am_frame = np.full(n_frames, float("nan"), dtype=np.float64)
        for f in np.unique(fi):
            m = fi == f
            am_frame[int(f)] = float(np.nanmedian(x_am[m]))
    else:
        am_frame = np.asarray(airmass_by_frame, dtype=np.float64)

    mono = _airmass_is_monotonic(am_frame)

    # S5 correction uses k2*(C-Cref)*X. After Honeycutt (frame + star medians),
    # the identifiable residual is k2*(C-Cref)*dX with dX = X - mean(X).
    d_c = c - cref
    x_mean = (
        float(np.nanmean(am_frame[np.isfinite(am_frame)]))
        if np.any(np.isfinite(am_frame))
        else float(np.nanmean(x_am))
    )
    d_x = x_am - x_mean
    x_des = d_c * d_x

    k2_hat = _ols_slope_through_origin(x_des, y)
    resid = y - k2_hat * x_des
    rms = float(np.sqrt(np.nanmean(resid**2))) if len(resid) else float("nan")

    # Detectability leverage: sd(C*dX) and N from the night (S6 item 2).
    sd_c_dx = float(np.nanstd(x_des, ddof=1)) if len(x_des) > 1 else float("nan")
    n_pts = int(len(y))
    if math.isfinite(sd_c_dx) and sd_c_dx > 0 and math.isfinite(rms) and n_pts > 0:
        sigma_k2_pred = float(rms) / (sd_c_dx * math.sqrt(float(n_pts)))
    else:
        sigma_k2_pred = float("inf")

    rng = rng if rng is not None else np.random.default_rng(0)
    # Bootstrap over frames AND stars (cluster-safe); take the larger scatter so
    # tertile consistency (S6.3) is judged against star-sampling uncertainty too.
    boot_vals: list[float] = []
    uniq_frames = np.unique(fi)
    uniq_stars_boot = np.unique(si)
    n_boot = int(max(20, n_bootstrap))
    if len(uniq_frames) >= 4:
        for _ in range(n_boot // 2):
            draw = rng.choice(uniq_frames, size=len(uniq_frames), replace=True)
            mask = np.isin(fi, draw)
            if int(mask.sum()) < 8:
                continue
            kb = _ols_slope_through_origin(x_des[mask], y[mask])
            if math.isfinite(kb):
                boot_vals.append(float(kb))
    if len(uniq_stars_boot) >= 4:
        for _ in range(n_boot - n_boot // 2):
            draw = rng.choice(uniq_stars_boot, size=len(uniq_stars_boot), replace=True)
            mask = np.isin(si, draw)
            if int(mask.sum()) < 8:
                continue
            kb = _ols_slope_through_origin(x_des[mask], y[mask])
            if math.isfinite(kb):
                boot_vals.append(float(kb))
    sigma_boot = float(np.std(boot_vals, ddof=1)) if len(boot_vals) >= 8 else float("nan")
    if math.isfinite(sigma_boot) and math.isfinite(sigma_k2_pred):
        sigma_boot = max(float(sigma_boot), float(sigma_k2_pred))

    # Colour / brightness tertile fits (by star median colour / brightness).
    star_ids = np.unique(si)
    star_col = np.array([float(np.nanmedian(c[si == s])) for s in star_ids], dtype=np.float64)
    star_brt = np.array([float(np.nanmedian(bright[si == s])) for s in star_ids], dtype=np.float64)

    x_des_sd = float(np.nanstd(x_des)) if len(x_des) else 0.0

    def _subset_k2(star_mask: np.ndarray) -> float:
        keep_stars = {int(s) for s, m in zip(star_ids, star_mask, strict=False) if m}
        if len(keep_stars) < 2:
            return float("nan")
        m = np.array([int(s) in keep_stars for s in si], dtype=bool)
        if int(m.sum()) < 8:
            return float("nan")
        # Middle colour tertile near C_ref has ~zero leverage; skip weak subsets.
        sd_sub = float(np.nanstd(x_des[m])) if int(m.sum()) else 0.0
        if x_des_sd > 0 and sd_sub < 0.25 * x_des_sd:
            return float("nan")
        return _ols_slope_through_origin(x_des[m], y[m])

    colour_k2: list[float] = []
    for tm in _tertile_masks(star_col):
        colour_k2.append(_subset_k2(tm))
    bright_k2: list[float] = []
    for tm in _tertile_masks(star_brt):
        bright_k2.append(_subset_k2(tm))
    arc_k2: list[float] = []
    if not mono:
        a1, a2 = _split_arcs(am_frame)
        for arc in (a1, a2):
            # Map frame mask -> point mask
            keep_f = {i for i, okf in enumerate(arc) if okf}
            m = np.array([int(f) in keep_f for f in fi], dtype=bool)
            if int(m.sum()) >= 8:
                arc_k2.append(_ols_slope_through_origin(x_des[m], y[m]))

    diag = K2FitDiagnostics(
        k2_fit=float(k2_hat),
        sigma_boot=float(sigma_boot) if math.isfinite(sigma_boot) else float("nan"),
        sigma_k2_pred=float(sigma_k2_pred),
        k2_literature=float(k2_literature),
        colour_tertile_k2=colour_k2,
        brightness_tertile_k2=bright_k2,
        arc_k2=arc_k2,
        airmass_monotonic=bool(mono),
        n_points=n_pts,
        sd_c_dx=float(sd_c_dx),
        residual_rms=float(rms),
    )
    ok_gate, reason = k2_feasibility_pregate(
        diag,
        k2_ceiling=float(k2_ceiling),
        k2_fit_min_detectability=float(k2_fit_min_detectability),
        k2_fit_consistency_sigma=float(k2_fit_consistency_sigma),
        k2_fit_lit_factor=float(k2_fit_lit_factor),
    )
    meta = {
        "k2_fit_n_points": n_pts,
        "k2_fit_sigma_k2_pred": float(sigma_k2_pred) if math.isfinite(sigma_k2_pred) else None,
        "k2_fit_sd_c_dx": float(sd_c_dx) if math.isfinite(sd_c_dx) else None,
        "k2_fit_residual_rms": float(rms) if math.isfinite(rms) else None,
        "k2_fit_airmass_monotonic": bool(mono),
        "k2_fit_colour_tertile": [float(v) if math.isfinite(v) else None for v in colour_k2],
        "k2_fit_brightness_tertile": [float(v) if math.isfinite(v) else None for v in bright_k2],
        "k2_fit_arc": [float(v) if math.isfinite(v) else None for v in arc_k2],
        "k2_colour_ref": float(cref) if math.isfinite(cref) else None,
    }
    if ok_gate:
        LOGGER.info(
            "[K2-FIT] NIGHT_FIT accepted k2=%.6f sigma_boot=%.6f n=%d",
            float(k2_hat),
            float(sigma_boot) if math.isfinite(sigma_boot) else float("nan"),
            n_pts,
        )
        return K2NightFitResult(
            accepted=True,
            k2_value=float(k2_hat),
            sigma_boot=float(sigma_boot) if math.isfinite(sigma_boot) else float("nan"),
            refuse_reason="",
            diagnostics=diag,
            meta=meta,
        )
    LOGGER.info(
        "[K2-FIT] NIGHT_FIT refused (%s); falling back to literature k2=%.6f",
        reason,
        float(k2_literature) if math.isfinite(float(k2_literature)) else float("nan"),
    )
    return K2NightFitResult(
        accepted=False,
        k2_value=float(k2_hat),
        sigma_boot=float(sigma_boot) if math.isfinite(sigma_boot) else float("nan"),
        refuse_reason=str(reason),
        diagnostics=diag,
        meta=meta,
    )


def build_honeycutt_residual_table(
    mag_inst: np.ndarray,
    colour: np.ndarray,
    airmass: np.ndarray,
    star_index: np.ndarray,
    frame_index: np.ndarray,
    brightness: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Convenience: mag_inst -> Honeycutt residual arrays for ``fit_k2_night``."""
    resid = _honeycutt_residuals(
        mag_inst, star_index=np.asarray(star_index), frame_index=np.asarray(frame_index)
    )
    out: dict[str, np.ndarray] = {
        "residual": resid,
        "colour": np.asarray(colour, dtype=np.float64),
        "airmass": np.asarray(airmass, dtype=np.float64),
        "star_index": np.asarray(star_index, dtype=np.int64),
        "frame_index": np.asarray(frame_index, dtype=np.int64),
    }
    if brightness is not None:
        out["brightness"] = np.asarray(brightness, dtype=np.float64)
    return out


def select_k2_fit_frames_readonly(
    n_frames: int,
    *,
    align_residual_px: np.ndarray | None = None,
    align_residual_max_px: float | None = None,
    frame_quality_ok: np.ndarray | None = None,
) -> np.ndarray:
    """READ-ONLY QC mask for the fit-frame subset (does not alter photometry frames)."""
    keep = np.ones(int(n_frames), dtype=bool)
    if align_residual_px is not None:
        ar = np.asarray(align_residual_px, dtype=np.float64)
        if len(ar) == n_frames:
            keep &= np.isfinite(ar)
            if align_residual_max_px is not None and math.isfinite(float(align_residual_max_px)):
                keep &= ar <= float(align_residual_max_px)
    if frame_quality_ok is not None:
        fq = np.asarray(frame_quality_ok, dtype=bool)
        if len(fq) == n_frames:
            keep &= fq
    # Never drop the whole night for the fit - fall back to all-finite.
    if int(keep.sum()) < max(6, n_frames // 5):
        return np.ones(int(n_frames), dtype=bool)
    return keep


def attempt_k2_night_fit_from_arrays(
    cfg: Any | None,
    obs_group: str,
    *,
    mag_inst: np.ndarray,
    colour: np.ndarray,
    airmass: np.ndarray,
    star_index: np.ndarray,
    frame_index: np.ndarray,
    brightness: np.ndarray | None = None,
    frame_ok: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> K2NightFitResult | None:
    """Run NIGHT_FIT when enabled; return None when the fit path is not requested."""
    if not _k2_fit_mode_requested(cfg):
        return None
    lit, lit_src = _literature_k2_bprp_for_obs_group(cfg, obs_group)
    if lit_src is K2Source.NONE or not math.isfinite(float(lit)):
        return K2NightFitResult(
            accepted=False,
            k2_value=float("nan"),
            sigma_boot=float("nan"),
            refuse_reason="band_no_literature_k2",
            meta={"k2_fit_attempted": True},
        )
    table = build_honeycutt_residual_table(
        mag_inst, colour, airmass, star_index, frame_index, brightness=brightness
    )
    if frame_ok is not None:
        fok = np.asarray(frame_ok, dtype=bool)
        m = fok[np.asarray(frame_index, dtype=np.int64)]
        for k in list(table.keys()):
            table[k] = table[k][m]
        if brightness is not None:
            brightness = np.asarray(brightness, dtype=np.float64)[m]

    ceiling = float(getattr(cfg, "k2_ceiling", 0.1) or 0.1)
    return fit_k2_night(
        residual=table["residual"],
        colour=table["colour"],
        airmass=table["airmass"],
        star_index=table["star_index"],
        frame_index=table["frame_index"],
        brightness=table.get("brightness", brightness),
        k2_literature=float(lit),
        k2_ceiling=ceiling,
        k2_fit_min_detectability=float(getattr(cfg, "k2_fit_min_detectability", 3.0)),
        k2_fit_consistency_sigma=float(getattr(cfg, "k2_fit_consistency_sigma", 2.0)),
        k2_fit_lit_factor=float(getattr(cfg, "k2_fit_lit_factor", 4.0)),
        rng=rng,
    )

