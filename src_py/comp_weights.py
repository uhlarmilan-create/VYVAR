"""COMP-ADMIT-03: continuous comparison-star weights (no admission rank cuts).

sigma_eff^2 = sigma_rms^2 + (c_col * |delta(BP-RP)|)^2 + (c_dist * r_deg)^2
w           = 1 / sigma_eff^2

Gates that remain elsewhere (not weights): saturation/non-linearity, known
variable, geometry (aligned footprint). See docs/VYVAR_DECISIONS.md COMP-ADMIT-03.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

# Named gap: refractive colour-dependent PSF / enclosed-flux systematic (~30 mmag
# over production radii on the wide lens) is registered but not yet reduced to a
# portable c_col term independent of k''. Mirror rigs should be ~0.
C_COL_PSF_TERM_GAP = (
    "c_col_psf_term: named zero until a portable measurement of colour-dependent "
    "enclosed-flux vs BP-RP is wired; k''*DeltaX is the only active colour term."
)


@dataclass(frozen=True)
class CompWeightCoeffs:
    """Colour and distance systematic coefficients (magnitudes)."""

    c_col_mag_per_bprp: float
    c_dist_mag_per_deg: float
    c_col_source: str
    c_dist_source: str
    notes: tuple[str, ...] = ()


def sigma_eff_mag(
    *,
    sigma_rms_mag: float,
    delta_bprp: float,
    r_deg: float,
    c_col_mag_per_bprp: float,
    c_dist_mag_per_deg: float,
) -> float:
    """Effective sigma in magnitudes for one comparison star relative to one target."""
    s = float(sigma_rms_mag)
    if not math.isfinite(s) or s < 0:
        s = float("nan")
    dc = float(delta_bprp) if math.isfinite(float(delta_bprp)) else 0.0
    rd = float(r_deg) if math.isfinite(float(r_deg)) else 0.0
    cc = float(c_col_mag_per_bprp) if math.isfinite(float(c_col_mag_per_bprp)) else 0.0
    cd = float(c_dist_mag_per_deg) if math.isfinite(float(c_dist_mag_per_deg)) else 0.0
    if not math.isfinite(s):
        return float("nan")
    # Floor tiny rms so a perfect flat artefact does not dominate with infinite weight.
    s_use = max(s, 1e-6)
    se2 = s_use * s_use + (cc * abs(dc)) ** 2 + (cd * abs(rd)) ** 2
    return float(math.sqrt(se2))


def weight_from_sigma_eff(sigma_eff: float) -> float:
    if not math.isfinite(float(sigma_eff)) or float(sigma_eff) <= 0:
        return 0.0
    return float(1.0 / (float(sigma_eff) ** 2))


def compute_comp_weights(
    *,
    catalog_ids: Sequence[str],
    sigma_rms_mag: Mapping[str, float],
    delta_bprp: Mapping[str, float],
    r_deg: Mapping[str, float],
    c_col_mag_per_bprp: float,
    c_dist_mag_per_deg: float,
) -> dict[str, float]:
    """Return w[cid] for each id; missing inputs yield weight 0."""
    out: dict[str, float] = {}
    for cid in catalog_ids:
        key = str(cid).strip()
        if not key:
            continue
        se = sigma_eff_mag(
            sigma_rms_mag=float(sigma_rms_mag.get(key, float("nan"))),
            delta_bprp=float(delta_bprp.get(key, float("nan"))),
            r_deg=float(r_deg.get(key, float("nan"))),
            c_col_mag_per_bprp=float(c_col_mag_per_bprp),
            c_dist_mag_per_deg=float(c_dist_mag_per_deg),
        )
        out[key] = weight_from_sigma_eff(se)
    return out


def c_col_from_k2_airmass(
    k2_bprp: float | None,
    airmass_span: float,
) -> tuple[float, str]:
    """c_col ~= |k''| * Delta(X); second-order extinction colour systematic (mag per BP-RP)."""
    if k2_bprp is None or not math.isfinite(float(k2_bprp)):
        return 0.0, "c_col=0 (no k2_bprp)"
    dx = float(airmass_span)
    if not math.isfinite(dx) or dx < 0:
        dx = 0.0
    val = abs(float(k2_bprp)) * dx
    return float(val), f"c_col=|k2_bprp|*DeltaX (|{float(k2_bprp):.6g}|*{dx:.4g})"


def measure_c_dist_mag_per_deg(
    *,
    r_deg: Sequence[float],
    residual_scatter_mag: Sequence[float],
    min_points: int = 8,
) -> tuple[float, str]:
    """Regress residual scatter vs separation; return slope (mag/deg) or 0 with named gap."""
    x = np.asarray(list(r_deg), dtype=np.float64)
    y = np.asarray(list(residual_scatter_mag), dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0)
    if int(ok.sum()) < int(min_points):
        return 0.0, (
            f"c_dist=0 named_gap:insufficient_points n={int(ok.sum())}<{int(min_points)}"
        )
    xx = x[ok]
    yy = y[ok]
    # Slope of scatter vs r; clamp negative (unphysical for a systematic floor) to 0.
    try:
        slope = float(np.polyfit(xx, yy, 1)[0])
    except Exception as exc:  # noqa: BLE001
        return 0.0, f"c_dist=0 named_gap:polyfit_failed ({exc})"
    if not math.isfinite(slope) or slope <= 0:
        return 0.0, f"c_dist=0 (measured_slope={slope!r} consistent_with_zero_or_negative)"
    return float(slope), f"c_dist=polyfit_scatter_vs_r slope={slope:.6g} mag/deg"


def resolve_comp_weight_coeffs(
    *,
    k2_bprp: float | None = None,
    airmass_span: float = 0.0,
    r_deg: Sequence[float] | None = None,
    residual_scatter_mag: Sequence[float] | None = None,
    c_col_override: float | None = None,
    c_dist_override: float | None = None,
) -> CompWeightCoeffs:
    """Derive c_col / c_dist; overrides when finite; else measure or named zero."""
    notes: list[str] = [C_COL_PSF_TERM_GAP]
    if c_col_override is not None and math.isfinite(float(c_col_override)):
        c_col = float(c_col_override)
        c_col_src = "config_override"
    else:
        c_col, c_col_src = c_col_from_k2_airmass(k2_bprp, airmass_span)

    if c_dist_override is not None and math.isfinite(float(c_dist_override)):
        c_dist = float(c_dist_override)
        c_dist_src = "config_override"
    elif r_deg is not None and residual_scatter_mag is not None:
        c_dist, c_dist_src = measure_c_dist_mag_per_deg(
            r_deg=r_deg, residual_scatter_mag=residual_scatter_mag
        )
    else:
        c_dist, c_dist_src = 0.0, "c_dist=0 named_gap:no_regression_inputs"

    return CompWeightCoeffs(
        c_col_mag_per_bprp=float(c_col),
        c_dist_mag_per_deg=float(c_dist),
        c_col_source=str(c_col_src),
        c_dist_source=str(c_dist_src),
        notes=tuple(notes),
    )


def weights_table(
    stars: pd.DataFrame,
    *,
    target_bprp: float,
    target_ra_deg: float,
    target_dec_deg: float,
    coeffs: CompWeightCoeffs,
    id_col: str = "catalog_id",
    rms_col: str = "comp_rms",
    bprp_col: str = "bp_rp",
    ra_col: str = "ra_deg",
    dec_col: str = "dec_deg",
) -> pd.DataFrame:
    """Attach delta_bprp, r_deg, sigma_eff, weight columns (population-independent)."""
    if stars is None or getattr(stars, "empty", True):
        return pd.DataFrame()
    out = stars.copy()
    ids = out[id_col].astype(str).str.strip()
    rms = pd.to_numeric(out.get(rms_col), errors="coerce")
    bpr = pd.to_numeric(out.get(bprp_col), errors="coerce")
    ra = pd.to_numeric(out.get(ra_col, out.get("ra")), errors="coerce")
    dec = pd.to_numeric(out.get(dec_col, out.get("dec")), errors="coerce")
    tb = float(target_bprp)
    tra = float(target_ra_deg)
    tde = float(target_dec_deg)

    def _sep(row_ra: float, row_dec: float) -> float:
        if not (math.isfinite(row_ra) and math.isfinite(row_dec) and math.isfinite(tra) and math.isfinite(tde)):
            return float("nan")
        # Small-angle haversine in degrees (adequate for FOV << 90 deg).
        dra = math.radians(row_ra - tra) * math.cos(math.radians(0.5 * (row_dec + tde)))
        dde = math.radians(row_dec - tde)
        return float(math.degrees(math.hypot(dra, dde)))

    deltas: list[float] = []
    rs: list[float] = []
    ses: list[float] = []
    ws: list[float] = []
    for i in range(len(out)):
        db = float(bpr.iloc[i] - tb) if math.isfinite(tb) and math.isfinite(float(bpr.iloc[i])) else float("nan")
        r = _sep(float(ra.iloc[i]), float(dec.iloc[i]))
        se = sigma_eff_mag(
            sigma_rms_mag=float(rms.iloc[i]),
            delta_bprp=db if math.isfinite(db) else 0.0,
            r_deg=r if math.isfinite(r) else 0.0,
            c_col_mag_per_bprp=coeffs.c_col_mag_per_bprp,
            c_dist_mag_per_deg=coeffs.c_dist_mag_per_deg,
        )
        deltas.append(abs(db) if math.isfinite(db) else float("nan"))
        rs.append(r)
        ses.append(se)
        ws.append(weight_from_sigma_eff(se))
    out["delta_bprp_abs"] = deltas
    out["r_deg"] = rs
    out["sigma_eff_mag"] = ses
    out["comp_weight"] = ws
    out[id_col] = ids
    return out
