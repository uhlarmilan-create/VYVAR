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
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

# COMP-POOL-02 Item 4 (draft 512, Zeiss 200 mm refractive): FWHM-rescaled COG
# enclosed-flux red-minus-blue = 29.485 mmag over BP-RP span 0.5 -> 1.5 (Delta=1.0).
# c_col_psf = 0.029485 / 1.0 mag per BP-RP. Mirror/Newton: prediction ~0.
C_COL_PSF_EE_MMAG = 29.485010546318453
C_COL_PSF_BPRP_SPAN = 1.0  # BP-RP 0.5 to 1.5
C_COL_PSF_REFRACTIVE_MAG_PER_BPRP = float(C_COL_PSF_EE_MMAG / 1000.0 / C_COL_PSF_BPRP_SPAN)
C_COL_PSF_SOURCE = (
    "MEASURED COMP-POOL-02 Item4: mmag_cog_fwhm_rescaled="
    f"{C_COL_PSF_EE_MMAG:.3f} over Delta(BP-RP)={C_COL_PSF_BPRP_SPAN:.3f} "
    f"(0.5->1.5); c_col_psf={C_COL_PSF_REFRACTIVE_MAG_PER_BPRP:.6g} mag/BP-RP"
)


@dataclass(frozen=True)
class CompWeightCoeffs:
    """Colour and distance systematic coefficients (magnitudes)."""

    c_col_mag_per_bprp: float
    c_dist_mag_per_deg: float
    c_col_source: str
    c_dist_source: str
    notes: tuple[str, ...] = ()
    c_col_k2_mag_per_bprp: float = 0.0
    c_col_psf_mag_per_bprp: float = 0.0
    c_dist_slope_unc_mag_per_deg: float = float("nan")
    c_dist_n: int = 0
    c_dist_r_value: float = float("nan")


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
        return 0.0, "c_col_k2=0 (no k2_bprp; CLEAR/unfiltered -> literature NONE)"
    dx = float(airmass_span)
    if not math.isfinite(dx) or dx < 0:
        dx = 0.0
    val = abs(float(k2_bprp)) * dx
    return float(val), f"c_col_k2=|k2_bprp|*DeltaX (|{float(k2_bprp):.6g}|*{dx:.4g})"


def c_col_psf_from_optics(optics_kind: str | None) -> tuple[float, str]:
    """Colour-dependent PSF / enclosed-flux term from optics class.

    refractive / telephoto / lens: MEASURED COMP-POOL-02 value.
    mirror / newton / cassegrain: predicted 0 (no refractive chromatic width).
    unknown: MEASURED refractive value with note (wide drafts in Archive are refractive).
    """
    kind = str(optics_kind or "unknown").strip().lower()
    if kind in ("mirror", "newton", "cassegrain", "ritchey", "rc"):
        return 0.0, "c_col_psf=0.0 predicted (mirror optics; no refractive chromatic width)"
    if kind in ("refractive", "telephoto", "lens", "zeiss", "unknown", ""):
        return (
            float(C_COL_PSF_REFRACTIVE_MAG_PER_BPRP),
            C_COL_PSF_SOURCE
            + ("" if kind in ("refractive", "telephoto", "lens", "zeiss") else " [optics_kind=unknown->refractive]"),
        )
    return (
        float(C_COL_PSF_REFRACTIVE_MAG_PER_BPRP),
        C_COL_PSF_SOURCE + f" [optics_kind={kind!r}->refractive_default]",
    )


def combine_c_col_quadrature(c_k2: float, c_psf: float) -> float:
    """Combine extinction and PSF colour terms in quadrature (independent systematics)."""
    a = float(c_k2) if math.isfinite(float(c_k2)) else 0.0
    b = float(c_psf) if math.isfinite(float(c_psf)) else 0.0
    return float(math.hypot(a, b))


def measure_c_dist_mag_per_deg(
    *,
    r_deg: Sequence[float],
    residual_scatter_mag: Sequence[float],
    min_points: int = 8,
) -> tuple[float, str, dict[str, float]]:
    """Regress residual scatter vs separation; return slope (mag/deg), note, stats.

    Estimator: ordinary least-squares ``np.polyfit`` degree 1 on (r, scatter).
    Uncertainty: OLS slope standard error from residual variance.
    Non-positive slope -> measured zero (universal answer on that rig).
    """
    stats: dict[str, float] = {
        "slope": float("nan"),
        "slope_unc": float("nan"),
        "n": 0.0,
        "r_value": float("nan"),
        "chi2_red": float("nan"),
    }
    x = np.asarray(list(r_deg), dtype=np.float64)
    y = np.asarray(list(residual_scatter_mag), dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0)
    n_ok = int(ok.sum())
    stats["n"] = float(n_ok)
    if n_ok < int(min_points):
        return 0.0, (
            f"c_dist=0 named_gap:insufficient_points n={n_ok}<{int(min_points)}"
        ), stats
    xx = x[ok]
    yy = y[ok]
    try:
        coeffs = np.polyfit(xx, yy, 1)
        slope = float(coeffs[0])
        intercept = float(coeffs[1])
        yhat = slope * xx + intercept
        resid = yy - yhat
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((yy - float(np.mean(yy))) ** 2))
        r_val = float(math.sqrt(max(0.0, 1.0 - ss_res / ss_tot))) if ss_tot > 0 else 0.0
        dof = max(1, n_ok - 2)
        chi2_red = ss_res / float(dof)
        # OLS slope SE: sigma^2 / Sxx
        sxx = float(np.sum((xx - float(np.mean(xx))) ** 2))
        sigma2 = ss_res / float(dof)
        slope_unc = float(math.sqrt(sigma2 / sxx)) if sxx > 0 else float("nan")
    except Exception as exc:  # noqa: BLE001
        return 0.0, f"c_dist=0 named_gap:polyfit_failed ({exc})", stats
    stats.update(
        {
            "slope": float(slope),
            "slope_unc": float(slope_unc),
            "r_value": float(r_val),
            "chi2_red": float(chi2_red),
        }
    )
    if not math.isfinite(slope):
        return 0.0, f"c_dist=0 (nonfinite_slope)", stats
    # Treat numerical dust and 1-sigma-consistent slopes as measured zero.
    if abs(slope) < 1e-12:
        return 0.0, (
            f"c_dist=0 MEASURED (slope={slope:.6g} numerical_zero; n={n_ok})"
        ), stats
    if (not math.isfinite(slope_unc) and slope <= 0) or (
        math.isfinite(slope_unc) and abs(slope) <= float(slope_unc)
    ):
        return 0.0, (
            f"c_dist=0 MEASURED (slope={slope:.6g}+/-{slope_unc:.6g} mag/deg "
            f"consistent_with_zero; n={n_ok}; r={r_val:.3f})"
        ), stats
    if slope <= 0:
        return 0.0, (
            f"c_dist=0 MEASURED (slope={slope:.6g} non_positive; n={n_ok})"
        ), stats
    return float(slope), (
        f"c_dist=MEASURED polyfit_scatter_vs_r slope={slope:.6g}+/-{slope_unc:.6g} "
        f"mag/deg n={n_ok} r={r_val:.3f}"
    ), stats


def resolve_comp_weight_coeffs(
    *,
    k2_bprp: float | None = None,
    airmass_span: float = 0.0,
    optics_kind: str | None = None,
    r_deg: Sequence[float] | None = None,
    residual_scatter_mag: Sequence[float] | None = None,
    c_col_override: float | None = None,
    c_dist_override: float | None = None,
) -> CompWeightCoeffs:
    """Derive c_col / c_dist; overrides when finite; else measure or named/measured zero."""
    notes: list[str] = []
    c_k2, k2_src = c_col_from_k2_airmass(k2_bprp, airmass_span)
    c_psf, psf_src = c_col_psf_from_optics(optics_kind)
    notes.append(k2_src)
    notes.append(psf_src)

    if c_col_override is not None and math.isfinite(float(c_col_override)):
        c_col = float(c_col_override)
        c_col_src = "config_override"
    else:
        c_col = combine_c_col_quadrature(c_k2, c_psf)
        c_col_src = f"quadrature(k2,psf)={c_col:.6g}"

    dist_unc = float("nan")
    dist_n = 0
    dist_r = float("nan")
    if c_dist_override is not None and math.isfinite(float(c_dist_override)):
        c_dist = float(c_dist_override)
        c_dist_src = "config_override"
    elif r_deg is not None and residual_scatter_mag is not None:
        c_dist, c_dist_src, st = measure_c_dist_mag_per_deg(
            r_deg=r_deg, residual_scatter_mag=residual_scatter_mag
        )
        dist_unc = float(st.get("slope_unc", float("nan")))
        dist_n = int(st.get("n", 0) or 0)
        dist_r = float(st.get("r_value", float("nan")))
    else:
        c_dist, c_dist_src = 0.0, "c_dist=0 named_gap:no_regression_inputs"

    return CompWeightCoeffs(
        c_col_mag_per_bprp=float(c_col),
        c_dist_mag_per_deg=float(c_dist),
        c_col_source=str(c_col_src),
        c_dist_source=str(c_dist_src),
        notes=tuple(notes),
        c_col_k2_mag_per_bprp=float(c_k2),
        c_col_psf_mag_per_bprp=float(c_psf),
        c_dist_slope_unc_mag_per_deg=float(dist_unc),
        c_dist_n=int(dist_n),
        c_dist_r_value=float(dist_r),
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


def infer_optics_kind_from_header_or_name(
    *,
    telescop: str | None = None,
    telescope_name: str | None = None,
    diameter_m: float | None = None,
    focal_m: float | None = None,
) -> str:
    """Best-effort optics class for c_col_psf. Prefer explicit name tokens."""
    blob = f"{telescop or ''} {telescope_name or ''}".lower()
    if any(t in blob for t in ("newton", "cassegrain", "ritchey", " rc", "mirror")):
        return "mirror"
    if any(t in blob for t in ("zeiss", "sonnar", "telephoto", "refractor", "lens")):
        return "refractive"
    # Heuristic: very fast small aperture + short focal often refractive telephoto.
    if (
        diameter_m is not None
        and focal_m is not None
        and math.isfinite(float(diameter_m))
        and math.isfinite(float(focal_m))
        and float(diameter_m) > 0
        and float(focal_m) / float(diameter_m) < 4.0
        and float(focal_m) < 0.5
    ):
        return "refractive"
    return "unknown"


def rewrite_comparison_stars_weights_csv(
    csv_path: Path | str,
    *,
    c_col_mag_per_bprp: float | None = None,
    c_dist_mag_per_deg: float = 0.0,
) -> dict[str, int | float]:
    """Rewrite ``comp_weight`` / ``sigma_eff_mag`` to match Phase-2A sigma_eff formula.

    PRE-IMPL-01: the Phase-1 CSV previously stored ``1/rms^2`` only, identical
    across targets. After COMP-ADMIT-03 the weights are the selection mechanism;
    the persisted artifact must describe what Phase 2A uses.
    """
    path = Path(csv_path)
    if not path.is_file():
        return {"ok": 0, "n_rows": 0}
    df = pd.read_csv(path)
    if df.empty or "catalog_id" not in df.columns:
        return {"ok": 0, "n_rows": 0}
    c_col = float(
        C_COL_PSF_REFRACTIVE_MAG_PER_BPRP if c_col_mag_per_bprp is None else c_col_mag_per_bprp
    )
    c_dist = float(c_dist_mag_per_deg) if math.isfinite(float(c_dist_mag_per_deg)) else 0.0
    weights: list[float] = []
    sigmas: list[float] = []
    for i in range(len(df)):
        rms = float(pd.to_numeric(df.iloc[i].get("comp_rms"), errors="coerce"))
        bpr = float(pd.to_numeric(df.iloc[i].get("bp_rp"), errors="coerce"))
        tb = float(pd.to_numeric(df.iloc[i].get("target_bp_rp"), errors="coerce"))
        db = abs(bpr - tb) if math.isfinite(bpr) and math.isfinite(tb) else 0.0
        ra = float(pd.to_numeric(df.iloc[i].get("ra_deg", df.iloc[i].get("ra")), errors="coerce"))
        dec = float(pd.to_numeric(df.iloc[i].get("dec_deg", df.iloc[i].get("dec")), errors="coerce"))
        # Target RA/Dec not always on row; use group median of comps for this target if needed.
        rdeg = 0.0
        if "r_deg" in df.columns:
            rdeg = float(pd.to_numeric(df.iloc[i].get("r_deg"), errors="coerce"))
            if not math.isfinite(rdeg):
                rdeg = 0.0
        se = sigma_eff_mag(
            sigma_rms_mag=rms if math.isfinite(rms) else float("nan"),
            delta_bprp=db,
            r_deg=rdeg,
            c_col_mag_per_bprp=c_col,
            c_dist_mag_per_deg=c_dist,
        )
        sigmas.append(se)
        weights.append(weight_from_sigma_eff(se))
    df["sigma_eff_mag"] = sigmas
    df["comp_weight"] = weights
    df.to_csv(path, index=False)
    # N_eff diversity check: unique N_eff across targets
    neffs = []
    if "target_catalog_id" in df.columns:
        for tid, sub in df.groupby(df["target_catalog_id"].astype(str)):
            w = pd.to_numeric(sub["comp_weight"], errors="coerce").to_numpy(dtype=float)
            w = w[np.isfinite(w) & (w > 0)]
            if w.size:
                neffs.append(float((np.sum(w) ** 2) / np.sum(w * w)))
    return {
        "ok": 1,
        "n_rows": int(len(df)),
        "n_targets": int(len(neffs)),
        "N_eff_min": float(min(neffs)) if neffs else float("nan"),
        "N_eff_max": float(max(neffs)) if neffs else float("nan"),
        "N_eff_unique_rounded": int(len({round(x, 3) for x in neffs})) if neffs else 0,
    }
