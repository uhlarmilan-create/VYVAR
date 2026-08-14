"""COMP-POOL-01: draft-level noise curve and comparison-star pool admission.

Parametric model (Howell 1989 + systematic floor)::

    sigma_mag^2(m) = (2.5/ln 10)^2 * (F/g + n_pix*(sky/g + (RN/g)^2)) / F^2
                     + sigma_sys^2

Non-parametric: running median of robust scatter in magnitude bins (assumes the
bulk of field stars are non-variable).

Stage 1: fit + derived thresholds (diagnostics).
Stage 2: ``admit_pool_stars`` applies those criteria; no pool-size cap.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mag_constants import MAG_ERR_SCALE
from sigma_budget import (
    OSBORN_CY_DEFAULT,
    resolve_rig_scintillation_params,
    scintillation_sigma,
)

LOGGER = logging.getLogger(__name__)

# Assumed: majority of field stars are non-variable (standard CSI / Broeg premise).
ASSUMPTION_BULK_NONVARIABLE = (
    "Bulk of field stars are non-variable; median scatter at magnitude traces noise."
)


@dataclass
class NoiseCurveFit:
    """Parametric noise-curve fit for one draft."""

    n_stars: int
    gain_e_per_adu: float
    read_noise_e: float
    sky_adu_median: float
    aperture_area_px_median: float
    zp_inst: float
    sigma_sys_mag: float
    sigma_sys_mag_err: float | None
    chi2_red: float | None
    n_fit: int
    scint_mag_predicted: float
    scint_rel_predicted: float
    scint_airmass_used: float
    scint_params: dict[str, Any] = field(default_factory=dict)
    telescope_diameter_m_used: float = float("nan")
    telescope_diameter_m_db: float = float("nan")
    diameter_note: str = ""
    assumption: str = ASSUMPTION_BULK_NONVARIABLE


@dataclass
class DerivedPoolThresholds:
    """Derived admission thresholds (Stage 1: reported only; Stage 2: applied)."""

    faint_limit_g: float | None
    faint_limit_snr_approx: float | None
    bright_limit_g: float | None
    bright_upturn_visible: bool
    default_lin_frac: float
    detect_frac_min: float
    detect_frac_rule: str
    dilution_threshold: float | None
    dilution_rule: str
    stability_excess_mad: float | None
    stability_excess_iqr: float | None
    stability_excess_inv_eta: float | None
    stability_rule: str
    nonparametric_min_bin_n: int
    nonparametric_usable_above_g: float | None


def _robust_scatter_mag(mags: np.ndarray) -> dict[str, float]:
    """MAD and IQR scatter (estimators; no point rejection)."""
    x = np.asarray(mags, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n < 3:
        return {
            "n": float(n),
            "mad_sigma": float("nan"),
            "iqr_sigma": float("nan"),
            "std": float("nan"),
            "inv_eta": float("nan"),
        }
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    mad_sigma = 1.4826 * mad
    q75, q25 = np.percentile(x, [75.0, 25.0])
    iqr = float(q75 - q25)
    iqr_sigma = iqr / 1.349  # normal consistency
    std = float(np.std(x, ddof=1))
    # von Neumann ratio eta = mean(diff^2) / variance; 1/eta large => smooth drift
    d2 = np.diff(x)
    mean_d2 = float(np.mean(d2 * d2)) if d2.size else float("nan")
    var = float(np.var(x, ddof=1))
    if math.isfinite(var) and var > 0 and math.isfinite(mean_d2) and mean_d2 > 0:
        eta = mean_d2 / var
        inv_eta = 1.0 / eta
    else:
        inv_eta = float("nan")
    return {
        "n": float(n),
        "mad_sigma": mad_sigma,
        "iqr_sigma": iqr_sigma,
        "std": std,
        "inv_eta": float(inv_eta),
    }


def instrumental_mag_from_flux(flux: float, zp: float = 25.0) -> float:
    f = float(flux)
    if not math.isfinite(f) or f <= 0:
        return float("nan")
    return float(zp) - 2.5 * math.log10(f)


def predicted_sigma_mag_phot(
    flux_adu: float,
    *,
    sky_adu: float,
    area_px: float,
    gain: float,
    read_noise_e: float,
) -> float:
    """Howell photon+sky+RN magnitude sigma (no systematic floor)."""
    f = float(flux_adu)
    g = float(gain) if math.isfinite(gain) and gain > 0 else 1.0
    rn = float(read_noise_e) if math.isfinite(read_noise_e) and read_noise_e >= 0 else 0.0
    sky = max(0.0, float(sky_adu)) if math.isfinite(sky_adu) else 0.0
    area = max(1e-6, float(area_px)) if math.isfinite(area_px) else 1.0
    if not math.isfinite(f) or f <= 0:
        return float("nan")
    # Variance in ADU^2: F/g + n_pix*(sky/g + (RN/g)^2)
    var_adu = f / g + area * (sky / g + (rn / g) ** 2)
    if not math.isfinite(var_adu) or var_adu < 0:
        return float("nan")
    return float(MAG_ERR_SCALE * math.sqrt(var_adu) / f)


def load_star_timeseries_from_proc_dir(
    proc_dir: Path | str,
    *,
    max_files: int | None = None,
) -> pd.DataFrame:
    """Stack proc CSVs into per-star epoch rows (catalog_id, flux, sky, aperture, mag_g)."""
    d = Path(proc_dir)
    files = sorted(p for p in d.glob("proc_*.csv") if p.is_file())
    if max_files is not None:
        files = files[: int(max_files)]
    rows: list[pd.DataFrame] = []
    usecols = [
        "catalog_id",
        "dao_flux",
        "flux",
        "sky_adu_per_px_annulus",
        "aperture_r_px",
        "phot_g_mean_mag",
        "catalog_mag",
        "mag",
        "vsx_known_variable",
        "gaia_dr3_variable_catalog",
        "zone",
        "airmass",
    ]
    for p in files:
        try:
            df = pd.read_csv(p, low_memory=False)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("skip proc %s: %s", p.name, exc)
            continue
        keep = [c for c in usecols if c in df.columns]
        sub = df[keep].copy()
        sub["source_file"] = p.name
        rows.append(sub)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    if "catalog_id" in out.columns:
        out["catalog_id"] = out["catalog_id"].apply(
            lambda v: str(int(v)) if pd.notna(v) and str(v).strip() != "" else ""
        )
        try:
            # int64 Gaia IDs may round-trip via float in some CSVs
            out.loc[out["catalog_id"] == "", "catalog_id"] = ""
        except Exception:  # noqa: BLE001
            pass
    return out


def summarize_stars(
    epochs: pd.DataFrame,
    *,
    zp_inst: float = 25.0,
    min_epochs: int = 10,
) -> pd.DataFrame:
    """One row per catalog_id with robust scatter and mean photometry params."""
    if epochs.empty or "catalog_id" not in epochs.columns:
        return pd.DataFrame()
    flux_col = "dao_flux" if "dao_flux" in epochs.columns else "flux"
    records: list[dict[str, Any]] = []
    for cid, g in epochs.groupby("catalog_id"):
        if not cid or str(cid).strip() == "":
            continue
        flux = pd.to_numeric(g[flux_col], errors="coerce")
        ok = flux.notna() & (flux > 0)
        if int(ok.sum()) < int(min_epochs):
            continue
        mags = np.array([instrumental_mag_from_flux(float(f), zp_inst) for f in flux[ok]], dtype=float)
        sc = _robust_scatter_mag(mags)
        sky = pd.to_numeric(g.get("sky_adu_per_px_annulus"), errors="coerce")
        rap = pd.to_numeric(g.get("aperture_r_px"), errors="coerce")
        gmag = pd.to_numeric(g.get("phot_g_mean_mag"), errors="coerce")
        if gmag.isna().all() and "catalog_mag" in g.columns:
            gmag = pd.to_numeric(g["catalog_mag"], errors="coerce")
        if gmag.isna().all() and "mag" in g.columns:
            gmag = pd.to_numeric(g["mag"], errors="coerce")
        am = pd.to_numeric(g.get("airmass"), errors="coerce")
        vsx = False
        if "vsx_known_variable" in g.columns:
            vsx = bool(g["vsx_known_variable"].fillna(False).astype(bool).any())
        gvar = False
        if "gaia_dr3_variable_catalog" in g.columns:
            gv = g["gaia_dr3_variable_catalog"]
            if pd.api.types.is_bool_dtype(gv) or str(gv.dtype) == "bool":
                gvar = bool(gv.fillna(False).astype(bool).any())
            else:
                # string / object: non-empty and not falsey tokens
                s = gv.fillna("").astype(str).str.strip().str.lower()
                gvar = bool(s.ne("").any() and s.isin(["true", "1", "yes", "variable", "var"]).any())
        records.append(
            {
                "catalog_id": str(cid),
                "n_epochs": int(ok.sum()),
                "n_frames_file": int(len(g)),
                "detect_frac": float(ok.sum()) / float(len(g)) if len(g) else float("nan"),
                "flux_median": float(np.median(flux[ok])),
                "mag_inst_median": float(np.median(mags[np.isfinite(mags)])),
                "mag_g": float(gmag.dropna().median()) if gmag.notna().any() else float("nan"),
                "sky_median": float(sky.dropna().median()) if sky.notna().any() else float("nan"),
                "aperture_r_median": float(rap.dropna().median()) if rap.notna().any() else float("nan"),
                "airmass_median": float(am.dropna().median()) if am.notna().any() else float("nan"),
                "scatter_mad": sc["mad_sigma"],
                "scatter_iqr": sc["iqr_sigma"],
                "scatter_std": sc["std"],
                "inv_eta": sc["inv_eta"],
                "vsx_known_variable": vsx,
                "gaia_variable_flag": gvar,
            }
        )
    return pd.DataFrame.from_records(records)


def nonparametric_noise_curve(
    stars: pd.DataFrame,
    *,
    mag_col: str = "mag_g",
    scatter_col: str = "scatter_mad",
    bin_width: float = 0.5,
    min_bin_n: int = 8,
) -> pd.DataFrame:
    """Running median scatter vs magnitude (non-variable bulk assumption)."""
    if stars.empty:
        return pd.DataFrame()
    m = pd.to_numeric(stars[mag_col], errors="coerce")
    s = pd.to_numeric(stars[scatter_col], errors="coerce")
    ok = m.notna() & s.notna() & (s > 0)
    # exclude catalogue variables from the noise-bulk estimate
    if "vsx_known_variable" in stars.columns:
        ok &= ~stars["vsx_known_variable"].fillna(False).astype(bool)
    if "gaia_variable_flag" in stars.columns:
        ok &= ~stars["gaia_variable_flag"].fillna(False).astype(bool)
    df = stars.loc[ok, [mag_col, scatter_col]].copy()
    if df.empty:
        return pd.DataFrame()
    m0 = float(np.floor(df[mag_col].min() / bin_width) * bin_width)
    m1 = float(np.ceil(df[mag_col].max() / bin_width) * bin_width)
    bins = np.arange(m0, m1 + bin_width, bin_width)
    rows: list[dict[str, Any]] = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (df[mag_col] >= lo) & (df[mag_col] < hi)
        n = int(mask.sum())
        if n < 1:
            continue
        sc = df.loc[mask, scatter_col]
        rows.append(
            {
                "mag_lo": float(lo),
                "mag_hi": float(hi),
                "mag_center": float(0.5 * (lo + hi)),
                "n": n,
                "scatter_median": float(sc.median()),
                "scatter_p16": float(sc.quantile(0.16)),
                "scatter_p84": float(sc.quantile(0.84)),
                "usable": n >= int(min_bin_n),
            }
        )
    return pd.DataFrame(rows)


def fit_parametric_noise_curve(
    stars: pd.DataFrame,
    *,
    gain: float,
    read_noise_e: float,
    draft_id: int | None,
    setup: str,
    telescope_diameter_m_override: float | None = None,
) -> tuple[NoiseCurveFit, np.ndarray, np.ndarray]:
    """Fit sigma_sys so sigma_total matches observed MAD scatter (bright non-variables)."""
    m = pd.to_numeric(stars["mag_g"], errors="coerce")
    sc = pd.to_numeric(stars["scatter_mad"], errors="coerce")
    flux = pd.to_numeric(stars["flux_median"], errors="coerce")
    sky = pd.to_numeric(stars["sky_median"], errors="coerce")
    rap = pd.to_numeric(stars["aperture_r_median"], errors="coerce")
    ok = m.notna() & sc.notna() & flux.notna() & (flux > 0) & (sc > 0)
    if "vsx_known_variable" in stars.columns:
        ok &= ~stars["vsx_known_variable"].fillna(False).astype(bool)
    if "gaia_variable_flag" in stars.columns:
        ok &= ~stars["gaia_variable_flag"].fillna(False).astype(bool)

    sky_med = float(np.nanmedian(sky[ok])) if ok.any() else float("nan")
    area_med = float(np.nanmedian(math.pi * rap[ok] ** 2)) if ok.any() else float("nan")

    # Photometric prediction at each star (full ok set)
    sig_phot = np.array(
        [
            predicted_sigma_mag_phot(
                float(f),
                sky_adu=float(sk) if math.isfinite(float(sk)) else sky_med,
                area_px=math.pi * float(r) ** 2 if math.isfinite(float(r)) else area_med,
                gain=float(gain),
                read_noise_e=float(read_noise_e),
            )
            for f, sk, r in zip(flux[ok], sky[ok], rap[ok])
        ],
        dtype=float,
    )
    obs = sc[ok].to_numpy(dtype=float)
    m_ok = m[ok].to_numpy(dtype=float)

    # Systematic floor from the bright asymptote, where photon noise is negligible.
    # Using G9-13.5 mixes in rising photon/variable excess and biases sigma_sys high (P-R1).
    bright = (m_ok >= 8.0) & (m_ok <= 10.5) & np.isfinite(obs) & np.isfinite(sig_phot)
    if int(np.count_nonzero(bright)) >= 8:
        # At the bright end sigma_phot << sigma_obs; floor ~= median(obs)
        # Correct for residual photon: sys^2 = median(max(0, obs^2 - phot^2))
        resid_var = obs[bright] * obs[bright] - sig_phot[bright] * sig_phot[bright]
        resid_var = resid_var[np.isfinite(resid_var)]
        sys_var = float(np.median(np.maximum(resid_var, 0.0))) if resid_var.size else float("nan")
        n_sys = int(resid_var.size)
    else:
        resid_var = obs * obs - sig_phot * sig_phot
        resid_var = resid_var[np.isfinite(resid_var)]
        sys_var = float(np.median(np.maximum(resid_var, 0.0))) if resid_var.size else float("nan")
        n_sys = int(resid_var.size)

    if resid_var.size and math.isfinite(sys_var) and sys_var > 0:
        mad_rv = float(np.median(np.abs(resid_var - np.median(resid_var))))
        sys_err = (1.4826 * mad_rv) / (2.0 * math.sqrt(sys_var) * math.sqrt(max(1, resid_var.size)))
    else:
        sys_err = float("nan")
    sigma_sys = math.sqrt(sys_var) if math.isfinite(sys_var) and sys_var >= 0 else float("nan")

    # Validation chi2 on the wider G8-13 band (not the floor-fit band alone)
    val = (m_ok >= 8.0) & (m_ok <= 13.0) & np.isfinite(obs) & np.isfinite(sig_phot)
    pred = np.sqrt(
        np.maximum(sig_phot * sig_phot + (sigma_sys**2 if math.isfinite(sigma_sys) else 0.0), 0.0)
    )
    if int(np.count_nonzero(val)) and np.any(pred[val] > 0):
        chi = ((obs[val] - pred[val]) / pred[val]) ** 2
        chi = chi[np.isfinite(chi)]
        chi2_red = float(np.mean(chi)) if chi.size else float("nan")
    else:
        chi2_red = float("nan")

    rig = resolve_rig_scintillation_params(draft_id=draft_id, setup=setup)
    diam_db = float(rig.telescope_diameter_m)
    diam_used = float(telescope_diameter_m_override) if telescope_diameter_m_override else diam_db
    diam_note = ""
    if telescope_diameter_m_override is not None:
        diam_note = (
            f"override D={telescope_diameter_m_override} m "
            f"(DB reported {diam_db} m)"
        )
    # COMP-POOL-02 item 1: do NOT reinterpret "200mm" as aperture.
    # TELESCOPE.FOCAL=200 mm, DIAMETER=70 mm; plate scale recovers f=200 mm.
    am_med = float(np.nanmedian(pd.to_numeric(stars.get("airmass_median"), errors="coerce")))
    if not math.isfinite(am_med) or am_med < 1.0:
        am_med = 1.2
    scint_rel = scintillation_sigma(
        telescope_diameter_m=diam_used,
        airmass=am_med,
        exposure_s=float(rig.exposure_s),
        altitude_m=float(rig.altitude_m),
        c_y=float(rig.c_y if math.isfinite(rig.c_y) else OSBORN_CY_DEFAULT),
    )
    scint_mag = float(scint_rel * MAG_ERR_SCALE) if math.isfinite(scint_rel) else float("nan")

    # ZP: mag_g ? zp - 2.5 log10(F)  => zp = median(mag_g + 2.5 log10 F)
    zp_vals = m[ok] + 2.5 * np.log10(flux[ok])
    zp_inst = float(np.nanmedian(zp_vals)) if ok.any() else 25.0

    fit = NoiseCurveFit(
        n_stars=int(ok.sum()),
        gain_e_per_adu=float(gain),
        read_noise_e=float(read_noise_e),
        sky_adu_median=sky_med,
        aperture_area_px_median=area_med,
        zp_inst=zp_inst,
        sigma_sys_mag=float(sigma_sys),
        sigma_sys_mag_err=float(sys_err) if math.isfinite(sys_err) else None,
        chi2_red=float(chi2_red) if math.isfinite(chi2_red) else None,
        n_fit=int(n_sys),
        scint_mag_predicted=scint_mag,
        scint_rel_predicted=float(scint_rel) if math.isfinite(scint_rel) else float("nan"),
        scint_airmass_used=am_med,
        scint_params=rig.to_dict(),
        telescope_diameter_m_used=diam_used,
        telescope_diameter_m_db=diam_db,
        diameter_note=diam_note,
    )
    return fit, obs, pred


def derive_pool_thresholds(
    stars: pd.DataFrame,
    np_curve: pd.DataFrame,
    fit: NoiseCurveFit,
    *,
    default_lin_frac: float = 0.85,
) -> DerivedPoolThresholds:
    """Derive admission thresholds from the draft's own distributions."""
    # Detect fraction: require median detect_frac of G<=14 population (data-driven)
    # SNR-GATE-01: ~1.0 through G14, 0.507 at G15. Use 16th percentile of detect_frac
    # among stars with mag_g <= 14 as the floor (must appear that often).
    m = pd.to_numeric(stars["mag_g"], errors="coerce")
    df = pd.to_numeric(stars["detect_frac"], errors="coerce")
    brightish = (m <= 14.0) & df.notna()
    if brightish.any():
        detect_frac_min = float(np.percentile(df[brightish], 16))
        detect_rule = "p16 of detect_frac among mag_g<=14 stars in this draft"
    else:
        detect_frac_min = float(np.nanmedian(df)) if df.notna().any() else float("nan")
        detect_rule = "median detect_frac (few mag_g<=14 stars)"

    # Faint limit (operative): mag where sigma_phot(m) == sigma_sys (photon = floor).
    # Fully determined by the fit; no free SNR constant. NP 1.5x asymptote kept as cross-check.
    faint_g = None
    faint_snr = None
    if (
        math.isfinite(fit.sigma_sys_mag)
        and fit.sigma_sys_mag > 0
        and math.isfinite(fit.zp_inst)
        and math.isfinite(fit.sky_adu_median)
        and math.isfinite(fit.aperture_area_px_median)
    ):
        faint_g = _mag_where_phot_equals_sys(fit)
        if faint_g is not None:
            flux_f = 10.0 ** ((fit.zp_inst - float(faint_g)) / 2.5)
            sp_f = predicted_sigma_mag_phot(
                flux_f,
                sky_adu=fit.sky_adu_median,
                area_px=fit.aperture_area_px_median,
                gain=fit.gain_e_per_adu,
                read_noise_e=fit.read_noise_e,
            )
            stot_f = math.sqrt(sp_f * sp_f + fit.sigma_sys_mag * fit.sigma_sys_mag)
            faint_snr = float(MAG_ERR_SCALE / stot_f) if stot_f > 0 else None

    # Bright limit / upturn: scatter rises again brighter than mag ~10
    bright_upturn = False
    bright_g = None
    if not np_curve.empty and (np_curve["usable"]).any():
        uc = np_curve[np_curve["usable"]].sort_values("mag_center")
        if len(uc) >= 4:
            mid = uc[(uc["mag_center"] >= 9.0) & (uc["mag_center"] <= 12.0)]
            bri = uc[uc["mag_center"] < 9.0]
            if not mid.empty and not bri.empty:
                if float(bri["scatter_median"].median()) > 1.2 * float(mid["scatter_median"].min()):
                    bright_upturn = True
                    bright_g = float(bri["mag_center"].max())

    # Dilution: derived when dilution_factor column present; else None (named gap).
    dilution_thr = None
    dilution_rule = (
        "not derived: attach dilution_factor via dilution.py then re-derive "
        "(p16 of D over stars that pass detect+mag+catalogue filters)"
    )
    if "dilution_factor" in stars.columns:
        dser = pd.to_numeric(stars["dilution_factor"], errors="coerce")
        ok_d = dser.notna() & (dser > 0) & (dser <= 1.0)
        if "vsx_known_variable" in stars.columns:
            ok_d &= ~stars["vsx_known_variable"].fillna(False).astype(bool)
        if "gaia_variable_flag" in stars.columns:
            ok_d &= ~stars["gaia_variable_flag"].fillna(False).astype(bool)
        if ok_d.any():
            # Prefer p16; if the distribution piles at D=1 (mostly isolated), step down
            # to p10 then p05 so the gate still rejects the contaminated tail.
            for pct, label in ((16, "p16"), (10, "p10"), (5, "p05")):
                thr_try = float(np.percentile(dser[ok_d], pct))
                if thr_try < 0.999:
                    dilution_thr = thr_try
                    dilution_rule = (
                        f"{label} of dilution_factor D among non-catalogue-variable stars "
                        f"with finite D (stepped from p16 when upper pile-up at 1.0); "
                        f"admit if D >= threshold when D is measured; missing D does not reject"
                    )
                    break
            else:
                dilution_thr = None
                dilution_rule = (
                    "inert: D piles at 1.0 through p05; isolation gate not informative "
                    "at this aperture/plate-scale (named; not a silent pass-as-derived)"
                )

    # Stability excess: p84 of (scatter/pred) among bulk; IQR and inv_eta analogous
    excess_mad = None
    excess_iqr = None
    excess_inv = None
    ratios_mad: list[float] = []
    ratios_iqr: list[float] = []
    inv_vals: list[float] = []
    if math.isfinite(fit.sigma_sys_mag) and stars is not None and not stars.empty:
        for _, st in stars.iterrows():
            if st.get("vsx_known_variable") or st.get("gaia_variable_flag"):
                continue
            mg = float(st.get("mag_g", float("nan")))
            scv = float(st.get("scatter_mad", float("nan")))
            sc_iqr = float(st.get("scatter_iqr", float("nan")))
            inv = float(st.get("inv_eta", float("nan")))
            if not (math.isfinite(mg) and math.isfinite(scv) and scv > 0):
                continue
            flux = float(st.get("flux_median", float("nan")))
            sky = float(st.get("sky_median", fit.sky_adu_median))
            rap = float(st.get("aperture_r_median", float("nan")))
            area = math.pi * rap * rap if math.isfinite(rap) else fit.aperture_area_px_median
            sp = predicted_sigma_mag_phot(
                flux,
                sky_adu=sky,
                area_px=area,
                gain=fit.gain_e_per_adu,
                read_noise_e=fit.read_noise_e,
            )
            if not math.isfinite(sp):
                continue
            stot = math.sqrt(sp * sp + fit.sigma_sys_mag * fit.sigma_sys_mag)
            if stot > 0:
                ratios_mad.append(scv / stot)
                if math.isfinite(sc_iqr) and sc_iqr > 0:
                    ratios_iqr.append(sc_iqr / stot)
            if math.isfinite(inv) and inv > 0:
                inv_vals.append(inv)
        if ratios_mad:
            excess_mad = float(np.percentile(ratios_mad, 84))
        if ratios_iqr:
            excess_iqr = float(np.percentile(ratios_iqr, 84))
        if inv_vals:
            excess_inv = float(np.percentile(inv_vals, 84))

    # Nonparametric usability: first bin (bright->faint) that fails min_n
    np_lim = None
    min_bin_n = 8  # CHOSEN: NP validation usability only; not an admission threshold
    if not np_curve.empty:
        for _, row in np_curve.sort_values("mag_center").iterrows():
            if not bool(row["usable"]):
                np_lim = float(row["mag_center"])
                break

    return DerivedPoolThresholds(
        faint_limit_g=faint_g,
        faint_limit_snr_approx=faint_snr,
        bright_limit_g=bright_g,
        bright_upturn_visible=bright_upturn,
        default_lin_frac=float(default_lin_frac),
        detect_frac_min=detect_frac_min,
        detect_frac_rule=detect_rule,
        dilution_threshold=dilution_thr,
        dilution_rule=dilution_rule,
        stability_excess_mad=excess_mad,
        stability_excess_iqr=excess_iqr if excess_iqr is not None else excess_mad,
        stability_excess_inv_eta=excess_inv,
        stability_rule=(
            "p84 of (scatter/sigma_total_parametric) for MAD and IQR; "
            "p84 of inv_eta (1/vonNeumann); reject above; "
            "CSI 2264 ~3x median noise; Kjeldsen ~1.5 variability index"
        ),
        nonparametric_min_bin_n=min_bin_n,
        nonparametric_usable_above_g=np_lim,
    )


def _mag_where_phot_equals_sys(fit: NoiseCurveFit) -> float | None:
    """Binary-search G mag where sigma_phot == sigma_sys."""
    sys = float(fit.sigma_sys_mag)
    if not (math.isfinite(sys) and sys > 0):
        return None

    def _sp(mag: float) -> float:
        flux = 10.0 ** ((fit.zp_inst - mag) / 2.5)
        return predicted_sigma_mag_phot(
            flux,
            sky_adu=fit.sky_adu_median,
            area_px=fit.aperture_area_px_median,
            gain=fit.gain_e_per_adu,
            read_noise_e=fit.read_noise_e,
        )

    lo, hi = 6.0, 18.0
    spo, sph = _sp(lo), _sp(hi)
    if not (math.isfinite(spo) and math.isfinite(sph)):
        return None
    if spo >= sys:
        return float(lo)
    if sph < sys:
        return float(hi)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        sm = _sp(mid)
        if not math.isfinite(sm):
            break
        if sm < sys:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def attach_dilution_to_stars(
    stars: pd.DataFrame,
    positions: pd.DataFrame,
    *,
    gaia_db_path: str,
    aperture_arcsec: float,
    mag_limit_delta: float = 5.0,
) -> pd.DataFrame:
    """Attach dilution_factor D from Gaia neighbours (Seager/Howell definition)."""
    from dilution import compute_dilution_factor  # noqa: PLC0415

    out = stars.copy()
    if out.empty or not gaia_db_path:
        out["dilution_factor"] = float("nan")
        return out
    pos = positions.copy()
    if "catalog_id" in pos.columns:
        pos["catalog_id"] = pos["catalog_id"].map(
            lambda v: str(int(v)) if pd.notna(v) and str(v).strip() != "" else ""
        )
    id_to_row: dict[str, dict[str, Any]] = {}
    for _, r in pos.iterrows():
        cid = str(r.get("catalog_id", "") or "").strip()
        if not cid:
            continue
        id_to_row[cid] = {
            "ra": float(pd.to_numeric(r.get("ra_deg"), errors="coerce")),
            "dec": float(pd.to_numeric(r.get("dec_deg"), errors="coerce")),
            "g": float(
                pd.to_numeric(
                    r.get("phot_g_mean_mag", r.get("catalog_mag", r.get("mag"))),
                    errors="coerce",
                )
            ),
        }
    ds: list[float] = []
    for cid in out["catalog_id"].astype(str):
        meta = id_to_row.get(str(cid))
        if meta is None or not all(math.isfinite(meta[k]) for k in ("ra", "dec", "g")):
            ds.append(float("nan"))
            continue
        try:
            res = compute_dilution_factor(
                meta["ra"],
                meta["dec"],
                meta["g"],
                float(aperture_arcsec),
                str(gaia_db_path),
                catalog_id=int(cid) if cid.isdigit() else None,
                mag_limit_delta=float(mag_limit_delta),
            )
            ds.append(float(res.get("dilution_factor", float("nan"))))
        except Exception:  # noqa: BLE001
            ds.append(float("nan"))
    out["dilution_factor"] = ds
    return out


def admit_pool_stars(
    stars: pd.DataFrame,
    fit: NoiseCurveFit,
    thr: DerivedPoolThresholds,
) -> pd.DataFrame:
    """Apply Stage-1/2 derived criteria; return stars with admit flag and reject reasons.

    No pool-size cap. Colour/spatial/magnitude proximity are Stage-3 assignment only.
    """
    if stars.empty:
        return stars.copy()
    rows: list[dict[str, Any]] = []
    for _, st in stars.iterrows():
        reasons: list[str] = []
        cid = str(st.get("catalog_id", ""))
        mg = float(st.get("mag_g", float("nan")))
        dfrac = float(st.get("detect_frac", float("nan")))
        sc_mad = float(st.get("scatter_mad", float("nan")))
        sc_iqr = float(st.get("scatter_iqr", float("nan")))
        inv = float(st.get("inv_eta", float("nan")))
        dil = float(st.get("dilution_factor", float("nan"))) if "dilution_factor" in st.index else float("nan")

        if bool(st.get("vsx_known_variable")):
            reasons.append("vsx_known_variable")
        if bool(st.get("gaia_variable_flag")):
            reasons.append("gaia_variable_flag")
        if math.isfinite(thr.detect_frac_min) and (
            not math.isfinite(dfrac) or dfrac < float(thr.detect_frac_min)
        ):
            reasons.append(f"detect_frac<{thr.detect_frac_min:.4g}")
        if thr.faint_limit_g is not None and math.isfinite(mg) and mg > float(thr.faint_limit_g):
            reasons.append(f"fainter_than_{thr.faint_limit_g:.3f}")
        if thr.bright_limit_g is not None and thr.bright_upturn_visible:
            if math.isfinite(mg) and mg < float(thr.bright_limit_g):
                reasons.append(f"brighter_than_upturn_{thr.bright_limit_g:.3f}")
        if thr.dilution_threshold is not None and math.isfinite(dil):
            if dil < float(thr.dilution_threshold):
                reasons.append(f"dilution<{thr.dilution_threshold:.4g}")
        # Missing dilution: do not reject (measurement gap, not a failed isolation test).

        flux = float(st.get("flux_median", float("nan")))
        sky = float(st.get("sky_median", fit.sky_adu_median))
        rap = float(st.get("aperture_r_median", float("nan")))
        area = math.pi * rap * rap if math.isfinite(rap) else fit.aperture_area_px_median
        sp = predicted_sigma_mag_phot(
            flux,
            sky_adu=sky,
            area_px=area,
            gain=fit.gain_e_per_adu,
            read_noise_e=fit.read_noise_e,
        )
        stot = (
            math.sqrt(sp * sp + fit.sigma_sys_mag * fit.sigma_sys_mag)
            if math.isfinite(sp) and math.isfinite(fit.sigma_sys_mag)
            else float("nan")
        )
        ratio_mad = sc_mad / stot if math.isfinite(stot) and stot > 0 and math.isfinite(sc_mad) else float("nan")
        ratio_iqr = sc_iqr / stot if math.isfinite(stot) and stot > 0 and math.isfinite(sc_iqr) else float("nan")
        if thr.stability_excess_mad is not None and math.isfinite(ratio_mad):
            if ratio_mad > float(thr.stability_excess_mad):
                reasons.append(f"mad_excess>{thr.stability_excess_mad:.3g}")
        if thr.stability_excess_iqr is not None and math.isfinite(ratio_iqr):
            if ratio_iqr > float(thr.stability_excess_iqr):
                reasons.append(f"iqr_excess>{thr.stability_excess_iqr:.3g}")
        if thr.stability_excess_inv_eta is not None and math.isfinite(inv):
            if inv > float(thr.stability_excess_inv_eta):
                reasons.append(f"inv_eta>{thr.stability_excess_inv_eta:.3g}")

        rows.append(
            {
                "catalog_id": cid,
                "mag_g": mg,
                "detect_frac": dfrac,
                "scatter_mad": sc_mad,
                "scatter_iqr": sc_iqr,
                "inv_eta": inv,
                "dilution_factor": dil,
                "sigma_total_model": stot,
                "ratio_mad": ratio_mad,
                "ratio_iqr": ratio_iqr,
                "admit": len(reasons) == 0,
                "reject_reasons": ";".join(reasons),
            }
        )
    return pd.DataFrame(rows)


def analyze_draft_comp_pool(
    proc_dir: Path | str,
    *,
    draft_id: int,
    setup: str,
    gain: float,
    read_noise_e: float,
    positions: pd.DataFrame | None = None,
    gaia_db_path: str | None = None,
    aperture_arcsec: float | None = None,
    telescope_diameter_m_override: float | None = None,
    default_lin_frac: float = 0.85,
) -> dict[str, Any]:
    """Fit noise curve, derive thresholds, optionally dilution, admit pool (no cap)."""
    epochs = load_star_timeseries_from_proc_dir(proc_dir)
    # provisional ZP for instrumental mags in summarize; refined in fit
    stars = summarize_stars(epochs, zp_inst=25.0)
    if stars.empty:
        return {"error": "no_stars", "n_proc": 0}
    np_curve = nonparametric_noise_curve(stars)
    fit, _obs, _pred = fit_parametric_noise_curve(
        stars,
        gain=float(gain),
        read_noise_e=float(read_noise_e),
        draft_id=int(draft_id),
        setup=str(setup),
        telescope_diameter_m_override=telescope_diameter_m_override,
    )
    # re-summarize with fitted ZP so instrumental mags match fit
    stars = summarize_stars(epochs, zp_inst=float(fit.zp_inst))
    np_curve = nonparametric_noise_curve(stars)
    fit, _obs, _pred = fit_parametric_noise_curve(
        stars,
        gain=float(gain),
        read_noise_e=float(read_noise_e),
        draft_id=int(draft_id),
        setup=str(setup),
        telescope_diameter_m_override=telescope_diameter_m_override,
    )
    if (
        positions is not None
        and gaia_db_path
        and aperture_arcsec is not None
        and float(aperture_arcsec) > 0
    ):
        stars = attach_dilution_to_stars(
            stars,
            positions,
            gaia_db_path=str(gaia_db_path),
            aperture_arcsec=float(aperture_arcsec),
        )
    thr = derive_pool_thresholds(stars, np_curve, fit, default_lin_frac=float(default_lin_frac))
    decisions = admit_pool_stars(stars, fit, thr)
    ratio = curve_ratio_table(np_curve, fit)
    usable = ratio[ratio["usable"]] if not ratio.empty and "usable" in ratio.columns else ratio
    med_ratio = float(usable["ratio_np_over_param"].median()) if not usable.empty else float("nan")
    n_admit = int(decisions["admit"].sum()) if not decisions.empty else 0
    return {
        "draft_id": int(draft_id),
        "setup": str(setup),
        "n_proc_files": int(len(list(Path(proc_dir).glob("proc_*.csv")))),
        "n_stars": int(len(stars)),
        "n_admitted": n_admit,
        "fit": fit_to_jsonable(fit),
        "thresholds": asdict(thr),
        "scint_vs_sys": {
            "sigma_sys_mag": fit.sigma_sys_mag,
            "scint_mag_predicted": fit.scint_mag_predicted,
            "ratio_sys_over_scint": (
                float(fit.sigma_sys_mag / fit.scint_mag_predicted)
                if math.isfinite(fit.scint_mag_predicted) and fit.scint_mag_predicted > 0
                else float("nan")
            ),
            "P_R2": "report both; do not adjust",
        },
        "curve_ratio_median_usable": med_ratio,
        "stars": stars,
        "np_curve": np_curve,
        "ratio": ratio,
        "decisions": decisions,
        "assumption": ASSUMPTION_BULK_NONVARIABLE,
    }


def curve_ratio_table(
    np_curve: pd.DataFrame,
    fit: NoiseCurveFit,
) -> pd.DataFrame:
    """Ratio nonparametric / parametric at bin centers."""
    if np_curve.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, row in np_curve.iterrows():
        mc = float(row["mag_center"])
        # flux from zp and mag_g ? mag_inst for matched stars
        flux = 10.0 ** ((fit.zp_inst - mc) / 2.5)
        sp = predicted_sigma_mag_phot(
            flux,
            sky_adu=fit.sky_adu_median,
            area_px=fit.aperture_area_px_median,
            gain=fit.gain_e_per_adu,
            read_noise_e=fit.read_noise_e,
        )
        stot = math.sqrt(sp * sp + fit.sigma_sys_mag * fit.sigma_sys_mag) if math.isfinite(sp) else float("nan")
        np_med = float(row["scatter_median"])
        ratio = np_med / stot if math.isfinite(stot) and stot > 0 else float("nan")
        rows.append(
            {
                "mag_center": mc,
                "n": int(row["n"]),
                "nonparametric": np_med,
                "parametric": stot,
                "ratio_np_over_param": ratio,
                "usable": bool(row["usable"]),
            }
        )
    return pd.DataFrame(rows)


def fit_to_jsonable(fit: NoiseCurveFit) -> dict[str, Any]:
    d = asdict(fit)
    return d
