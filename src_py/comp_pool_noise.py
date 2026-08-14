"""COMP-POOL-01 Stage 1: draft-level noise curve for comparison-star pool admission.

Parametric model (Howell 1989 + systematic floor)::

    sigma_mag^2(m) = (2.5/ln 10)^2 * (F/g + n_pix*(sky/g + (RN/g)^2)) / F^2
                     + sigma_sys^2

Non-parametric: running median of robust scatter in magnitude bins (assumes the
bulk of field stars are non-variable).

Diagnostics only in Stage 1 -- selection code does not call these thresholds yet.
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
            f"(DB reported {diam_db} m; name implies 200 mm wide-field)"
        )
    elif diam_db < 0.15 and draft_id in (510, 512, 435, 509):
        # Known Jirny Carl-Zeiss 200 mm: DB DIAMETER sometimes wrong
        diam_used = 0.2
        diam_note = f"corrected D=0.2 m for 200 mm wide-field (DB had {diam_db} m)"
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

    # Faint limit: where usable NP median exceeds 1.5 x bright asymptote (G8-10 usable bins)
    bright_asym = float("nan")
    if not np_curve.empty and (np_curve["usable"]).any():
        uc = np_curve[np_curve["usable"]]
        bright_bins = uc[(uc["mag_center"] >= 8.0) & (uc["mag_center"] <= 10.0)]
        if not bright_bins.empty:
            bright_asym = float(bright_bins["scatter_median"].median())
    faint_g = None
    faint_snr = None
    if math.isfinite(bright_asym) and bright_asym > 0 and not np_curve.empty:
        for _, row in np_curve.sort_values("mag_center").iterrows():
            if not row["usable"]:
                continue
            if float(row["scatter_median"]) >= 1.5 * bright_asym and float(row["mag_center"]) > 10.0:
                faint_g = float(row["mag_center"])
                faint_snr = float(MAG_ERR_SCALE / float(row["scatter_median"]))
                break

    # Bright limit / upturn: scatter rises again brighter than mag ~10
    bright_upturn = False
    bright_g = None
    if not np_curve.empty and (np_curve["usable"]).any():
        uc = np_curve[np_curve["usable"]].sort_values("mag_center")
        if len(uc) >= 4:
            # look for minimum then rise toward bright end
            mid = uc[(uc["mag_center"] >= 9.0) & (uc["mag_center"] <= 12.0)]
            bri = uc[uc["mag_center"] < 9.0]
            if not mid.empty and not bri.empty:
                if float(bri["scatter_median"].median()) > 1.2 * float(mid["scatter_median"].min()):
                    bright_upturn = True
                    bright_g = float(bri["mag_center"].max())

    # Dilution: not measured in Stage 1 without Gaia neighbour query; placeholder rule
    dilution_thr = None
    dilution_rule = (
        "Stage 1: not derived here (needs dilution.py batch). "
        "Stage 2 will use p16 of D among isolated-looking stars."
    )

    # Stability excess: compare scatter/model; threshold = p84 of (scatter/pred) among bulk
    excess_mad = None
    if fit.sigma_sys_mag == fit.sigma_sys_mag and stars is not None and not stars.empty:
        # use nonparametric median as pred when available
        ratios: list[float] = []
        for _, st in stars.iterrows():
            if st.get("vsx_known_variable") or st.get("gaia_variable_flag"):
                continue
            mg = float(st.get("mag_g", float("nan")))
            scv = float(st.get("scatter_mad", float("nan")))
            if not (math.isfinite(mg) and math.isfinite(scv) and scv > 0):
                continue
            # parametric prediction
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
                ratios.append(scv / stot)
        if ratios:
            excess_mad = float(np.percentile(ratios, 84))

    # Nonparametric usability: faintest mag where all brighter usable bins have n>=min
    np_lim = None
    min_bin_n = 8
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
        stability_excess_iqr=excess_mad,  # same population rule until separate IQR fit
        stability_excess_inv_eta=None,
        stability_rule=(
            "p84 of (scatter_mad / sigma_total_parametric) among non-catalogue-variable stars; "
            "CSI 2264 used ~3x median noise; Kjeldsen ~1.5 variability index"
        ),
        nonparametric_min_bin_n=min_bin_n,
        nonparametric_usable_above_g=np_lim,
    )


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
