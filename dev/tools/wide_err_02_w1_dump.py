#!/usr/bin/env python3
"""WIDE-ERR-02 Part W1: production error component dump (measurement only).

Honours W1b/W1c STOP gates. Equipment id from draft_manifest.rig.equipment_id.
G for LC targets from masterstars (empirical clean set has zero overlap with LC
targets on draft 515 - named as a spec/scope defect in the JSON).
"""
from __future__ import annotations

import json
import math
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src_py"))

from config import AppConfig  # noqa: E402
from database import VyvarDatabase  # noqa: E402
from mag_constants import MAG_ERR_SCALE  # noqa: E402
from param_resolver import resolve_gain, resolve_read_noise  # noqa: E402
from photometry_core import _photometric_error  # noqa: E402
from sigma_budget import scintillation_sigma  # noqa: E402
from sigma_floor_core import resolve_sigma_sys_mag  # noqa: E402

RUN_SHA = "da9cce4a5edd1392b8ba842d3c8488589b9d0ac9"
DRAFT = ROOT / "Archive" / "Drafts" / "draft_000515"
SETUP = "NoFilter_60_2"
PHOT = DRAFT / "platesolve" / SETUP / "photometry"
LC = PHOT / "lightcurves"
EMP = ROOT / "dev" / "results" / "wide_err_515_empirical.csv"
MS = DRAFT / "platesolve" / SETUP / "masterstars_full_match.csv"
OUT = ROOT / "dev" / "results" / "WIDE_ERR_02_prod_components.json"

FITTED_GAIN_LO = 0.24
FITTED_GAIN_HI = 0.32
TEL_D_M = 0.070  # wide rig 70 mm
EXPTIME_S = 60.0


def rel_to_mmag(rel: float) -> float:
    if not math.isfinite(rel) or rel <= 0:
        return float("nan")
    return float(MAG_ERR_SCALE * rel * 1000.0)


def mag_to_mmag(m: float) -> float:
    if not math.isfinite(m) or m <= 0:
        return float("nan")
    return float(m * 1000.0)


def med_finite(vals: list[float]) -> float:
    a = np.asarray([v for v in vals if math.isfinite(v)], dtype=float)
    return float(np.median(a)) if a.size else float("nan")


def equipment_id_from_manifest(man: dict) -> int | None:
    for key in ("equipment_id", "equipmentId"):
        if man.get(key) is not None:
            try:
                return int(man[key])
            except (TypeError, ValueError):
                pass
    rig = man.get("rig")
    if isinstance(rig, dict) and rig.get("equipment_id") is not None:
        try:
            return int(rig["equipment_id"])
        except (TypeError, ValueError):
            pass
    equip = man.get("equipment")
    if isinstance(equip, dict) and equip.get("id") is not None:
        try:
            return int(equip["id"])
        except (TypeError, ValueError):
            pass
    return None


def main() -> None:
    cfg = AppConfig()
    db = VyvarDatabase(Path(cfg.database_path))
    man = json.loads((DRAFT / "draft_manifest.json").read_text(encoding="utf-8"))
    eq_id = equipment_id_from_manifest(man)

    meta = json.loads((PHOT / "pipeline_meta.json").read_text(encoding="utf-8"))
    dyn = meta.get("dynamic_params") or {}
    resolved = meta.get("resolved_facts") or {}
    observer = meta.get("observer_location") or {}
    alt_m = float(observer.get("alt_m") or 275.0)
    ap_r = float(dyn.get("aperture_r_px") or 4.0)
    area = math.pi * ap_r * ap_r
    prod_gain_meta = float(dyn.get("gain") or float("nan"))
    prod_rn_meta = float(dyn.get("read_noise") or float("nan"))

    hdr = None
    fits_candidates = list((DRAFT / "detrended_aligned" / "lights" / SETUP).glob("*.fits"))[:1]
    if fits_candidates:
        from astropy.io import fits  # noqa: PLC0415

        hdr = fits.getheader(fits_candidates[0])

    g_res = resolve_gain(hdr, db=db, equipment_id=eq_id, cfg=cfg)
    rn_res = resolve_read_noise(hdr, db=db, equipment_id=eq_id, cfg=cfg)
    gain = float(g_res.value if g_res.ok else (prod_gain_meta if math.isfinite(prod_gain_meta) else cfg.gain))
    rn = float(rn_res.value if rn_res.ok else (prod_rn_meta if math.isfinite(prod_rn_meta) else cfg.read_noise))
    # Prefer the stamped run values when present (binning-scaled RN etc.)
    if math.isfinite(prod_gain_meta) and prod_gain_meta > 0:
        gain_for_model = prod_gain_meta
        gain_model_source = "pipeline_meta.dynamic_params.gain"
    else:
        gain_for_model = gain
        gain_model_source = str(getattr(g_res, "source", None))
    if math.isfinite(prod_rn_meta) and prod_rn_meta > 0:
        rn_for_model = prod_rn_meta
        rn_model_source = "pipeline_meta.dynamic_params.read_noise"
    else:
        rn_for_model = rn
        rn_model_source = str(getattr(rn_res, "source", None))

    ssm = resolve_sigma_sys_mag(eq_id, cfg, rig_label=SETUP)

    # Typical airmass from LC jd/airmass if present
    airmass_med = 1.2
    sample_lcs = sorted(LC.glob("lightcurve_*.csv"))[:3]
    am_vals: list[float] = []
    for p in sample_lcs:
        df = pd.read_csv(p, nrows=200)
        if "airmass" in df.columns:
            am_vals.extend(
                float(x)
                for x in pd.to_numeric(df["airmass"], errors="coerce").to_numpy()
                if math.isfinite(float(x)) and float(x) >= 1.0
            )
    if am_vals:
        airmass_med = float(np.median(am_vals))
    scint_rel = scintillation_sigma(
        telescope_diameter_m=TEL_D_M,
        airmass=airmass_med,
        exposure_s=EXPTIME_S,
        altitude_m=alt_m,
        c_y=1.5,
    )
    scint_mmag = rel_to_mmag(scint_rel)
    sys_mmag = mag_to_mmag(ssm)

    emp = pd.read_csv(EMP, dtype={"catalog_id": str})
    emp["catalog_id"] = emp["catalog_id"].astype(str).str.strip()
    clean_ids = set(emp["catalog_id"])

    ms = pd.read_csv(MS, dtype={"catalog_id": str}, low_memory=False)
    ms["catalog_id"] = ms["catalog_id"].astype(str).str.strip()
    gcol = "phot_g_mean_mag" if "phot_g_mean_mag" in ms.columns else "mag"
    g_by_id = {str(r.catalog_id): float(getattr(r, gcol)) for r in ms.itertuples()}

    # --- W1a/b from published LC targets (full production assembly) ---
    per_star: dict[str, dict] = {}
    for p in sorted(LC.glob("lightcurve_*.csv")):
        tid = p.stem.replace("lightcurve_", "")
        df = pd.read_csv(p)

        def med_mmag_rel(col: str, *, allow_zero: bool = False) -> float:
            a = pd.to_numeric(df.get(col), errors="coerce").to_numpy(dtype=float)
            a = a[np.isfinite(a)]
            if not allow_zero:
                a = a[a > 0]
            else:
                a = a[a >= 0]
            if a.size == 0:
                return float("nan")
            med = float(np.median(a))
            if med == 0.0:
                return 0.0
            return rel_to_mmag(med)

        ssm_col = pd.to_numeric(df.get("sigma_sys_mag"), errors="coerce")
        ssm_med = float(np.nanmedian(ssm_col)) if ssm_col is not None and len(ssm_col) else float("nan")
        g = g_by_id.get(tid, float("nan"))
        if not math.isfinite(g):
            hit = emp[emp["catalog_id"] == tid]
            if not hit.empty:
                g = float(hit.iloc[0]["G"])

        ep = med_mmag_rel("err_photon")
        sm = med_mmag_rel("err_sem_rel")
        sc = med_mmag_rel("err_scint_rel")
        sy = med_mmag_rel("err_sigma_sys_rel", allow_zero=True)
        et = med_mmag_rel("err")
        per_star[tid] = {
            "catalog_id": tid,
            "role": "lc_target",
            "G": g,
            "in_empirical_clean_set": tid in clean_ids,
            "n_epochs": int(pd.to_numeric(df.get("err"), errors="coerce").notna().sum()),
            "err_photon_mmag_median": ep,
            "err_sem_mmag_median": sm,
            "err_scint_mmag_median": sc,
            "err_sys_mmag_median": sy,
            "err_total_mmag_median": et,
            "sys_plus_scint_hypot_mmag": (
                math.hypot(sy if math.isfinite(sy) else 0.0, sc)
                if math.isfinite(sc)
                else float("nan")
            ),
            "sigma_sys_mag_scalar": ssm_med,
            "sigma_sys_mmag_scalar": mag_to_mmag(ssm_med) if math.isfinite(ssm_med) else float("nan"),
        }

    # --- Clean-star photon recompute (production Howell params; SEM not available) ---
    clean_rows: list[dict] = []
    for r in emp.itertuples():
        flux = float(r.F)
        sbk = float(r.sbk)
        g = float(r.G)
        if not (math.isfinite(flux) and flux > 0 and math.isfinite(sbk) and sbk >= 0):
            continue
        e_rel = _photometric_error(flux, sbk, area, gain=gain_for_model, read_noise=rn_for_model)
        e_ph = rel_to_mmag(e_rel)
        # Production assembly without SEM (unknown for non-LC stars): photon + scint + sys
        e_no_sem = float("nan")
        if math.isfinite(e_ph):
            terms = [(e_ph / 1000.0 / MAG_ERR_SCALE) ** 2]
            if math.isfinite(scint_rel) and scint_rel > 0:
                terms.append(scint_rel * scint_rel)
            if ssm > 0:
                terms.append((ssm / MAG_ERR_SCALE) ** 2)
            e_no_sem = rel_to_mmag(math.sqrt(sum(terms)))
        clean_rows.append(
            {
                "catalog_id": str(r.catalog_id),
                "G": g,
                "bin": str(r.bin),
                "scat_global_zp_mmag": float(r.scat_mmag),
                "err_photon_prod_mmag": e_ph,
                "err_scint_mmag": scint_mmag,
                "err_sys_mmag": sys_mmag if sys_mmag > 0 else 0.0,
                "err_total_no_sem_mmag": e_no_sem,
                "ratio_scat_over_photon_prod": (
                    float(r.scat_mmag) / e_ph if math.isfinite(e_ph) and e_ph > 0 else float("nan")
                ),
            }
        )

    bins = [(8 + 0.5 * i, 8 + 0.5 * (i + 1)) for i in range(15)]

    def bin_label(lo: float, hi: float) -> str:
        return f"({lo:.1f}, {hi:.1f}]"

    by_bin_lc: dict[str, dict] = {}
    by_bin_clean: dict[str, dict] = {}
    for lo, hi in bins:
        rows = [r for r in per_star.values() if math.isfinite(r["G"]) and lo < r["G"] <= hi]
        if not rows:
            by_bin_lc[bin_label(lo, hi)] = {"n": 0}
        else:
            by_bin_lc[bin_label(lo, hi)] = {
                "n_lc_targets": len(rows),
                "median_err_total_mmag": med_finite([r["err_total_mmag_median"] for r in rows]),
                "median_err_photon_mmag": med_finite([r["err_photon_mmag_median"] for r in rows]),
                "median_err_sem_mmag": med_finite([r["err_sem_mmag_median"] for r in rows]),
                "median_err_scint_mmag": med_finite([r["err_scint_mmag_median"] for r in rows]),
                "median_err_sys_mmag": med_finite([r["err_sys_mmag_median"] for r in rows]),
                "median_sys_plus_scint_hypot_mmag": med_finite(
                    [r["sys_plus_scint_hypot_mmag"] for r in rows]
                ),
            }

        crows = [r for r in clean_rows if math.isfinite(r["G"]) and lo < r["G"] <= hi]
        if not crows:
            by_bin_clean[bin_label(lo, hi)] = {"n": 0}
        else:
            by_bin_clean[bin_label(lo, hi)] = {
                "n_clean_stars": len(crows),
                "median_scat_global_zp_mmag": med_finite([r["scat_global_zp_mmag"] for r in crows]),
                "median_err_photon_prod_mmag": med_finite([r["err_photon_prod_mmag"] for r in crows]),
                "median_err_total_no_sem_mmag": med_finite([r["err_total_no_sem_mmag"] for r in crows]),
                "median_ratio_scat_over_photon_prod": med_finite(
                    [r["ratio_scat_over_photon_prod"] for r in crows]
                ),
                "median_err_scint_mmag": scint_mmag,
                "median_err_sys_mmag": sys_mmag if sys_mmag > 0 else 0.0,
            }

    # G 8-9 = union of half-bins (8.0, 8.5] and (8.5, 9.0]
    bright = [r for r in per_star.values() if math.isfinite(r["G"]) and 8.0 < r["G"] <= 9.0]
    # Also report (8,10] for context when half-bin n is tiny
    bright_8_10 = [
        r for r in per_star.values() if math.isfinite(r["G"]) and 8.0 < r["G"] <= 10.0
    ]
    bright_summary = {
        "n_lc_targets_G8_9": len(bright),
        "median_sys_mmag": med_finite([r["err_sys_mmag_median"] for r in bright]),
        "median_scint_mmag": med_finite([r["err_scint_mmag_median"] for r in bright]),
        "median_sys_plus_scint_hypot_mmag": med_finite(
            [r["sys_plus_scint_hypot_mmag"] for r in bright]
        ),
        "median_err_total_mmag": med_finite([r["err_total_mmag_median"] for r in bright]),
        "median_err_photon_mmag": med_finite([r["err_photon_mmag_median"] for r in bright]),
        "median_err_sem_mmag": med_finite([r["err_sem_mmag_median"] for r in bright]),
        "stars": [
            {
                "catalog_id": r["catalog_id"],
                "G": r["G"],
                "err_total_mmag": r["err_total_mmag_median"],
                "err_photon_mmag": r["err_photon_mmag_median"],
                "err_sem_mmag": r["err_sem_mmag_median"],
                "err_scint_mmag": r["err_scint_mmag_median"],
                "err_sys_mmag": r["err_sys_mmag_median"],
                "sys_plus_scint_hypot_mmag": r["sys_plus_scint_hypot_mmag"],
            }
            for r in sorted(bright, key=lambda x: x["G"])
        ],
        "G8_10_context": {
            "n_lc_targets": len(bright_8_10),
            "median_sys_plus_scint_hypot_mmag": med_finite(
                [r["sys_plus_scint_hypot_mmag"] for r in bright_8_10]
            ),
            "median_err_total_mmag": med_finite([r["err_total_mmag_median"] for r in bright_8_10]),
            "median_err_scint_mmag": med_finite([r["err_scint_mmag_median"] for r in bright_8_10]),
            "stars": [
                {
                    "catalog_id": r["catalog_id"],
                    "G": r["G"],
                    "sys_plus_scint_hypot_mmag": r["sys_plus_scint_hypot_mmag"],
                    "err_total_mmag": r["err_total_mmag_median"],
                    "err_photon_mmag": r["err_photon_mmag_median"],
                    "err_scint_mmag": r["err_scint_mmag_median"],
                }
                for r in sorted(bright_8_10, key=lambda x: x["G"])
            ],
        },
    }

    sys_scint = bright_summary["median_sys_plus_scint_hypot_mmag"]
    w1b_stop = False
    w1b_note = ""
    if math.isfinite(sys_scint):
        if 2.5 <= sys_scint <= 5.0:
            w1b_note = (
                f"sys+scint hypot ~{sys_scint:.2f} mmag in ~3-4 mmag band; "
                "bright deficit vs LC-frame 6-8 mmag is reproduced - W1b OK to proceed."
            )
        elif sys_scint >= 6.0:
            w1b_stop = True
            w1b_note = (
                f"STOP: sys+scint hypot ~{sys_scint:.2f} mmag already at LC-frame "
                "6-8 mmag; deficit lives elsewhere - revisit W3 design."
            )
        else:
            w1b_note = (
                f"sys+scint hypot ~{sys_scint:.2f} mmag (outside 3-4 and 6-8); "
                "report; W3 design note required."
            )
    else:
        w1b_note = "W1b inconclusive: no finite sys+scint on G8-9 LC targets."

    # W1c: compare production model gain to fitted effective
    gain_cmp = gain_for_model
    fitted_mid = 0.5 * (FITTED_GAIN_LO + FITTED_GAIN_HI)
    ratio_gain = gain_cmp / fitted_mid
    w1c_stop = False
    if FITTED_GAIN_LO * 0.7 <= gain_cmp <= FITTED_GAIN_HI * 1.3:
        w1c_note = (
            f"Production gain {gain_cmp:.4f} e-/ADU within ~30% of fitted effective "
            f"{FITTED_GAIN_LO}-{FITTED_GAIN_HI}; photon term confirmed end-to-end."
        )
    elif gain_cmp > 2 * FITTED_GAIN_HI or gain_cmp < FITTED_GAIN_LO / 2:
        w1c_stop = True
        w1c_note = (
            f"STOP: production gain {gain_cmp:.4f} disagrees >2x with fitted "
            f"{FITTED_GAIN_LO}-{FITTED_GAIN_HI} (ratio to mid {ratio_gain:.2f}); "
            "photon term suspect - do not implement W3 calibration (would mask it)."
        )
    else:
        w1c_note = (
            f"Production gain {gain_cmp:.4f} vs fitted {FITTED_GAIN_LO}-{FITTED_GAIN_HI} "
            f"(ratio to mid {ratio_gain:.2f}); outside 30% but <2x - report."
        )

    # DB equipment row citation
    eq_row = None
    try:
        con = sqlite3.connect(str(Path(cfg.database_path)))
        row = con.execute(
            "SELECT ID, CAMERANAME, ALIAS, SENSORTYPE, GAIN_ADU, READNOISE_E FROM EQUIPMENTS WHERE ID=?",
            (int(eq_id) if eq_id is not None else -1,),
        ).fetchone()
        if row:
            eq_row = {
                "table": "EQUIPMENTS",
                "ID": row[0],
                "CAMERANAME": row[1],
                "ALIAS": row[2],
                "SENSORTYPE": row[3],
                "GAIN_ADU": row[4],
                "READNOISE_E": row[5],
            }
        con.close()
    except Exception as exc:  # noqa: BLE001
        eq_row = {"error": str(exc)}

    overlap = len(clean_ids & set(per_star.keys()))
    payload = {
        "task": "WIDE-ERR-02 Part W1",
        "run_sha": RUN_SHA,
        "draft_id": 515,
        "setup": SETUP,
        "frame_lc": "production LC err columns (rel-flux -> mmag via MAG_ERR_SCALE*rel*1000)",
        "frame_clean_recompute": (
            "Howell photon from empirical F,sbk with production gain/RN/aperture; "
            "scint Young/Osborn; sys from resolve_sigma_sys_mag; SEM omitted"
        ),
        "gain": {
            "value_e_per_adu_resolver": gain,
            "resolver_source": str(getattr(g_res, "source", None)),
            "value_e_per_adu_model": gain_for_model,
            "model_source": gain_model_source,
            "pipeline_meta_dynamic_gain": prod_gain_meta,
            "resolved_facts_gain": (resolved.get("gain") or {}),
            "cfg_gain_fallback": float(cfg.gain),
            "architect_fitted_effective_range": [FITTED_GAIN_LO, FITTED_GAIN_HI],
            "ratio_model_to_fitted_mid": ratio_gain,
            "equipment_id": eq_id,
            "equipment_id_source": "draft_manifest.rig.equipment_id",
            "equipment_row": eq_row,
            "header_GAIN_raw": float(hdr["GAIN"]) if hdr is not None and "GAIN" in hdr else None,
            "w1c_note": w1c_note,
            "w1c_stop": w1c_stop,
        },
        "read_noise": {
            "value_e_resolver": rn,
            "resolver_source": str(getattr(rn_res, "source", None)),
            "value_e_model": rn_for_model,
            "model_source": rn_model_source,
            "note": "dynamic_params RN 15.2 with binning 2x2; EQUIPMENTS.READNOISE_E is 7.6",
        },
        "aperture_r_px": ap_r,
        "aperture_area_px2": area,
        "sigma_sys_mag_resolved": ssm,
        "scintillation": {
            "telescope_diameter_m": TEL_D_M,
            "exposure_s": EXPTIME_S,
            "altitude_m": alt_m,
            "airmass_median_sample": airmass_med,
            "c_y": 1.5,
            "sigma_rel": scint_rel,
            "sigma_mmag": scint_mmag,
        },
        "bright_G8_9": bright_summary,
        "w1b_note": w1b_note,
        "w1b_stop": w1b_stop,
        "by_G_bin_lc_targets": by_bin_lc,
        "by_G_bin_clean_recompute": by_bin_clean,
        "per_star_lc": per_star,
        "n_lc_targets": len(per_star),
        "n_clean_stars_recomputed": len(clean_rows),
        "n_overlap_lc_vs_empirical_clean": overlap,
        "spec_defects": [
            "equipment_id is under draft_manifest.rig, not top-level (initial dump used cfg.gain=1.0).",
            "Empirical clean set (2589) and LC targets (49) have ZERO catalog_id overlap on draft 515; "
            "published err_* cannot answer W1a for the clean-star concept without recompute.",
            "Clean-star recompute omits ensemble SEM (not persisted per non-target star).",
            "sigma_sys_mag config map has key '4' only; equipment_id=1 resolves to 0.",
            "Architect fitted effective gain 0.24-0.32 e-/ADU vs production 3.17 (W1c STOP).",
        ],
        "implementation_gate": {
            "w1b_stop": w1b_stop,
            "w1c_stop": w1c_stop,
            "implement_w3": not (w1b_stop or w1c_stop),
        },
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("WROTE", OUT, flush=True)
    print("gain_model", gain_for_model, "eq", eq_id, flush=True)
    print("W1b", w1b_note, "STOP", w1b_stop, flush=True)
    print("W1c", w1c_note, "STOP", w1c_stop, flush=True)
    print("bright", {k: bright_summary[k] for k in bright_summary if k != "stars"}, flush=True)
    print("overlap", overlap, "clean_recomputed", len(clean_rows), flush=True)
    # Mid-bin clean ratio for quick console check
    mid = by_bin_clean.get("(12.0, 12.5]", {})
    print("clean G12-12.5", mid, flush=True)
    if w1b_stop or w1c_stop:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
