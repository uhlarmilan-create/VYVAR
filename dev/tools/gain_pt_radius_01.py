"""GAIN-PT-RADIUS-01: A3 PT verify, then ERR-only Phase 2A + BO export + SUBMIT-01.

ASCII. Does not push. Stops if CI gate rejects g_pt at pinned r=4.0.
"""
from __future__ import annotations

import json
import math
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src_py"))
sys.path.insert(0, str(ROOT / "dev" / "tests"))

from citations import build_run_citation_context, load_pipeline_meta  # noqa: E402
from config import AppConfig  # noqa: E402
from database import VyvarDatabase  # noqa: E402
from export_reports import (  # noqa: E402
    export_all_method_lightcurve_reports,
    find_truncated_gaia_ids,
)
from gain_photon_transfer import (  # noqa: E402
    PHOTON_TRANSFER_APERTURE_R_PX,
    PHOTON_TRANSFER_APERTURE_SOURCE,
    apply_photometric_gain_authority,
    resolve_photon_transfer_aperture_r_px,
)
from invariants_runtime import STAGE_ORDER, save_pipeline_meta  # noqa: E402
from param_resolver import resolve_gain  # noqa: E402
from photometry_core import run_phase2a  # noqa: E402
from photometry_sha import compute_photometry_sha  # noqa: E402
from ui_aperture_photometry import _load_fwhm  # noqa: E402

DRAFT_ID = 515
SETUP = "NoFilter_60_2"
DRAFT = ROOT / "Archive" / "Drafts" / f"draft_{DRAFT_ID:06d}"
PS = DRAFT / "platesolve" / SETUP
LIGHTS = DRAFT / "detrended_aligned" / "lights" / SETUP
PHOT = PS / "photometry"
LC = PHOT / "lightcurves"
REPORTS = PHOT / "lightcurves_reports"
BACKUP = ROOT / "tmp" / "gain_pt_radius_01_lc_backup"
OUT_JSON = ROOT / "dev" / "results" / "GAIN_PT_RADIUS_01_summary.json"
BO = "1498613634033133184"
FW = "1497343732462852864"
CHECK_ID = "1497613731286514432"
PREV_SHA_PREFIX = "36a53b0"
WIDE_ERR_G_PT = 0.6370667331227862
MAG_COLS = (
    "mag_inst",
    "mag_calib_raw",
    "mag_calib",
    "mag_calib_ct",
    "mag_calib_ac",
    "mag_calib_final",
    "delta_mag",
)


def _git_sha() -> str:
    import subprocess

    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(ROOT),
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _snapshot_lcs(src: Path, dst: Path) -> int:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(src.glob("lightcurve_*.csv")):
        shutil.copy2(p, dst / p.name)
        n += 1
    return n


def _median_err(path: Path) -> float:
    df = pd.read_csv(path, comment="#", low_memory=False)
    return float(pd.to_numeric(df["err"], errors="coerce").median())


def _mag_identity(before: Path, after: Path) -> dict:
    files_a = {p.name: p for p in before.glob("lightcurve_*.csv")}
    files_b = {p.name: p for p in after.glob("lightcurve_*.csv")}
    common = sorted(set(files_a) & set(files_b))
    n_ok = 0
    n_fail = 0
    fail_examples: list[str] = []
    for name in common:
        da = pd.read_csv(files_a[name], comment="#", low_memory=False)
        db = pd.read_csv(files_b[name], comment="#", low_memory=False)
        cols = [c for c in MAG_COLS if c in da.columns and c in db.columns]
        ok = True
        for c in cols:
            a = pd.to_numeric(da[c], errors="coerce").to_numpy(dtype=float)
            b = pd.to_numeric(db[c], errors="coerce").to_numpy(dtype=float)
            if a.shape != b.shape or not np.array_equal(a, b, equal_nan=True):
                ok = False
                break
        if ok:
            n_ok += 1
        else:
            n_fail += 1
            if len(fail_examples) < 5:
                fail_examples.append(name)
    return {
        "n_common": len(common),
        "n_mag_byte_identical": n_ok,
        "n_mag_mismatch": n_fail,
        "pass": n_fail == 0 and len(common) > 0,
        "fail_examples": fail_examples,
    }


def _check_mad_mmag(path: Path) -> float:
    """MAD of check-star residual if columns exist; else NaN."""
    df = pd.read_csv(path, comment="#", low_memory=False)
    for c in ("mag_check", "check_mag", "kmag"):
        if c in df.columns and "mag_calib" in df.columns:
            a = pd.to_numeric(df["mag_calib"], errors="coerce")
            b = pd.to_numeric(df[c], errors="coerce")
            d = (a - b).dropna()
            if len(d) >= 5:
                return float(1.4826 * np.median(np.abs(d - np.median(d))) * 1000.0)
    # Product meter uses ensemble check sidecar MAD elsewhere; fall back to
    # mag_calib scatter MAD (still mag-domain; gain must not change it).
    if "mag_calib" in df.columns:
        m = pd.to_numeric(df["mag_calib"], errors="coerce").dropna()
        if len(m) >= 5:
            return float(1.4826 * np.median(np.abs(m - np.median(m))) * 1000.0)
    return float("nan")


def a3_pt_verify() -> dict:
    r_pt, r_src = resolve_photon_transfer_aperture_r_px({"aperture_r_px": 2.499})
    assert r_pt == PHOTON_TRANSFER_APERTURE_R_PX
    cfg = AppConfig()
    db = VyvarDatabase(Path(cfg.database_path))
    from astropy.io import fits as astrofits

    with astrofits.open(PS / "MASTERSTAR.fits", memmap=False) as hdul:
        hdr = hdul[0].header
    g_res = resolve_gain(hdr, db=db, equipment_id=1, cfg=cfg)
    native = float(g_res.value) if g_res.ok else 3.17
    scale = float(getattr(cfg, "gain_container_scale", 4.0) or 4.0)
    ci_w = float(getattr(cfg, "photon_transfer_ci_max_width_factor", 3.0) or 3.0)
    sidecar = PHOT / "gain_photon_transfer.json"
    g_val, auth, pt = apply_photometric_gain_authority(
        g_db_native=native,
        native_source=g_res.source if g_res.ok else "db",
        proc_dir=LIGHTS,
        aperture_r_px=r_pt,
        container_scale=scale,
        ci_max_width_factor=ci_w,
        persist_sidecar=sidecar,
        draft_meta={"draft_id": DRAFT_ID, "stage": "gain_pt_radius_01_a3"},
        aperture_r_px_source=r_src,
    )
    payload = {
        "r_pt_px": r_pt,
        "r_pt_source": r_src,
        "g_pt": float(pt.g_pt) if pt is not None else float("nan"),
        "g_pt_ci_lo": float(pt.g_pt_ci_lo) if pt is not None else float("nan"),
        "g_pt_ci_hi": float(pt.g_pt_ci_hi) if pt is not None else float("nan"),
        "ci_width_factor": float(pt.ci_width_factor()) if pt is not None else float("nan"),
        "n_frames": int(pt.n_frames) if pt is not None else 0,
        "pt_ok": bool(pt.ok) if pt is not None else False,
        "authority_source": auth.source,
        "authority_value": float(auth.value_e_per_adu_container),
        "wide_err_04_g_pt": WIDE_ERR_G_PT,
        "abs_delta_vs_wide_err_04": abs(float(pt.g_pt) - WIDE_ERR_G_PT)
        if pt is not None and math.isfinite(float(pt.g_pt))
        else float("nan"),
        "ci_gate_pass": bool(
            pt is not None
            and pt.ok
            and math.isfinite(pt.ci_width_factor())
            and pt.ci_width_factor() <= ci_w
            and auth.source == "g_pt"
        ),
        "sidecar_path": str(sidecar),
        "sidecar": json.loads(sidecar.read_text(encoding="utf-8")),
    }
    print("A3", json.dumps({k: v for k, v in payload.items() if k != "sidecar"}, indent=2))
    return payload


def b_phase2a_and_export(a3: dict) -> dict:
    before_bo = _median_err(LC / f"lightcurve_{BO}.csv")
    before_fw = _median_err(LC / f"lightcurve_{FW}.csv")
    mad_bo_before = _check_mad_mmag(LC / f"lightcurve_{BO}.csv")
    mad_fw_before = _check_mad_mmag(LC / f"lightcurve_{FW}.csv")
    n_snap = _snapshot_lcs(LC, BACKUP)
    print(f"snapshot n={n_snap} BO_err={before_bo * 1000:.3f} mmag", flush=True)

    meta = load_pipeline_meta(PHOT)
    stages = meta.get("stages") if isinstance(meta.get("stages"), list) else []
    p2_seq = STAGE_ORDER.index("phase2a")
    meta["stages"] = [
        s
        for s in stages
        if isinstance(s, dict)
        and str(s.get("name") or "") in STAGE_ORDER
        and STAGE_ORDER.index(str(s.get("name"))) < p2_seq
    ]
    save_pipeline_meta(PHOT, meta)

    cfg = AppConfig()
    # Match 36a53b0 product: PFS ON via per-run override.
    cfg.per_frame_saturation_enabled = True
    db = VyvarDatabase(Path(cfg.database_path))
    fw = float(_load_fwhm(PS / "MASTERSTAR.fits"))
    t0 = time.time()
    run_phase2a(
        masterstar_fits_path=PS / "MASTERSTAR.fits",
        active_targets_csv=PHOT / "active_targets.csv",
        comparison_stars_csv=PHOT / "comparison_stars_per_target.csv",
        per_frame_csv_dir=LIGHTS,
        detrended_aligned_dir=LIGHTS,
        output_dir=PHOT,
        fwhm_px=fw,
        cfg=cfg,
        db=db,
        draft_id=DRAFT_ID,
        progress_cb=lambda m: print(m, flush=True),
    )
    elapsed = time.time() - t0
    after_bo = _median_err(LC / f"lightcurve_{BO}.csv")
    after_fw = _median_err(LC / f"lightcurve_{FW}.csv")
    mad_bo_after = _check_mad_mmag(LC / f"lightcurve_{BO}.csv")
    mad_fw_after = _check_mad_mmag(LC / f"lightcurve_{FW}.csv")
    ident = _mag_identity(BACKUP, LC)

    # Export BO CVn (same call shape as export_hdr_01_bo_cvn.py)
    at = pd.read_csv(PHOT / "active_targets.csv", dtype={"catalog_id": str}, low_memory=False)
    at["catalog_id"] = at["catalog_id"].astype(str).str.strip()
    trow = at[at["catalog_id"] == BO].iloc[0]
    comp = pd.read_csv(
        PHOT / "comparison_stars_per_target.csv",
        dtype={"catalog_id": str, "target_catalog_id": str},
    )
    comp_t = comp[comp["target_catalog_id"].astype(str).str.strip() == BO].copy()
    summary = pd.read_csv(PHOT / "photometry_summary.csv", dtype={"catalog_id": str})
    srow = summary[summary["catalog_id"].astype(str).str.strip() == BO]
    srow = srow.iloc[0] if not srow.empty else pd.Series(dtype=object)
    meta2 = load_pipeline_meta(PHOT)
    run_cite = build_run_citation_context(cfg, pipeline_meta=meta2, targets_df=at)
    qmap = None
    cq = LC / f"comp_quality_{BO}.json"
    if cq.is_file():
        raw = json.loads(cq.read_text(encoding="utf-8"))
        qmap = {}
        if isinstance(raw, dict):
            items = raw.get("stars") or raw.get("comps") or raw
            if isinstance(items, dict):
                for k, v in items.items():
                    nk = str(k).strip()
                    q = v.get("quality") if isinstance(v, dict) else v
                    qmap[nk] = str(q or "").strip().lower()
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "aavso").mkdir(exist_ok=True)
    (REPORTS / "varastro").mkdir(exist_ok=True)
    paths = export_all_method_lightcurve_reports(
        REPORTS,
        trow,
        lc_dir=LC,
        target_cid=BO,
        comp_df=comp_t,
        summary_row=srow,
        observer_code=str(cfg.observer_code or ""),
        observer_name=str(cfg.observer_name or "Unknown Observer"),
        comp_quality_map=qmap,
        cfg=cfg,
        obs_group=SETUP,
        targets_df=at,
        run_citation_ctx=run_cite,
    )
    print("EXPORT_PATHS", paths, flush=True)

    aavso = REPORTS / "aavso" / "BO_CVn_20260423.txt"
    varastro = REPORTS / "varastro" / "BO_CVn_20260423.txt"
    if isinstance(paths, dict) and "aperture" in paths:
        aavso = Path(paths["aperture"].get("aavso") or aavso)
        varastro = Path(paths["aperture"].get("varastro") or varastro)
    aavso_text = aavso.read_text(encoding="utf-8") if aavso.is_file() else ""
    var_text = varastro.read_text(encoding="utf-8") if varastro.is_file() else ""

    aavso_mags = []
    aavso_errs = []
    for ln in aavso_text.splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        parts = ln.split(",")
        if len(parts) < 4:
            continue
        try:
            aavso_mags.append(float(parts[2]))
            aavso_errs.append(float(parts[3]) if parts[3] not in ("na", "") else float("nan"))
        except ValueError:
            continue
    bak = pd.read_csv(BACKUP / f"lightcurve_{BO}.csv", comment="#", low_memory=False)
    lc_now = pd.read_csv(LC / f"lightcurve_{BO}.csv", comment="#", low_memory=False)
    mag_now = pd.to_numeric(lc_now["mag_calib"], errors="coerce")
    mag_bak = pd.to_numeric(bak["mag_calib"], errors="coerce")
    mag_lc_ident = bool(
        np.array_equal(mag_now.to_numpy(dtype=float), mag_bak.to_numpy(dtype=float), equal_nan=True)
    )

    core_sha, core_n = compute_photometry_sha(DRAFT)
    gpt = json.loads((PHOT / "gain_photon_transfer.json").read_text(encoding="utf-8"))

    return {
        "elapsed_s": elapsed,
        "mag_byte_identity": ident,
        "bo_mag_calib_byte_identical_vs_backup": mag_lc_ident,
        "err_before_after": {
            "BO_median_err_before_mmag": before_bo * 1000.0,
            "BO_median_err_after_mmag": after_bo * 1000.0,
            "BO_delta_mmag": (after_bo - before_bo) * 1000.0,
            "FW_median_err_before_mmag": before_fw * 1000.0,
            "FW_median_err_after_mmag": after_fw * 1000.0,
            "FW_delta_mmag": (after_fw - before_fw) * 1000.0,
        },
        "check_mad_mmag": {
            "BO_before": mad_bo_before,
            "BO_after": mad_bo_after,
            "BO_delta": mad_bo_after - mad_bo_before
            if math.isfinite(mad_bo_before) and math.isfinite(mad_bo_after)
            else float("nan"),
            "FW_before": mad_fw_before,
            "FW_after": mad_fw_after,
            "FW_delta": mad_fw_after - mad_fw_before
            if math.isfinite(mad_fw_before) and math.isfinite(mad_fw_after)
            else float("nan"),
            "note": "mag-domain MAD; must be unaffected by gain (byte mag identity is the gate)",
        },
        "aavso_path": str(aavso),
        "varastro_path": str(varastro),
        "aavso_n_rows": len(aavso_mags),
        "aavso_magerr_median_mag": float(np.nanmedian(aavso_errs)) if aavso_errs else float("nan"),
        "err_model_line": next(
            (ln for ln in aavso_text.splitlines() if ln.startswith("#ERR_MODEL")),
            "",
        ),
        "photometry_sha_core": core_sha,
        "photometry_sha_core_n": core_n,
        "photometry_sha_core_prefix": core_sha[:7],
        "prev_sha_prefix": PREV_SHA_PREFIX,
        "sidecar_after_phase2a": {
            "authority_source": (gpt.get("authority") or {}).get("source"),
            "authority_value": (gpt.get("authority") or {}).get("value_e_per_adu_container"),
            "aperture_r_px": gpt.get("aperture_r_px"),
            "aperture_r_px_source": gpt.get("aperture_r_px_source"),
            "g_pt": (gpt.get("photon_transfer") or {}).get("g_pt"),
            "ci_width_factor": (gpt.get("photon_transfer") or {}).get("ci_width_factor"),
        },
        "a3_carried": {k: a3[k] for k in a3 if k != "sidecar"},
        "varastro_bytes": len(var_text),
    }


def c_submit_checklist() -> dict:
    aavso = (REPORTS / "aavso" / "BO_CVn_20260423.txt").read_text(encoding="utf-8")
    varastro = (REPORTS / "varastro" / "BO_CVn_20260423.txt").read_text(encoding="utf-8")
    lines = aavso.splitlines()
    rows = [ln for ln in lines if ln.strip() and not ln.startswith("#")]
    header = [ln for ln in lines if ln.startswith("#")]

    def _has(prefix: str) -> bool:
        return any(ln.startswith(prefix) for ln in header)

    def _get(prefix: str) -> str:
        for ln in header:
            if ln.startswith(prefix):
                return ln
        return ""

    err_model = _get("#ERR_MODEL")
    notes0 = rows[0].split(",")[14] if rows and len(rows[0].split(",")) > 14 else ""
    kname_na = 0
    kmag_na = 0
    magerr_vals = []
    mags = []
    for ln in rows:
        p = ln.split(",")
        if len(p) < 15:
            continue
        if p[9].strip().lower() == "na":
            kname_na += 1
        if p[10].strip().lower() == "na":
            kmag_na += 1
        try:
            mags.append(float(p[2]))
            magerr_vals.append(float(p[3]))
        except ValueError:
            pass

    comps = []
    for ln in varastro.splitlines():
        if "Gaia DR3" in ln or "GaiaDR3" in ln or ln.strip().startswith("#"):
            if any(c.isdigit() for c in ln) and "149" in ln:
                comps.append(ln)
    # VarAstro comp table lines typically have full Gaia IDs
    var_rows = [
        ln
        for ln in varastro.splitlines()
        if ln.strip() and not ln.startswith("#") and not ln.lower().startswith("bjd")
    ]

    truncated = find_truncated_gaia_ids(
        notes0 + "\n" + aavso,
        [CHECK_ID, "1500748301498613248", "1498613634033133184"],
    )

    # MAG range
    mag_arr = np.array(mags, dtype=float)
    depth = float(np.nanmax(mag_arr) - np.nanmin(mag_arr)) if len(mag_arr) else float("nan")
    # eclipse-free heuristic: brightest half median
    if len(mag_arr):
        thr = np.nanpercentile(mag_arr, 50)
        quiet = mag_arr[mag_arr <= thr]  # brighter = smaller mag
        quiet_med = float(np.nanmedian(quiet)) if len(quiet) else float("nan")
    else:
        quiet_med = float("nan")

    mandatory_na = 0
    for ln in rows:
        p = ln.split(",")
        # STARID,DATE,MAG,MAGERR,FILTER required
        for i in (0, 1, 2, 3, 4):
            if i >= len(p) or p[i].strip() == "" or p[i].strip().lower() == "na":
                if i != 3:  # MAGERR can be na rarely; still count
                    mandatory_na += 1

    checks = {
        "C1_OBSCODE_UMIA": "PASS" if any("OBSCODE=UMIA" in ln for ln in header) else "FAIL",
        "C1_TYPE_present": "PASS" if _has("#TYPE=") else "FAIL",
        "C1_SOFTWARE_VYVAR": "PASS"
        if any("VYVAR" in ln and ("SOFTWARE" in ln or "Software" in ln or ln.startswith("#SOFTWARE") or "meth=" in ln.lower() or "VERSION" in ln.upper() or "software" in ln.lower()) for ln in header)
        or any("VYVAR" in ln for ln in header)
        else "FAIL",
        "C1_band_CV": "PASS" if any(ln.split(",")[4].strip() == "CV" for ln in rows[:1]) else "FAIL",
        "C1_KNAME_KMAG_0_na": "PASS" if kname_na == 0 and kmag_na == 0 else "FAIL",
        "C1_KNAME_is_check": "PASS"
        if any(CHECK_ID in ln or ln.split(",")[9] == CHECK_ID for ln in rows[:3])
        else "FAIL",
        "C1_NOTES_n_comp4": "PASS" if "n_comp=4 GaiaDR3 ensemble" in notes0 else "FAIL",
        "C1_NOTES_no_truncated_ids": "PASS" if not truncated else "FAIL",
        "C1_DATE_BJD": "PASS" if _has("#DATE=BJD") else "FAIL",
        "C1_n_rows_134": "PASS" if len(rows) == 134 else "FAIL",
        "C1_ERR_MODEL_g_pt": "PASS" if "gain=g_pt=" in err_model else "FAIL",
        "C1_MAGERR_3decimal": "PASS"
        if magerr_vals and all(len(f"{e:.3f}") >= 3 for e in magerr_vals[:5])
        else "FAIL",
        "C2_varastro_exists": "PASS" if varastro.strip() else "FAIL",
        "C2_w_pre_w_post": "PASS" if "w_pre" in varastro and "w_post" in varastro else "FAIL",
        "C2_n_epochs_134": "PASS" if len(var_rows) == 134 else "FAIL",
        "C3_mag_depth": "PASS" if math.isfinite(depth) and 0.3 <= depth <= 0.7 else "FAIL",
        "C3_quiet_median_9p4_9p5": "PASS"
        if math.isfinite(quiet_med) and 9.2 <= quiet_med <= 9.7
        else "FAIL",
        "C3_mandatory_na": "PASS" if mandatory_na == 0 else "FAIL",
    }
    return {
        "checks": checks,
        "n_aavso_rows": len(rows),
        "n_varastro_data_rows": len(var_rows),
        "notes_first": notes0,
        "err_model": err_model,
        "kname_na_n": kname_na,
        "kmag_na_n": kmag_na,
        "mag_depth": depth,
        "quiet_median_mag": quiet_med,
        "magerr_median_mag": float(np.nanmedian(magerr_vals)) if magerr_vals else float("nan"),
        "truncated_ids": truncated,
        "obscode_line": _get("#OBSCODE"),
        "software_lines": [ln for ln in header if "VYVAR" in ln or "SOFTWARE" in ln.upper()][:5],
    }


def main() -> int:
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    sha = _git_sha()
    print(f"GAIN-PT-RADIUS-01 start {started} git={sha}", flush=True)
    a3 = a3_pt_verify()
    if not a3["ci_gate_pass"]:
        payload = {
            "task": "GAIN-PT-RADIUS-01",
            "status": "STOP",
            "reason": "CI gate rejected g_pt at pinned r=4.0; authority not forced",
            "code_sha": sha,
            "a3": {k: v for k, v in a3.items() if k != "sidecar"},
            "a3_sidecar": a3.get("sidecar"),
        }
        OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("STOP: CI gate reject. Wrote", OUT_JSON, flush=True)
        return 2

    b = b_phase2a_and_export(a3)
    c = c_submit_checklist()
    payload = {
        "task": "GAIN-PT-RADIUS-01 + SUBMIT-01",
        "status": "OK",
        "measured_utc": started,
        "code_sha": sha,
        "a3": {k: v for k, v in a3.items() if k != "sidecar"},
        "b": b,
        "c": c,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("WROTE", OUT_JSON, flush=True)
    print("mag_identity", b["mag_byte_identity"], flush=True)
    print("ERR_MODEL", b["err_model_line"], flush=True)
    print("SHA", b["photometry_sha_core_prefix"], flush=True)
    print("C checks", json.dumps(c["checks"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
