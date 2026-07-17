"""TODO-8 ePSF final verify: re-export + Phase 2A on draft_000321 (PSF on, chi2=50)."""
from __future__ import annotations

import json
import logging
import re
import sys
import time
from io import StringIO
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import AppConfig  # noqa: E402
from database import VyvarDatabase  # noqa: E402
from photometry_core import run_phase2a  # noqa: E402
from pipeline import export_per_frame_catalogs  # noqa: E402
from psf_photometry import build_epsf_model  # noqa: E402
from ui_aperture_photometry import _load_fwhm  # noqa: E402

DRAFT = _ROOT / "Archive" / "Drafts" / "draft_000321"
SETUP = "NoFilter_60_2"
DRAFT_ID = 321
APERTURE_BASELINE = {"BO CVn_lc_rms": 0.1465770255311493, "FW CVn_lc_rms": 0.0159449004910286}


def main() -> int:
    ps = DRAFT / "platesolve" / SETUP
    aligned = DRAFT / "detrended_aligned" / "lights" / SETUP
    phot = ps / "photometry"

    buf = StringIO()
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    for h in (logging.StreamHandler(sys.stdout), logging.StreamHandler(buf)):
        h.setLevel(logging.DEBUG)
        h.setFormatter(fmt)
        root.addHandler(h)
    for name in ("photometry_core", "pipeline", "psf_photometry"):
        logging.getLogger(name).setLevel(logging.DEBUG)

    cfg = AppConfig()
    print(
        f"psf_photometry_enabled={cfg.psf_photometry_enabled} "
        f"psf_chi2_threshold={cfg.psf_chi2_threshold}"
    )
    if not cfg.psf_photometry_enabled:
        print("WARNING: config has psf_photometry_enabled=False")
    bo_cid = None
    try:
        import pandas as pd

        s0 = pd.read_csv(phot / "photometry_summary.csv", dtype={"catalog_id": str})
        bo = s0[s0["vsx_name"].astype(str).str.strip() == "BO CVn"]
        if not bo.empty:
            bo_cid = str(bo["catalog_id"].iloc[0]).strip()
    except Exception:  # noqa: BLE001
        pass

    report: dict = {"baseline_aperture": APERTURE_BASELINE}
    t_total = time.perf_counter()

    t0 = time.perf_counter()
    db = VyvarDatabase(Path(cfg.database_path))
    try:
        epsf_path = build_epsf_model(
            masterstar_fits_path=ps / "MASTERSTAR.fits",
            masterstars_csv_path=ps / "masterstars_full_match.csv",
            db=db,
            draft_id=DRAFT_ID,
        )
        print(f"build_epsf_model -> {epsf_path}")
    finally:
        db.conn.close()
    report["epsf_build_s"] = round(time.perf_counter() - t0, 1)

    t1 = time.perf_counter()
    export_per_frame_catalogs(
        frames_root=aligned,
        platesolve_dir=ps,
        masterstars_csv=ps / "masterstars_full_match.csv",
        masterstar_fits=ps / "MASTERSTAR.fits",
        use_master_fast_path=True,
        app_config=cfg,
        draft_id=DRAFT_ID,
        equipment_id=1,
    )
    report["export_catalogs_s"] = round(time.perf_counter() - t1, 1)

    # Spot-check frame 050
    import pandas as pd

    p050 = aligned / "proc_BO_CVn_Light_050.csv"
    if bo_cid and p050.is_file():
        df = pd.read_csv(p050, dtype={"catalog_id": str}, low_memory=False)
        sub = df[df["catalog_id"].astype(str).str.strip() == bo_cid]
        if not sub.empty:
            r = sub.iloc[0]
            report["frame050"] = {
                "psf_chi2": float(pd.to_numeric(r.get("psf_chi2"), errors="coerce")),
                "psf_fit_ok": bool(r.get("psf_fit_ok")),
                "psf_flux_finite": bool(pd.notna(pd.to_numeric(r.get("psf_flux"), errors="coerce"))),
            }
        chi = pd.to_numeric(df["psf_chi2"], errors="coerce")
        ok = df["psf_fit_ok"].fillna(False).astype(bool)
        report["frame050_all"] = {
            "psf_fit_ok_true": int(ok.sum()),
            "n_rows": len(df),
            "chi2_lt_50": int((chi < 50).sum()),
        }

    if bo_cid:
        fin = ok_cnt = 0
        for p in aligned.glob("proc_*.csv"):
            d = pd.read_csv(p, dtype={"catalog_id": str}, usecols=["catalog_id", "psf_fit_ok"], low_memory=False)
            sub = d[d["catalog_id"].astype(str).str.strip() == bo_cid]
            if sub.empty:
                continue
            ok_cnt += int(sub["psf_fit_ok"].fillna(False).astype(bool).sum())
            fin += len(sub)
        report["bo_all_frames"] = {"psf_fit_ok_true": ok_cnt, "n_frames": fin}

    t2 = time.perf_counter()
    fw = float(_load_fwhm(ps / "MASTERSTAR.fits"))
    rc = 0
    out: dict = {}
    try:
        out = run_phase2a(
            masterstar_fits_path=ps / "MASTERSTAR.fits",
            active_targets_csv=phot / "active_targets.csv",
            comparison_stars_csv=phot / "comparison_stars_per_target.csv",
            per_frame_csv_dir=aligned,
            detrended_aligned_dir=aligned,
            output_dir=phot,
            fwhm_px=fw,
            cfg=cfg,
            draft_id=DRAFT_ID,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"phase2a ERROR {exc}")
        import traceback

        traceback.print_exc()
        rc = 1
    report["phase2a_s"] = round(time.perf_counter() - t2, 1)
    report["total_s"] = round(time.perf_counter() - t_total, 1)
    report["exit_code"] = rc
    report["n_lightcurves"] = out.get("n_lightcurves")

    log = buf.getvalue()
    log_path = _ROOT / "todo8_final_verify.log"
    log_path.write_text(log, encoding="utf-8")

    metrics = {}
    if rc == 0:
        s = pd.read_csv(phot / "photometry_summary.csv", dtype={"catalog_id": str})
        for name in ("BO CVn", "FW CVn"):
            m = s["vsx_name"].astype(str).str.strip() == name
            if m.any():
                v = float(s.loc[m, "lc_rms"].iloc[0])
                metrics[f"{name}_lc_rms"] = v
                base = APERTURE_BASELINE.get(f"{name}_lc_rms")
                if base is not None:
                    metrics[f"{name}_delta_vs_aperture"] = v - base
    report["metrics"] = metrics

    epsf_lines = [ln for ln in log.splitlines() if "[ePSF]" in ln]
    report["epsf_log_line_count"] = len(epsf_lines)
    if bo_cid:
        for ln in epsf_lines:
            if bo_cid in ln and "frames using PSF flux" in ln:
                report["bo_psf_lc_line"] = ln.strip()
                m = re.search(r"(\d+)/(\d+) frames using PSF flux", ln)
                if m:
                    report["bo_frames_using_psf"] = int(m.group(1))
                    report["bo_frames_total"] = int(m.group(2))
        if "flux selector active" not in str(report.get("bo_psf_lc_line", "")):
            for ln in epsf_lines:
                if "flux selector active" in ln and str(bo_cid) in ln:
                    report["bo_flux_selector_line"] = ln.strip()

    errs = [ln for ln in log.splitlines() if "Traceback" in ln or " ERROR " in ln]
    psf_fail = [ln for ln in log.splitlines() if "per-frame PSF failed" in ln]
    report["error_lines"] = len(errs)
    report["psf_fail_lines"] = len(psf_fail)
    if errs:
        report["errors_sample"] = errs[:5]

    out_path = _ROOT / "todo8_final_verify_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("\n=== FINAL REPORT ===")
    print(json.dumps(report, indent=2, default=str))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
