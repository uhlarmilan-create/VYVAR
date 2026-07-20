"""TODO-8 ePSF full verify on draft_000321 - Run1 PSF off, Run2 PSF on."""
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


def _setup_logging(log_path: Path) -> StringIO:
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
    return buf


def _summary_metrics(phot: Path) -> dict[str, float | int | None]:
    import pandas as pd

    s = pd.read_csv(phot / "photometry_summary.csv", dtype={"catalog_id": str})
    out: dict[str, float | int | None] = {}
    for name in ("BO CVn", "FW CVn"):
        m = s["vsx_name"].astype(str).str.strip() == name
        if m.any():
            r = s.loc[m].iloc[0]
            out[f"{name}_lc_rms"] = float(r.get("lc_rms")) if pd.notna(r.get("lc_rms")) else None
    return out


def _epsf_log_stats(log: str, bo_cid: str | None) -> dict[str, str | int]:
    stats: dict[str, str | int] = {}
    for pat, key in (
        (r"\[ePSF\] After isolation filter \(3xFWHM=[\d.]+px\): (\d+) PSF stars", "epsf_n_stars"),
        (r"\[ePSF\] Model built:", "epsf_built"),
        (r"PSF ePSF: extract_stars retained (\d+)", "extract_stars"),
    ):
        m = re.search(pat, log)
        if m:
            stats[key] = int(m.group(1)) if m.group(1).isdigit() else m.group(0)
    if bo_cid:
        m = re.search(
            rf"\[ePSF\] {re.escape(bo_cid)}: (\d+)/(\d+) frames using PSF flux",
            log,
        )
        if m:
            stats["bo_psf_frames"] = int(m.group(1))
            stats["bo_total_frames"] = int(m.group(2))
        m2 = re.search(rf"\[ePSF\] flux selector active for target {re.escape(bo_cid)}", log)
        if m2:
            stats["bo_flux_selector"] = 1
    return stats


def _spot_check_psf_flux(aligned: Path, bo_cid: str) -> dict[str, float | int]:
    import pandas as pd

    csvs = sorted(aligned.glob("proc_*.csv"))[:5]
    if not csvs:
        return {"spot_files": 0}
    n_finite = 0
    n_rows = 0
    for p in csvs:
        df = pd.read_csv(p, dtype={"catalog_id": str}, low_memory=False)
        if "psf_flux" not in df.columns:
            continue
        sub = df[df["catalog_id"].astype(str).str.strip() == bo_cid]
        if sub.empty:
            continue
        v = pd.to_numeric(sub["psf_flux"], errors="coerce")
        n_rows += len(v)
        n_finite += int(v.notna().sum())
    return {"spot_files": len(csvs), "bo_psf_rows": n_rows, "bo_psf_finite": n_finite}


def run_phase2a_only(cfg: AppConfig, ps: Path, aligned: Path, phot: Path, label: str) -> dict:
    buf = _setup_logging(_ROOT / f"todo8_{label}.log")
    fw = float(_load_fwhm(ps / "MASTERSTAR.fits"))
    cfg.psf_photometry_enabled = label == "run2_psf_on"
    print(f"\n=== {label} phase2a psf_enabled={cfg.psf_photometry_enabled} ===")
    t0 = time.perf_counter()
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
        print(f"ERROR {type(exc).__name__}: {exc}")
        import traceback

        traceback.print_exc()
        rc = 1
    elapsed = time.perf_counter() - t0
    log = buf.getvalue()
    (_ROOT / f"todo8_{label}.log").write_text(log, encoding="utf-8")
    metrics = _summary_metrics(phot) if rc == 0 else {}
    errs = [ln for ln in log.splitlines() if "Traceback" in ln or " ERROR " in ln]
    return {
        "label": label,
        "rc": rc,
        "elapsed_s": elapsed,
        "n_lightcurves": out.get("n_lightcurves"),
        "log": log,
        "metrics": metrics,
        "error_lines": len(errs),
        "errors_sample": errs[:5],
    }


def run2_preflight(cfg: AppConfig, ps: Path, aligned: Path) -> tuple[int, str, float]:
    """build_epsf + refresh per-frame CSVs with PSF columns."""
    buf = _setup_logging(_ROOT / "todo8_run2_preflight.log")
    cfg.psf_photometry_enabled = True
    t0 = time.perf_counter()
    rc = 0
    db = VyvarDatabase(Path(cfg.database_path))
    try:
        epsf_path = build_epsf_model(
            masterstar_fits_path=ps / "MASTERSTAR.fits",
            masterstars_csv_path=ps / "masterstars_full_match.csv",
            db=db,
            draft_id=DRAFT_ID,
        )
        print(f"build_epsf_model -> {epsf_path}")
        if not epsf_path.is_file():
            rc = 1
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
    except Exception as exc:  # noqa: BLE001
        print(f"preflight ERROR {type(exc).__name__}: {exc}")
        import traceback

        traceback.print_exc()
        rc = 1
    finally:
        db.conn.close()
    elapsed = time.perf_counter() - t0
    log = buf.getvalue()
    (_ROOT / "todo8_run2_preflight.log").write_text(log, encoding="utf-8")
    return rc, log, elapsed


def main() -> int:
    ps = DRAFT / "platesolve" / SETUP
    aligned = DRAFT / "detrended_aligned" / "lights" / SETUP
    phot = ps / "photometry"
    cfg = AppConfig()
    cfg.psf_photometry_enabled = False

    report: dict = {"draft": str(DRAFT), "setup": SETUP}

    r1 = run_phase2a_only(cfg, ps, aligned, phot, "run1_psf_off")
    report["run1"] = {k: v for k, v in r1.items() if k != "log"}

    import pandas as pd

    bo_cid: str | None = None
    s = pd.read_csv(phot / "photometry_summary.csv", dtype={"catalog_id": str})
    bo = s[s["vsx_name"].astype(str).str.strip() == "BO CVn"]
    if not bo.empty:
        bo_cid = str(bo["catalog_id"].iloc[0]).strip()

    pre_rc, pre_log, pre_elapsed = run2_preflight(cfg, ps, aligned)
    report["run2_preflight"] = {
        "rc": pre_rc,
        "elapsed_s": pre_elapsed,
        "epsf_stats": _epsf_log_stats(pre_log, bo_cid),
    }

    r2 = run_phase2a_only(cfg, ps, aligned, phot, "run2_psf_on")
    report["run2"] = {k: v for k, v in r2.items() if k != "log"}
    report["run2"]["epsf_stats"] = _epsf_log_stats(r2.get("log", ""), bo_cid)
    if bo_cid:
        report["run2"]["spot_check"] = _spot_check_psf_flux(aligned, bo_cid)

    out_path = _ROOT / "todo8_verify_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("\n=== REPORT ===")
    print(json.dumps(report, indent=2, default=str))
    rc = 0
    if r1["rc"] != 0 or pre_rc != 0 or r2["rc"] != 0:
        rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
