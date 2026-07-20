"""Re-export proc_*.csv + Phase 2A for draft_000307 (catalog_id float64 fix validation)."""
from __future__ import annotations

import logging
import re
import sys
import time
from io import StringIO
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import AppConfig  # noqa: E402
from pipeline import export_per_frame_catalogs  # noqa: E402
from photometry_core import run_phase2a  # noqa: E402
from ui_aperture_photometry import _load_fwhm  # noqa: E402

DRAFT = Path(r"C:\ASTRO\python\VYVAR\Archive\Drafts\draft_000307")
SETUP = "NoFilter_60_2"
CID_CHECK = "1504293848541056896"


def _setup_log_capture() -> tuple[logging.Logger, StringIO]:
    buf = StringIO()
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    root.addHandler(sh)
    fh = logging.StreamHandler(buf)
    fh.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(fh)
    return root, buf


def main() -> int:
    ps = DRAFT / "platesolve" / SETUP
    aligned = DRAFT / "detrended_aligned" / "lights" / SETUP
    phot = ps / "photometry"
    master_fits = ps / "MASTERSTAR.fits"
    master_csv = ps / "masterstars_full_match.csv"
    at_csv = phot / "active_targets.csv"
    comp_csv = phot / "comparison_stars_per_target.csv"

    for p in (master_fits, master_csv, at_csv, comp_csv, aligned):
        if not Path(p).is_file() and p != aligned:
            if not Path(p).is_dir():
                raise FileNotFoundError(p)

    cfg = AppConfig()
    _setup_log_capture()
    buf_all = StringIO()

    class _Tee(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            buf_all.write(self.format(record) + "\n")

    logging.getLogger().addHandler(_Tee())

    print("[1/2] export_per_frame_catalogs ...")
    t0 = time.time()
    per = export_per_frame_catalogs(
        frames_root=aligned,
        platesolve_dir=ps,
        max_catalog_rows=12000,
        catalog_match_max_sep_arcsec=10.0,
        saturate_level_fraction=0.95,
        faintest_mag_limit=18.0,
        dao_threshold_sigma=3.5,
        masterstars_csv=master_csv,
        masterstar_fits=master_fits,
        use_master_fast_path=True,
        app_config=cfg,
        draft_id=307,
        equipment_id=None,
    )
    print(f"    written={per.get('written')} elapsed={time.time() - t0:.1f}s")

    # Spot-check one proc CSV catalog_id dtype
    sample = next(aligned.glob("proc_*.csv"), None)
    if sample:
        import pandas as pd

        row = pd.read_csv(sample, nrows=3, dtype={"catalog_id": str, "name": str})
        print(f"    sample {sample.name} catalog_id[0]={row['catalog_id'].iloc[0]!r}")

    fw = float(_load_fwhm(master_fits))
    print(f"[2/2] run_phase2a fwhm={fw:.4f} ...")
    t1 = time.time()
    run_phase2a(
        masterstar_fits_path=master_fits,
        active_targets_csv=at_csv,
        comparison_stars_csv=comp_csv,
        per_frame_csv_dir=aligned,
        detrended_aligned_dir=aligned,
        output_dir=phot,
        fwhm_px=fw,
        cfg=cfg,
        draft_id=307,
    )
    print(f"    phase2a elapsed={time.time() - t1:.1f}s")

    log_text = buf_all.getvalue()
    report_path = phot / "_reexport_phase2a_report.txt"
    report_path.write_text(log_text, encoding="utf-8")

    xy = len(re.findall(r"XY fallback wrong star", log_text))
    am_skip = len(re.findall(r"Airmass detrend preskoceny", log_text))
    print(f"\n=== METRICS ===")
    print(f"XY fallback wrong star warnings: {xy}")
    print(f"Airmass detrend preskoceny warnings: {am_skip}")
    print(f"log saved: {report_path}")

    import pandas as pd

    summ = phot / "photometry_summary.csv"
    if summ.is_file():
        df = pd.read_csv(summ, dtype={"catalog_id": str})
        det = df["am_detrended"].astype(str).str.lower().isin(("true", "1", "yes"))
        print(f"photometry_summary: am_detrended=True {int(det.sum())} / {len(df)} total")

    # Hockey Stick (RMS) - same session as Variability UI
    try:
        from ui_variability import run_variability_detection_session  # noqa: PLC0415

        results, _n_cand, _sig = run_variability_detection_session(
            cfg=cfg,
            draft_dir=DRAFT,
            obs_group=SETUP,
            flux_col="dao_flux",
            min_frames_pct=80,
            sigma_thr=2.3,
            mag_limit=18.0,
        )
        rms_df = results.get("rms_df")
        if rms_df is not None and not rms_df.empty and "is_variable_candidate" in rms_df.columns:
            n_rms = int(pd.to_numeric(rms_df["is_variable_candidate"], errors="coerce").fillna(0).astype(bool).sum())
            print(f"Hockey Stick is_variable_candidate (RMS): {n_rms}")
        vc = phot / "variability_candidates.csv"
        if vc.is_file():
            vcdf = pd.read_csv(vc, dtype={"catalog_id": str})
            if "is_candidate_combined" in vcdf.columns:
                n_comb = int(vcdf["is_candidate_combined"].astype(str).str.lower().isin(("true", "1")).sum())
                print(f"variability_candidates.csv is_candidate_combined: {n_comb}")
    except Exception as exc:  # noqa: BLE001
        print(f"Hockey Stick run skipped: {exc}")

    at = pd.read_csv(at_csv, dtype={"catalog_id": str})
    comp = pd.read_csv(comp_csv, dtype={"catalog_id": str, "target_catalog_id": str})
    cid = CID_CHECK
    in_at = at[at["catalog_id"].astype(str).str.strip() == cid]
    in_comp = comp[comp["catalog_id"].astype(str).str.strip() == cid]
    print(f"\ncid={cid}:")
    if not in_at.empty:
        print(f"  active_targets: {in_at.iloc[0].get('vsx_name', '?')}")
    else:
        print("  active_targets: not found")
    print(f"  comparison_stars rows: {len(in_comp)} (comp-only if 0 in AT)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
