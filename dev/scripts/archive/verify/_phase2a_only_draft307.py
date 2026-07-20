"""Phase 2A only on draft_000307 (after catalog_only LC skip)."""
from __future__ import annotations

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
from photometry_core import run_phase2a  # noqa: E402
from ui_aperture_photometry import _load_fwhm  # noqa: E402

DRAFT = Path(r"C:\ASTRO\python\VYVAR\Archive\Drafts\draft_000307")
SETUP = "NoFilter_60_2"


def main() -> int:
    ps = DRAFT / "platesolve" / SETUP
    aligned = DRAFT / "detrended_aligned" / "lights" / SETUP
    phot = ps / "photometry"
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

    cfg = AppConfig()
    fw = float(_load_fwhm(ps / "MASTERSTAR.fits"))
    t0 = time.time()
    run_phase2a(
        masterstar_fits_path=ps / "MASTERSTAR.fits",
        active_targets_csv=phot / "active_targets.csv",
        comparison_stars_csv=phot / "comparison_stars_per_target.csv",
        per_frame_csv_dir=aligned,
        detrended_aligned_dir=aligned,
        output_dir=phot,
        fwhm_px=fw,
        cfg=cfg,
        draft_id=307,
    )
    log = buf.getvalue()
    (phot / "_phase2a_catalog_only_skip_report.txt").write_text(log, encoding="utf-8")
    xy = len(re.findall(r"XY fallback wrong star", log))
    am = len(re.findall(r"Airmass detrend preskoceny", log))
    skip_n = re.search(r"Skipping (\d+) catalog_only targets", log)
    print(f"elapsed={time.time()-t0:.1f}s skip_catalog_only={skip_n.group(1) if skip_n else '?'}")
    print(f"XY fallback={xy} airmass_skip={am}")
    import pandas as pd

    s = pd.read_csv(phot / "photometry_summary.csv", dtype={"catalog_id": str})
    det = s["am_detrended"].astype(str).str.lower().isin(("true", "1"))
    print(f"summary rows={len(s)} am_detrended={int(det.sum())}/{len(s)}")
    co = 0
    if "zone_flag" in s.columns:
        co = int((s["zone_flag"].astype(str).str.lower() == "catalog_only").sum())
    print(f"catalog_only in summary={co}")
    try:
        from ui_variability import run_variability_detection_session  # noqa: PLC0415

        res, _, _ = run_variability_detection_session(
            cfg=cfg, draft_dir=DRAFT, obs_group=SETUP, flux_col="dao_flux",
            min_frames_pct=80, sigma_thr=2.3, mag_limit=18.0,
        )
        rms = res.get("rms_df")
        if rms is not None and "is_variable_candidate" in rms.columns:
            print(f"HS is_variable_candidate={int(rms['is_variable_candidate'].sum())}")
    except Exception as exc:  # noqa: BLE001
        print(f"HS skipped: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
