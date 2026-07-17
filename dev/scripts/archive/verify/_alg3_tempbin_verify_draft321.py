"""Phase 2A replay on draft_000321 — TODO-ALG-3 temporal binning verify."""
from __future__ import annotations

import logging
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

DRAFT = _ROOT / "Archive" / "Drafts" / "draft_000321"
SETUP = "NoFilter_60_2"


def main() -> int:
    ps = DRAFT / "platesolve" / SETUP
    aligned = DRAFT / "detrended_aligned" / "lights" / SETUP
    phot = ps / "photometry"
    log_path = _ROOT / "alg3_tempbin_verify.log"

    buf = StringIO()
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    for h in (logging.StreamHandler(sys.stdout), logging.StreamHandler(buf)):
        h.setLevel(logging.DEBUG)
        h.setFormatter(fmt)
        root.addHandler(h)
    logging.getLogger("photometry_core").setLevel(logging.DEBUG)

    cfg = AppConfig()
    print(
        f"temporal_binning_enabled={cfg.temporal_binning_enabled} "
        f"temporal_bin_window={cfg.temporal_bin_window} "
        f"pytics_enabled={cfg.pytics_enabled}"
    )
    fw = float(_load_fwhm(ps / "MASTERSTAR.fits"))
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
            draft_id=321,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR {type(exc).__name__}: {exc}")
        import traceback

        traceback.print_exc()
        rc = 1

    elapsed = time.perf_counter() - t0
    log = buf.getvalue()
    log_path.write_text(log, encoding="utf-8")

    print(f"exit={rc} elapsed_s={elapsed:.1f}")
    if rc == 0:
        print(f"n_lightcurves={out.get('n_lightcurves')}")
        import pandas as pd

        s = pd.read_csv(phot / "photometry_summary.csv", dtype={"catalog_id": str})
        for name in ("BO CVn", "FW CVn"):
            if "vsx_name" not in s.columns:
                continue
            m = s["vsx_name"].astype(str).str.strip() == name
            if m.any():
                r = s.loc[m].iloc[0]
                print(
                    f"{name}: lc_rms={r.get('lc_rms')} "
                    f"n_good_comp={r.get('n_good_comp')}"
                )
        errs = [
            ln
            for ln in log.splitlines()
            if "[ERROR]" in ln
            or "Traceback" in ln
            or (("ERROR" in ln) and "logging" not in ln.lower())
        ]
        print(f"error_lines={len(errs)}")
        if errs[:10]:
            print("--- first errors ---")
            for ln in errs[:10]:
                print(ln)
        tempbin = [ln for ln in log.splitlines() if "[ALG-3 TempBin]" in ln]
        print(f"tempbin_lines={len(tempbin)}")
        for ln in tempbin[:5]:
            print(ln)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
