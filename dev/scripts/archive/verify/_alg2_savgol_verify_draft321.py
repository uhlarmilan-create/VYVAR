"""Phase 2A replay draft_000321 — TODO-ALG-2 Savitzky-Golay verify (2 runs)."""
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


def _run_once(*, sg_enabled: bool, log_suffix: str) -> int:
    ps = DRAFT / "platesolve" / SETUP
    aligned = DRAFT / "detrended_aligned" / "lights" / SETUP
    phot = ps / "photometry"
    log_path = _ROOT / f"alg2_savgol_verify_{log_suffix}.log"

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
    cfg.savgol_detrend_enabled = bool(sg_enabled)
    cfg.savgol_window_frac = 0.3
    cfg.savgol_polyorder = 2

    label = "enabled" if sg_enabled else "disabled"
    print(f"\n=== Run: savgol_detrend_enabled={cfg.savgol_detrend_enabled} ({label}) ===")
    fw = float(_load_fwhm(ps / "MASTERSTAR.fits"))
    t0 = time.perf_counter()
    rc = 0
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
        out = {}

    elapsed = time.perf_counter() - t0
    log = buf.getvalue()
    log_path.write_text(log, encoding="utf-8")

    print(f"exit={rc} elapsed_s={elapsed:.1f} log={log_path.name}")
    if rc == 0:
        print(f"n_lightcurves={out.get('n_lightcurves')}")
        import pandas as pd

        s = pd.read_csv(phot / "photometry_summary.csv", dtype={"catalog_id": str})
        for name in ("BO CVn", "FW CVn"):
            m = s["vsx_name"].astype(str).str.strip() == name
            if m.any():
                r = s.loc[m].iloc[0]
                print(f"{name}: lc_rms={r.get('lc_rms')} n_good_comp={r.get('n_good_comp')}")
        errs = [
            ln
            for ln in log.splitlines()
            if "Traceback" in ln or ("[ERROR]" in ln)
        ]
        print(f"error_lines={len(errs)}")
        sg = [ln for ln in log.splitlines() if "[ALG-2 SG]" in ln]
        print(f"alg2_sg_lines={len(sg)}")
        for ln in sg[:5]:
            print(ln)
    return rc


def main() -> int:
    rc1 = _run_once(sg_enabled=False, log_suffix="disabled")
    rc2 = _run_once(sg_enabled=True, log_suffix="enabled")
    return rc1 or rc2


if __name__ == "__main__":
    raise SystemExit(main())
