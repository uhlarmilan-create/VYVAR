"""One-off: Fáza 0+1 + 2A pre smoke log / active_targets (draft_000287)."""
from __future__ import annotations

import logging
import sys
import time
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
from photometry_core import run_phase0_and_phase1, run_phase2a  # noqa: E402


def main() -> int:
    draft = Path(r"C:\ASTRO\python\VYVAR\Archive\Drafts\draft_000287")
    setup = "NoFilter_60_2"
    ps = draft / "platesolve" / setup
    aligned = draft / "detrended_aligned" / "lights" / setup
    log_path = ps / "photometry" / "_cursor_smoke" / "run.log"
    phase01_out = ps / "photometry" / "_cursor_smoke" / "phase01"
    phase2a_out = ps / "photometry" / "_cursor_smoke" / "phase2a"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    phase01_out.mkdir(parents=True, exist_ok=True)
    phase2a_out.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(levelname)s %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)

    cfg = AppConfig()
    margin = int(getattr(cfg, "phase01_chip_interior_margin_px", 100) or 100)

    t0 = time.time()
    p01 = run_phase0_and_phase1(
        variable_targets_csv=ps / "variable_targets.csv",
        masterstars_csv=ps / "masterstars_full_match.csv",
        per_frame_csv_dir=aligned,
        output_dir=phase01_out,
        fwhm_px=3.016,
        frame_w_px=2082,
        frame_h_px=1397,
        chip_interior_margin_px=margin,
        max_mag_diff=float(getattr(cfg, "phase01_comparison_max_mag_diff", 0.25) or 0.25),
        n_comp_min=int(getattr(cfg, "phase01_comparison_n_comp_min", 3) or 3),
        n_comp_max=int(getattr(cfg, "phase01_comparison_n_comp_max", 12) or 12),
        max_comp_rms=float(getattr(cfg, "phase01_comparison_max_comp_rms", 0.05) or 0.05),
        cfg=cfg,
    )
    logging.info("[SMOKE] phase01 done %.1fs -> %s", time.time() - t0, phase01_out)

    t1 = time.time()
    _cfg2a = p01.get("cfg_effective_for_photometry") or cfg
    p2a = run_phase2a(
        masterstar_fits_path=ps / "MASTERSTAR.fits",
        active_targets_csv=Path(str(p01["active_targets_csv"])),
        comparison_stars_csv=Path(str(p01["comparison_stars_csv"])),
        per_frame_csv_dir=aligned,
        detrended_aligned_dir=aligned,
        output_dir=phase2a_out,
        fwhm_px=3.016,
        cfg=_cfg2a,
    )
    logging.info("[SMOKE] phase2a done %.1fs -> %s", time.time() - t1, phase2a_out)
    logging.info("[SMOKE] phase2a keys: %s", sorted(p2a.keys()))
    print("LOG_FILE", log_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
