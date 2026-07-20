"""GS11 Step B validation: enable gs11, phase2a + phase1 comp selection, report."""
from __future__ import annotations

import json
import logging
import re
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from config import AppConfig, config_json_path  # noqa: E402
from photometry_core import run_phase2a  # noqa: E402
from ui_aperture_photometry import _load_fwhm  # noqa: E402

DRAFT = _ROOT / "Archive/Drafts/draft_000342"
SETUP = "NoFilter_60_2"
PHOT = DRAFT / "platesolve" / SETUP / "photometry"
BLENDED = {
    "1499974726349018112",
    "1498688469542868992",
    "1498072743031629824",
    "1497368849430107904",
    "1499187269863874304",
}


def _set_gs11_enabled(enabled: bool) -> dict:
    path = config_json_path(_ROOT)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["gs11_dilution_enabled"] = bool(enabled)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return data


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    buf: list[str] = []

    class _H(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            buf.append(record.getMessage())

    logging.getLogger().addHandler(_H())

    print("=== Enable gs11_dilution_enabled=True ===")
    _set_gs11_enabled(True)
    cfg = AppConfig()
    assert cfg.gs11_dilution_enabled is True

    print("\n=== Phase 2A re-run ===")
    ps = DRAFT / "platesolve" / SETUP
    t0 = time.time()
    run_phase2a(
        masterstar_fits_path=ps / "MASTERSTAR.fits",
        active_targets_csv=PHOT / "active_targets.csv",
        comparison_stars_csv=PHOT / "comparison_stars_per_target.csv",
        per_frame_csv_dir=DRAFT / "detrended_aligned" / "lights" / SETUP,
        detrended_aligned_dir=DRAFT / "detrended_aligned" / "lights" / SETUP,
        output_dir=PHOT,
        fwhm_px=float(_load_fwhm(ps / "MASTERSTAR.fits")),
        cfg=cfg,
        draft_id=342,
    )
    print(f"phase2a elapsed={time.time() - t0:.1f}s")

    gs11_corr = [ln for ln in buf if "GS11 dilution correction" in ln]
    gs11_skip = [ln for ln in buf if "GS11:" in ln and "too low" in ln]
    print(f"targets corrected (log lines): {len(gs11_corr)}")
    print(f"targets skipped low-D (log lines): {len(gs11_skip)}")

    summ = PHOT / "photometry_summary.csv"
    if summ.is_file():
        import pandas as pd

        df = pd.read_csv(summ, dtype={"catalog_id": str})
        print("\nphotometry_summary dilution_factor (first 10):")
        cols = ["catalog_id", "vsx_name", "dilution_factor", "dilution_delta_mag", "n_neighbors_aperture"]
        cols = [c for c in cols if c in df.columns]
        print(df[cols].head(10).to_string(index=False))

    print("\n=== Phase 1 comp selection (full re-run, gs11 on) ===")
    buf.clear()
    from photometry_core import run_phase0_and_phase1  # noqa: E402

    t1 = time.time()
    run_phase0_and_phase1(
        variable_targets_csv=PHOT / "variable_targets.csv",
        masterstars_csv=ps / "masterstars_full_match.csv",
        per_frame_csv_dir=DRAFT / "detrended_aligned" / "lights" / SETUP,
        output_dir=PHOT,
        fwhm_px=float(_load_fwhm(ps / "MASTERSTAR.fits")),
        plate_scale_arcsec_px=float(cfg.phase01_plate_scale_arcsec_per_px or 1.3) or 1.3,
        cfg=cfg,
    )
    print(f"phase0+1 elapsed={time.time() - t1:.1f}s")

    gs11_rej = [ln for ln in buf if "GS11 dilution filter vylucil" in ln]
    n_rej_comps = sum(int(m.group(1)) for ln in gs11_rej for m in [re.search(r"vylucil (\d+)", ln)] if m)
    print(f"GS11 comp reject log events: {len(gs11_rej)} (sum comps={n_rej_comps})")

    comp = PHOT / "comparison_stars_per_target.csv"
    import pandas as pd

    cdf = pd.read_csv(comp, dtype={"catalog_id": str})
    for cid in BLENDED:
        rows = cdf[cdf["catalog_id"].astype(str).str.strip() == cid]
        print(f"  blended comp {cid} still in comp csv: {len(rows)} rows")

    print("\n=== Restore gs11_dilution_enabled=False ===")
    _set_gs11_enabled(False)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
