from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

# Ensure repo root importable when run as a script.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline import export_per_frame_catalogs  # noqa: E402
from photometry_core import run_phase0_and_phase1, run_phase2a  # noqa: E402
from vyvar_alignment_frame import _alignment_compute_one_frame, _alignment_detect_xy  # noqa: E402


def main() -> int:
    draft_root = Path(r"C:\ASTRO\python\VYVAR\Archive\Drafts\draft_000244").resolve()
    setup = "NoFilter_60_2"

    processed_root = (draft_root / "processed" / "lights" / setup).resolve()
    aligned_root = (draft_root / "detrended_aligned" / "lights" / setup).resolve()
    platesolve_dir = (draft_root / "platesolve" / setup).resolve()

    if not processed_root.is_dir():
        raise FileNotFoundError(f"Missing processed_root: {processed_root}")
    platesolve_dir.mkdir(parents=True, exist_ok=True)
    aligned_root.mkdir(parents=True, exist_ok=True)

    # Inputs produced earlier (we reuse, do not rebuild MASTERSTAR).
    masterstar_fits = platesolve_dir / "MASTERSTAR.fits"
    masterstars_csv = platesolve_dir / "masterstars_full_match.csv"
    variable_targets_csv = platesolve_dir / "variable_targets.csv"

    if not masterstar_fits.is_file():
        raise FileNotFoundError(f"Missing {masterstar_fits}")
    if not masterstars_csv.is_file():
        raise FileNotFoundError(f"Missing {masterstars_csv}")
    if not variable_targets_csv.is_file():
        raise FileNotFoundError(f"Missing {variable_targets_csv}")

    files = sorted(processed_root.glob("proc_*.fits"))
    if not files:
        raise FileNotFoundError(f"No proc_*.fits in {processed_root}")

    # Align *to MASTERSTAR* so the output grid + WCS matches masterstars_full_match.csv,
    # enabling robust Gaia IDs in per-frame CSVs.
    ref_fp = masterstar_fits
    with fits.open(ref_fp, memmap=False) as hdul:
        ref_hdr = hdul[0].header.copy()
        ref_data = np.asarray(hdul[0].data, dtype=np.float32)

    ref_xy = _alignment_detect_xy(
        ref_data,
        want_max=200,
        det_sigma=3.5,
        fwhm_px=3.0,
        label=ref_fp.name,
        log_sink=None,
    )

    ctx = {
        "ref_data": ref_data,
        "ref_hdr": ref_hdr,
        "ref_fp_name": ref_fp.name,
        "fixed_target_pts": np.asarray(ref_xy, dtype=np.float32),
        "reference_list": np.asarray(ref_xy, dtype=np.float32).tolist(),
        "platesolve_dir": str(platesolve_dir),
        "align_star_cap": 200,
        "min_detected_stars": 100,
        "max_detected_stars": 500,
        "fb_align": 3.0,
        "rotation_ref_angle_deg": None,
        "has_ref_wcs": True,
    }

    # ── Alignment (force regenerate aligned FITS + report) ──
    star_counts: list[dict] = []
    t0 = time.time()
    for i, fp in enumerate(files, start=1):
        out = _alignment_compute_one_frame(fp, i, ctx, None)
        sc = out.get("star_count") or {}
        if out.get("kind") == "aligned":
            hdr = out["hdr"]
            data = out["aligned_data"]
            tgt = aligned_root / fp.name
            fits.writeto(tgt, np.asarray(data, dtype=np.float32), header=hdr, overwrite=True)
            star_counts.append(sc)
        else:
            # Keep report row anyway.
            star_counts.append(sc)

        if i == 1 or i == len(files) or (len(files) > 10 and i % max(1, len(files) // 10) == 0):
            print(f"[align] {i}/{len(files)} {fp.name} -> {out.get('kind')} {out.get('aligned_method')}")

    rep_path = platesolve_dir / "alignment_report.csv"
    pd.DataFrame(star_counts).to_csv(rep_path, index=False)
    print(f"[align] done: {len(files)} frames in {time.time()-t0:.1f}s report={rep_path}")

    # ── Per-frame catalogs (regenerate) ──
    t1 = time.time()
    per = export_per_frame_catalogs(
        frames_root=aligned_root,
        platesolve_dir=platesolve_dir,
        max_catalog_rows=12000,
        catalog_match_max_sep_arcsec=10.0,
        saturate_level_fraction=0.95,
        faintest_mag_limit=18.0,
        dao_threshold_sigma=3.5,
        masterstars_csv=masterstars_csv,
        masterstar_fits=masterstar_fits,
        use_master_fast_path=True,
        draft_id=None,
        equipment_id=None,
    )
    print(f"[csv] done: written={per.get('written')} in {time.time()-t1:.1f}s index={per.get('index_csv')}")

    # ── Phase 0 + 1 ──
    phase01_out = (platesolve_dir / "photometry" / "phase01").resolve()
    t2 = time.time()
    p01 = run_phase0_and_phase1(
        variable_targets_csv=variable_targets_csv,
        masterstars_csv=masterstars_csv,
        per_frame_csv_dir=aligned_root,
        output_dir=phase01_out,
        fwhm_px=3.3136,
        frame_w_px=2082,
        frame_h_px=1397,
    )
    print(f"[phase0+1] done in {time.time()-t2:.1f}s -> {phase01_out}")

    # ── Phase 2A ──
    active_targets_csv = Path(str(p01["active_targets_csv"]))
    comparison_stars_csv = Path(str(p01["comparison_stars_csv"]))
    phase2a_out = (platesolve_dir / "photometry" / "phase2a").resolve()
    t3 = time.time()
    p2a = run_phase2a(
        masterstar_fits_path=masterstar_fits,
        active_targets_csv=active_targets_csv,
        comparison_stars_csv=comparison_stars_csv,
        per_frame_csv_dir=aligned_root,
        detrended_aligned_dir=aligned_root,
        output_dir=phase2a_out,
        fwhm_px=3.3136,
    )
    print(f"[phase2a] done in {time.time()-t3:.1f}s -> {phase2a_out}")
    print(f"[phase2a] summary: {p2a}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

