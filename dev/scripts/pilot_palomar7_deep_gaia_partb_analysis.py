#!/usr/bin/env python3
"""Finish Part B analysis only (after deep-Gaia pipeline already ran)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  (repo layout: src_py + dev on sys.path)
_ROOT = _bootstrap.REPO_ROOT
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import AppConfig  # noqa: E402
from database import VyvarDatabase  # noqa: E402
from psf_photometry import get_epsf_fwhm_from_context  # noqa: E402

_pal7_spec = importlib.util.spec_from_file_location(
    "pal7_ab", _ROOT / "scripts" / "pilot_palomar7_deep_gaia_ab.py"
)
_pal7 = importlib.util.module_from_spec(_pal7_spec)
assert _pal7_spec.loader is not None
_pal7_spec.loader.exec_module(_pal7)
_collect_match_stats = _pal7._collect_match_stats
_extended_part_b = _pal7._extended_part_b

DRAFT_ID = 364
SETUP = "Luminance_180_2"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cfg = AppConfig()
    db = VyvarDatabase(cfg.database_path)
    draft_dir = Path(cfg.archive_root) / "Drafts" / f"draft_{DRAFT_ID:06d}"
    ps = draft_dir / "platesolve" / SETUP
    aligned = draft_dir / "detrended_aligned" / "lights" / SETUP
    ms_csv = ps / "masterstars_full_match.csv"
    ms_fits = ps / "MASTERSTAR.fits"

    out: dict = {"match_stats": _collect_match_stats(draft_dir, SETUP)}
    meta_json = ps / "masterstar_epsf_meta.json"
    if meta_json.is_file():
        out["epsf_meta"] = json.loads(meta_json.read_text(encoding="utf-8"))
    out["epsf_path"] = str(ps / "masterstar_epsf.fits")
    out["fwhm_px"] = float(get_epsf_fwhm_from_context(ms_fits, db, DRAFT_ID))

    diag_path = _ROOT / "scripts" / "diagnose_psf_elongation_362.py"
    spec = importlib.util.spec_from_file_location("diag", diag_path)
    diag = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(diag)

    part_b_df, note, star_df = _extended_part_b(
        aligned, ms_csv, ps, fwhm_px=out["fwhm_px"], draft_dir=draft_dir, diag=diag
    )
    out["part_b_note"] = note
    diag_dir = draft_dir / "diagnostics" / "psf_aperture_pal7_deep"
    diag_dir.mkdir(parents=True, exist_ok=True)
    mag_csv = diag_dir / "d364_aperture_vs_psf_by_mag.csv"
    crowd_csv = diag_dir / "d364_aperture_vs_psf_crowding.csv"
    stars_csv = diag_dir / "d364_aperture_vs_psf_per_star.csv"
    if not part_b_df.empty:
        part_b_df.to_csv(mag_csv, index=False)
        out["mag_bin_csv"] = str(mag_csv)
    if not star_df.empty:
        star_df.to_csv(stars_csv, index=False)
        out["per_star_csv"] = str(stars_csv)
    crowd_df = star_df.groupby(["mag_bin", "crowding_class"], dropna=False).agg(
        N=("catalog_id", "count"),
        median_rms_aperture=("rms_aperture", "median"),
        median_rms_psf=("rms_psf", "median"),
        median_ratio_psf_aper=("ratio_psf_aper", "median"),
    ).reset_index()
    if not crowd_df.empty:
        crowd_df.to_csv(crowd_csv, index=False)
        out["crowding_csv"] = str(crowd_csv)
    out["mag_bin_table"] = part_b_df.to_dict(orient="records")
    out["crowding_table"] = crowd_df.to_dict(orient="records")
    out["n_stars_analyzed"] = int(len(star_df))

    result_path = _ROOT / "tmp" / "pilot_palomar7_deep_gaia_ab_partb_analysis.json"
    result_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
