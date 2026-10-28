"""E2E pre-calibrated per-frame proc CSV naming (Fix A regression).

Fails when export writes ``<fits_basename>.csv`` instead of ``proc_<stem>.csv``.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from proc_frame_store import (
    PROC_CSV_GLOB,
    ProcFrameStore,
    list_proc_csvs,
    proc_csv_path_for_aligned_fits,
)

_REPO = Path(__file__).resolve().parents[2]
_DRAFT381 = _REPO / "Archive" / "Drafts" / "draft_000381"
_SETUP = "V_20_2"
_CHI_H_GLOB = "Chi_H*.fits"


def test_proc_csv_path_for_aligned_fits_idempotent() -> None:
    raw = Path("aligned/proc_V842_Her_Light_066.fits")
    pre = Path("aligned/Chi_H_2025-08-10_21-54-13_V_0027.fits")
    assert proc_csv_path_for_aligned_fits(raw).name == "proc_V842_Her_Light_066.csv"
    assert proc_csv_path_for_aligned_fits(pre).name == "proc_Chi_H_2025-08-10_21-54-13_V_0027.csv"
    # Raw path unchanged vs legacy with_suffix behaviour
    assert proc_csv_path_for_aligned_fits(raw) == raw.with_suffix(".csv")


@pytest.mark.slow
def test_pre_cal_export_writes_proc_csv_glob() -> None:
    """Chi_H_* aligned FITS -> export (defer) -> proc_*.csv -> ProcFrameStore."""
    src_aligned = _DRAFT381 / "detrended_aligned" / "lights" / _SETUP
    src_ps = _DRAFT381 / "platesolve" / _SETUP
    src_phot = src_ps / "photometry"
    if not src_aligned.is_dir() or not (src_ps / "MASTERSTAR.fits").is_file():
        pytest.skip("draft_000381 V_20_2 fixture not on disk")

    fits_src = sorted(src_aligned.glob(_CHI_H_GLOB))[:3]
    if len(fits_src) < 3:
        pytest.skip("need >=3 Chi_H aligned FITS in draft_000381")

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        aligned = root / "aligned"
        aligned.mkdir()
        ps = root / "platesolve"
        ps.mkdir()
        phot = ps / "photometry"
        phot.mkdir()

        for fp in fits_src:
            shutil.copy2(fp, aligned / fp.name)
        shutil.copy2(src_ps / "MASTERSTAR.fits", ps / "MASTERSTAR.fits")
        shutil.copy2(src_ps / "masterstars_full_match.csv", ps / "masterstars_full_match.csv")
        shutil.copy2(src_phot / "comparison_stars_per_target.csv", phot / "comparison_stars_per_target.csv")

        os.environ["VYVAR_PARALLEL_WORKERS"] = "1"
        from config import AppConfig
        from pipeline import export_per_frame_catalogs

        cfg = AppConfig()
        field_db = _REPO / "catalogs" / "vyvar_gaia_dr3_chiandh_field.db"
        if field_db.is_file():
            cfg.gaia_db_path = str(field_db.resolve())

        per = export_per_frame_catalogs(
            frames_root=aligned,
            platesolve_dir=ps,
            dao_fwhm_px=3.5,
            catalog_match_max_sep_arcsec=3.0,
            masterstars_csv=ps / "masterstars_full_match.csv",
            masterstar_fits=ps / "MASTERSTAR.fits",
            defer_disk_writes=True,
            app_config=cfg,
            draft_id=381,
        )

        deferred = per.get("deferred_csv_writes", [])
        assert deferred, "export must produce deferred CSV writes"
        for pcsv, _df in deferred:
            assert Path(pcsv).name.startswith("proc_"), f"non-canonical CSV path: {pcsv}"
        for row in per.get("frames", []):
            csv_ref = str(row.get("csv") or "")
            if csv_ref:
                assert Path(csv_ref).name.startswith("proc_"), csv_ref

        for pcsv, df in deferred:
            df.to_csv(pcsv, index=False)

        legacy_sidecars = [p for p in aligned.glob("*.csv") if not p.name.startswith("proc_")]
        assert not legacy_sidecars, f"legacy sidecar names must not be written: {legacy_sidecars}"

        proc_csvs = list_proc_csvs(aligned)
        assert proc_csvs, "list_proc_csvs must find proc_*.csv"
        assert all(p.name.startswith("proc_") for p in proc_csvs)

        store = ProcFrameStore.build(aligned, glob_pattern=PROC_CSV_GLOB)
        assert len(store) > 0

