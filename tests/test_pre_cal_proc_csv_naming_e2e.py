"""E2E pre-calibrated per-frame proc CSV naming (Fix A regression).

Fails when export writes ``<fits_basename>.csv`` instead of ``proc_<stem>.csv``.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pandas as pd
import pytest

from comp_qa_core import compute_comp_qa
from proc_frame_store import (
    PROC_CSV_GLOB,
    ProcFrameStore,
    list_proc_csvs,
    proc_csv_path_for_aligned_fits,
)

_REPO = Path(__file__).resolve().parents[1]
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
def test_pre_cal_export_writes_proc_csv_glob_and_comp_qa_n_clean() -> None:
    """Chi_H_* aligned FITS -> export (defer) -> proc_*.csv -> ProcFrameStore + comp_qa."""
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
        field_db = _REPO / "GAIA_DR3" / "vyvar_gaia_dr3_chiandh_field.db"
        if field_db.is_file():
            cfg.gaia_db_path = str(field_db.resolve())

        per = export_per_frame_catalogs(
            frames_root=aligned,
            platesolve_dir=ps,
            dao_threshold_sigma=3.5,
            dao_fwhm_px=3.5,
            max_catalog_rows=20000,
            catalog_match_max_sep_arcsec=3.0,
            masterstars_csv=ps / "masterstars_full_match.csv",
            masterstar_fits=ps / "MASTERSTAR.fits",
            use_master_fast_path=True,
            defer_disk_writes=True,
            app_config=cfg,
            plate_solve_fov_deg=1.25,
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

        # Build a minimal comparison table from catalog_ids present in all exported frames.
        from comp_qa_core import load_proc_pivot

        id_sets: list[set[str]] = []
        for pcsv, df in deferred:
            if "catalog_id" in df.columns:
                id_sets.append(
                    set(df["catalog_id"].astype(str).str.strip().replace({"": pd.NA}).dropna())
                )
        common = set.intersection(*id_sets) if id_sets else set()
        assert len(common) >= 4, "need target + comps shared across exported frames"
        picked = sorted(common)[:4]
        target_id, *comp_ids = picked
        mini_comp = pd.DataFrame(
            {
                "target_catalog_id": [target_id] * len(comp_ids),
                "catalog_id": comp_ids,
                "target_vsx_name": ["e2e_target"] * len(comp_ids),
            }
        )
        mini_comp.to_csv(phot / "comparison_stars_per_target.csv", index=False)

        flux_w, _t = load_proc_pivot(aligned, {target_id, *comp_ids})
        assert not flux_w.empty
        assert target_id in flux_w.columns

        result = compute_comp_qa(photometry_dir=phot, proc_dir=aligned, min_comps=3, max_comps=8)
        per_target = result.get("per_target") or {}
        assert per_target, "comp_qa must process at least one target"
        n_clean_vals = [int(v["n_clean"]) for v in per_target.values()]
        assert n_clean_vals
        assert not all(pd.isna(nc) for nc in n_clean_vals)
