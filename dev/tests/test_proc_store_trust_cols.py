"""G2-F002b: ProcFrameStore must preserve catalog_match_mode (production-path regression)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from catalog_match_trust import (
    NONDET_CATALOG_MATCH_MODES,
    TRUSTED_CATALOG_MATCH_MODES,
    UNTRUSTED_FLUX_CATALOG_MATCH_MODES,
    normalize_catalog_match_mode,
)
from proc_frame_store import PROC_STORE_COLS, ProcFrameStore

_KNOWN_EXPORT_MODES = (
    TRUSTED_CATALOG_MATCH_MODES
    | UNTRUSTED_FLUX_CATALOG_MATCH_MODES
    | NONDET_CATALOG_MATCH_MODES
)


def _synthetic_proc_frame(path: Path, *, mode: str) -> None:
    pd.DataFrame(
        {
            "catalog_id": ["111", "222"],
            "name": ["111", "222"],
            "bjd_tdb_mid": [2459000.1, 2459000.2],
            "hjd_mid": [2459000.0, 2459000.1],
            "jd_mid": [2459000.0, 2459000.1],
            "dao_flux": [1000.0, 500.0],
            "airmass": [1.2, 1.2],
            "catalog_match_mode": [mode, mode],
        }
    ).to_csv(path, index=False)


def _frame_catalog_match_mode_from_df(df: pd.DataFrame) -> str:
    """Mirror run_phase2a frame_time_lookup catalog_match_mode extraction."""
    if df is None or df.empty or "catalog_match_mode" not in df.columns:
        return ""
    _cmm_s = df["catalog_match_mode"].dropna()
    if len(_cmm_s) == 0:
        return ""
    return normalize_catalog_match_mode(str(_cmm_s.iloc[0]))


_PSF_LC_STORE_COLS = (
    "psf_flux",
    "psf_flux_err",
    "psf_fit_ok",
    "psf_quality",
    "psf_quality_fallback",
    "psf_snr",
    "psf_ac_factor",
    "psf_ac_n_used",
    "psf_ac_applied",
)


def _synthetic_psf_proc_frame(path: Path) -> None:
    pd.DataFrame(
        {
            "catalog_id": ["111"],
            "name": ["111"],
            "bjd_tdb_mid": [2459000.1],
            "dao_flux": [1000.0],
            "psf_flux": [950.0],
            "psf_flux_err": [12.5],
            "psf_fit_ok": [True],
            "psf_quality": ["good"],
            "psf_quality_fallback": [False],
            "psf_snr": [18.0],
            "psf_ac_factor": [1.02],
            "psf_ac_n_used": [6],
            "psf_ac_applied": [True],
        }
    ).to_csv(path, index=False)


def test_proc_store_cols_includes_psf_lc_columns() -> None:
    for col in _PSF_LC_STORE_COLS:
        assert col in PROC_STORE_COLS


def test_phase2a_store_path_preserves_psf_columns() -> None:
    """Entry-path equivalence for PSF LC columns (catalog_match_mode lesson)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir)
        csv_path = p / "proc_test.csv"
        _synthetic_psf_proc_frame(csv_path)

        store = ProcFrameStore.build(p)
        store_df = store.get(str(csv_path))
        direct_df = pd.read_csv(
            csv_path,
            usecols=lambda c: c in set(PROC_STORE_COLS),
            low_memory=False,
        )
        assert store_df is not None
        for col in _PSF_LC_STORE_COLS:
            assert col in store_df.columns, col
            assert col in direct_df.columns, col
            assert store_df[col].iloc[0] == direct_df[col].iloc[0], col


def test_proc_store_cols_includes_catalog_match_mode() -> None:
    assert "catalog_match_mode" in PROC_STORE_COLS


def test_store_projection_preserves_catalog_match_mode() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir)
        _synthetic_proc_frame(p / "proc_frame001.csv", mode="master_reference_sky")
        store = ProcFrameStore.build(p)
        key = str(p / "proc_frame001.csv")
        df = store.get_frame(key)
        assert df is not None
        assert "catalog_match_mode" in df.columns
        assert df["catalog_match_mode"].iloc[0] == "master_reference_sky"

        df_proj = store.get_frame(key, cols=["catalog_id", "catalog_match_mode"])
        assert df_proj is not None
        assert list(df_proj.columns) == ["catalog_id", "catalog_match_mode"]


def test_phase2a_store_path_matches_direct_csv_path() -> None:
    """Entry-path equivalence: ProcFrameStore vs direct read on same proc CSV."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir)
        csv_path = p / "proc_BO_CVn_Light_009.csv"
        _synthetic_proc_frame(csv_path, mode="master_reference_sky")

        store = ProcFrameStore.build(p)
        store_df = store.get(str(csv_path))

        direct_df = pd.read_csv(
            csv_path,
            usecols=lambda c: c in set(PROC_STORE_COLS),
            low_memory=False,
        )

        mode_store = _frame_catalog_match_mode_from_df(store_df)
        mode_direct = _frame_catalog_match_mode_from_df(direct_df)
        assert mode_store == mode_direct == "master_reference_sky"


def test_exported_catalog_match_modes_are_known_sets() -> None:
    samples = [
        "master_reference_sky",
        "master_reference_pixel",
        "nondet_no_wcs",
        "nondet_unaligned_sky",
        "full_cone",
    ]
    for mode in samples:
        assert normalize_catalog_match_mode(mode) in _KNOWN_EXPORT_MODES
