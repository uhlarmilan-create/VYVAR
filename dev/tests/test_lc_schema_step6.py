"""OUTPUT-REVIEW-2026-10 step 6 (Milan 2026-10-21): light curve, check star, small CSVs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[2] / "src_py"
sys.path.insert(0, str(SRC))

import photometry_core  # noqa: F401,E402  (import order)
from check_star_kmag import (  # noqa: E402
    check_star_block,
    load_check_star,
    save_check_kmag_sidecar,
)
from photometry_lightcurve import _select_export_lc_rows  # noqa: E402
from frame_export import PER_FRAME_INDEX_COLUMNS, write_per_frame_index  # noqa: E402
from photometry_lightcurve import published_mag_column, save_lightcurve_csv  # noqa: E402
from photometry_report import _publication_lc_mag_column  # noqa: E402

LC_BASE = [
    "bjd", "hjd", "jd", "time_base", "airmass", "mag_inst", "mag_calib", "delta_mag", "err",
    "err_photon", "err_sem_rel", "err_scint_rel", "err_sigma_sys_rel", "aperture_r_px",
    "flag", "flag_reason", "source_file",
]


def _write(tmp_path: Path, **kw) -> pd.DataFrame:
    n = 3
    z = np.zeros(n)
    p = tmp_path / "lightcurve_1.csv"
    save_lightcurve_csv(
        p, bjd=z + 2460000.5, hjd=z + 2460000.4, jd=z + 2460000.3, airmass=z + 1.2, mag_inst=z - 9.0,
        mag_calib=z + 13.0, delta_mag=z + 0.5, err=z + 0.01, aperture_r_px=z + 5.0,
        flags=["normal"] * n, source_files=["f"] * n, err_photon=z + 0.009, err_sem_rel=z,
        err_scint_rel=z, err_sigma_sys_rel=z, **kw,
    )
    return pd.read_csv(p)


def test_lc_columns_without_colour_term(tmp_path: Path) -> None:
    df = _write(tmp_path)
    assert list(df.columns) == LC_BASE
    assert published_mag_column(df) == "mag_calib"


def test_lc_columns_with_colour_term_and_gspc(tmp_path: Path) -> None:
    df = _write(tmp_path, mag_calib_ct=np.full(3, 13.02), mag_gspc=np.full(3, 15.5))
    assert list(df.columns[:9]) == ["bjd", "hjd", "jd", "time_base", "airmass", "mag_inst", "mag_calib",
                                    "mag_calib_ct", "mag_gspc"]
    assert published_mag_column(df) == "mag_calib_ct"
    assert _select_export_lc_rows(df)["mag_calib"].tolist() == [13.02] * 3


def test_lc_k2_term_column(tmp_path: Path) -> None:
    """EPSF-K2-2026-10: the applied k'' term per epoch (k'' x colour difference x airmass)."""
    df = _write(tmp_path, k2_coeff=-0.016 * 1.5)
    assert list(df.columns) == LC_BASE[:13] + ["k2_delta_mag"] + LC_BASE[13:]
    assert np.allclose(df["k2_delta_mag"], -0.016 * 1.5 * 1.2)


def test_removed_columns_not_written(tmp_path: Path) -> None:
    df = _write(tmp_path, mag_calib_ct=np.full(3, 13.0))
    for col in ("mag_calib_raw", "mag_calib_final", "ct_correction", "ct_ok", "k2_source", "k2_value",
                "lunar_risk", "sigma_sys_mag", "aperture_f", "fwhm_night_median_px", "method",
                "lc_flux_method", "skip_reason", "is_flipped", "wcs_untrusted", "catalog_match_mode",
                "alignment_failed", "err_scatter_unmatched", "dilution_factor",
                "delta_mag_sysrem", "mag_calib_ac"):
        assert col not in df.columns


def test_published_column_of_old_drafts() -> None:
    old = pd.DataFrame({"bjd": [1.0], "mag_calib": [12.5], "mag_calib_ct": [12.4],
                        "mag_calib_final": [12.4], "flag": ["normal"]})
    assert published_mag_column(old) == "mag_calib_ct"
    assert _publication_lc_mag_column(old.columns) == "mag_calib_ct"
    assert _select_export_lc_rows(old)["mag_calib"].tolist() == [12.4]
    legacy = pd.DataFrame({"bjd": [1.0], "mag_calib": [12.5], "flag": ["normal"]})
    assert _select_export_lc_rows(legacy)["mag_calib"].tolist() == [12.5]


class _Ens:
    check_sparse = True
    trust_R = 1.25
    trust_R_lo = 0.9
    trust_R_hi = 1.7
    comp_stability_p = 0.03
    x2_pair_mag2 = 2.5e-5
    triangulation_clipped = False
    sparse_flags = ["R_hi_exceeds_T_green"]
    k_source = ""
    k_colour_offset = float("nan")
    trust_R_detrend = float("nan")
    trust_R_detrend_lo = float("nan")
    trust_R_detrend_hi = float("nan")


def test_check_star_block_in_comp_quality(tmp_path: Path) -> None:
    lc_dir = tmp_path / "lightcurves"
    lc_dir.mkdir()
    save_check_kmag_sidecar(lc_dir / "check_kmag_7.csv", bjd=np.array([1.0, 2.0]), source_files=["a", "b"],
                            kmag=np.array([12.0, 12.01]), kmag_err=np.array([0.01, 0.01]))
    assert list(pd.read_csv(lc_dir / "check_kmag_7.csv").columns) == ["bjd", "source_file", "kmag", "kmag_err"]
    (lc_dir / "comp_quality_7.json").write_text(json.dumps({"9": "good", "check_star": check_star_block("123", _Ens())}))
    blk = load_check_star(lc_dir, "7")
    assert blk["catalog_id"] == "123" and blk["check_sparse"] is True and blk["sparse_flags"] == ["R_hi_exceeds_T_green"]
    from trust_flag_core import read_sparse_trust_sidecar

    tr = read_sparse_trust_sidecar(tmp_path, "7")
    assert tr["trust_R_hi"] == 1.7 and tr["check_sparse"] and tr["sparse_flags"] == ("R_hi_exceeds_T_green",)
    from comp_table import parse_comp_quality_json_map

    assert set(parse_comp_quality_json_map(json.loads((lc_dir / "comp_quality_7.json").read_text()))) == {"9"}


def test_check_star_of_old_drafts(tmp_path: Path) -> None:
    lc_dir = tmp_path / "lightcurves"
    lc_dir.mkdir()
    pd.DataFrame({"check_catalog_id": ["2172318699396962560"] * 2, "bjd": [1.0, 2.0], "source_file": ["a", "b"],
                  "kmag": [12.0, 12.1], "check_sparse": [1, 1], "trust_R": [2.3, 2.3], "trust_R_lo": [1.8, 1.8],
                  "trust_R_hi": [3.1, 3.1], "comp_stability_p": [0.1, 0.1], "x2_pair_mag2": [5e-5, 5e-5],
                  "triangulation_clipped": [0, 0], "sparse_flags": ["a;b", "a;b"]}).to_csv(
        lc_dir / "check_kmag_7.csv", index=False)
    blk = load_check_star(lc_dir, "7")
    assert blk["catalog_id"] == "2172318699396962560" and blk["trust_R_hi"] == 3.1 and blk["sparse_flags"] == ["a", "b"]


def test_per_frame_index_columns(tmp_path: Path) -> None:
    rows = [{"file": "a.fits", "status": "ok", "csv": "proc_a.csv", "n_detected": 10, "n_matched": 9,
             "raw_peaks_used": True, "debug_pixel_match": {"x": 1}, "frame_values": {}}]
    p = write_per_frame_index(rows, tmp_path / "per_frame_catalog_index.csv")
    assert tuple(pd.read_csv(p).columns) == PER_FRAME_INDEX_COLUMNS


def test_qc_metrics_not_written_columns() -> None:
    from pipeline_calibrate import QC_METRICS_NOT_WRITTEN

    assert set(QC_METRICS_NOT_WRITTEN) == {"n_sources", "cal_pstbg", "sky_surface_fwhm_px", "sky_surface_source_mask_px"}


def test_report_fwhm_reads_fwhm_px() -> None:
    s = (SRC / "photometry_report.py").read_text(encoding="utf-8")
    assert 'dfq.get("self.fwhm_px")' not in s and 'dfq.get("fwhm_px")' in s


def test_invariants_read_the_photometry_active_targets(tmp_path: Path) -> None:
    import invariants_runtime as ir

    ps = tmp_path / "platesolve" / "g_120_4"
    phot = ps / "photometry"
    phot.mkdir(parents=True)
    pd.DataFrame({"vsx_name": ["X"], "catalog_id": ["1"]}).to_csv(phot / "active_targets.csv", index=False)
    pd.DataFrame({"vsx_name": ["X"], "catalog_id": ["2"]}).to_csv(ps / "variable_targets.csv", index=False)
    meta: dict = {"provenance": {"config_snapshot": {"vsx_out_of_scope_types": ["ELL"]}}}
    try:
        ir.validate_config_behavior(meta, phot)
    except ir.InvariantViolation as exc:
        assert "INV-TARGET-ID" in str(exc)
    else:
        raise AssertionError("INV-TARGET-ID must fail on a catalog_id mismatch")
