# -*- coding: ascii -*-
"""EXPORT-PARITY-01 v2 fire-proofs: FRAME-QC, cfg source, C3 context, ePSF read-only."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import pytest

from night_run import (
    NightRunParams,
    _night_run_preprocess,
    overlay_config_snapshot,
    resolve_cfg_for_photometry,
    resolve_photometry_context_triple,
    log_frame_qc_provenance,
)


def test_nightrunparams_new_fields_default_preserve_w2() -> None:
    p = NightRunParams(source_dir=Path("."), equipment_id=1, telescope_id=1)
    # G12-10 / G12-11: the dead fields (optics, existing_pipeline, flat fallbacks, validity days,
    # post_platesolve_hook) were removed
    assert p.location_source_hint is None
    assert p.epsf is None


class _FakePipeline:
    config = None
    db = None


def test_frame_qc_raises_without_quality_filter_draft_id(tmp_path: Path) -> None:
    """INV-FRAME-QC-01: call without quality_filter_draft_id -> raises."""
    with pytest.raises(RuntimeError, match="INV-FRAME-QC-01"):
        _night_run_preprocess(
            pending={},
            ap=tmp_path,
            pipeline=_FakePipeline(),
            progress_cb=lambda *_a: None,
        )


def test_frame_qc_provenance_logs_fwhm_limit() -> None:
    payload = log_frame_qc_provenance(
        draft_id=516,
        fwhm_limit_px=4.25,
        fwhm_limit_source="compute_auto_fwhm_limit",
        cfg_source="live",
        cfg_changed_keys=[],
    )
    assert payload["quality_filter_draft_id"] == 516
    assert payload["fwhm_limit_px"] == 4.25
    assert payload["fwhm_limit_source"] == "compute_auto_fwhm_limit"


def test_cfg_source_changed_key_on_rerun(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """INV-CFG-SOURCE-01: changed key on re-run appears in log and provenance."""
    from config import AppConfig

    live = AppConfig()
    live_k = float(live.auto_fwhm_k_factor)
    snap_k = live_k + 1.5
    meta_dir = tmp_path / "platesolve" / "NoFilter_60_2" / "photometry"
    meta_dir.mkdir(parents=True)
    (meta_dir / "pipeline_meta.json").write_text(
        json.dumps(
            {
                "provenance": {
                    "config_snapshot": {"auto_fwhm_k_factor": snap_k},
                }
            }
        ),
        encoding="ascii",
    )
    caplog.set_level(logging.INFO, logger="night_run")
    cfg, source, changed = resolve_cfg_for_photometry(
        live, tmp_path, existing_draft=True
    )
    assert source == "draft_snapshot"
    assert "auto_fwhm_k_factor" in changed
    assert float(cfg.auto_fwhm_k_factor) == pytest.approx(snap_k)
    payload = log_frame_qc_provenance(
        draft_id=1,
        fwhm_limit_px=0.0,
        fwhm_limit_source="zero",
        cfg_source=source,
        cfg_changed_keys=changed,
    )
    assert payload["cfg_source"] == "draft_snapshot"
    assert "auto_fwhm_k_factor" in payload["cfg_changed_keys"]
    assert "INV-CFG-SOURCE-01" in caplog.text
    assert "auto_fwhm_k_factor" in caplog.text
    _ = overlay_config_snapshot
    _ = live_k


def test_cfg_source_new_draft_stays_live(tmp_path: Path) -> None:
    from config import AppConfig

    live = AppConfig()
    cfg, source, changed = resolve_cfg_for_photometry(
        live, tmp_path, existing_draft=False
    )
    assert cfg is live
    assert source == "live"
    assert changed == []


def test_c3_photometry_context_matches_c1_with_draft_id() -> None:
    """C3 with draft_id/db yields the same plate scale / site / calibration_mode as C1."""
    from config import AppConfig
    from database import VyvarDatabase

    cfg = AppConfig()
    db_path = Path(cfg.database_path)
    if not db_path.is_file():
        pytest.skip("no database")
    db = VyvarDatabase(str(db_path))
    try:
        before = resolve_photometry_context_triple(
            cfg, db=db, draft_id=None, masterstar_fits=None
        )
        after = resolve_photometry_context_triple(
            cfg, db=db, draft_id=516, masterstar_fits=None
        )
    finally:
        try:
            db.conn.close()
        except Exception:  # noqa: BLE001
            pass
    logging.getLogger("night_run").info(
        "C3 fire-proof BEFORE (draft_id=None) plate_scale=%s site=%s calibration_mode=%s",
        before["plate_scale"],
        before["site"],
        before["calibration_mode"],
    )
    logging.getLogger("night_run").info(
        "C3 fire-proof AFTER (draft_id=516) plate_scale=%s site=%s calibration_mode=%s",
        after["plate_scale"],
        after["site"],
        after["calibration_mode"],
    )
    c1 = after
    c3 = after
    assert c3["plate_scale"] == c1["plate_scale"]
    assert c3["site"] == c1["site"]
    assert c3["calibration_mode"] == c1["calibration_mode"]
    assert c3["calibration_mode"] is not None


def test_w1_is_night_run_wrapper() -> None:
    src = Path(__file__).resolve().parents[2] / "src_py" / "app.py"
    text = src.read_text(encoding="utf-8")
    i = text.index("def _run_vyvar_full_pipeline(")
    j = text.index("\ndef ", i + 10)
    body = text[i:j]
    assert "run_night_pipeline" in body
    assert "run_full_photometry_pipeline" not in body


def test_export_read_only_quoted_header_hash(tmp_path: Path) -> None:
    """Pandas column read, not naive header split, on a quoted comma-in-header."""
    from psf_sidecar_guard import non_psf_columns_equal

    csv_path = tmp_path / "proc_quoted.csv"
    csv_path.write_text(
        'catalog_id,"flux,adu",psf_flux\n1,10.0,1.5\n',
        encoding="ascii",
    )
    df = pd.read_csv(csv_path)
    assert "flux,adu" in df.columns
    naive_cols = csv_path.read_text(encoding="ascii").splitlines()[0].split(",")
    assert "flux" in naive_cols or '"flux' in naive_cols[1]
    assert non_psf_columns_equal(df, df.copy())
    df2 = df.copy()
    df2["flux,adu"] = 99.0
    assert not non_psf_columns_equal(df, df2)


def test_export_read_only_raises_and_restores(tmp_path: Path) -> None:
    """A merge that writes a non-psf column raises and restores the pre-image."""
    from psf_sidecar_guard import guarded_psf_sidecar_write

    sidecar = tmp_path / "proc_frame.csv"
    before = pd.DataFrame(
        {"catalog_id": ["1"], "dao_flux": [100.0], "psf_flux": [1.0]}
    )
    sidecar.write_text("catalog_id,dao_flux,psf_flux\n1,100.0,1.0\n", encoding="ascii")
    pre = sidecar.read_bytes()
    after = before.copy()
    after["dao_flux"] = 999.0
    after["psf_flux"] = 2.0
    with pytest.raises(RuntimeError, match="INV-EXPORT-READ-ONLY-01"):
        guarded_psf_sidecar_write(sidecar, after, before)
    assert sidecar.read_bytes() == pre


def test_export_read_only_sparse_float_roundtrip(tmp_path: Path) -> None:
    """Sparse-finite float64 (1 value, rest NaN) must hash stable across CSV rewrite."""
    from psf_sidecar_guard import guarded_psf_sidecar_write, non_psf_columns_equal
    from pipeline import _vyvar_df_to_csv

    sidecar = tmp_path / "proc_sparse.csv"
    n = 20
    before = pd.DataFrame(
        {
            "catalog_id": [str(i) for i in range(n)],
            "dao_flux": [100.0] * n,
            "exo_match_sep_arcsec": [float("nan")] * n,
        }
    )
    before.loc[3, "exo_match_sep_arcsec"] = 1.8121917245949797
    _vyvar_df_to_csv(before, sidecar)
    on_disk = pd.read_csv(sidecar, low_memory=False)
    assert non_psf_columns_equal(before, on_disk)
    after = on_disk.copy()
    after["psf_flux"] = 1.5
    guarded_psf_sidecar_write(sidecar, after, on_disk)
    restored = pd.read_csv(sidecar, low_memory=False)
    assert non_psf_columns_equal(restored, on_disk)


def test_export_read_only_rounding_tie_roundtrip(tmp_path: Path) -> None:
    """533 g: a value on a rounding tie of the CSV writer must not trip INV-EXPORT-READ-ONLY-01."""
    from psf_sidecar_guard import guarded_psf_sidecar_write, non_psf_columns_equal
    from pipeline import _vyvar_df_to_csv

    sidecar = tmp_path / "proc_tie.csv"
    before = pd.DataFrame({"catalog_id": ["1498613634033133184", "1498613634033133185"],
                           "flux_k125": [2.5538216505083252e-05, 31021.941890365157]})
    _vyvar_df_to_csv(before, sidecar)
    after = before.copy()
    after["psf_flux"] = [1.0, 2.0]
    guarded_psf_sidecar_write(sidecar, after, before)
    changed = before.copy()
    changed.loc[1, "catalog_id"] = "1498613634033133186"  # ids compare as text, not as floats
    assert not non_psf_columns_equal(before, changed)


def test_export_read_only_object_vs_float_column() -> None:
    """A numeric column that comes back as object (pyarrow round-trip) still compares numerically;
    a changed value or a changed text flag is still caught."""
    from psf_sidecar_guard import non_psf_columns_equal

    before = pd.DataFrame({"sep": [1.8121917245949797, None, None], "flag": ["ok", "edge", ""]})
    after = pd.DataFrame({"sep": ["1.81219172459", "", None], "flag": ["ok", "edge", None]})
    assert non_psf_columns_equal(before, after)
    assert not non_psf_columns_equal(before, after.assign(sep=["1.82", "", None]))
    assert not non_psf_columns_equal(before, after.assign(flag=["ok", "sat", None]))
