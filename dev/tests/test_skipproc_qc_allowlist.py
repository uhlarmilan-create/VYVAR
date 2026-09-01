"""Preprocess QC allowlist (design C+A) and QC-01 gate tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from config import AppConfig
from invariants_runtime import InvariantViolation, check_qc01_skipproc_alignment
from pipeline import (
    _qc_enrich_calibrated_in_place,
    build_prefilter_rejected_map,
    filter_files_by_qc_metrics_allowlist,
    load_qc_metrics_status_by_path,
    norm_fits_path_key,
    qc_enrich_calibrated_lights_in_place,
)


def _write_light_fits(path: Path, *, seed: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(seed * 1000) % 2**31)
    data = rng.normal(loc=1000.0, scale=20.0, size=(32, 32)).astype(np.float32)
    fits.PrimaryHDU(data).writeto(path, overwrite=True)


def _write_qc_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_allowlist_ok_included_rejected_excluded(tmp_path: Path) -> None:
    root = tmp_path / "calibrated" / "lights" / "setup"
    ok_fp = root / "ok.fits"
    bad_fp = root / "bad.fits"
    ghost_fp = root / "ghost.fits"
    _write_light_fits(ok_fp)
    _write_light_fits(bad_fp, seed=2.0)
    _write_light_fits(ghost_fp, seed=3.0)
    qc = tmp_path / "calibrated" / "lights" / "qc_metrics.csv"
    _write_qc_csv(
        qc,
        [
            {"src": str(ok_fp.resolve()), "dst": str(ok_fp.resolve()), "status": "ok"},
            {
                "src": str(bad_fp.resolve()),
                "dst": str(bad_fp.resolve()),
                "status": "rejected_prefilter_fwhm",
            },
        ],
    )
    selected, _ = filter_files_by_qc_metrics_allowlist([ok_fp, bad_fp, ghost_fp], qc)
    assert [p.name for p in selected] == ["ok.fits"]


def test_allowlist_missing_csv_raises(tmp_path: Path) -> None:
    qc = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError, match="qc_metrics.csv not found"):
        load_qc_metrics_status_by_path(qc)


def test_align_requires_qc_csv_message(tmp_path: Path) -> None:
    from pipeline import find_qc_metrics_csv

    ap = tmp_path / "draft"
    (ap / "calibrated" / "lights").mkdir(parents=True)
    qc = find_qc_metrics_csv(ap, app_config=AppConfig())
    assert qc is None
    with pytest.raises(FileNotFoundError, match="Preprocess QC step required"):
        if qc is None or not qc.is_file():
            raise FileNotFoundError(
                "Preprocess QC step required; run Analyze/preprocess first to produce qc_metrics.csv"
            )


def test_full_set_visitation_prefilter_stamping(tmp_path: Path) -> None:
    root = tmp_path / "calibrated" / "lights" / "grp"
    keep = root / "keep.fits"
    drop = root / "drop.fits"
    _write_light_fits(keep, seed=1.0)
    _write_light_fits(drop, seed=2.0)
    pre = build_prefilter_rejected_map([keep, drop], [keep])
    cfg = AppConfig()
    out = _qc_enrich_calibrated_in_place(
        tmp_path / "calibrated" / "lights",
        app_config=cfg,
        prefilter_rejected=pre,
    )
    results = out.get("results") or []
    assert len(results) == 2
    by_name = {Path(r["src"]).name: r["status"] for r in results}
    assert by_name["keep.fits"] == "ok"
    assert by_name["drop.fits"] == "rejected_prefilter_fwhm"
    with fits.open(drop) as hdul:
        assert str(hdul[0].header.get("VY_QC", "")).strip() == "rejected_prefilter_fwhm"
        assert hdul[0].header.get("VYVARPR") is True
    qc_csv = tmp_path / "calibrated" / "lights" / "qc_metrics.csv"
    assert qc_csv.is_file()
    df = pd.read_csv(qc_csv)
    assert len(df) == 2


def test_skip_mode_never_emits_segmentation_reject_statuses(tmp_path: Path) -> None:
    root = tmp_path / "calibrated" / "lights"
    fp = root / "one.fits"
    _write_light_fits(fp)
    cfg = AppConfig()
    cfg.qc_fwhm_limit = 0.5
    out = _qc_enrich_calibrated_in_place(root, app_config=cfg, fwhm_reject_limit=0.5)
    statuses = {str(r.get("status")) for r in out.get("results") or []}
    assert "rejected_fwhm" not in statuses
    assert "rejected_elong" not in statuses


def test_qc01_gate_fail_on_non_ok_selection(tmp_path: Path) -> None:
    root = tmp_path / "lights"
    ok_fp = root / "a.fits"
    bad_fp = root / "b.fits"
    _write_light_fits(ok_fp)
    _write_light_fits(bad_fp, seed=2.0)
    qc = root / "qc_metrics.csv"
    _write_qc_csv(
        qc,
        [
            {"src": str(ok_fp.resolve()), "dst": str(ok_fp.resolve()), "status": "ok"},
            {
                "src": str(bad_fp.resolve()),
                "dst": str(bad_fp.resolve()),
                "status": "rejected_prefilter_fwhm",
            },
        ],
    )
    check_qc01_skipproc_alignment([ok_fp], qc)
    with pytest.raises(InvariantViolation, match="QC-01"):
        check_qc01_skipproc_alignment([ok_fp, bad_fp], qc)


def test_norm_fits_path_key_casefold(tmp_path: Path) -> None:
    p = tmp_path / "Light.FITS"
    p.write_bytes(b"x")
    k1 = norm_fits_path_key(p)
    k2 = norm_fits_path_key(str(p).upper())
    assert k1 == k2


def test_preprocess_creates_no_processed_directory(tmp_path: Path) -> None:
    lights = tmp_path / "calibrated" / "lights" / "NoFilter_60_2"
    fp = lights / "Light_001.fits"
    _write_light_fits(fp)
    qc_enrich_calibrated_lights_in_place(
        calibrated_root=tmp_path / "calibrated" / "lights",
        app_config=AppConfig(),
    )
    assert not (tmp_path / "processed").exists()
    assert (tmp_path / "calibrated" / "lights" / "qc_metrics.csv").is_file()


def test_known_removed_skip_processed_directory_logs_info(tmp_path: Path, caplog) -> None:
    import json
    import logging

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({"skip_processed_directory": True}), encoding="utf-8")
    caplog.set_level(logging.INFO)
    from config import AppConfig

    AppConfig(project_root=tmp_path)
    assert any("skip_processed_directory removed 2026-07" in r.message for r in caplog.records)


def test_known_removed_global_comp_pool_enabled_logs_info(tmp_path: Path, caplog) -> None:
    import json
    import logging

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({"global_comp_pool_enabled": False}), encoding="utf-8")
    caplog.set_level(logging.INFO)
    from config import AppConfig

    cfg = AppConfig(project_root=tmp_path)
    assert not hasattr(cfg, "global_comp_pool_enabled")
    assert any("global_comp_pool_enabled removed 2026-09-01" in r.message for r in caplog.records)
    assert not any(r.levelno >= logging.WARNING for r in caplog.records)


def test_known_removed_export_err_mode_logs_info(tmp_path: Path, caplog) -> None:
    import json
    import logging

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({"export_err_mode": "model"}), encoding="utf-8")
    caplog.set_level(logging.INFO)
    from config import AppConfig

    cfg = AppConfig(project_root=tmp_path)
    assert not hasattr(cfg, "export_err_mode")
    assert any("export_err_mode removed 2026-09-01" in r.message for r in caplog.records)
    assert not any(r.levelno >= logging.WARNING for r in caplog.records)


def test_known_removed_err_background_mode_logs_info(tmp_path: Path, caplog) -> None:
    import json
    import logging

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({"err_background_mode": "howell"}), encoding="utf-8")
    caplog.set_level(logging.INFO)
    from config import AppConfig

    assert not any(r.levelno >= logging.WARNING for r in caplog.records)


def test_known_removed_masterstar_accept_mode_logs_info(tmp_path: Path, caplog) -> None:
    import json
    import logging

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({"masterstar_accept_mode": "fraction"}), encoding="utf-8")
    caplog.set_level(logging.INFO)
    from config import AppConfig

    assert not any(r.levelno >= logging.WARNING for r in caplog.records)


def test_known_removed_psf_ac_policy_logs_info(tmp_path: Path, caplog) -> None:
    import json
    import logging

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({"psf_ac_policy": "chi2_lt5_legacy"}), encoding="utf-8")
    caplog.set_level(logging.INFO)
    from config import AppConfig

    cfg = AppConfig(project_root=tmp_path)
    assert not hasattr(cfg, "psf_ac_policy")
    assert any("psf_ac_policy removed 2026-09-01" in r.message for r in caplog.records)
    assert not any(r.levelno >= logging.WARNING for r in caplog.records)
