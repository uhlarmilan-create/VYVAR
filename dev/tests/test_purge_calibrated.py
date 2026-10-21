# -*- coding: ascii -*-
"""OUTPUT-REVIEW-2026-10: calibrated light frames are deleted after a finished run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from astropy.io import fits

from draft_provenance import load_draft_manifest, purge_calibrated_light_frames


def _fits(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(np.zeros((4, 4), dtype=np.float32)).writeto(path)


def _draft(tmp_path: Path, *, raw: bool) -> Path:
    d = tmp_path / "draft_000001"
    if raw:
        _fits(d / "Raw" / "lights" / "NoFilter_60_2" / "Light_001.fits")
    _fits(d / "calibrated" / "lights" / "NoFilter_60_2" / "Light_001.fits")
    _fits(d / "calibrated" / "lights" / "NoFilter_60_2_R" / "Light_001.fit")
    (d / "calibrated" / "lights" / "qc_metrics.csv").write_text("file,fwhm_px\nLight_001.fits,3.0\n")
    (d / "draft_manifest.json").write_text(json.dumps({"draft_id": 1, "calibration_mode": "vyvar"}))
    return d


def test_purge_deletes_only_calibrated_fits_and_stamps_manifest(tmp_path: Path) -> None:
    d = _draft(tmp_path, raw=True)
    out = purge_calibrated_light_frames(d, draft_id=1)
    assert out["n_files"] == 2 and out["bytes"] > 0 and out["skipped"] == ""
    assert not list((d / "calibrated" / "lights").rglob("*.fit*"))
    assert (d / "calibrated" / "lights" / "qc_metrics.csv").is_file()
    assert (d / "Raw" / "lights" / "NoFilter_60_2" / "Light_001.fits").is_file()
    stamp = load_draft_manifest(d)["calibrated_frames_purged"]
    assert stamp["n_files"] == 2 and stamp["bytes"] == out["bytes"]
    assert load_draft_manifest(d)["calibration_mode"] == "vyvar"


def test_purge_keeps_frames_without_source_frames(tmp_path: Path) -> None:
    d = _draft(tmp_path, raw=False)
    out = purge_calibrated_light_frames(d, draft_id=1)
    assert out["n_files"] == 0 and "no source frames" in out["skipped"]
    assert len(list((d / "calibrated" / "lights").rglob("*.fit*"))) == 2
    assert "calibrated_frames_purged" not in load_draft_manifest(d)


def test_purge_keeps_precalibrated_source(tmp_path: Path) -> None:
    d = tmp_path / "draft_000002"
    _fits(d / "non_calibrated" / "lights" / "g" / "Light_001.fits")
    _fits(d / "calibrated" / "lights" / "g" / "Light_001.fits")
    out = purge_calibrated_light_frames(d / "non_calibrated", draft_id=2)
    assert out["n_files"] == 1
    assert (d / "non_calibrated" / "lights" / "g" / "Light_001.fits").is_file()
