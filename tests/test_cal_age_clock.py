"""CAL-AGE-CLOCK: unified master validity on header capture date."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from astropy.io import fits

from calibration import get_master_age_days, resolve_master_age
from importer import (
    _age_days,
    _reset_master_age_mtime_warnings,
    get_calibration_status,
)


def _write_master(path: Path, *, cdate: str | None = None, date_obs: str | None = None) -> None:
    data = [[1.0, 2.0], [3.0, 4.0]]
    hdu = fits.PrimaryHDU(data=data)
    if cdate is not None:
        hdu.header["VY_CDATE"] = (cdate, "capture")
    if date_obs is not None:
        hdu.header["DATE-OBS"] = (date_obs, "obs")
    hdu.header["IMAGETYP"] = "MASTER DARK"
    hdu.header["XBINNING"] = 1
    hdu.header["YBINNING"] = 1
    path.parent.mkdir(parents=True, exist_ok=True)
    hdu.writeto(path, overwrite=True)


def _mtime_days_ago(path: Path, days: float) -> None:
    ts = (datetime.now(timezone.utc) - timedelta(days=days)).timestamp()
    os.utime(path, (ts, ts))


@pytest.fixture
def cal_lib(tmp_path: Path) -> Path:
    root = tmp_path / "CalibrationLibrary"
    root.mkdir()
    return root


def test_resolve_master_age_header_priority_vy_cdate(tmp_path: Path) -> None:
    p = tmp_path / "md_test.fits"
    old = (datetime.now(timezone.utc) - timedelta(days=120)).strftime("%Y-%m-%dT%H:%M:%SZ")
    fresh = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    _write_master(p, cdate=old, date_obs=fresh)
    info = resolve_master_age(p)
    assert info.source == "header"
    assert info.header_key == "VY_CDATE"
    assert info.age_days >= 119.0


def test_copy_scenario_header_old_mtime_fresh_rejected(cal_lib: Path) -> None:
    """Header date old + mtime fresh -> import scan rejects (the copy bug fix)."""
    p = cal_lib / "md_copy.fits"
    old = (datetime.now(timezone.utc) - timedelta(days=120)).strftime("%Y-%m-%dT%H:%M:%SZ")
    _write_master(p, cdate=old)
    _mtime_days_ago(p, 1.0)  # simulate copy to new machine

    assert get_master_age_days(p) > 90.0
    mtime_age = (datetime.now(timezone.utc).timestamp() - os.path.getmtime(p)) / 86400.0
    assert mtime_age < 5.0

    _reset_master_age_mtime_warnings()
    age = _age_days(p)
    assert age is not None
    assert age > 90.0


def test_header_fresh_mtime_old_accepted(cal_lib: Path) -> None:
    p = cal_lib / "md_fresh.fits"
    fresh = (datetime.now(timezone.utc) - timedelta(days=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
    _write_master(p, cdate=fresh)
    _mtime_days_ago(p, 200.0)

    _reset_master_age_mtime_warnings()
    age = _age_days(p)
    assert age is not None
    assert age <= 90.0


def test_no_header_date_mtime_fallback_warns(cal_lib: Path, caplog: pytest.LogCaptureFixture) -> None:
    p = cal_lib / "legacy.fits"
    _write_master(p)
    _mtime_days_ago(p, 5.0)

    warnings: list[str] = []
    _reset_master_age_mtime_warnings()
    age = _age_days(p, warnings=warnings)
    assert age is not None
    assert 4.0 <= age <= 6.0
    assert len(warnings) == 1
    assert "mtime fallback" in warnings[0]
    assert p.name in warnings[0]
    # one warning per file per scan
    _age_days(p, warnings=warnings)
    assert len(warnings) == 1


def test_boundary_age_at_limit_inclusive_ok(cal_lib: Path) -> None:
    """age <= validity_days is valid (UI and import); expired only when age > limit."""
    p = cal_lib / "md_boundary.fits"
    capture = datetime.now(timezone.utc) - timedelta(days=89, hours=12)
    _write_master(p, cdate=capture.strftime("%Y-%m-%dT%H:%M:%SZ"))
    age = get_master_age_days(p)
    assert age < 90.0
    stt = get_calibration_status(p, kind="Master Dark", validity_days=90)
    assert stt.status == "ok"


def test_boundary_age_just_over_limit_expired(cal_lib: Path) -> None:
    p = cal_lib / "md_over.fits"
    over = (datetime.now(timezone.utc) - timedelta(days=90, hours=6)).strftime("%Y-%m-%dT%H:%M:%SZ")
    _write_master(p, cdate=over)
    stt = get_calibration_status(p, kind="Master Dark", validity_days=90)
    assert stt.status == "expired"


def test_naive_date_obs_treated_as_utc(tmp_path: Path) -> None:
    p = tmp_path / "md_naive.fits"
    _write_master(p, date_obs="2024-01-15T22:00:00")
    info = resolve_master_age(p)
    assert info.source == "header"
    assert info.header_key == "DATE-OBS"
    assert info.capture_utc is not None
    assert info.capture_utc.tzinfo is not None


def test_smart_scan_uses_header_age_not_mtime(cal_lib: Path, tmp_path: Path) -> None:
    p = cal_lib / "md_scan.fits"
    old = (datetime.now(timezone.utc) - timedelta(days=150)).strftime("%Y-%m-%dT%H:%M:%SZ")
    _write_master(p, cdate=old)
    _mtime_days_ago(p, 2.0)

    source = tmp_path / "source"
    source.mkdir()
    # minimal empty source  scan still evaluates library paths via observation groups if lights exist
    # Direct status check mirrors import path
    _reset_master_age_mtime_warnings()
    stt = get_calibration_status(p, kind="Master Dark", validity_days=90)
    assert stt.status == "expired"


def test_flat_boundary_200_days_inclusive(cal_lib: Path) -> None:
    p = cal_lib / "mf_boundary.fits"
    capture = datetime.now(timezone.utc) - timedelta(days=199, hours=12)
    hdu = fits.PrimaryHDU(data=[[1.0]])
    hdu.header["VY_CDATE"] = capture.strftime("%Y-%m-%dT%H:%M:%SZ")
    hdu.header["IMAGETYP"] = "MASTER FLAT"
    hdu.writeto(p, overwrite=True)
    stt = get_calibration_status(p, kind="Master Flat", validity_days=200)
    assert stt.status == "ok"
