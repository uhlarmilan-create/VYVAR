"""LOG-NOISE-2026-10 (repeated library warnings once), FITS card comments that fit one card, and
DISK-PREFLIGHT-2026-10 (a run that cannot fit on the disk stops before the import)."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

import infolog
import parallel_frames as pf

SRC = Path(__file__).resolve().parents[2] / "src_py"


def test_repeated_library_warning_is_shown_once_and_counted(caplog, monkeypatch):
    monkeypatch.setattr(infolog, "_repeat_filter", None)
    filt = infolog.quiet_repeated_library_warnings()
    try:
        lg = logging.getLogger("astropy")
        with caplog.at_level(logging.INFO):
            for x in (10.5, 11.25, 12.0):
                lg.warning("The star at (%s, 3) has been excluded from ePSF fitting", x)
            lg.warning("another warning")
            infolog.log_repeated_warnings_summary()
        shown = [r.getMessage() for r in caplog.records if r.name == "astropy"]
        assert len(shown) == 2
        assert any("repeated 2 more time(s)" in r.getMessage() for r in caplog.records)
    finally:
        for lg in infolog._library_loggers():
            lg.removeFilter(filt)


def test_astroquery_logger_stays_its_own_class(monkeypatch):
    """The filter must not create plain loggers that break astroquery ('_set_defaults')."""
    monkeypatch.setattr(infolog, "_repeat_filter", None)
    filt = infolog.quiet_repeated_library_warnings()
    try:
        from astroquery import log as aq_log
        from astroquery.simbad import Simbad  # noqa: F401  imports call log._set_defaults

        assert hasattr(aq_log, "_set_defaults")
    finally:
        for lg in infolog._library_loggers():
            lg.removeFilter(filt)


def test_header_card_comments_fit_one_card():
    """hdr["KEY"] = (value, "comment"): 8 + '= ' + 20 value columns + ' / ' leaves 47 for the comment."""
    pat = re.compile(r"""\[["']([A-Z0-9_-]{1,8})["']\]\s*=\s*\(\s*(.+?),\s*f?(["'])(.+?)\3\s*,?\s*\)""", re.S)
    long_comments = []
    for p in sorted(SRC.glob("*.py")):
        for m in pat.finditer(p.read_text(encoding="utf-8")):
            if len(m.group(4)) > 47:
                long_comments.append(f"{p.name}:{m.group(1)}")
    assert long_comments == []


def _light(path: Path, ny: int, nx: int) -> Path:
    fits.PrimaryHDU(np.zeros((ny, nx), dtype=np.uint16)).writeto(path)
    return path


def test_disk_need_is_raw_copy_plus_two_float_frames(tmp_path):
    lights = [_light(tmp_path / f"l{i}.fits", 100, 200) for i in range(3)]
    raw = sum(p.stat().st_size for p in lights)
    assert pf.run_disk_need(lights) == int(pf.DISK_MARGIN * (raw + 2 * 3 * 100 * 200 * 4))


def test_disk_check_stops_when_the_run_does_not_fit(tmp_path, monkeypatch):
    lights = [_light(tmp_path / "l.fits", 100, 200)]
    import shutil

    monkeypatch.setattr(shutil, "disk_usage", lambda _p: shutil._ntuple_diskusage(10**9, 10**9 - 1000, 1000))
    with pytest.raises(OSError, match="not enough disk space"):
        pf.check_disk_for_run(tmp_path / "Archive", lights)
    monkeypatch.setattr(shutil, "disk_usage", lambda _p: shutil._ntuple_diskusage(10**12, 0, 10**12))
    assert "GB free" in pf.check_disk_for_run(tmp_path / "Archive", lights)
