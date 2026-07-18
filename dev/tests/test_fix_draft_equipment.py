"""Tests for fix_draft_equipment maintenance script."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from database import VyvarDatabase
from tools.reference_seed import seed_reference_observatory

ROOT = Path(__file__).resolve().parent.parent


def _load_fix_module():
    path = ROOT / "scripts" / "fix_draft_equipment.py"
    spec = importlib.util.spec_from_file_location("fix_draft_equipment", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _seed_equipment_db(db_path: Path) -> VyvarDatabase:
    db = VyvarDatabase(db_path)
    # Reference fixture (C5A-150M id=4, Carl-Zeiss id=1, ...) — not product seed.
    seed_reference_observatory(db)
    db.conn.execute(
        """
        INSERT OR IGNORE INTO EQUIPMENTS
          (ID, CAMERANAME, SENSORTYPE, SENSORSIZE, PIXELSIZE, GAIN_ADU, READNOISE_E)
        VALUES (2, 'C3-26000', 'IMX571', '6252*4176', 3.76, 0.78, 1.5);
        """
    )
    db.conn.execute(
        """
        UPDATE EQUIPMENTS
        SET CAMERANAME = 'C5A-150M', SENSORTYPE = 'IMX411', SENSORSIZE = '14208*10656',
            PIXELSIZE = 3.76, GAIN_ADU = 1.0, READNOISE_E = 1.5
        WHERE ID = 4;
        """
    )
    db.conn.execute(
        """
        INSERT INTO OBS_DRAFT (ID, ID_EQUIPMENTS, ID_TELESCOPE)
        VALUES (426, 4, 1);
        """
    )
    db.conn.commit()
    return db


def _write_sample_fits(
    draft_dir: Path,
    *,
    setup: str = "g_60_4",
    instrume: str = "C5A-150M",
    naxis: tuple[int, int] = (3552, 2664),
    binning: int = 4,
    gain: float = 12.48,
) -> Path:
    lights = draft_dir / "non_calibrated" / "lights" / setup
    lights.mkdir(parents=True, exist_ok=True)
    path = lights / "sample.fits"
    hdu = fits.PrimaryHDU(data=np.zeros((8, 8), dtype=np.float32))
    hdu.header["INSTRUME"] = instrume
    hdu.header["NAXIS1"] = naxis[0]
    hdu.header["NAXIS2"] = naxis[1]
    hdu.header["XBINNING"] = binning
    hdu.header["YBINNING"] = binning
    hdu.header["GAIN"] = gain
    hdu.header["IMAGETYP"] = "OBJECT"
    hdu.header["EXPTIME"] = 60.0
    hdu.writeto(path, overwrite=True)
    return path


def test_verify_draft_equipment_scores_c5a_geometry(tmp_path):
    db_path = tmp_path / "vyvar.db"
    _seed_equipment_db(db_path)
    draft_dir = tmp_path / "Drafts" / "draft_000426"
    _write_sample_fits(draft_dir)

    class _Cfg:
        database_path = str(db_path)
        archive_root = str(tmp_path)

    mod = _load_fix_module()
    report = mod.verify_draft_equipment(426, cfg=_Cfg())
    assert report["verdict"] == "proceed"
    assert report["verified_equipment_id"] == 4
    assert report["current_equipment_id"] == 4
    scores = {row["equipment_id"]: row["score"] for row in report["equipment_scores"]}
    assert scores[4] > scores[2]


@pytest.mark.skipif(
    not (ROOT / "Archive" / "Drafts" / "draft_000426").is_dir(),
    reason="requires on-disk draft_000426 (skipped after ARCHIVE-CLEANUP 2026-07-15)",
)
def test_fix_draft_equipment_real_draft426_dry_run():
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "fix_draft_equipment.py"),
            "--draft",
            "426",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "verified=4" in proc.stdout
    assert "No change needed" in proc.stdout
