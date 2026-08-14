# -*- coding: ascii -*-
"""INV-CAL-02 / INV-SAT-01 disk-evidence regression (COG-A1-01 E1/E2)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src_py"
sys.path.insert(0, str(SRC))

from cal_stage import stamp_cal_stage_headers  # noqa: E402
from invariants_runtime import (  # noqa: E402
    check_cal_stage,
    check_sat_diag,
    inv_check,
)


def test_sat_diag_loads_draft_root_json(tmp_path: Path) -> None:
    draft = tmp_path / "draft_000999"
    photo = draft / "platesolve" / "NoFilter_60_2" / "photometry"
    photo.mkdir(parents=True)
    sd = {
        "sat_adu": 65535.0,
        "sat_source": "DERIVED",
        "lin_source": "DEFAULT_FRAC",
    }
    (draft / "sat_diag.json").write_text(json.dumps(sd), encoding="ascii")
    meta: dict = {}
    check_sat_diag(meta, photometry_dir=photo)
    rec = [x for x in meta["invariants"] if x["id"] == "INV-SAT-01"][-1]
    assert rec["ok"] is True
    assert "loaded sat_diag.json" in rec["detail"]


def test_cal_stage_reads_vy_calstage_header(tmp_path: Path) -> None:
    draft = tmp_path / "draft_000998"
    cal = draft / "calibrated" / "lights"
    photo = draft / "platesolve" / "NoFilter_60_2" / "photometry"
    cal.mkdir(parents=True)
    photo.mkdir(parents=True)
    data = np.full((8, 8), 120.0, dtype=np.float32)
    hdr = fits.Header()
    stamp_cal_stage_headers(hdr, data, stage="SKYSF_2")
    fits.writeto(cal / "light.fits", data, header=hdr, overwrite=True)
    meta: dict = {}
    check_cal_stage(meta, photometry_dir=photo)
    rec = [x for x in meta["invariants"] if x["id"] == "INV-CAL-02"][-1]
    assert rec["ok"] is True
    assert "VY_CALSTAGE='SKYSF_2'" in rec["detail"] or "VY_CALSTAGE='SKYSF_2'" in rec["detail"].replace('"', "'")


@pytest.mark.skipif(
    not (REPO / "Archive" / "Drafts" / "draft_000512").is_dir(),
    reason="draft 512 not present",
)
def test_fire_proof_draft512_sat_and_cal_gates() -> None:
    photo = REPO / "Archive" / "Drafts" / "draft_000512" / "platesolve" / "NoFilter_60_2" / "photometry"
    meta: dict = {}
    check_sat_diag(meta, photometry_dir=photo)
    check_cal_stage(meta, photometry_dir=photo)
    sat = [x for x in meta["invariants"] if x["id"] == "INV-SAT-01"][-1]
    cal = [x for x in meta["invariants"] if x["id"] == "INV-CAL-02"][-1]
    assert sat["ok"] is True
    assert "sat_diag not stamped" not in sat["detail"]
    assert cal["ok"] is True
    assert "cal_stage not stamped" not in cal["detail"]
