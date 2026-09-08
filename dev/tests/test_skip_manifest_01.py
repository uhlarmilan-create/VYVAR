# -*- coding: ascii -*-
"""D-LC-SKIP-MANIFEST-01: sidecar outside photometry SHA globs."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd

from skip_manifest import (
    SKIP_MANIFEST_NAME,
    build_skip_manifest_payload,
    skip_manifest_path,
    write_skip_manifest,
)
from tests.photometry_sha import (
    _SHA_PATTERN_COMP_QA,
    _SHA_PATTERNS_CORE,
    photometry_sha_files,
)


def test_manifest_name_outside_sha_globs() -> None:
    """Name must not match either SHA glob family."""
    name = SKIP_MANIFEST_NAME
    assert name == "lightcurves_skip_manifest.json"
    assert not name.startswith("lightcurve_")
    assert not name.endswith(".csv")
    assert not name.startswith("comp_quality_")
    assert not name.startswith("comp_qa_")
    for pat in list(_SHA_PATTERNS_CORE) + [_SHA_PATTERN_COMP_QA]:
        assert "skip_manifest" not in pat


def test_manifest_path_not_collected_by_sha_helper(tmp_path: Path) -> None:
    photo = tmp_path / "platesolve" / "NoFilter_60_2" / "photometry"
    photo.mkdir(parents=True)
    (photo / SKIP_MANIFEST_NAME).write_text("{}", encoding="utf-8")
    (photo / "lightcurves").mkdir()
    (photo / "lightcurves" / "lightcurve_1.csv").write_text("bjd,mag_calib_final\n1,10\n", encoding="utf-8")
    core = {p.name for p in photometry_sha_files(tmp_path, include_comp_qa=False)}
    ext = {p.name for p in photometry_sha_files(tmp_path, include_comp_qa=True)}
    assert SKIP_MANIFEST_NAME not in core
    assert SKIP_MANIFEST_NAME not in ext


def test_manifest_three_classes_and_always_written(tmp_path: Path) -> None:
    lc_dir = tmp_path / "lightcurves"
    lc_dir.mkdir()
    empty_lc = lc_dir / "lightcurve_EXPORT.csv"
    empty_lc.write_text(
        "bjd,mag_calib_final,flag\n,,no_data\n",
        encoding="utf-8",
    )
    summary = [
        {
            "catalog_id": "NOCOMPS",
            "vsx_name": "Stub",
            "ac_skip_reason": "no_comps",
            "lc_csv": "",
        },
        {
            "catalog_id": "PINABORT",
            "vsx_name": "Pinned",
            "ac_skip_reason": "no_comps",
            "lc_csv": "",
        },
        {
            "catalog_id": "EXPORT",
            "vsx_name": "EmptyLC",
            "ac_skip_reason": "",
            "lc_csv": str(empty_lc),
        },
    ]

    def _pin(cid: str, path: Path | None = None) -> bool:
        _ = path
        return str(cid) == "PINABORT"

    with patch("pinned_ensembles.is_pinned_target", _pin):
        path = write_skip_manifest(
            tmp_path,
            lc_dir=lc_dir,
            summary_rows=summary,
            timestamp="2026-09-08T00:00:00Z",
        )
    assert path.name == SKIP_MANIFEST_NAME
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    classes = {t["catalog_id"]: t["class"] for t in data["targets"]}
    assert classes["NOCOMPS"] == "no_comps_stub"
    assert classes["PINABORT"] == "pin_rms_abort"
    assert classes["EXPORT"] == "export_empty"
    assert data["schema"] == "vyvar.lightcurves_skip_manifest.v1"


def test_manifest_empty_list_when_nothing_skipped(tmp_path: Path) -> None:
    lc_dir = tmp_path / "lightcurves"
    lc_dir.mkdir()
    good = lc_dir / "lightcurve_OK.csv"
    good.write_text(
        "bjd,mag_calib_final,flag\n2461154.3,10.1,normal\n",
        encoding="utf-8",
    )
    active = pd.DataFrame(
        [{"catalog_id": "OK", "vsx_name": "Fine", "ac_skip_reason": ""}]
    )
    path = write_skip_manifest(
        tmp_path,
        lc_dir=lc_dir,
        summary_rows=[],
        active_df=active,
        timestamp="2026-09-08T00:00:00Z",
    )
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["targets"] == []
    assert skip_manifest_path(tmp_path).is_file()


def test_build_payload_other_class() -> None:
    payload = build_skip_manifest_payload(
        summary_rows=[],
        active_df=pd.DataFrame([{"catalog_id": "MISSING", "vsx_name": "Gone"}]),
        lc_dir=Path("no_such_lc_dir"),
        timestamp="2026-09-08T00:00:00Z",
    )
    assert payload["targets"][0]["class"] == "other"
