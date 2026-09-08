# -*- coding: ascii -*-
"""Skip-manifest sidecar (D-LC-SKIP-MANIFEST-01).

Written at Phase 2A finalize (``photometry_phase2a._phase2a_finalize_exports``).
Path: ``<photometry>/lightcurves_skip_manifest.json``.

Outside ``photometry_sha_files`` globs (``dev/tests/photometry_sha.py:100-105``):

* ``**/photometry/**/lightcurve_*.csv``
* ``**/photometry/**/comp_quality_*.json``
* ``**/platesolve/**/comparison_stars_per_target.csv``
* ``**/photometry/**/lightcurves/comp_qa_*.json``  (extended)

Always written. ``targets`` is an empty list when nothing is skipped.
LC CSV bytes are not touched. A ``skip_reason`` column in the LC CSVs
is deferred to the next natural era re-cut.

Classes
-------
no_comps_stub
    Phase 2A counted drop (``ac_skip_reason=no_comps``), target not pinned.
    Producer: ``photometry_lightcurve.py:77-95``.
pin_rms_abort
    Pinned target, no LC / ``no_comps`` after
    ``PinnedEnsembleInsufficientError`` (``pinned_ensembles.py:659-664``;
    ``phase01_run.py:772-779``).
export_empty
    LC CSV present, zero exportable points
    (``export_reports.py:893``, ``:997-1004``).
other
    LC absent or empty for another reason.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

SKIP_MANIFEST_NAME = "lightcurves_skip_manifest.json"
SKIP_CLASSES = ("no_comps_stub", "pin_rms_abort", "export_empty", "other")


def skip_manifest_path(photometry_dir: Path) -> Path:
    return Path(photometry_dir) / SKIP_MANIFEST_NAME


def _exportable_count(lc_path: Path) -> tuple[int, int]:
    if not lc_path.is_file():
        return 0, 0
    df = pd.read_csv(lc_path, comment="#")
    n_rows = int(len(df))
    try:
        from export_reports import _select_export_lc_rows  # noqa: PLC0415

        n_exp = int(len(_select_export_lc_rows(df)))
    except Exception:  # noqa: BLE001
        mag = pd.to_numeric(df.get("mag_calib_final", df.get("mag_calib")), errors="coerce")
        n_exp = int(mag.notna().sum()) if mag is not None else 0
    return n_rows, n_exp


def classify_skip_row(
    *,
    catalog_id: str,
    name: str,
    ac_skip_reason: str,
    lc_path: Path,
    pinned: bool,
) -> dict[str, Any] | None:
    exists = lc_path.is_file()
    n_rows, n_exp = _exportable_count(lc_path) if exists else (0, 0)
    reason = str(ac_skip_reason or "").strip()
    if pinned and (reason == "no_comps" or not exists):
        klass = "pin_rms_abort"
        stage = "pinned_ensembles.py:659-664"
    elif reason == "no_comps":
        klass = "no_comps_stub"
        stage = "photometry_lightcurve.py:77-95"
    elif exists and n_exp == 0:
        klass = "export_empty"
        stage = "export_reports.py:997-1004"
    elif not exists:
        klass = "other"
        stage = "photometry_phase2a.py:3612"
    else:
        return None
    return {
        "catalog_id": str(catalog_id),
        "name": str(name or ""),
        "class": klass,
        "ac_skip_reason": reason,
        "n_rows": int(n_rows),
        "n_exportable": int(n_exp),
        "producing_stage": stage,
    }


def build_skip_manifest_payload(
    *,
    summary_rows: list[dict[str, Any]] | None,
    active_df: pd.DataFrame | None,
    lc_dir: Path,
    timestamp: str | None = None,
) -> dict[str, Any]:
    try:
        from pinned_ensembles import is_pinned_target  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        def is_pinned_target(_cid: str, path: Path | None = None) -> bool:  # noqa: ARG001
            return False

    rows: dict[str, dict[str, Any]] = {}
    for r in summary_rows or []:
        cid = str(r.get("catalog_id", "") or "").strip()
        if not cid:
            continue
        rec = classify_skip_row(
            catalog_id=cid,
            name=str(r.get("vsx_name", r.get("name", "")) or ""),
            ac_skip_reason=str(r.get("ac_skip_reason", "") or ""),
            lc_path=Path(lc_dir) / f"lightcurve_{cid}.csv",
            pinned=bool(is_pinned_target(cid)),
        )
        if rec:
            rows[cid] = rec

    if active_df is not None and not getattr(active_df, "empty", True):
        id_col = "catalog_id" if "catalog_id" in active_df.columns else active_df.columns[0]
        name_col = "vsx_name" if "vsx_name" in active_df.columns else (
            "name" if "name" in active_df.columns else id_col
        )
        for _, ar in active_df.iterrows():
            cid = str(ar.get(id_col, "") or "").strip()
            if not cid or cid in rows:
                continue
            rec = classify_skip_row(
                catalog_id=cid,
                name=str(ar.get(name_col, "") or ""),
                ac_skip_reason=str(ar.get("ac_skip_reason", "") or ""),
                lc_path=Path(lc_dir) / f"lightcurve_{cid}.csv",
                pinned=bool(is_pinned_target(cid)),
            )
            if rec:
                rows[cid] = rec

    ts = timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "schema": "vyvar.lightcurves_skip_manifest.v1",
        "timestamp_utc": ts,
        "targets": [rows[k] for k in sorted(rows)],
    }


def write_skip_manifest(
    photometry_dir: Path,
    *,
    lc_dir: Path | None = None,
    summary_rows: list[dict[str, Any]] | None = None,
    active_df: pd.DataFrame | None = None,
    timestamp: str | None = None,
) -> Path:
    photometry_dir = Path(photometry_dir)
    photometry_dir.mkdir(parents=True, exist_ok=True)
    out = skip_manifest_path(photometry_dir)
    payload = build_skip_manifest_payload(
        summary_rows=summary_rows,
        active_df=active_df,
        lc_dir=Path(lc_dir) if lc_dir is not None else photometry_dir / "lightcurves",
        timestamp=timestamp,
    )
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out
