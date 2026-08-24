"""PSF-only sidecar merge for RUN ePSF (INV-PSF-ADDITIVE-01)."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from astropy.io import fits

from gaia_catalog_id import read_vyvar_csv
from invariants_runtime import InvariantViolation, inv_check
from proc_frame_store import proc_csv_path_for_aligned_fits

LOGGER = logging.getLogger(__name__)

INV_PSF_ADDITIVE_01 = "INV-PSF-ADDITIVE-01"

PSF_MERGE_COLUMNS: tuple[str, ...] = (
    "psf_flux",
    "psf_flux_err",
    "psf_chi2",
    "psf_fit_ok",
    "psf_quality",
    "psf_snr",
    "psf_ac_factor",
    "psf_ac_n_used",
    "psf_ac_applied",
    "psf_quality_fallback",
)


def is_psf_column(name: str) -> bool:
    return str(name).startswith("psf_")


def non_psf_columns(columns) -> list[str]:
    return [c for c in columns if not is_psf_column(c)]


def assert_inv_psf_additive_01(
    before: pd.DataFrame,
    after: pd.DataFrame,
    *,
    frame_name: str,
    pipeline_meta: dict[str, Any] | None = None,
) -> None:
    """Non-PSF columns must be unchanged after PSF merge (numeric tolerance for float noise)."""
    meta = pipeline_meta if pipeline_meta is not None else {}
    b_cols = non_psf_columns(before.columns)
    a_cols = non_psf_columns(after.columns)
    if b_cols != a_cols:
        detail = f"frame={frame_name} non_psf column set changed"
        inv_check(meta, INV_PSF_ADDITIVE_01, False, policy="FAIL", detail=detail)
        raise InvariantViolation(f"{INV_PSF_ADDITIVE_01}: {detail}")

    if len(before) != len(after):
        detail = f"frame={frame_name} row count {len(before)} -> {len(after)}"
        inv_check(meta, INV_PSF_ADDITIVE_01, False, policy="FAIL", detail=detail)
        raise InvariantViolation(f"{INV_PSF_ADDITIVE_01}: {detail}")

    if "catalog_id" in before.columns:
        b_ids = before["catalog_id"].astype(str).tolist()
        a_ids = after["catalog_id"].astype(str).tolist()
        if b_ids != a_ids:
            detail = f"frame={frame_name} catalog_id row order changed"
            inv_check(meta, INV_PSF_ADDITIVE_01, False, policy="FAIL", detail=detail)
            raise InvariantViolation(f"{INV_PSF_ADDITIVE_01}: {detail}")

    for col in b_cols:
        a = before[col]
        b = after[col]
        if pd.api.types.is_numeric_dtype(a) or pd.api.types.is_numeric_dtype(b):
            na = pd.to_numeric(a, errors="coerce")
            nb = pd.to_numeric(b, errors="coerce")
            ok = np.isclose(na, nb, rtol=0.0, atol=1e-9, equal_nan=True) | (na.isna() & nb.isna())
            if not bool(ok.all()):
                n_bad = int((~ok).sum())
                detail = f"frame={frame_name} column={col} numeric drift n={n_bad}"
                inv_check(meta, INV_PSF_ADDITIVE_01, False, policy="FAIL", detail=detail)
                raise InvariantViolation(f"{INV_PSF_ADDITIVE_01}: {detail}")
        else:
            if not a.astype(str).equals(b.astype(str)):
                detail = f"frame={frame_name} column={col} value drift"
                inv_check(meta, INV_PSF_ADDITIVE_01, False, policy="FAIL", detail=detail)
                raise InvariantViolation(f"{INV_PSF_ADDITIVE_01}: {detail}")

    inv_check(meta, INV_PSF_ADDITIVE_01, True, policy="FAIL", detail=f"frame={frame_name} ok")


def merge_psf_into_sidecar(
    *,
    fits_path: Path,
    sidecar_path: Path,
    st: dict[str, Any],
    target_ids: set[str] | None,
    pipeline_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Read existing sidecar, update psf_* columns only, enforce INV-PSF-ADDITIVE-01."""
    from pipeline import _fill_psf_catalog_columns, _vyvar_df_to_csv

    if not sidecar_path.is_file():
        raise FileNotFoundError(
            f"Missing proc sidecar for {fits_path.name}: {sidecar_path} "
            "(PSF merge refuses to fabricate a catalog via full export)"
        )

    before = read_vyvar_csv(sidecar_path, low_memory=False)
    before_snap = before.copy()

    with fits.open(fits_path, memmap=True) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32)
        hdr = hdul[0].header.copy()

    st = dict(st)
    st["epsf_frame_name"] = fits_path.name
    st["_run_epsf"] = True
    st["_psf_merge_only"] = True

    after = _fill_psf_catalog_columns(
        before.copy(),
        data,
        hdr,
        st,
        target_ids=target_ids,
    )
    assert_inv_psf_additive_01(
        before_snap,
        after,
        frame_name=fits_path.name,
        pipeline_meta=pipeline_meta,
    )
    _vyvar_df_to_csv(after, sidecar_path)
    rec = st.get("_psf_frame_record")
    return {
        "file": fits_path.name,
        "status": "ok",
        "csv": str(sidecar_path),
        "psf_frame_record": dict(rec) if isinstance(rec, dict) else None,
    }


def run_epsf_psf_merge_job(
    *,
    frames_root: Path,
    platesolve_dir: Path,
    app_config: Any,
    draft_id: int | None = None,
    equipment_id: int | None = None,
    progress_cb: Callable[[int, int, str], None] | None = None,
    pipeline_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """RUN ePSF photometry pass: science-light frames only, PSF columns merged into existing sidecars."""
    from epsf_frame_accounting import finalize_epsf_frame_job, list_epsf_science_light_fits
    from epsf_science_set import build_epsf_science_set
    from pipeline import _epsf_fit_catalog_ids, _export_catalog_psf_st_fields

    ps = Path(platesolve_dir)
    root = Path(frames_root)
    files = list_epsf_science_light_fits(root)
    if not files:
        raise FileNotFoundError(f"No science-light FITS under {root}")

    _cfg = app_config
    if not bool(getattr(_cfg, "psf_photometry_enabled", False)):
        LOGGER.warning("[ePSF merge] psf_photometry_enabled=False - PSF columns will be empty")

    _sci = build_epsf_science_set(ps)
    if not _sci.catalog_ids:
        raise ValueError(
            "ePSF science set is empty"
            + (f": {_sci.empty_reason}" if _sci.empty_reason else "")
        )
    _epsf_science_meta = _sci.to_meta_dict()
    _target_ids = _epsf_fit_catalog_ids(ps, psf_photometry_enabled=True)

    st_base = _export_catalog_psf_st_fields(_cfg, ps)
    st_base["platesolve_dir"] = str(ps.resolve())
    st_base["draft_id"] = int(draft_id) if draft_id is not None else None
    st_base["equipment_id"] = int(equipment_id) if equipment_id is not None else None
    _frame_index_by_name = {p.name: i for i, p in enumerate(files)}

    rows_out: list[dict[str, Any]] = []
    total = len(files)
    for i, fp in enumerate(files, start=1):
        if progress_cb is not None:
            progress_cb(i, total, f"PSF merge: {fp.name}")
        st = dict(st_base)
        st["epsf_frame_index_by_name"] = _frame_index_by_name
        st["epsf_frame_index"] = _frame_index_by_name.get(fp.name)
        sidecar = proc_csv_path_for_aligned_fits(fp)
        try:
            row = merge_psf_into_sidecar(
                fits_path=fp,
                sidecar_path=sidecar,
                st=st,
                target_ids=_target_ids,
                pipeline_meta=pipeline_meta,
            )
        except Exception as exc:  # noqa: BLE001
            row = {
                "file": fp.name,
                "status": f"error: {exc}",
                "csv": str(sidecar),
                "psf_frame_record": {
                    "frame_name": fp.name,
                    "frame_index": _frame_index_by_name.get(fp.name),
                    "n_fit": 0,
                    "n_ok": 0,
                    "exception_class": type(exc).__name__,
                    "exception_message": str(exc),
                    "traceback_tail": None,
                },
            }
        rows_out.append(row)

    _psf_recs = [
        r["psf_frame_record"]
        for r in rows_out
        if isinstance(r.get("psf_frame_record"), dict)
    ]
    _epsf_job_summary = None
    if _psf_recs:
        _epsf_job_summary = finalize_epsf_frame_job(
            _psf_recs,
            platesolve_dir=ps,
            science_set_meta=_epsf_science_meta,
            pipeline_meta=pipeline_meta,
        )

    n_ok = sum(1 for r in rows_out if r.get("status") == "ok")
    psf_lc: dict[str, Any] | None = None
    try:
        from psf_internal_lc import write_internal_psf_lightcurves_after_epsf_job  # noqa: PLC0415

        psf_lc = write_internal_psf_lightcurves_after_epsf_job(
            platesolve_dir=ps,
            frames_root=root,
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[ePSF] internal PSF LC write skipped: %s", exc)
        psf_lc = {"error": str(exc)}
    return {
        "written": int(n_ok),
        "frames_total": len(files),
        "per_frame_dir": str(root),
        "frames": rows_out,
        "epsf_job_summary": _epsf_job_summary,
        "science_set": _epsf_science_meta,
        "internal_psf_lc": psf_lc,
    }
