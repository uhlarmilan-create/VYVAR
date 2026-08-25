"""PSF-only sidecar merge for RUN ePSF (INV-PSF-ADDITIVE-01)."""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
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
    "psf_ac_policy",
    "psf_quality_fallback",
    "psf_group_n",
    "x_fit",
    "y_fit",
)

# Fit coordinates are PSF-only (EPSF-SHAPE-01-F F3); treated as PSF columns
# so INV-PSF-ADDITIVE-01 non-PSF byte-identity still holds.
_PSF_FIT_COORD_COLS = frozenset({"x_fit", "y_fit"})


def is_psf_column(name: str) -> bool:
    n = str(name)
    return n.startswith("psf_") or n in _PSF_FIT_COORD_COLS


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


def stamp_p4_none_on_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Invert stored AC and stamp ``p4_none`` on PSF columns. Non-PSF columns untouched."""
    from psf_photometry import invert_applied_ac

    out = df.copy()
    n = len(out)
    flux = pd.to_numeric(out["psf_flux"], errors="coerce") if "psf_flux" in out.columns else pd.Series([np.nan] * n)
    err = (
        pd.to_numeric(out["psf_flux_err"], errors="coerce")
        if "psf_flux_err" in out.columns
        else pd.Series([np.nan] * n)
    )
    fac = (
        pd.to_numeric(out["psf_ac_factor"], errors="coerce")
        if "psf_ac_factor" in out.columns
        else pd.Series([np.nan] * n)
    )
    if "psf_ac_applied" in out.columns:
        applied = out["psf_ac_applied"].astype(str).str.lower().str.strip().isin(
            ("1", "true", "t", "yes", "y")
        )
    else:
        applied = pd.Series([False] * n)
    new_flux = []
    new_err = []
    new_fac = []
    new_applied = []
    new_n = []
    new_pol = []
    for i in range(n):
        fl = float(flux.iloc[i]) if i < len(flux) else float("nan")
        er = float(err.iloc[i]) if i < len(err) else float("nan")
        fa = float(fac.iloc[i]) if i < len(fac) else float("nan")
        ap = bool(applied.iloc[i]) if i < len(applied) else False
        if math.isfinite(fl) or math.isfinite(fa):
            nfl, ner = invert_applied_ac(fl, er, fa, ap)
            new_flux.append(nfl)
            new_err.append(ner)
            new_fac.append(1.0)
            new_applied.append(False)
            new_n.append(0)
            new_pol.append("p4_none")
        else:
            new_flux.append(fl)
            new_err.append(er)
            new_fac.append(fa)
            new_applied.append(ap)
            new_n.append(out["psf_ac_n_used"].iloc[i] if "psf_ac_n_used" in out.columns else float("nan"))
            new_pol.append("")
    out["psf_flux"] = new_flux
    out["psf_flux_err"] = new_err
    out["psf_ac_factor"] = new_fac
    out["psf_ac_applied"] = new_applied
    out["psf_ac_n_used"] = new_n
    out["psf_ac_policy"] = new_pol
    return out


def stamp_p4_none_sidecar(
    sidecar_path: Path,
    *,
    pipeline_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Rewrite one proc sidecar PSF columns to P4 (uncorrected). ADDITIVE-01 asserted."""
    from pipeline import _vyvar_df_to_csv

    before = read_vyvar_csv(sidecar_path, low_memory=False)
    after = stamp_p4_none_on_dataframe(before)
    assert_inv_psf_additive_01(
        before,
        after,
        frame_name=sidecar_path.name,
        pipeline_meta=pipeline_meta,
    )
    _vyvar_df_to_csv(after, sidecar_path)
    return {"csv": str(sidecar_path), "status": "ok"}


def psf_ac_policy_params(policy: str) -> dict[str, Any]:
    """Named parameters stamped with the F6 AC policy."""
    p = str(policy or "").strip().lower()
    if p == "chi2_lt5_legacy":
        return {"chi2_limit": 5.0, "min_ref_stars": 5}
    return {"psf_ac_factor": 1.0, "psf_ac_applied": False}


def write_epsf_ac_merge_meta(
    platesolve_dir: Path,
    *,
    policy: str,
    n_sidecars: int,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Stamp policy name + parameters into F6 merge meta (platesolve sidecar)."""
    meta: dict[str, Any] = {
        "psf_ac_policy": str(policy),
        "psf_ac_params": psf_ac_policy_params(policy),
        "n_sidecars": int(n_sidecars),
        "stamped_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "task": "EPSF-AC-02-WIRE",
    }
    if extra:
        meta.update(extra)
    out = Path(platesolve_dir) / "epsf_ac_merge_meta.json"
    out.write_text(json.dumps(meta, indent=2) + "\n", encoding="ascii")
    return out


def stamp_p4_none_science_sidecars(
    frames_root: Path,
    *,
    platesolve_dir: Path | None = None,
    pipeline_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Invert stored AC on every ``proc_*.csv`` under frames_root. ADDITIVE-01 per file."""
    root = Path(frames_root)
    files = sorted(root.glob("proc_*.csv"))
    if not files:
        raise FileNotFoundError(f"No proc_*.csv under {root}")
    rows: list[dict[str, Any]] = []
    n_ok = 0
    for p in files:
        rec = stamp_p4_none_sidecar(p, pipeline_meta=pipeline_meta)
        rec["status"] = "ok"
        rows.append(rec)
        n_ok += 1
    meta_path = None
    if platesolve_dir is not None:
        meta_path = write_epsf_ac_merge_meta(
            Path(platesolve_dir),
            policy="p4_none",
            n_sidecars=n_ok,
            extra={"frames_root": str(root)},
        )
        phot = Path(platesolve_dir) / "photometry"
        if phot.is_dir():
            try:
                from photometry_core import merge_photometry_pipeline_meta

                merge_photometry_pipeline_meta(
                    phot,
                    {
                        "psf_ac_policy": "p4_none",
                        "psf_ac_params": psf_ac_policy_params("p4_none"),
                    },
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("[ePSF] pipeline_meta AC stamp skipped: %s", exc)
    return {
        "written": n_ok,
        "frames_total": len(files),
        "policy": "p4_none",
        "merge_meta": str(meta_path) if meta_path is not None else None,
        "frames": rows,
    }


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
    _policy = str(getattr(_cfg, "psf_ac_policy", "p4_none") or "p4_none")
    _ac_params = psf_ac_policy_params(_policy)
    if _epsf_job_summary is not None:
        _epsf_job_summary["psf_ac_policy"] = _policy
        _epsf_job_summary["psf_ac_params"] = _ac_params
        from epsf_frame_accounting import persist_epsf_job_summary

        persist_epsf_job_summary(_epsf_job_summary, ps)
    write_epsf_ac_merge_meta(ps, policy=_policy, n_sidecars=int(n_ok))
    try:
        from photometry_core import merge_photometry_pipeline_meta

        merge_photometry_pipeline_meta(
            ps / "photometry",
            {"psf_ac_policy": _policy, "psf_ac_params": _ac_params},
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[ePSF] pipeline_meta AC stamp skipped: %s", exc)
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
