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
from photutils import __version__ as _photutils_version

from gaia_catalog_id import normalize_gaia_source_id, read_vyvar_csv
from invariants_runtime import InvariantViolation, inv_check
from proc_frame_store import proc_csv_path_for_aligned_fits

LOGGER = logging.getLogger(__name__)

INV_PSF_ADDITIVE_01 = "INV-PSF-ADDITIVE-01"

# Fit coordinates are PSF-only (EPSF-SHAPE-01-F F3); treated as PSF columns
# so INV-PSF-ADDITIVE-01 non-PSF byte-identity still holds.
_PSF_FIT_COORD_COLS = frozenset({"x_fit", "y_fit"})


def is_psf_column(name: str) -> bool:
    n = str(name)
    return n.startswith("psf_") or n in _PSF_FIT_COORD_COLS


def non_psf_columns(columns) -> list[str]:
    return [c for c in columns if not is_psf_column(c)]


def non_psf_columns_equal(before: pd.DataFrame, after: pd.DataFrame, *, rtol: float = 1e-9) -> bool:
    """True when the non-psf columns of two tables hold the same values.

    Numeric values compare with a relative tolerance (``rtol``) because the CSV writer keeps ~11
    significant digits: an exact digit fingerprint of a re-read file differs whenever a value sits
    on a rounding tie (533 g: flux_k125 2.5538216505083e-05 -> 2.5538216505e-05 failed a .10g
    fingerprint). Columns whose entries are all numbers on both sides compare numerically, whatever
    the dtype; the others (names, flags) and integer IDs longer than 15 digits (Gaia source_id, where
    a tolerance would hide a changed ID) compare as exact text; NaN / empty equal.
    """
    cols = non_psf_columns(before.columns)
    if cols != non_psf_columns(after.columns) or len(before) != len(after):
        return False
    for c in cols:
        ta, tb = _as_text(before[c]), _as_text(after[c])
        if ta == tb:
            continue
        if _looks_like_ids(ta) or _looks_like_ids(tb):
            return False
        va, vb = _as_numbers(before[c]), _as_numbers(after[c])
        if va is not None and vb is not None:
            if not bool(np.isclose(va, vb, rtol=rtol, atol=0.0, equal_nan=True).all()):
                return False
        else:
            return False
    return True


def _as_numbers(s: pd.Series) -> np.ndarray | None:
    """Float values of a column whose every non-empty entry is a number (object columns from a
    pyarrow round-trip included), else None."""
    num = pd.to_numeric(s, errors="coerce")
    empty = s.isna() | (s.astype(str).str.strip() == "")
    if bool((num.isna() & ~empty).any()):
        return None
    return num.to_numpy(dtype=float)


def _looks_like_ids(text: list[str]) -> bool:
    """Integers longer than a float mantissa (Gaia source_id): exact text only."""
    return any(len(t.lstrip("-")) > 15 and t.lstrip("-").isdigit() for t in text)


def _as_text(s: pd.Series) -> list[str]:
    return ["" if (v is None or (isinstance(v, float) and math.isnan(v))) else str(v) for v in s.tolist()]


def guarded_psf_sidecar_write(
    sidecar_path: Path,
    after: pd.DataFrame,
    before: pd.DataFrame,
) -> None:
    """Write sidecar; INV-EXPORT-READ-ONLY-01 restores if non-psf columns change."""
    from pipeline import _vyvar_df_to_csv

    pre_bytes = sidecar_path.read_bytes() if sidecar_path.is_file() else None
    try:
        _vyvar_df_to_csv(after, sidecar_path)
        on_disk = read_vyvar_csv(sidecar_path, low_memory=False)
        if not non_psf_columns_equal(before, on_disk):
            raise RuntimeError(
                "INV-EXPORT-READ-ONLY-01: non-psf columns changed; sidecar restored"
            )
    except Exception:
        if pre_bytes is not None:
            sidecar_path.write_bytes(pre_bytes)
        raise


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


# OUTPUT-REVIEW-2026-10 step 4: the PSF columns a proc file carries (read by the PSF light curves,
# lc_outlier and the night locus). The fit's diagnostics (Fisher-only error, flags, S/N, local sky,
# group size, neighbour light) stay in memory.
PSF_PROC_COLUMNS: tuple[str, ...] = (
    "psf_flux",
    "psf_flux_err",
    "psf_chi2",
    "psf_converged",
    "psf_sat_masked",
    "psf_edge_masked",
)


def merge_psf_frame_result(
    sidecar_path: Path,
    fit_df: pd.DataFrame,
    *,
    frame_name: str,
    pipeline_meta: dict[str, Any] | None = None,
) -> int:
    """Replace the PSF columns of one proc sidecar with ``fit_df`` (INV-PSF-ADDITIVE-01 enforced).

    Earlier PSF columns are dropped (the per-star cutout fitter left psf_quality / psf_ac_* /
    psf_group_n); ``psf_fit_ok`` stays False until the night chi^2 locus sets it. Returns the
    number of rows the fit covered.
    """
    if not sidecar_path.is_file():
        raise FileNotFoundError(f"Missing proc sidecar for {frame_name}: {sidecar_path}")
    before = read_vyvar_csv(sidecar_path, low_memory=False)
    after = before[non_psf_columns(before.columns)].copy()
    key = after["catalog_id"].map(lambda v: str(normalize_gaia_source_id(v)).strip() if pd.notna(v) else "")
    fit = fit_df.drop_duplicates("catalog_id").set_index("catalog_id")
    fit = fit[[c for c in PSF_PROC_COLUMNS if c in fit.columns]]
    for col in fit.columns:
        after[col] = key.map(fit[col])
    from proc_frame_store import round_proc_columns  # noqa: PLC0415

    after = round_proc_columns(after, columns=list(fit.columns))  # proc precision; other columns untouched
    after["psf_fit_ok"] = False
    after["psf_converged"] = after["psf_converged"].astype("boolean").fillna(False).astype(bool)
    assert_inv_psf_additive_01(before, after, frame_name=frame_name, pipeline_meta=pipeline_meta)
    guarded_psf_sidecar_write(sidecar_path, after, before)
    return int(key.isin(fit.index).sum())


def frame_saturation_limit_adu(sidecar_path: Path) -> float:
    """The frame's peak non-linearity / saturation limit (``saturate_limit_adu_85pct`` of its proc
    sidecar, INV-SAT-LIMIT): PSF fit pixels at or above it are masked. Raises when it is unknown."""
    df = read_vyvar_csv(sidecar_path, low_memory=False, usecols=lambda c: c == "saturate_limit_adu_85pct")
    vals = pd.to_numeric(df.get("saturate_limit_adu_85pct"), errors="coerce").dropna()
    if vals.empty:
        raise ValueError(f"saturate_limit_adu_85pct missing in {Path(sidecar_path).name}")
    return float(vals.median())


_FIT_CTX: dict[str, Any] = {}


def _fit_worker_init(ctx: dict[str, Any]) -> None:
    _FIT_CTX.clear()
    _FIT_CTX.update(ctx)


def _fit_frame_task(fp: Path) -> dict[str, Any]:
    """Frame ePSF + frame PSF fit of the fit set (``_FIT_CTX``); the merge into the sidecar is the
    parent's (INV-PSF-ADDITIVE-01 guard, serial). A failure is returned, not raised."""
    from epsf_frame import build_frame_model, fit_frame_psf  # noqa: PLC0415
    from frame_footprint import aligned_footprint  # noqa: PLC0415
    from psf_photometry import _psf_resolve_gain_read_noise  # noqa: PLC0415

    c = _FIT_CTX
    out: dict[str, Any] = {"mrec": None, "fit_df": None, "exception_class": None, "exception_message": None,
                           "traceback_tail": None}
    try:
        model, mrec = build_frame_model(fp, c["build_stars"], cutout_size=c["cutout"], fwhm_px=c["fwhm_ms"],
                                        delta=c["delta"], grid=c["grid"])
        out["mrec"] = mrec
        with fits.open(fp, memmap=True) as hdul:
            data = np.asarray(hdul[0].data, dtype=np.float32)
            hdr = hdul[0].header.copy()
        covered = aligned_footprint(hdr, data.shape)
        gain, rn = _psf_resolve_gain_read_noise(hdr, photometry_dir=Path(c["platesolve_dir"]) / "photometry", cfg=c["app_config"])
        sidecar = proc_csv_path_for_aligned_fits(fp)
        out["fit_df"] = fit_frame_psf(
            data, model, c["fit_stars"], c["neighbours"], fwhm_px=mrec["fwhm_px"], gain=gain, read_noise_e=rn,
            sat_limit_adu=frame_saturation_limit_adu(sidecar), covered=covered,
        )
    except Exception as exc:  # noqa: BLE001 - recorded per frame, INV-PSF-FRAME-01 decides the job
        _record_exception(out, exc)
    return out


def _record_exception(rec: dict[str, Any], exc: BaseException) -> None:
    import traceback  # noqa: PLC0415

    rec.update({"exception_class": type(exc).__name__, "exception_message": str(exc),
                "traceback_tail": "\n".join(traceback.format_exc().splitlines()[-8:])})


def run_epsf_psf_merge_job(
    *,
    frames_root: Path,
    platesolve_dir: Path,
    app_config: Any,
    progress_cb: Callable[[int, int, str], None] | None = None,
    fit_set: Any,
    pipeline_meta: dict[str, Any] | None = None,
    max_frames: int | None = None,
) -> dict[str, Any]:
    """RUN ePSF photometry pass (EPSF-SPATIAL-2026-10).

    1. Night spatial perturbation of the ePSF on a 3x3 / 2x2 grid (``epsf_frame.night_spatial_delta``).
    2. Per frame: the frame's ePSF from the MASTERSTAR build stars + the perturbation
       (``epsf_frame.build_frame_model``, ``epsf_frames/``), PSF photometry of ``fit_set`` (the selected
       targets + their comparison stars, ``epsf_science_set.build_epsf_fit_set``) in one frame-level fit
       (``epsf_frame.fit_frame_psf``), PSF columns merged into the proc sidecar.
    3. Night chi^2 locus sets ``psf_fit_ok`` (D-EPSF-CHI2-LOCUS-01); INV-PSF-FRAME-01 accounting.
    """
    from epsf_frame import OVERSAMPLING as EPSF_OVERSAMPLING
    from epsf_frame import frame_fit_sources, night_spatial_delta
    from parallel_frames import map_frames, ram_capped_workers
    from epsf_frame_accounting import finalize_epsf_frame_job, list_epsf_science_light_fits
    from psf_photometry import (
        EPSF_BUILD_STARS_META_NAME,
        EPSF_BUILD_STARS_NAME,
        EPSF_FRAMES_DIR,
        EPSF_FRAMES_MANIFEST,
        EPSF_SPATIAL_DELTA_NAME,
    )
    from psf_chi2_locus import finalize_night_locus_for_inv_psf_frame_01

    ps = Path(platesolve_dir)
    root = Path(frames_root)
    files = list_epsf_science_light_fits(root)
    if max_frames is not None:
        files = files[: max(0, int(max_frames))]
    if not files:
        raise FileNotFoundError(f"No science-light FITS under {root}")

    if not fit_set.catalog_ids:
        raise ValueError(f"ePSF fit set is empty: {fit_set.empty_reason}")
    meta_p, stars_p = ps / EPSF_BUILD_STARS_META_NAME, ps / EPSF_BUILD_STARS_NAME
    for need in (meta_p, stars_p):
        if not need.is_file():
            raise FileNotFoundError(f"{need.name} missing - run the build-star selection (Step ePSF-1) first")
    build_meta = json.loads(meta_p.read_text(encoding="utf-8"))
    build_stars = pd.read_csv(stars_p, dtype={"catalog_id": str})
    cutout, fwhm_ms = int(build_meta["cutout_size"]), float(build_meta["fwhm_px"])
    frames_dir = ps / EPSF_FRAMES_DIR

    # EPSF-MERGE-2026-10: unresolved companions are judged at the night's typical seeing (median of the
    # frames' fwhm_px_for_aperture, NIGHT-FWHM-2026-10), not at the MASTERSTAR (the best frame).
    from photometry_comp import night_median_fwhm_px

    fwhm_night, fwhm_night_src = night_median_fwhm_px([proc_csv_path_for_aligned_fits(f) for f in files], float("nan"))
    if not (math.isfinite(fwhm_night) and fwhm_night > 0):
        raise ValueError("ePSF: no per-frame fwhm_px_for_aperture in the proc sidecars - night FWHM unknown")
    fit_stars, neighbours, sources_meta = frame_fit_sources(
        ps, fit_set.catalog_ids, cutout_size=cutout, fwhm_px=fwhm_ms, merge_fwhm_px=fwhm_night
    )
    sources_meta["merge_fwhm_source"] = fwhm_night_src

    if progress_cb is not None:
        progress_cb(0, len(files), "night spatial ePSF perturbation")
    # A worker holds the frame (float32) and the fit's float64 images (data, two error maps, two model
    # images, masks): ~6 x 8 bytes per pixel.
    workers = ram_capped_workers(app_config, files, bytes_per_pixel=8 * 6)
    spatial = night_spatial_delta(files, build_stars, cutout_size=cutout, fwhm_px=fwhm_ms,
                                  out_path=frames_dir / EPSF_SPATIAL_DELTA_NAME, workers=workers)
    delta = fits.getdata(frames_dir / EPSF_SPATIAL_DELTA_NAME).astype(np.float64) if spatial["file"] else None
    grid = (int(spatial["grid"][0]), int(spatial["grid"][1]))

    ctx = {"build_stars": build_stars, "cutout": cutout, "fwhm_ms": fwhm_ms, "delta": delta, "grid": grid,
           "platesolve_dir": str(ps), "app_config": app_config, "fit_stars": fit_stars, "neighbours": neighbours}
    records: list[dict[str, Any]] = []
    models: list[dict[str, Any]] = []
    done = map_frames(_fit_frame_task, files, workers=workers, initializer=_fit_worker_init, ctx=ctx)
    for i, (fp, res) in enumerate(zip(files, done, strict=True), start=1):
        if progress_cb is not None:
            progress_cb(i, len(files), f"PSF fit: {fp.name}")
        rec: dict[str, Any] = {"frame_name": fp.name, "frame_index": i - 1, "n_fit": 0, "n_ok": 0,
                               "exception_class": None, "exception_message": None, "traceback_tail": None}
        if res["mrec"] is not None:
            models.append(res["mrec"])
        if res["exception_class"] is None:
            try:
                rec["n_fit"] = merge_psf_frame_result(
                    proc_csv_path_for_aligned_fits(fp), res["fit_df"], frame_name=fp.name, pipeline_meta=pipeline_meta
                )
            except Exception as exc:  # noqa: BLE001 - recorded per frame, INV-PSF-FRAME-01 decides the job
                _record_exception(rec, exc)
        else:
            rec.update({k: res[k] for k in ("exception_class", "exception_message", "traceback_tail")})
        if rec["exception_class"] is not None:
            LOGGER.warning("[ePSF] frame %s failed: %s: %s", fp.name, rec["exception_class"], rec["exception_message"])
        records.append(rec)

    created = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    manifest = {
        "created_utc": created,
        "mode": "per_frame_plus_night_spatial",
        "photutils_version": _photutils_version,
        "cutout_size": int(cutout),
        "oversampling": int(EPSF_OVERSAMPLING),
        "build_stars_file": EPSF_BUILD_STARS_NAME,
        "n_build_stars": int(len(build_stars)),
        "spatial": spatial,
        "fit_set": fit_set.to_meta_dict(),
        "fit_sources": sources_meta,
        "frames": models,
    }
    frames_dir.mkdir(parents=True, exist_ok=True)
    (frames_dir / EPSF_FRAMES_MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    locus = finalize_night_locus_for_inv_psf_frame_01(root, records)
    if pipeline_meta is not None:
        pipeline_meta["psf_chi2_locus_night"] = locus
    summary = finalize_epsf_frame_job(records, science_set_meta=fit_set.to_meta_dict(), pipeline_meta=pipeline_meta)
    summary["psf_chi2_locus_night"] = {k: locus.get(k) for k in ("n_sigma", "night_locus", "n_rewritten", "n_kept_frame")}

    return {
        "written": int(sum(1 for r in records if r["exception_class"] is None)),
        "frames_total": len(files),
        "per_frame_dir": str(root),
        "frames": records,
        "epsf_job_summary": summary,
        "fit_set": fit_set.to_meta_dict(),
        "spatial": spatial,
        "epsf_frames_manifest": str(frames_dir / EPSF_FRAMES_MANIFEST),
    }
