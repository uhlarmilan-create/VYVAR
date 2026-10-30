"""Moved from pipeline.py (CONSOLIDATE-01E6a). Facade re-exports these names.

Alignment orchestration, plate solve, masterstar build, VSX/exo merge.
The four giants stay in pipeline.py this wave (E6b).
"""
from __future__ import annotations

import logging
import math
import os
import warnings
from pathlib import Path
from typing import Any, Sequence
import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.wcs import FITSFixedWarning
import pandas as pd
from config import AppConfig
from catalog_db import query_local_gaia_by_source_ids, query_local_vsx, vsx_amplitude_mag
from database import DraftTechnicalMetadataError, VyvarDatabase
from gaia_catalog_id import normalize_gaia_source_id
from infolog import log_event
from optics_selection import resolve_optics_ids_for_platesolve
from optics import optics_hint
from utils import iter_fits_paths_recursive as _iter_fits_recursive, strip_celestial_wcs_keys, strip_vendor_platesolve_metadata
import fits_meta
from pipeline_constants import _EXO_HOST_ANNOTATION_COLUMNS
from pipeline_calibrate import (
    _filter_light_paths_maybe,
    _qc_center_crop_for_stars,
)
from aperture_policy import clamp_fwhm_px, fwhm_from_header_vy_fwhm
from proc_frame_store import _vyvar_df_to_csv  # LAYER-2026-10: moved there
from pipeline_preprocess import _path_is_under_tree, _path_segments_forbidden_for_masterstar_physical_source, _pick_preferred_masterstar_basename_hit, _resolve_best_effort_path_under, fits_basename_index  # LAYER-2026-10: moved there

# Same named logger as pipeline.LOGGER (logging.getLogger singleton).
# Avoids pipeline -> pipeline_astrometry -> pipeline at module load.
LOGGER = logging.getLogger("pipeline")

# Target selection through aperture photometry (active_targets.csv, zone_flag, skip_photometry) lives in
# ``photometry_core`` - ``run_target_and_comp_selection`` / ``run_aperture_photometry``, not in this file.

def _ensure_parent_dirs_for_aligned_fits(out_path: Path) -> None:
    """Ensure parent folders exist for nested outputs under ``detrended_aligned/lights/...``."""
    os.makedirs(str(out_path.parent), exist_ok=True)


def _assert_alignment_produced_fits(aligned_root: Path) -> None:
    """Fail fast if alignment wrote no FITS under the group folder (recursive; not only top-level)."""
    n = len(_iter_fits_recursive(aligned_root))
    if n == 0:
        raise RuntimeError(
            "Alignment failed - no output files found! "
            f"(no FITS in {aligned_root.resolve()} including subdirectories.)"
        )


def draft_obs_group_count(archive_path: Path | str) -> int:
    """Number of obs_group subdirs under draft lights root that contain FITS."""
    from draft_provenance import draft_archive_root, resolve_draft_lights_root

    ap = draft_archive_root(Path(archive_path).expanduser())
    pl = resolve_draft_lights_root(ap)
    if not pl.is_dir():
        pl = ap / "processed" / "lights"
    if not pl.is_dir():
        return 0
    return sum(
        1
        for d in pl.iterdir()
        if d.is_dir() and any(_iter_fits_recursive(d))
    )


def draft_is_multi_group_obs(archive_path: Path | str) -> bool:
    return int(draft_obs_group_count(archive_path)) > 1


def _sort_masterstar_paths_by_fwhm(files: list[Path]) -> list[Path]:
    """Best first: lowest header VY_FWHM, unknown last."""
    scored: list[tuple[float, str, Path]] = []
    for fp in files:
        v: float | None = None
        try:
            with fits.open(fp, memmap=False) as h:
                v = fwhm_from_header_vy_fwhm(h[0].header)
        except Exception:  # noqa: BLE001
            pass
        score = float(v) if v is not None and math.isfinite(v) and v > 0 else float("inf")
        scored.append((score, str(fp).casefold(), fp))
    scored.sort(key=lambda t: (t[0], t[1]))
    return [p for _, _, p in scored]


def _strip_external_platesolve_header(hdr: fits.Header) -> None:
    """Drop celestial WCS and common third-party plate-solve keywords (ASTAP, astrometry.net, ...).

    VYVAR establishes astrometry with its own solver only (``astrosolve``, PLATESOLVE-2026-10).
    """
    strip_celestial_wcs_keys(hdr)
    strip_vendor_platesolve_metadata(hdr)
    for _k in (
        "WCSAXES",
        "WCSDIM",
        "CROTA1",
        "CROTA2",
        "WCSNAME",
        "VY_PSOLV",
        "VY_SIPRF",
    ):
        try:
            del hdr[_k]
        except KeyError:
            pass


def build_masterstar_from_detrended(
    *,
    detrended_root: Path,
    output_fits: Path,
    only_paths: "Sequence[Path | str]",
    fwhm_fallback_px: float | None = None,
    pre_calibrated: bool = False,
) -> dict[str, Any]:
    """Build MASTERSTAR by copying the best (lowest VY_FWHM) of the candidate FITS ``only_paths``."""
    import shutil

    import numpy as np

    root = Path(detrended_root).resolve()
    _forbid_kw = {"pre_calibrated": bool(pre_calibrated)}
    all_fits = [
        fp
        for fp in _iter_fits_recursive(root)
        if _path_is_under_tree(root, fp)
        and not _path_segments_forbidden_for_masterstar_physical_source(fp, **_forbid_kw)
    ]
    files = _filter_light_paths_maybe(all_fits, only_paths)
    if not files:
        remapped: list[Path] = []
        seen_r: set[str] = set()
        _bn_index = fits_basename_index(root, pre_calibrated=pre_calibrated)  # G03-41: one scan
        for op in only_paths:
            hit = _resolve_best_effort_path_under(root, str(op), pre_calibrated=pre_calibrated, index=_bn_index)
            if hit is None or not _path_is_under_tree(root, hit):
                continue
            if _path_segments_forbidden_for_masterstar_physical_source(hit, **_forbid_kw):
                log_event(f"MASTERSTAR: mapping dropped a RAW path -> {hit}")
                continue
            try:
                rk = str(hit.resolve()).casefold()
            except OSError:
                rk = str(hit).casefold()
            if rk in seen_r:
                continue
            seen_r.add(rk)
            remapped.append(hit)
        if remapped:
            files = remapped
            log_event(
                f"MASTERSTAR: selection merged via best-effort mapping ({len(files)} FITS; path filter was empty)."
            )
    if files:
        log_event(
            f"Found {len(files)} files for MASTERSTAR in {root} (candidate selection; "
            f"in the tree {len(all_fits)} FITS in total)."
        )
    else:
        try:
            _want = ", ".join(Path(str(x)).name for x in only_paths[:8])
        except Exception:  # noqa: BLE001
            _want = str(only_paths)
        msg = (
            f"MASTERSTAR: explicit selection does not match the files under {root}: {_want}"
            + (" ..." if len(list(only_paths)) > 8 else "")
        )
        log_event(msg)
        raise FileNotFoundError(msg)
    try:
        _comp_names = [Path(p).name for p in files]
        log_event(f"[folder] MASTERSTAR COMPOSITION: Using [{', '.join(_comp_names)}]")
    except Exception:  # noqa: BLE001
        pass

    sorted_files = _sort_masterstar_paths_by_fwhm(files)

    output_fits.parent.mkdir(parents=True, exist_ok=True)

    reference_path = Path(sorted_files[0])
    if _path_segments_forbidden_for_masterstar_physical_source(
        reference_path, **_forbid_kw
    ) or not _path_is_under_tree(root, reference_path):
        raise FileNotFoundError(
            f"MASTERSTAR: rejected source outside the lights tree or from RAW: {reference_path}"
        )
    if not reference_path.exists():
        log_event(f"[X] MASTERSTAR FAIL: Reference file {reference_path} not found.")
        fallback_hits = [
            x
            for x in _iter_fits_recursive(root)
            if x.name == reference_path.name
            and _path_is_under_tree(root, x)
            and not _path_segments_forbidden_for_masterstar_physical_source(x, **_forbid_kw)
        ]
        if fallback_hits:
            reference_path = _pick_preferred_masterstar_basename_hit(fallback_hits) or fallback_hits[0]
            log_event(f"[OK] MASTERSTAR fallback reference found: {reference_path}")
        else:
            raise FileNotFoundError(f"MASTERSTAR reference file not found: {reference_path}")

    best_frame_fwhm_px: float | None = None
    try:
        from astropy.io.fits import getheader

        best_frame_fwhm_px = fwhm_from_header_vy_fwhm(getheader(str(reference_path)))  # G03-29
    except Exception:  # noqa: BLE001
        best_frame_fwhm_px = None

    try:
        with fits.open(reference_path, memmap=False) as hdul0:
            _d0 = hdul0[0].data
            _sh = getattr(_d0, "shape", None)
            if _d0 is None or _sh is None or len(_sh) != 2:
                raise ValueError("MASTERSTAR: reference FITS is not a valid 2D primary.")
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"MASTERSTAR: cannot read reference FITS: {exc}") from exc

    shutil.copy2(reference_path, output_fits)
    if len(files) <= 1:
        _ms_pick_msg = "single candidate"
    else:
        _ms_pick_msg = f"candidates: {len(files)}; best by VY_FWHM"
    log_event(f"MASTERSTAR: clean copy -> {output_fits} (source {reference_path.name}, {_ms_pick_msg}).")

    # FWHM-ONE-2026-10: MASTERSTAR is one frame, so its FWHM is that frame's own VY_FWHM
    # (frame QC elliptical Gaussian fit). Measured here only if the header lacks it.
    if best_frame_fwhm_px is not None:
        _fwhm_auto = float(best_frame_fwhm_px)
        log_event(f"MASTERSTAR: FWHM = {_fwhm_auto:.3f} px (VY_FWHM of the source frame)")
    else:
        from frame_metrics import measure_frame  # noqa: PLC0415

        with fits.open(output_fits, memmap=False) as _hm:
            _arr = np.asarray(_hm[0].data, dtype=np.float32)
        # G03-29: the same crop and measurement as frame QC (the VY_FWHM it would have stamped)
        _m = measure_frame(_qc_center_crop_for_stars(_arr))
        _fwhm_meas = clamp_fwhm_px(_m.get("fwhm_mean"))
        if _fwhm_meas is not None:
            _fwhm_auto = float(_fwhm_meas)
            log_event(f"MASTERSTAR: FWHM = {_fwhm_auto:.3f} px (measured on MASTERSTAR, VY_FWHM missing)")
        elif clamp_fwhm_px(fwhm_fallback_px) is not None:
            _fwhm_auto = float(clamp_fwhm_px(fwhm_fallback_px))
            log_event(f"MASTERSTAR: FWHM = {_fwhm_auto:.3f} px (night median QC FWHM; not measurable on MASTERSTAR)")
        else:
            raise ValueError(
                f"MASTERSTAR: no FWHM - {reference_path.name} has no VY_FWHM, none measurable and no night QC FWHM."
            )

    try:
        with fits.open(output_fits, mode="update", memmap=False) as h:
            hdr = h[0].header
            vy_fwhm = float(_fwhm_auto)
            hdr["VY_FWHM"] = (round(vy_fwhm, 4), "FWHM [pix] elliptical Gaussian (frame QC)")
            log_event(f"MASTERSTAR: VY_FWHM = {vy_fwhm:.3f} px written to FITS.")

            _strip_external_platesolve_header(hdr)
            log_event(
                "MASTERSTAR: external WCS/plate-solve metadata removed from the MASTERSTAR copy "
                "(ASTAP, astrometry.net, ...) - astrometry is set only by the VYVAR Gaia solver."
            )

            h.flush()
    except Exception as _exc:  # noqa: BLE001
        log_event(f"MASTERSTAR: VY_FWHM write failed: {_exc!s}")

    return {
        "masterstar_path": str(output_fits),
        "frames_used": int(len(files)),
        "reference_path": str(reference_path),
        "copied_from": str(reference_path),
        "best_frame_fwhm_px": best_frame_fwhm_px,
    }


def _update_masterstar_obs_file_status(
    *,
    cfg: AppConfig,
    draft_id: int | None,
    selected_ref_path: Path | None,
    wcs_ok: bool,
    n_stars: int,
) -> None:
    if draft_id is None or selected_ref_path is None:
        return
    try:
        from draft_provenance import load_or_init_manifest, patch_draft_manifest, resolve_draft_dir_for_id

        app_cfg = cfg
        db = VyvarDatabase.for_config(app_cfg)
        ar = getattr(app_cfg, "archive_root", None)
        if ar:
            db._archive_root_override = Path(str(ar)).expanduser().resolve()
        try:
            root = resolve_draft_dir_for_id(db, int(draft_id))
            if root is None:
                return
            manifest = load_or_init_manifest(root, int(draft_id))
            files = manifest.get("files")
            if not isinstance(files, list):
                return
            ref_name = selected_ref_path.name
            ref_l = ref_name.casefold()
            raw_l = ref_l[5:] if ref_l.startswith("proc_") else ref_l
            proc_l = raw_l if raw_l.startswith("proc_") else f"proc_{raw_l}"
            updated = 0
            for i, entry in enumerate(files):
                if not isinstance(entry, dict):
                    continue
                fp = str(entry.get("file_path") or "")
                fp_l = Path(fp).name.casefold()
                if not (
                    fp_l == raw_l
                    or fp_l == proc_l
                    or fp_l == ref_l
                    or raw_l in fp_l
                    or proc_l in fp_l
                    or ref_l in fp_l
                ):
                    continue
                patched = dict(entry)
                insp = dict(patched.get("inspection") or {})
                insp["wcs"] = 1 if bool(wcs_ok) else 0
                insp["stars"] = int(max(0, int(n_stars)))
                patched["inspection"] = insp
                files[i] = patched
                updated += 1
            if updated:
                patch_draft_manifest(root, int(draft_id), files=files)
            log_event(
                f"MASTERSTAR manifest update: DRAFT_ID={int(draft_id)}, WCS={1 if wcs_ok else 0}, "
                f"Stars={int(n_stars)}, rows={int(updated)}"
            )
        finally:
            db.conn.close()
    except Exception as exc:  # noqa: BLE001
        log_event(f"MASTERSTAR manifest update skipped: {exc!s}")


def get_masterstar_candidate_rows(
    draft_id: int,
    percentage: float,
    *,
    fwhm_max_px: float | None = None,
    db: VyvarDatabase,
) -> "pd.DataFrame":
    """Rank draft light frames by quality metrics for MASTERSTAR selection.

    Strict filter: include only rows with ``IS_REJECTED`` 0/NULL.
    Score (higher is better) is a normalized version of:

    Score = (1 / fwhm) * (1 / sky_level) * snr_estimate
    where ``snr_estimate`` is approximated from ``STAR_COUNT`` and ``SKY_LEVEL`` when no explicit SNR exists.
    """
    import numpy as np
    import pandas as pd

    did = int(draft_id)
    pct = float(percentage)
    pct = float(max(0.1, min(100.0, pct)))

    rows = db.fetch_draft_light_rows_for_quality(did)
    if not rows:
        return pd.DataFrame(columns=["FILE_PATH", "FWHM", "SKY_LEVEL", "STAR_COUNT", "SNR_EST", "SCORE"])

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["FILE_PATH", "FWHM", "SKY_LEVEL", "STAR_COUNT", "SNR_EST", "SCORE"])

    # Strict: IS_REJECTED == 0 (or NULL)
    if "IS_REJECTED" in df.columns:
        df["IS_REJECTED"] = pd.to_numeric(df["IS_REJECTED"], errors="coerce").fillna(0).astype(int)
        df = df[df["IS_REJECTED"] == 0].copy()
    lim_active = bool(fwhm_max_px is not None and float(fwhm_max_px) > 0)
    lim_v = float(fwhm_max_px) if lim_active else 0.0
    if lim_active:
        _f = pd.to_numeric(df.get("FWHM"), errors="coerce")
        df = df[_f.notna() & (_f <= lim_v)].copy()
    if df.empty:
        return pd.DataFrame(columns=["FILE_PATH", "FWHM", "SKY_LEVEL", "STAR_COUNT", "SNR_EST", "SCORE"])

    df["FWHM"] = pd.to_numeric(df.get("FWHM"), errors="coerce")
    df["SKY_LEVEL"] = pd.to_numeric(df.get("SKY_LEVEL"), errors="coerce")
    df["STAR_COUNT"] = pd.to_numeric(df.get("STAR_COUNT"), errors="coerce").fillna(0).astype(int)

    # Robust fallbacks: median scales for normalization.
    f_med = float(np.nanmedian(df["FWHM"].values)) if np.isfinite(np.nanmedian(df["FWHM"].values)) else 0.0
    s_med = (
        float(np.nanmedian(df["SKY_LEVEL"].values))
        if np.isfinite(np.nanmedian(df["SKY_LEVEL"].values))
        else 0.0
    )
    if not (math.isfinite(f_med) and f_med > 0):
        f_med = 1.0
    if not (math.isfinite(s_med) and s_med > 0):
        s_med = 1.0

    eps = 1e-9
    f = df["FWHM"].astype(float)
    sky = df["SKY_LEVEL"].astype(float)
    stars = df["STAR_COUNT"].astype(float)

    # snr_estimate: more stars and lower sky -> better SNR (proxy; true SNR not stored in manifest files[]).
    snr_est = (stars + 1.0) / np.sqrt(np.maximum(sky, 0.0) + 1.0)

    f_norm_inv = f_med / (np.maximum(f, 0.0) + eps)
    sky_norm_inv = s_med / (np.maximum(sky, 0.0) + eps)

    score = f_norm_inv * sky_norm_inv * snr_est
    score = np.where(np.isfinite(score), score, 0.0)

    df["SNR_EST"] = snr_est
    df["SCORE"] = score

    df = df.sort_values(["SCORE", "STAR_COUNT"], ascending=[False, False], kind="mergesort").reset_index(
        drop=True
    )

    k = int(max(1, math.ceil(len(df) * (pct / 100.0))))
    return df.head(k).loc[:, ["FILE_PATH", "FWHM", "SKY_LEVEL", "STAR_COUNT", "SNR_EST", "SCORE"]]


def get_masterstar_candidates(draft_id: int, percentage: float, *, db: VyvarDatabase) -> list[str]:
    """Return FILE_PATH list of top-ranked MASTERSTAR candidates (temporary list; manifest files[] is not modified)."""
    df = get_masterstar_candidate_rows(int(draft_id), float(percentage), db=db)
    if df.empty:
        return []
    return [str(x) for x in df["FILE_PATH"].tolist() if str(x).strip()]


def _vyvar_open_database(cfg: AppConfig) -> VyvarDatabase | None:
    try:
        return VyvarDatabase.for_config(cfg)
    except Exception:  # noqa: BLE001
        return None


def _plate_solve_input_bundle(
    fits_path: Path,
    *,
    app_config: AppConfig,
    equipment_id: int | None,
    draft_id: int | None = None,
    telescope_id: int | None = None,
) -> dict[str, Any]:
    """Open DB once: metadata, effective pixel, focal length, expected plate scale [arcsec/pixel]."""
    cfg_u = app_config
    db_u = _vyvar_open_database(cfg_u)
    _eq_use, _tel_use = resolve_optics_ids_for_platesolve(
        db_u, draft_id, equipment_id=equipment_id, telescope_id=telescope_id
    )
    out: dict[str, Any] = {
        "meta": {},
        "header": None,
        "eff_um": None,
        "focal_mm": None,
        "expected_arcsec_per_px": None,
    }
    try:
        with fits.open(fits_path, memmap=False) as hdul:
            out["header"] = hdul[0].header.copy()
        out["meta"] = fits_meta.extract_fits_metadata(
            fits_path,
            db=db_u,
            id_equipment=_eq_use,
            draft_id=draft_id,
        )
        if draft_id is not None:
            _m = out["meta"]
            if _m.get("focal_length") is None or _m.get("effective_pixel_um_plate_scale") is None:
                raise DraftTechnicalMetadataError(int(draft_id))
        # T08 optics module (G03-19): one rule - DB rig values first, header second, no rewriting
        hint = optics_hint(
            out["header"],
            db_u,
            equipment_id=_eq_use,
            telescope_id=_tel_use,
            binning=(
                max(1, int(out["meta"].get("binning", 1) or 1)),
                max(1, int(out["meta"].get("binning_y", out["meta"].get("binning", 1)) or 1)),
            ),
        )
        out["focal_mm"] = hint.focal_mm
        out["eff_um"] = hint.pixel_effective_um
        out["expected_arcsec_per_px"] = hint.arcsec_per_px
        if hint.arcsec_per_px is not None:
            log_event(
                f"MATH CHECK: ({hint.pixel_effective_um} / {hint.focal_mm}) * 206.265 = {hint.arcsec_per_px} "
                f"(focal: {hint.focal_source}, pixel: {hint.pixel_source})"
            )
    except DraftTechnicalMetadataError:
        raise  # FAIL-LOUD-2026-10 (code review 2 G03-18): the caller (app) handles missing optics metadata
    except Exception as exc:  # noqa: BLE001
        from except_fix_counters import get_except_fix_counters

        get_except_fix_counters().plate_solve_bundle_fail += 1
        LOGGER.error("[PLATE-SOLVE] _plate_solve_input_bundle failed: %s", exc)
        out["bundle_error"] = str(exc)
    finally:
        if db_u is not None:
            try:
                db_u.conn.close()
            except Exception:  # noqa: BLE001
                pass
    return out


def compute_plate_scale_from_db(
    equipment_id: int | None,
    telescope_id: int | None,
    db: Any,
    *,
    binning: int = 1,
) -> float | None:
    """Expected plate scale [arcsec/px] of the draft rig from ``EQUIPMENTS`` + ``TELESCOPE`` only
    (:func:`optics.optics_hint` without a header); None when either value is missing."""
    hint = optics_hint(
        None, db, equipment_id=equipment_id, telescope_id=telescope_id, binning=(int(binning), int(binning))
    )
    return hint.arcsec_per_px


def _solve_wcs_external(
    fits_path: Path,
    *,
    hint_ra_deg: float | None = None,
    hint_dec_deg: float | None = None,
    scale_hint_arcsec: float | None = None,
    app_config: AppConfig,
) -> dict[str, Any]:
    """Plate solve a FITS in place with VYVAR astrosolve (PLATESOLVE-2026-10): hinted with the
    header / caller position, blind index as fallback. The solver needs neither a field size
    nor a plate scale; a scale hint only orders the catalogue cones."""
    import astrosolve  # noqa: PLC0415

    cfg_u = app_config
    gaia_db = str(cfg_u.gaia_db_path or "").strip()
    if not gaia_db:
        return {"solved": False,
                "reason": "VYVAR solver: set the path to the local Gaia DR3 SQLite DB in Settings (gaia_db_path)."}
    r = astrosolve.solve_fits(
        Path(fits_path),
        gaia_db=gaia_db,
        index_dir=astrosolve.default_index_dir(gaia_db, str(getattr(cfg_u, "astrometry_index_dir", "") or "")),
        hint_ra_deg=hint_ra_deg,
        hint_dec_deg=hint_dec_deg,
        scale_hint_arcsec=scale_hint_arcsec,
        write=True,
        log=log_event,
    )
    if not r.get("solved"):
        r["reason"] = r.get("message")
    return r


def _wcs_field_center_radec_deg(fits_path: Path) -> tuple[float, float] | None:
    """RA/Dec (deg) of image center from existing celestial WCS."""

    try:
        with fits.open(fits_path, memmap=False) as hdul:
            hdr = hdul[0].header.copy()
        h = int(hdr.get("NAXIS2", 0) or 0)
        wpx = int(hdr.get("NAXIS1", 0) or 0)
        if h <= 0 or wpx <= 0:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            w = WCS(hdr)
        if not w.has_celestial:
            return None
        c = SkyCoord.from_pixel((wpx - 1) / 2.0, (h - 1) / 2.0, wcs=w, origin=0)
        return float(c.ra.deg), float(c.dec.deg)
    except Exception:  # noqa: BLE001
        return None


def _merge_vsx_exoplanet_variable_targets(
    vsx_df: pd.DataFrame,
    exo_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge VSX + exoplanet promotion rows; same Gaia id -> one row with VSX labels + exo_*."""
    from gaia_catalog_id import normalize_gaia_source_id_series

    vsx = vsx_df.copy() if vsx_df is not None else pd.DataFrame()
    exo = exo_df.copy() if exo_df is not None else pd.DataFrame()
    if vsx.empty and exo.empty:
        return vsx
    if "catalog_id" in vsx.columns:
        vsx["catalog_id"] = normalize_gaia_source_id_series(vsx["catalog_id"])
    if not exo.empty and "catalog_id" in exo.columns:
        exo["catalog_id"] = normalize_gaia_source_id_series(exo["catalog_id"])

    exo_extra = list(_EXO_HOST_ANNOTATION_COLUMNS) + ["target_origin"]
    for col in exo_extra:
        # object columns: they take the exoplanet values below, numbers too (pandas 3 refuses a
        # float in a str column)
        vsx[col] = vsx[col].astype(object) if col in vsx.columns else pd.Series("", index=vsx.index, dtype=object)
    if exo.empty:
        return vsx

    if vsx.empty:
        return exo

    exo_by_cid: dict[str, pd.Series] = {}
    for _, er in exo.iterrows():
        cid = str(er.get("catalog_id", "") or "").strip()
        if cid:
            exo_by_cid[cid] = er

    vsx_cids: set[str] = set()
    for i in vsx.index:
        cid = str(vsx.at[i, "catalog_id"] or "").strip()
        if not cid:
            continue
        vsx_cids.add(cid)
        er = exo_by_cid.get(cid)
        if er is not None:
            for col in exo_extra:
                if col in er.index:
                    vsx.at[i, col] = er[col]

    exo_only = exo.loc[~exo["catalog_id"].astype(str).str.strip().isin(vsx_cids)].copy()
    if exo_only.empty:
        return vsx
    merged = pd.concat([vsx, exo_only], ignore_index=True)
    log_event(
        f"[EXO TARGET] variable_targets merge: VSX={len(vsx)} exo-only added={len(exo_only)} "
        f"total={len(merged)}"
    )
    return merged


def _query_vsx_local_frame_bbox(
    *,
    wcs: Any,
    width_px: int,
    height_px: int,
    vsx_db_path: Path | None,
    margin_px: float = 50.0,
    center: SkyCoord | None = None,
    require_db: bool = False,
) -> pd.DataFrame:
    """Query **local VSX** within the FRAME footprint (frame bbox + ``margin_px``), spatial-first.

    Completeness must not depend on row order: the frame bbox is tiny (sub-degree), so the SQL
    result is small and the global ``catalog_query_max_rows`` cap never truncates it (unlike the
    3.5 deg cone in ``_query_vsx_local``, which hit the 15000-row cap and dropped a contiguous Dec
    slice). Returns raw VSX rows over the bbox; the caller applies the precise in-frame pixel
    filter (matching ``margin_px``). RA wrap is handled via centre-relative offsets.
    """
    from catalog_db import count_vsx_rows, require_vsx_db

    if require_db:
        vp = require_vsx_db(vsx_db_path)
        n_total = count_vsx_rows(vp)
    elif vsx_db_path is None:
        return pd.DataFrame()
    else:
        vp = Path(vsx_db_path).expanduser().resolve()
        if not vp.is_file():
            return pd.DataFrame()
        n_total = 0
    m = float(margin_px)
    w = float(width_px)
    h = float(height_px)
    # Pixel-space samples covering the frame border + margin (corners + edge midpoints).
    xs_px = np.asarray([-m, w * 0.5, w + m, -m, w + m, -m, w * 0.5, w + m], dtype=np.float64)
    ys_px = np.asarray([-m, -m, -m, h * 0.5, h * 0.5, h + m, h + m, h + m], dtype=np.float64)
    try:
        world = wcs.all_pix2world(xs_px, ys_px, 0)
        ras = np.asarray(world[0], dtype=np.float64)
        decs = np.asarray(world[1], dtype=np.float64)
    except Exception as exc:  # noqa: BLE001
        from except_fix_counters import get_except_fix_counters

        get_except_fix_counters().vsx_frame_bbox_wcs_fail += 1
        LOGGER.error("[VSX] frame-bbox WCS all_pix2world failed: %s", exc)
        return pd.DataFrame()
    ok = np.isfinite(ras) & np.isfinite(decs)
    if not bool(ok.any()):
        return pd.DataFrame()
    ras = ras[ok]
    decs = decs[ok]
    de_min = float(np.min(decs))
    de_max = float(np.max(decs))
    if center is not None:
        try:
            ra0 = float(center.icrs.ra.deg)
        except Exception:  # noqa: BLE001
            ra0 = float(np.median(ras))
    else:
        ra0 = float(np.median(ras))
    dra = ((ras - ra0 + 180.0) % 360.0) - 180.0
    ra_min = ra0 + float(np.min(dra))
    ra_max = ra0 + float(np.max(dra))
    pad = 1.0 / 3600.0  # 1 arcsec rounding pad
    rows = query_local_vsx(
        vp,
        ra_min=ra_min - pad,
        ra_max=ra_max + pad,
        dec_min=de_min - pad,
        dec_max=de_max + pad,
        max_rows=None,  # frame bbox is tiny -> no cap needed (spatial-first completeness)
    )
    if not rows:
        if require_db:
            log_event(
                f"VSX cone=0 on {vp} ({n_total} total rows) - field genuinely empty "
                f"(frame bbox+{int(m)}px, RA=[{ra_min:.4f},{ra_max:.4f}], Dec=[{de_min:.4f},{de_max:.4f}])"
            )
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "ra_deg" not in df.columns or "dec_deg" not in df.columns:
        if require_db:
            from catalog_db import VSXCatalogError

            raise VSXCatalogError(f"VSX query returned unexpected columns from {vp}.")
        return pd.DataFrame()
    try:
        log_event(
            f"CATALOG SEARCH (VSX local, frame bbox+{int(m)}px): {len(df)} sources "
            f"(RA=[{ra_min:.4f},{ra_max:.4f}], Dec=[{de_min:.4f},{de_max:.4f}], no cap)"
        )
    except Exception:  # noqa: BLE001
        pass
    return df


def _equipment_saturate_adu_from_db(equipment_id: int | None, cfg: AppConfig) -> float | None:
    """Read ``EQUIPMENTS.SATURATE_ADU`` (run database) when a valid equipment id is given."""
    if equipment_id is None:
        return None
    try:
        eid = int(equipment_id)
    except (TypeError, ValueError):
        return None
    if eid <= 0:
        return None
    try:
        db = VyvarDatabase.for_config(cfg)
        return db.get_equipment_saturation_adu(eid)
    except Exception as exc:  # noqa: BLE001 - optional source: the header / measured level decide then
        logging.warning("[SAT] equipment %s saturation not read from the database: %s", eid, exc)
        return None


def select_comparison_stars_spatial_grid(
    df: pd.DataFrame,
    *,
    width_px: float,
    height_px: float,
    n_comp: int = 0,
    require_non_variable: bool = True,
    safe_bbox: tuple[float, float, float, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Comparison pool for ensemble photometry: all usable, catalog-matched, photometry-ok, linear stars
    inside ``safe_bbox``, sorted by flux (brightest first), uncapped (COMP-POOL-01).

    G03-13: the former spatial-grid cap (``n_comp > 0``) had no production caller and was removed; the
    per-target comp choice happens later (photometry_comp). ``n_comp`` must be 0.

    When ``require_non_variable`` is True and column ``catalog_known_variable`` exists, stars flagged as
    known variables (VSX and/or the Gaia DR3 variability catalogue) are excluded from the ensemble.

    B4-2026-10 (G03-14): no fixed-angle proximity veto around VSX stars (10" was 1 px on one rig and
    18 px on another); light from neighbours - known variables at their maximum - is judged by the
    neighbour-light model (comp_blend.blend_contamination_for_pool, COMP_BLEND_NEIGHBOUR_LIGHT).
    """
    import numpy as np

    if int(n_comp) > 0:
        raise ValueError("n_comp > 0: the spatial-grid comp cap was removed (G03-13); the pool is uncapped")
    if df.empty:
        return pd.DataFrame(), {"n_selected": 0, "grid_nx": 0, "grid_ny": 0, "reason": "empty"}

    work = df.copy()
    if "is_usable" in work.columns:
        work = work[work["is_usable"].fillna(False).astype(bool)]
    if work.empty:
        return pd.DataFrame(), {"n_selected": 0, "grid_nx": 0, "grid_ny": 0, "reason": "no_rows_after_usable_filter"}
    # Forced-photometry rows (Gaia positions measured without a detection) never enter the pool.
    if "forced_photometry" in work.columns:
        _fp = work["forced_photometry"].astype(str).str.strip().str.lower()
        work = work[~_fp.isin(["true", "1", "yes"])]
    if work.empty:
        return pd.DataFrame(), {"n_selected": 0, "grid_nx": 0, "grid_ny": 0, "reason": "no_rows_after_seed_pool_filter"}
    if "catalog" in work.columns:
        work = work[work["catalog"].fillna("").astype(str).str.strip() != ""]
    if "photometry_ok" in work.columns:
        work = work[work["photometry_ok"].fillna(True).astype(bool)]
    if require_non_variable and "catalog_known_variable" in work.columns:
        work = work[~work["catalog_known_variable"].fillna(False).astype(bool)]
    # Annulus-aware border filter: keep only stars whose centroid is within the safe bbox
    # (intersection of aligned frames shrunk by sky-annulus outer radius).
    if safe_bbox is not None and "x" in work.columns and "y" in work.columns:
        from aperture_policy import stars_fit_on_chip  # noqa: PLC0415

        x0b, y0b, x1b, y1b = safe_bbox
        before = int(len(work))
        # 4-tuple naxis = precomputed safe bbox (already shrunk by resolver r_out).
        _on = stars_fit_on_chip(
            work["x"], work["y"], (0.0, 0.0, 0.0), (float(x0b), float(y0b), float(x1b), float(y1b))
        )
        work = work[np.asarray(_on, dtype=bool)]
        removed = before - int(len(work))
        if removed > 0:
            logging.info(
                f"[BORDER] Comp selection: removed {removed} candidates outside safe bbox "
                f"(annulus-aware intersection)"
            )
    if "likely_nonlinear" in work.columns:
        _ln = pd.to_numeric(work["likely_nonlinear"], errors="coerce").fillna(0).astype(int)
        work = work[_ln == 0]

    if work.empty:
        return pd.DataFrame(), {"n_selected": 0, "grid_nx": 0, "grid_ny": 0, "reason": "no_rows_after_filter"}

    w = float(width_px)
    h = float(height_px)
    if w <= 0 or h <= 0:
        w = float(work["x"].max()) + 1.0
        h = float(work["y"].max()) + 1.0

    if "flux" not in work.columns:
        work["flux"] = 0.0
    work["_flux_key"] = pd.to_numeric(work["flux"], errors="coerce").fillna(0.0)

    # COMP-POOL-01: uncapped pool; meta keys kept for pipeline_meta comparison_selection
    picked = work.sort_values("_flux_key", ascending=False).drop(
        columns=["_flux_key"], errors="ignore"
    )
    picked.insert(0, "comp_id", [f"COMP_{i+1:04d}" for i in range(len(picked))])
    picked.insert(1, "role", "comparison")
    meta = {
        "n_selected": int(len(picked)),
        "grid_nx": 0,
        "grid_ny": 0,
        "n_comp_requested": 0,
        "pool_uncapped": True,
        "width_px": float(w),
        "height_px": float(h),
        "require_non_variable": bool(require_non_variable),
        "exclude_nonlinear_badcolumn": True,
    }
    return picked, meta


COMP_POOL_CSV_COLUMNS: tuple[str, ...] = (
    "comp_id",
    "role",
    "catalog_id",
    "ra_deg",
    "dec_deg",
    "x",
    "y",
    "bp_rp",
    "phot_g_mean_mag",
)


def _plan_safe_bbox(*, _cfg_plan, aligned_files, aligned_ram_frames, hdr, ps, require_safe_bbox):
    """Comparison-safe bbox: intersection of the aligned frames shrunk by the sky-annulus outer radius (BORDER)."""
    import numpy as np  # noqa: PLC0415

    _safe_bbox: tuple[float, float, float, float] | None = None
    _r_out: float | None = None
    try:
        from aperture_policy import fwhm_from_header_vy_fwhm, resolve_aperture_geometry  # noqa: PLC0415

        _fwhm_px = fwhm_from_header_vy_fwhm(hdr)
        if _fwhm_px is None:
            raise ValueError("MASTERSTAR VY_FWHM missing; cannot resolve annulus r_out")
        _, _, _r_out = resolve_aperture_geometry(
            f=float(_cfg_plan.aperture_fwhm_factor),
            fwhm_px=float(_fwhm_px),
            annulus_inner_fwhm=float(_cfg_plan.annulus_inner_fwhm),
            annulus_outer_fwhm=float(_cfg_plan.annulus_outer_fwhm),
        )

        draft_root = ps.parent.parent
        aligned_dir = draft_root / "detrended_aligned" / "lights" / ps.name
        # BORDER-FOOTPRINT-2026-10 (Milan 2026-10-24): the border filter is the overlap of the data
        # footprints of ALL aligned frames (as AstroImageJ crops the common area), from each frame's
        # frame->reference transform VY_AM0..8. The former bbox of "finite pixels" never cut anything:
        # the aligner fills uncovered pixels with a constant, not NaN (MN Boo 538: frame 116 shifted
        # -80 px, bbox = whole chip). The aligned FITS keep the raw names (no proc_ prefix; the old
        # proc_*.fits glob found nothing - ALIGNED-GLOB-01). No dominant-group exception on a field jump.
        if aligned_files is not None:
            all_aligned = [Path(p) for p in aligned_files if Path(p).is_file()]
        else:
            all_aligned = sorted(aligned_dir.glob("*.fits"))
        ram_headers = [
            (_hdr, tuple(np.asarray(_arr).shape[:2]))
            for _name, _hdr, _arr in (aligned_ram_frames or [])
            if _hdr is not None and _arr is not None
        ]
        n_frame_sources = len(all_aligned) or len(ram_headers)
        if n_frame_sources == 0:
            if require_safe_bbox:
                raise RuntimeError(
                    f"[BORDER] Post-alignment border filter requires aligned frames but none were found in {aligned_dir}"
                )
            log_event("[BORDER] No aligned frames yet (before the alignment) - border filter computed after it")
        elif n_frame_sources < 2:
            if require_safe_bbox:
                raise RuntimeError(
                    f"[BORDER] Post-alignment border filter requires >=2 aligned frames, got {n_frame_sources}"
                )
            log_event(f"[BORDER] Not enough aligned frames for intersection bbox: {n_frame_sources}")
        else:
            from photometry_shared import (  # noqa: PLC0415
                common_field_intersection_bbox_px,
                common_field_intersection_bbox_px_from_headers,
            )

            if all_aligned:
                raw_bbox = common_field_intersection_bbox_px(frame_paths=all_aligned)
            else:
                raw_bbox = common_field_intersection_bbox_px_from_headers(headers=ram_headers)
            if raw_bbox is not None:
                log_event(
                    f"[BORDER] Overlap of the data footprints of all {n_frame_sources} aligned frames: "
                    f"x=[{raw_bbox[0]:.0f},{raw_bbox[2]:.0f}] y=[{raw_bbox[1]:.0f},{raw_bbox[3]:.0f}]"
                )

            if raw_bbox is None:
                if require_safe_bbox:
                    raise RuntimeError(
                        "[BORDER] Post-alignment border filter failed: the aligned frames have no common area "
                        f"({n_frame_sources} frames)"
                    )
                log_event("[BORDER] intersection bbox returned None - no border filter applied")
            else:
                x0, y0, x1, y1 = raw_bbox
                sb = (float(x0) + _r_out, float(y0) + _r_out, float(x1) - _r_out, float(y1) - _r_out)
                if sb[2] > sb[0] and sb[3] > sb[1]:
                    _safe_bbox = sb
                    log_event(
                        f"[BORDER] Safe bbox (shrunk by r_out={_r_out:.1f}px): "
                        f"x=[{_safe_bbox[0]:.0f},{_safe_bbox[2]:.0f}] y=[{_safe_bbox[1]:.0f},{_safe_bbox[3]:.0f}]"
                    )
                elif require_safe_bbox:
                    raise RuntimeError(
                        f"[BORDER] Post-alignment safe bbox invalid after shrink (r_out={_r_out:.1f}px)"
                    )
                else:
                    log_event(
                        f"[BORDER] Safe bbox invalid after shrink (r_out={_r_out:.1f}px); skipping border filter"
                    )
                    _safe_bbox = None
    except RuntimeError:
        raise
    except Exception as _bbox_exc:  # noqa: BLE001
        if require_safe_bbox:
            raise RuntimeError(f"[BORDER] safe_bbox computation failed: {_bbox_exc!s}") from _bbox_exc
        log_event(f"[BORDER] safe_bbox computation failed: {_bbox_exc!s} - skipping border filter")
        _safe_bbox = None
    return _r_out, _safe_bbox


def _plan_vsx_targets(*, _cfg_plan, df, h, hdr, wpx):
    """variable_targets.csv rows: VSX stars in the frame footprint, identified with Gaia (fails loud)."""
    from pipeline_catalog import _effective_field_catalog_cone_radius_deg  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    var_cols = [
        "name",
        "catalog_id",
        "catalog",
        "ra_deg",
        "dec_deg",
        "priority",
        "notes",
        "vsx_name",
        "vsx_type",
        "vsx_period",
        "x",
        "y",
        "mag",
        "zone",
        "gaia_match_arcsec",
        "gaia_match_quality",
        "gaia_match_source",
        "vsx_mag_max",
        "vsx_amplitude_mag",
        "exo_host_obj_id",
        "exo_host_name",
        "exo_cat_source",
        "exo_disposition",
        "exo_match_sep_arcsec",
        "target_origin",
    ]
    vsx_out = pd.DataFrame(columns=var_cols)
    _vsx_n_cone = 0
    _vsx_diag: dict[str, Any] = {}
    try:
        from catalog_db import VSXCatalogError, require_vsx_db

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            w0 = WCS(hdr)
        if not bool(getattr(w0, "has_celestial", False)):
            raise RuntimeError("MASTERSTAR WCS has no celestial axes.")
        if w0.has_celestial:
            center, radius_deg = _effective_field_catalog_cone_radius_deg(w0, int(h), int(wpx))
            _vsx_p3 = require_vsx_db(_cfg_plan.objects_db_path)
            # B-cap fix: variable_targets are driven by the FRAME footprint (bbox + the same
            # 50px margin used by the in-frame pixel filter below), not the 3.5 deg cone box. The
            # cone box hit the 15000-row catalog_query_max_rows cap (no ORDER BY) and silently
            # dropped a contiguous Dec slice (northern half of the field, incl. bright named
            # variables). The frame bbox is tiny -> no cap -> spatial-first completeness. The list
            # also gives the VSX amplitudes for the neighbour light of known variables (B4-2026-10).
            vsx_df = _query_vsx_local_frame_bbox(
                wcs=w0,
                width_px=int(wpx),
                height_px=int(h),
                vsx_db_path=_vsx_p3,
                margin_px=50.0,
                center=center,
                require_db=True,
            )
            n_vsx_in_cone = int(len(vsx_df)) if vsx_df is not None else 0
            _vsx_n_cone = n_vsx_in_cone

            # Masterstars catalogue (target-selection DAO membership only; not the VSX->Gaia match RHS).
            ga = df.copy()
            if "catalog_id" in ga.columns and "ra_deg" in ga.columns and "dec_deg" in ga.columns:
                ga["catalog_id"] = ga["catalog_id"].fillna("").astype(str).str.strip()
                ga = ga[ga["catalog_id"].ne("")].copy()
                ga["ra_deg"] = pd.to_numeric(ga["ra_deg"], errors="coerce")
                ga["dec_deg"] = pd.to_numeric(ga["dec_deg"], errors="coerce")
                ga = ga[ga["ra_deg"].notna() & ga["dec_deg"].notna()].copy()
            else:
                ga = ga.iloc[0:0].copy()
            from gaia_catalog_id import normalize_gaia_source_id  # noqa: PLC0415

            _ms_ids: set[str] = set()
            if not ga.empty:
                _ms_ids = {
                    str(normalize_gaia_source_id(x)).strip()
                    for x in ga["catalog_id"].astype(str).tolist()
                    if str(x).strip()
                }

            if vsx_df is not None and not vsx_df.empty and "ra_deg" in vsx_df.columns and "dec_deg" in vsx_df.columns:
                v = vsx_df.copy()
                v["ra_deg"] = pd.to_numeric(v["ra_deg"], errors="coerce")
                v["dec_deg"] = pd.to_numeric(v["dec_deg"], errors="coerce")
                v = v[v["ra_deg"].notna() & v["dec_deg"].notna()].copy()
                if not v.empty:
                    # Pixel coords from WCS (origin=0).
                    xy = w0.all_world2pix(v["ra_deg"].astype(float).to_numpy(), v["dec_deg"].astype(float).to_numpy(), 0)
                    x = np.asarray(xy[0], dtype=float)
                    y = np.asarray(xy[1], dtype=float)
                    v["x"] = x
                    v["y"] = y
                    in_frame = (v["x"] >= -50.0) & (v["y"] >= -50.0) & (v["x"] <= float(wpx) + 50.0) & (v["y"] <= float(h) + 50.0)
                    v = v.loc[in_frame].copy()

                n_vsx_in_frame = int(len(v))

                # Density-aware VSX -> Gaia DR3 cross-match (Marrese 2017/2019; Sutherland & Saunders 1992).
                # RHS is the deep local Gaia catalogue over the frame bbox; masterstars membership is derived after.
                cat_id_out = [""] * int(len(v))
                mag_out: list[float] = [float("nan")] * int(len(v))
                zone_out: list[str] = [""] * int(len(v))
                sep_out: list[float] = [float("nan")] * int(len(v))
                quality_out: list[str] = [""] * int(len(v))
                source_out: list[str] = [""] * int(len(v))
                _cm_diag: dict[str, Any] = {}
                if len(v) > 0:
                    from vsx_gaia_crossmatch import (  # noqa: PLC0415
                        VsxGaiaCrossmatchError,
                        catalog_depth_for_frame_bbox,
                        count_in_field,
                        field_area_deg2_from_wcs,
                        match_vsx_to_gaia_density_aware,
                        query_gaia_for_frame_bbox,
                    )

                    try:
                        gaia_db = str(_cfg_plan.gaia_db_path or "").strip()
                    except Exception:  # noqa: BLE001
                        gaia_db = ""
                    if not gaia_db:
                        raise RuntimeError("VSX->Gaia cross-match refused: gaia_db_path not configured")

                    _field_area = field_area_deg2_from_wcs(w0, int(wpx), int(h))
                    _catalog_depth_g = catalog_depth_for_frame_bbox(
                        gaia_db, w0, int(wpx), int(h), margin_px=50.0, center=center
                    )
                    _gaia_rows = query_gaia_for_frame_bbox(
                        gaia_db,
                        w0,
                        int(wpx),
                        int(h),
                        margin_px=50.0,
                        center=center,
                    )
                    _gdf = pd.DataFrame(_gaia_rows) if _gaia_rows else pd.DataFrame()
                    if _gdf.empty:
                        raise RuntimeError(
                            "VSX->Gaia cross-match refused: no Gaia DR3 sources in frame bbox"
                        )
                    _sid_col = "source_id" if "source_id" in _gdf.columns else "catalog_id"
                    _ra_col = "ra" if "ra" in _gdf.columns else "ra_deg"
                    _dec_col = "dec" if "dec" in _gdf.columns else "dec_deg"
                    _gdf = _gdf.copy()
                    _gdf["catalog_id"] = _gdf[_sid_col].apply(
                        lambda x: str(normalize_gaia_source_id(x)).strip()
                    )
                    _gdf["ra_deg"] = pd.to_numeric(_gdf[_ra_col], errors="coerce")
                    _gdf["dec_deg"] = pd.to_numeric(_gdf[_dec_col], errors="coerce")
                    _gdf = _gdf[
                        _gdf["catalog_id"].ne("")
                        & _gdf["ra_deg"].notna()
                        & _gdf["dec_deg"].notna()
                    ].copy()
                    _pmra = (
                        pd.to_numeric(_gdf["pmra"], errors="coerce").to_numpy(dtype=float)
                        if "pmra" in _gdf.columns
                        else None
                    )
                    _pmdec = (
                        pd.to_numeric(_gdf["pmdec"], errors="coerce").to_numpy(dtype=float)
                        if "pmdec" in _gdf.columns
                        else None
                    )
                    _vsx_mag_max = (
                        pd.to_numeric(v.get("mag_max"), errors="coerce").to_numpy(dtype=float)
                        if "mag_max" in v.columns
                        else None
                    )
                    try:
                        _match_rows, _cm_diag_obj = match_vsx_to_gaia_density_aware(
                            v["ra_deg"].astype(float).to_numpy(),
                            v["dec_deg"].astype(float).to_numpy(),
                            _gdf["catalog_id"].astype(str).to_numpy(),
                            _gdf["ra_deg"].astype(float).to_numpy(),
                            _gdf["dec_deg"].astype(float).to_numpy(),
                            field_area_deg2=float(_field_area),
                            n_gaia_in_field=count_in_field(
                                w0, int(wpx), int(h),
                                _gdf["ra_deg"].astype(float).to_numpy(),
                                _gdf["dec_deg"].astype(float).to_numpy(),
                            ),
                            pmra=_pmra,
                            pmdec=_pmdec,
                            catalog_depth_g=_catalog_depth_g,
                            vsx_mag_max=_vsx_mag_max,
                            masterstars_ids=_ms_ids,
                        )
                        _cm_diag = {
                            "q_fit": _cm_diag_obj.q_fit,
                            "w_fit": _cm_diag_obj.w_fit,
                            "sigma_narrow_arcsec": _cm_diag_obj.sigma_narrow_arcsec,
                            "sigma_broad_arcsec": _cm_diag_obj.sigma_broad_arcsec,
                            "sigma_arcsec": _cm_diag_obj.sigma_narrow_arcsec,
                            "rho_per_deg2": _cm_diag_obj.rho_per_deg2,
                            "mean_nn_arcsec": _cm_diag_obj.mean_nn_arcsec,
                            "pm_path": _cm_diag_obj.pm_path,
                            "pm_columns_present": _cm_diag_obj.pm_columns_present,
                            "n_pm_finite": _cm_diag_obj.n_pm_finite,
                            "n_vsx": _cm_diag_obj.n_vsx,
                            "n_gaia": _cm_diag_obj.n_gaia,
                            "n_accepted": _cm_diag_obj.n_accepted,
                            "expected_contamination_fraction": _cm_diag_obj.expected_contamination_fraction,
                            "r_max_arcsec": _cm_diag_obj.r_max_arcsec,
                            "candidate_multiplicity": dict(_cm_diag_obj.candidate_multiplicity),
                            "multi_candidate_fraction": _cm_diag_obj.multi_candidate_fraction,
                            "fit_degenerate_warn": _cm_diag_obj.fit_degenerate_warn,
                            "catalog_depth_g": _cm_diag_obj.catalog_depth_g,
                            "vsx_fainter_than_catalog": _cm_diag_obj.vsx_fainter_than_catalog,
                            "sep_quantiles_before_pm": _cm_diag_obj.sep_quantiles_before_pm,
                            "sep_quantiles_after_pm": _cm_diag_obj.sep_quantiles_after_pm,
                            "sep_quantiles_accepted": _cm_diag_obj.sep_quantiles_accepted,
                            "masterstars_in_frame": _cm_diag_obj.masterstars_in_frame,
                            "masterstars_eligible": _cm_diag_obj.masterstars_eligible,
                            "masterstars_accepted": _cm_diag_obj.masterstars_accepted,
                            "outcome_check": _cm_diag_obj.outcome_check,
                        }
                        _gaia_by_cid = {
                            str(r["catalog_id"]): r for _, r in _gdf.iterrows()
                        }
                        for i, mr in enumerate(_match_rows):
                            if mr.accepted and mr.catalog_id:
                                cat_id_out[i] = mr.catalog_id
                                sep_out[i] = float(mr.sep_arcsec)
                                quality_out[i] = mr.quality
                                if mr.catalog_id in _ms_ids:
                                    source_out[i] = "masterstars"
                                    gro = ga.loc[ga["catalog_id"].astype(str) == mr.catalog_id]
                                    if not gro.empty:
                                        gro = gro.iloc[0]
                                        try:
                                            _mg = gro.get("mag")
                                            mag_out[i] = (
                                                float(_mg)
                                                if _mg is not None and str(_mg).strip() != ""
                                                else float("nan")
                                            )
                                        except (TypeError, ValueError):
                                            mag_out[i] = float("nan")
                                        try:
                                            zr = gro.get("zone")
                                            zone_out[i] = str(zr).strip().lower() if zr is not None else ""
                                        except Exception:  # noqa: BLE001
                                            zone_out[i] = ""
                                else:
                                    source_out[i] = "gaia_dr3_direct"
                                    grow = _gaia_by_cid.get(mr.catalog_id)
                                    if grow is not None:
                                        try:
                                            _gm = grow.get("g_mag")
                                            mag_out[i] = (
                                                float(_gm)
                                                if _gm is not None and str(_gm).strip() != ""
                                                else float("nan")
                                            )
                                        except (TypeError, ValueError):
                                            mag_out[i] = float("nan")
                            else:
                                source_out[i] = "no_match"
                                try:
                                    vsx_name0 = str(v.iloc[i].get("name", "") or "").strip()
                                    vsx_ra = float(v.iloc[i]["ra_deg"])
                                    vsx_dec = float(v.iloc[i]["dec_deg"])
                                    log_event(
                                        f"VSX no Gaia match: {vsx_name0} ra={vsx_ra:.4f} dec={vsx_dec:.4f} "
                                        "- star will not be tracked"
                                    )
                                except Exception:  # noqa: BLE001
                                    pass
                    except VsxGaiaCrossmatchError as _cm_exc:
                        log_event(f"VSX->Gaia DR3 cross-match refused: {_cm_exc!s}")
                        raise RuntimeError(str(_cm_exc)) from _cm_exc

                # Period column varies by VSX schema.
                _pcol = None
                for c in ("period", "varperiod", "var_period", "Period", "VarPeriod", "Var_Period"):
                    if c in v.columns:
                        _pcol = c
                        break
                if _pcol is None:
                    v["vsx_period"] = np.nan
                else:
                    v["vsx_period"] = pd.to_numeric(v[_pcol], errors="coerce")

                vname = v.get("name", pd.Series([""] * len(v))).fillna("").astype(str).str.strip()
                vtype = v.get("var_type", pd.Series([""] * len(v))).fillna("").astype(str).str.strip()
                notes = []
                for t, p in zip(vtype.tolist(), v["vsx_period"].tolist(), strict=False):
                    t0 = str(t or "").strip()
                    p0 = float(p) if p is not None and pd.notna(p) else None
                    if p0 is not None:
                        notes.append(f"{t0} P={p0}")
                    else:
                        notes.append(t0)

                _mmx = pd.to_numeric(v.get("mag_max"), errors="coerce") if "mag_max" in v.columns else pd.Series(
                    [float("nan")] * len(v), dtype=float
                )
                vsx_out = pd.DataFrame(
                    {
                        "name": vname,
                        "catalog_id": cat_id_out,
                        "catalog": "VSX",
                        "ra_deg": v["ra_deg"].astype(float).to_numpy(),
                        "dec_deg": v["dec_deg"].astype(float).to_numpy(),
                        "priority": 1,
                        "notes": notes,
                        "vsx_name": vname,
                        "vsx_type": vtype,
                        "vsx_period": v["vsx_period"].to_numpy(),
                        "x": pd.to_numeric(v.get("x"), errors="coerce"),
                        "y": pd.to_numeric(v.get("y"), errors="coerce"),
                        "mag": np.asarray(mag_out, dtype=float),
                        "zone": zone_out,
                        "gaia_match_arcsec": np.asarray(sep_out, dtype=float),
                        "gaia_match_quality": quality_out,
                        "gaia_match_source": source_out,
                        "vsx_mag_max": _mmx.to_numpy(dtype=float),
                        # B4-2026-10: neighbour light of a known variable is judged at its maximum
                        "vsx_amplitude_mag": np.array(
                            [vsx_amplitude_mag(a, b) for a, b in zip(_mmx, v.get("mag_min", pd.Series(np.nan, index=v.index)), strict=True)],
                            dtype=float,
                        ),
                    }
                )
                n_gaia_ok = int(sum(1 for c in cat_id_out if str(c).strip()))
                _zone_hist: dict[str, int] = {}
                n_target_selection_hint = int(n_gaia_ok)
                for i in range(len(zone_out)):
                    if not str(cat_id_out[i]).strip():
                        continue
                    z = str(zone_out[i]).strip().lower()
                    _zone_hist[z] = int(_zone_hist.get(z, 0)) + 1
                _vsx_diag = {
                    "vsx_rows_in_frame_bbox": int(n_vsx_in_cone),
                    "vsx_rows_after_in_frame_margin": int(n_vsx_in_frame),
                    "n_with_masterstar_match": int(n_gaia_ok),
                    "vsx_rows_written_csv": int(len(vsx_out)),
                    "gaia_matches_within_10arcsec": int(n_gaia_ok),
                    "masterstars_zone_counts_among_gaia_matched": _zone_hist,
                    "target_selection_active_targets_hint_count": int(n_target_selection_hint),
                    "vsx_gaia_crossmatch": dict(_cm_diag),
                    "target_selection_note": (
                        "Target selection (`select_active_targets` in photometry_core.py): all zones from masterstars "
                        "go to active_targets.csv with `zone_flag`; saturated ones have `skip_photometry=True` "
                        "and aperture photometry does not measure them (see `run_aperture_photometry`). Only an empty "
                        "`catalog_id` and targets off the frame (border filter) are excluded."
                    ),
                }
                _vsx_n_cone = int(n_vsx_in_cone)
    except VSXCatalogError:
        raise
    except Exception as _vsx_exc:
        # FAIL-LOUD-2026-10 (code review 2 G03-06): no VSX targets means known variables can become comparison
        # stars - a failed VSX block stops the run with its cause instead of writing an empty variable_targets.csv
        raise RuntimeError(f"variable_targets.csv (VSX export) failed: {_vsx_exc!s}") from _vsx_exc
    return _vsx_diag, _vsx_n_cone, vsx_out


def _plan_write_comp_and_targets(
    *,
    _cfg_plan,
    _safe_bbox,
    _vsx_diag,
    _vsx_n_cone,
    comp_path,
    df,
    h,
    hdr,
    n_comparison_stars,
    require_non_variable,
    var_path,
    vsx_out,
    wpx,
):
    """Exoplanet promotions, comparison pool and the comparison / variable-target CSVs."""
    from pipeline_catalog import _build_exoplanet_promotion_rows_from_masterstars  # noqa: PLC0415

    exo_promo = _build_exoplanet_promotion_rows_from_masterstars(
        df,
        hdr,
        _cfg_plan,
        frame_w_px=int(wpx),
        frame_h_px=int(h),
        margin_px=50.0,
    )
    merged_var = _merge_vsx_exoplanet_variable_targets(vsx_out, exo_promo)
    comp_df, cmeta = select_comparison_stars_spatial_grid(
        df,
        width_px=float(wpx),
        height_px=float(h),
        n_comp=int(n_comparison_stars),
        require_non_variable=bool(require_non_variable),
        safe_bbox=_safe_bbox,
    )
    try:
        from gaia_catalog_id import normalize_gaia_source_id_series  # noqa: PLC0415

        if "catalog_id" in comp_df.columns:
            comp_df = comp_df.copy()
            comp_df["catalog_id"] = normalize_gaia_source_id_series(comp_df["catalog_id"])
    except Exception:  # noqa: BLE001
        pass
    # OUTPUT-REVIEW-2026-10 step 5: the pool file carries what its reader (the colour-term fit)
    # takes, plus the position; the rest of the MASTERSTAR row is in masterstars_full_match.csv.
    _vyvar_df_to_csv(comp_df[[c for c in COMP_POOL_CSV_COLUMNS if c in comp_df.columns]], comp_path)

    # Always overwrite (even if it exists) so UI sees current field cone + exo promotions.
    try:
        from gaia_catalog_id import normalize_gaia_source_id_series  # noqa: PLC0415

        if "catalog_id" in merged_var.columns:
            merged_var = merged_var.copy()
            merged_var["catalog_id"] = normalize_gaia_source_id_series(merged_var["catalog_id"])
    except Exception:  # noqa: BLE001
        pass
    _vyvar_df_to_csv(merged_var, var_path)
    if _vsx_diag:
        log_event(
            "variable_targets.csv (VSX): "
            f"VSX in frame bbox={_vsx_diag.get('vsx_rows_in_frame_bbox')} -> "
            f"in-frame margin={_vsx_diag.get('vsx_rows_after_in_frame_margin')} -> "
            f"detection-limited (DAO+Gaia match)={_vsx_diag.get('n_with_masterstar_match')} -> "
            f"CSV={int(len(merged_var))}. "
            f"Estimated VSX with Gaia ID (target selection then cross-matches to masterstars): {_vsx_diag.get('target_selection_active_targets_hint_count')}."
        )
    else:
        log_event(
            f"variable_targets.csv (VSX export): cone={_vsx_n_cone} -> written={int(len(merged_var))} "
            f"({var_path.name})"
        )
    return cmeta,


def write_photometry_plan_files(
    *,
    platesolve_dir: Path,
    masterstar_fits: Path,
    masterstars_csv: Path,
    n_comparison_stars: int = 0,
    require_non_variable: bool = True,
    cfg: AppConfig,
    aligned_files: list[Path] | None = None,
    aligned_ram_frames: list[tuple[str, Any, Any]] | None = None,
    require_safe_bbox: bool = False,
    enrichment_expected: bool = True,
) -> dict[str, Any]:
    """Write ``comparison_stars.csv`` from ``masterstars.csv`` + image size; stub ``variable_targets.csv``."""

    import json


    ps = Path(platesolve_dir)
    ps.mkdir(parents=True, exist_ok=True)

    if not masterstars_csv.is_file():
        return {"comparison_stars_csv": "", "variable_targets_csv": "", "error": "missing masterstars.csv"}

    # Read IDs as strings (avoid float64 precision loss for 19-digit Gaia IDs).
    df = pd.read_csv(masterstars_csv, low_memory=False, dtype={"catalog_id": str, "name": str})
    try:
        from gaia_catalog_id import catalog_id_series_for_masterstars_export  # noqa: PLC0415

        if "catalog_id" in df.columns:
            df = df.copy()
            df["catalog_id"] = catalog_id_series_for_masterstars_export(df)
    except Exception:  # noqa: BLE001
        pass

    _enrichment_provenance: dict[str, Any] = {"source": "masterstars_full_match.csv"}
    try:
        from masterstars_enrichment import missing_comp_selection_enrichment_columns  # noqa: PLC0415

        _missing_enrich = missing_comp_selection_enrichment_columns(df)
        if _missing_enrich:
            _msg = (
                "comp-selection enrichment unavailable for this draft: columns "
                f"{_missing_enrich} absent from masterstars_full_match.csv - "
                "selection proceeds without nonlinearity/bad-column exclusion"
            )
            if enrichment_expected:
                logging.error("[COMP-SELECTION] %s", _msg)
                log_event(f"ERROR: {_msg}")
            else:
                # First plan pass inside MASTERSTAR build runs before the enrichment
                # columns are attached; the later passes carry them.
                logging.debug("[COMP-SELECTION] (pre-enrichment pass) %s", _msg)
            _enrichment_provenance = {
                "source": "masterstars_full_match.csv",
                "enrichment_missing_columns": list(_missing_enrich),
                "nonlinearity_badcolumn_exclusion": False,
                "message": _msg,
            }
        else:
            _enrichment_provenance["nonlinearity_badcolumn_exclusion"] = True
    except Exception as _enrich_exc:  # noqa: BLE001
        logging.error("[COMP-SELECTION] enrichment column check failed: %s", _enrich_exc)
    try:
        with fits.open(masterstar_fits, memmap=False) as hdul:
            hdr = hdul[0].header
    except Exception as _ms_open_exc:  # noqa: BLE001
        log_event(f"write_photometry_plan_files: could not open MASTERSTAR.fits ({masterstar_fits}): {_ms_open_exc!s}")
        return {"comparison_stars_csv": "", "variable_targets_csv": "", "error": "MASTERSTAR open failed"}
    wpx = int(hdr.get("NAXIS1", 0) or 0)
    h = int(hdr.get("NAXIS2", 0) or 0)
    if wpx <= 0 or h <= 0:
        return {"comparison_stars_csv": "", "error": "MASTERSTAR has no data"}

    _cfg_plan = cfg

    comp_path = ps / "comparison_stars.csv"
    var_path = ps / "variable_targets.csv"

    # --- Annulus-aware intersection bbox (Variant A2) ---
    # Compute a "safe" bbox from the intersection of aligned frames, shrunk by the sky-annulus outer radius.
    # CONSOLIDATE-01B A2: FWHM source stays MASTERSTAR VY_FWHM (qc vs header differ on all
    # 134 era04 frames at FITS-card rounding; night-median qc vs MASTERSTAR is also not
    # last-ulp identical). Do not invent 3.5 / 10.5.
    _r_out, _safe_bbox = _plan_safe_bbox(
        _cfg_plan=_cfg_plan,
        aligned_files=aligned_files,
        aligned_ram_frames=aligned_ram_frames,
        hdr=hdr,
        ps=ps,
        require_safe_bbox=require_safe_bbox,
    )

    # VSX variable targets for the field (frame bbox), with pixel coords from MASTERSTAR WCS.
    _vsx_diag, _vsx_n_cone, vsx_out = _plan_vsx_targets(_cfg_plan=_cfg_plan, df=df, h=h, hdr=hdr, wpx=wpx)

    cmeta, = _plan_write_comp_and_targets(
        _cfg_plan=_cfg_plan,
        _safe_bbox=_safe_bbox,
        _vsx_diag=_vsx_diag,
        _vsx_n_cone=_vsx_n_cone,
        comp_path=comp_path,
        df=df,
        h=h,
        hdr=hdr,
        n_comparison_stars=n_comparison_stars,
        require_non_variable=require_non_variable,
        var_path=var_path,
        vsx_out=vsx_out,
        wpx=wpx,
    )

    plan_path = ps / "photometry_plan.json"
    # OUTPUT-REVIEW-2026-10: the plan keeps only what is read back (target and comparison-star selection
    # and aperture photometry: the border-safe box); the comparison selection and VSX diagnostics are in the log / return value.
    plan = {
        "safe_bbox_px": list(_safe_bbox) if _safe_bbox is not None else None,
        "safe_bbox_r_out_px": float(_r_out) if _r_out is not None else None,
    }
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    return {
        "comparison_stars_csv": str(comp_path),
        "variable_targets_csv": str(var_path),
        "photometry_plan_json": str(plan_path),
        "comparison_selection": cmeta,
        "comp_selection_enrichment": _enrichment_provenance,
    }


def _fill_masterstars_gaia_matched_bp_rp_from_local_db(
    df: pd.DataFrame,
    *,
    gaia_db_path: str,
) -> tuple[pd.DataFrame, int, int]:
    """Fill ``bp_rp`` / ``b_v`` from the local Gaia SQLite for ``GAIA_MATCHED`` rows without colour in masterstars CSV.

    The frame-wide Gaia query (``ORDER BY g_mag LIMIT``) often misses stars already matched on the frame;
    this batch goes directly by ``source_id``.
    """
    if df is None or getattr(df, "empty", True):
        return df, 0, 0
    need = {"catalog_id", "bp_rp", "source_type"}
    if not need.issubset(df.columns):
        return df, 0, 0

    st_ok = df["source_type"].astype(str).str.strip().eq("GAIA_MATCHED")
    bpr = pd.to_numeric(df["bp_rp"], errors="coerce")
    gid = df["catalog_id"].map(normalize_gaia_source_id)
    gid_ok = gid.ne("")
    mask = st_ok & bpr.isna() & gid_ok
    n_missing = int(mask.sum())
    if n_missing <= 0:
        return df, 0, 0

    gdb = str(gaia_db_path or "").strip()
    try:
        gdb_ok = bool(gdb) and Path(gdb).is_file()
    except OSError:
        gdb_ok = False
    if not gdb_ok:
        return df, 0, n_missing

    out = df.copy()

    sub_idx = out.index[mask]
    keys_series = gid.loc[sub_idx]
    uniq_keys = sorted({k for k in keys_series.tolist() if k})
    if not uniq_keys:
        return out, 0, n_missing

    gaia_map = query_local_gaia_by_source_ids(gdb, uniq_keys)
    bprp_raw = keys_series.map(lambda k: (gaia_map.get(k) or {}).get("bp_rp"))
    bprp_num = pd.to_numeric(bprp_raw, errors="coerce")
    fill_ok = bprp_num.notna()
    to_fill = bprp_num.index[fill_ok]
    if len(to_fill) > 0:
        out.loc[to_fill, "bp_rp"] = bprp_num.loc[to_fill].astype(float)
    n_filled = int(fill_ok.sum())
    return out, n_filled, n_missing


#: effective wavelength (nm) per filter token, to pick the spectrally nearest solved sibling filter
FILTER_EFFECTIVE_WAVELENGTH_NM: dict[str, float] = {
    "u": 354.0,
    "g": 477.0,
    "r": 623.0,
    "i": 762.0,
    "z": 913.0,
    "y": 1035.0,
    "b": 440.0,
    "v": 551.0,
    "clear": 550.0,
    "none": 550.0,
    "nofilter": 550.0,
}


def filter_code_from_setup_name(setup: str) -> str:
    """Filter token of a setup folder name (``g_60_4`` -> ``g``)."""
    s = str(setup or "").strip()
    if not s or s.casefold() == "(root)":
        return ""
    return s.split("_")[0].strip().lower()


def pick_sibling_donor_setup(recipient_setup: str, solved_setups: Sequence[str]) -> str | None:
    """The solved setup to hint a failed one (G03-35: donors keyed by setup, not by filter - a
    failed g_120_2 can use the solved g_60_2): the same filter first, then the spectrally nearest
    filter; ties by setup name. The recipient itself is never its own donor."""
    rec = str(recipient_setup or "").strip()
    cands = sorted({str(x).strip() for x in solved_setups if str(x).strip() and str(x).strip() != rec})
    if not cands:
        return None
    f_r = filter_code_from_setup_name(rec)
    lam_r = float(FILTER_EFFECTIVE_WAVELENGTH_NM.get(f_r, 550.0))

    def _key(setup: str) -> tuple[int, float, str]:
        f = filter_code_from_setup_name(setup)
        return (0 if f == f_r else 1, abs(float(FILTER_EFFECTIVE_WAVELENGTH_NM.get(f, 550.0)) - lam_r), setup)

    return min(cands, key=_key)


