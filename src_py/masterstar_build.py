"""Moved from pipeline.py (CONSOLIDATE-01E6b). Facade re-exports this name.

generate_masterstar_and_catalog: masterstar FITS build, plate-solve,
comparison-star sync, VSX/exo merge.
"""
from __future__ import annotations

import logging
import math
import warnings
from pathlib import Path
from typing import Any, Sequence

from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.wcs import FITSFixedWarning
import astropy.units as u
import pandas as pd

from config import AppConfig, resolve_detection_threshold_sigma
from database import VyvarDatabase
from infolog import log_event
from optics_selection import resolve_optics_ids_for_platesolve
from gaia_catalog_id import read_vyvar_csv
from aperture_policy import clamp_fwhm_px, fwhm_from_header_vy_fwhm
from photometry_provenance import merge_photometry_pipeline_meta
from dao_reconcile import reconcile_to_pipeline_meta
from optics import plate_scale_arcsec_per_px_from_wcs, report_scale_mismatch
from pipeline_astrometry import _fill_masterstars_gaia_matched_bp_rp_from_local_db
from pipeline_constants import (
    DAO_FWHM_FALLBACK_PX,
    LINEARITY_FRAC,
    SAT_LIMIT_CONTAINER_CLIP_ADU,
    SAT_LIMIT_NO_KNEE_FRAC,
    _MASTERSTAR_SOLVER_USE_DRAFT_MEDIAN_IF_HINT_SEP_DEG,
)
from utils import (
    iter_fits_paths_recursive as _iter_fits_recursive,
    fits_binning_xy_from_header,
)
from fits_meta import pointing_hint_from_header as _pointing_hint_from_header
from pipeline_calibrate import (
    _effective_saturation_limit,
    _has_valid_wcs,
    draft_median_pointing_icrs_deg,
)
from pipeline_preprocess import resolve_obs_file_to_processed_fits
from pipeline_astrometry import (
    _equipment_saturate_adu_from_db,
    _plate_solve_input_bundle,
    _update_masterstar_obs_file_status,
    _vyvar_open_database,
    build_masterstar_from_detrended,
    compute_plate_scale_from_db,
    draft_is_multi_group_obs,
    get_masterstar_candidate_rows,
    get_masterstar_candidates,
    write_photometry_plan_files,
)
from pipeline_preprocess import (
    _path_is_under_tree,
    _path_segments_forbidden_for_masterstar_physical_source,
    _resolve_best_effort_path_under,
    fits_basename_index,
    resolve_masterstar_input_root,
)
from proc_frame_store import _vyvar_df_to_csv
from pipeline_catalog import (
    _annotate_masterstars_flux_zones,
    _effective_field_catalog_cone_radius_deg,
    _invalidate_field_catalog_cone_cache_if_needed,
)
from pipeline_constants import CATALOG_MATCH_SEP_ARCSEC, MASTERSTAR_MAX_DAO_ROWS  # noqa: E402

LOGGER = logging.getLogger("pipeline")


def _frame_matching_depth_g(det_meta: dict[str, Any]) -> float:
    """Depth of the DAO <-> Gaia matching phase: the frame limit capped by the catalogue
    (FRAME-DEPTH-2026-10); without a measured limit the catalogue depth itself."""
    fd = det_meta.get("frame_depth") or {}
    for key in ("g_lim", "db_depth_g"):
        v = fd.get(key)
        try:
            if v is not None and math.isfinite(float(v)):
                return float(v)
        except (TypeError, ValueError):
            continue
    return float("inf")


def _store_masterstar_path(cfg: Any, draft_id: int | None, path: str, out: dict[str, Any]) -> None:
    """Record the MASTERSTAR path on the draft; a DB error goes to ``out['masterstar_path_store_error']``."""
    if draft_id is None:
        return
    try:
        db = VyvarDatabase.for_config(cfg)
        try:
            db.set_obs_draft_masterstar_fits_path(int(draft_id), path)
        finally:
            db.conn.close()
    except Exception as exc:  # noqa: BLE001
        out["masterstar_path_store_error"] = str(exc)


def _ms_measure_draft_saturation(*, ap, equipment_saturate_adu):
    """SAT-DIAG: saturation level measured on the draft's frames; may replace the equipment value."""
    _ms_measured_sat: float | None = None
    try:
        from sat_diag import run_sat_diag  # noqa: PLC0415

        _sd_ctx_ms = run_sat_diag(ap, equipment_adu=equipment_saturate_adu)
        _ms_measured_sat = _sd_ctx_ms.measured_sat_adu()
        if _sd_ctx_ms.sat_adu is not None and _sd_ctx_ms.may_exclude_saturation():
            equipment_saturate_adu = float(_sd_ctx_ms.sat_adu)
        logging.info(
            "[SAT-DIAG] MASTERSTAR saturation level %s ADU (source %s, measured=%s)",
            _sd_ctx_ms.sat_adu, _sd_ctx_ms.sat_source, _ms_measured_sat is not None,
        )
    except Exception as _sd_ms_exc:  # noqa: BLE001
        logging.warning("[SAT-DIAG] MASTERSTAR saturation level not measured: %s", _sd_ms_exc)
    return _ms_measured_sat, equipment_saturate_adu


def _ms_build_or_load_masterstar(
    *,
    ap,
    app_config,
    draft_id,
    masterstar_basename,
    masterstar_candidate_paths,
    masterstar_selection_pct,
    masterstar_skip_build,
    platesolve_dir,
    setup_name,
    source_root,
):
    """MASTERSTAR.fits: the existing file (platesolve-from-disk) or a copy of the best candidate frame."""
    import numpy as np  # noqa: PLC0415

    detrended_root: Path | None = None
    if masterstar_skip_build:
        ps = Path(platesolve_dir) if platesolve_dir is not None else (ap / "platesolve")
        platesolve_dir = ps
        platesolve_dir.mkdir(parents=True, exist_ok=True)
        _ms_name = str(masterstar_basename or "MASTERSTAR.fits").strip() or "MASTERSTAR.fits"
        masterstar_fits = Path(platesolve_dir) / _ms_name
        if not masterstar_fits.is_file():
            raise FileNotFoundError(
                f"MASTERSTAR plate-solve: in {platesolve_dir} the file {_ms_name} is missing. "
                "First run **MAKE MASTERSTAR** on the archive or create MASTERSTAR another way (FITS QA -> reference frame)."
            )
        _match_sep_eff = CATALOG_MATCH_SEP_ARCSEC
        log_event(
            f"MASTERSTAR platesolve-from-disk: {masterstar_fits.resolve()} - VYVAR solver + catalog "
            "(no new build from processed)."
        )
        ms_selection_meta = {
            "source": "platesolve_existing",
            "file": str(masterstar_fits.resolve()),
        }
        try:
            _ms_resolved = str(masterstar_fits.resolve())
        except OSError:
            _ms_resolved = str(masterstar_fits)
        info = {
            "masterstar_path": _ms_resolved,
            "frames_used": 1,
            "reference_path": _ms_resolved,
        }
    if not masterstar_skip_build:
        # MASTERSTAR-only reads from processed/lights/setup_name (robust folder-based discovery).
        if source_root is not None:
            detrended_root = Path(source_root)
        else:
            detrended_root = resolve_masterstar_input_root(
                ap,
                setup_name=setup_name,
                draft_id=draft_id,
            )
            if detrended_root is None:
                raise FileNotFoundError(
                    f"MASTERSTAR input root for setup {str(setup_name)!r} not found under "
                    f"{ap / 'processed' / 'lights'} (refusing cross-group fallback)."
                )
        if not detrended_root.exists():
            if setup_name:
                raise FileNotFoundError(
                    f"MASTERSTAR input root for setup {str(setup_name)!r} not found: {detrended_root}"
                )
            log_event(f"[X] MASTERSTAR FAIL: Input path {detrended_root} not found.")
            processed_lights = ap / "processed" / "lights"
            if processed_lights.is_dir():
                subdirs = sorted(
                    [d for d in processed_lights.iterdir() if d.is_dir()],
                    key=lambda p: p.name.casefold(),
                )
                if subdirs:
                    detrended_root = subdirs[0]
                    log_event(f"[OK] MASTERSTAR fallback input found: {detrended_root}")
        if not detrended_root.exists():
            raise FileNotFoundError(f"Missing processed/detrended lights: {detrended_root}")
        # If root exists but has no FITS, try first setup subfolder under processed lights (single-group only).
        if not setup_name and not _iter_fits_recursive(detrended_root):
            processed_lights = ap / "processed" / "lights"
            if processed_lights.is_dir():
                subdirs = sorted(
                    [d for d in processed_lights.iterdir() if d.is_dir()],
                    key=lambda p: p.name.casefold(),
                )
                for sd in subdirs:
                    if _iter_fits_recursive(sd):
                        log_event(f"[OK] MASTERSTAR fallback to setup subdir: {sd}")
                        detrended_root = sd
                        break

        log_event(f"[search] MASTERSTAR: Searching for candidates in {Path(detrended_root).resolve()}")
        log_event(f"Input folder for Masterstar:{Path(detrended_root).resolve()}")
        from draft_provenance import is_pre_calibrated_draft

        _pre_cal_ms = is_pre_calibrated_draft(ap, draft_id=draft_id)
        if _pre_cal_ms:
            log_event(
                "MASTERSTAR: pre-calibrated draft - candidates resolved directly under "
                f"{Path(detrended_root).resolve()} (no processed/calibrated remap)."
            )
        _match_sep_eff = CATALOG_MATCH_SEP_ARCSEC

        ps = Path(platesolve_dir) if platesolve_dir is not None else (ap / "platesolve")
        platesolve_dir = ps
        platesolve_dir.mkdir(parents=True, exist_ok=True)
        _ms_name = str(masterstar_basename or "MASTERSTAR.fits").strip() or "MASTERSTAR.fits"
        masterstar_fits = Path(platesolve_dir) / _ms_name
        only_ms_paths: list[Path] | None = None
        ms_selection_meta: dict[str, Any] = {}
        #: When True, ``masterstar_candidate_paths`` mapped to disk - do not append unrelated FITS
        #: for "best-of-N" pool (that would override a deliberate single-frame pick in the UI).
        explicit_ui_masterstar_paths = False

        def _map_qc_paths_to_disk(raw_paths: list[str]) -> list[Path]:
            """Map UI / DB paths onto draft lights FITS under ``detrended_root``.

            Pre-calibrated: match by basename under ``non_calibrated/lights/<setup>/``.
            VYVAR-calibrated: prefer ``processed/lights/.../proc_*.fits`` via remap helpers.
            """

            def _mapped_hit_ok(hit: Path) -> bool:
                if not hit.is_file() or _path_segments_forbidden_for_masterstar_physical_source(
                    hit, pre_calibrated=_pre_cal_ms
                ):
                    return False
                if _pre_cal_ms:
                    return _path_is_under_tree(Path(detrended_root), hit)
                pl = ap / "processed" / "lights"
                if pl.is_dir():
                    try:
                        hit.resolve().relative_to(pl.resolve())
                        return True
                    except ValueError:
                        return False
                return _path_is_under_tree(Path(detrended_root), hit)

            out: list[Path] = []
            _bn_index = fits_basename_index(Path(detrended_root), pre_calibrated=_pre_cal_ms)  # G03-41: one scan
            for rp in raw_paths:
                s = str(rp).strip()
                if not s:
                    continue
                hit = _resolve_best_effort_path_under(
                    Path(detrended_root),
                    s,
                    pre_calibrated=_pre_cal_ms,
                    index=_bn_index,
                )
                if hit is not None and _mapped_hit_ok(hit):
                    out.append(hit)
                    continue
                if _pre_cal_ms:
                    continue
                try:
                    hit2 = resolve_obs_file_to_processed_fits(
                        ap,
                        s,
                        setup_name=setup_name,
                        draft_id=draft_id,
                    )
                except Exception:  # noqa: BLE001
                    hit2 = None
                if hit2 is not None and _mapped_hit_ok(hit2):
                    out.append(hit2)
            return out

        def _disk_stack_fallback_paths(input_dir: Path, *, max_frames: int = 8) -> list[Path]:
            """When QC paths / DB mapping fail: pick best frames from disk (deterministic order)."""
            all_on_disk = sorted(
                (
                    fp
                    for fp in _iter_fits_recursive(input_dir)
                    if _path_is_under_tree(input_dir, fp)
                    and not _path_segments_forbidden_for_masterstar_physical_source(
                        fp, pre_calibrated=_pre_cal_ms
                    )
                ),
                key=lambda p: str(p).casefold(),
            )
            if not all_on_disk:
                return []
            n = max(1, min(int(max_frames), len(all_on_disk)))
            return all_on_disk[:n]

        try:
            _pct_eff = float(masterstar_selection_pct) if masterstar_selection_pct is not None else 10.0
        except (TypeError, ValueError):
            _pct_eff = 10.0
        if not math.isfinite(_pct_eff) or _pct_eff <= 0:
            _pct_eff = 10.0
        _pct_eff = max(0.1, min(100.0, _pct_eff))

        cand_paths = [str(x) for x in (masterstar_candidate_paths or []) if str(x).strip()]
        if cand_paths:
            mapped = _map_qc_paths_to_disk(cand_paths)
            if mapped:
                only_ms_paths = mapped
                explicit_ui_masterstar_paths = True
                ms_selection_meta = {
                    "source": "ui_paths",
                    "requested": int(len(cand_paths)),
                    "mapped_found": int(len(mapped)),
                    "explicit_ui_lock": True,
                }
            else:
                raise FileNotFoundError(
                    "MASTERSTAR: UI/job passed explicit reference-frame paths, but none was found "
                    f"as ``processed/lights/.../proc_*.fits`` (selection root: {Path(detrended_root).resolve()}). "
                    "Check preprocess, archive and the FITS QA selection (confirm again after **RUN VYVAR**). "
                    f"Requested ({len(cand_paths)}): " + "; ".join(cand_paths[:6]) + (" ..." if len(cand_paths) > 6 else "")
                )

        _multi_obs = draft_is_multi_group_obs(ap)
        if only_ms_paths is None:
            # QC-PER-GROUP-2026-10: best frame of this group from frame QC (frame_qc.json).
            from frame_qc import best_path_for_setup  # noqa: PLC0415

            _qc_best = best_path_for_setup(ap, setup_name)
            if _qc_best is None and not _multi_obs:
                _qc_best = best_path_for_setup(ap, None)
            if _qc_best:
                mapped_qc = _map_qc_paths_to_disk([_qc_best])
                if mapped_qc:
                    only_ms_paths = mapped_qc[:1]
                    explicit_ui_masterstar_paths = True
                    ms_selection_meta = {
                        "source": "frame_qc_best_of_group",
                        "setup": str(setup_name or ""),
                        "frame": Path(_qc_best).name,
                    }
                    log_event(
                        f"MASTERSTAR: frame QC best of group {setup_name or '(single)'} -> {mapped_qc[0].name}"
                    )
                else:
                    log_event(
                        f"MASTERSTAR: frame QC best {Path(_qc_best).name} not found under {detrended_root}"
                    )
        if only_ms_paths is None and draft_id is not None and not (_multi_obs and setup_name):
            _db_ms = _vyvar_open_database(app_config)
            if _db_ms is not None:
                try:
                    # FITS QA 'Confirm MASTERSTAR selection' -> ``get_obs_draft_masterstar_source_path``
                    # (``draft manifest.MASTERSTAR_PATH`` = source frame, not the finished ``MASTERSTAR.fits``).
                    # Must run before the automatic top-% from ``manifest files[]``, else the user's choice is overwritten.
                    _src = _db_ms.get_obs_draft_masterstar_source_path(int(draft_id))
                    if _src and str(_src).strip():
                        mapped_src = _map_qc_paths_to_disk([str(_src).strip()])
                        if mapped_src:
                            only_ms_paths = mapped_src
                            explicit_ui_masterstar_paths = True
                            ms_selection_meta = {
                                "source": "db_masterstar_source_path",
                                "draft_id": int(draft_id),
                                "mapped_found": int(len(mapped_src)),
                                "explicit_ui_lock": True,
                            }
                            log_event(
                                f"MASTERSTAR: FITS QA selection (DB source, draft {int(draft_id)}) -> {mapped_src[0].name}"
                            )
                    if only_ms_paths is None:
                        db_paths = get_masterstar_candidates(int(draft_id), _pct_eff, db=_db_ms)
                        mapped_db = _map_qc_paths_to_disk([str(x) for x in db_paths if str(x).strip()])
                        if mapped_db:
                            only_ms_paths = mapped_db
                            ms_selection_meta = {
                                "source": "db_top_pct",
                                "draft_id": int(draft_id),
                                "pct": float(_pct_eff),
                                "mapped_found": int(len(mapped_db)),
                            }
                            log_event(
                                f"MASTERSTAR: selection from DB (draft {int(draft_id)}, top {_pct_eff:g} %) -> "
                                f"{len(mapped_db)} candidates (the best is copied to platesolve)."
                            )
                        else:
                            log_event(
                                f"MASTERSTAR: DB selection (draft {int(draft_id)}) could not be mapped to FITS under {detrended_root}."
                            )
                except Exception as exc:  # noqa: BLE001
                    logging.error("MASTERSTAR: DB candidate selection failed: %s", exc)
                    log_event(f"MASTERSTAR: DB candidate selection failed ({exc!s}).")
                finally:
                    try:
                        _db_ms.conn.close()
                    except Exception:  # noqa: BLE001
                        pass

        if only_ms_paths is None:
            disk_batch = _disk_stack_fallback_paths(Path(detrended_root), max_frames=8)
            if disk_batch:
                log_event(
                    "! MASTERSTAR: no frame QC result for this group - falling back to the first "
                    "frames on disk (not quality ranked)."
                )
                only_ms_paths = disk_batch
                ms_selection_meta = {
                    "source": "disk_fallback_stack",
                    "mapped_found": int(len(disk_batch)),
                }
                log_event(
                    f"MASTERSTAR disk fallback: {len(disk_batch)} candidates from disk (no valid QC selection)."
                )

        if only_ms_paths is None:
            raise FileNotFoundError(
                f"MASTERSTAR: no FITS to select from in {detrended_root}, even after UI/DB."
            )

        _cfg_stack = app_config
        # G03-29: the night median QC FWHM is the only fallback for the MASTERSTAR FWHM (no constant)
        _ms_fwhm_fb: float | None = None
        if draft_id is not None:
            _dbc_fw = _vyvar_open_database(_cfg_stack)
            if _dbc_fw is not None:
                try:
                    _fdf = get_masterstar_candidate_rows(int(draft_id), 100.0, db=_dbc_fw)
                    if _fdf is not None and not _fdf.empty and "FWHM" in _fdf.columns:
                        _vals = [v for v in (clamp_fwhm_px(x) for x in _fdf["FWHM"].tolist()) if v is not None]
                        if _vals:
                            _ms_fwhm_fb = float(np.median(_vals))
                except Exception:  # noqa: BLE001
                    pass
                finally:
                    try:
                        _dbc_fw.conn.close()
                    except Exception:  # noqa: BLE001
                        pass

        # Build MASTERSTAR with best-of-N fallback: try a few top candidates if build/selection is brittle.
        try:
            _best_n = int(float(_cfg_stack.masterstar_best_of_n))
        except (TypeError, ValueError):
            _best_n = 10
        _best_n = max(1, min(25, int(_best_n)))
        _cand_all = [Path(p) for p in (only_ms_paths or []) if Path(p).is_file()]
        # If UI/DB mapping yields too few candidates, expand from disk for best-of-N robustness -
        # but never when the user explicitly passed ``masterstar_candidate_paths`` (would replace e.g.
        # a single chosen frame with unrelated lights and pick lowest VY_FWHM among them).
        try:
            if not explicit_ui_masterstar_paths and len(_cand_all) < max(2, _best_n):
                _disk_more = _disk_stack_fallback_paths(Path(detrended_root), max_frames=max(8, _best_n * 2))
                for p in _disk_more:
                    if p not in _cand_all and p.is_file():
                        _cand_all.append(p)
        except Exception:  # noqa: BLE001
            pass
        if not _cand_all:
            raise FileNotFoundError(f"MASTERSTAR: no FITS to select from in {detrended_root}.")
        _cand_singletons = _cand_all[:_best_n]

        last_exc: Exception | None = None
        info = {}
        # Try pool first (as before), then single best-of-N frames.
        attempt_lists: list[tuple[str, list[Path]]] = [("pool", _cand_all)]
        for i, p in enumerate(_cand_singletons, start=1):
            attempt_lists.append((f"single_{i:02d}_of_{len(_cand_singletons):02d}", [p]))
        for label, paths_try in attempt_lists:
            try:
                log_event(f"MASTERSTAR build attempt: {label} (n={len(paths_try)})")
                info = build_masterstar_from_detrended(
                    detrended_root=detrended_root,
                    output_fits=masterstar_fits,
                    only_paths=paths_try,
                    fwhm_fallback_px=_ms_fwhm_fb,
                    pre_calibrated=_pre_cal_ms,
                )
                # Update selection metadata for traceability.
                ms_selection_meta = dict(ms_selection_meta or {})
                ms_selection_meta["best_of_n"] = int(_best_n)
                ms_selection_meta["build_attempt"] = str(label)
                ms_selection_meta["build_only_paths"] = [str(p.name) for p in paths_try]
                _bfpx = info.get("best_frame_fwhm_px")
                if _bfpx is not None:
                    try:
                        _bfv = float(_bfpx)
                        if math.isfinite(_bfv) and 0.5 < _bfv < 80.0:
                            ms_selection_meta["best_frame_fwhm_px"] = float(_bfv)
                    except (TypeError, ValueError):
                        pass
                last_exc = None
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                log_event(f"MASTERSTAR build attempt failed: {label}: {exc!s}")
                continue
        if last_exc is not None:
            raise last_exc
        try:
            _legacy_master = Path(detrended_root) / "MASTERSTAR.fits"
            if _legacy_master.is_file() and _legacy_master.resolve() != masterstar_fits.resolve():
                _legacy_master.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass
    return _match_sep_eff, info, masterstar_fits, ms_selection_meta, platesolve_dir


def _ms_selected_reference_path(*, info):
    """Path of the frame the MASTERSTAR was copied from (None if unknown)."""
    _selected_ref_path: Path | None = None
    try:
        _rp = str(info.get("reference_path") or "").strip()
        _selected_ref_path = Path(_rp) if _rp else None
    except Exception:  # noqa: BLE001
        _selected_ref_path = None
    return _selected_ref_path,


#: G04-22: field_catalog_cone.csv of the MASTERSTAR being built, parsed once (it was read three
#: times: catalogue membership, Gaia-complete enrich, identity enrich). Keyed by resolved path, size
#: and mtime, so a rewritten cone (catalogue top-up re-runs the detection) is read again.
_CONE_CSV_CACHE: dict[tuple[str, int, int], pd.DataFrame] = {}


def _read_field_cone_csv(path: Path) -> pd.DataFrame:
    """``read_vyvar_csv(path, low_memory=False)``, parsed once per file version; a copy per call."""
    p = Path(path).resolve()
    st = p.stat()
    key = (str(p), int(st.st_size), int(st.st_mtime_ns))
    hit = _CONE_CSV_CACHE.get(key)
    if hit is None:
        hit = read_vyvar_csv(p, low_memory=False)
        _CONE_CSV_CACHE.clear()
        _CONE_CSV_CACHE[key] = hit
    return hit.copy(deep=True)


def _ms_plate_scale_hint(*, app_config, draft_id, equipment_id, masterstar_fits, telescope_id):
    """Read the MASTERSTAR and resolve the expected plate scale (DB optics, FITS, bundle) for the solver."""
    import numpy as np  # noqa: PLC0415

    with fits.open(masterstar_fits, memmap=False) as hdul:
        hdr = hdul[0].header.copy()
        data = np.array(hdul[0].data, dtype=np.float32, copy=True)

    _cfg_ms = app_config

    _dao_sigma_eff = resolve_detection_threshold_sigma(_cfg_ms.detection_threshold_sigma)
    log_event(
        f"MASTERSTAR: DAO threshold sigmaxRMS = {_dao_sigma_eff:.2f} "
        f"(config detection_threshold_sigma; plate solve + catalog)"
    )

    _full_db = str(_cfg_ms.gaia_db_path or "").strip()
    if not _full_db:
        raise RuntimeError(
            "MASTERSTAR: set gaia_db_path in Settings (full local Gaia DR3 SQLite DB)."
        )
    log_event("MASTERSTAR WCS: VYVAR astrosolve + full Gaia DB (gaia_db_path).")
    try:
        _xb_ms, _yb_ms = fits_binning_xy_from_header(hdr)
        _bin_ms = max(1, int(_xb_ms), int(_yb_ms))
    except Exception:  # noqa: BLE001
        _bin_ms = 1

    _auto_scale_ms: float | None = None
    _db_scale = _vyvar_open_database(_cfg_ms)
    _eq_ms: int | None = int(equipment_id) if equipment_id is not None else None
    _tel_ms: int | None = None
    if _db_scale is not None:
        try:
            _eq_ms, _tel_ms = resolve_optics_ids_for_platesolve(
                _db_scale, draft_id, equipment_id=equipment_id, telescope_id=telescope_id
            )
            _auto_scale_ms = compute_plate_scale_from_db(
                _eq_ms, _tel_ms, _db_scale, binning=_bin_ms
            )
        except Exception:  # noqa: BLE001
            _auto_scale_ms = None
        finally:
            try:
                _db_scale.conn.close()
            except Exception:  # noqa: BLE001
                pass

    if _auto_scale_ms is not None:
        log_event(
            f"INFO: Plate scale from DB (Equipment+Telescope): {_auto_scale_ms:.4f} arcsec/px"
        )
    else:
        log_event(
            "WARNING: Plate scale from DB unavailable - solver derives the scale from FITS or None"
        )

    _plate_scale_ms = _auto_scale_ms or None
    # Pull more complete optics hints (focal + effective pixel) from DB/FITS.
    # This is critical when FITS headers lack FOCALLEN/PIXSIZE and the solver would otherwise
    # overestimate FOV / cone radius and fail triangle matching.
    _bundle = _plate_solve_input_bundle(
        Path(masterstar_fits),
        app_config=_cfg_ms,
        equipment_id=_eq_ms,
        draft_id=int(draft_id) if draft_id is not None else None,
        telescope_id=_tel_ms if _tel_ms is not None else telescope_id,
    )
    _expected_bundle = _bundle.get("expected_arcsec_per_px")
    # D1/S6: a FITS/config/UI scale must not overwrite Equipment+Telescope DB
    # scale. On 520 g_60_4 the bundle used 200 mm / 15.511 "/px (Zeiss-wide
    # default) while the AZ800 row is 0.566 "/px; the triangle filter then
    # rejected every match. First auto-scale from DB wins.
    try:
        _bundle_scale = (
            float(_expected_bundle)
            if _expected_bundle is not None
            and math.isfinite(float(_expected_bundle))
            and float(_expected_bundle) > 0
            else None
        )
    except (TypeError, ValueError):
        _bundle_scale = None
    if _auto_scale_ms is not None:
        _plate_scale_ms = float(_auto_scale_ms)
        if (
            _bundle_scale is not None
            and abs(float(_bundle_scale) - float(_auto_scale_ms)) / float(_auto_scale_ms) > 0.05
        ):
            log_event(
                f"WARNING: MASTERSTAR plate-scale from FITS/config/UI "
                f"({_bundle_scale:.4f} arcsec/px) disagrees with DB Equipment+Telescope "
                f"({_auto_scale_ms:.4f} arcsec/px) - keeping DB scale for the triangle filter."
            )
    elif _bundle_scale is not None:
        _plate_scale_ms = float(_bundle_scale)
    return _cfg_ms, _eq_ms, _full_db, _plate_scale_ms, data, hdr


def _ms_solve_wcs(
    *,
    _cfg_ms,
    _full_db,
    _plate_scale_ms,
    data,
    draft_id,
    hdr,
    hint_dec_deg,
    hint_ra_deg,
    masterstar_fits,
    masterstar_platesolve_skip_solve,
):
    """VYVAR astrosolve: solve (or confirm) the MASTERSTAR WCS; returns the reloaded image, header and solve meta."""
    import numpy as np  # noqa: PLC0415

    _skip_independent_solve = bool(masterstar_platesolve_skip_solve)
    solve_meta: dict[str, Any] = {}
    if _skip_independent_solve:
        # WCS-CONFIRM-2026-10: a WCS handed over from another MASTERSTAR (sibling filter, OSC
        # channel) is verified (log-odds) and refined by astrosolve like a solved one; a WCS
        # that does not verify stops the build (it would give wrong catalogue identities).
        import astrosolve as _asol_c  # noqa: PLC0415

        log_event("MASTERSTAR: WCS taken over from another MASTERSTAR - verification and refinement (astrosolve confirm).")
        solve_meta = _asol_c.confirm_fits(masterstar_fits, gaia_db=str(_full_db), write=True, log=log_event)
        if not bool(solve_meta.get("solved", False)):
            raise RuntimeError(
                "MASTERSTAR: taken-over WCS failed verification against Gaia - " + str(solve_meta.get("message") or solve_meta)
            )
        with fits.open(masterstar_fits, memmap=False) as hdul:
            hdr = hdul[0].header.copy()

    if not _skip_independent_solve:

        _mra, _mde, _ = _pointing_hint_from_header(hdr)
        if hint_ra_deg is not None and hint_dec_deg is not None:
            try:
                _hra_ov = float(hint_ra_deg)
                _hde_ov = float(hint_dec_deg)
                if math.isfinite(_hra_ov) and math.isfinite(_hde_ov):
                    _mra, _mde = _hra_ov, _hde_ov
                    log_event(
                        "MASTERSTAR: hint_ra_deg / hint_dec_deg from the call override the FITS hint "
                        "(second MASTERSTAR / detrended aligned)."
                    )
            except (TypeError, ValueError):
                pass
        _hint_sep_thr = float(_MASTERSTAR_SOLVER_USE_DRAFT_MEDIAN_IF_HINT_SEP_DEG)
        if draft_id is not None:
            _dbc_hint = _vyvar_open_database(_cfg_ms)
            if _dbc_hint is not None:
                try:
                    med_ra, med_de = draft_median_pointing_icrs_deg(_dbc_hint, int(draft_id))
                    if med_ra is not None and med_de is not None:
                        if _mra is None or _mde is None:
                            _mra, _mde = med_ra, med_de
                            log_event(
                                "MASTERSTAR solve: using median RA/Dec from manifest files[] (header has no reliable hint)."
                            )
                        else:
                            sc_h = SkyCoord(ra=float(_mra) * u.deg, dec=float(_mde) * u.deg, frame="icrs")
                            sc_d = SkyCoord(ra=float(med_ra) * u.deg, dec=float(med_de) * u.deg, frame="icrs")
                            sep = float(sc_h.separation(sc_d).deg)
                            if sep > float(_hint_sep_thr):
                                log_event(
                                    f"MASTERSTAR solve: hint vs draft median = {sep:.3f} deg > {_hint_sep_thr} deg "
                                    "- using draft median from manifest files[]."
                                )
                                _mra, _mde = med_ra, med_de
                            elif sep > 0.05:
                                log_event(
                                    f"MASTERSTAR solve: hint vs draft median = {sep:.3f} deg (check pointing)."
                                )
                finally:
                    try:
                        _dbc_hint.conn.close()
                    except Exception:  # noqa: BLE001
                        pass

        # PLATESOLVE-2026-10: astrometry.net-style solver (astrosolve): hinted with the
        # position above (scale only orders the search), blind index as fallback;
        # Bayesian log-odds verification, TAN+SIP with order by BIC.
        import astrosolve as _asol  # noqa: PLC0415

        solve_meta = _asol.solve_fits(
            masterstar_fits,
            gaia_db=str(_full_db),
            index_dir=_asol.default_index_dir(str(_full_db), str(getattr(_cfg_ms, "astrometry_index_dir", "") or "")),
            hint_ra_deg=_mra,
            hint_dec_deg=_mde,
            scale_hint_arcsec=float(_plate_scale_ms) if _plate_scale_ms is not None else None,
            write=True,
            log=log_event,
        )
        if not bool(solve_meta.get("solved", False)):
            raise RuntimeError(
                "MASTERSTAR plate solve failed: " + str(solve_meta.get("message") or solve_meta)
                + ". Check gaia_db_path; without a position in the header the blind index is needed "
                "(python catalogs/build_astro_index.py)."
            )
        with fits.open(masterstar_fits, memmap=False) as hdul:
            hdr = hdul[0].header.copy()
            data = np.array(hdul[0].data, dtype=np.float32, copy=True)
        if not _has_valid_wcs(hdr):
            raise RuntimeError("MASTERSTAR: no valid WCS after plate-solve.")
    return data, hdr, solve_meta


def _masterstar_fwhm_px(det_meta: dict, hdr) -> float:
    """FWHM [px] of the MASTERSTAR detection: the DAO kernel FWHM of the detection, else VY_FWHM
    (the one reader, G03-29); neither -> RuntimeError (no 3.5 px default)."""
    try:
        v = float(det_meta.get("dao_fwhm_px") or 0.0)
    except (TypeError, ValueError):
        v = 0.0
    if math.isfinite(v) and v > 0.0:
        return v
    vy = fwhm_from_header_vy_fwhm(hdr)
    if vy is None:
        raise RuntimeError("MASTERSTAR: no DAO FWHM and no valid VY_FWHM")
    return float(vy)


def _ms_stamp_plate_scale_and_check_wcs(*, _plate_scale_ms, data, hdr, masterstar_fits, solve_meta):
    """Write VY_PLTS and run the WCS round-trip check on the solved MASTERSTAR."""
    _exp_scale_apx: float | None = None
    if _plate_scale_ms is not None:
        try:
            _ea2 = float(_plate_scale_ms)
            if math.isfinite(_ea2) and _ea2 > 0:
                _exp_scale_apx = float(_ea2)
        except (TypeError, ValueError):
            _exp_scale_apx = None
    if _exp_scale_apx is None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FITSFixedWarning)
                _w_hdr = WCS(hdr)
            if getattr(_w_hdr, "has_celestial", False):
                # G04-30: proj_plane_pixel_scales (the diagonal of the scale matrix is ~0 at 90 deg rotation)
                _exp_scale_apx = plate_scale_arcsec_per_px_from_wcs(_w_hdr)
        except Exception:  # noqa: BLE001
            _exp_scale_apx = None
    if _exp_scale_apx is None or (not math.isfinite(_exp_scale_apx)) or _exp_scale_apx <= 0:
        # derive-or-None (DR6 pattern): all principled sources exhausted - do not guess.
        _exp_scale_apx = None
        log_event(
            "WARNING: MASTERSTAR plate scale not derivable (DB/FITS/WCS exhausted) - "
            "expected scale unknown; VY_PLTS will not be written."
        )

    solve_meta["wcs_plate_scale_adjustment"] = {"rescaled": False, "reason": "astrosolve solution kept"}

    # G13-06: a DB / header optics entry that disagrees with the solved scale is reported, never rewritten
    if isinstance(solve_meta, dict):
        report_scale_mismatch(_plate_scale_ms, solve_meta.get("plate_scale_arcsec_px"), context="MASTERSTAR")

    # Write calibrated plate scale to MASTERSTAR header
    _vy_plts = None
    try:
        if isinstance(solve_meta, dict):
            _vy_plts = solve_meta.get("plate_scale_arcsec_px")
        if _vy_plts is None:
            _vy_plts = _exp_scale_apx
    except Exception:  # noqa: BLE001
        _vy_plts = None

    if _vy_plts is not None:
        try:
            _vy_plts_f = float(_vy_plts)
            if math.isfinite(_vy_plts_f) and _vy_plts_f > 0:
                with fits.open(masterstar_fits, mode="update") as hdul:
                    hdul[0].header["VY_PLTS"] = (
                        _vy_plts_f,
                        "VYVAR plate scale arcsec/px",
                    )
                    hdul.flush()
                log_event(f"VY_PLTS={_vy_plts_f:.4f} written to MASTERSTAR.fits")
        except Exception as exc:  # noqa: BLE001
            log_event(f"Could not write VY_PLTS to MASTERSTAR: {exc}")
    else:
        log_event(
            "WARNING: MASTERSTAR VY_PLTS not written - plate scale not derivable "
            "(derive-or-None; no rig/global constant written to header)."
        )

    try:
        from wcs_invertibility import evaluate_wcs_roundtrip

        _nax1 = int(hdr.get("NAXIS1") or (data.shape[1] if data.ndim >= 2 else 0))
        _nax2 = int(hdr.get("NAXIS2") or (data.shape[0] if data.ndim >= 1 else 0))
        if _has_valid_wcs(hdr) and _nax1 > 0 and _nax2 > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FITSFixedWarning)
                _w_rt = WCS(hdr)
            _rt0 = evaluate_wcs_roundtrip(_w_rt, naxis1=_nax1, naxis2=_nax2)
            if isinstance(solve_meta, dict):
                solve_meta["wcs_roundtrip_p99_px"] = _rt0.get("wcs_roundtrip_p99_px")
                solve_meta["wcs_roundtrip_pass"] = bool(_rt0.get("pass"))
            if not _rt0.get("pass"):
                log_event(
                    f"WARNING: MASTERSTAR initial-solve WCS round-trip p99="
                    f"{_rt0.get('wcs_roundtrip_p99_px'):.4f}px (threshold "
                    f"{_rt0.get('p99_threshold_px')}px) - provenance flag set; continuing."
                )
            else:
                log_event(
                    f"MASTERSTAR WCS round-trip PASS (initial solve): p99="
                    f"{_rt0.get('wcs_roundtrip_p99_px'):.4f}px"
                )
    except Exception as _rt_exc:  # noqa: BLE001
        log_event(f"MASTERSTAR WCS round-trip check skipped: {_rt_exc!s}")


def _ms_load_for_detection(*, data, hdr, masterstar_fits, ms_selection_meta, platesolve_dir):
    """DAO FWHM for the MASTERSTAR, the image for detection and the cone-cache check."""
    import numpy as np  # noqa: PLC0415

    _ms_fwhm = DAO_FWHM_FALLBACK_PX
    _ms_meta = ms_selection_meta if isinstance(ms_selection_meta, dict) else {}
    _best_fwhm = _ms_meta.get("best_frame_fwhm_px")
    try:
        _best_fwhm_f = float(_best_fwhm) if _best_fwhm is not None else float("nan")
    except (TypeError, ValueError):
        _best_fwhm_f = float("nan")
    if (
        math.isfinite(_best_fwhm_f)
        and 1.2 <= _best_fwhm_f <= 20.0
    ):
        dao_fwhm_px_for_ms = float(_best_fwhm_f)
        _dao_fwhm_bypass_hdr = True
        log_event(
            f"MASTERSTAR DAO: dao_fwhm_px={dao_fwhm_px_for_ms:.3f} from best_frame_fwhm_px "
            f"(header VY_FWHM median ignored)"
        )
    else:
        dao_fwhm_px_for_ms = float(_ms_fwhm)
        _dao_fwhm_bypass_hdr = False
        log_event(
            f"MASTERSTAR DAO: dao_fwhm_px={dao_fwhm_px_for_ms:.3f} from DAO_FWHM_FALLBACK_PX / header VY_FWHM"
        )

    with fits.open(masterstar_fits, memmap=False) as hdul:
        hdr = hdul[0].header.copy()
        data = np.array(hdul[0].data, dtype=np.float32, copy=True)
    data = np.ascontiguousarray(data, dtype=np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    # Global median (BOJ) background subtraction is intentionally OFF (handled downstream).
    _ms_mean = float(np.nanmean(data))
    log_event("MASTERSTAR: global median subtraction (BOJ) disabled.")
    log_event(f"MASTERSTAR after nan_to_num: mean={_ms_mean:.6f}")
    _ms_min = float(np.nanmin(data))
    _ms_max = float(np.nanmax(data))
    log_event(f"MASTERSTAR levels: noise_floor(min)={_ms_min:.2f}, saturation_proxy(max)={_ms_max:.2f}")

    if platesolve_dir is None:
        raise ValueError("generate_masterstar_and_catalog: platesolve_dir is required (got None).")
    platesolve_dir.mkdir(parents=True, exist_ok=True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            _w_pre = WCS(hdr)
        if _w_pre.has_celestial:
            _, _r_cat_need = _effective_field_catalog_cone_radius_deg(_w_pre, int(data.shape[0]), int(data.shape[1]))
            _invalidate_field_catalog_cone_cache_if_needed(
                platesolve_dir / "field_catalog_cone.csv",
                effective_radius_deg=float(_r_cat_need),
            )
    except Exception as exc:  # noqa: BLE001
        log_event(f"Catalog: field_catalog_cone cache check skipped - {exc!s}")
    return _dao_fwhm_bypass_hdr, dao_fwhm_px_for_ms, data, hdr


def _ms_detect_and_match(
    *,
    _cfg_ms,
    _dao_fwhm_bypass_hdr,
    _eq_ms,
    _full_db,
    _ms_measured_sat,
    _selected_ref_path,
    dao_fwhm_px_for_ms,
    data,
    draft_id,
    equipment_saturate_adu,
    hdr,
    masterstars_csv_basename,
    platesolve_dir,
):
    """DAO detection + Gaia identities (with the catalogue top-up); the catalogue stays in memory.

    G04-25: no write / re-read here. Each row keeps its own ``vy_dao_pass`` and ``ambiguous_owner``
    (the former re-read remapped them by catalog_id, so every DAO_ONLY row got the last DAO_ONLY
    row's values); the export normaliser runs at the final write.
    """
    from catalog_match import detect_stars_and_match_catalog  # noqa: PLC0415

    _ms_max_catalog_rows_eff = MASTERSTAR_MAX_DAO_ROWS
    _detect_kw: dict[str, Any] = {
        "gaia_db_path": Path(_full_db),
        "field_catalog_export_path": platesolve_dir / "field_catalog_cone.csv",
        "dao_fwhm_px": dao_fwhm_px_for_ms,
        "equipment_saturate_adu": equipment_saturate_adu,
        "measured_saturate_adu": _ms_measured_sat,
        "dao_fwhm_bypass_header": bool(_dao_fwhm_bypass_hdr),
        "cfg": _cfg_ms,
    }
    # GAIN-ONE-2026-10: the SNR of the MASTERSTAR catalogue (snr_min gates) with the run's gain
    from rig_photometry import photometric_gain_read_noise  # noqa: PLC0415

    _detect_kw["photometric_gain"] = float(photometric_gain_read_noise(hdr, _cfg_ms, _eq_ms).gain)
    df_out, det_meta = detect_stars_and_match_catalog(data, hdr, **_detect_kw)
    # FRAME-DEPTH-2026-10 step 2: the frame reaches fainter than the main Gaia DB covers this field
    # -> fetch the field (+-1 deg) down to the frame limit into the main DB and match again.
    # Offline the run continues with the local DB; the limitation is logged and reported.
    _gaia_topup: dict[str, Any] = {"status": "not_needed"}
    _fd0 = det_meta.get("frame_depth") or {}
    if _fd0.get("limited_by_catalogue") and _fd0.get("g_lim_frame") is not None:
        from gaia_coverage import ensure_field_depth  # noqa: PLC0415

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            _w_fd = WCS(hdr)
        _c_fd, _r_fd = _effective_field_catalog_cone_radius_deg(_w_fd, int(data.shape[0]), int(data.shape[1]))
        # One more sigma of the frame's own completeness width beyond the estimated limit: the
        # catalogue then holds the whole completeness drop, so after the top-up the fit (not the
        # detector estimate) gives the final limit (BO CVn, DB cut at G 15: estimate 15.14, fit 15.28).
        _sig_fd = float(_fd0.get("sigma_mag") or 0.0)
        _g_need = math.ceil((float(_fd0["g_lim_frame"]) + max(0.0, _sig_fd)) * 10.0 - 1e-9) / 10.0
        try:
            _gaia_topup = ensure_field_depth(
                Path(_full_db), center=_c_fd, radius_deg=float(_r_fd), g_needed=_g_need
            )
        except Exception as _tu_exc:  # noqa: BLE001 - the build continues with the local DB
            _gaia_topup = {"status": "error", "error": f"{type(_tu_exc).__name__}: {str(_tu_exc)[:200]}"}
            log_event(f"GAIA TOP-UP: failed ({_gaia_topup['error']}); continuing with the local DB")
        if _gaia_topup.get("status") == "ok":
            log_event(
                f"MASTERSTAR: Gaia DB topped up to G {_g_need:.1f} for this field "
                f"({_gaia_topup.get('n_new')} new stars) - catalogue match repeated"
            )
            df_out, det_meta = detect_stars_and_match_catalog(data, hdr, **_detect_kw)
    det_meta["gaia_topup"] = _gaia_topup


    if "b_v" in df_out.columns and "bp_rp" not in df_out.columns:
        df_out = df_out.copy()
        df_out["bp_rp"] = pd.to_numeric(df_out["b_v"], errors="coerce")
    if "mag" in df_out.columns:
        df_out = df_out.copy()
        df_out["phot_g_mean_mag"] = pd.to_numeric(df_out["mag"], errors="coerce")

    if int(det_meta.get("n_detected", 0)) == 0:
        raise RuntimeError("No stars detected on MASTERSTAR.")
    _n_det_raw = int(det_meta.get("n_detected", 0) or 0)
    _n_mat_raw = int(det_meta.get("n_matched", 0) or 0)
    _rate_raw = (100.0 * float(_n_mat_raw) / float(_n_det_raw)) if _n_det_raw > 0 else 0.0
    # G04-08: no Gaia->DAO completeness here (the cone is not the reference); it is computed once
    # the census exists (dao_reconcile.gaia_detection_counts).
    log_event(
        f"[chart] MATCH STATS (raw): Found {_n_det_raw} stars on image | {_n_mat_raw} matched with Gaia | "
        f"Match Rate: {_rate_raw:.2f}%"
    )
    _update_masterstar_obs_file_status(
        cfg=_cfg_ms,
        draft_id=draft_id,
        selected_ref_path=_selected_ref_path,
        wcs_ok=bool(_has_valid_wcs(hdr)),
        n_stars=_n_det_raw,
    )
    _msc_name = str(masterstars_csv_basename or "masterstars_full_match.csv").strip() or "masterstars_full_match.csv"
    csv_path = platesolve_dir / _msc_name
    # WCS-CONFIRM-2026-10: every MASTERSTAR WCS is verified by astrosolve (solved or confirmed),
    # so the catalogue identities of detect_stars_and_match_catalog (IDENTITY-2026-10) are final;
    # the former optimizer re-match (fixed arcsec radii, nearest pixel) is gone.
    return _ms_max_catalog_rows_eff, csv_path, det_meta, df_out


def _ms_wcs_roundtrip_and_identity_qa(*, _cfg_ms, df_final, masterstar_fits):
    """WCS round-trip p99 and identity QA of the final MASTERSTAR catalogue."""
    _wcs_rt_p99: float | None = None
    _wcs_rt_pass: bool | None = None
    _identity_qa: dict[str, Any] = {}
    try:
        from wcs_invertibility import (
            evaluate_matched_world2pix_identity_px,
            evaluate_wcs_roundtrip,
            finalize_masterstar_sky_coords,
        )

        with fits.open(masterstar_fits, memmap=False) as _hf:
            _hdr_fin = _hf[0].header
            _w_fin = WCS(_hdr_fin)
        _gaia_fin: dict[str, tuple[float, float]] = {}
        df_final = finalize_masterstar_sky_coords(
            df_final,
            _w_fin,
            gaia_db_path=str(_cfg_ms.gaia_db_path or ""),
            log_fn=log_event,
            gaia_coords_out=_gaia_fin,
        )
        _nax1f = int(_hdr_fin.get("NAXIS1") or 0)
        _nax2f = int(_hdr_fin.get("NAXIS2") or 0)
        _rt_fin = evaluate_wcs_roundtrip(_w_fin, naxis1=_nax1f, naxis2=_nax2f)
        _wcs_rt_p99 = _rt_fin.get("wcs_roundtrip_p99_px")
        _wcs_rt_pass = bool(_rt_fin.get("pass"))
        # G02-44: reuse the finalize lookup (same ids, same DB); query only when it found nothing / failed
        _identity_qa = evaluate_matched_world2pix_identity_px(
            df_final,
            _w_fin,
            gaia_ra_dec_by_cid=_gaia_fin,
            gaia_db_path="" if _gaia_fin else str(_cfg_ms.gaia_db_path or ""),
            log_fn=log_event,
        )
        _p95 = _identity_qa.get("matched_world2pix_identity_p95_px")
        try:
            _p95f = float(_p95) if _p95 is not None else float("nan")
        except (TypeError, ValueError):
            _p95f = float("nan")
        # Standing series WARN (Anchor #3 / draft_435 baseline p95~1.54 px): soft threshold only.
        # INV-WCS-01: same band, recorded into pipeline_meta invariants at merge below.
        _IDENTITY_P95_WARN_PX = 2.0
        if math.isfinite(_p95f) and _p95f > _IDENTITY_P95_WARN_PX:
            logging.warning(
                "[IDENTITY-QA] matched_world2pix_identity_p95_px=%.3f exceeds WARN threshold %.1f px "
                "(series baseline draft_435 p95~1.54; no FAIL)",
                _p95f,
                _IDENTITY_P95_WARN_PX,
            )
            log_event(
                f"IDENTITY-QA WARN: p95={_p95f:.3f} px > {_IDENTITY_P95_WARN_PX:.1f} px threshold"
            )
        try:
            from invariants_runtime import check_wcs_identity_p95  # noqa: PLC0415
            from invariants_runtime import inv_check  # noqa: PLC0415

            _ok_w, _det_w = check_wcs_identity_p95(_p95f if math.isfinite(_p95f) else None)
            _inv_meta_wcs: dict = {"invariants": []}
            inv_check(_inv_meta_wcs, "INV-WCS-01", _ok_w, policy="WARN", detail=_det_w)
            _identity_qa = dict(_identity_qa or {})
            _identity_qa["_inv_wcs_01"] = _inv_meta_wcs.get("invariants") or []
        except Exception as _inv_wcs_exc:  # noqa: BLE001
            logging.debug("[INV-WCS-01] record skipped: %s", _inv_wcs_exc)
    except Exception as _fin_exc:  # noqa: BLE001
        log_event(f"MASTERSTAR coordinate finalization / round-trip QA skipped: {_fin_exc!s}")
        _wcs_rt_p99 = None
        _wcs_rt_pass = None
        _identity_qa = {}
    return _identity_qa, _wcs_rt_p99, _wcs_rt_pass, df_final


def _ms_expand_catalog_membership(
    *,
    _cfg_ms,
    _ms_measured_sat,
    det_meta,
    df_final,
    equipment_saturate_adu,
    hdr,
    platesolve_dir,
):
    """Catalogue membership: on-chip Gaia stars without a detection get a row; flux zones annotated."""
    import numpy as np  # noqa: PLC0415

    _chip_ms: pd.DataFrame | None = None
    _membership_expand_meta: dict[str, Any] = {}
    _cone_gaia_pre = Path(platesolve_dir) / "field_catalog_cone.csv"
    _wcs_ok_pre = bool(_has_valid_wcs(hdr))
    if _cone_gaia_pre.is_file() and _wcs_ok_pre:
        from masterstar_gaia_accounting import (  # noqa: PLC0415
            expand_detection_to_catalog_membership,
            gaia_on_chip_from_cone,
        )
        from astropy.wcs import WCS as _WCS_expand  # noqa: PLC0415

        LOGGER.info(
            "[M1] catalog membership expand: cone=%s wcs_ok=%s n_ms_in=%d",
            True,
            True,
            int(len(df_final)),
        )
        _cone_df_pre = _read_field_cone_csv(_cone_gaia_pre)
        _nax1_pre = int(hdr.get("NAXIS1") or 0)
        _nax2_pre = int(hdr.get("NAXIS2") or 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            _wcs_pre = _WCS_expand(hdr)
        _ra_pre = pd.to_numeric(_cone_df_pre["ra_deg"], errors="coerce").to_numpy(dtype=np.float64)
        _de_pre = pd.to_numeric(_cone_df_pre["dec_deg"], errors="coerce").to_numpy(dtype=np.float64)
        _ok_pre = np.isfinite(_ra_pre) & np.isfinite(_de_pre)
        _gx_pre, _gy_pre = _wcs_pre.world_to_pixel_values(_ra_pre[_ok_pre], _de_pre[_ok_pre])
        _chip_ms = gaia_on_chip_from_cone(
            _cone_df_pre, gx=_gx_pre, gy=_gy_pre, ok_mask=_ok_pre, wpx=_nax1_pre, h=_nax2_pre
        )
        _membership_depth_g = _frame_matching_depth_g(det_meta)
        df_final, _membership_expand_meta = expand_detection_to_catalog_membership(
            df_final,
            _chip_ms,
            membership_depth_g=_membership_depth_g,
            wpx=_nax1_pre,
            h=_nax2_pre,
        )
        det_meta["catalog_derived_membership"] = dict(_membership_expand_meta)
        LOGGER.info(
            "[M1] catalog-derived membership: +%d Gaia rows (depth G<=%.1f), n_out=%d",
            int(_membership_expand_meta.get("n_catalog_rows_added", 0)),
            _membership_depth_g,
            int(_membership_expand_meta.get("n_rows_out", len(df_final))),
        )
        log_event(
            "MASTERSTAR catalog-derived membership: "
            f"+{int(_membership_expand_meta.get('n_catalog_rows_added', 0))} Gaia rows "
            f"(depth G<={_membership_depth_g:.1f}), "
            f"n_out={int(_membership_expand_meta.get('n_rows_out', len(df_final)))}"
        )
    elif _cone_gaia_pre.is_file() or _wcs_ok_pre:
        raise RuntimeError(
            "INV-MS-EXPAND-01: catalog membership expand blocked "
            f"(cone={_cone_gaia_pre.is_file()} wcs_ok={_wcs_ok_pre})"
        )
    # VSX stamp deferred until after write_photometry_plan_files (VT CSV created there).
    df_final = _annotate_masterstars_flux_zones(
        df_final,
        equipment_saturate_adu=equipment_saturate_adu,
        saturate_limit_adu_fallback=det_meta.get("saturate_limit_adu"),
        saturate_limit_fraction=float(LINEARITY_FRAC),
        sigma_px=det_meta.get("bg_sigma_adu"),
        sky_median_adu=det_meta.get("sky_median_adu"),
        frame_max_adu=det_meta.get("frame_max_adu"),
        empirical_clip_adu=det_meta.get("empirical_clip_adu"),
        measured_sat_adu=_ms_measured_sat,
        dao_detection_n_equiv=(
            det_meta.get("dao_detection_n_equiv")
            if det_meta.get("dao_detection_n_equiv") is not None
            else float(_cfg_ms.detection_threshold_sigma)
        ),
    )
    return _chip_ms, _membership_expand_meta, df_final


def _ms_gaia_accounting_and_reconcile(
    *,
    _cfg_ms,
    _chip_ms,
    _membership_expand_meta,
    det_meta,
    df_final,
    hdr,
    platesolve_dir,
):
    """Gaia accounting (locks, census) and the DAO-Gaia reconciliation / DAO_ONLY classes."""
    import numpy as np  # noqa: PLC0415

    _dao_class_meta: dict[str, Any] = {}
    _recon_ms: dict[str, Any] | None = None
    try:
        cid = df_final.get("catalog_id", pd.Series([""] * len(df_final))).fillna("").astype(str).str.strip()
        df_final["source_type"] = np.where(cid.ne(""), "GAIA_MATCHED", "DAO_ONLY")
        from masterstar_gaia_accounting import (  # noqa: PLC0415
            enrich_masterstar_gaia_complete,
            gaia_on_chip_from_cone,
            verify_gaia_census,
        )
        from astropy.wcs import WCS as _WCS_enrich  # noqa: PLC0415

        _cone_gaia = Path(platesolve_dir) / "field_catalog_cone.csv"
        if not _cone_gaia.is_file():
            raise RuntimeError("MASTERSTAR Gaia-complete enrich: missing field_catalog_cone.csv")
        if not _has_valid_wcs(hdr):
            raise RuntimeError("MASTERSTAR Gaia-complete enrich: MASTERSTAR WCS missing/invalid")
        _cone_df = _read_field_cone_csv(_cone_gaia)
        _nax1_g = int(hdr.get("NAXIS1") or 0)
        _nax2_g = int(hdr.get("NAXIS2") or 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            _wcs_g = _WCS_enrich(hdr)
        _ra_g = pd.to_numeric(_cone_df["ra_deg"], errors="coerce").to_numpy(dtype=np.float64)
        _de_g = pd.to_numeric(_cone_df["dec_deg"], errors="coerce").to_numpy(dtype=np.float64)
        _ok_g = np.isfinite(_ra_g) & np.isfinite(_de_g)
        _gx, _gy = _wcs_g.world_to_pixel_values(_ra_g[_ok_g], _de_g[_ok_g])
        _chip = (
            _chip_ms
            if _chip_ms is not None and len(_chip_ms)
            else gaia_on_chip_from_cone(
                _cone_df, gx=_gx, gy=_gy, ok_mask=_ok_g, wpx=_nax1_g, h=_nax2_g
            )
        )
        _fwhm_g = _masterstar_fwhm_px(det_meta, hdr)
        # FRAME-DEPTH-2026-10: census and the certificate go to the frame limit
        # (was a fixed 17.5; membership a fixed 15).
        _census_depth_g = _frame_matching_depth_g(det_meta)
        df_final, _gaia_census, _gaia_meta = enrich_masterstar_gaia_complete(
            df_final,
            gaia_on_chip=_chip,
            wpx=_nax1_g,
            h=_nax2_g,
            fwhm_px=_fwhm_g,
            target_depth_g=_census_depth_g,
            identity_lock_only=False,
            catalog_derived_membership=bool(_membership_expand_meta),
            tolerance_overrides=det_meta.get("dao_gaia_derived_tol"),
        )
        _census_inv = verify_gaia_census(_gaia_census, n_on_chip=len(_chip))
        if _membership_expand_meta:
            from masterstar_gaia_accounting import verify_ms_expand_guard  # noqa: PLC0415

            _ok_exp, _det_exp = verify_ms_expand_guard(_membership_expand_meta)
            if not _ok_exp:
                from invariants_runtime import InvariantViolation  # noqa: PLC0415

                raise InvariantViolation("INV-MS-EXPAND-01", _det_exp)
            log_event(f"INV-MS-EXPAND-01 PASS: {_det_exp}")
        det_meta["gaia_census_meta"] = _gaia_meta
        det_meta["gaia_census_invariants"] = _census_inv.get("invariants") or []
        log_event(
            f"MASTERSTAR Gaia census: {len(_chip)} on-chip, "
            f"by_state={_gaia_meta.get('census_by_state', {})}, "
            f"leftover_promotions={_gaia_meta.get('n_leftover_promotions', 0)}, "
            f"INV-MS-CENSUS-01 {_census_inv.get('detail')}"
        )
        _gdb_fill = str(_cfg_ms.gaia_db_path or "").strip()
        df_final, _n_bp_fill, _n_bp_miss = _fill_masterstars_gaia_matched_bp_rp_from_local_db(
            df_final,
            gaia_db_path=_gdb_fill,
        )
        if _n_bp_miss > 0:
            log_event(f"masterstars bp_rp fallback: {_n_bp_fill}/{_n_bp_miss} filled from Gaia DB")
        # G04-08: detections and raw completeness against on-chip Gaia to the matching depth
        # (one helper with the reconciliation below).
        from dao_reconcile import gaia_detection_counts  # noqa: PLC0415

        det_meta["gaia_detection_counts"] = gaia_detection_counts(
            df_final, _chip, depth_g=_census_depth_g
        )
        if _gdb_fill:
            from dao_reconcile import (  # noqa: PLC0415
                annotate_dao_only_magnitude_classes,
                compute_gaia_dao_reconcile,
                format_dao_only_census_log,
            )

            _fwhm_cls = _masterstar_fwhm_px(det_meta, hdr)
            _noise_cls = det_meta.get("dao_flux_noise_adu")
            _nax1_cls = int(hdr.get("NAXIS1") or 0)
            _nax2_cls = int(hdr.get("NAXIS2") or 0)
            _wcs_cls = None
            _plate_cls = None
            if _has_valid_wcs(hdr) and _nax1_cls > 0 and _nax2_cls > 0:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FITSFixedWarning)
                    _wcs_cls = WCS(hdr)
                _plate_cls = plate_scale_arcsec_per_px_from_wcs(_wcs_cls)
            if _wcs_cls is not None:
                _recon_ms = compute_gaia_dao_reconcile(
                    df_final,
                    frame_depth=det_meta.get("frame_depth"),
                    gaia_db_path=_gdb_fill,
                    wcs=_wcs_cls,
                    naxis1=_nax1_cls,
                    naxis2=_nax2_cls,
                    fwhm_px=_fwhm_cls,
                    plate_scale_arcsec=_plate_cls,
                    match_sep_arcsec=float(
                        det_meta.get("match_sep_arcsec_effective")
                        or det_meta.get("match_sep_arcsec_requested")
                        or 8.0
                    ),
                    gaia_on_chip=_chip,
                    matching_depth_g=_census_depth_g,
                )
            df_final, _dao_class_meta = annotate_dao_only_magnitude_classes(
                df_final,
                gaia_db_path=_gdb_fill,
                frame_depth=det_meta.get("frame_depth"),
                frame_noise_adu=_noise_cls,
            )
            if _recon_ms is not None:
                _recon_ms["dao_only_class_meta"] = _dao_class_meta
            _ms_info_msg = format_dao_only_census_log(_dao_class_meta, n_total=len(df_final))
            LOGGER.info(_ms_info_msg)
            log_event(_ms_info_msg)
        else:
            try:
                from invariants_runtime import dao_only_fraction_from_masterstars  # noqa: PLC0415

                _frac_ms = float(dao_only_fraction_from_masterstars(df_final))
                _n_dao_ms = int(round(_frac_ms * float(len(df_final))))
                _ms_info_msg = (
                    f"MASTERSTAR DAO_ONLY census: {_n_dao_ms}/{len(df_final)} "
                    f"(fraction={_frac_ms:.3f}) -- informational, not a gate"
                )
                LOGGER.info(_ms_info_msg)
                log_event(_ms_info_msg)
            except Exception as _ms_census_exc:  # noqa: BLE001
                LOGGER.debug("[MASTERSTAR-DAO-CENSUS] skipped: %s", _ms_census_exc)
    except Exception as exc:  # noqa: BLE001
        from invariants_runtime import InvariantViolation  # noqa: PLC0415

        if isinstance(exc, InvariantViolation):
            raise
        _cone_gaia_fail = Path(platesolve_dir) / "field_catalog_cone.csv"
        if _cone_gaia_fail.is_file() and _has_valid_wcs(hdr):
            raise RuntimeError(
                f"MASTERSTAR Gaia-complete enrich failed on production path: {exc!s}"
            ) from exc
        LOGGER.exception("[M1] MASTERSTAR source_type annotate failed: %s", exc)
        log_event(f"MASTERSTAR source_type annotate failed: {exc!s}")
    return _dao_class_meta, _recon_ms, df_final


def _ms_write_catalog_and_meta(
    *,
    _cfg_ms,
    _dao_class_meta,
    _identity_qa,
    _recon_ms,
    _wcs_rt_p99,
    _wcs_rt_pass,
    csv_path,
    det_meta,
    df_final,
    masterstar_fits,
    platesolve_dir,
):
    """Write the final catalogue, log the match rates, merge the MASTERSTAR facts into pipeline_meta, stamp VY_NDAO."""
    _vyvar_df_to_csv(df_final, csv_path)
    # G04-08: detections only (catalogue-membership rows are not detections); the raw Gaia->DAO
    # completeness is against on-chip Gaia to the matching depth (dao_reconcile.gaia_detection_counts).
    from dao_reconcile import gaia_detection_counts  # noqa: PLC0415

    _counts = det_meta.get("gaia_detection_counts") or gaia_detection_counts(df_final, None, depth_g=None)
    _n_det = int(_counts["n_detection_rows"])
    _n_mat = int(_counts["n_gaia_detection_rows"])
    _rate = (100.0 * float(_n_mat) / float(_n_det)) if _n_det > 0 else 0.0
    _cat_rows_opt = int(det_meta.get("catalog_rows", 0) or 0)
    _n_ref = _counts.get("n_gaia_reference")
    _n_gaia_det_opt = _counts.get("n_gaia_detected")
    _gaia_rate_opt = _counts.get("completeness_raw_pct")
    log_event(
        f"[chart] MATCH STATS: Found {_n_det} stars on image | {_n_mat} matched with Gaia | "
        f"Match Rate: {_rate:.2f}% | Gaia->DAO: "
        + (
            f"{_gaia_rate_opt:.2f}% ({_n_gaia_det_opt}/{_n_ref} on-chip Gaia to "
            f"G<={float(_counts['reference_depth_g']):.2f})"
            if _gaia_rate_opt is not None
            else "n/a"
        )
    )
    log_event(
        f"MASTERSTAR JSON consistency: n_raw_dao={int(det_meta.get('n_detected_dao_raw', 0) or 0)}, "
        f"detected_stars={_n_det}, catalog_matched={_n_mat}, catalog_rows_total={len(df_final)}, "
        f"gaia_dao_completeness_raw_pct={_gaia_rate_opt}, "
        f"n_gaia_undetected={_counts.get('n_gaia_undetected')}"
    )
    # TODO-25: persist to pipeline_meta.json so UI can read single source of truth
    if _cat_rows_opt > 0:
        _meta_patch: dict[str, Any] = {
            "catalog_rows": int(_cat_rows_opt),
        }
        if _gaia_rate_opt is not None:
            _meta_patch.update(
                {
                    "gaia_dao_completeness_raw_pct": float(_gaia_rate_opt),
                    "n_gaia_reference": int(_n_ref),
                    "gaia_reference_depth_g": float(_counts["reference_depth_g"]),
                    "n_gaia_detected": int(_n_gaia_det_opt),
                    "n_gaia_undetected": int(_counts["n_gaia_undetected"]),
                }
            )
        _idg_stamp = dict(det_meta.get("identity_gate") or {})
        _gmeta = det_meta.get("gaia_census_meta") or {}
        if isinstance(_gmeta, dict) and "n_lock_geometry_reject" in _gmeta:
            _idg_stamp["n_lock_geometry_reject"] = int(_gmeta.get("n_lock_geometry_reject") or 0)
        if _idg_stamp:
            _meta_patch["identity_gate"] = _idg_stamp
        for _mk in (
            "match_sep_arcsec_requested",
            "match_sep_arcsec_effective",
            "match_sep_formula_inputs",
        ):
            if det_meta.get(_mk) is not None:
                _meta_patch[_mk] = det_meta.get(_mk)
        if det_meta.get("frame_depth"):
            _meta_patch["frame_depth"] = {
                k: v for k, v in dict(det_meta["frame_depth"]).items() if k != "completeness_bins"
            }
        if det_meta.get("gaia_topup"):
            _meta_patch["gaia_topup"] = dict(det_meta["gaia_topup"])
        if _wcs_rt_p99 is not None:
            _meta_patch["wcs_roundtrip_p99_px"] = float(_wcs_rt_p99)
            _meta_patch["wcs_roundtrip_pass"] = bool(_wcs_rt_pass)
        if _identity_qa:
            _inv_wcs_recs = _identity_qa.pop("_inv_wcs_01", None)
            _meta_patch.update(_identity_qa)
            if _inv_wcs_recs:
                _meta_patch.setdefault("invariants", [])
                if isinstance(_meta_patch["invariants"], list):
                    _meta_patch["invariants"].extend(list(_inv_wcs_recs))
        try:
            if _recon_ms is not None:
                _meta_patch.update(reconcile_to_pipeline_meta(_recon_ms))
            elif _dao_class_meta:
                from dao_reconcile import dao_only_class_meta_flat  # noqa: PLC0415

                _meta_patch.update(dao_only_class_meta_flat(_dao_class_meta))
            else:
                # The reconciliation runs once, above (FRAME-DEPTH-2026-10); without it only the raw rate.
                _meta_patch["gaia_dao_completeness_pct"] = _gaia_rate_opt
        except Exception as exc:  # noqa: BLE001
            log_event(f"MASTERSTAR Gaia reconcile decomposition skipped: {exc!s}")
            _meta_patch["gaia_dao_completeness_pct"] = _gaia_rate_opt
        if det_meta.get("dao_false_detection"):
            _meta_patch["dao_false_detection"] = dict(det_meta["dao_false_detection"])
        merge_photometry_pipeline_meta(
            Path(platesolve_dir) / "photometry",
            _meta_patch,
            _cfg_ms,
            entry_point="generate_masterstar_and_catalog",
        )
        # INV-DAG-01: masterstar stage stamp (cold-start OK if earlier stages absent).
        try:
            from invariants_runtime import stamp_stage_on_disk  # noqa: PLC0415

            stamp_stage_on_disk(
                Path(platesolve_dir) / "photometry",
                "masterstar",
                enforce_upstream=True,
            )
        except Exception as _dag_exc:  # noqa: BLE001
            logging.debug("[INV-DAG-01] masterstar stamp skipped: %s", _dag_exc)
    log_event(
        f"MASTERSTAR catalog: {Path(csv_path).name} - {len(df_final)} rows "
        f"(DAO + catalog over the full field; no cut by distance from the frame centre)."
    )
    # FWHM-ONE-2026-10: VY_FWHM is the one FWHM (frame QC elliptical Gaussian fit); no second
    # FWHM measurement or copy on MASTERSTAR.
    try:
        _ms_path = Path(masterstar_fits)
        with fits.open(_ms_path, mode="update", memmap=False) as _hdul:
            _n_raw_dao_hdr = int(det_meta.get("n_detected_dao_raw", 0) or 0)
            _hdul[0].header["VY_NDAO"] = (
                _n_raw_dao_hdr,
                "Raw DAO detections on MASTERSTAR",
            )
            _hdul.flush()
    except Exception as e:  # noqa: BLE001
        log_event(f"[ERROR] VY_NDAO stamp failed: {e}")
    return _n_det, _n_mat


def _ms_write_plan_and_summary(
    *,
    _cfg_ms,
    _match_sep_eff,
    _ms_max_catalog_rows_eff,
    _n_det,
    _n_mat,
    csv_path,
    det_meta,
    df_final,
    info,
    masterstar_fits,
    ms_selection_meta,
    n_comparison_stars,
    platesolve_dir,
    require_non_variable_comparisons,
    solve_meta,
):
    """Photometry plan files, VSX stamp and the MASTERSTAR result dict."""
    photo_plan = write_photometry_plan_files(
        platesolve_dir=platesolve_dir,
        masterstar_fits=masterstar_fits,
        masterstars_csv=csv_path,
        n_comparison_stars=int(n_comparison_stars),
        require_non_variable=bool(require_non_variable_comparisons),
        enrichment_expected=False,
        cfg=_cfg_ms,
    )
    # F-431 / C1: stamp AFTER VT exists (write_photometry_plan_files).
    try:
        _vt_stamp_path = platesolve_dir / "variable_targets.csv"
        if _vt_stamp_path.is_file():
            from photometry_shared import stamp_vsx_known_variable_on_masterstars  # noqa: PLC0415

            _ms_for_stamp = read_vyvar_csv(csv_path, low_memory=False)
            _vt_stamp_df = read_vyvar_csv(_vt_stamp_path, low_memory=False)
            _ms_for_stamp, _vsx_stamp_final = stamp_vsx_known_variable_on_masterstars(
                _ms_for_stamp,
                _vt_stamp_df,
                log_fn=log_event,
            )
            _vyvar_df_to_csv(_ms_for_stamp, csv_path)
            df_final = _ms_for_stamp
            log_event(
                f"MASTERSTAR VSX catalog_id stamp (post-VT): "
                f"id_join={_vsx_stamp_final.get('id_join')} "
                f"positional_fallback={_vsx_stamp_final.get('positional_fallback')}"
            )
        else:
            log_event("MASTERSTAR VSX stamp skipped: variable_targets.csv missing after photometry plan.")
    except Exception as _vsx_final_exc:  # noqa: BLE001
        log_event(f"MASTERSTAR VSX catalog_id stamp (post-VT) skipped: {_vsx_final_exc!s}")
    # G03-09 (Milan 2026-10-29): each setup keeps its own comparison pool (no copy between filters)

    out: dict[str, Any] = {
        "masterstar_fits": str(masterstar_fits),
        "masterstars_csv": str(csv_path),
        "frames_used": int(info.get("frames_used", 0)),
        "masterstar_selection": ms_selection_meta or None,
        "n_raw_dao": int(det_meta.get("n_detected_dao_raw", 0) or 0),
        "detected_stars": int(_n_det),
        "catalog_matched": int(_n_mat),
        "catalog_rows": int(det_meta.get("catalog_rows", 0)),
        "catalog_match_max_sep_arcsec": float(_match_sep_eff),
        "max_catalog_rows": int(_ms_max_catalog_rows_eff),
        "n_likely_saturated": int(det_meta.get("n_likely_saturated", 0)),
        "saturate_limit_adu": det_meta.get("saturate_limit_adu"),
        "saturate_limit_source": det_meta.get("saturate_limit_source"),
        "solve": solve_meta,
        "n_comparison_stars_requested": int(n_comparison_stars),
        "field_catalog_cone_csv": det_meta.get("field_catalog_cone_csv"),
        "catalog_derived_membership": det_meta.get("catalog_derived_membership"),
        "dao_threshold_sigma": det_meta.get("dao_threshold_sigma"),
        "masterstar_match_png": "",
    }
    out.update(photo_plan)
    return df_final, out


def _ms_enrich_from_gaia_db(
    *,
    _cfg_ms,
    csv_path,
    det_meta,
    df_final,
    equipment_id,
    equipment_saturate_adu,
    hdr,
    masterstar_fits,
    n_comparison_stars,
    out,
    platesolve_dir,
    require_non_variable_comparisons,
):
    """Enrichment of the MASTERSTAR catalogue (likely_nonlinear, NSS, G flux error) and the photometry-plan rewrite."""

    # G04-09: every row from itself - own peak for likely_nonlinear, NSS / G flux error by catalog_id from the
    # run's Gaia cone (the former second DB box query + 2" match let a DAO_ONLY neighbour overwrite a verdict)
    from masterstars_enrichment import enrich_masterstars_by_identity  # noqa: PLC0415

    sat_limit = det_meta.get("saturate_limit_adu")
    if sat_limit is None:
        _eq_sat_ms = equipment_saturate_adu
        if _eq_sat_ms is None and equipment_id is not None:
            _eq_sat_ms = _equipment_saturate_adu_from_db(equipment_id, _cfg_ms)
        sat_limit, _ = _effective_saturation_limit(hdr, fallback_adu=None, equipment_saturate_adu=_eq_sat_ms)
    if sat_limit is not None and math.isfinite(float(sat_limit)) and float(sat_limit) > 0:
        sat_thr = float(sat_limit) * float(LINEARITY_FRAC)
    else:
        # INV-SAT-LIMIT: never admit against +inf.
        sat_thr = float(SAT_LIMIT_CONTAINER_CLIP_ADU) * float(SAT_LIMIT_NO_KNEE_FRAC)
        logging.warning("[INV-SAT-LIMIT] MASTERSTAR sat_thr unresolved; using peak-test %.1f ADU (%.2f x container clip)",
                        sat_thr, float(SAT_LIMIT_NO_KNEE_FRAC))
    _cone_p = det_meta.get("field_catalog_cone_csv")
    _cone = _read_field_cone_csv(Path(_cone_p)) if _cone_p and Path(_cone_p).is_file() else None
    df_final = enrich_masterstars_by_identity(df_final, _cone, sat_thr)
    _vyvar_df_to_csv(df_final, csv_path)
    try:
        _wp2 = write_photometry_plan_files(
            platesolve_dir=platesolve_dir,
            masterstar_fits=masterstar_fits,
            masterstars_csv=csv_path,
            n_comparison_stars=int(n_comparison_stars),
            require_non_variable=bool(require_non_variable_comparisons),
            cfg=_cfg_ms,
        )
        out.update(_wp2)
    except Exception as _wp2_exc:  # noqa: BLE001
        log_event(
            f"MASTERSTAR: enriched photometry-plan rewrite failed ({_wp2_exc!s}); "
            "keeping the prior photometry plan."
        )


def generate_masterstar_and_catalog(
    *,
    archive_path: Path,
    source_root: Path | None = None,
    platesolve_dir: Path | None = None,
    n_comparison_stars: int = 0,
    require_non_variable_comparisons: bool = True,
    equipment_saturate_adu: float | None = None,
    app_config: AppConfig,
    equipment_id: int | None = None,
    draft_id: int | None = None,
    telescope_id: int | None = None,
    masterstar_candidate_paths: "Sequence[str] | None" = None,
    masterstar_selection_pct: float | None = None,
    setup_name: str | None = None,
    masterstar_basename: str = "MASTERSTAR.fits",
    masterstars_csv_basename: str = "masterstars_full_match.csv",
    masterstar_fits_only: bool = False,
    masterstar_skip_build: bool = False,
    masterstar_platesolve_only: bool = False,
    masterstar_platesolve_skip_solve: bool = False,
    hint_ra_deg: float | None = None,
    hint_dec_deg: float | None = None,
) -> dict[str, Any]:
    """Create MASTERSTAR.fits, plate-solve it, and export masterstars.csv.

    If ``masterstar_fits_only=True``, stop after building the FITS in ``platesolve/`` (no plate-solve, no CSV).
    If ``masterstar_skip_build=True``, skip the build from processed - use the existing ``MASTERSTAR.fits`` in
    ``platesolve/`` and run solver + catalog.
    If ``masterstar_platesolve_only=True``, stop after a successful plate-solve and WCS scale adjustment (no DAO CSV,
    ``masterstars_full_match.csv``, photometric plan or MASTER_SOURCES write).
    """
    ap = Path(archive_path).expanduser()
    # Draft UI may pass .../draft_x/non_calibrated - MASTERSTAR and platesolve belong under the draft root.
    if ap.name.casefold() == "non_calibrated":
        ap = ap.parent
    # SAT-PRECAL-2026-10 (Milan 2026-10-13): the saturation level is measured on this draft's
    # own light frames (SAT-DIAG) BEFORE the MASTERSTAR zones are set; a measured level wins
    # over EQUIPMENTS / header values (SAT-LIN-2026-10). Before, the MASTERSTAR read
    # sat_diag.json, which is written later (per-frame export) - a first run of a
    # non_calibrated draft (532) set its zones from a guessed 65535 clip.
    _ms_measured_sat, equipment_saturate_adu = _ms_measure_draft_saturation(
        ap=ap,
        equipment_saturate_adu=equipment_saturate_adu,
    )
    _match_sep_eff, info, masterstar_fits, ms_selection_meta, platesolve_dir = _ms_build_or_load_masterstar(
        ap=ap,
        app_config=app_config,
        draft_id=draft_id,
        masterstar_basename=masterstar_basename,
        masterstar_candidate_paths=masterstar_candidate_paths,
        masterstar_selection_pct=masterstar_selection_pct,
        masterstar_skip_build=masterstar_skip_build,
        platesolve_dir=platesolve_dir,
        setup_name=setup_name,
        source_root=source_root,
    )
    _selected_ref_path, = _ms_selected_reference_path(info=info)

    if masterstar_fits_only:
        _cfg_fast = app_config
        try:
            _ms_out = str(Path(masterstar_fits).resolve())
        except OSError:
            _ms_out = str(masterstar_fits)
        log_event(
            f"MASTERSTAR (FITS only, no plate-solve): written {_ms_out} | "
            f"frames used={info.get('frames_used', '?')}"
        )
        out_fast: dict[str, Any] = {
            "masterstar_fits": _ms_out,
            "masterstars_csv": "",
            "frames_used": int(info.get("frames_used", 0)),
            "masterstar_selection": ms_selection_meta or None,
            "masterstar_build_info": info,
            "n_raw_dao": 0,
            "detected_stars": 0,
            "catalog_matched": 0,
            "catalog_rows": 0,
            "catalog_match_max_sep_arcsec": float(_match_sep_eff),
            "solve": {"skipped": True, "reason": "masterstar_fits_only"},
        }
        _store_masterstar_path(_cfg_fast, draft_id, _ms_out, out_fast)
        return out_fast

    # Solve WCS (MASTERSTAR): exclusively the VYVAR local Gaia solver (no ASTAP / astrometry.net).

    _cfg_ms, _eq_ms, _full_db, _plate_scale_ms, data, hdr = _ms_plate_scale_hint(
        app_config=app_config,
        draft_id=draft_id,
        equipment_id=equipment_id,
        masterstar_fits=masterstar_fits,
        telescope_id=telescope_id,
    )

    data, hdr, solve_meta = _ms_solve_wcs(
        _cfg_ms=_cfg_ms,
        _full_db=_full_db,
        _plate_scale_ms=_plate_scale_ms,
        data=data,
        draft_id=draft_id,
        hdr=hdr,
        hint_dec_deg=hint_dec_deg,
        hint_ra_deg=hint_ra_deg,
        masterstar_fits=masterstar_fits,
        masterstar_platesolve_skip_solve=masterstar_platesolve_skip_solve,
    )

    _ms_stamp_plate_scale_and_check_wcs(
        _plate_scale_ms=_plate_scale_ms,
        data=data,
        hdr=hdr,
        masterstar_fits=masterstar_fits,
        solve_meta=solve_meta,
    )

    if masterstar_platesolve_only:
        _cfg_early = app_config
        try:
            _ms_out_early = str(Path(masterstar_fits).resolve())
        except OSError:
            _ms_out_early = str(masterstar_fits)
        log_event(
            f"ONLY MASTER (test): plate-solve + WCS scale adjustment done -> {_ms_out_early} "
            "(skipping DAO export, masterstars CSV, photometric plan, MASTER_SOURCES)."
        )
        out_ps: dict[str, Any] = {
            "masterstar_fits": _ms_out_early,
            "masterstars_csv": "",
            "frames_used": int(info.get("frames_used", 0)),
            "masterstar_selection": ms_selection_meta or None,
            "masterstar_build_info": info,
            "n_raw_dao": 0,
            "detected_stars": 0,
            "catalog_matched": 0,
            "catalog_rows": 0,
            "catalog_match_max_sep_arcsec": float(_match_sep_eff),
            "solve": solve_meta,
            "masterstar_platesolve_only": True,
            "comparison_stars_csv": "",
            "variable_targets_csv": "",
            "photometry_plan_json": "",
        }
        _store_masterstar_path(_cfg_early, draft_id, _ms_out_early, out_ps)
        return out_ps

    # _cfg_ms / _dao_sigma_eff set above (same DAO sigma for plate solve and catalog).
    _dao_fwhm_bypass_hdr, dao_fwhm_px_for_ms, data, hdr = _ms_load_for_detection(
        data=data,
        hdr=hdr,
        masterstar_fits=masterstar_fits,
        ms_selection_meta=ms_selection_meta,
        platesolve_dir=platesolve_dir,
    )

    # Full-field MASTERSTAR depth: keep deeper Gaia and larger catalog rows for corner recovery.
    _ms_max_catalog_rows_eff, csv_path, det_meta, df_final = _ms_detect_and_match(
        _cfg_ms=_cfg_ms,
        _dao_fwhm_bypass_hdr=_dao_fwhm_bypass_hdr,
        _eq_ms=_eq_ms,
        _full_db=_full_db,
        _ms_measured_sat=_ms_measured_sat,
        _selected_ref_path=_selected_ref_path,
        dao_fwhm_px_for_ms=dao_fwhm_px_for_ms,
        data=data,
        draft_id=draft_id,
        equipment_saturate_adu=equipment_saturate_adu,
        hdr=hdr,
        masterstars_csv_basename=masterstars_csv_basename,
        platesolve_dir=platesolve_dir,
    )
    _identity_qa, _wcs_rt_p99, _wcs_rt_pass, df_final = _ms_wcs_roundtrip_and_identity_qa(
        _cfg_ms=_cfg_ms,
        df_final=df_final,
        masterstar_fits=masterstar_fits,
    )
    # DAO-GAIA-ERA-01 M1: expand detection table to catalog-derived membership before zones/enrich.
    # INV-MS-EXPAND-01: when cone+WCS exist, expand must succeed or raise (no silent skip).
    _chip_ms, _membership_expand_meta, df_final = _ms_expand_catalog_membership(
        _cfg_ms=_cfg_ms,
        _ms_measured_sat=_ms_measured_sat,
        det_meta=det_meta,
        df_final=df_final,
        equipment_saturate_adu=equipment_saturate_adu,
        hdr=hdr,
        platesolve_dir=platesolve_dir,
    )
    _dao_class_meta, _recon_ms, df_final = _ms_gaia_accounting_and_reconcile(
        _cfg_ms=_cfg_ms,
        _chip_ms=_chip_ms,
        _membership_expand_meta=_membership_expand_meta,
        det_meta=det_meta,
        df_final=df_final,
        hdr=hdr,
        platesolve_dir=platesolve_dir,
    )
    _n_det, _n_mat = _ms_write_catalog_and_meta(
        _cfg_ms=_cfg_ms,
        _dao_class_meta=_dao_class_meta,
        _identity_qa=_identity_qa,
        _recon_ms=_recon_ms,
        _wcs_rt_p99=_wcs_rt_p99,
        _wcs_rt_pass=_wcs_rt_pass,
        csv_path=csv_path,
        det_meta=det_meta,
        df_final=df_final,
        masterstar_fits=masterstar_fits,
        platesolve_dir=platesolve_dir,
    )
    df_final, out = _ms_write_plan_and_summary(
        _cfg_ms=_cfg_ms,
        _match_sep_eff=_match_sep_eff,
        _ms_max_catalog_rows_eff=_ms_max_catalog_rows_eff,
        _n_det=_n_det,
        _n_mat=_n_mat,
        csv_path=csv_path,
        det_meta=det_meta,
        df_final=df_final,
        info=info,
        masterstar_fits=masterstar_fits,
        ms_selection_meta=ms_selection_meta,
        n_comparison_stars=n_comparison_stars,
        platesolve_dir=platesolve_dir,
        require_non_variable_comparisons=require_non_variable_comparisons,
        solve_meta=solve_meta,
    )
    # Comp-selection columns of masterstars_full_match.csv (MS-SOURCES-RETIRE): likely_nonlinear
    # (SAT-LIN-2026-10: peak >= LINEARITY_FRAC x saturation) and the Gaia quality flags of the star
    # matched within 2". The photometry plan is rewritten with them (the first pass above runs without).
    # CODE-REVIEW-2026-10 step 6: the safe-comp / stress-test / phot-category columns had no reader.
    _ms_enrich_from_gaia_db(
        _cfg_ms=_cfg_ms,
        csv_path=csv_path,
        det_meta=det_meta,
        df_final=df_final,
        equipment_id=equipment_id,
        equipment_saturate_adu=equipment_saturate_adu,
        hdr=hdr,
        masterstar_fits=masterstar_fits,
        n_comparison_stars=n_comparison_stars,
        out=out,
        platesolve_dir=platesolve_dir,
        require_non_variable_comparisons=require_non_variable_comparisons,
    )
    # Persist MASTERSTAR path on draft for later UI reloads / Step 3 continuity.
    _store_masterstar_path(_cfg_ms, draft_id, str(Path(masterstar_fits).resolve()), out)
    return out
