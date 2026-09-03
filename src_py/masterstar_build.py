"""Moved from pipeline.py (CONSOLIDATE-01E6b). Facade re-exports this name.

generate_masterstar_and_catalog: masterstar FITS build, plate-solve,
comparison-star sync, VSX/exo merge.
"""
from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Sequence

import time
import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.wcs import FITSFixedWarning
import astropy.units as u
import pandas as pd

from config import AppConfig
from database import VyvarDatabase, query_local_gaia
from infolog import log_event
from optics_selection import resolve_optics_ids_for_platesolve
from gaia_catalog_id import read_vyvar_csv
from masterstar_context import header_core_fwhm_px
from photometry_core import _fwhm_moment_at, merge_photometry_pipeline_meta
from dao_reconcile import reconcile_to_pipeline_meta
from photometry import (
    common_field_intersection_bbox_px,
    recommended_aperture_by_color,
    stress_test_relative_rms_from_sidecars,
    vsx_is_known_variable_top3_per_bin,
)
from pipeline_astrometry import _fill_masterstars_gaia_matched_bp_rp_from_local_db
from pipeline_constants import (
    SAT_LIMIT_CONTAINER_CLIP_ADU,
    SAT_LIMIT_NO_KNEE_FRAC,
    _MASTERSTAR_OPTIMIZER_MIRROR_EXTRA_LOG,
    _MASTERSTAR_PLATESOLVE_NN_REFINE_MAX_RMS_PX,
    _MASTERSTAR_PLATESOLVE_PREWRITE_RELAXED_RMS_MAX_PX,
    _MASTERSTAR_PLATESOLVE_PREWRITE_RMS_MAX_PX,
    _MASTERSTAR_SIP_FORCE_RMS_GUARD_RATIO,
    _MASTERSTAR_SOLVER_USE_DRAFT_MEDIAN_IF_HINT_SEP_DEG,
    _PLATESOLVE_ANISOTROPY_THRESHOLD,
)
from utils import (
    iter_fits_paths_recursive as _iter_fits_recursive,
    fits_binning_xy_from_header,
    masterstar_wcs_quality,
)
from vyvar_platesolver import pointing_hint_from_header as _pointing_hint_from_header
from plain_stats import plain_mean_med_std
from pipeline_calibrate import (
    _effective_saturation_limit,
    _has_valid_wcs,
    draft_median_pointing_icrs_deg,
)
from pipeline_preprocess import resolve_obs_file_to_processed_fits
from pipeline_astrometry import (
    _equipment_saturate_adu_from_db,
    _merge_platesolve_gaia_pairs_into_masterstars_df,
    _path_is_under_tree,
    _path_segments_forbidden_for_masterstar_physical_source,
    _plate_solve_input_bundle,
    _resolve_best_effort_path_under,
    _sat_adu_from_draft_sat_diag,
    _sync_comparison_stars_across_setups,
    _try_rescale_masterstar_linear_wcs_to_expected_plate_scale,
    _update_masterstar_obs_file_status,
    _vyvar_df_to_csv,
    _vyvar_open_database,
    build_masterstar_from_detrended,
    compute_plate_scale_from_db,
    draft_is_multi_group_obs,
    get_masterstar_candidate_rows,
    get_masterstar_candidates,
    resolve_masterstar_input_root,
    resolve_plate_solve_fov_deg_hint,
    write_photometry_plan_files,
)
from pipeline_catalog import (
    _annotate_masterstars_flux_zones,
    _effective_field_catalog_cone_radius_deg,
    _invalidate_field_catalog_cone_cache_if_needed,
)

LOGGER = logging.getLogger("pipeline")


def generate_masterstar_and_catalog(

    *,
    archive_path: Path,
    max_catalog_rows: int = 12000,
    astrometry_api_key: str | None = None,
    source_root: Path | None = None,
    platesolve_dir: Path | None = None,
    platesolve_backend: str = "vyvar",
    plate_solve_fov_deg: float = 1.0,
    catalog_match_max_sep_arcsec: float = 25.0,
    saturate_level_fraction: float = 0.999,
    n_comparison_stars: int = 0,
    require_non_variable_comparisons: bool = True,
    faintest_mag_limit: float | None = None,
    dao_threshold_sigma: float = 3.5,
    equipment_saturate_adu: float | None = None,
    catalog_local_gaia_only: bool | None = None,
    app_config: AppConfig | None = None,
    equipment_id: int | None = None,
    draft_id: int | None = None,
    telescope_id: int | None = None,
    master_dark_path: Path | str | None = None,
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

    Ak je ``masterstar_fits_only=True``, po zostaveni FITS v ``platesolve/`` sa skonci (ziadny plate-solve ani CSV).
    Ak je ``masterstar_skip_build=True``, preskoci sa build z processed - pouzije sa existujuci ``MASTERSTAR.fits`` v ``platesolve/`` a bezi solver + katalog.
    Ak je ``masterstar_platesolve_only=True``, po uspesnom plate-solve a uprave mierky WCS sa skonci (bez DAO CSV, ``masterstars_full_match.csv``, fotometrickeho planu a zapisu MASTER_SOURCES).
    """
    from pipeline import detect_stars_and_match_catalog  # noqa: PLC0415  # call-time: giant; patch analysis in Phase B

    max_catalog_rows = max(int(max_catalog_rows), 100000)
    import numpy as np

    ap = Path(archive_path).expanduser()
    # Draft UI moze poslat .../draft_x/non_calibrated - MASTERSTAR a platesolve patria pod koren draftu.
    if ap.name.casefold() == "non_calibrated":
        ap = ap.parent
    if equipment_saturate_adu is None:
        _sd_clip = _sat_adu_from_draft_sat_diag(ap)
        if _sd_clip is not None:
            equipment_saturate_adu = _sd_clip
            logging.info(
                "[INV-SAT-LIMIT] EQUIPMENTS.SATURATE_ADU missing; using sat_diag.json sat_adu=%.0f",
                _sd_clip,
            )
    detrended_root: Path | None = None
    if masterstar_skip_build:
        ps = Path(platesolve_dir) if platesolve_dir is not None else (ap / "platesolve")
        platesolve_dir = ps
        platesolve_dir.mkdir(parents=True, exist_ok=True)
        _ms_name = str(masterstar_basename or "MASTERSTAR.fits").strip() or "MASTERSTAR.fits"
        masterstar_fits = Path(platesolve_dir) / _ms_name
        if not masterstar_fits.is_file():
            raise FileNotFoundError(
                f"MASTERSTAR plate-solve: v {platesolve_dir} chyba subor {_ms_name}. "
                "Najprv spusti **MAKE MASTERSTAR** na archive alebo vytvor MASTERSTAR inak (FITS QA -> referencny snimok)."
            )
        _match_sep_eff = max(10.0, float(catalog_match_max_sep_arcsec))
        if _match_sep_eff > float(catalog_match_max_sep_arcsec) + 1e-9:
            log_event(
                f"MASTERSTAR: catalog match sep eff={_match_sep_eff:.2f} arcsec (min 10 for initial match)."
            )
        log_event(
            f"MASTERSTAR platesolve-from-disk: {masterstar_fits.resolve()} - VYVAR solver + katalog "
            "(bez noveho buildu z processed)."
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
            "reference_index": 0,
            "stacked": False,
            "frames_combined": 1,
        }
    if not masterstar_skip_build:
        # MASTERSTAR-only reads from processed/lights/setup_name (robust folder-based discovery).
        if source_root is not None:
            detrended_root = Path(source_root)
        else:
            detrended_root = resolve_masterstar_input_root(
                ap,
                setup_name=setup_name,
                app_config=app_config,
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
        log_event(f"Vstupny priecinok pre Masterstar: {Path(detrended_root).resolve()}")
        from draft_provenance import is_pre_calibrated_draft

        _pre_cal_ms = is_pre_calibrated_draft(ap, draft_id=draft_id)
        if _pre_cal_ms:
            log_event(
                "MASTERSTAR: pre-calibrated draft - candidates resolved directly under "
                f"{Path(detrended_root).resolve()} (no processed/calibrated remap)."
            )
        _match_sep_eff = max(10.0, float(catalog_match_max_sep_arcsec))
        if _match_sep_eff > float(catalog_match_max_sep_arcsec) + 1e-9:
            log_event(
                f"MASTERSTAR: catalog match sep zvyseny na {_match_sep_eff:.2f}\" "
                f"(pozadovane minimum pre pociatocny match)."
            )

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
            for rp in raw_paths:
                s = str(rp).strip()
                if not s:
                    continue
                hit = _resolve_best_effort_path_under(
                    Path(detrended_root),
                    s,
                    pre_calibrated=_pre_cal_ms,
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
                        app_config=app_config,
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
                    "MASTERSTAR: z UI/job prisli explicitne cesty k referencnemu snimku, ale ziadna sa nenasla "
                    f"ako ``processed/lights/.../proc_*.fits`` (koren vyberu: {Path(detrended_root).resolve()}). "
                    "Skontroluj preprocess, archiv a vyber vo FITS QA (potvrd znovu po **Create Archive & Do Calibration**). "
                    f"Pozadovane ({len(cand_paths)}): " + "; ".join(cand_paths[:6]) + (" ..." if len(cand_paths) > 6 else "")
                )

        _multi_obs = draft_is_multi_group_obs(ap)
        if only_ms_paths is None and draft_id is not None and not (_multi_obs and setup_name):
            _db_ms = _vyvar_open_database(app_config or AppConfig())
            if _db_ms is not None:
                try:
                    # FITS QA 'Potvrdit vyber MASTERSTAR' -> ``get_obs_draft_masterstar_source_path``
                    # (``draft manifest.MASTERSTAR_PATH`` = zdrojovy frame, nie hotovy ``MASTERSTAR.fits``).
                    # Musi ist pred automatickym top-% z ``manifest files[]``, inak sa pouzivatelsky vyber prepise.
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
                                f"MASTERSTAR: FITS QA vyber (DB source, draft {int(draft_id)}) -> {mapped_src[0].name}"
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
                                f"MASTERSTAR: vyber z DB (draft {int(draft_id)}, top {_pct_eff:g} %) -> "
                                f"{len(mapped_db)} kandidatov (najlepsi sa skopiruje do platesolve)."
                            )
                        else:
                            log_event(
                                f"MASTERSTAR: DB vyber (draft {int(draft_id)}) sa nepodarilo namapovat na FITS pod {detrended_root}."
                            )
                except Exception as exc:  # noqa: BLE001
                    # EXC-0394: T4 -- `_dbc_fw.conn.close()` cleanup only; no radiometry or frame data touched. (EXCEPT-BULK 2026-07-08)
                    logging.error('[EXC-0393] DB FWHM median fetch `pass` leaves `_ms_fwhm_fb` at config default instead of draft QC ...: %s', exc)
                    log_event(f"MASTERSTAR: DB vyber kandidatov zlyhal ({exc!s}).")
                finally:
                    try:
                        _db_ms.conn.close()
                    except Exception:  # noqa: BLE001
                        pass

        if only_ms_paths is None:
            disk_batch = _disk_stack_fallback_paths(Path(detrended_root), max_frames=8)
            if disk_batch:
                only_ms_paths = disk_batch
                ms_selection_meta = {
                    "source": "disk_fallback_stack",
                    "mapped_found": int(len(disk_batch)),
                }
                log_event(
                    f"MASTERSTAR disk fallback: {len(disk_batch)} kandidatov z disku (bez platneho QC vyberu)."
                )

        if only_ms_paths is None:
            raise FileNotFoundError(
                f"MASTERSTAR: v {detrended_root} nie su ziadne FITS pre vyber ani po UI/DB."
            )

        _cfg_stack = app_config or AppConfig()
        try:
            _ms_fwhm_fb = float(_cfg_stack.sips_dao_fwhm_px)
        except (TypeError, ValueError):
            _ms_fwhm_fb = 2.5
        if not math.isfinite(_ms_fwhm_fb) or _ms_fwhm_fb <= 0:
            _ms_fwhm_fb = 2.5
        if draft_id is not None:
            _dbc_fw = _vyvar_open_database(_cfg_stack)
            if _dbc_fw is not None:
                try:
                    _fdf = get_masterstar_candidate_rows(int(draft_id), 100.0, db=_dbc_fw)
                    if _fdf is not None and not _fdf.empty and "FWHM" in _fdf.columns:
                        _vals = pd.to_numeric(_fdf["FWHM"], errors="coerce").to_numpy(dtype=float)
                        _vals = _vals[np.isfinite(_vals) & (_vals > 0.5) & (_vals < 80.0)]
                        if _vals.size:
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
            raise FileNotFoundError(f"MASTERSTAR: v {detrended_root} nie su ziadne FITS pre vyber.")
        _cand_singletons = _cand_all[:_best_n]

        _db_ms_build: VyvarDatabase | None = None
        if draft_id is not None:
            _db_ms_build = _vyvar_open_database(_cfg_stack)
        try:
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
                        fwhm_fallback_px=float(_ms_fwhm_fb),
                        app_config=_cfg_stack,
                        draft_id=draft_id,
                        db=_db_ms_build,
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
        finally:
            if _db_ms_build is not None:
                try:
                    _db_ms_build.conn.close()
                except Exception:  # noqa: BLE001
                    pass
        try:
            _legacy_master = Path(detrended_root) / "MASTERSTAR.fits"
            if _legacy_master.is_file() and _legacy_master.resolve() != masterstar_fits.resolve():
                _legacy_master.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass
    _selected_ref_path: Path | None = None
    try:
        _rp = str(info.get("reference_path") or "").strip()
        _selected_ref_path = Path(_rp) if _rp else None
    except Exception:  # noqa: BLE001
        _selected_ref_path = None

    if masterstar_fits_only:
        _cfg_fast = app_config or AppConfig()
        try:
            _ms_out = str(Path(masterstar_fits).resolve())
        except OSError:
            _ms_out = str(masterstar_fits)
        log_event(
            f"MASTERSTAR (len FITS, bez plate-solve): zapisane {_ms_out} | "
            f"zkombinovanych snimok={info.get('frames_combined', info.get('frames_used', '?'))}"
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
        try:
            if draft_id is not None:
                _db_ms = VyvarDatabase(Path(_cfg_fast.database_path))
                try:
                    _db_ms.set_obs_draft_masterstar_fits_path(int(draft_id), _ms_out)
                finally:
                    _db_ms.conn.close()
        except Exception as exc:  # noqa: BLE001
            out_fast["masterstar_path_store_error"] = str(exc)
        return out_fast

    # Solve WCS (MASTERSTAR): vyhradne VYVAR lokalny Gaia solver (ziadny ASTAP / astrometry.net).

    with fits.open(masterstar_fits, memmap=False) as hdul:
        hdr = hdul[0].header.copy()
        data = np.array(hdul[0].data, dtype=np.float32, copy=True)

    _cfg_ms = app_config or AppConfig()

    try:
        _dao_sigma_eff = float(_cfg_ms.masterstar_dao_threshold_sigma)
    except (TypeError, ValueError):
        _dao_sigma_eff = 1.8
    if not math.isfinite(_dao_sigma_eff) or _dao_sigma_eff <= 0:
        _dao_sigma_eff = 1.8
    _dao_sigma_eff = max(0.1, min(6.0, float(_dao_sigma_eff)))
    log_event(
        f"MASTERSTAR: DAO threshold sigmaxRMS = {_dao_sigma_eff:.2f} "
        f"(config masterstar_dao_threshold_sigma; plate solve + katalog)"
    )

    _full_db = str(_cfg_ms.gaia_db_path or "").strip()
    if not _full_db:
        raise RuntimeError(
            "MASTERSTAR: v Settings nastavte gaia_db_path (plna lokalna Gaia DR3 SQLite DB)."
        )
    from vyvar_platesolver import solve_wcs_with_local_gaia

    log_event("MASTERSTAR WCS: VYVAR solver + plna Gaia DB (gaia_db_path).")
    try:
        _sip_ms = int(_cfg_ms.masterstar_platesolve_sip_max_order)
    except (TypeError, ValueError):
        _sip_ms = 5
    _sip_ms = max(2, min(5, _sip_ms))
    try:
        _sip_lo = int(_cfg_ms.masterstar_platesolve_sip_min_order)
    except (TypeError, ValueError):
        _sip_lo = 3
    _sip_lo = max(2, min(5, _sip_lo))
    if _sip_lo > _sip_ms:
        _sip_lo = _sip_ms
    log_event(
        f"MASTERSTAR: SIP skusanie {_sip_ms}->...->{_sip_lo} (config max/min plate-solve SIP)."
    )
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
                _eq_ms, _tel_ms, _db_scale.conn, binning=_bin_ms
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
            f"INFO: Plate scale z DB (Equipment+Telescope): {_auto_scale_ms:.4f} arcsec/px"
        )
    else:
        log_event(
            "WARNING: Plate scale z DB nedostupna - solver odvodi mierku z FITS alebo None"
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
    _eff_um = _bundle.get("eff_um")
    _foc_mm = _bundle.get("focal_mm")
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

    _skip_independent_solve = bool(masterstar_platesolve_skip_solve) or (
        str(hdr.get("VY_CRT", "")).strip().lower() == "sibling_recovered"
        and _has_valid_wcs(hdr)
    )
    solve_meta: dict[str, Any] = {}
    if _skip_independent_solve:
        log_event(
            "MASTERSTAR: sibling-recovered WCS on disk - skipping independent Pass-1 plate-solve."
        )
        try:
            _vy_sodd = int(hdr.get("VY_SODD", 0) or 0)
        except (TypeError, ValueError):
            _vy_sodd = 0
        solve_meta = {
            "solved": True,
            "method": "sibling_recovered",
            "match_rate": 1.0,
            "sip_meta": {
                "masterstar_verified": True,
                "route": "sibling_recovered",
                "n_matched_tight": _vy_sodd,
            },
        }

    if not _skip_independent_solve:

        _mra, _mde, _ = _pointing_hint_from_header(hdr)
        if hint_ra_deg is not None and hint_dec_deg is not None:
            try:
                _hra_ov = float(hint_ra_deg)
                _hde_ov = float(hint_dec_deg)
                if math.isfinite(_hra_ov) and math.isfinite(_hde_ov):
                    _mra, _mde = _hra_ov, _hde_ov
                    log_event(
                        "MASTERSTAR: hint_ra_deg / hint_dec_deg z volania prepisuju hint z FITS "
                        "(druhy MASTERSTAR / detrended aligned)."
                    )
            except (TypeError, ValueError):
                pass
        try:
            _hint_sep_thr = float(_MASTERSTAR_SOLVER_USE_DRAFT_MEDIAN_IF_HINT_SEP_DEG)
        except (TypeError, ValueError):
            _hint_sep_thr = 1.0
        if not math.isfinite(_hint_sep_thr) or _hint_sep_thr < 0:
            _hint_sep_thr = 1.0
        if draft_id is not None:
            _dbc_hint = _vyvar_open_database(_cfg_ms)
            if _dbc_hint is not None:
                try:
                    med_ra, med_de = draft_median_pointing_icrs_deg(_dbc_hint, int(draft_id))
                    if med_ra is not None and med_de is not None:
                        if _mra is None or _mde is None:
                            _mra, _mde = med_ra, med_de
                            log_event(
                                "MASTERSTAR solve: pouzivam median RA/Dec z manifest files[] (hlavicka bez spolahliveho hintu)."
                            )
                        else:
                            sc_h = SkyCoord(ra=float(_mra) * u.deg, dec=float(_mde) * u.deg, frame="icrs")
                            sc_d = SkyCoord(ra=float(med_ra) * u.deg, dec=float(med_de) * u.deg, frame="icrs")
                            sep = float(sc_h.separation(sc_d).deg)
                            if sep > float(_hint_sep_thr):
                                log_event(
                                    f"MASTERSTAR solve: hint vs draft median = {sep:.3f} deg > {_hint_sep_thr} deg "
                                    "- pouzivam draft median z manifest files[]."
                                )
                                _mra, _mde = med_ra, med_de
                            elif sep > 0.05:
                                log_event(
                                    f"MASTERSTAR solve: hint vs draft median = {sep:.3f} deg (skontrolujte pointing)."
                                )
                finally:
                    try:
                        _dbc_hint.conn.close()
                    except Exception:  # noqa: BLE001
                        pass

        _fov_ms_solve = resolve_plate_solve_fov_deg_hint(
            hdr,
            int(data.shape[0]),
            int(data.shape[1]),
            database_path=_cfg_ms.database_path,
            equipment_id=_eq_ms,
            draft_id=int(draft_id) if draft_id is not None else None,
        )
        if _fov_ms_solve is None:
            try:
                _pf_ms = float(plate_solve_fov_deg)
                if math.isfinite(_pf_ms) and _pf_ms > 0:
                    _fov_ms_solve = _pf_ms
            except (TypeError, ValueError):
                pass
        if _fov_ms_solve is None:
            _fov_ms_solve = float(_cfg_ms.plate_solve_fov_deg)
        _prms = _MASTERSTAR_PLATESOLVE_PREWRITE_RMS_MAX_PX
        _prms_r = _MASTERSTAR_PLATESOLVE_PREWRITE_RELAXED_RMS_MAX_PX
        _nnrms = _MASTERSTAR_PLATESOLVE_NN_REFINE_MAX_RMS_PX
        # MASTERSTAR platesolve: always single best processed FITS (copy mode).
        _ms_vyvar_max_rows = 30000

        def _run_masterstar_vyvar_solve(*, enable_sip: bool, sip_max_order: int, fov_deg: float, max_rows: int) -> dict[str, Any]:
            return solve_wcs_with_local_gaia(
                masterstar_fits,
                hint_ra_deg=_mra,
                hint_dec_deg=_mde,
                fov_diameter_deg=float(fov_deg),
                gaia_db_path=Path(_full_db),
                enable_sip=bool(enable_sip),
                sip_max_order=int(sip_max_order),
                ransac_refinement=True,
                max_catalog_rows=int(max_rows),
                faintest_mag_limit=18.0,
                dao_threshold_sigma=float(_dao_sigma_eff),
                effective_pixel_um=float(_eff_um) if _eff_um is not None else None,
                focal_length_mm=float(_foc_mm) if _foc_mm is not None else None,
                expected_plate_scale_arcsec_per_px=(
                    float(_plate_scale_ms) if _plate_scale_ms is not None else None
                ),
                masterstar_prewrite_rms_max_px=float(_prms) if _prms is not None else None,
                masterstar_prewrite_relaxed_rms_max_px=float(_prms_r) if _prms_r is not None else None,
                masterstar_nn_refine_max_rms_px=float(_nnrms) if _nnrms is not None else None,
                masterstar_sip_min_order=int(_sip_lo),
                app_config=_cfg_ms,
                solver_use_cone_for_sip=True,
                solver_fits_header_hint_sep_escape=True,
                solver_legacy_masterstar_mirror_sweep=True,
                solver_apply_roworder_yflip=False,
            )

        solve_meta = _run_masterstar_vyvar_solve(
            enable_sip=True,
            sip_max_order=int(_sip_ms),
            fov_deg=float(_fov_ms_solve),
            max_rows=int(_ms_vyvar_max_rows),
        )
        if not isinstance(solve_meta, dict) or not bool(solve_meta.get("solved", False)):
            raise RuntimeError(
                "MASTERSTAR plate-solve zlyhal. "
                f"Back-end returned: {solve_meta!r}. "
                "Cannot safely continue with photometry / source extraction."
            )

        # Refresh header/data after solve attempt (solver overwrote MASTERSTAR.fits header)
        with fits.open(masterstar_fits, memmap=False) as hdul:
            hdr = hdul[0].header.copy()
            data = np.array(hdul[0].data, dtype=np.float32, copy=True)
        if not _has_valid_wcs(hdr):
            raise RuntimeError(
                "MASTERSTAR: po plate-solve chyba platny WCS. Skontroluj gaia_db_path, RA/Dec a mierku v hlavicke "
                "(FOCALLEN/PIXSIZE alebo SECPIX) a vystup solvera."
            )

        # Pipeline-level acceptance criteria (stricter than solver's minimal guard):
        # - match_rate: allow 60% on the first solve (optimizer refines later)
        try:
            _mr = float(solve_meta.get("match_rate", 0.0) or 0.0)
        except (TypeError, ValueError):
            _mr = 0.0
        from vyvar_platesolver import MASTERSTAR_PLATESOLVE_MIN_MATCH_RATE

        _min_mr = float(MASTERSTAR_PLATESOLVE_MIN_MATCH_RATE)
        if _mr < _min_mr:
            raise RuntimeError(
                f"MASTERSTAR plate-solve zamietnuty: match_rate={_mr * 100.0:.1f}% < {_min_mr * 100.0:.0f}%. "
                "Skus zvysit n_stack alebo upravit hint/DAO prahy."
            )

        try:
            _aniso_thr = float(_PLATESOLVE_ANISOTROPY_THRESHOLD)
        except (TypeError, ValueError):
            _aniso_thr = 1.3
        if not math.isfinite(_aniso_thr) or _aniso_thr <= 0:
            _aniso_thr = 1.3
        _aniso_thr = max(1.01, min(5.0, float(_aniso_thr)))

        # Post-solve anisotropy validation: reject strongly anisotropic pixel scale and retry solver once.
        try:
            wcs0 = WCS(hdr)
            scale_x = abs(float(wcs0.pixel_scale_matrix[0, 0])) * 3600.0  # arcsec/px
            scale_y = abs(float(wcs0.pixel_scale_matrix[1, 1])) * 3600.0  # arcsec/px
            if math.isfinite(scale_x) and math.isfinite(scale_y) and scale_x > 0 and scale_y > 0:
                scale_ratio = max(scale_x, scale_y) / min(scale_x, scale_y)
            else:
                scale_ratio = float("nan")
        except Exception:  # noqa: BLE001
            scale_ratio = float("nan")

        if math.isfinite(scale_ratio) and scale_ratio > _aniso_thr:
            log_event(
                f"VAROVANIE: Anizotropna mierka ratio={scale_ratio:.2f} - plate-solve zamietnuty, restartujem solver (relaxed)."
            )
            # Retry with relaxed knobs:
            # - slightly larger FOV diameter (hint-vs-solved tolerance),
            # - more Gaia rows,
            # - no SIP (simpler model can be more stable when the fit goes off-rails).
            solve_meta2 = _run_masterstar_vyvar_solve(
                enable_sip=False,
                sip_max_order=0,
                fov_deg=float(_fov_ms_solve) * 1.25,
                max_rows=int(max(_ms_vyvar_max_rows, 30000)),
            )
            if not isinstance(solve_meta2, dict) or not bool(solve_meta2.get("solved", False)):
                raise RuntimeError(
                    f"MASTERSTAR platesolve retry zlyhal po anizotropii. Back-end returned: {solve_meta2!r}"
                )
            solve_meta = solve_meta2
            # Reload header after retry
            with fits.open(masterstar_fits, memmap=False) as hdul:
                hdr = hdul[0].header.copy()
                data = np.array(hdul[0].data, dtype=np.float32, copy=True)
            if not _has_valid_wcs(hdr):
                raise RuntimeError("MASTERSTAR: po retry plate-solve chyba platny WCS.")
            try:
                wcs1 = WCS(hdr)
                sx = abs(float(wcs1.pixel_scale_matrix[0, 0])) * 3600.0
                sy = abs(float(wcs1.pixel_scale_matrix[1, 1])) * 3600.0
                if math.isfinite(sx) and math.isfinite(sy) and sx > 0 and sy > 0:
                    scale_ratio2 = max(sx, sy) / min(sx, sy)
                else:
                    scale_ratio2 = float("nan")
            except Exception:  # noqa: BLE001
                scale_ratio2 = float("nan")
            if math.isfinite(scale_ratio2) and scale_ratio2 > _aniso_thr:
                raise RuntimeError(
                    f"MASTERSTAR plate-solve zamietnuty: anizotropna mierka po retry ratio={scale_ratio2:.2f} (>{_aniso_thr})."
                )

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
                _pm0 = _w_hdr.pixel_scale_matrix
                _sx0 = abs(float(_pm0[0, 0])) * 3600.0
                _sy0 = abs(float(_pm0[1, 1])) * 3600.0
                if math.isfinite(_sx0) and math.isfinite(_sy0) and _sx0 > 0 and _sy0 > 0:
                    _exp_scale_apx = float((_sx0 + _sy0) / 2.0)
        except Exception:  # noqa: BLE001
            _exp_scale_apx = None
    if _exp_scale_apx is None or (not math.isfinite(_exp_scale_apx)) or _exp_scale_apx <= 0:
        # derive-or-None (DR6 pattern): all principled sources exhausted - do not guess.
        _exp_scale_apx = None
        log_event(
            "WARNING: MASTERSTAR plate scale not derivable (DB/FITS/WCS exhausted) - "
            "expected scale unknown; VY_PLTS will not be written."
        )

    _wcs_ok = False
    try:
        with fits.open(masterstar_fits, memmap=False) as _hd_wq:
            _w_check = WCS(_hd_wq[0].header)
        if _exp_scale_apx is None:
            log_event(
                "MASTERSTAR WCS quality: expected plate scale unknown "
                "(no DB/FITS/WCS/config) - check skipped."
            )
            _wcs_q = None
        else:
            _wcs_q = masterstar_wcs_quality(
                _w_check, float(_exp_scale_apx), anisotropy_limit=float(_aniso_thr)
            )
            _wcs_ok = bool(_wcs_q.get("ok", False))
        if _wcs_q is not None and not _wcs_ok:
            _rq = _wcs_q.get("ratio")
            _se = _wcs_q.get("scale_err_pct")
            try:
                _rq_s = f"{float(_rq):.2f}" if _rq is not None and math.isfinite(float(_rq)) else str(_rq)
            except (TypeError, ValueError):
                _rq_s = str(_rq)
            try:
                _se_s = f"{float(_se):.1f}" if _se is not None and math.isfinite(float(_se)) else str(_se)
            except (TypeError, ValueError):
                _se_s = str(_se)
            log_event(
                f"MASTERSTAR WCS kvalita: zla (ratio={_rq_s}, scale_err={_se_s}%) - "
                "pokracujem bez externeho plate-solve (ocakava sa FITS metadata / buduci blind solver)."
            )
    except Exception as _wq_exc:  # noqa: BLE001
        log_event(f"MASTERSTAR WCS check failed: {_wq_exc}")
        _wcs_ok = False

    try:
        _pscale_adj = _try_rescale_masterstar_linear_wcs_to_expected_plate_scale(
            masterstar_fits,
            app_config=app_config or AppConfig(),
            equipment_id=equipment_id,
            draft_id=draft_id,
        )
    except Exception as exc:  # noqa: BLE001
        log_event(f"WCS PLATE SCALE: neocakavana chyba - {exc!s}")
        _pscale_adj = {"rescaled": False, "error": str(exc)}
    solve_meta["wcs_plate_scale_adjustment"] = _pscale_adj

    # Write calibrated plate scale to MASTERSTAR header
    _vy_plts = None
    try:
        if isinstance(solve_meta, dict):
            _vy_plts = solve_meta.get("plate_scale_arcsec_px")
        if _vy_plts is None and isinstance(_pscale_adj, dict):
            _vy_plts = _pscale_adj.get("new_scale_arcsec_per_px") or _pscale_adj.get(
                "expected_arcsec_per_px"
            )
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

    if masterstar_platesolve_only:
        _cfg_early = app_config or AppConfig()
        try:
            _ms_out_early = str(Path(masterstar_fits).resolve())
        except OSError:
            _ms_out_early = str(masterstar_fits)
        log_event(
            f"ONLY MASTER (test): plate-solve + uprava mierky WCS hotove -> {_ms_out_early} "
            "(preskakujem DAO export, masterstars CSV, fotometricky plan, MASTER_SOURCES)."
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
        try:
            if draft_id is not None:
                _db_early = VyvarDatabase(Path(_cfg_early.database_path))
                try:
                    _db_early.set_obs_draft_masterstar_fits_path(int(draft_id), _ms_out_early)
                finally:
                    _db_early.conn.close()
        except Exception as exc:  # noqa: BLE001
            out_ps["masterstar_path_store_error"] = str(exc)
        return out_ps

    # _cfg_ms / _dao_sigma_eff uz vyssie (rovnake DAO sigma pre plate solve aj katalog).
    _ms_fwhm = float(_cfg_ms.sips_dao_fwhm_px)
    if not math.isfinite(_ms_fwhm) or _ms_fwhm <= 0:
        _ms_fwhm = 2.5
    _ms_meta = ms_selection_meta if isinstance(ms_selection_meta, dict) else {}
    _best_fwhm = _ms_meta.get("best_frame_fwhm_px")
    try:
        _best_fwhm_f = float(_best_fwhm) if _best_fwhm is not None else float("nan")
    except (TypeError, ValueError):
        _best_fwhm_f = float("nan")
    _use_best_frame_fwhm = bool(_cfg_ms.masterstar_use_best_frame_fwhm)
    if (
        _use_best_frame_fwhm
        and math.isfinite(_best_fwhm_f)
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
            f"MASTERSTAR DAO: dao_fwhm_px={dao_fwhm_px_for_ms:.3f} from sips_dao_fwhm_px / header VY_FWHM"
        )

    with fits.open(masterstar_fits, memmap=False) as hdul:
        hdr = hdul[0].header.copy()
        data = np.array(hdul[0].data, dtype=np.float32, copy=True)
    data = np.ascontiguousarray(data, dtype=np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    # Global median (BOJ) background subtraction is intentionally OFF (handled downstream).
    _ms_mean = float(np.nanmean(data))
    log_event("MASTERSTAR: globalne odcitanie medianu (BOJ) vypnute.")
    log_event(f"MASTERSTAR po nan_to_num: mean={_ms_mean:.6f}")
    _ms_min = float(np.nanmin(data))
    _ms_max = float(np.nanmax(data))
    log_event(f"MASTERSTAR levels: noise_floor(min)={_ms_min:.2f}, saturation_proxy(max)={_ms_max:.2f}")

    if platesolve_dir is None:
        raise ValueError("generate_masterstar_and_catalog: platesolve_dir is required (got None).")
    platesolve_dir.mkdir(parents=True, exist_ok=True)
    _fov_job: float | None = None
    try:
        _fj = float(plate_solve_fov_deg)
        if math.isfinite(_fj) and _fj > 0:
            _fov_job = _fj
    except (TypeError, ValueError):
        _fov_job = None
    if _fov_job is None:
        _fov_job = resolve_plate_solve_fov_deg_hint(
            hdr,
            int(data.shape[0]),
            int(data.shape[1]),
            database_path=_cfg_ms.database_path,
            equipment_id=int(equipment_id) if equipment_id is not None else None,
            draft_id=int(draft_id) if draft_id is not None else None,
        )
    if _fov_job is None:
        _fov_job = float(_cfg_ms.plate_solve_fov_deg)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            _w_pre = WCS(hdr)
        if _w_pre.has_celestial:
            _, _r_cat_need = _effective_field_catalog_cone_radius_deg(
                _w_pre, int(data.shape[0]), int(data.shape[1]), float(_fov_job), fits_header=hdr
            )
            _invalidate_field_catalog_cone_cache_if_needed(
                platesolve_dir / "field_catalog_cone.csv",
                plate_solve_fov_deg=float(_fov_job),
                effective_radius_deg=float(_r_cat_need),
            )
    except Exception as exc:  # noqa: BLE001
        log_event(f"Katalog: kontrola cache field_catalog_cone preskocena - {exc!s}")

    # Full-field MASTERSTAR depth: keep deeper Gaia and larger catalog rows for corner recovery.
    _ms_max_catalog_rows_eff = max(int(max_catalog_rows), 100000)
    if faintest_mag_limit is None:
        _ms_faintest_mag_eff: float | None = 18.0
    else:
        try:
            _ms_faintest_mag_eff = max(float(faintest_mag_limit), 18.0)
        except (TypeError, ValueError):
            _ms_faintest_mag_eff = 18.0
    df_out, det_meta = detect_stars_and_match_catalog(
        data,
        hdr,
        max_catalog_rows=int(_ms_max_catalog_rows_eff),
        cat_df=None,
        match_sep_arcsec=float(_match_sep_eff),
        saturate_level_fraction=float(saturate_level_fraction),
        faintest_mag_limit=_ms_faintest_mag_eff,
        field_catalog_export_path=platesolve_dir / "field_catalog_cone.csv",
        dao_threshold_sigma=float(_dao_sigma_eff),
        dao_fwhm_px=dao_fwhm_px_for_ms,
        equipment_saturate_adu=equipment_saturate_adu,
        catalog_local_gaia_only=catalog_local_gaia_only,
        plate_solve_fov_deg=float(_fov_job),
        fov_database_path=_cfg_ms.database_path,
        fov_equipment_id=int(equipment_id) if equipment_id is not None else None,
        fov_draft_id=int(draft_id) if draft_id is not None else None,
        prematch_peak_sigma_floor=float(
            _cfg_ms.masterstar_prematch_peak_sigma_floor
        ),
        dao_fwhm_bypass_header=bool(_dao_fwhm_bypass_hdr),
    )
    try:
        if isinstance(solve_meta, dict) and bool(solve_meta.get("solved")):
            _px = solve_meta.get("pairs_x")
            _py = solve_meta.get("pairs_y")
            _pra = solve_meta.get("pairs_ra")
            _pde = solve_meta.get("pairs_de")
            _pids = solve_meta.get("pairs_catalog_id")
            if (
                isinstance(_px, list)
                and isinstance(_py, list)
                and isinstance(_pra, list)
                and isinstance(_pde, list)
                and isinstance(_pids, list)
                and len(_px) > 0
                and len(_px) == len(_py) == len(_pra) == len(_pde) == len(_pids)
            ):
                _sm0 = solve_meta.get("sip_meta") if isinstance(solve_meta.get("sip_meta"), dict) else {}
                _mir = str((_sm0 or {}).get("det_mirror_orientation") or "").strip()
                df_out = _merge_platesolve_gaia_pairs_into_masterstars_df(
                    df_out,
                    pairs_x=[float(t) for t in _px],
                    pairs_y=[float(t) for t in _py],
                    pairs_ra=[float(t) for t in _pra],
                    pairs_de=[float(t) for t in _pde],
                    pairs_catalog_id=[str(t) for t in _pids],
                )
                log_event(
                    f"MASTERSTAR: VYVAR pary ({len(_px)}) zlucene do katalogu "
                    f"(mirror={_mir or 'native'}, pre astrometry optimizer)."
                )
    except Exception as exc:  # noqa: BLE001
        log_event(f"MASTERSTAR: zlucenie VYVAR parov preskocene - {exc!s}")

    _fwhm_dao = float(det_meta.get("dao_fwhm_px") or 0.0)
    if not math.isfinite(_fwhm_dao) or _fwhm_dao <= 0:
        _fwhm_dao = float((det_meta.get("identity_gate") or {}).get("fwhm_px") or 1.25)

    try:
        from wcs_invertibility import (
            accumulate_identity_gate,
            apply_post_match_identity_gate_df,
            gaia_radec_map_from_table,
        )

        _gmap_ms: dict[str, tuple[float, float]] = {}
        _cone_p = Path(platesolve_dir) / "field_catalog_cone.csv"
        if _cone_p.is_file():
            _cone_df = pd.read_csv(_cone_p, low_memory=False, dtype={"catalog_id": str, "source_id": str})
            _gmap_ms.update(gaia_radec_map_from_table(_cone_df))
        if isinstance(solve_meta, dict):
            _pids = solve_meta.get("pairs_catalog_id") or []
            _pra = solve_meta.get("pairs_ra") or []
            _pde = solve_meta.get("pairs_de") or []
            from gaia_catalog_id import normalize_gaia_source_id as _norm_gid

            for _i, _pid in enumerate(_pids):
                _k = _norm_gid(str(_pid))
                if _k and _i < len(_pra) and _i < len(_pde):
                    _gmap_ms[_k] = (float(_pra[_i]), float(_pde[_i]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FITSFixedWarning)
            _w_gate = WCS(hdr)
        df_out, _idc_merge = apply_post_match_identity_gate_df(
            df_out,
            _w_gate,
            gaia_ra_dec_by_cid=_gmap_ms,
            fwhm_px=_fwhm_dao,
            log_fn=log_event,
        )
        _acc = dict(det_meta.get("identity_gate") or {})
        _n_out = int(
            df_out.get("catalog_id", pd.Series([""] * len(df_out)))
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
            .sum()
        )
        det_meta["identity_gate"] = accumulate_identity_gate(_acc, _idc_merge, _n_out)
        det_meta["identity_gate"]["fwhm_px"] = float(_fwhm_dao)
    except Exception as _mg_exc:  # noqa: BLE001
        log_event(f"post_match_identity_gate (post-merge) skipped: {_mg_exc!s}")

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
    _cat_rows = int(det_meta.get("catalog_rows", 0) or 0)
    if "catalog_id" in df_out.columns:
        _cid_raw = df_out["catalog_id"].fillna("").astype(str).str.strip()
        _n_gaia_det_raw = int(_cid_raw[_cid_raw != ""].nunique())
    else:
        _n_gaia_det_raw = int(_n_mat_raw)
    _gaia_rate_raw = (100.0 * float(_n_gaia_det_raw) / float(_cat_rows)) if _cat_rows > 0 else 0.0
    log_event(
        f"[chart] MATCH STATS (raw): Found {_n_det_raw} stars on image | {_n_mat_raw} matched with Gaia | "
        f"Match Rate: {_rate_raw:.2f}% | Gaia->DAO: {_gaia_rate_raw:.2f}% ({_n_gaia_det_raw}/{_cat_rows})"
    )
    if _cat_rows > 0:
        LOGGER.info(
            "[MASTERSTAR] Gaia->DAO completeness: "
            "%d/%d (%.1f%%) | catalog_only: %d",
            _n_gaia_det_raw,
            _cat_rows,
            _gaia_rate_raw,
            _cat_rows - _n_gaia_det_raw,
        )
    _update_masterstar_obs_file_status(
        cfg=_cfg_ms,
        draft_id=draft_id,
        selected_ref_path=_selected_ref_path,
        wcs_ok=bool(_has_valid_wcs(hdr)),
        n_stars=_n_det_raw,
    )
    temp_csv = platesolve_dir / "masterstars.csv"
    _msc_name = str(masterstars_csv_basename or "masterstars_full_match.csv").strip() or "masterstars_full_match.csv"
    csv_path = platesolve_dir / _msc_name
    _vyvar_df_to_csv(df_out, temp_csv)
    _opt_stats_last: dict[str, Any] = {}
    try:
        from astrometry_optimizer import optimize_masterstar_matches

        _gdb_opt = str(_cfg_ms.gaia_db_path or "").strip()
        if _gdb_opt:
            _mir_extra = bool(_MASTERSTAR_OPTIMIZER_MIRROR_EXTRA_LOG)
            _idg_n_out = int((det_meta.get("identity_gate") or {}).get("n_matched_out") or 0)
            try:
                with fits.open(masterstar_fits, mode="update", memmap=False) as _hf_dao:
                    _hf_dao[0].header["VY_FWHM_DAO"] = (
                        float(_fwhm_dao),
                        "DAO-domain FWHM [pix] for identity gate",
                    )
                    _hf_dao.flush()
            except Exception as _dao_hdr_exc:  # noqa: BLE001
                log_event(f"MASTERSTAR VY_FWHM_DAO stamp skipped: {_dao_hdr_exc!s}")
            csv_path = optimize_masterstar_matches(
                masterstars_csv=temp_csv,
                masterstar_fits=masterstar_fits,
                gaia_db_path=_gdb_opt,
                output_csv=csv_path,
                gaia_mag_limit=float(_ms_faintest_mag_eff),
                gaia_max_catalog_rows=int(_ms_max_catalog_rows_eff),
                mirror_orientation_extra_log=_mir_extra,
                sip_force_rms_guard_ratio=_MASTERSTAR_SIP_FORCE_RMS_GUARD_RATIO,
                fwhm_dao_px=float(_fwhm_dao),
                identity_gate_n_out=_idg_n_out,
                stats_out=_opt_stats_last,
            )
            # Force one more pass after WCS displacement update for final edge recovery.
            # Identity-count contract is first-entry only; rematch may add honest pairs.
            csv_path = optimize_masterstar_matches(
                masterstars_csv=csv_path,
                masterstar_fits=masterstar_fits,
                gaia_db_path=_gdb_opt,
                output_csv=csv_path,
                gaia_mag_limit=float(_ms_faintest_mag_eff),
                gaia_max_catalog_rows=int(_ms_max_catalog_rows_eff),
                mirror_orientation_extra_log=_mir_extra,
                sip_force_rms_guard_ratio=_MASTERSTAR_SIP_FORCE_RMS_GUARD_RATIO,
                fwhm_dao_px=float(_fwhm_dao),
                identity_gate_n_out=None,
                stats_out=_opt_stats_last,
            )
            log_event("MASTERSTAR optimizer: forced final re-match pass completed.")
            # Final safety: repair any residual precision-loss IDs in masterstars_full_match.csv via Gaia RA/DEC lookup.
            try:
                from repair_catalog_ids import repair_csv_catalog_ids_from_gaia_db  # noqa: PLC0415

                rep = repair_csv_catalog_ids_from_gaia_db(
                    csv_path=Path(csv_path),
                    gaia_db_path=Path(_gdb_opt),
                    id_col="catalog_id",
                    backup=True,
                    max_sep_arcsec=2.0,
                    log_fn=log_event,
                    skip_unmatched_placeholders=True,
                )
                if int(rep.get("repaired") or 0) > 0:
                    log_event(
                        f"MASTERSTAR repair: repaired={rep.get('repaired')} warnings={rep.get('warnings')} ({Path(csv_path).name})"
                    )
            except Exception as _rep_exc:  # noqa: BLE001
                log_event(f"MASTERSTAR repair skipped: {_rep_exc!s}")
        else:
            _vyvar_df_to_csv(df_out, csv_path)
    except Exception as exc:  # noqa: BLE001
        from invariants_runtime import InvariantViolation as _InvMatchId  # noqa: PLC0415

        if isinstance(exc, _InvMatchId):
            raise
        log_event(f"MASTERSTAR optimizer skipped/fallback: {exc!s}")
        _vyvar_df_to_csv(df_out, csv_path)
    try:
        # Critical: keep Gaia IDs as strings (avoid float/scientific precision loss).
        df_final = pd.read_csv(csv_path, low_memory=False, dtype={"catalog_id": str, "name": str})
        # Preserve DAO pass provenance through astrometry_optimizer CSV round-trip.
        for _prov_col in ("vy_dao_pass", "ambiguous_owner"):
            if _prov_col not in df_out.columns:
                continue
            _cid_out = df_out.get("catalog_id", pd.Series([""] * len(df_out))).map(
                lambda c: str(c).strip()
            )
            _map = df_out.assign(_cid=_cid_out).drop_duplicates("_cid", keep="last").set_index("_cid")[
                _prov_col
            ]
            _cid_fin = df_final.get("catalog_id", pd.Series([""] * len(df_final))).map(
                lambda c: str(c).strip()
            )
            df_final[_prov_col] = _cid_fin.map(_map)
            if _prov_col == "vy_dao_pass":
                df_final[_prov_col] = pd.to_numeric(df_final[_prov_col], errors="coerce").fillna(1)
            elif _prov_col == "ambiguous_owner":
                df_final[_prov_col] = df_final[_prov_col].fillna(False).astype(bool)
        if len(df_final) == len(df_out):
            for _idcol in ("vy_identity_gate", "gaia_dao_resid_px"):
                if _idcol in df_out.columns and _idcol not in df_final.columns:
                    df_final[_idcol] = df_out[_idcol].to_numpy()
    except Exception as _df_final_exc:  # noqa: BLE001
        log_event(
            f"MASTERSTAR: re-read of {Path(csv_path).name} failed ({_df_final_exc!s}); "
            "using in-memory df_out and re-asserting catalog_id/name as str."
        )
        df_final = df_out.copy()
        # df_out.copy() can carry catalog_id/name as non-string dtypes -> re-assert to avoid
        # reintroducing float/scientific precision loss on Gaia IDs downstream.
        for _idcol in ("catalog_id", "name"):
            if _idcol in df_final.columns:
                df_final[_idcol] = df_final[_idcol].astype(str)
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
        df_final = finalize_masterstar_sky_coords(
            df_final,
            _w_fin,
            gaia_db_path=str(_cfg_ms.gaia_db_path or ""),
            log_fn=log_event,
        )
        _nax1f = int(_hdr_fin.get("NAXIS1") or 0)
        _nax2f = int(_hdr_fin.get("NAXIS2") or 0)
        _rt_fin = evaluate_wcs_roundtrip(_w_fin, naxis1=_nax1f, naxis2=_nax2f)
        _wcs_rt_p99 = _rt_fin.get("wcs_roundtrip_p99_px")
        _wcs_rt_pass = bool(_rt_fin.get("pass"))
        _identity_qa = evaluate_matched_world2pix_identity_px(
            df_final,
            _w_fin,
            gaia_db_path=str(_cfg_ms.gaia_db_path or ""),
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
    # DAO-GAIA-ERA-01 M1: expand detection table to catalog-derived membership before zones/enrich.
    # INV-MS-EXPAND-01: when cone+WCS exist, expand must succeed or raise (no silent skip).
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
        _cone_df_pre = read_vyvar_csv(_cone_gaia_pre, low_memory=False, dtype={"catalog_id": str})
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
        _membership_depth_g = float(
            getattr(_cfg_ms, "masterstar_gaia_census_target_depth_g", None) or 15.0
        )
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
        noise_floor_adu=det_meta.get("noise_floor_adu"),
        equipment_saturate_adu=equipment_saturate_adu,
        saturate_limit_adu_fallback=det_meta.get("saturate_limit_adu"),
        saturate_limit_fraction=float(_cfg_ms.saturate_limit_fraction),
        sigma_px=det_meta.get("bg_sigma_adu"),
        sky_median_adu=det_meta.get("sky_median_adu"),
        prematch_peak_sigma_floor=det_meta.get("prematch_peak_sigma_floor"),
        frame_max_adu=det_meta.get("frame_max_adu"),
        empirical_clip_adu=det_meta.get("empirical_clip_adu"),
        dao_detection_n_equiv=(
            det_meta.get("dao_detection_n_equiv")
            if det_meta.get("dao_detection_n_equiv") is not None
            else float(_cfg_ms.dao_detection_n_equiv)
        ),
    )
    _dao_class_meta: dict[str, Any] = {}
    _recon_ms: dict[str, Any] | None = None
    try:
        cid = df_final.get("catalog_id", pd.Series([""] * len(df_final))).fillna("").astype(str).str.strip()
        df_final["source_type"] = np.where(cid.ne(""), "GAIA_MATCHED", "DAO_ONLY")
        from masterstar_gaia_accounting import (  # noqa: PLC0415
            enrich_masterstar_gaia_complete,
            gaia_on_chip_from_cone,
            write_gaia_census_and_verify,
        )
        from astropy.wcs import WCS as _WCS_enrich  # noqa: PLC0415

        _cone_gaia = Path(platesolve_dir) / "field_catalog_cone.csv"
        if not _cone_gaia.is_file():
            raise RuntimeError("MASTERSTAR Gaia-complete enrich: missing field_catalog_cone.csv")
        if not _has_valid_wcs(hdr):
            raise RuntimeError("MASTERSTAR Gaia-complete enrich: MASTERSTAR WCS missing/invalid")
        _cone_df = read_vyvar_csv(_cone_gaia, low_memory=False, dtype={"catalog_id": str})
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
        with fits.open(masterstar_fits, memmap=False) as _hg:
            _raw_g = np.asarray(_hg[0].data, dtype=np.float32)
        _, _med_g, _ = plain_mean_med_std(_raw_g, sigma=3.0, maxiters=3)
        _data0_g = np.nan_to_num(
            (_raw_g - _med_g).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        _fwhm_g = float(det_meta.get("dao_fwhm_px") or header_core_fwhm_px(hdr) or 3.5)
        _membership_depth_g = float(
            getattr(_cfg_ms, "masterstar_gaia_census_target_depth_g", None) or 15.0
        )
        _census_depth_g = 17.5  # M1-amend: census accounting depth; G 15-17.5 census-only
        df_final, _gaia_census, _gaia_meta = enrich_masterstar_gaia_complete(
            df_final,
            data0=_data0_g,
            gaia_on_chip=_chip,
            cfg=_cfg_ms,
            wpx=_nax1_g,
            h=_nax2_g,
            fwhm_px=_fwhm_g,
            target_depth_g=_census_depth_g,
            sat_limit_adu=det_meta.get("saturate_limit_adu"),
            identity_lock_only=False,
            catalog_derived_membership=bool(_membership_expand_meta),
            tolerance_overrides=det_meta.get("dao_gaia_derived_tol"),
        )
        _census_inv = write_gaia_census_and_verify(
            _gaia_census,
            n_on_chip=len(_chip),
            census_path=Path(platesolve_dir) / "gaia_source_state_census.csv",
        )
        try:
            from dao_gaia_calibration import (  # noqa: PLC0415
                build_calibration_certificate,
                write_calibration_certificate,
            )

            _setup_nm = str(setup_name or platesolve_dir.name or "MASTERSTAR")
            _tol_d = det_meta.get("dao_gaia_derived_tol") or {}
            _dao_x_f = pd.to_numeric(df_final.get("x"), errors="coerce").to_numpy(dtype=np.float64)
            _dao_y_f = pd.to_numeric(df_final.get("y"), errors="coerce").to_numpy(dtype=np.float64)
            _cert = build_calibration_certificate(
                setup=_setup_nm,
                wcs_obj=_wcs_g,
                data0=_data0_g,
                dao_x=_dao_x_f,
                dao_y=_dao_y_f,
                gaia_x=pd.to_numeric(_chip.get("x_gaia"), errors="coerce").to_numpy(dtype=np.float64),
                gaia_y=pd.to_numeric(_chip.get("y_gaia"), errors="coerce").to_numpy(dtype=np.float64),
                gaia_g=pd.to_numeric(_chip.get("g_mag"), errors="coerce").to_numpy(dtype=np.float64),
                fwhm_px=float(_fwhm_g),
                pass1_sigma=float(_tol_d.get("pass1_sigma") or _cfg_ms.masterstar_dao_threshold_sigma),
                pass2_sigma=float(_tol_d.get("pass2_sigma") or _cfg_ms.masterstar_dao_pass2_sigma),
                seed_snr_min=float(_cfg_ms.masterstar_forced_seed_snr_min),
                target_depth_g=float(_census_depth_g),
                edge_margin_px=float(_cfg_ms.masterstar_gaia_census_edge_margin_px),
                cfg=_cfg_ms,
                ms_df=df_final,
                census_df=_gaia_census,
                repo_root=Path(__file__).resolve().parent.parent,
            )
            _cert_path = write_calibration_certificate(
                _cert, Path(platesolve_dir), fail_closed=True
            )
            if _membership_expand_meta:
                from masterstar_gaia_accounting import verify_ms_expand_guard  # noqa: PLC0415

                _ok_exp, _det_exp = verify_ms_expand_guard(
                    _membership_expand_meta,
                    census_path=Path(platesolve_dir) / "gaia_source_state_census.csv",
                    cert_path=_cert_path,
                )
                if not _ok_exp:
                    from invariants_runtime import InvariantViolation  # noqa: PLC0415

                    raise InvariantViolation("INV-MS-EXPAND-01", _det_exp)
                log_event(f"INV-MS-EXPAND-01 PASS: {_det_exp}")
            det_meta["dao_gaia_calibration"] = _cert.to_dict()
            det_meta["dao_gaia_calibration_path"] = str(_cert_path)
            log_event(
                f"DAO-Gaia calibration certificate {_cert.status}: "
                f"match_r={_cert.derived.match_radius_px:.1f}px "
                f"centroid={_cert.derived.pass2_center_tol_px:.1f}px "
                f"empty-sky det={_cert.empty_sky.inv_det} seed={_cert.empty_sky.inv_seed}"
            )
        except Exception as _cal_exc:  # noqa: BLE001
            from invariants_runtime import InvariantViolation  # noqa: PLC0415

            if isinstance(_cal_exc, InvariantViolation):
                raise
            if _membership_expand_meta:
                raise RuntimeError(
                    f"INV-MS-EXPAND-01: certificate write failed: {_cal_exc!s}"
                ) from _cal_exc
            log_event(f"DAO-Gaia calibration certificate skipped: {_cal_exc!s}")
        det_meta["gaia_census_meta"] = _gaia_meta
        det_meta["gaia_census_invariants"] = _census_inv.get("invariants") or []
        log_event(
            f"MASTERSTAR Gaia census: {len(_chip)} on-chip, "
            f"forced_seed={_gaia_meta.get('n_forced_seed', 0)}, "
            f"leftover_promotions={_gaia_meta.get('n_leftover_promotions', 0)}, "
            f"INV-MS-CENSUS-01 {_census_inv.get('detail')}"
        )
        _gdb_fill = str(_cfg_ms.gaia_db_path or "").strip()
        df_final, _n_bp_fill, _n_bp_miss = _fill_masterstars_gaia_matched_bp_rp_from_local_db(
            df_final,
            gaia_db_path=_gdb_fill,
        )
        if _n_bp_miss > 0:
            log_event(f"masterstars bp_rp fallback: {_n_bp_fill}/{_n_bp_miss} doplnenych z Gaia DB")
        _fleming_sigma: float | None = None
        if _gdb_fill:
            try:
                from dao_reconcile import (  # noqa: PLC0415
                    annotate_dao_only_magnitude_classes,
                    compute_gaia_dao_reconcile,
                    fit_fleming_completeness,
                    format_dao_only_census_log,
                    resolve_effective_match_depth,
                )

                _cone_df_cls = None
                _cone_csv_cls = Path(platesolve_dir) / "field_catalog_cone.csv"
                if _cone_csv_cls.is_file():
                    _cone_df_cls = read_vyvar_csv(_cone_csv_cls, low_memory=False, dtype={"catalog_id": str})
                _fwhm_cls = float(det_meta.get("dao_fwhm_px") or 0.0)
                if not (_fwhm_cls > 0.0):
                    _fwhm_cls = float(header_core_fwhm_px(hdr) or 3.5)
                _md_cls = resolve_effective_match_depth(det_meta, is_masterstar=True)
                _cone_lim_cls: float | None = None
                try:
                    _raw_lim = det_meta.get("faintest_mag_limit")
                    if _raw_lim is not None and math.isfinite(float(_raw_lim)):
                        _cone_lim_cls = float(_raw_lim)
                except (TypeError, ValueError):
                    _cone_lim_cls = None
                _noise_cls = det_meta.get("noise_floor_adu")
                _nax1_cls = int(hdr.get("NAXIS1") or 0)
                _nax2_cls = int(hdr.get("NAXIS2") or 0)
                _wcs_cls = None
                _plate_cls = None
                if _has_valid_wcs(hdr) and _nax1_cls > 0 and _nax2_cls > 0:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", FITSFixedWarning)
                        _wcs_cls = WCS(hdr)
                    try:
                        from astropy.wcs.utils import proj_plane_pixel_scales

                        _plate_cls = float(np.mean(proj_plane_pixel_scales(_wcs_cls) * 3600.0))
                    except Exception:  # noqa: BLE001
                        pass
                if _wcs_cls is not None:
                    _recon_ms = compute_gaia_dao_reconcile(
                        df_final,
                        gaia_db_path=_gdb_fill,
                        wcs=_wcs_cls,
                        naxis1=_nax1_cls,
                        naxis2=_nax2_cls,
                        fwhm_px=_fwhm_cls,
                        plate_scale_arcsec=_plate_cls,
                        mag_limit=float(det_meta.get("faintest_mag_limit") or 18.0),
                        match_sep_arcsec=float(
                            det_meta.get("match_sep_arcsec_effective")
                            or det_meta.get("match_sep_arcsec_requested")
                            or 8.0
                        ),
                        cone_df=_cone_df_cls,
                    )
                    _recon_ms.update(_md_cls)
                    _ff = fit_fleming_completeness(_recon_ms.get("completeness_curve") or [])
                    _fleming_sigma = _ff.sigma_mag
                df_final, _dao_class_meta = annotate_dao_only_magnitude_classes(
                    df_final,
                    gaia_db_path=_gdb_fill,
                    effective_match_depth=_md_cls.get("match_depth"),
                    cone_query_mag_limit=_cone_lim_cls,
                    fleming_sigma_mag=_fleming_sigma,
                    frame_noise_adu=_noise_cls,
                )
                if _recon_ms is not None:
                    _recon_ms["dao_only_class_meta"] = _dao_class_meta
                _ms_info_msg = format_dao_only_census_log(_dao_class_meta, n_total=len(df_final))
                LOGGER.info(_ms_info_msg)
                log_event(_ms_info_msg)
            except Exception as _ms_census_exc:  # noqa: BLE001
                LOGGER.debug("[MASTERSTAR-DAO-CENSUS] skipped: %s", _ms_census_exc)
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
                except Exception:  # noqa: BLE001
                    pass
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
    _vyvar_df_to_csv(df_final, csv_path)
    _n_det = int(len(df_final))
    _n_mat = int(
        df_final.get("catalog_id", pd.Series([""] * len(df_final)))
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .sum()
    )
    _rate = (100.0 * float(_n_mat) / float(_n_det)) if _n_det > 0 else 0.0
    _cat_rows_opt = int(det_meta.get("catalog_rows", 0) or 0)
    if "catalog_id" in df_final.columns:
        _cid_opt = df_final["catalog_id"].fillna("").astype(str).str.strip()
        _n_gaia_det_opt = int(_cid_opt[_cid_opt != ""].nunique())
    else:
        _n_gaia_det_opt = int(_n_mat)
    _gaia_rate_opt = (100.0 * float(_n_gaia_det_opt) / float(_cat_rows_opt)) if _cat_rows_opt > 0 else 0.0
    log_event(
        f"[chart] MATCH STATS (optimized): Found {_n_det} stars on image | {_n_mat} matched with Gaia | "
        f"Match Rate: {_rate:.2f}% | Gaia->DAO: {_gaia_rate_opt:.2f}% ({_n_gaia_det_opt}/{_cat_rows_opt})"
    )
    if _cat_rows_opt > 0:
        LOGGER.info(
            "[MASTERSTAR] Gaia->DAO completeness: "
            "%d/%d (%.1f%%) | catalog_only: %d",
            _n_gaia_det_opt,
            _cat_rows_opt,
            _gaia_rate_opt,
            _cat_rows_opt - _n_gaia_det_opt,
        )
    log_event(
        f"MASTERSTAR JSON consistency: n_raw_dao={int(det_meta.get('n_detected_dao_raw', 0) or 0)}, "
        f"detected_stars={_n_det}, catalog_matched={_n_mat}, "
        f"gaia_dao_completeness_pct={round(_gaia_rate_opt, 2) if _cat_rows_opt > 0 else None}, "
        f"n_gaia_undetected={(_cat_rows_opt - _n_gaia_det_opt) if _cat_rows_opt > 0 else None}"
    )
    # TODO-25: persist to pipeline_meta.json so UI can read single source of truth
    if _cat_rows_opt > 0:
        _meta_patch: dict[str, Any] = {
            "gaia_dao_completeness_raw_pct": round(float(_gaia_rate_opt), 2),
            "catalog_rows": int(_cat_rows_opt),
            "n_gaia_detected": int(_n_gaia_det_opt),
            "n_gaia_undetected": int(_cat_rows_opt - _n_gaia_det_opt),
        }
        _idg_stamp = dict(det_meta.get("identity_gate") or {})
        _gmeta = det_meta.get("gaia_census_meta") or {}
        if isinstance(_gmeta, dict) and "n_lock_geometry_reject" in _gmeta:
            _idg_stamp["n_lock_geometry_reject"] = int(_gmeta.get("n_lock_geometry_reject") or 0)
        if _idg_stamp:
            _meta_patch["identity_gate"] = _idg_stamp
        try:
            from dao_gaia_calibration import effective_tol_stamps  # noqa: PLC0415

            _meta_patch["dao_gaia_tol"] = effective_tol_stamps(
                det_meta.get("dao_gaia_derived_tol")
                if isinstance(det_meta.get("dao_gaia_derived_tol"), dict)
                else None,
                _cfg_ms,
                fwhm_px=float(det_meta.get("dao_fwhm_px") or _idg_stamp.get("fwhm_px") or 3.5),
                census_meta=_gmeta if isinstance(_gmeta, dict) else None,
            )
        except Exception:  # noqa: BLE001
            pass
        for _mk in (
            "match_sep_arcsec_requested",
            "match_sep_arcsec_effective",
            "match_sep_formula_inputs",
            "wcs_gaia_pixel_refine_iters",
        ):
            if det_meta.get(_mk) is not None:
                _meta_patch[_mk] = det_meta.get(_mk)
        if _opt_stats_last:
            _meta_patch["optimizer_refit"] = dict(_opt_stats_last)
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
                _cone_df = None
                _cone_csv = Path(platesolve_dir) / "field_catalog_cone.csv"
                if _cone_csv.is_file():
                    _cone_df = read_vyvar_csv(_cone_csv, low_memory=False, dtype={"catalog_id": str})
                _fwhm_recon = float(det_meta.get("dao_fwhm_px") or 0.0)
                if not (_fwhm_recon > 0.0):
                    _fwhm_recon = float(header_core_fwhm_px(hdr) or 3.5)
                _wcs_recon = None
                _plate_recon = None
                _nax1 = int(hdr.get("NAXIS1") or 0)
                _nax2 = int(hdr.get("NAXIS2") or 0)
                if _has_valid_wcs(hdr) and _nax1 > 0 and _nax2 > 0:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", FITSFixedWarning)
                        _wcs_recon = WCS(hdr)
                    try:
                        from astropy.wcs.utils import proj_plane_pixel_scales

                        _plate_recon = float(np.mean(proj_plane_pixel_scales(_wcs_recon) * 3600.0))
                    except Exception:  # noqa: BLE001
                        pass
                _gdb_recon = str(_cfg_ms.gaia_db_path or "").strip()
                _faintest_recon = float(det_meta.get("faintest_mag_limit") or 18.0)
                _match_sep_recon = float(
                    det_meta.get("match_sep_arcsec_effective")
                    or det_meta.get("match_sep_arcsec_requested")
                    or 8.0
                )
                if _wcs_recon is not None and _gdb_recon:
                    _recon = compute_gaia_dao_reconcile(
                        df_final,
                        gaia_db_path=_gdb_recon,
                        wcs=_wcs_recon,
                        naxis1=_nax1,
                        naxis2=_nax2,
                        fwhm_px=_fwhm_recon,
                        plate_scale_arcsec=_plate_recon,
                        mag_limit=_faintest_recon,
                        match_sep_arcsec=_match_sep_recon,
                        cone_df=_cone_df,
                    )
                    _md = resolve_effective_match_depth(det_meta, is_masterstar=True)
                    _recon.update(_md)
                    _meta_patch.update(reconcile_to_pipeline_meta(_recon))
        except Exception as exc:  # noqa: BLE001
            log_event(f"MASTERSTAR Gaia reconcile decomposition skipped: {exc!s}")
            _meta_patch["gaia_dao_completeness_pct"] = round(float(_gaia_rate_opt), 2)
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
        f"MASTERSTAR katalog: {Path(csv_path).name} - {len(df_final)} riadkov "
        f"(DAO + katalog na celom poli; ziadne orezanie podla vzdialenosti od stredu snimku)."
    )
    # Gaussian FWHM (2D fit) -> hlavicka; VY_FWHM je DAO odhad, nie moment FWHM - nepouzivaj 0.619.
    masterstars_df = df_final
    if (
        masterstars_df is None
        or len(masterstars_df) == 0
        or "x" not in masterstars_df.columns
        or "y" not in masterstars_df.columns
    ):
        masterstars_df = df_out
    try:
        from photometry_phase2a import measure_fwhm_from_masterstar

        _ms_path = Path(masterstar_fits)
        if "mag" in masterstars_df.columns:
            _star_pos = masterstars_df[["x", "y", "mag"]].dropna().head(50)
        elif "phot_g_mean_mag" in masterstars_df.columns:
            _star_pos = (
                masterstars_df[["x", "y", "phot_g_mean_mag"]]
                .dropna()
                .rename(columns={"phot_g_mean_mag": "mag"})
                .head(50)
            )
        else:
            _star_pos = masterstars_df[["x", "y"]].dropna().head(50)
        with fits.open(_ms_path, memmap=False) as _hint_hdul:
            _vy_hint = _hint_hdul[0].header.get("VY_FWHM", 3.5)
            _vy_fwhm_hint = float(_vy_hint) if _vy_hint is not None else 3.5
        _gaussian_fwhm = measure_fwhm_from_masterstar(
            _ms_path,
            _star_pos,
            dao_fwhm_hint=_vy_fwhm_hint,
            n_stars=30,
        )
        with fits.open(_ms_path, mode="update", memmap=False) as _hdul:
            _hdul[0].header["VY_FWHM_GAUSS"] = (
                round(float(_gaussian_fwhm), 4),
                "Gaussian FWHM px (2D fit)",
            )
            _n_raw_dao_hdr = int(det_meta.get("n_detected_dao_raw", 0) or 0)
            _hdul[0].header["VY_NDAO"] = (
                _n_raw_dao_hdr,
                "VYVAR: raw DAO detections on MASTERSTAR (stars/Mpx density)",
            )
            _hdul.flush()
        logging.info(
            f"[MASTERSTAR] VY_FWHM_GAUSS={float(_gaussian_fwhm):.3f}px ulozene do hlavicky (2D fit)"
        )
    except Exception as e:  # noqa: BLE001
        logging.error('[EXC-0409] Cross-setup `comparison_stars.csv` sync failure leaves B/V/R setups with inconsistent c...: %s', e)
        log_event(f"[ERROR] VY_FWHM_GAUSS fit ZLYHAL: {e}\n{traceback.format_exc()}")
    try:
        _ms_path_tag = Path(masterstar_fits)
        _n_raw_dao_hdr = int(det_meta.get("n_detected_dao_raw", 0) or 0)
        with fits.open(_ms_path_tag, mode="update", memmap=False) as _hdul_tag:
            if "VY_NDAO" not in _hdul_tag[0].header:
                _hdul_tag[0].header["VY_NDAO"] = (
                    _n_raw_dao_hdr,
                    "VYVAR: raw DAO detections on MASTERSTAR (stars/Mpx density)",
                )
                _hdul_tag.flush()
    except Exception:  # noqa: BLE001
        pass
    # Small flush pause: UI may read CSV immediately after this returns.
    time.sleep(0.5)
    # Drop stale pre-optimizer dataframe to avoid accidental reuse ("ghost rows").
    try:
        del df_out
    except Exception:  # noqa: BLE001
        pass
    # Keep platesolve clean: remove temporary/duplicate artifacts.
    for _dup in (
        platesolve_dir / "MASTERSTAR_full.fits",
        platesolve_dir / "MASTERSTAR_full.jpg",
        temp_csv,
    ):
        try:
            if Path(_dup).is_file() and Path(_dup).resolve() != Path(csv_path).resolve():
                Path(_dup).unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass
    photo_plan = write_photometry_plan_files(
        platesolve_dir=platesolve_dir,
        masterstar_fits=masterstar_fits,
        masterstars_csv=csv_path,
        n_comparison_stars=int(n_comparison_stars),
        require_non_variable=bool(require_non_variable_comparisons),
    )
    # F-431 / C1: stamp AFTER VT exists (write_photometry_plan_files).
    try:
        _vt_stamp_path = platesolve_dir / "variable_targets.csv"
        if _vt_stamp_path.is_file():
            from photometry_core import stamp_vsx_known_variable_on_masterstars  # noqa: PLC0415

            _ms_for_stamp = pd.read_csv(
                csv_path, low_memory=False, dtype={"catalog_id": str, "name": str}
            )
            _vt_stamp_df = pd.read_csv(_vt_stamp_path, low_memory=False, dtype={"catalog_id": str})
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
    # Multi-filter support: keep comparison stars consistent across platesolve/<setup>/ folders.
    try:
        _sync_comparison_stars_across_setups(Path(platesolve_dir).parent)
    except Exception as _sync_exc:  # noqa: BLE001
        log_event(
            f"MASTERSTAR: comparison-star cross-setup sync failed ({_sync_exc!s}); "
            "B/V/R comp sets may be inconsistent across setups."
        )

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
        "faintest_mag_limit": det_meta.get("faintest_mag_limit"),
        "n_dropped_fainter_than_limit": det_meta.get("n_dropped_fainter_than_limit"),
        "field_catalog_cone_csv": det_meta.get("field_catalog_cone_csv"),
        "catalog_derived_membership": det_meta.get("catalog_derived_membership"),
        "dao_threshold_sigma": det_meta.get("dao_threshold_sigma"),
        "masterstar_match_png": "",
    }
    out.update(photo_plan)
    # Enrichment columns for masterstars_full_match.csv (formerly MASTER_SOURCES DB).
    try:
        gaia_db = str(_cfg_ms.gaia_db_path or "").strip()
        if gaia_db and draft_id is not None and "ra_deg" in df_final.columns and "dec_deg" in df_final.columns:
            det = df_final.copy()
            det["ra_deg"] = pd.to_numeric(det["ra_deg"], errors="coerce")
            det["dec_deg"] = pd.to_numeric(det["dec_deg"], errors="coerce")
            det = det[det["ra_deg"].notna() & det["dec_deg"].notna()].copy()
            if not det.empty:
                ra_min = float(det["ra_deg"].min()) - 0.01
                ra_max = float(det["ra_deg"].max()) + 0.01
                de_min = float(det["dec_deg"].min()) - 0.01
                de_max = float(det["dec_deg"].max()) + 0.01
                ga = query_local_gaia(
                    gaia_db,
                    ra_min=ra_min,
                    ra_max=ra_max,
                    dec_min=de_min,
                    dec_max=de_max,
                    mag_limit=None,
                )
                if ga:
                    gdf = pd.DataFrame(ga)
                    gcoo = SkyCoord(
                        ra=pd.to_numeric(gdf["ra"], errors="coerce").astype(float).values * u.deg,
                        dec=pd.to_numeric(gdf["dec"], errors="coerce").astype(float).values * u.deg,
                        frame="icrs",
                    )
                    dcoo = SkyCoord(
                        ra=det["ra_deg"].astype(float).values * u.deg,
                        dec=det["dec_deg"].astype(float).values * u.deg,
                        frame="icrs",
                    )
                    idx, sep2d, _ = dcoo.match_to_catalog_sky(gcoo)
                    ok = sep2d.to(u.arcsec).value <= 2.0
                    if bool(np.any(ok)):
                        # Geometry + blending pruning and dynamic photometric binning.
                        nax1 = int(hdr.get("NAXIS1", 0) or 0) or int(data.shape[1])
                        nax2 = int(hdr.get("NAXIS2", 0) or 0) or int(data.shape[0])
                        border_px = 50.0

                        try:
                            from astropy.coordinates import search_around_sky

                            pairs_i, pairs_j, _, _ = search_around_sky(gcoo, gcoo, 5.0 * u.arcsec)
                            gmag_all = (
                                pd.to_numeric(gdf.get("g_mag"), errors="coerce")
                                .astype(float)
                                .to_numpy()
                            )
                            blended_idx: set[int] = set()
                            for a, b in zip(pairs_i, pairs_j, strict=False):
                                ia = int(a)
                                ib = int(b)
                                if ia == ib:
                                    continue
                                ma = gmag_all[ia] if ia < len(gmag_all) else float("nan")
                                mb = gmag_all[ib] if ib < len(gmag_all) else float("nan")
                                if not (math.isfinite(ma) and math.isfinite(mb)):
                                    continue
                                if abs(ma - mb) < 3.0:
                                    blended_idx.add(ia)
                                    blended_idx.add(ib)
                        except Exception:  # noqa: BLE001
                            blended_idx = set()

                        filt = str(det_meta.get("filter") or hdr.get("FILTER") or "Clear").strip() or "Clear"
                        if filt.lower() in {"nofilter", "none", "null"}:
                            filt = "Clear"

                        def _bin_step(v: float, step: float) -> float:
                            if not math.isfinite(v):
                                return float("nan")
                            return math.floor((float(v) / float(step)) + 0.5) * float(step)

                        # Saturation threshold for MASTERSTAR (FITS + EQUIPMENTS; no global config fallback)
                        sat_limit = det_meta.get("saturate_limit_adu")
                        if sat_limit is None:
                            _eq_sat_ms = equipment_saturate_adu
                            if _eq_sat_ms is None and equipment_id is not None:
                                _eq_sat_ms = _equipment_saturate_adu_from_db(equipment_id)
                            sat_limit, _ = _effective_saturation_limit(
                                hdr,
                                fallback_adu=None,
                                equipment_saturate_adu=_eq_sat_ms,
                            )
                        if (
                            sat_limit is not None
                            and math.isfinite(float(sat_limit))
                            and float(sat_limit) > 0
                        ):
                            sat_thr = float(sat_limit) * float(saturate_level_fraction)
                        else:
                            # INV-SAT-LIMIT: never admit against +inf.
                            sat_thr = float(SAT_LIMIT_CONTAINER_CLIP_ADU) * float(SAT_LIMIT_NO_KNEE_FRAC)
                            logging.warning(
                                "[INV-SAT-LIMIT] MASTERSTAR sat_thr unresolved; "
                                "using peak-test %.1f ADU (0.80 x container clip)",
                                sat_thr,
                            )

                        rows_ms: list[dict[str, Any]] = []
                        det_ok = det.iloc[np.where(ok)[0]].reset_index(drop=True)
                        g_ok = gdf.iloc[idx[np.where(ok)[0]]].reset_index(drop=True)
                        g_ok_idx = idx[np.where(ok)[0]]
                        # Aperture optimization: estimate per-star FWHM on MASTERSTAR, then take medians per color.
                        try:
                            import numpy as _np

                            arr_ms = _np.asarray(data, dtype=_np.float32)

                            fwhm_est = [
                                _fwhm_moment_at(
                                    arr_ms,
                                    float(det_ok["x"].iloc[i]) if "x" in det_ok.columns and pd.notna(det_ok["x"].iloc[i]) else float("nan"),
                                    float(det_ok["y"].iloc[i]) if "y" in det_ok.columns and pd.notna(det_ok["y"].iloc[i]) else float("nan"),
                                    half=6,
                                )
                                for i in range(len(det_ok))
                            ]
                            fwhm_med_px = float(_np.nanmedian(_np.asarray(fwhm_est, dtype=_np.float64)))
                        except Exception:  # noqa: BLE001
                            fwhm_est = [float("nan")] * len(det_ok)
                            fwhm_med_px = float("nan")

                        if not (math.isfinite(fwhm_med_px) and fwhm_med_px > 0):
                            try:
                                fwhm_med_px = float(det_meta.get("dao_fwhm_px") or _ms_fwhm)
                            except Exception:  # noqa: BLE001
                                fwhm_med_px = float(_ms_fwhm)
                        if not (math.isfinite(fwhm_med_px) and fwhm_med_px > 0):
                            fwhm_med_px = float(_ms_fwhm)

                        # Median per coarse color category.
                        def _color_bucket(bp_rp: float) -> str:
                            if not math.isfinite(bp_rp):
                                return "neutral"
                            if bp_rp < 0.5:
                                return "blue"
                            if bp_rp <= 1.5:
                                return "neutral"
                            return "red"

                        by_col: dict[str, list[float]] = {"blue": [], "neutral": [], "red": []}
                        for i in range(len(det_ok)):
                            bprp_v0 = (
                                float(g_ok["bp_rp"].iloc[i])
                                if "bp_rp" in g_ok.columns and pd.notna(g_ok["bp_rp"].iloc[i])
                                else float("nan")
                            )
                            fe = float(fwhm_est[i]) if i < len(fwhm_est) else float("nan")
                            if math.isfinite(fe) and fe > 0:
                                by_col[_color_bucket(bprp_v0)].append(fe)
                        fwhm_blue = float(_np.median(by_col["blue"])) if by_col["blue"] else fwhm_med_px
                        fwhm_neu = float(_np.median(by_col["neutral"])) if by_col["neutral"] else fwhm_med_px
                        fwhm_red = float(_np.median(by_col["red"])) if by_col["red"] else fwhm_med_px

                        # Gaia neighbour veto radius in arcsec: 3x median FWHM (px) x plate scale.
                        try:
                            from astropy.wcs.utils import proj_plane_pixel_scales

                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", FITSFixedWarning)
                                _w_ms = WCS(hdr)
                            sc = proj_plane_pixel_scales(_w_ms.celestial)  # deg/pix
                            scale_arcsec_per_px = float(_np.nanmedian(_np.asarray(sc, dtype=_np.float64))) * 3600.0
                        except Exception:  # noqa: BLE001
                            scale_arcsec_per_px = float("nan")
                        veto_radius_arcsec = (
                            float(3.0 * fwhm_med_px * scale_arcsec_per_px)
                            if math.isfinite(scale_arcsec_per_px) and scale_arcsec_per_px > 0
                            else float("nan")
                        )
                        import numpy as _np
                        # photometry.py is legacy; use the merged core module.
                        from photometry_core import bad_columns_for_light_frame

                        _thr_nl = float("nan")
                        _peaks_nl: list[float] = []
                        for _i in range(len(det_ok)):
                            _pm = (
                                det_ok["peak_max_adu"].iloc[_i]
                                if "peak_max_adu" in det_ok.columns and pd.notna(det_ok["peak_max_adu"].iloc[_i])
                                else None
                            )
                            if _pm is not None and math.isfinite(float(_pm)):
                                _peaks_nl.append(float(_pm))
                        if _peaks_nl:
                            _pp = float(_cfg_ms.nonlinearity_peak_percentile)
                            _thr_nl = float(
                                _np.percentile(
                                    _np.asarray(_peaks_nl, dtype=_np.float64),
                                    min(100.0, max(0.0, 100.0 - _pp)),
                                )
                            )
                        _nl_ratio = float(_cfg_ms.nonlinearity_fwhm_ratio)
                        _bpm_js = None
                        if master_dark_path and str(master_dark_path).strip():
                            _mdp = Path(str(master_dark_path))
                            _bj = _mdp.parent / f"{_mdp.stem}_dark_bpm.json"
                            if _bj.is_file():
                                try:
                                    _bpm_js = json.loads(_bj.read_text(encoding="utf-8"))
                                except Exception:  # noqa: BLE001
                                    _bpm_js = None
                        _bad_x = bad_columns_for_light_frame(_bpm_js, light_header=hdr)
                        for i in range(len(det_ok)):
                            x = float(det_ok["x"].iloc[i]) if "x" in det_ok.columns else None
                            y = float(det_ok["y"].iloc[i]) if "y" in det_ok.columns else None
                            pmax = float(det_ok["peak_max_adu"].iloc[i]) if "peak_max_adu" in det_ok.columns and pd.notna(det_ok["peak_max_adu"].iloc[i]) else None
                            is_sat = 1 if (pmax is not None and math.isfinite(pmax) and pmax > sat_thr) else 0
                            var_flag = str(g_ok.get("var_flag").iloc[i]) if "var_flag" in g_ok.columns else ""
                            is_var = 1 if var_flag.strip() not in ("", "0", "False", "false", "NO", "No") else 0

                            is_border = (
                                x is not None
                                and y is not None
                                and (
                                    x < border_px
                                    or y < border_px
                                    or x > float(nax1) - border_px
                                    or y > float(nax2) - border_px
                                )
                            )
                            gi = int(g_ok_idx[i]) if i < len(g_ok_idx) else -1
                            is_blend = bool(gi in blended_idx) if gi >= 0 else False
                            excl = "Border" if is_border else ("Blended" if is_blend else None)

                            fe_i = float(fwhm_est[i]) if i < len(fwhm_est) else float("nan")
                            likely_nl = False
                            if (
                                math.isfinite(fe_i)
                                and math.isfinite(fwhm_med_px)
                                and fwhm_med_px > 0
                                and pmax is not None
                                and math.isfinite(float(pmax))
                                and math.isfinite(_thr_nl)
                                and float(pmax) >= _thr_nl
                                and fe_i > _nl_ratio * fwhm_med_px
                            ):
                                likely_nl = True
                            on_bad = False
                            if x is not None and _bad_x:
                                if int(round(float(x))) in _bad_x:
                                    on_bad = True

                            # New Gaia stability/multiplicity filters.
                            gfer = None
                            if "g_flux_error_rel" in g_ok.columns and pd.notna(g_ok["g_flux_error_rel"].iloc[i]):
                                try:
                                    gfer = float(g_ok["g_flux_error_rel"].iloc[i])
                                except (TypeError, ValueError):
                                    gfer = None
                            nss = 0
                            if "non_single_star" in g_ok.columns and pd.notna(g_ok["non_single_star"].iloc[i]):
                                try:
                                    nss = int(float(g_ok["non_single_star"].iloc[i]))
                                except (TypeError, ValueError):
                                    nss = 0
                            pvf = ""
                            if "phot_variable_flag" in g_ok.columns and pd.notna(g_ok["phot_variable_flag"].iloc[i]):
                                pvf = str(g_ok["phot_variable_flag"].iloc[i]).strip()

                            if excl is None:
                                if gfer is not None and math.isfinite(gfer) and float(gfer) > 0.02:
                                    excl = "CatalogNoise"
                                elif int(nss) > 0:
                                    excl = "NonSingle"
                                elif pvf.upper() == "VARIABLE":
                                    excl = "Variable"
                                else:
                                    # Neighbour veto: exclude if Gaia neighbour would change mag by > 0.001.
                                    if (
                                        veto_radius_arcsec is not None
                                        and math.isfinite(float(veto_radius_arcsec))
                                        and float(veto_radius_arcsec) > 0
                                        and "g_mag" in g_ok.columns
                                        and pd.notna(g_ok["g_mag"].iloc[i])
                                    ):
                                        try:
                                            m0 = float(g_ok["g_mag"].iloc[i])
                                        except (TypeError, ValueError):
                                            m0 = float("nan")
                                        if math.isfinite(m0):
                                            try:
                                                from astropy.coordinates import search_around_sky as _sas

                                                # Query neighbours in the Gaia window itself (fast; same gcoo).
                                                # Use the matched Gaia index gi.
                                                gi2 = int(g_ok_idx[i]) if i < len(g_ok_idx) else -1
                                                if gi2 >= 0 and gi2 < len(gcoo):
                                                    c0 = gcoo[gi2]
                                                    _, jj, _, _ = _sas(
                                                        c0,
                                                        gcoo,
                                                        float(veto_radius_arcsec) * u.arcsec,
                                                    )
                                                    ratios: list[float] = []
                                                    for jx in list(jj):
                                                        j = int(jx)
                                                        if j == gi2:
                                                            continue
                                                        try:
                                                            mj = float(gdf["g_mag"].iloc[j])
                                                        except Exception:  # noqa: BLE001
                                                            continue
                                                        if not math.isfinite(mj):
                                                            continue
                                                        ratios.append(10.0 ** (-0.4 * (mj - m0)))
                                                    if ratios:
                                                        dm = -2.5 * math.log10(1.0 + float(sum(ratios)))
                                                        if abs(dm) > 0.001:
                                                            excl = "Gaia neighbor blend"
                                            except Exception:  # noqa: BLE001
                                                pass
                            if likely_nl and excl is None:
                                excl = "Nonlinear FWHM"
                            if on_bad and excl is None:
                                excl = "Bad column"
                            safe = 0 if excl is not None else 1

                            gmag_v = (
                                float(g_ok["g_mag"].iloc[i])
                                if "g_mag" in g_ok.columns and pd.notna(g_ok["g_mag"].iloc[i])
                                else float("nan")
                            )
                            bprp_v = (
                                float(g_ok["bp_rp"].iloc[i])
                                if "bp_rp" in g_ok.columns and pd.notna(g_ok["bp_rp"].iloc[i])
                                else float("nan")
                            )
                            mb = _bin_step(gmag_v, 0.5)
                            cb = _bin_step(bprp_v, 0.25)
                            phot_cat = (
                                f"{filt}_mag_{mb:.1f}_col_{cb:.2f}"
                                if math.isfinite(mb) and math.isfinite(cb)
                                else f"{filt}_mag_nan_col_nan"
                            )
                            rows_ms.append(
                                {
                                    "x_master": x,
                                    "y_master": y,
                                    "ra": float(g_ok["ra"].iloc[i]) if pd.notna(g_ok["ra"].iloc[i]) else float(det_ok["ra_deg"].iloc[i]),
                                    "dec": float(g_ok["dec"].iloc[i]) if pd.notna(g_ok["dec"].iloc[i]) else float(det_ok["dec_deg"].iloc[i]),
                                    "g_mag": float(g_ok["g_mag"].iloc[i]) if "g_mag" in g_ok.columns and pd.notna(g_ok["g_mag"].iloc[i]) else None,
                                    "bp_rp": float(g_ok["bp_rp"].iloc[i]) if "bp_rp" in g_ok.columns and pd.notna(g_ok["bp_rp"].iloc[i]) else None,
                                    "is_var": is_var,
                                    "is_saturated": is_sat,
                                    "source_id_gaia": str(g_ok["source_id"].iloc[i]) if "source_id" in g_ok.columns else "",
                                    "g_flux_error_rel": gfer,
                                    "non_single_star": int(nss),
                                    "phot_variable_flag": pvf,
                                    "filter_name": filt,
                                    "phot_category": phot_cat,
                                    "recommended_aperture": recommended_aperture_by_color(
                                        bp_rp=bprp_v if math.isfinite(bprp_v) else None,
                                        median_fwhm_blue=fwhm_blue,
                                        median_fwhm_neutral=fwhm_neu,
                                        median_fwhm_red=fwhm_red,
                                    ),
                                    "is_safe_comp": safe,
                                    "exclusion_reason": excl,
                                    "safe_override": 0,
                                    "likely_nonlinear": 1 if likely_nl else 0,
                                    "on_bad_column": 1 if on_bad else 0,
                                }
                            )
                        try:
                            from masterstars_enrichment import (  # noqa: PLC0415
                                apply_common_field_bbox_exclusion,
                                apply_stress_rms_to_rows_ms,
                                apply_vsx_variable_flags,
                                merge_enrichment_into_masterstars_df,
                            )

                            df_final = merge_enrichment_into_masterstars_df(df_final, rows_ms)
                            _vyvar_df_to_csv(df_final, csv_path)
                            out["masterstars_enrichment_written"] = int(len(rows_ms))
                            try:
                                _wp2 = write_photometry_plan_files(
                                    platesolve_dir=platesolve_dir,
                                    masterstar_fits=masterstar_fits,
                                    masterstars_csv=csv_path,
                                    n_comparison_stars=int(n_comparison_stars),
                                    require_non_variable=bool(require_non_variable_comparisons),
                                    draft_id=int(draft_id),
                                )
                                out.update(_wp2)
                            except Exception as _wp2_exc:  # noqa: BLE001
                                log_event(
                                    f"MASTERSTAR: enriched photometry-plan rewrite failed ({_wp2_exc!s}); "
                                    "keeping the prior photometry plan."
                                )
                        except Exception as exc:  # noqa: BLE001
                            out["masterstars_enrichment_error"] = str(exc)

                        # Stress-test: 10% random sample, exclude Border/Blended by default (soft-crop).
                        try:
                            from masterstars_enrichment import merge_enrichment_into_masterstars_df  # noqa: PLC0415

                            root_frames = (
                                Path(source_root)
                                if source_root is not None
                                else (Path(detrended_root) if detrended_root is not None else ap)
                            )
                            # Common field intersection bbox across MASTERSTAR input frames (finite data overlap).
                            try:
                                _ms_inputs: list[Path] = []
                                if only_ms_paths is not None:
                                    _ms_inputs = [Path(p) for p in only_ms_paths if Path(p).is_file()]
                                else:
                                    # Fallback: approximate using a subset of aligned frames.
                                    _ms_inputs = sorted(_iter_fits_recursive(root_frames))[
                                        : max(2, int(info.get("frames_used", 0)))
                                    ]
                                bbox = common_field_intersection_bbox_px(frame_paths=_ms_inputs, finite_stride=16)
                                if bbox is not None:
                                    x0b, y0b, x1b, y1b = bbox
                                    apply_common_field_bbox_exclusion(
                                        rows_ms,
                                        x0=float(x0b),
                                        x1=float(x1b),
                                        y0=float(y0b),
                                        y1=float(y1b),
                                    )
                                    df_final = merge_enrichment_into_masterstars_df(df_final, rows_ms)
                                    _vyvar_df_to_csv(df_final, csv_path)
                                    out["common_field_bbox_px"] = [float(x0b), float(y0b), float(x1b), float(y1b)]
                            except Exception as exc:  # noqa: BLE001
                                out["common_field_error"] = str(exc)

                            safe_ids = [
                                str(r.get("source_id_gaia") or "").strip()
                                for r in rows_ms
                                if int(r.get("is_safe_comp") or 0) == 1
                            ]
                            st_res = stress_test_relative_rms_from_sidecars(
                                frames_root=root_frames,
                                source_ids=safe_ids,
                                sample_frac=0.10,
                                seed=42,
                            )
                            out["stress_frames_sampled"] = int(st_res.frames_sampled)
                            out["stress_frames_used"] = int(st_res.frames_used)

                            by_bin: dict[str, list[float]] = {}
                            for rr in rows_ms:
                                if int(rr.get("is_safe_comp") or 0) != 1:
                                    continue
                                sid = str(rr.get("source_id_gaia") or "").strip()
                                if not sid or sid not in st_res.per_source_rms:
                                    continue
                                b = str(rr.get("phot_category") or "").strip()
                                if b:
                                    by_bin.setdefault(b, []).append(float(st_res.per_source_rms[sid]))
                            med_by_bin = {b: float(pd.Series(v).median()) for b, v in by_bin.items() if v}
                            apply_stress_rms_to_rows_ms(rows_ms, st_res.per_source_rms, med_by_bin)

                            packed = [
                                {
                                    "source_id_gaia": rr.get("source_id_gaia"),
                                    "phot_category": rr.get("phot_category"),
                                    "stress_rms": rr.get("stress_rms"),
                                    "ra": rr.get("ra"),
                                    "dec": rr.get("dec"),
                                }
                                for rr in rows_ms
                                if rr.get("stress_rms") is not None
                            ]
                            var_ids = vsx_is_known_variable_top3_per_bin(rows=packed)
                            if var_ids:
                                apply_vsx_variable_flags(rows_ms, set(var_ids))
                                out["vsx_flagged_variables"] = int(len(var_ids))

                            df_final = merge_enrichment_into_masterstars_df(df_final, rows_ms)
                            _vyvar_df_to_csv(df_final, csv_path)
                        except Exception as exc:  # noqa: BLE001
                            out["stress_test_error"] = str(exc)
    except Exception as exc:  # noqa: BLE001
        out["masterstars_enrichment_error"] = str(exc)
    # Persist MASTERSTAR path on draft for later UI reloads / Step 3 continuity.
    try:
        if draft_id is not None:
            _db_ms = VyvarDatabase(Path(_cfg_ms.database_path))
            try:
                _db_ms.set_obs_draft_masterstar_fits_path(int(draft_id), str(Path(masterstar_fits).resolve()))
            finally:
                _db_ms.conn.close()
    except Exception as exc:  # noqa: BLE001
        out["masterstar_path_store_error"] = str(exc)
    return out
