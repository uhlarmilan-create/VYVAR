"""Headless VYVAR night pipeline runner.

Extracted from ``app.py`` ``_run_vyvar_full_pipeline``.
Called by:
  - ``simulate_night_run.py`` (CLI / e2e test)
  - ``app.py`` ``_run_vyvar_full_pipeline`` (UI wrapper - deferred)
  - Future: TODO-11 auto-trigger watchdog

No Streamlit dependencies. Progress via ``logging`` and optional callback.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from config import AppConfig
from importer import smart_import_session, smart_scan_source
from pipeline import AstroPipeline

LOGGER = logging.getLogger(__name__)

_DEFAULT_MASTERSTAR_SELECTION_PCT = 10.0


@dataclass
class NightRunParams:
    """Input parameters for a headless night pipeline run."""

    source_dir: Path
    equipment_id: int
    telescope_id: int
    config_path: Path | None = None
    location_id: int | None = None
    platesolve_equipment_id: int | None = None
    sysrem_enabled: bool | None = None
    sysrem_n_iter: int | None = None
    progress_cb: Callable[[str], None] | None = None
    dry_run: bool = False
    manual_flat_map: dict[str, str] | None = None
    roundness_reject_above: float = 1.25
    # Preprocess / platesolve (UI defaults from render_live_view)
    plate_fov_deg: float | None = None
    dao_fwhm_px: float | None = None
    dao_threshold_sigma: float | None = None
    # UI RUN VYVAR hardcodes cat_match_arc=2.0; pipeline floors with max(10, sep) for
    # MASTERSTAR initial match - so NightRun default follows UI intent (effective 10").
    catalog_match_max_sep_arcsec: float = 2.0
    max_catalog_rows: int = 12000
    max_extra_platesolve: int = 0
    min_detected_stars: int = 100
    max_detected_stars: int = 500
    max_control_points: int = 80
    saturate_level_fraction: float = 0.95
    post_platesolve_hook: Callable[[int, Path, AppConfig, AstroPipeline], None] | None = None
    pre_calibrated_mode: bool = False


@dataclass
class NightRunResult:
    """Result of a headless night pipeline run."""

    success: bool
    draft_id: int | None = None
    draft_dir: Path | None = None
    output_dir: Path | None = None
    n_lightcurves: int = 0
    n_frames: int = 0
    lc_rms_median: float = float("nan")
    sysrem_improvement_pct: float = float("nan")
    phase_timings: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    photometry_completeness: dict[str, dict[str, Any]] = field(default_factory=dict)


# Phase 2A must cover at least this fraction of active_targets in photometry_summary.
_PHOTOMETRY_COMPLETENESS_MIN_RATIO = 0.90


def _load_app_config(config_path: Path | None) -> AppConfig:
    """Load ``AppConfig``; optional explicit ``config.json`` path."""
    cfg = AppConfig()
    if config_path is None:
        return cfg
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    cfg.project_root = path.parent
    cfg.__post_init__()
    return cfg


def _compute_masterstar_score(df: pd.DataFrame) -> pd.Series:
    """MASTERSTAR frame score (higher = better). Mirrors ``ui_quality_dashboard``."""
    score = pd.Series(0.0, index=df.index)
    if len(df) == 0:
        return score

    def _norm_inverse(s: pd.Series) -> pd.Series:
        mn, mx = s.min(), s.max()
        if not math.isfinite(float(mn)) or not math.isfinite(float(mx)) or mx <= mn:
            return pd.Series(1.0, index=s.index)
        return 1.0 - (s - mn) / (mx - mn)

    def _norm_direct(s: pd.Series) -> pd.Series:
        mn, mx = s.min(), s.max()
        if not math.isfinite(float(mn)) or not math.isfinite(float(mx)) or mx <= mn:
            return pd.Series(1.0, index=s.index)
        return (s - mn) / (mx - mn)

    fwhm = pd.to_numeric(df["FWHM"], errors="coerce")
    elong = pd.to_numeric(
        df.get("ELONGATION_MEAN", pd.Series(np.nan, index=df.index)), errors="coerce"
    )
    stars = pd.to_numeric(df["STAR_COUNT"], errors="coerce")
    sky = pd.to_numeric(df["SKY_LEVEL"], errors="coerce")

    if fwhm.notna().sum() >= 2:
        score += 0.45 * _norm_inverse(fwhm.fillna(fwhm.max()))
    if elong.notna().sum() >= 2:
        score += 0.30 * _norm_inverse(elong.fillna(elong.max()))
    if stars.notna().sum() >= 2:
        score += 0.15 * _norm_direct(stars.fillna(stars.min()))
    if sky.notna().sum() >= 2:
        score += 0.10 * _norm_inverse(sky.fillna(sky.max()))
    return score


def _masterstar_candidate_path_for_job(
    archive: Path,
    path_any: str,
    *,
    draft_id: int | None = None,
    db: Any = None,
) -> str:
    """Prefer draft lights FITS for MASTERSTAR job input (processed or pre-cal non_calibrated)."""
    from pipeline import resolve_obs_file_to_processed_fits

    def _has_raw_seg(pp: Path) -> bool:
        try:
            parts = pp.resolve().parts
        except OSError:
            parts = pp.parts
        return any(seg.casefold() == "raw" for seg in parts)

    p = (path_any or "").strip()
    if not p:
        return ""
    if not archive.is_dir():
        return p
    try:
        from draft_provenance import is_pre_calibrated_draft

        pre_cal = is_pre_calibrated_draft(archive, draft_id=draft_id, db=db)
    except Exception:  # noqa: BLE001
        pre_cal = False
    for key in (p, Path(p).name):
        try:
            hit = resolve_obs_file_to_processed_fits(
                archive,
                str(key),
                draft_id=draft_id,
                db=db,
            )
        except Exception:  # noqa: BLE001
            hit = None
        if hit is not None and hit.is_file():
            if pre_cal or not _has_raw_seg(hit):
                return str(hit.resolve())
    if not pre_cal and _has_raw_seg(Path(p)):
        LOGGER.warning(
            "MASTERSTAR: could not map RAW/non_calibrated path to processed - %s",
            Path(p).name,
        )
        return ""
    if Path(p).is_file():
        return str(Path(p).resolve())
    return p


def _make_progress_cb(
    user_cb: Callable[[str], None] | None,
) -> Callable[[int, int, str], None]:
    def _inner(i: int, total: int, msg: str) -> None:
        line = f"[{i}/{max(total, 1)}] {msg}"
        LOGGER.info("[NightRun] %s", line)
        if user_cb is not None:
            user_cb(line)

    return _inner


def _night_run_preprocess(
    *,
    pending: dict[str, Any],
    ap: Path,
    pipeline: AstroPipeline,
    progress_cb: Callable[[int, int, str], None],
) -> None:
    """Headless equivalent of ``app._vyvar_execute_preprocess_pending``."""
    from draft_provenance import resolve_draft_lights_root
    from pipeline import (
        build_prefilter_rejected_map,
        calibrated_paths_for_draft_apply_filters,
        preprocess_calibrated_to_processed,
        qc_enrich_calibrated_lights_in_place,
        _iter_light_fits,
    )

    _app_cfg = pipeline.config
    ap_root = ap.parent if ap.name.casefold() == "non_calibrated" else ap
    proc_root = ap_root / "processed" / "lights"
    _dqf = pending.get("quality_filter_draft_id")
    _fwhm_lim = float(pending.get("fwhm_limit_px") or 0.0)
    source_dir = (
        resolve_draft_lights_root(
            ap_root,
            draft_id=int(_dqf) if _dqf is not None else None,
            db=pipeline.db,
        )
        if _dqf is not None
        else ap_root / "calibrated" / "lights"
    )

    _pp_kw = dict(
        reject_fwhm_px=(float(_fwhm_lim) if float(_fwhm_lim) > 0.0 else None),
        reject_elongation=None,
        use_gpu_if_available=False,
        inject_pointing_ra_deg=pending.get("inject_pointing_ra_deg"),
        inject_pointing_dec_deg=pending.get("inject_pointing_dec_deg"),
        inject_pointing_only_if_missing=False,
        app_config=_app_cfg,
    )

    if _dqf is not None:
        p1, _unused = calibrated_paths_for_draft_apply_filters(
            ap_root,
            pipeline.db,
            int(_dqf),
            fwhm_max_px=_fwhm_lim,
            source_dir=source_dir,
        )
        if not p1:
            why = ["IS_REJECTED=0"]
            if _fwhm_lim > 0:
                why.append("FWHM <= limit or FWHM is NULL")
            raise FileNotFoundError("QC filter: no frames matching " + ", ".join(why) + ".")
        dfs_pp: list[pd.DataFrame] = []
        _all_lights = _iter_light_fits(source_dir)
        _prefilter_map = build_prefilter_rejected_map(_all_lights, p1)
        tot_pp = len(_all_lights)
        off_pp = 0

        def _pcb_pp(off0: int):
            def _inner(i: int, _t: int, msg: str) -> None:
                progress_cb(off0 + i, max(tot_pp, 1), msg)

            return _inner

        if not source_dir.exists():
            raise FileNotFoundError("Missing source lights directory for preprocess filter.")
        dfs_pp.append(
            qc_enrich_calibrated_lights_in_place(
                calibrated_root=source_dir,
                only_paths=None,
                prefilter_rejected=_prefilter_map,
                progress_cb=_pcb_pp(off_pp),
                db=pipeline.db,
                draft_id=int(_dqf),
                **_pp_kw,
            )
        )
        df = pd.concat(dfs_pp, ignore_index=True) if dfs_pp else pd.DataFrame()
    else:
        if not source_dir.exists():
            raise FileNotFoundError("Missing source lights directory. Run calibration/import first.")
        df = qc_enrich_calibrated_lights_in_place(
            calibrated_root=source_dir,
            progress_cb=progress_cb,
            db=pipeline.db,
            draft_id=None,
            **_pp_kw,
        )

    pipeline.quick_preprocess_last_import(archive_path=ap_root, run=False)

    _ph_done = pending.get("processing_hash")
    _dqf_done = pending.get("quality_filter_draft_id")
    if _dqf_done is not None and _ph_done:
        try:
            pipeline.db.record_qc_processing_apply(
                int(_dqf_done),
                str(_ph_done),
                overwrite=bool(pending.get("overwrite_qc_processing")),
            )
            pipeline.db.update_obs_draft_status(int(_dqf_done), "PROCESSED")
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("QC snapshot / draft status write failed: %s", exc)


def _night_run_platesolve(
    *,
    pending: dict[str, Any],
    ap: Path,
    pipeline: AstroPipeline,
    plan: Any,
    progress_cb: Callable[[int, int, str], None],
) -> dict[str, Any]:
    """Headless equivalent of ``app._vyvar_execute_platesolve_pending``."""
    from pipeline import astrometry_align_and_build_masterstar

    cfg_run = pipeline.config
    try:
        cfg_run.sips_dao_fwhm_px = float(
            pending.get("dao_fwhm_px", getattr(cfg_run, "sips_dao_fwhm_px", 2.5))
        )
    except (TypeError, ValueError):
        pass
    try:
        cfg_run.sips_dao_threshold_sigma = float(
            pending.get("dao_threshold_sigma", getattr(cfg_run, "sips_dao_threshold_sigma", 3.5))
        )
    except (TypeError, ValueError):
        pass

    _ms_pct = pending.get("masterstar_selection_pct")
    try:
        ms_pct_f = float(_ms_pct) if _ms_pct is not None else _DEFAULT_MASTERSTAR_SELECTION_PCT
    except (TypeError, ValueError):
        ms_pct_f = _DEFAULT_MASTERSTAR_SELECTION_PCT

    md_ps: Path | None = None
    if plan is not None and getattr(plan, "dark_master", None):
        md_ps = Path(plan.dark_master)
        if not md_ps.exists():
            md_ps = None

    _peq = pending.get("id_equipment")
    _draft_ps = pending.get("draft_id")

    return astrometry_align_and_build_masterstar(
        archive_path=ap,
        app_config=cfg_run,
        astrometry_api_key=(str(pending.get("astrometry_api_key", "")).strip() or None),
        max_control_points=int(pending.get("max_control_points", cfg_run.alignment_max_control_points)),
        min_detected_stars=int(pending.get("min_detected_stars", 100)),
        max_detected_stars=int(pending.get("max_detected_stars", 500)),
        platesolve_backend=str(pending.get("platesolve_backend", "vyvar")),
        plate_solve_fov_deg=float(pending.get("plate_solve_fov_deg", 1.0)),
        max_extra_platesolve=int(pending.get("max_extra_platesolve", 0)),
        catalog_match_max_sep_arcsec=float(pending.get("catalog_match_max_sep_arcsec", 25.0)),
        saturate_level_fraction=float(pending.get("saturate_level_fraction", 0.999)),
        max_catalog_rows=int(pending.get("max_catalog_rows", 12000)),
        n_comparison_stars=int(pending.get("n_comparison_stars", 150)),
        faintest_mag_limit=(
            None if pending.get("faintest_mag_limit") is None else float(pending["faintest_mag_limit"])
        ),
        dao_threshold_sigma=float(
            pending.get("dao_threshold_sigma", getattr(cfg_run, "sips_dao_threshold_sigma", 3.5))
        ),
        id_equipment=int(_peq) if _peq is not None else None,
        draft_id=int(_draft_ps) if _draft_ps is not None else None,
        catalog_local_gaia_only=True,
        build_masterstar_and_catalogs=bool(pending.get("build_masterstar_and_catalogs", False)),
        ram_align_and_catalog=True,
        progress_cb=progress_cb,
        masterstar_candidate_paths=list(pending.get("masterstar_candidate_paths") or []),
        masterstar_selection_pct=ms_pct_f,
        master_dark_path=md_ps,
    )


def audit_photometry_completeness(
    output_dir: Path,
    *,
    min_ratio: float = _PHOTOMETRY_COMPLETENESS_MIN_RATIO,
) -> dict[str, Any]:
    """False-success guard: did photometry process every *measurable* active target?

    Compares ``photometry_summary.csv`` rows to ``active_targets.csv``, but the verdict is taken
    against **measurable** targets only. A target is counted *unmeasurable* (honest, must NOT fail
    the run) when it produced no summary row AND is fainter than the deepest target that *was*
    measured - i.e. below the achieved per-setup detection depth (undetected / too faint / RED).
    Targets that are missing yet bright enough to be measurable (<= achieved depth) are *measurable
    misses* and still fail the run - this preserves the silent-truncation guard (draft_383/385:
    e.g. 69/373), where a cut-short process drops bright, on-frame, detectable targets.

    Depth is taken from the data itself (faintest measured target's catalog mag), so no new
    threshold/parameter is introduced. Conservative fallbacks (no mag, nothing measured) count a
    miss as *measurable* so truncation can never masquerade as honest unmeasurability.
    """
    output_dir = Path(output_dir)
    active_csv = output_dir / "active_targets.csv"
    summary_csv = output_dir / "photometry_summary.csv"
    out: dict[str, Any] = {
        "output_dir": str(output_dir),
        "n_active_targets": 0,
        "n_summary_rows": 0,
        "n_unmeasurable_missing": 0,
        "n_measurable_active": 0,
        "n_measurable_missing": 0,
        "achieved_depth_mag": float("nan"),
        "min_ratio": float(min_ratio),
        "ratio": float("nan"),
        "measurable_ratio": float("nan"),
        "ok": False,
    }
    if not active_csv.is_file():
        out["error"] = f"missing {active_csv.name}"
        return out
    if not summary_csv.is_file():
        out["error"] = f"missing {summary_csv.name}"
        return out
    try:
        at_df = pd.read_csv(active_csv, low_memory=False)
        sm_df = pd.read_csv(summary_csv, low_memory=False)
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)
        return out
    n_active = int(len(at_df))
    n_summary = int(len(sm_df))
    out["n_active_targets"] = n_active
    out["n_summary_rows"] = n_summary
    if n_active <= 0:
        out["ratio"] = 1.0 if n_summary == 0 else 0.0
        out["measurable_ratio"] = out["ratio"]
        out["ok"] = n_summary == 0
        return out

    raw_ratio = n_summary / n_active
    out["ratio"] = float(raw_ratio)

    # Classify missing targets (active but no summary row) by achieved depth.
    def _norm_id(s: pd.Series) -> pd.Series:
        return s.astype(str).str.strip()

    n_measurable_missing = n_active - n_summary  # conservative default (treat all as measurable)
    achieved_depth = float("nan")
    if "catalog_id" in at_df.columns and "catalog_id" in sm_df.columns:
        measured_ids = set(_norm_id(sm_df["catalog_id"]).tolist())
        at_ids = _norm_id(at_df["catalog_id"])
        is_measured = at_ids.isin(measured_ids)
        at_mag = pd.to_numeric(at_df.get("mag"), errors="coerce") if "mag" in at_df.columns else None
        if at_mag is not None:
            measured_mag = at_mag[is_measured].dropna()
            if not measured_mag.empty:
                achieved_depth = float(measured_mag.max())
            missing_mag = at_mag[~is_measured]
            if math.isfinite(achieved_depth):
                # Unmeasurable = missing AND fainter than the deepest measured target.
                unmeasurable_mask = (~is_measured) & at_mag.notna() & (at_mag > achieved_depth)
                n_unmeasurable = int(unmeasurable_mask.sum())
            else:
                # Nothing measured -> cannot assert depth -> all misses are measurable (truncation).
                n_unmeasurable = 0
            n_measurable_missing = int((~is_measured).sum()) - n_unmeasurable
            out["n_unmeasurable_missing"] = n_unmeasurable

    n_measurable_missing = max(0, int(n_measurable_missing))
    out["n_measurable_missing"] = n_measurable_missing
    n_measurable_active = n_summary + n_measurable_missing
    out["n_measurable_active"] = int(n_measurable_active)
    out["achieved_depth_mag"] = float(achieved_depth)

    measurable_ratio = (n_summary / n_measurable_active) if n_measurable_active > 0 else 1.0
    out["measurable_ratio"] = float(measurable_ratio)
    out["ok"] = measurable_ratio >= float(min_ratio)
    return out


def _collect_photometry_metrics(output_dir: Path) -> tuple[int, float]:
    """Read ``photometry_summary.csv`` for LC count and median ``lc_rms``."""
    summary = output_dir / "photometry_summary.csv"
    if not summary.is_file():
        return 0, float("nan")
    try:
        df = pd.read_csv(summary, low_memory=False)
    except Exception as exc:  # noqa: BLE001
        logging.warning('[EXC-0113] intent unclear (try: / df = pd.read_csv(summary, low_memory=False) / except Exception: ...: %s', exc)
        return 0, float("nan")
    n_lc = int(len(df))
    if "lc_rms" not in df.columns:
        return n_lc, float("nan")
    rms = pd.to_numeric(df["lc_rms"], errors="coerce").dropna()
    med = float(rms.median()) if len(rms) else float("nan")
    return n_lc, med


def run_night_pipeline(params: NightRunParams) -> NightRunResult:
    """Run full VYVAR pipeline headless (mirrors ``_run_vyvar_full_pipeline``)."""
    result = NightRunResult(success=False)
    timings: dict[str, float] = {}
    t_run = time.time()

    from infolog import ensure_infolog_logging  # noqa: PLC0415

    ensure_infolog_logging()

    def _p(msg: str) -> None:
        LOGGER.info("[NightRun] %s", msg)
        if params.progress_cb is not None:
            params.progress_cb(msg)
        if msg.startswith("Step "):
            from infolog import log_phase_boundary  # noqa: PLC0415

            _phase = msg.split(":", 1)[0].replace("Step ", "").strip()
            if _phase:
                log_phase_boundary(_phase, status="start")

    def _t(label: str, t0: float) -> None:
        elapsed = time.time() - t0
        timings[label] = elapsed
        LOGGER.info("[NightRun] [OK] %s - %.1fs", label, elapsed)
        from infolog import log_milestone  # noqa: PLC0415

        log_milestone(f"[PHASE] {label} done {elapsed:.1f}s")

    prog_cb = _make_progress_cb(params.progress_cb)

    try:
        t0 = time.time()
        cfg = _load_app_config(params.config_path)
        if params.sysrem_enabled is not None:
            cfg.sysrem_enabled = bool(params.sysrem_enabled)
        if params.sysrem_n_iter is not None:
            cfg.sysrem_n_iter = int(params.sysrem_n_iter)
        pipeline = AstroPipeline(cfg)
        _t("config", t0)

        source = Path(params.source_dir).expanduser().resolve()
        if not source.is_dir():
            result.errors.append(f"Invalid source directory: {source}")
            result.phase_timings = timings
            return result

        eq_id = int(params.equipment_id)
        tel_id = int(params.telescope_id)
        # Same camera as import/scan (platesolve_equipment_id deprecated alias).
        ps_eq = int(eq_id)

        if params.dry_run:
            _p(f"DRY RUN - scan only: {source}")
            t0 = time.time()
            plan = smart_scan_source(
                source_root=source,
                calibration_library_root=cfg.calibration_library_root,
                masterdark_validity_days=int(cfg.masterdark_validity_days),
                masterflat_validity_days=int(cfg.masterflat_validity_days),
                db=pipeline.db,
                id_equipments=eq_id,
                id_telescope=tel_id,
                calibration_master_ccd_temp_tolerance_c=cfg.calibration_master_ccd_temp_tolerance_c,
            )
            _t("smart_scan_source", t0)
            lights_bad = any(
                r.type == "Lights" and r.status in ("missing", "empty") for r in plan.scan_rows
            )
            if lights_bad or not plan.lights_files:
                result.errors.append("Scan: no light frames found in source directory")
            else:
                result.success = True
                _p(f"DRY RUN OK - {len(plan.lights_files)} light file(s) in plan")
            result.phase_timings = timings
            return result

        # Step 1: Scan
        _p("Step 1: Scan source")
        t0 = time.time()
        plan = smart_scan_source(
            source_root=source,
            calibration_library_root=cfg.calibration_library_root,
            masterdark_validity_days=int(cfg.masterdark_validity_days),
            masterflat_validity_days=int(cfg.masterflat_validity_days),
            db=pipeline.db,
            id_equipments=eq_id,
            id_telescope=tel_id,
            calibration_master_ccd_temp_tolerance_c=cfg.calibration_master_ccd_temp_tolerance_c,
        )
        _t("smart_scan_source", t0)

        lights_bad = any(
            r.type == "Lights" and r.status in ("missing", "empty") for r in plan.scan_rows
        )
        if lights_bad:
            result.errors.append("Scan plan is missing or has empty light frames")
            result.phase_timings = timings
            return result

        if params.manual_flat_map:
            for flt, pth in params.manual_flat_map.items():
                if pth and Path(pth).exists():
                    plan.masterflat_by_filter[flt] = pth

        if params.pre_calibrated_mode:
            from draft_provenance import (
                CALIBRATION_MODE_PRE,
                apply_pre_calibrated_import_plan,
                calibration_mode_report_line,
            )

            apply_pre_calibrated_import_plan(plan)
            _p(calibration_mode_report_line(CALIBRATION_MODE_PRE))

        # Step 2: Import
        _p("Step 2: Import session")
        from observer_location import (  # noqa: PLC0415
            apply_resolved_observer_location_to_config,
            resolve_observer_location_for_run,
        )

        _resolved_site = resolve_observer_location_for_run(
            cfg.database_path,
            explicit_location_id=params.location_id,
            cfg=cfg,
            source_hint="cli_arg" if params.location_id is not None else None,
        )
        apply_resolved_observer_location_to_config(cfg, _resolved_site)
        t0 = time.time()
        import_result = smart_import_session(
            plan=plan,
            pipeline=pipeline,
            id_equipment=eq_id,
            id_telescope=tel_id,
            id_location=_resolved_site.location_id,
            location_source=_resolved_site.source,
            cfg=cfg,
        )
        _t("smart_import_session", t0)

        if getattr(import_result, "draft_id", None) is None:
            result.errors.append("Import did not return draft_id")
            result.phase_timings = timings
            return result

        draft_id = int(import_result.draft_id)
        result.draft_id = draft_id
        ap = Path(str(import_result.archive_path))
        ap_root = ap.parent if ap.name.casefold() == "non_calibrated" else ap
        result.draft_dir = ap_root.resolve()

        from infolog import log_milestone, log_phase_boundary, start_infolog_session  # noqa: PLC0415

        start_infolog_session(result.draft_dir)
        log_milestone(_resolved_site.milestone_line())

        from draft_provenance import (
            CALIBRATION_MODE_PRE,
            CALIBRATION_MODE_VYVAR,
            record_draft_calibration_provenance,
            record_observer_location_provenance,
        )

        record_draft_calibration_provenance(
            db=pipeline.db,
            archive_path=ap_root,
            draft_id=draft_id,
            calibration_mode=CALIBRATION_MODE_PRE if params.pre_calibrated_mode else CALIBRATION_MODE_VYVAR,
        )
        record_observer_location_provenance(
            archive_path=ap_root,
            draft_id=draft_id,
            resolved=_resolved_site,
        )

        md = Path(plan.dark_master) if getattr(plan, "dark_master", None) else None
        mf_map: dict[str, Path | None] = {}
        for k, v in (getattr(plan, "masterflat_by_filter", None) or {}).items():
            mf_map[str(k)] = Path(v) if v else None
        mf_obs: dict[str, Path | None] = {}
        for k, v in (getattr(plan, "masterflat_by_obs_key", None) or {}).items():
            mf_obs[str(k)] = Path(str(v)) if v else None
        dm_obs: dict[str, Path | None] = {}
        for k, v in (getattr(plan, "dark_master_by_obs_key", None) or {}).items():
            dm_obs[str(k)] = Path(str(v)) if v else None

        if params.pre_calibrated_mode:
            md = None
            mf_map = {}
            mf_obs = {}
            dm_obs = {}
            _p(
                "Step 3: skipped - pre-calibrated import; downstream reads non_calibrated/lights "
                "(no calibrated/ directory)"
            )
            _cal_out: dict[str, Any] = {"pre_calibrated_skip_calibration": True}
            timings["calibration"] = 0.0
        else:
            _p("Step 3: Calibration")
            t0 = time.time()
            _cal_out = pipeline.quick_calibrate_last_import(
                archive_path=ap,
                master_dark_path=md if (md and md.exists()) else None,
                masterflat_by_filter=mf_map,
                progress_cb=prog_cb,
                equipment_id=eq_id,
                draft_id=draft_id,
                observation_id=getattr(import_result, "observation_id", None),
                masterflat_by_obs_key=mf_obs or None,
                master_dark_by_obs_key=dm_obs or None,
                roundness_reject_above=float(params.roundness_reject_above),
            )
            _t("calibration", t0)

        from draft_provenance import resolve_draft_lights_root

        lights_root = resolve_draft_lights_root(ap_root, draft_id=draft_id, db=pipeline.db)
        if not lights_root.exists():
            _msg = (
                "Missing non_calibrated/lights after pre-cal import"
                if params.pre_calibrated_mode
                else "Missing /calibrated/lights after calibration"
            )
            result.errors.append(_msg)
            result.phase_timings = timings
            return result

        qsum: dict[str, Any] = dict(_cal_out.get("perf10_qsum") or {})

        # Step 4: Memory profile
        from pipeline import (
            estimate_archive_memory_profile,
            generate_observation_hash,
            resolve_preprocess_target_coordinates,
            run_draft_ram_calibration_qc_to_obs_files,
            scan_calibrated_lights_pointing,
        )

        t0 = time.time()
        estimate_archive_memory_profile(ap_root)
        _t("memory_profile", t0)

        # Step 5: RAM QC (skipped when PERF-10: DAO QC already done in calibration)
        _perf10_ok = (
            not params.pre_calibrated_mode
            and bool(getattr(cfg, "dao_qc_in_calibrate", False))
            and int(qsum.get("n_successful_fwhm") or 0) > 0
        )
        if not _perf10_ok:
            _p("Step 5: RAM QC -> manifest files[]")
            t0 = time.time()
            qsum = run_draft_ram_calibration_qc_to_obs_files(
                db=pipeline.db,
                draft_id=draft_id,
                archive_path=ap_root,
                master_dark_path=md if (md and md.exists()) else None,
                masterflat_by_filter=mf_map,
                masterflat_by_obs_key=mf_obs or None,
                master_dark_by_obs_key=dm_obs or None,
                equipment_id=eq_id,
                pipeline_config=pipeline.config,
                progress_cb=prog_cb,
                roundness_reject_above=float(params.roundness_reject_above),
            )
            _t("ram_qc", t0)
        else:
            LOGGER.info("[PERF-10] Step 5 skipped - DAO QC computed during calibration")
            timings["ram_qc"] = 0.0

        # Step 6: Pointing
        t0 = time.time()
        pointing = scan_calibrated_lights_pointing(lights_root, max_files=None)
        r_pref = next(
            (
                r
                for r in pointing["rows"]
                if r.get("display_ra_deg") is not None and r.get("display_dec_deg") is not None
            ),
            None,
        )
        ra_ui: float | None = None
        de_ui: float | None = None
        if r_pref:
            ra_ui = float(r_pref["display_ra_deg"])
            de_ui = float(r_pref["display_dec_deg"])
        try:
            mra_q = qsum.get("median_ra_deg")
            mde_q = qsum.get("median_de_deg")
            if mra_q is not None and math.isfinite(float(mra_q)):
                ra_ui = float(mra_q)
            if mde_q is not None and math.isfinite(float(mde_q)):
                de_ui = float(mde_q)
        except (TypeError, ValueError):
            pass
        _t("pointing_scan", t0)

        # Step 7: Auto FWHM
        fwhm_lim = 0.0
        if bool(cfg.auto_fwhm_enabled):
            _p("Step 7: Auto FWHM limit")
            t0 = time.time()
            from photometry_core import compute_auto_fwhm_limit

            rows_f = pipeline.db.fetch_draft_light_rows_for_quality(draft_id)
            if rows_f:
                df_f = pd.DataFrame(rows_f)
                col = next((c for c in ("fwhm_mean", "FWHM", "fwhm") if c in df_f.columns), None)
                if col:
                    vals = df_f[col].dropna().values
                    ar = compute_auto_fwhm_limit(vals, k=float(cfg.auto_fwhm_k_factor))
                    if ar.get("auto_limit") is not None:
                        fwhm_lim = float(ar["auto_limit"])
                        LOGGER.info(
                            "[NightRun] Auto FWHM limit=%.3f px (k=%.2f)",
                            fwhm_lim,
                            float(cfg.auto_fwhm_k_factor),
                        )
            _t("auto_fwhm", t0)

        # Step 8: MASTERSTAR TOP1
        _p("Step 8: Auto-select MASTERSTAR (TOP1)")
        t0 = time.time()
        from pipeline import draft_is_multi_group_obs

        _multi_obs = draft_is_multi_group_obs(ap_root)
        use_path = ""
        if _multi_obs:
            LOGGER.info(
                "[NightRun] Multi-group draft (%s): per-group MASTERSTAR from each obs_group; "
                "skipping global TOP1 cross-group path",
                ap_root,
            )
            try:
                pipeline.db.set_obs_draft_masterstar_source_path(draft_id, None)
            except Exception as exc:  # noqa: BLE001
                result.warnings.append(f"MASTERSTAR DB path clear: {exc}")
        else:
            rows_ms = pipeline.db.fetch_draft_light_rows_for_quality(draft_id)
            if not rows_ms:
                result.errors.append("Empty manifest files[] for draft")
                result.phase_timings = timings
                return result
            df_ms = pd.DataFrame(rows_ms)
            for col in (
                "CALIB_FLAGS",
                "FWHM",
                "SKY_LEVEL",
                "STAR_COUNT",
                "REJECTED_AUTO",
                "IS_REJECTED",
                "ELONGATION_MEAN",
            ):
                if col not in df_ms.columns:
                    df_ms[col] = np.nan if col != "IS_REJECTED" else 0
            df_ms["FWHM"] = pd.to_numeric(df_ms["FWHM"], errors="coerce")
            df_ms["IS_REJECTED"] = pd.to_numeric(df_ms["IS_REJECTED"], errors="coerce").fillna(0).astype(int)
            ms_eligible = df_ms[df_ms["IS_REJECTED"] == 0].copy()
            if fwhm_lim > 0.0:
                ms_eligible = ms_eligible[
                    ms_eligible["FWHM"].notna() & (ms_eligible["FWHM"] <= fwhm_lim)
                ]
            if ms_eligible.empty:
                result.errors.append("No light row after IS_REJECTED and FWHM filter")
                result.phase_timings = timings
                return result
            ms_eligible = ms_eligible.copy()
            ms_eligible["_ms_score"] = _compute_masterstar_score(ms_eligible)
            ms_eligible = ms_eligible.sort_values("_ms_score", ascending=False).reset_index(drop=True)
            top_path = str(ms_eligible["FILE_PATH"].iloc[0]).strip()
            use_path = (
                _masterstar_candidate_path_for_job(
                    ap_root, top_path, draft_id=draft_id, db=pipeline.db
                )
                or top_path
            ).strip()
            if not use_path:
                result.errors.append("TOP1 MASTERSTAR path is empty")
                result.phase_timings = timings
                return result
            try:
                pipeline.db.set_obs_draft_masterstar_source_path(draft_id, use_path)
            except Exception as exc:  # noqa: BLE001
                result.warnings.append(f"MASTERSTAR DB path write: {exc}")
        _t("masterstar_resolve", t0)

        # Steps 9-10: Coordinates + hash
        t0 = time.time()
        ira, ide = resolve_preprocess_target_coordinates(
            db=pipeline.db,
            draft_id=draft_id,
            ui_ra_deg=ra_ui,
            ui_dec_deg=de_ui,
        )
        coords_ok = False
        try:
            coords_ok = (
                ira is not None
                and ide is not None
                and math.isfinite(float(ira))
                and math.isfinite(float(ide))
                and not (abs(float(ira)) < 1e-9 and abs(float(ide)) < 1e-9)
            )
        except (TypeError, ValueError):
            coords_ok = False
        ph: str | None = None
        try:
            ph = generate_observation_hash(pipeline.db, draft_id)
        except Exception:  # noqa: BLE001
            ph = None
        _t("preprocess_prep", t0)

        dao_fwhm = float(
            params.dao_fwhm_px
            if params.dao_fwhm_px is not None
            else max(1.0, min(8.0, float(getattr(cfg, "sips_dao_fwhm_px", 2.5))))
        )
        dao_sigma = float(
            params.dao_threshold_sigma
            if params.dao_threshold_sigma is not None
            else max(1.0, min(10.0, float(getattr(cfg, "sips_dao_threshold_sigma", 3.5))))
        )
        plate_fov = float(
            params.plate_fov_deg
            if params.plate_fov_deg is not None
            else float(getattr(cfg, "plate_solve_fov_deg", 1.0))
        )

        pipeline.config.sips_dao_fwhm_px = dao_fwhm
        pipeline.config.sips_dao_threshold_sigma = dao_sigma

        job_ms: dict[str, Any] = {
            "kind": "make_masterstar",
            "archive_path": str(ap),
            "fwhm_limit_px": float(fwhm_lim),
            "inject_pointing_ra_deg": (float(ira) if coords_ok else None),
            "inject_pointing_dec_deg": (float(ide) if coords_ok else None),
            "quality_filter_draft_id": draft_id,
            "max_control_points": int(cfg.alignment_max_control_points),
            "min_detected_stars": int(params.min_detected_stars),
            "max_detected_stars": int(params.max_detected_stars),
            "astrometry_api_key": "",
            "platesolve_backend": "vyvar",
            "plate_solve_fov_deg": plate_fov,
            "max_extra_platesolve": int(params.max_extra_platesolve),
            "catalog_match_max_sep_arcsec": float(params.catalog_match_max_sep_arcsec),
            "saturate_level_fraction": float(params.saturate_level_fraction),
            "max_catalog_rows": int(params.max_catalog_rows),
            "n_comparison_stars": 150,
            "faintest_mag_limit": None,
            "dao_threshold_sigma": dao_sigma,
            "dao_fwhm_px": dao_fwhm,
            "id_equipment": ps_eq,
            "draft_id": draft_id,
            "catalog_local_gaia_only": True,
            "build_masterstar_and_catalogs": True,
            "masterstar_candidate_paths": [] if _multi_obs else [use_path],
            "masterstar_selection_pct": _DEFAULT_MASTERSTAR_SELECTION_PCT,
        }
        if ph:
            job_ms["processing_hash"] = ph
            job_ms["overwrite_qc_processing"] = True

        # Step 11: Preprocess
        _p("Step 11: Preprocess (calibrated -> processed)")
        t0 = time.time()
        _night_run_preprocess(pending=job_ms, ap=ap, pipeline=pipeline, progress_cb=prog_cb)
        _t("preprocess", t0)

        # Step 12: Platesolve + MASTERSTAR
        _p("Step 12: Plate solve + alignment + MASTERSTAR")
        t0 = time.time()
        ps_out = _night_run_platesolve(
            pending=job_ms,
            ap=ap,
            pipeline=pipeline,
            plan=plan,
            progress_cb=prog_cb,
        )
        _t("platesolve_align_masterstar", t0)
        if isinstance(ps_out, dict) and ps_out.get("error"):
            result.errors.append(f"Platesolve failed: {ps_out.get('error')}")
            result.phase_timings = timings
            return result

        draft_dir = (Path(cfg.archive_root) / "Drafts" / f"draft_{draft_id:06d}").resolve()
        if params.post_platesolve_hook is not None:
            _p("Step 12b: post-platesolve hook")
            t0 = time.time()
            try:
                params.post_platesolve_hook(draft_id, draft_dir, cfg, pipeline)
            except Exception as hook_exc:  # noqa: BLE001
                result.errors.append(f"post_platesolve_hook failed: {hook_exc}")
                result.phase_timings = timings
                return result
            _t("post_platesolve_hook", t0)

        # Steps 13-14: Photometry
        from photometry_core import run_full_photometry_pipeline
        from ui_aperture_photometry import _find_phase2a_paths

        _p("Step 13: Discover obs groups")
        t0 = time.time()
        all_setups = _find_phase2a_paths(cfg, draft_id, draft_dir_override=None)
        _t("discover_obs_groups", t0)

        if not all_setups:
            result.errors.append("No platesolve setups (per_frame_catalog_index.csv)")
            result.phase_timings = timings
            return result

        draft_dir = (Path(cfg.archive_root) / "Drafts" / f"draft_{draft_id:06d}").resolve()
        aligned_root = draft_dir / "detrended_aligned" / "lights"
        run_groups: list[str] = []
        for nm in sorted(all_setups.keys()):
            og_dir = aligned_root / str(nm)
            if og_dir.is_dir() and any(og_dir.glob("proc_*.csv")):
                run_groups.append(str(nm))
        if not run_groups:
            run_groups = list(sorted(all_setups.keys()))

        def _prog_phot(msg: str) -> None:
            _p(str(msg))

        phot_errors: list[str] = []
        completeness_issues: list[str] = []
        completeness_by_setup: dict[str, dict[str, Any]] = {}
        total_lc = 0
        total_frames = 0
        lc_rms_values: list[float] = []

        for nm in run_groups:
            p = all_setups.get(str(nm)) or {}
            ms_fits = Path(p["masterstar_fits"]) if p.get("masterstar_fits") else None
            og_dir = Path(p["obs_group_dir"]) if p.get("obs_group_dir") else None
            ms_csv = (og_dir / "masterstars_full_match.csv") if og_dir is not None else None
            vt_csv = (og_dir / "variable_targets.csv") if og_dir is not None else None
            pf_dir = Path(p["per_frame_csv_dir"]) if p.get("per_frame_csv_dir") else None
            dt_dir = Path(p["detrended_aligned_dir"]) if p.get("detrended_aligned_dir") else None
            out_d = Path(p["output_dir"]) if p.get("output_dir") else None

            missing: list[str] = []
            if ms_fits is None or not ms_fits.exists():
                missing.append("MASTERSTAR.fits")
            if ms_csv is None or not ms_csv.exists():
                missing.append("masterstars_full_match.csv")
            if vt_csv is None or not vt_csv.exists():
                missing.append("variable_targets.csv")
            if pf_dir is None or not pf_dir.exists():
                missing.append("per-frame CSV directory")
            if dt_dir is None or not dt_dir.exists():
                missing.append("detrended_aligned directory")
            if out_d is None:
                missing.append("output_dir")
            if missing:
                phot_errors.append(f"{nm}: missing {', '.join(missing)}")
                continue

            _p(f"Step 14: Photometry - {nm}")
            t0 = time.time()
            try:
                phot_result = run_full_photometry_pipeline(
                    masterstar_fits_path=ms_fits,
                    variable_targets_csv=vt_csv,
                    masterstars_csv=ms_csv,
                    per_frame_csv_dir=pf_dir,
                    detrended_aligned_dir=dt_dir,
                    output_dir=out_d,
                    cfg=cfg,
                    db=pipeline.db,
                    draft_id=draft_id,
                    progress_cb=_prog_phot,
                )
            except Exception as exc_nm:  # noqa: BLE001
                phot_errors.append(f"{nm}: {exc_nm}")
                continue
            _t(f"photometry_{nm}", t0)

            p2a = phot_result.get("phase2a") or {}
            total_lc += int(p2a.get("n_lightcurves") or 0)
            total_frames += int(p2a.get("n_frames") or 0)
            result.output_dir = out_d

            sysrem = phot_result.get("sysrem")
            if sysrem:
                try:
                    result.sysrem_improvement_pct = float(sysrem.get("rms_improvement_pct", float("nan")))
                except (TypeError, ValueError):
                    pass

            n_sum, med_rms = _collect_photometry_metrics(out_d)
            if n_sum > 0:
                total_lc = max(total_lc, n_sum)
            if math.isfinite(med_rms):
                lc_rms_values.append(med_rms)

            audit = audit_photometry_completeness(out_d)
            completeness_by_setup[str(nm)] = audit
            if not audit.get("ok"):
                n_sm = int(audit.get("n_summary_rows") or 0)
                n_at = int(audit.get("n_active_targets") or 0)
                n_meas = int(audit.get("n_measurable_active") or 0)
                n_unmeas = int(audit.get("n_unmeasurable_missing") or 0)
                mratio = audit.get("measurable_ratio")
                mratio_s = (
                    f"{float(mratio):.1%}"
                    if mratio is not None and math.isfinite(float(mratio))
                    else "n/a"
                )
                err_detail = audit.get("error")
                if err_detail:
                    completeness_issues.append(f"{nm}: {err_detail}")
                else:
                    completeness_issues.append(
                        f"{nm}: photometry_summary {n_sm}/{n_meas} measurable targets "
                        f"({mratio_s} coverage, min {_PHOTOMETRY_COMPLETENESS_MIN_RATIO:.0%} required; "
                        f"{n_unmeas} of {n_at} active are below achieved depth -> unmeasurable)"
                    )

            # Step 15: PDF per group
            try:
                from photometry_report import generate_all_method_photometry_reports

                pdf_paths = generate_all_method_photometry_reports(
                    draft_dir=draft_dir,
                    obs_group=str(nm),
                    tess_results={},
                    base_report_title="VYVAR - Summary Measure Report",
                )
                for pdf_path in pdf_paths:
                    _p(f"PDF report: {Path(pdf_path).name}")
            except Exception as pdf_err:  # noqa: BLE001
                result.warnings.append(f"PDF report {nm}: {pdf_err}")

        result.photometry_completeness = completeness_by_setup

        if phot_errors:
            result.errors.extend(phot_errors)
            result.phase_timings = timings
            return result

        if completeness_issues:
            result.errors.append("Photometry completeness gate FAILED")
            result.errors.extend(completeness_issues)
            result.phase_timings = timings
            return result

        result.n_lightcurves = total_lc
        result.n_frames = total_frames
        if lc_rms_values:
            result.lc_rms_median = float(np.median(lc_rms_values))

        result.success = True
        timings["total"] = time.time() - t_run
        result.phase_timings = timings
        if result.draft_dir is not None:
            from infolog import write_run_infolog  # noqa: PLC0415

            saved = write_run_infolog(result.draft_dir)
            if saved:
                _p(f"Infolog saved: {Path(saved).name}")
        _p(
            f"Night run complete - draft {draft_id}, {total_lc} light curve(s), "
            f"{total_frames} frame(s)"
        )
        return result

    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("[NightRun] FATAL: %s", exc)
        result.errors.append(str(exc))
        timings["total"] = time.time() - t_run
        result.phase_timings = timings
        return result
