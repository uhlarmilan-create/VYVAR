"""Headless VYVAR night pipeline runner.

Production entry: ``run_night_pipeline``. UI RUN VYVAR
(``app._run_vyvar_full_pipeline``) is a wrapper that resolves optics /
location / flats into ``NightRunParams`` and writes Streamlit state from
the result. C3 (Aperture Photometry page) calls ``run_night_photometry``.

No Streamlit dependencies. Progress via ``logging`` and optional callback.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import sys
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
    existing_pipeline: Any | None = None
    location_id: int | None = None
    location_source_hint: str | None = None
    platesolve_equipment_id: int | None = None
    sysrem_enabled: bool | None = None
    sysrem_n_iter: int | None = None
    progress_cb: Callable[[str], None] | None = None
    dry_run: bool = False
    manual_flat_map: dict[str, str] | None = None
    # UI session_state vyvar_flatfb_{group_key} -> source obs_key. Empty = bookkeeping only.
    flat_fallback_choices: dict[str, str] | None = None
    apply_smart_plan_flat_fallbacks: bool = False
    masterdark_validity_days: int | None = None
    masterflat_validity_days: int | None = None
    # Resolved optics from W1; None = use equipment_id/telescope_id (current W2).
    optics: Any | None = None
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
    # INV-EPSF-STAGE-01: ePSF after aperture. None = read config epsf_auto_run (default OFF).
    # Explicit True/False overrides the config key (UI buttons / gates).
    epsf: bool | None = None


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
    import_result: Any = None
    plan: Any = None
    archive_path: Path | None = None
    ram_qc_summary: dict[str, Any] = field(default_factory=dict)
    pointing: dict[str, Any] = field(default_factory=dict)
    fwhm_limit_px: float = 0.0
    fwhm_limit_source: str = "unset"
    masterstar_path: str = ""
    processing_hash: str | None = None
    platesolve_job_output: Any = None
    ra_ui: float | None = None
    de_ui: float | None = None
    memory_profile: Any = None
    cfg_source: str = "live"
    cfg_changed_keys: list[str] = field(default_factory=list)
    zero_target_setups: list[str] = field(default_factory=list)
    completed_setups: list[str] = field(default_factory=list)
    photometry_context: dict[str, Any] = field(default_factory=dict)
    calibration_output: dict[str, Any] = field(default_factory=dict)
    epsf_stage: dict[str, Any] = field(default_factory=dict)


# Phase 2A must cover at least this fraction of active_targets in photometry_summary.
_PHOTOMETRY_COMPLETENESS_MIN_RATIO = 0.90

_NIGHT_RUN_INPUT_CAMERA = "camera"
_NIGHT_RUN_INPUT_TELESCOPE = "telescope"
_NIGHT_RUN_INPUT_SITE = "observing site"


def resolve_epsf_run(epsf: bool | None, cfg: Any | None = None) -> bool:
    """Whether to run the ePSF night-run stage.

    Explicit True/False wins. None (or omitted) reads ``cfg.epsf_auto_run``
    (default False). ``run_epsf_stage(params=None)`` is the button path and
    always runs -- it does not call this helper.
    """
    if epsf is True:
        return True
    if epsf is False:
        return False
    if cfg is None:
        return False
    return bool(getattr(cfg, "epsf_auto_run", False))


def _positive_id(value: Any) -> int | None:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def resolve_night_run_cli_ids(
    *,
    equipment_id: int | None = None,
    telescope_id: int | None = None,
    location_id: int | None = None,
    draft_dir: Path | str | None = None,
    cfg: Any | None = None,
) -> tuple[int | None, int | None, int | None, list[str]]:
    """Resolve camera / telescope / observing site the way W1 does.

    Explicit CLI ids win. Else draft manifest ``rig`` (equipment_id,
    telescope_id, location_id). Else ``cfg.observer_location_id`` for site
    only (W1). Missing names: camera, telescope, observing site.
    """
    eq = _positive_id(equipment_id)
    tel = _positive_id(telescope_id)
    loc = _positive_id(location_id)
    if draft_dir is not None:
        from draft_provenance import _manifest_rig_pair, _optional_int, load_draft_manifest

        manifest = load_draft_manifest(Path(draft_dir))
        if manifest:
            m_eq, m_tel = _manifest_rig_pair(manifest)
            rig = manifest.get("rig") if isinstance(manifest.get("rig"), dict) else {}
            m_loc = _optional_int(rig.get("location_id"))
            if eq is None:
                eq = _positive_id(m_eq)
            if tel is None:
                tel = _positive_id(m_tel)
            if loc is None:
                loc = _positive_id(m_loc)
    if loc is None and cfg is not None:
        loc = _positive_id(getattr(cfg, "observer_location_id", 0))
    missing: list[str] = []
    if eq is None:
        missing.append(_NIGHT_RUN_INPUT_CAMERA)
    if tel is None:
        missing.append(_NIGHT_RUN_INPUT_TELESCOPE)
    if loc is None:
        missing.append(_NIGHT_RUN_INPUT_SITE)
    return eq, tel, loc, missing


def missing_night_run_inputs(
    *,
    equipment_id: int | None = None,
    telescope_id: int | None = None,
    location_id: int | None = None,
    draft_dir: Path | str | None = None,
    cfg: Any | None = None,
) -> list[str]:
    """Names of unresolved required night-run inputs (camera, telescope, observing site)."""
    _eq, _tel, _loc, missing = resolve_night_run_cli_ids(
        equipment_id=equipment_id,
        telescope_id=telescope_id,
        location_id=location_id,
        draft_dir=draft_dir,
        cfg=cfg,
    )
    return missing


def night_run_missing_message(missing: list[str]) -> str:
    named = ", ".join(missing)
    return (
        f"Night run refused: missing {named}. "
        "Provide --camera/--eq, --telescope/--tel, and --site/--location "
        "(or a draft manifest that resolves them; site may also come from "
        "config observer_location_id)."
    )


def parse_night_run_cli(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "VYVAR night run. Requires the same three inputs as UI RUN VYVAR: "
            "telescope, camera, observing site."
        )
    )
    parser.add_argument("--source", required=True, help="Source directory with FITS files")
    parser.add_argument(
        "--camera",
        "--eq",
        dest="equipment_id",
        type=int,
        default=None,
        help="Camera / equipment DB id (no silent default)",
    )
    parser.add_argument(
        "--telescope",
        "--tel",
        dest="telescope_id",
        type=int,
        default=None,
        help="Telescope DB id (no silent default)",
    )
    parser.add_argument(
        "--site",
        "--location",
        dest="location_id",
        type=int,
        default=None,
        help="Observing site LOCATION id (else config observer_location_id)",
    )
    parser.add_argument(
        "--draft-dir",
        type=Path,
        default=None,
        help="Optional draft directory whose manifest can supply the three ids",
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to config.json")
    parser.add_argument("--dry-run", action="store_true", help="Scan only; no pipeline")
    return parser.parse_args(argv)


def apply_smart_plan_flat_fallbacks(
    plan: Any,
    choices: dict[str, str] | None = None,
) -> None:
    """Apply per-observation flat fallbacks (Streamlit-free).

    ``choices`` maps group_key -> source obs_key. Missing / ``__skip__``
    leaves that group unchanged and only refreshes missing-flat bookkeeping.
    """
    ogs = getattr(plan, "observation_groups", None) or {}
    if not ogs:
        return
    mf = dict(getattr(plan, "masterflat_by_obs_key", None) or {})
    choices = dict(choices or {})
    for p in getattr(plan, "flat_fallback_prompts", None) or []:
        gk = str(p.get("group_key") or "")
        if not gk:
            continue
        choice = choices.get(gk, "__skip__")
        if choice and choice != "__skip__":
            src = mf.get(str(choice))
            if src and Path(str(src)).is_file():
                mf[gk] = src
    missing = sorted(
        gk
        for gk in ogs
        if not (mf.get(gk) and str(mf[gk]).strip() and Path(str(mf[gk])).is_file())
    )
    plan.masterflat_by_obs_key = mf
    plan.missing_obs_keys = missing
    plan.missing_flat_filters = sorted({ogs[gk]["filter"] for gk in missing})
    mfb: dict[str, Any] = {}
    for gk, g in ogs.items():
        fln = g["filter"]
        pth = mf.get(gk)
        if fln not in mfb or pth is not None and mfb[fln] is None:
            mfb[fln] = pth
    plan.masterflat_by_filter = mfb


def load_draft_config_snapshot(draft_dir: Path) -> dict[str, Any] | None:
    """Return ``provenance.config_snapshot`` from a draft ``pipeline_meta.json``."""
    root = Path(draft_dir)
    paths: list[Path] = []
    if root.is_dir():
        paths.extend(sorted(root.glob("platesolve/*/photometry/pipeline_meta.json")))
    direct = root / "photometry" / "pipeline_meta.json"
    if direct.is_file():
        paths.append(direct)
    for meta_path in paths:
        try:
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        prov = raw.get("provenance") if isinstance(raw, dict) else None
        snap = prov.get("config_snapshot") if isinstance(prov, dict) else None
        if isinstance(snap, dict) and snap:
            return dict(snap)
    return None


def overlay_config_snapshot(
    live: AppConfig,
    snapshot: dict[str, Any],
) -> tuple[AppConfig, list[str]]:
    """Copy snapshot keys onto a deepcopy of live cfg. Return (cfg, changed_keys)."""
    cfg = copy.deepcopy(live)
    live_d = live.to_dict() if hasattr(live, "to_dict") else {}
    changed: list[str] = []
    for k, v in snapshot.items():
        if not str(k) or str(k).startswith("_"):
            continue
        if not hasattr(cfg, k):
            continue
        old = live_d.get(k, getattr(cfg, k, None))
        if old == v:
            continue
        try:
            setattr(cfg, k, v)
        except (TypeError, ValueError, AttributeError):
            continue
        changed.append(str(k))
    return cfg, sorted(changed)


def resolve_cfg_for_photometry(
    live_cfg: AppConfig,
    draft_dir: Path | None,
    *,
    existing_draft: bool,
) -> tuple[AppConfig, str, list[str]]:
    """INV-CFG-SOURCE-01: snapshot cfg on re-run; live cfg only for a new draft."""
    if not existing_draft or draft_dir is None:
        return live_cfg, "live", []
    snap = load_draft_config_snapshot(Path(draft_dir))
    if not snap:
        return live_cfg, "live_no_snapshot", []
    cfg, changed = overlay_config_snapshot(live_cfg, snap)
    LOGGER.info(
        "[NightRun] INV-CFG-SOURCE-01 source=draft_snapshot changed_keys=%s",
        changed,
    )
    return cfg, "draft_snapshot", changed


def stamp_frame_qc_provenance(
    ap_root: Path,
    *,
    draft_id: int,
    fwhm_limit_px: float,
    fwhm_limit_source: str,
    cfg_source: str = "live",
    cfg_changed_keys: list[str] | None = None,
) -> Path:
    """Write FRAME-QC + cfg-source stamp next to the draft archive."""
    payload = {
        "quality_filter_draft_id": int(draft_id),
        "fwhm_limit_px": float(fwhm_limit_px),
        "fwhm_limit_source": str(fwhm_limit_source),
        "cfg_source": str(cfg_source),
        "cfg_changed_keys": list(cfg_changed_keys or []),
    }
    path = Path(ap_root) / "night_run_qc_provenance.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="ascii")
    LOGGER.info("[NightRun] FRAME-QC provenance: %s", payload)
    return path


def resolve_photometry_context_triple(
    cfg: AppConfig,
    *,
    db: Any,
    draft_id: int | None,
    masterstar_fits: Path | None,
) -> dict[str, Any]:
    """Phase 2A plate scale / site / calibration_mode (C3 fire-proof)."""
    from draft_provenance import resolve_calibration_mode
    from param_resolver import resolve_site
    from photometry_core import _get_plate_scale_from_cfg

    hdr = None
    ms = Path(masterstar_fits) if masterstar_fits is not None else None
    if ms is not None and ms.is_file():
        try:
            from astropy.io import fits as astrofits

            with astrofits.open(ms, memmap=False) as hdul:
                hdr = hdul[0].header.copy()
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("MASTERSTAR header for context triple skipped: %s", exc)
    plate = _get_plate_scale_from_cfg(
        cfg,
        db=db,
        draft_id=draft_id,
        fits_path=ms,
        ms_header=hdr,
    )
    site = resolve_site(hdr, db=db, draft_id=draft_id, cfg=cfg)
    cal = resolve_calibration_mode(draft_id=draft_id, db=db)
    site_s = (
        f"{site.source}:{site.lat}:{site.lon}:{site.elev}"
        if site is not None
        else None
    )
    out = {
        "plate_scale": plate,
        "site": site_s,
        "calibration_mode": cal,
    }
    LOGGER.info(
        "[NightRun] photometry context plate_scale=%s site=%s calibration_mode=%s",
        out["plate_scale"],
        out["site"],
        out["calibration_mode"],
    )
    return out


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
        raise RuntimeError(
            "INV-FRAME-QC-01: quality_filter_draft_id is required; "
            "the unfiltered qc_enrich else-branch is removed"
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
        n_comparison_stars=int(pending.get("n_comparison_stars", 0)),
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
    require_psf: bool = False,
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

    psf_audit = audit_psf_lc_completeness(output_dir, require=bool(require_psf))
    out["psf"] = psf_audit
    if psf_audit.get("applicable") and not psf_audit.get("ok"):
        out["ok"] = False
        out["psf_error"] = psf_audit.get("error")
    return out


def audit_psf_lc_completeness(
    output_dir: Path,
    *,
    require: bool = False,
) -> dict[str, Any]:
    """INV-EPSF-COMPLETE-01: every aperture LC has a PSF LC with n_full>0 or a reason.

    A run whose PSF LCs are all-dropped (n_full=0 for every target) FAILS even
    when drop reasons are recorded. Aperture-only trees with zero PSF files are
    not applicable unless ``require`` is True.
    """
    output_dir = Path(output_dir)
    lc_dir = output_dir / "lightcurves"
    out: dict[str, Any] = {
        "applicable": False,
        "ok": True,
        "n_aperture_lc": 0,
        "n_psf_lc": 0,
        "n_missing_psf": 0,
        "n_full_positive": 0,
        "n_all_dropped": 0,
        "n_no_reason": 0,
        "error": None,
    }
    if not lc_dir.is_dir():
        if require:
            out["applicable"] = True
            out["ok"] = False
            out["error"] = "PSF LC audit required but lightcurves/ missing"
        return out
    ap_ids: list[str] = []
    for p in sorted(lc_dir.glob("lightcurve_*.csv")):
        stem = p.stem
        if stem.endswith("_psf") or stem.endswith("_adaptive"):
            continue
        cid = stem.replace("lightcurve_", "", 1)
        if cid:
            ap_ids.append(cid)
    out["n_aperture_lc"] = len(ap_ids)
    psf_files = list(lc_dir.glob("lightcurve_*_psf.csv"))
    out["n_psf_lc"] = len(psf_files)
    if not ap_ids:
        return out
    if not psf_files and not require:
        return out
    out["applicable"] = True
    missing = 0
    n_full_pos = 0
    n_dropped = 0
    n_no_reason = 0
    for cid in ap_ids:
        psf_path = lc_dir / f"lightcurve_{cid}_psf.csv"
        if not psf_path.is_file():
            missing += 1
            continue
        n_full, has_reason = _psf_lc_n_full_and_reason(psf_path)
        if n_full > 0:
            n_full_pos += 1
        else:
            n_dropped += 1
            if not has_reason:
                n_no_reason += 1
    out["n_missing_psf"] = missing
    out["n_full_positive"] = n_full_pos
    out["n_all_dropped"] = n_dropped
    out["n_no_reason"] = n_no_reason
    if missing:
        out["ok"] = False
        out["error"] = f"{missing} aperture LC(s) missing a PSF LC"
        return out
    if n_no_reason:
        out["ok"] = False
        out["error"] = f"{n_no_reason} PSF LC(s) have n_full=0 and no recorded drop reason"
        return out
    if n_dropped > 0 and n_full_pos == 0:
        out["ok"] = False
        out["error"] = (
            f"all {n_dropped} PSF LC(s) dropped (n_full=0); refusing empty-file success"
        )
        return out
    out["ok"] = True
    return out


def _psf_lc_n_full_and_reason(path: Path) -> tuple[int, bool]:
    n_full = 0
    n_dropped_hdr = 0
    has_reason = False
    try:
        with Path(path).open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.startswith("#"):
                    break
                if line.startswith("# psf_lc_n_epochs_full="):
                    try:
                        n_full = int(line.split("=", 1)[1].strip())
                    except (TypeError, ValueError):
                        n_full = 0
                elif line.startswith("# psf_lc_n_epochs_dropped_pin="):
                    try:
                        n_dropped_hdr = int(line.split("=", 1)[1].strip())
                    except (TypeError, ValueError):
                        n_dropped_hdr = 0
    except OSError:
        return 0, False
    if n_dropped_hdr > 0:
        has_reason = True
    try:
        df = pd.read_csv(path, comment="#", low_memory=False, nrows=8)
    except Exception:  # noqa: BLE001
        df = pd.DataFrame()
    for col in ("psf_epoch_drop_reason", "drop_reason"):
        if col in df.columns:
            vals = df[col].dropna().astype(str).str.strip()
            if any(v and v.lower() not in ("nan", "none") for v in vals.tolist()):
                has_reason = True
                break
    return int(n_full), bool(has_reason)


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


def run_night_photometry(
    *,
    cfg: AppConfig,
    pipeline: AstroPipeline,
    draft_id: int | None,
    draft_dir_override: Path | None = None,
    progress_cb: Callable[[str], None] | None = None,
    timings: dict[str, float] | None = None,
    write_pdfs: bool = True,
    existing_draft: bool = False,
    epsf: bool | None = None,
) -> dict[str, Any]:
    """Photometry-only slice shared by ``run_night_pipeline`` (C1/C2) and C3.

    Forwards ``draft_id`` and ``db``. Completeness gate is shared: a UI run
    can FAIL loudly. INV-CFG-SOURCE-01 applies when ``existing_draft`` is True.
    """
    from photometry_core import merge_photometry_pipeline_meta, run_full_photometry_pipeline
    from ui_aperture_photometry import _find_phase2a_paths

    timings = timings if timings is not None else {}
    out: dict[str, Any] = {
        "errors": [],
        "warnings": [],
        "n_lightcurves": 0,
        "n_frames": 0,
        "lc_rms_median": float("nan"),
        "sysrem_improvement_pct": float("nan"),
        "output_dir": None,
        "photometry_completeness": {},
        "zero_target_setups": [],
        "completed_setups": [],
        "photometry_context": {},
        "cfg_source": "live",
        "cfg_changed_keys": [],
        "epsf_stage": {},
    }

    def _p(msg: str) -> None:
        LOGGER.info("[NightRun] %s", msg)
        if progress_cb is not None:
            progress_cb(msg)

    def _t(label: str, t0: float) -> None:
        elapsed = time.time() - t0
        timings[label] = elapsed
        LOGGER.info("[NightRun] [OK] %s - %.1fs", label, elapsed)

    draft_dir = None
    if draft_dir_override is not None:
        draft_dir = Path(draft_dir_override).resolve()
    elif draft_id is not None:
        draft_dir = (Path(cfg.archive_root) / "Drafts" / f"draft_{int(draft_id):06d}").resolve()

    phot_cfg, cfg_source, changed_keys = resolve_cfg_for_photometry(
        cfg,
        draft_dir,
        existing_draft=bool(existing_draft),
    )
    out["cfg_source"] = cfg_source
    out["cfg_changed_keys"] = list(changed_keys)
    if changed_keys:
        _p(f"INV-CFG-SOURCE-01 changed_keys={changed_keys}")

    _p("Step 13: Discover obs groups")
    t0 = time.time()
    all_setups = _find_phase2a_paths(
        phot_cfg, draft_id, draft_dir_override=draft_dir_override
    )
    _t("discover_obs_groups", t0)

    if not all_setups:
        out["errors"].append("No platesolve setups (per_frame_catalog_index.csv)")
        return out

    if draft_dir is None:
        out["errors"].append("No draft_dir for photometry")
        return out

    aligned_root = draft_dir / "detrended_aligned" / "lights"
    run_groups: list[str] = []
    for nm in sorted(all_setups.keys()):
        og_dir = aligned_root / str(nm)
        if og_dir.is_dir() and any(og_dir.glob("proc_*.csv")):
            run_groups.append(str(nm))
    if not run_groups:
        run_groups = list(sorted(all_setups.keys()))

    first_ms = None
    for nm0 in run_groups:
        p0 = all_setups.get(str(nm0)) or {}
        if p0.get("masterstar_fits"):
            first_ms = Path(p0["masterstar_fits"])
            break
    ctx = resolve_photometry_context_triple(
        phot_cfg,
        db=pipeline.db,
        draft_id=draft_id,
        masterstar_fits=first_ms,
    )
    out["photometry_context"] = ctx

    phot_errors: list[str] = []
    completeness_issues: list[str] = []
    completeness_by_setup: dict[str, dict[str, Any]] = {}
    total_lc = 0
    total_frames = 0
    lc_rms_values: list[float] = []
    zero_target_setups: list[str] = []
    completed: list[str] = []
    sysrem_pct = float("nan")
    last_out: Path | None = None

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
                cfg=phot_cfg,
                db=pipeline.db,
                draft_id=draft_id,
                progress_cb=_p,
            )
        except Exception as exc_nm:  # noqa: BLE001
            phot_errors.append(f"{nm}: {exc_nm}")
            continue
        _t(f"photometry_{nm}", t0)

        try:
            merge_photometry_pipeline_meta(
                out_d,
                {
                    "cfg_source": cfg_source,
                    "cfg_changed_keys": list(changed_keys),
                    "photometry_context": ctx,
                },
            )
        except Exception as meta_exc:  # noqa: BLE001
            out["warnings"].append(f"{nm}: provenance stamp: {meta_exc}")

        if phot_result.get("zero_targets"):
            zero_target_setups.append(str(nm))
            completed.append(str(nm))
            _p(f"{nm}: 0 active targets - photometry not run")
            continue
        if phot_result.get("error"):
            phot_errors.append(f"{nm}: {phot_result.get('error')}")
            continue

        p2a = phot_result.get("phase2a") or {}
        total_lc += int(p2a.get("n_lightcurves") or 0)
        total_frames += int(p2a.get("n_frames") or 0)
        last_out = out_d
        completed.append(str(nm))

        sysrem = phot_result.get("sysrem")
        if sysrem:
            try:
                sysrem_pct = float(sysrem.get("rms_improvement_pct", float("nan")))
            except (TypeError, ValueError):
                pass

        n_sum, med_rms = _collect_photometry_metrics(out_d)
        if n_sum > 0:
            total_lc = max(total_lc, n_sum)
        if math.isfinite(med_rms):
            lc_rms_values.append(med_rms)

        do_epsf = resolve_epsf_run(epsf, phot_cfg)
        if do_epsf:
            from epsf_stage import EpsfStagePaths, run_epsf_stage

            _p(f"Step 14b: ePSF photometry - {nm}")
            t0 = time.time()
            try:
                epsf_out = run_epsf_stage(
                    params=None,
                    paths=EpsfStagePaths(
                        platesolve_dir=og_dir,
                        frames_root=dt_dir,
                        masterstar_fits=ms_fits,
                        masterstars_csv=ms_csv,
                        photometry_dir=out_d,
                    ),
                    cfg=phot_cfg,
                    progress_cb=_p,
                    db=pipeline.db,
                    draft_id=draft_id,
                )
                out["epsf_stage"][str(nm)] = {
                    "epsf_model_sha256": epsf_out.get("epsf_model_sha256"),
                    "n_stars": epsf_out.get("n_stars"),
                    "lc": epsf_out.get("lc"),
                    "merge": {
                        "written": (epsf_out.get("merge") or {}).get("written"),
                        "frames_total": (epsf_out.get("merge") or {}).get("frames_total"),
                    },
                }
            except Exception as epsf_exc:  # noqa: BLE001
                phot_errors.append(f"{nm}: ePSF stage failed: {epsf_exc}")
                _t(f"epsf_{nm}", t0)
                continue
            _t(f"epsf_{nm}", t0)

        audit = audit_photometry_completeness(out_d, require_psf=bool(do_epsf))
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
            err_detail = audit.get("error") or audit.get("psf_error")
            if err_detail:
                completeness_issues.append(f"{nm}: {err_detail}")
            else:
                completeness_issues.append(
                    f"{nm}: photometry_summary {n_sm}/{n_meas} measurable targets "
                    f"({mratio_s} coverage, min {_PHOTOMETRY_COMPLETENESS_MIN_RATIO:.0%} required; "
                    f"{n_unmeas} of {n_at} active are below achieved depth -> unmeasurable)"
                )

        if write_pdfs:
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
                out["warnings"].append(f"PDF report {nm}: {pdf_err}")

    out["photometry_completeness"] = completeness_by_setup
    out["zero_target_setups"] = zero_target_setups
    out["completed_setups"] = completed
    out["output_dir"] = last_out
    out["sysrem_improvement_pct"] = sysrem_pct
    out["n_lightcurves"] = total_lc
    out["n_frames"] = total_frames
    if lc_rms_values:
        out["lc_rms_median"] = float(np.median(lc_rms_values))

    if phot_errors:
        out["errors"].extend(phot_errors)
        return out

    if completeness_issues:
        out["errors"].append("Photometry completeness gate FAILED")
        out["errors"].extend(completeness_issues)
        return out

    return out


def run_ui_night_photometry(
    *,
    cfg: AppConfig,
    pipeline: AstroPipeline,
    draft_id: int | None,
    draft_dir_override: Path | None = None,
    progress_cb: Callable[[str], None] | None = None,
    timings: dict[str, float] | None = None,
    write_pdfs: bool = False,
    existing_draft: bool = True,
    epsf: bool | None = None,
) -> dict[str, Any]:
    """W1/C3 photometry entry (INV-ONE-ENTRY-01). Defaults match a draft re-run."""
    return run_night_photometry(
        cfg=cfg,
        pipeline=pipeline,
        draft_id=draft_id,
        draft_dir_override=draft_dir_override,
        progress_cb=progress_cb,
        timings=timings,
        write_pdfs=write_pdfs,
        existing_draft=existing_draft,
        epsf=epsf,
    )


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
        if params.existing_pipeline is not None:
            pipeline = params.existing_pipeline
            cfg = pipeline.config
        else:
            cfg = _load_app_config(params.config_path)
            pipeline = AstroPipeline(cfg)
        if params.sysrem_enabled is not None:
            cfg.sysrem_enabled = bool(params.sysrem_enabled)
        if params.sysrem_n_iter is not None:
            cfg.sysrem_n_iter = int(params.sysrem_n_iter)
        _t("config", t0)

        source = Path(params.source_dir).expanduser().resolve()
        if not source.is_dir():
            result.errors.append(f"Invalid source directory: {source}")
            result.phase_timings = timings
            return result

        eq_id = int(params.equipment_id)
        tel_id = int(params.telescope_id)
        if params.optics is not None:
            eq_id = int(params.optics.equipment_id)
            tel_id = int(params.optics.telescope_id)
        # Same camera as import/scan (platesolve_equipment_id deprecated alias).
        ps_eq = int(eq_id)
        _md_days = (
            int(params.masterdark_validity_days)
            if params.masterdark_validity_days is not None
            else int(cfg.masterdark_validity_days)
        )
        _mf_days = (
            int(params.masterflat_validity_days)
            if params.masterflat_validity_days is not None
            else int(cfg.masterflat_validity_days)
        )

        if params.dry_run:
            _p(f"DRY RUN - scan only: {source}")
            t0 = time.time()
            plan = smart_scan_source(
                source_root=source,
                calibration_library_root=cfg.calibration_library_root,
                masterdark_validity_days=_md_days,
                masterflat_validity_days=_mf_days,
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
            masterdark_validity_days=_md_days,
            masterflat_validity_days=_mf_days,
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

        if params.apply_smart_plan_flat_fallbacks and not params.pre_calibrated_mode:
            apply_smart_plan_flat_fallbacks(plan, params.flat_fallback_choices)

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

        _loc_hint = params.location_source_hint
        if _loc_hint is None:
            _loc_hint = "cli_arg" if params.location_id is not None else None
        _resolved_site = resolve_observer_location_for_run(
            cfg.database_path,
            explicit_location_id=params.location_id,
            cfg=cfg,
            source_hint=_loc_hint,
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
        result.import_result = import_result
        result.plan = plan
        ap = Path(str(import_result.archive_path))
        ap_root = ap.parent if ap.name.casefold() == "non_calibrated" else ap
        result.draft_dir = ap_root.resolve()
        result.archive_path = ap

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
        result.calibration_output = dict(_cal_out or {})

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
        result.memory_profile = estimate_archive_memory_profile(ap_root)
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
        result.ram_qc_summary = dict(qsum or {})

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
        result.pointing = pointing if isinstance(pointing, dict) else {}
        result.ra_ui = ra_ui
        result.de_ui = de_ui
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

        fwhm_limit_source = (
            "compute_auto_fwhm_limit" if bool(cfg.auto_fwhm_enabled) else "auto_fwhm_disabled"
        )
        result.fwhm_limit_px = float(fwhm_lim)
        result.fwhm_limit_source = fwhm_limit_source

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
            "max_control_points": int(params.max_control_points),
            "min_detected_stars": int(params.min_detected_stars),
            "max_detected_stars": int(params.max_detected_stars),
            "astrometry_api_key": "",
            "platesolve_backend": "vyvar",
            "plate_solve_fov_deg": plate_fov,
            "max_extra_platesolve": int(params.max_extra_platesolve),
            "catalog_match_max_sep_arcsec": float(params.catalog_match_max_sep_arcsec),
            "saturate_level_fraction": float(params.saturate_level_fraction),
            "max_catalog_rows": int(params.max_catalog_rows),
            "n_comparison_stars": 0,
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
        result.processing_hash = ph
        result.masterstar_path = str(use_path or "")
        stamp_frame_qc_provenance(
            ap_root,
            draft_id=draft_id,
            fwhm_limit_px=float(fwhm_lim),
            fwhm_limit_source=fwhm_limit_source,
            cfg_source=result.cfg_source,
            cfg_changed_keys=result.cfg_changed_keys,
        )

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
        result.platesolve_job_output = ps_out

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

        # Steps 13-15: Photometry (shared slice)
        _p("Step 13: Discover obs groups")
        phot = run_night_photometry(
            cfg=cfg,
            pipeline=pipeline,
            draft_id=draft_id,
            draft_dir_override=None,
            progress_cb=params.progress_cb,
            timings=timings,
            write_pdfs=True,
            existing_draft=False,
            epsf=params.epsf,
        )
        result.cfg_source = str(phot.get("cfg_source") or "live")
        result.cfg_changed_keys = list(phot.get("cfg_changed_keys") or [])
        result.photometry_context = dict(phot.get("photometry_context") or {})
        result.zero_target_setups = list(phot.get("zero_target_setups") or [])
        result.completed_setups = list(phot.get("completed_setups") or [])
        result.photometry_completeness = dict(phot.get("photometry_completeness") or {})
        result.epsf_stage = dict(phot.get("epsf_stage") or {})
        result.warnings.extend(list(phot.get("warnings") or []))
        if phot.get("output_dir") is not None:
            result.output_dir = Path(phot["output_dir"])
        try:
            result.sysrem_improvement_pct = float(
                phot.get("sysrem_improvement_pct", float("nan"))
            )
        except (TypeError, ValueError):
            pass

        phot_errors = list(phot.get("errors") or [])
        if phot_errors:
            result.errors.extend(phot_errors)
            result.phase_timings = timings
            return result

        result.n_lightcurves = int(phot.get("n_lightcurves") or 0)
        result.n_frames = int(phot.get("n_frames") or 0)
        try:
            result.lc_rms_median = float(phot.get("lc_rms_median", float("nan")))
        except (TypeError, ValueError):
            pass

        result.success = True
        timings["total"] = time.time() - t_run
        result.phase_timings = timings
        if result.draft_dir is not None:
            from infolog import write_run_infolog  # noqa: PLC0415

            saved = write_run_infolog(result.draft_dir)
            if saved:
                _p(f"Infolog saved: {Path(saved).name}")
        _p(
            f"Night run complete - draft {draft_id}, {result.n_lightcurves} light curve(s), "
            f"{result.n_frames} frame(s)"
        )
        return result

    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("[NightRun] FATAL: %s", exc)
        result.errors.append(str(exc))
        timings["total"] = time.time() - t_run
        result.phase_timings = timings
        return result


def main(argv: list[str] | None = None) -> int:
    """CLI entry: same three required inputs as UI RUN VYVAR."""
    args = parse_night_run_cli(argv)
    cfg = _load_app_config(args.config)
    eq, tel, loc, missing = resolve_night_run_cli_ids(
        equipment_id=args.equipment_id,
        telescope_id=args.telescope_id,
        location_id=args.location_id,
        draft_dir=args.draft_dir,
        cfg=cfg,
    )
    if missing:
        print(night_run_missing_message(missing), file=sys.stderr)
        return 2
    params = NightRunParams(
        source_dir=Path(args.source),
        equipment_id=int(eq),
        telescope_id=int(tel),
        location_id=int(loc),
        location_source_hint="cli_arg",
        config_path=args.config,
        dry_run=bool(args.dry_run),
        progress_cb=lambda msg: LOGGER.info("[Progress] %s", msg),
    )
    result = run_night_pipeline(params)
    if result.success:
        return 0
    for err in result.errors:
        print(f"ERROR: {err}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

