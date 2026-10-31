"""Headless VYVAR night pipeline runner.

Production entry: ``run_night_pipeline``. UI RUN VYVAR
(``app._run_vyvar_full_pipeline``) builds the same ``NightRunParams`` as the CLI
(source, camera, telescope, site, mode), checks ``check_run_preconditions`` and
starts ``python -m run_job --job`` (run_job), so the GUI and the night runner run
the same code (INV-ONE-ENTRY-01). Everything else comes from the run's config.

No Streamlit dependencies. Progress via ``logging`` and optional callback.
"""

from __future__ import annotations

import astro_env  # noqa: F401  (IERS-OFF-2026-10: no IERS downloads)

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

from gaia_catalog_id import read_vyvar_csv

from config import AppConfig, resolve_detection_threshold_sigma
from importer import smart_import_session, smart_scan_source
from pipeline import AstroPipeline

LOGGER = logging.getLogger(__name__)

_DEFAULT_MASTERSTAR_SELECTION_PCT = 10.0


@dataclass
class NightRunParams:
    """Input parameters for a headless night pipeline run.

    G12-10 / G12-11 / G14-07 (Milan 2026-10-31): only the run inputs the GUI and the CLI both give;
    validity windows, the DAO threshold and the optics come from the run's config and DB, so the GUI
    and the CLI build identical params for the same inputs."""

    source_dir: Path
    equipment_id: int
    telescope_id: int
    config_path: Path | None = None
    location_id: int | None = None
    location_source_hint: str | None = None
    # Draft rig.location_id when resolving without an explicit UI/CLI site
    # (precedence: explicit location_id > manifest_location_id > config).
    manifest_location_id: int | None = None
    progress_cb: Callable[[str], None] | None = None
    dry_run: bool = False
    pre_calibrated_mode: bool = False
    #: CAL-2026-10: masters next to the LIGHTs ("lights", default) or from the
    #: Calibration Library ("library") are preferred; the other source is the fallback.
    #: LIGHT-folder masters that are used get imported into the library.
    masters_from: str = "lights"
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
    cfg_source: str = "live"
    cfg_changed_keys: list[str] = field(default_factory=list)
    zero_target_setups: list[str] = field(default_factory=list)
    completed_setups: list[str] = field(default_factory=list)
    photometry_context: dict[str, Any] = field(default_factory=dict)
    calibration_output: dict[str, Any] = field(default_factory=dict)
    epsf_stage: dict[str, Any] = field(default_factory=dict)
    #: frame_qc per observation group (limits, rejections, best frame).
    frame_qc: dict[str, Any] = field(default_factory=dict)
    #: G12-15: the run stopped on ``check_run_preconditions`` (nothing was imported).
    preconditions_failed: bool = False
    #: N5: ``logs/run_preflight_error_<ts>.log`` of a run that failed before its infolog opened.
    failure_log: str | None = None


# Aperture photometry must cover at least this fraction of active_targets in photometry_summary.
_PHOTOMETRY_COMPLETENESS_MIN_RATIO = 0.90

_NIGHT_RUN_INPUT_CAMERA = "camera"
_NIGHT_RUN_INPUT_TELESCOPE = "telescope"
_NIGHT_RUN_INPUT_SITE = "observing site"


def resolve_epsf_run(epsf: bool | None, cfg: Any | None = None) -> bool:
    """Whether to run the ePSF night-run stage.

    Explicit True/False wins. None (or omitted) reads ``cfg.epsf_auto_run``
    (default False). The UI ePSF button calls ``run_epsf_stage`` directly and
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
) -> tuple[int | None, int | None, int | None, list[str], str | None]:
    """Resolve camera / telescope / observing site the way W1 does.

    Explicit CLI/UI ids win. Else draft manifest ``rig`` (equipment_id,
    telescope_id, location_id). Else the default site (LOCATION.IS_DEFAULT,
    ``cfg.observer_location_id``) for site only. Returns ``(eq, tel, loc, missing, loc_source)``
    where ``loc_source`` is ``cli_arg`` / ``manifest`` / ``db_default`` / None.
    """
    eq = _positive_id(equipment_id)
    tel = _positive_id(telescope_id)
    loc_explicit = _positive_id(location_id)
    loc = loc_explicit
    loc_source: str | None = "cli_arg" if loc_explicit is not None else None
    m_loc: int | None = None
    if draft_dir is not None:
        from draft_provenance import _manifest_rig_pair, _optional_int, load_draft_manifest

        manifest = load_draft_manifest(Path(draft_dir))
        if manifest:
            m_eq, m_tel = _manifest_rig_pair(manifest)
            rig = manifest.get("rig") if isinstance(manifest.get("rig"), dict) else {}
            m_loc = _positive_id(_optional_int(rig.get("location_id")))
            if eq is None:
                eq = _positive_id(m_eq)
            if tel is None:
                tel = _positive_id(m_tel)
            if loc is None and m_loc is not None:
                loc = m_loc
                loc_source = "manifest"
    if loc is None and cfg is not None:
        loc = _positive_id(getattr(cfg, "observer_location_id", 0))
        if loc is not None:
            loc_source = "db_default"
    missing: list[str] = []
    if eq is None:
        missing.append(_NIGHT_RUN_INPUT_CAMERA)
    if tel is None:
        missing.append(_NIGHT_RUN_INPUT_TELESCOPE)
    if loc is None:
        missing.append(_NIGHT_RUN_INPUT_SITE)
        loc_source = None
    return eq, tel, loc, missing, loc_source


def night_run_missing_message(missing: list[str]) -> str:
    named = ", ".join(missing)
    return (
        f"Night run refused: missing {named}. "
        "Provide --camera/--eq, --telescope/--tel, and --site/--location "
        "(or a draft manifest that resolves them; the site may also be the default "
        "site, the default in Database Explorer > Sites)."
    )


def parse_night_run_cli(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "VYVAR night run. Requires the same three inputs as UI RUN VYVAR: "
            "telescope, camera, observing site."
        )
    )
    _src = parser.add_mutually_exclusive_group(required=True)
    _src.add_argument("--source", default=None, help="Source directory with FITS files")
    _src.add_argument(
        "--job",
        type=Path,
        default=None,
        help="Run a job spec written by the GUI (job.json); all other inputs come from the spec",
    )
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
        help="Observing site LOCATION id (else the default site, LOCATION.IS_DEFAULT)",
    )
    parser.add_argument(
        "--draft-dir",
        type=Path,
        default=None,
        help="Optional draft directory whose manifest can supply the three ids",
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to config.json")
    parser.add_argument("--dry-run", action="store_true", help="Scan only; no pipeline")
    parser.add_argument(
        "--pre-calibrated",
        action="store_true",
        help="LIGHTs are used as they are (calibrated by another program, or deliberately uncalibrated)",
    )
    parser.add_argument(
        "--masters-from",
        choices=("lights", "library"),
        default="lights",
        help="Preferred source of master DARK/FLAT: next to the LIGHTs (default) or the Calibration Library",
    )
    return parser.parse_args(argv)


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


def log_frame_qc_provenance(
    *,
    draft_id: int,
    fwhm_limit_px: float,
    fwhm_limit_source: str,
    cfg_source: str = "live",
    cfg_changed_keys: list[str] | None = None,
) -> dict[str, Any]:
    """Log the FRAME-QC + cfg-source stamp (infolog); no file is written."""
    payload = {
        "quality_filter_draft_id": int(draft_id),
        "fwhm_limit_px": float(fwhm_limit_px),
        "fwhm_limit_source": str(fwhm_limit_source),
        "cfg_source": str(cfg_source),
        "cfg_changed_keys": list(cfg_changed_keys or []),
    }
    LOGGER.info("[NightRun] FRAME-QC provenance: %s", payload)
    return payload


def resolve_photometry_context_triple(
    cfg: AppConfig,
    *,
    db: Any,
    draft_id: int | None,
    masterstar_fits: Path | None,
) -> dict[str, Any]:
    """Aperture-photometry plate scale / site / calibration_mode (C3 fire-proof)."""
    from draft_provenance import resolve_calibration_mode
    from param_resolver import resolve_site
    from optics import frame_plate_scale

    hdr = None
    ms = Path(masterstar_fits) if masterstar_fits is not None else None
    if ms is not None and ms.is_file():
        try:
            from astropy.io import fits as astrofits

            with astrofits.open(ms, memmap=False) as hdul:
                hdr = hdul[0].header.copy()
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("MASTERSTAR header for context triple skipped: %s", exc)
    plate = frame_plate_scale(ms, header=hdr, db=db, draft_id=draft_id)
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
    """The run config: the install's config.json, or exactly ``config_path`` (``--config``, G12-03)."""
    return AppConfig() if config_path is None else AppConfig.from_file(config_path)


def gaia_db_problem(cfg: Any) -> str | None:
    """Why the Gaia DR3 database of ``cfg`` cannot be used for MASTERSTAR (None: usable). The one
    check of the GUI RUN button and the night run (G12-15)."""
    from catalog_db import validate_gaia_db_schema  # noqa: PLC0415

    path = str(getattr(cfg, "gaia_db_path", "") or "").strip()
    try:
        ok, msg = validate_gaia_db_schema(path)
    except Exception as exc:  # noqa: BLE001 - reported as the problem
        ok, msg = False, f"{type(exc).__name__}: {exc}"
    return None if ok else f"Gaia DR3 database not usable ({path or 'no path set'}): {msg}"


@dataclass
class RunPreconditions:
    """Result of :func:`check_run_preconditions`: the problems (empty = the run may start) and the
    observing site resolved for the run (G12-16: resolved once, the run uses this one)."""

    problems: list[str]
    site: Any | None = None


def check_run_preconditions(cfg: AppConfig, params: NightRunParams, db: Any) -> RunPreconditions:
    """G12-15 / G12-16: what RUN VYVAR needs before anything is imported - the source folder, the
    camera and telescope rows, the observing site and (not for a dry run) the Gaia DR3 database. The
    GUI runs it before it starts the job, ``run_night_pipeline`` before the scan (CLI exit code 2).
    ``cfg`` is not changed."""
    from observer_location import resolve_observer_location_for_run  # noqa: PLC0415

    problems: list[str] = []
    src = Path(params.source_dir).expanduser()
    if not src.is_dir():
        problems.append(f"Invalid source directory: {src}")
    for table, label, value in (
        ("EQUIPMENTS", _NIGHT_RUN_INPUT_CAMERA, params.equipment_id),
        ("TELESCOPE", _NIGHT_RUN_INPUT_TELESCOPE, params.telescope_id),
    ):
        rid = _positive_id(value)
        if rid is None or not db._fk_row_exists(table, rid):
            problems.append(f"{label} id={value} not found in {table}")
    hint = params.location_source_hint
    if hint is None and params.location_id is not None:
        hint = "cli_arg"
    site = None
    try:
        site = resolve_observer_location_for_run(
            cfg.database_path,
            explicit_location_id=params.location_id,
            manifest_location_id=params.manifest_location_id,
            cfg=cfg,
            source_hint=hint,
        )
    except ValueError as exc:
        problems.append(str(exc))
    if not params.dry_run:
        gaia = gaia_db_problem(cfg)
        if gaia:
            problems.append(gaia)
    return RunPreconditions(problems=problems, site=site)


def _write_failure_log(cfg: Any | None, *, step: str, exc: BaseException, db: Any = None) -> str | None:
    """N5: ``<data_dir>/logs/run_preflight_error_<ts>.log`` for a run that failed before its infolog
    opened (the GUI writes the same file for its own early failures). Returns the path."""
    from config import resolve_data_root  # noqa: PLC0415
    from run_preflight_log import write_run_preflight_error_log  # noqa: PLC0415

    try:
        data_root = getattr(cfg, "data_root", None) or resolve_data_root(Path(__file__).resolve().parent.parent)
        return str(write_run_preflight_error_log(data_root, step=step, exc=exc, db=db, cfg=cfg))
    except Exception as log_exc:  # noqa: BLE001 - the run's own error is already in the result
        LOGGER.warning("[NightRun] preflight / failure log not written: %s", log_exc)
        return None


def _masterstar_candidate_path_for_job(
    archive: Path | None,
    path_any: str,
    *,
    draft_id: int | None = None,
    db: Any = None,
) -> str:
    """Prefer draft lights FITS for MASTERSTAR job input (processed or pre-cal non_calibrated)."""
    from pipeline_preprocess import resolve_obs_file_to_processed_fits

    def _has_raw_seg(pp: Path) -> bool:
        try:
            parts = pp.resolve().parts
        except OSError:
            parts = pp.parts
        return any(seg.casefold() == "raw" for seg in parts)

    p = (path_any or "").strip()
    if not p:
        return ""
    if archive is None or not archive.is_dir():
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


class _RunStages:
    """G12-34: the one status / stage / timing helper of a run (``run_night_pipeline`` and the
    photometry slice it calls): a stage is started once, every timing writes ``[PHASE] <label> done``."""

    def __init__(self, timings: dict[str, float], progress_cb: Callable[[str], None] | None) -> None:
        self.timings = timings
        self._cb = progress_cb
        self.current = "start"

    def progress(self, msg: str) -> None:
        LOGGER.info("[NightRun] %s", msg)
        if self._cb is not None:
            self._cb(msg)

    def stage(self, name: str) -> None:
        """Status line + infolog milestone for one named pipeline process."""
        from infolog import log_phase_boundary  # noqa: PLC0415

        self.current = name
        self.progress(name)
        log_phase_boundary(name, status="start")

    def done(self, label: str, t0: float) -> None:
        from infolog import log_milestone  # noqa: PLC0415

        elapsed = time.time() - t0
        self.timings[label] = elapsed
        LOGGER.info("[NightRun] [OK] %s - %.1fs", label, elapsed)
        log_milestone(f"[PHASE] {label} done {elapsed:.1f}s")


def _night_run_preprocess(
    *,
    pending: dict[str, Any],
    ap: Path,
    pipeline: AstroPipeline,
    progress_cb: Callable[[int, int, str], None],
) -> None:
    """Preprocess (calibrated -> processed) for RUN VYVAR (GUI and CLI)."""
    from draft_provenance import resolve_draft_lights_root
    from pipeline_preprocess import (
        build_prefilter_rejected_map,
        calibrated_paths_for_draft_apply_filters,
        qc_enrich_calibrated_lights_in_place,
    )
    from pipeline_calibrate import _iter_light_fits

    _app_cfg = pipeline.config
    ap_root = ap.parent if ap.name.casefold() == "non_calibrated" else ap
    _dqf = pending.get("quality_filter_draft_id")
    if _dqf is None:
        raise RuntimeError("INV-FRAME-QC-01: quality_filter_draft_id is required")
    source_dir = resolve_draft_lights_root(ap_root, draft_id=int(_dqf), db=pipeline.db)
    if not source_dir.exists():
        raise FileNotFoundError("Missing source lights directory for preprocess filter.")

    p1 = calibrated_paths_for_draft_apply_filters(ap_root, pipeline.db, int(_dqf), source_dir=source_dir)
    if not p1:
        raise FileNotFoundError("QC filter: no frames matching IS_REJECTED=0.")
    _all_lights = _iter_light_fits(source_dir)
    _n_all = max(len(_all_lights), 1)
    qc_enrich_calibrated_lights_in_place(
        calibrated_root=source_dir,
        only_paths=None,
        prefilter_rejected=build_prefilter_rejected_map(_all_lights, p1),
        progress_cb=lambda i, _t, msg: progress_cb(i, _n_all, msg),
        db=pipeline.db,
        draft_id=int(_dqf),
        inject_pointing_ra_deg=pending.get("inject_pointing_ra_deg"),
        inject_pointing_dec_deg=pending.get("inject_pointing_dec_deg"),
        inject_pointing_only_if_missing=False,
        app_config=_app_cfg,
    )

    _ph_done = pending.get("processing_hash")
    _dqf_done = pending.get("quality_filter_draft_id")
    if _dqf_done is not None and _ph_done:
        try:
            pipeline.db.record_qc_processing_apply(
                int(_dqf_done),
                str(_ph_done),
            )
            pipeline.db.update_obs_draft_status(int(_dqf_done), "PROCESSED")
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("QC snapshot / draft status write failed: %s", exc)


def _night_run_platesolve(
    *,
    pending: dict[str, Any],
    ap: Path,
    pipeline: AstroPipeline,
    progress_cb: Callable[[int, int, str], None],
) -> dict[str, Any]:
    """Plate solve + alignment + MASTERSTAR for RUN VYVAR (GUI and CLI)."""
    from astrometry_orchestration import astrometry_align_and_build_masterstar

    # The run config already holds the resolved DAO sigma (run_night_pipeline, the one place it is set).
    cfg_run = pipeline.config

    _ms_pct = pending.get("masterstar_selection_pct")
    try:
        ms_pct_f = float(_ms_pct) if _ms_pct is not None else _DEFAULT_MASTERSTAR_SELECTION_PCT
    except (TypeError, ValueError):
        ms_pct_f = _DEFAULT_MASTERSTAR_SELECTION_PCT

    _peq = pending.get("id_equipment")
    _draft_ps = pending.get("draft_id")

    return astrometry_align_and_build_masterstar(
        archive_path=ap,
        app_config=cfg_run,
        n_comparison_stars=int(pending.get("n_comparison_stars", 0)),
        id_equipment=int(_peq) if _peq is not None else None,
        draft_id=int(_draft_ps) if _draft_ps is not None else None,
        build_masterstar_and_catalogs=bool(pending.get("build_masterstar_and_catalogs", True)),
        ram_align_and_catalog=True,
        progress_cb=progress_cb,
        masterstar_candidate_paths=list(pending.get("masterstar_candidate_paths") or []),
        masterstar_selection_pct=ms_pct_f,
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
        # FIX-CID-DTYPE-2026-10: Gaia ids as text (a NaN in the column turned them into floats)
        at_df = pd.read_csv(active_csv, low_memory=False, dtype={"catalog_id": str})
        sm_df = pd.read_csv(summary_csv, low_memory=False, dtype={"catalog_id": str})
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
    def _strip_id_series(s: pd.Series) -> pd.Series:
        return s.astype(str).str.strip()

    n_measurable_missing = n_active - n_summary  # conservative default (treat all as measurable)
    achieved_depth = float("nan")
    if "catalog_id" in at_df.columns and "catalog_id" in sm_df.columns:
        measured_ids = set(_strip_id_series(sm_df["catalog_id"]).tolist())
        at_ids = _strip_id_series(at_df["catalog_id"])
        is_measured = at_ids.isin(measured_ids)
        at_mag = pd.to_numeric(at_df.get("mag"), errors="coerce") if "mag" in at_df.columns else None
        if at_mag is not None:
            measured_mag = at_mag[is_measured].dropna()
            if not measured_mag.empty:
                achieved_depth = float(measured_mag.max())
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
        from report_methods import lc_csv_path  # noqa: PLC0415
        psf_path = lc_csv_path(lc_dir, cid, "psf")
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
        df = read_vyvar_csv(summary, low_memory=False)
    except Exception as exc:  # noqa: BLE001
        logging.warning('[EXC-0113] intent unclear (try: / df = pd.read_csv(summary, low_memory=False) / except Exception: ...: %s', exc)
        return 0, float("nan")
    # FIX-NLC-2026-10: rows of skipped / no-comparison targets (n_frames 0) are not light curves
    if "n_frames" in df.columns:
        df = df[pd.to_numeric(df["n_frames"], errors="coerce").fillna(0) > 0]
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
    stages: _RunStages | None = None,
) -> dict[str, Any]:
    """Photometry-only slice shared by ``run_night_pipeline`` (C1/C2) and C3.

    Forwards ``draft_id`` and ``db``. Completeness gate is shared: a UI run
    can FAIL loudly. INV-CFG-SOURCE-01 applies when ``existing_draft`` is True.
    """
    from photometry_provenance import merge_photometry_pipeline_meta
    from photometry_core import run_full_photometry_pipeline
    from platesolve_ui_paths import find_photometry_setups

    timings = timings if timings is not None else {}
    out: dict[str, Any] = {
        "errors": [],
        "warnings": [],
        "n_lightcurves": 0,
        "n_frames": 0,
        "lc_rms_median": float("nan"),
        "output_dir": None,
        "photometry_completeness": {},
        "zero_target_setups": [],
        "completed_setups": [],
        "photometry_context": {},
        "cfg_source": "live",
        "cfg_changed_keys": [],
        "epsf_stage": {},
    }

    if stages is None:
        stages = _RunStages(timings, progress_cb)
    _p, _stage, _t = stages.progress, stages.stage, stages.done

    draft_dir = None
    if draft_dir_override is not None:
        draft_dir = Path(draft_dir_override).resolve()
    elif draft_id is not None:
        from draft_provenance import draft_dir_for_id  # noqa: PLC0415
        draft_dir = draft_dir_for_id(cfg.archive_root, int(draft_id)).resolve()

    phot_cfg, cfg_source, changed_keys = resolve_cfg_for_photometry(
        cfg,
        draft_dir,
        existing_draft=bool(existing_draft),
    )
    out["cfg_source"] = cfg_source
    out["cfg_changed_keys"] = list(changed_keys)
    if changed_keys:
        _p(f"INV-CFG-SOURCE-01 changed_keys={changed_keys}")

    _stage("Discover observation groups")
    t0 = time.time()
    all_setups = find_photometry_setups(phot_cfg, draft_id, draft_dir_override=draft_dir)
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
    last_out: Path | None = None

    for nm in run_groups:
        p = all_setups.get(str(nm)) or {}
        ms_fits = Path(p["masterstar_fits"]) if p.get("masterstar_fits") else None
        og_dir = Path(p["obs_group_dir"]) if p.get("obs_group_dir") else None
        ms_csv = (og_dir / "masterstars_full_match.csv") if og_dir is not None else None
        vt_csv = (og_dir / "variable_targets.csv") if og_dir is not None else None
        pf_dir = Path(p["per_frame_csv_dir"]) if p.get("per_frame_csv_dir") else None
        dt_dir = pf_dir  # G03-34: the per-frame CSVs live next to the aligned frames (one key)
        out_d = Path(p["photometry_dir"]) if p.get("photometry_dir") else None

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

        _stage(f"Photometry - {nm}")
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

        p2a = phot_result.get("aperture_photometry") or {}
        total_lc += int(p2a.get("n_lightcurves") or 0)
        total_frames += int(p2a.get("n_frames") or 0)
        last_out = out_d
        completed.append(str(nm))

        n_sum, med_rms = _collect_photometry_metrics(out_d)
        if n_sum > 0:
            total_lc = max(total_lc, n_sum)
        if math.isfinite(med_rms):
            lc_rms_values.append(med_rms)

        do_epsf = resolve_epsf_run(epsf, phot_cfg)
        if do_epsf:
            from epsf_stage import EpsfStagePaths, run_epsf_stage

            _stage(f"ePSF photometry - {nm}")
            t0 = time.time()
            try:
                epsf_out = run_epsf_stage(
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
                from photometry_report import generate_photometry_report

                pdf_path = generate_photometry_report(
                    draft_dir, str(nm), None, tess_results={}, report_title="VYVAR - Summary Measure Report",
                    cfg=phot_cfg,
                )
                if pdf_path is not None:
                    _p(f"PDF report: {Path(pdf_path).name}")
            except Exception as pdf_err:  # noqa: BLE001
                out["warnings"].append(f"PDF report {nm}: {pdf_err}")

    out["photometry_completeness"] = completeness_by_setup
    out["zero_target_setups"] = zero_target_setups
    out["completed_setups"] = completed
    out["output_dir"] = last_out
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


def _nr_scan_source(*, _t, cfg, eq_id, params, pipeline, source, tel_id):
    """Smart scan of the source folder (rig ids, light groups) and the scan facts in the result."""
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
        precalibrated=bool(params.pre_calibrated_mode),
        masters_from=str(params.masters_from or "lights"),
        import_source_masters=True,
    )
    _t("smart_scan_source", t0)

    lights_bad = any(
        r.type == "Lights" and r.status in ("missing", "empty") for r in plan.scan_rows
    )
    return lights_bad, plan


def _nr_import(*, _p, _stage, _t, cfg, eq_id, params, pipeline, plan, tel_id, site):
    """Pre-calibrated / calibrated import plan, observer site and the import session into a new draft.
    ``site``: the site resolved by ``check_run_preconditions`` (G12-16: resolved once)."""
    from parallel_frames import check_disk_for_run  # noqa: PLC0415

    # DISK-PREFLIGHT-2026-10: fail before the import, not in the middle of the night
    _p(check_disk_for_run(cfg.archive_root, list(plan.lights_files)))

    if params.pre_calibrated_mode:
        from draft_provenance import (
            CALIBRATION_MODE_PRE,
            apply_pre_calibrated_import_plan,
            calibration_mode_report_line,
        )

        apply_pre_calibrated_import_plan(plan)
        _p(calibration_mode_report_line(CALIBRATION_MODE_PRE))

    # Import - the observing site of this run (product rule SITE-UI-01) goes into the run's own cfg.
    _stage("Import session")
    from observer_location import apply_resolved_observer_location_to_config  # noqa: PLC0415

    _resolved_site = site
    apply_resolved_observer_location_to_config(cfg, _resolved_site)
    _p(_resolved_site.milestone_line())
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
    return _resolved_site, import_result


def _nr_calibrate(
    *,
    _p,
    _resolved_site,
    _stage,
    _t,
    eq_id,
    import_result,
    params,
    pipeline,
    plan,
    prog_cb,
    result,
    timings,
):
    """Draft provenance and calibration (VYVAR masters or pre-calibrated lights) of the imported draft."""
    draft_id = int(import_result.draft_id)
    result.draft_id = draft_id
    result.import_result = import_result
    result.plan = plan
    ap = Path(str(import_result.archive_path))
    ap_root = ap.parent if ap.name.casefold() == "non_calibrated" else ap
    result.draft_dir = ap_root.resolve()
    result.archive_path = ap

    from infolog import log_milestone, start_infolog_session  # noqa: PLC0415

    start_infolog_session(result.draft_dir)
    log_milestone(_resolved_site.milestone_line())
    from parallel_frames import resource_budget  # noqa: PLC0415

    log_milestone(f"[RESOURCES] {resource_budget().describe()}")

    from draft_provenance import (
        CALIBRATION_MODE_PRE,
        CALIBRATION_MODE_VYVAR,
        record_draft_calibration_provenance,
        record_observer_location_provenance,
    )

    record_draft_calibration_provenance(
        archive_path=ap_root,
        draft_id=draft_id,
        calibration_mode=CALIBRATION_MODE_PRE if params.pre_calibrated_mode else CALIBRATION_MODE_VYVAR,
    )
    record_observer_location_provenance(
        archive_path=ap_root,
        draft_id=draft_id,
        resolved=_resolved_site,
    )
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
        mf_map = {}
        mf_obs = {}
        dm_obs = {}
        _p(
            "Calibration skipped - pre-calibrated import; downstream reads non_calibrated/lights "
            "(no calibrated/ directory)"
        )
        _cal_out: dict[str, Any] = {"pre_calibrated_skip_calibration": True}
        timings["calibration"] = 0.0
    else:
        _stage("Calibration")
        t0 = time.time()
        _cal_out = pipeline.quick_calibrate_last_import(
            archive_path=ap,
            master_dark_path=None,
            masterflat_by_filter=mf_map,
            progress_cb=prog_cb,
            equipment_id=eq_id,
            draft_id=draft_id,
            observation_id=getattr(import_result, "observation_id", None),
            masterflat_by_obs_key=mf_obs or None,
            master_dark_by_obs_key=dm_obs or None,
            flat_substitute_by_obs_key=dict(getattr(plan, "flat_substitute_by_obs_key", None) or {}),
        )
        _t("calibration", t0)
    result.calibration_output = dict(_cal_out or {})

    from draft_provenance import resolve_draft_lights_root

    lights_root = resolve_draft_lights_root(ap_root, draft_id=draft_id, db=pipeline.db)
    return _cal_out, ap, ap_root, dm_obs, draft_id, lights_root, mf_map, mf_obs


def _ui_pointing(qsum: Any, lights_root: Path) -> tuple[dict[str, Any], float | None, float | None]:
    """Field pointing for the UI: the RAM-QC medians, else the first calibrated light with display
    coordinates (``scan_calibrated_lights_pointing``). Returns (scan result or {}, ra, dec)."""
    from pipeline_calibrate import scan_calibrated_lights_pointing  # noqa: PLC0415

    ra_q: float | None = None
    de_q: float | None = None
    try:
        mra_q = qsum.get("median_ra_deg")
        mde_q = qsum.get("median_de_deg")
        if mra_q is not None and math.isfinite(float(mra_q)):
            ra_q = float(mra_q)
        if mde_q is not None and math.isfinite(float(mde_q)):
            de_q = float(mde_q)
    except (TypeError, ValueError):
        pass
    # G12-38: the header pointing scan (every calibrated light, header + WCS) is needed only when
    # the RAM-QC medians do not give both coordinates; its rows were not read otherwise.
    pointing: dict[str, Any] = {}
    ra_ui: float | None = ra_q
    de_ui: float | None = de_q
    if ra_q is None or de_q is None:
        pointing = scan_calibrated_lights_pointing(lights_root, max_files=None)
        r_pref = next(
            (
                r
                for r in pointing["rows"]
                if r.get("display_ra_deg") is not None and r.get("display_dec_deg") is not None
            ),
            None,
        )
        if r_pref:
            ra_ui = ra_q if ra_q is not None else float(r_pref["display_ra_deg"])
            de_ui = de_q if de_q is not None else float(r_pref["display_dec_deg"])
    return pointing, ra_ui, de_ui


def _nr_frame_qc(
    *,
    _cal_out,
    _stage,
    _t,
    ap_root,
    cfg,
    dm_obs,
    draft_id,
    eq_id,
    lights_root,
    mf_map,
    mf_obs,
    params,
    pipeline,
    prog_cb,
    result,
    timings,
):
    """Frame QC of the calibrated lights (RAM QC, per-group best frame, rejections) and the QC provenance."""
    qsum: dict[str, Any] = dict(_cal_out.get("perf10_qsum") or {})

    from pipeline_calibrate import run_draft_ram_calibration_qc_to_obs_files


    # RAM QC (skipped when PERF-10: DAO QC already done in calibration)
    _perf10_ok = (
        not params.pre_calibrated_mode
        and int(qsum.get("n_successful_fwhm") or 0) > 0
    )
    if not _perf10_ok:
        _stage("Frame quality check (RAM QC)")
        t0 = time.time()
        qsum = run_draft_ram_calibration_qc_to_obs_files(
            db=pipeline.db,
            draft_id=draft_id,
            archive_path=ap_root,
            master_dark_path=None,
            masterflat_by_filter=mf_map,
            masterflat_by_obs_key=mf_obs or None,
            master_dark_by_obs_key=dm_obs or None,
            equipment_id=eq_id,
            progress_cb=prog_cb,
            pedestal_temp_tol_c=float(pipeline.config.calibration_master_ccd_temp_tolerance_c),
            flat_substitute_by_obs_key=dict(getattr(result.plan, "flat_substitute_by_obs_key", None) or {}),
        )
        _t("ram_qc", t0)
    else:
        LOGGER.info("[PERF-10] RAM QC skipped - DAO QC computed during calibration")
        timings["ram_qc"] = 0.0
    result.ram_qc_summary = dict(qsum or {})

    # Pointing
    t0 = time.time()
    pointing, ra_ui, de_ui = _ui_pointing(qsum, lights_root)
    result.pointing = pointing if isinstance(pointing, dict) else {}
    result.ra_ui = ra_ui
    result.de_ui = de_ui
    _t("pointing_scan", t0)

    # Frame QC per channel (frame_qc.py, QC-PER-GROUP-2026-10): per observation
    # group FWHM / elongation limits (median + k*1.4826*MAD, one pass), rejections written
    # to the manifest (IS_REJECTED + reason) and frame_qc.json, best frame per group.
    _stage("Frame QC per channel")
    t0 = time.time()
    from frame_qc import run_frame_qc

    _qc_groups = run_frame_qc(
        pipeline.db,
        draft_id,
        ap_root,
        k=float(cfg.auto_fwhm_k_factor),
        reject=bool(cfg.auto_fwhm_enabled),
    )
    _t("frame_qc", t0)
    # Rejections are in IS_REJECTED; preprocess applies no extra pooled FWHM cap.
    fwhm_lim = 0.0
    fwhm_limit_source = (
        f"frame_qc per group (k={float(cfg.auto_fwhm_k_factor):g})"
        if bool(cfg.auto_fwhm_enabled)
        else "frame_qc ranking only (auto_fwhm_enabled=False)"
    )
    result.fwhm_limit_px = float(fwhm_lim)
    result.fwhm_limit_source = fwhm_limit_source
    result.frame_qc = {gk: g.to_dict() for gk, g in _qc_groups.items()}

    # MASTERSTAR source = best frame of each group (frame_qc.json). A single-group
    # draft also stores it as the draft MASTERSTAR source path (read by MASTERSTAR QA).
    _stage("MASTERSTAR reference frame per channel")
    t0 = time.time()
    from pipeline_astrometry import draft_is_multi_group_obs

    _multi_obs = draft_is_multi_group_obs(ap_root)
    use_path = ""
    return _multi_obs, _qc_groups, de_ui, fwhm_lim, fwhm_limit_source, ra_ui, t0, use_path


def _nr_preprocess_and_platesolve(
    *,
    _multi_obs,
    _qc_groups,
    _stage,
    _t,
    ap,
    ap_root,
    cfg,
    de_ui,
    draft_id,
    fwhm_lim,
    fwhm_limit_source,
    params,
    pipeline,
    prog_cb,
    ps_eq,
    ra_ui,
    result,
    t0,
    use_path,
):
    """Reference coordinates, preprocess (QC enrich) and the plate solve / alignment / MASTERSTAR job."""
    from fits_meta import generate_observation_hash  # noqa: PLC0415
    from pipeline_preprocess import resolve_preprocess_target_coordinates  # noqa: PLC0415

    if not _multi_obs and len(_qc_groups) == 1:
        top_path = str(next(iter(_qc_groups.values())).best_path or "").strip()
        use_path = (
            _masterstar_candidate_path_for_job(
                ap_root, top_path, draft_id=draft_id, db=pipeline.db
            )
            or top_path
        ).strip()
    try:
        pipeline.db.set_obs_draft_masterstar_source_path(draft_id, use_path or None)
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
    except Exception as exc:  # noqa: BLE001
        ph = None  # G12-22: no QC apply record / PROCESSED status for this run - said, not silent
        LOGGER.warning("observation hash failed for draft %s (QC apply not recorded, status stays): %s",
                       draft_id, exc)
        from infolog import log_event  # noqa: PLC0415

        log_event(f"WARNING: observation hash failed for draft {draft_id}: {exc} "
                  "(QC apply not recorded, field status not set to PROCESSED)")
    _t("preprocess_prep", t0)

    dao_sigma = resolve_detection_threshold_sigma(cfg.detection_threshold_sigma)

    pipeline.config.detection_threshold_sigma = dao_sigma

    job_ms: dict[str, Any] = {
        "kind": "make_masterstar",
        "archive_path": str(ap),
        "inject_pointing_ra_deg": (float(ira) if coords_ok else None),
        "inject_pointing_dec_deg": (float(ide) if coords_ok else None),
        "quality_filter_draft_id": draft_id,
        "n_comparison_stars": 0,
        "dao_threshold_sigma": dao_sigma,
        "id_equipment": ps_eq,
        "draft_id": draft_id,
        "build_masterstar_and_catalogs": True,
        "masterstar_candidate_paths": [use_path] if use_path else [],
    }
    if ph:
        job_ms["processing_hash"] = ph
        job_ms["overwrite_qc_processing"] = True
    result.processing_hash = ph
    result.masterstar_path = str(use_path or "")
    log_frame_qc_provenance(
        draft_id=draft_id,
        fwhm_limit_px=float(fwhm_lim),
        fwhm_limit_source=fwhm_limit_source,
        cfg_source=result.cfg_source,
        cfg_changed_keys=result.cfg_changed_keys,
    )

    _stage("Preprocess (calibrated -> processed)")
    t0 = time.time()
    _night_run_preprocess(pending=job_ms, ap=ap, pipeline=pipeline, progress_cb=prog_cb)
    _t("preprocess", t0)

    _stage("Plate solve + alignment + MASTERSTAR")
    t0 = time.time()
    ps_out = _night_run_platesolve(
        pending=job_ms,
        ap=ap,
        pipeline=pipeline,
        progress_cb=prog_cb,
    )
    _t("platesolve_align_masterstar", t0)
    from manifest_centre import record_manifest_centres_from_masterstars  # noqa: PLC0415  (G14-03, per setup)

    record_manifest_centres_from_masterstars(ap, int(draft_id))
    return ps_out,


def _nr_photometry(*, cfg, draft_id, params, pipeline, result, stages):
    """Night photometry of every setup and its outcome in the result."""
    phot = run_night_photometry(
        cfg=cfg,
        pipeline=pipeline,
        draft_id=draft_id,
        draft_dir_override=None,
        progress_cb=params.progress_cb,
        timings=stages.timings,
        write_pdfs=True,
        existing_draft=False,
        epsf=params.epsf,
        stages=stages,
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

    phot_errors = list(phot.get("errors") or [])
    return phot, phot_errors


def run_night_pipeline(params: NightRunParams, cfg: AppConfig | None = None) -> NightRunResult:
    """Run full VYVAR pipeline headless (the GUI's job child and the CLI).

    ``cfg``: the run config when the caller already built it (N4, CLI ``main``); else it is loaded
    from ``params.config_path`` (the install's config.json when None)."""
    result = NightRunResult(success=False)
    timings: dict[str, float] = {}
    t_run = time.time()

    from infolog import quiet_repeated_library_warnings, setup_run_logging  # noqa: PLC0415

    # G12-07: one logging setup per run (CLI and GUI job child alike)
    _teardown_logging = setup_run_logging()
    quiet_repeated_library_warnings()
    from except_fix_counters import reset_except_fix_counters  # noqa: PLC0415

    reset_except_fix_counters()  # G12-30: once per run; workers' counts are merged by map_frames

    stages = _RunStages(timings, params.progress_cb)
    _p, _stage, _t = stages.progress, stages.stage, stages.done

    prog_cb = _make_progress_cb(params.progress_cb)

    pipeline = None
    fatal: BaseException | None = None
    try:
        t0 = time.time()
        if cfg is None:
            cfg = _load_app_config(params.config_path)
        pipeline = AstroPipeline(cfg)
        _t("config", t0)

        source = Path(params.source_dir).expanduser().resolve()
        eq_id = int(params.equipment_id)
        tel_id = int(params.telescope_id)
        ps_eq = int(eq_id)

        # G12-15: the GUI's preconditions, before anything is imported
        pre = check_run_preconditions(cfg, params, pipeline.db)
        if pre.problems:
            result.preconditions_failed = True
            result.errors.extend(f"Precondition: {m}" for m in pre.problems)
            result.phase_timings = timings
            return result
        from optics_selection import VyvarOpticsSelection, log_active_optics  # noqa: PLC0415

        # G12-08: the optics line is written by the run (kept until its infolog opens)
        log_active_optics(pipeline.db, VyvarOpticsSelection(eq_id, tel_id), context="RUN VYVAR (camera / telescope)")

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
                precalibrated=bool(params.pre_calibrated_mode),
                masters_from=str(params.masters_from or "lights"),
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

        _stage("Scan source")
        lights_bad, plan = _nr_scan_source(
            _t=_t,
            cfg=cfg,
            eq_id=eq_id,
            params=params,
            pipeline=pipeline,
            source=source,
            tel_id=tel_id,
        )
        if lights_bad:
            result.errors.append("Scan plan is missing or has empty light frames")
            result.phase_timings = timings
            return result
        _resolved_site, import_result = _nr_import(
            _p=_p,
            _stage=_stage,
            _t=_t,
            cfg=cfg,
            eq_id=eq_id,
            params=params,
            pipeline=pipeline,
            plan=plan,
            tel_id=tel_id,
            site=pre.site,
        )

        if getattr(import_result, "draft_id", None) is None:
            result.errors.append("Import did not return draft_id")
            result.phase_timings = timings
            return result

        _cal_out, ap, ap_root, dm_obs, draft_id, lights_root, mf_map, mf_obs = _nr_calibrate(
            _p=_p,
            _resolved_site=_resolved_site,
            _stage=_stage,
            _t=_t,
            eq_id=eq_id,
            import_result=import_result,
            params=params,
            pipeline=pipeline,
            plan=plan,
            prog_cb=prog_cb,
            result=result,
            timings=timings,
        )
        if not lights_root.exists():
            _msg = (
                "Missing non_calibrated/lights after pre-cal import"
                if params.pre_calibrated_mode
                else "Missing /calibrated/lights after calibration"
            )
            result.errors.append(_msg)
            result.phase_timings = timings
            return result

        _multi_obs, _qc_groups, de_ui, fwhm_lim, fwhm_limit_source, ra_ui, t0, use_path = _nr_frame_qc(
            _cal_out=_cal_out,
            _stage=_stage,
            _t=_t,
            ap_root=ap_root,
            cfg=cfg,
            dm_obs=dm_obs,
            draft_id=draft_id,
            eq_id=eq_id,
            lights_root=lights_root,
            mf_map=mf_map,
            mf_obs=mf_obs,
            params=params,
            pipeline=pipeline,
            prog_cb=prog_cb,
            result=result,
            timings=timings,
        )
        if not any(g.best_path for g in _qc_groups.values()):
            result.errors.append("Frame QC: no usable frame in any group (all rejected or no FWHM)")
            result.phase_timings = timings
            return result
        ps_out, = _nr_preprocess_and_platesolve(
            _multi_obs=_multi_obs,
            _qc_groups=_qc_groups,
            _stage=_stage,
            _t=_t,
            ap=ap,
            ap_root=ap_root,
            cfg=cfg,
            de_ui=de_ui,
            draft_id=draft_id,
            fwhm_lim=fwhm_lim,
            fwhm_limit_source=fwhm_limit_source,
            params=params,
            pipeline=pipeline,
            prog_cb=prog_cb,
            ps_eq=ps_eq,
            ra_ui=ra_ui,
            result=result,
            t0=t0,
            use_path=use_path,
        )
        if isinstance(ps_out, dict) and ps_out.get("error"):
            result.errors.append(f"Platesolve failed: {ps_out.get('error')}")
            result.phase_timings = timings
            return result
        result.platesolve_job_output = ps_out


        # Steps 13-15: Photometry (shared slice; it starts its own stages, G12-34)
        phot, phot_errors = _nr_photometry(
            cfg=cfg,
            draft_id=draft_id,
            params=params,
            pipeline=pipeline,
            result=result,
            stages=stages,
        )
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

        try:
            from draft_provenance import purge_calibrated_light_frames  # noqa: PLC0415

            _purge = purge_calibrated_light_frames(ap_root, draft_id=draft_id)
            if _purge["n_files"]:
                _p(f"Calibrated light frames removed: {_purge['n_files']} file(s), "
                   f"{_purge['bytes'] / 1e9:.2f} GB (the raw frames stay)")
            else:
                LOGGER.info("[NightRun] calibrated frames not removed: %s", _purge["skipped"])
        except OSError as _purge_exc:
            result.warnings.append(f"calibrated frames not removed: {_purge_exc}")
            LOGGER.warning("[NightRun] calibrated frames not removed: %s", _purge_exc)

        result.success = True
        timings["total"] = time.time() - t_run
        result.phase_timings = timings
        _p(
            f"Night run complete - draft {draft_id}, {result.n_lightcurves} light curve(s), "
            f"{result.n_frames} frame(s)"
        )
        return result

    except Exception as exc:  # noqa: BLE001
        # G12-07: the record carries the traceback into the infolog (and stdout)
        LOGGER.exception("[NightRun] FATAL: %s", exc)
        result.errors.append(str(exc))
        timings["total"] = time.time() - t_run
        result.phase_timings = timings
        fatal = exc
        return result
    finally:
        from infolog import end_infolog_session, infolog_session_open, log_repeated_warnings_summary  # noqa: PLC0415

        log_repeated_warnings_summary()
        if not result.success and not infolog_session_open():
            # N5: no infolog of the field - the failure goes to logs/run_preflight_error_<ts>.log
            _exc = fatal or RuntimeError("; ".join(result.errors) or "night run failed")
            result.failure_log = _write_failure_log(
                cfg, step=stages.current, exc=_exc, db=getattr(pipeline, "db", None)
            )
        # INFOLOG-ONE-2026-10: the run's single infolog closes after its last line
        end_infolog_session()
        _teardown_logging()


def main(argv: list[str] | None = None) -> int:
    """CLI entry: same three required inputs as UI RUN VYVAR."""
    args = parse_night_run_cli(argv)
    if args.job is not None:
        from run_job import run_job_file  # noqa: PLC0415

        return run_job_file(args.job)
    from infolog import setup_run_logging  # noqa: PLC0415
    from vyvar_runtime import ensure_data_dir_for_entry  # noqa: PLC0415

    teardown = setup_run_logging()  # G12-07: the CLI logs like the run (stdout + infolog)
    try:
        ensure_data_dir_for_entry()  # G12-29: the GUI's data-dir bootstrap
        cfg = _load_app_config(args.config)  # N4: built once, passed to the run
        eq, tel, loc, missing, loc_source = resolve_night_run_cli_ids(
            equipment_id=args.equipment_id,
            telescope_id=args.telescope_id,
            location_id=args.location_id,
            draft_dir=args.draft_dir,
            cfg=cfg,
        )
        if missing:
            print(night_run_missing_message(missing), file=sys.stderr)
            return 2
        # Pass only an explicit CLI site as location_id; otherwise pass the
        # resolved id via manifest_location_id / config through the resolver.
        _explicit = _positive_id(args.location_id)
        _manifest = int(loc) if loc_source == "manifest" else None
        params = NightRunParams(
            source_dir=Path(args.source),
            equipment_id=int(eq),
            telescope_id=int(tel),
            location_id=_explicit,
            manifest_location_id=_manifest,
            location_source_hint="cli_arg" if _explicit is not None else None,
            config_path=args.config,
            dry_run=bool(args.dry_run),
            pre_calibrated_mode=bool(getattr(args, "pre_calibrated", False)),
            masters_from=str(getattr(args, "masters_from", "lights") or "lights"),
        )
        # S6-FIX RUN-LOCK-2026-10: one run per data directory, like the GUI's jobs (run_job.spawn_job)
        from run_job import JobBusyError, fail_job, finish_job, params_to_job, register_external_job  # noqa: PLC0415

        try:
            job_dir = register_external_job(cfg.data_root, spec=params_to_job(params), label="Night run (CLI)",
                                            prefix="cli")
        except JobBusyError as exc:
            print(f"ERROR: not started - {exc}", file=sys.stderr)
            return 2
        try:
            result = run_night_pipeline(params, cfg=cfg)
        except BaseException as exc:
            fail_job(job_dir, exc, None)
            raise
        finish_job(job_dir, result)
        if result.success:
            return 0
        for err in result.errors:
            print(f"ERROR: {err}", file=sys.stderr)
        if result.failure_log:
            print(f"Preflight log: {result.failure_log}", file=sys.stderr)
        return 2 if result.preconditions_failed else 1
    finally:
        teardown()


if __name__ == "__main__":
    raise SystemExit(main())

