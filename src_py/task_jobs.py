"""Long GUI actions as separate job processes (G14-04, G09-43, G14-14; Milan 2026-10-31).

RUN ePSF and the other GUI actions that write science products run like RUN VYVAR: the GUI writes a
job spec and starts ``python -m task_jobs --job <spec>`` (``run_job.spawn_job``: one job at a time per
data directory, G14-37). The child reports into ``run_status.json``; the GUI follows it with the RUN
monitor fragment from any view, so a click or a view switch never interrupts the work and an abort
never happens between two files inside the Streamlit thread.

A task on a processed field runs with the field's stored config snapshot, exactly as a night re-run
(``night_run.resolve_cfg_for_photometry(existing_draft=True)``, INV-CFG-SOURCE-01), never with the
Settings of the browser session.

``job.json``: ``{"task": <name in TASKS>, "args": {...}}``; ``result.json``: the task's result dict.
This module never imports Streamlit (CORE-NO-UI-2026-10).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable

import astro_env  # noqa: F401  (IERS-OFF-2026-10, as in run_job)
from run_job import JOB_FILE, _now, _write_json_atomic, read_status, spawn_job, write_status

LOGGER = logging.getLogger(__name__)

RESULT_JSON = "result.json"

Progress = Callable[[str], None]


def _live_cfg() -> Any:
    from config import AppConfig  # noqa: PLC0415

    return AppConfig()


def draft_task_cfg(draft_dir: Path) -> tuple[Any, str, list[str]]:
    """The config of a task on a processed field: its stored snapshot over the install config (as a
    night re-run). Returns ``(cfg, cfg_source, changed_keys)``."""
    from night_run import resolve_cfg_for_photometry  # noqa: PLC0415

    return resolve_cfg_for_photometry(_live_cfg(), Path(draft_dir), existing_draft=True)


def _db(cfg: Any) -> Any:
    from database import VyvarDatabase  # noqa: PLC0415

    return VyvarDatabase.for_config(cfg)


def _setup_dirs(draft_dir: Path, setup: str) -> tuple[Path, Path, Path]:
    """(platesolve/<setup>, its photometry dir, its aligned frames dir)."""
    ps = Path(draft_dir) / "platesolve" / str(setup)
    return ps, ps / "photometry", Path(draft_dir) / "detrended_aligned" / "lights" / str(setup)


# --------------------------------------------------------------------------------------------- tasks
def task_run_epsf(args: dict[str, Any], progress: Progress) -> dict[str, Any]:
    """RUN ePSF of the ePSF tab: every given setup (EPSF-MULTISETUP-2026-10), every target with an aperture light
    curve; the same ``run_epsf_stage`` call and paths as the night run's ``epsf_auto_run``."""
    from epsf_stage import EpsfStagePaths, run_epsf_stage  # noqa: PLC0415
    from platesolve_ui_paths import find_photometry_setups  # noqa: PLC0415

    draft_dir = Path(args["draft_dir"])
    if args.get("draft_id") is None:
        raise ValueError(f"RUN ePSF: the processed field {draft_dir} has no id in its manifest")
    draft_id = int(args["draft_id"])
    names = [str(n) for n in args.get("setups") or []]
    if not names:
        raise ValueError("RUN ePSF job without setups.")
    cfg, cfg_source, changed = draft_task_cfg(draft_dir)
    db = _db(cfg)
    all_setups = find_photometry_setups(cfg, draft_id, draft_dir_override=draft_dir)
    lines: list[str] = []
    n_lc_total = 0
    for k, name in enumerate(names, start=1):
        p = all_setups.get(name)
        if p is None:
            raise FileNotFoundError(f"{name}: setup not found in {draft_dir / 'platesolve'}")
        og_dir = Path(p["obs_group_dir"])
        ms_fits = Path(p["masterstar_fits"])
        ms_csv = og_dir / "masterstars_full_match.csv"
        frames = Path(p["per_frame_csv_dir"]) if p.get("per_frame_csv_dir") else None
        for f, what in ((ms_fits, "MASTERSTAR.fits"), (ms_csv, "masterstars_full_match.csv")):
            if not f.is_file():
                raise FileNotFoundError(f"{name}: {what} not found: {f}")
        if frames is None or not frames.is_dir():
            raise FileNotFoundError(f"{name}: aligned frames dir not found: {frames}")
        tag = f"setup {k}/{len(names)} {name}"
        out = run_epsf_stage(
            paths=EpsfStagePaths(platesolve_dir=og_dir, frames_root=frames, masterstar_fits=ms_fits,
                                 masterstars_csv=ms_csv, photometry_dir=Path(p["photometry_dir"])),
            cfg=cfg,
            progress_cb=lambda m, _t=tag: progress(f"{_t}: {m}"),
            db=db,
            draft_id=draft_id,
        )
        n_tg = len((out.get("fit_set") or {}).get("target_types") or {})
        n_lc = int((out.get("lc") or {}).get("n_written") or 0)
        written = int((out.get("merge") or {}).get("written") or 0)
        n_lc_total += n_lc
        exp = out.get("export")
        lines.append(f"{name}: {n_tg} target(s), {n_lc} PSF light curve(s), {written} frame(s) fitted"
                     + (f", {exp['n_written']} light-curve file(s)" if exp else ""))
    msg = f"ePSF photometry done ({len(names)} setup(s)): " + "; ".join(lines)
    if changed:
        msg += f" - config snapshot of the field, keys that differ from Settings: {', '.join(changed)}"
    return {"message": msg, "n_lightcurves": n_lc_total, "cfg_source": cfg_source, "cfg_changed_keys": changed}


def task_variability_search(args: dict[str, Any], progress: Progress) -> dict[str, Any]:
    """Variability tab "Run the variability search now"."""
    from variability_field import run_field_variability  # noqa: PLC0415

    draft_dir, setup = Path(args["draft_dir"]), str(args["setup"])
    ps, phot, frames = _setup_dirs(draft_dir, setup)
    cfg, _, _ = draft_task_cfg(draft_dir)
    progress(f"variability search {setup}")
    out = run_field_variability(frames, phot, cfg, platesolve_dir=ps)
    return {"message": f"Variability search {setup}: {out.get('status', 'done')}"}


def task_variability_catalog_check(args: dict[str, Any], progress: Progress) -> dict[str, Any]:
    """Variability tab "Check catalogues again" (online catalogues for every new candidate)."""
    from catalog_crossmatch import _catalog_check  # noqa: PLC0415
    from variability_field import load_variability_candidates, write_candidate_catalogs  # noqa: PLC0415

    draft_dir, setup = Path(args["draft_dir"]), str(args["setup"])
    _, phot, _ = _setup_dirs(draft_dir, setup)
    cfg, _, _ = draft_task_cfg(draft_dir)
    cand = load_variability_candidates(phot)
    progress(f"querying the catalogues for {len(cand)} candidate(s)")
    write_candidate_catalogs(phot, {str(x["catalog_id"]): _catalog_check(x, cfg) for _, x in cand.iterrows()})
    return {"message": f"Catalogues checked for {len(cand)} candidate(s) of {setup}"}


def task_tess(args: dict[str, Any], progress: Progress) -> dict[str, Any]:
    """Variability tab "Download TESS and analyse" for one star."""
    from tess_runner import run_tess_for_star  # noqa: PLC0415

    draft_dir, setup, cid = Path(args["draft_dir"]), str(args["setup"]), str(args["catalog_id"])
    _, phot, _ = _setup_dirs(draft_dir, setup)
    cfg, _, _ = draft_task_cfg(draft_dir)
    run_tess_for_star(phot, cid, cfg, progress=lambda m, v: progress(f"{m} ({100 * float(v):.0f} %)"))
    return {"message": f"TESS done for {cid}"}


def task_hrd_enrich(args: dict[str, Any], progress: Progress) -> dict[str, Any]:
    """HRD tab "Look up the notable stars online": Gaia TAP + SIMBAD into ``_hrd_cache/hrd_enrich.json``
    (NEW-1: the tab itself only reads that file). Same call as the PDF report of the night run."""
    from hrd_analysis import (  # noqa: PLC0415
        HRD_PARALLAX_MIN_MAS,
        HRD_PARALLAX_SNR_MIN,
        build_hrd_dataframe,
        get_top_interesting_stars,
    )

    draft_dir, setup = Path(args["draft_dir"]), str(args["setup"])
    ps, _, _ = _setup_dirs(draft_dir, setup)
    cfg, _, _ = draft_task_cfg(draft_dir)
    gdb = Path(str(getattr(cfg, "gaia_db_path", "") or "").strip())
    if not gdb.is_file():
        raise FileNotFoundError(f"Gaia DB not found: {gdb} (Settings gaia_db_path)")
    df = build_hrd_dataframe(ps / "masterstars_full_match.csv", gdb, parallax_min_mas=HRD_PARALLAX_MIN_MAS,
                             parallax_snr_min=HRD_PARALLAX_SNR_MIN)
    progress("Gaia TAP + SIMBAD lookup")
    top = get_top_interesting_stars(df, cfg=cfg, cache_path=ps / "_hrd_cache" / "hrd_enrich.json")
    return {"message": f"Notable stars of {setup} looked up online ({len(top)} in the table)"}


#: What "Build masters" builds: auto (by IMAGETYP / exposure) or one kind.
BUILD_WHAT = ("auto", "dark", "darkflat", "flat")


def build_masters_from_folder(
    src: str | Path,
    *,
    what: str,
    db: Any,
    camera_id: int,
    telescope_id: int,
    library_root: str | Path,
    method: str,
) -> tuple[list[str], list[str]]:
    """Calibration Library "Build masters": the raw DARK / FLAT frames of a folder (subfolders included,
    by IMAGETYP) -> ``calib_plan.build_masters_from_source``. Returns (built, messages).

    Raises ``FileNotFoundError`` / ``ValueError`` when the folder or its frames are missing."""
    from calib_plan import build_masters_from_source, is_dark_typed  # noqa: PLC0415
    from importer import _collect_fits_by_type, _looks_like_master  # noqa: PLC0415

    if what not in BUILD_WHAT:
        raise ValueError(f"build_masters_from_folder: what must be one of {BUILD_WHAT}, got {what!r}")
    p = Path(src)
    if not p.is_dir():
        raise FileNotFoundError(f"Folder not found: {p}")
    by = _collect_fits_by_type(p, db=db)
    darks = [f for f in by.get("dark", []) if not _looks_like_master(f)]
    flats = [f for f in by.get("flat", []) if not _looks_like_master(f)]
    dark_kind = None
    if what in ("dark", "darkflat"):
        dark_kind, flats = what, []
        # dark-flat frames may be typed FLAT DARK; they are dark frames here
        darks = darks + [f for f in by.get("flat", []) if not _looks_like_master(f) and is_dark_typed(f)]
    elif what == "flat":
        darks = []
        flats = [f for f in flats if not is_dark_typed(f)]
    if not darks and not flats:
        raise ValueError(f"No raw {what.upper() if what != 'auto' else 'DARK or FLAT'} frames found in {p} "
                         "(IMAGETYP must say dark / flat).")
    # G14-16: a DB error must not silently build an OSC flat as mono (no Bayer mask) - it surfaces
    bayer = db.get_equipment_bayermask(int(camera_id))
    return build_masters_from_source(
        dark_files=darks, flat_files=flats, light_exposures=[], db=db, camera_id=int(camera_id),
        telescope_id=int(telescope_id), library_root=library_root, method=method, equipment_bayer_mask=bayer,
        dark_kind=dark_kind,
    )


def task_build_masters(args: dict[str, Any], progress: Progress) -> dict[str, Any]:
    """Calibration Library "Build masters" (install config: the library is not part of a field)."""
    cfg = _live_cfg()
    progress(f"building masters from {args['src']}")
    built, msgs = build_masters_from_folder(
        args["src"], what=str(args["what"]), db=_db(cfg), camera_id=int(args["camera_id"]),
        telescope_id=int(args["telescope_id"]), library_root=Path(cfg.calibration_library_root),
        method=str(args["method"]),
    )
    msg = f"{len(built)} master(s) built" + (": " + ", ".join(built) if built else "")
    return {"message": msg + ("; " + "; ".join(msgs) if msgs else ""), "built": list(built), "messages": list(msgs)}


TASKS: dict[str, Callable[[dict[str, Any], Progress], dict[str, Any]]] = {
    "run_epsf": task_run_epsf,
    "variability_search": task_variability_search,
    "variability_catalog_check": task_variability_catalog_check,
    "tess": task_tess,
    "hrd_enrich": task_hrd_enrich,
    "build_masters": task_build_masters,
}


# ----------------------------------------------------------------------------------------- protocol
def launch_task(task: str, args: dict[str, Any], data_root: Path | str, *, label: str) -> Path:
    """Start one task as a job process; ``run_job.JobBusyError`` when a job of the data directory runs."""
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; known: {sorted(TASKS)}")
    json.dumps(args)  # args must travel through JSON
    return spawn_job(data_root, spec={"task": task, "args": dict(args)}, module="task_jobs", label=label,
                     prefix=task)


def load_task_result(job_dir: Path | str) -> dict[str, Any] | None:
    p = Path(job_dir) / RESULT_JSON
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def run_task_file(job_path: Path | str) -> int:
    """Child side: run the task of ``job.json`` and report into ``run_status.json``."""
    job_path = Path(job_path)
    job_dir = job_path.parent
    job = json.loads(job_path.read_text(encoding="utf-8"))
    label = (read_status(job_dir) or {}).get("label") or str(job.get("task"))
    import os  # noqa: PLC0415

    write_status(job_dir, state="running", pid=os.getpid(), message="Running...")

    def _progress(msg: str) -> None:
        LOGGER.info("[%s] %s", label, msg)
        try:
            write_status(job_dir, state="running", process=label, message=str(msg)[:800])
        except OSError as exc:  # progress must never stop the task
            LOGGER.warning("[%s] WARNING: status not written: %s", label, exc)

    try:
        fn = TASKS[str(job["task"])]
        result = dict(fn(dict(job.get("args") or {}), _progress))
    except BaseException as exc:  # noqa: BLE001
        LOGGER.exception("[%s] failed", label)
        write_status(job_dir, state="failed", message=f"{type(exc).__name__}: {exc}"[:800],
                     errors=[f"{type(exc).__name__}: {exc}"], finished_utc=_now())
        return 1
    _write_json_atomic(job_dir / RESULT_JSON, result)
    write_status(job_dir, state="done", message=str(result.get("message") or "Done")[:800], finished_utc=_now())
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="task_jobs", description="Run one GUI task job spec.")
    ap.add_argument("--job", type=Path, required=True, help=f"{JOB_FILE} written by launch_task")
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(levelname)s %(message)s")
    return run_task_file(ap.parse_args(argv).job)


if __name__ == "__main__":
    raise SystemExit(main())
