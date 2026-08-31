"""Night-run ePSF stage: build, per-frame PSF fit+merge, internal PSF LCs.

INV-EPSF-STAGE-01: one function consumed by ``run_night_pipeline``, the ePSF
dashboard, the RUN ePSF job, and ``psf_runner.main``. No subprocess.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

LOGGER = logging.getLogger(__name__)

ProgressCb = Callable[[str], None]


@dataclass
class EpsfStagePaths:
    """Filesystem inputs for one obs-group ePSF stage."""

    platesolve_dir: Path
    frames_root: Path
    masterstar_fits: Path | None = None
    masterstars_csv: Path | None = None
    photometry_dir: Path | None = None


def _p(progress_cb: ProgressCb | None, msg: str) -> None:
    LOGGER.info("[ePSF stage] %s", msg)
    if progress_cb is not None:
        progress_cb(msg)


def _resolve_paths(paths: EpsfStagePaths | dict[str, Any]) -> EpsfStagePaths:
    if isinstance(paths, EpsfStagePaths):
        ps = Path(paths.platesolve_dir)
        frames = Path(paths.frames_root)
        ms = Path(paths.masterstar_fits) if paths.masterstar_fits is not None else ps / "MASTERSTAR.fits"
        csv = (
            Path(paths.masterstars_csv)
            if paths.masterstars_csv is not None
            else ps / "masterstars_full_match.csv"
        )
        phot = Path(paths.photometry_dir) if paths.photometry_dir is not None else ps / "photometry"
        return EpsfStagePaths(
            platesolve_dir=ps,
            frames_root=frames,
            masterstar_fits=ms,
            masterstars_csv=csv,
            photometry_dir=phot,
        )
    ps = Path(paths["platesolve_dir"])
    frames = Path(paths["frames_root"])
    ms_raw = paths.get("masterstar_fits")
    csv_raw = paths.get("masterstars_csv")
    phot_raw = paths.get("photometry_dir")
    return EpsfStagePaths(
        platesolve_dir=ps,
        frames_root=frames,
        masterstar_fits=Path(ms_raw) if ms_raw else ps / "MASTERSTAR.fits",
        masterstars_csv=Path(csv_raw) if csv_raw else ps / "masterstars_full_match.csv",
        photometry_dir=Path(phot_raw) if phot_raw else ps / "photometry",
    )


def run_epsf_stage(
    params: Any,
    paths: EpsfStagePaths | dict[str, Any],
    cfg: Any,
    progress_cb: ProgressCb | None = None,
    *,
    db: Any = None,
    draft_id: int | None = None,
    equipment_id: int | None = None,
    do_build: bool = True,
    do_fit_merge: bool = True,
    do_lc: bool = True,
    max_frames: int | None = None,
    dry_run: bool = False,
    pipeline_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build ePSF, fit+merge psf_* into proc sidecars, write internal PSF LCs.

    ``params.epsf`` True/False overrides config. None reads ``cfg.epsf_auto_run``
    (default OFF). When ``params`` is None the stage always runs (UI ePSF button
    / dashboard). Sub-steps are selected by ``do_build`` / ``do_fit_merge`` /
    ``do_lc``.
    """
    if params is not None:
        flag = getattr(params, "epsf", None)
        if flag is False:
            _p(progress_cb, "skipped (NightRunParams.epsf=False)")
            return {"skipped": True, "reason": "epsf=False"}
        if flag is not True:
            auto = bool(getattr(cfg, "epsf_auto_run", False)) if cfg is not None else False
            if not auto:
                _p(progress_cb, "skipped (epsf_auto_run=False)")
                return {"skipped": True, "reason": "epsf=False"}

    resolved = _resolve_paths(paths)
    ps = resolved.platesolve_dir
    frames = resolved.frames_root
    ms_fits = resolved.masterstar_fits
    ms_csv = resolved.masterstars_csv
    phot = resolved.photometry_dir
    assert ms_fits is not None and ms_csv is not None and phot is not None

    did = draft_id
    if did is None and params is not None:
        did = getattr(params, "draft_id", None)
    eq = equipment_id
    if eq is None and params is not None:
        eq = getattr(params, "equipment_id", None)

    cfg_stage = copy.copy(cfg)
    try:
        cfg_stage.psf_photometry_enabled = True
        cfg_stage.photometry_mode = "both"
    except Exception:  # noqa: BLE001
        pass

    out: dict[str, Any] = {
        "skipped": False,
        "platesolve_dir": str(ps),
        "frames_root": str(frames),
        "build": None,
        "merge": None,
        "lc": None,
        "epsf_model_sha256": "",
        "n_stars": -1,
        "epsf_path": None,
    }

    if dry_run:
        _p(progress_cb, f"dry-run build={do_build} fit={do_fit_merge} lc={do_lc} {ps}")
        out["dry_run"] = True
        return out

    if do_build:
        if not ms_fits.is_file():
            raise FileNotFoundError(f"MASTERSTAR.fits not found: {ms_fits}")
        if not ms_csv.is_file():
            raise FileNotFoundError(f"masterstars_full_match.csv not found: {ms_csv}")
        if db is None:
            raise ValueError("run_epsf_stage build requires db")
        if did is None:
            raise ValueError("run_epsf_stage build requires draft_id")
        from psf_photometry import build_epsf_model
        from psf_internal_lc import load_epsf_build_meta

        _p(progress_cb, "Step ePSF-1: build model")
        epsf_path = build_epsf_model(
            masterstar_fits_path=ms_fits,
            masterstars_csv_path=ms_csv,
            db=db,
            draft_id=int(did),
        )
        meta = load_epsf_build_meta(ps)
        out["build"] = {
            "epsf_path": str(epsf_path),
            "epsf_model_sha256": meta.get("model_sha256") or "",
            "n_stars": meta.get("n_stars"),
        }
        out["epsf_path"] = str(epsf_path)
        out["epsf_model_sha256"] = str(meta.get("model_sha256") or "")
        try:
            out["n_stars"] = int(meta.get("n_stars"))
        except (TypeError, ValueError):
            out["n_stars"] = -1
        _p(
            progress_cb,
            f"ePSF model {Path(str(epsf_path)).name} n_stars={out['n_stars']}",
        )

    if do_fit_merge:
        from epsf_psf_merge import run_epsf_psf_merge_job

        _p(progress_cb, "Step ePSF-2/3: per-frame PSF fit + guarded sidecar merge")

        def _merge_cb(i: int, total: int, msg: str) -> None:
            _p(progress_cb, f"ePSF fit {i}/{total}: {msg}")

        merge_out = run_epsf_psf_merge_job(
            frames_root=frames,
            platesolve_dir=ps,
            app_config=cfg_stage,
            draft_id=int(did) if did is not None else None,
            equipment_id=int(eq) if eq is not None else None,
            progress_cb=_merge_cb,
            pipeline_meta=pipeline_meta,
            write_internal_lc=False,
            max_frames=max_frames,
        )
        out["merge"] = {
            "written": merge_out.get("written"),
            "frames_total": merge_out.get("frames_total"),
            "science_set": merge_out.get("science_set"),
            "epsf_job_summary": merge_out.get("epsf_job_summary"),
        }

    if do_lc:
        from psf_internal_lc import write_internal_psf_lightcurves

        _p(progress_cb, "Step ePSF-4: internal PSF light curves")
        lc_out = write_internal_psf_lightcurves(
            platesolve_dir=ps,
            frames_root=frames,
            photometry_dir=phot,
            cfg=cfg_stage,
        )
        out["lc"] = {
            "n_written": lc_out.get("n_written"),
            "n_skipped": lc_out.get("n_skipped"),
        }

    if out["epsf_model_sha256"] == "" and (ps / "masterstar_epsf.fits").is_file():
        from psf_internal_lc import load_epsf_build_meta

        meta = load_epsf_build_meta(ps)
        out["epsf_model_sha256"] = str(meta.get("model_sha256") or "")
        try:
            out["n_stars"] = int(meta.get("n_stars"))
        except (TypeError, ValueError):
            pass

    try:
        from photometry_core import merge_photometry_pipeline_meta

        merge_photometry_pipeline_meta(
            phot,
            {
                "epsf_model_sha256": out.get("epsf_model_sha256") or "",
                "epsf_n_stars": out.get("n_stars"),
            },
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[ePSF stage] pipeline_meta stamp skipped: %s", exc)

    return out
