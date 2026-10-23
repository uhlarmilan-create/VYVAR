"""Night-run ePSF stage: build, per-frame PSF fit+merge, internal PSF LCs.

INV-EPSF-STAGE-01: one function consumed by ``run_night_pipeline``, the ePSF
dashboard and the RUN ePSF job. No subprocess.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pipeline_constants import EPSF_MIN_BUILD_STARS

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


def remove_stale_epsf_products(photometry_dir: Path, setup: str, target_ids: frozenset[str] | set[str]) -> list[str]:
    """EPSF-STALE-2026-10: the fit + merge rewrites every psf_* column of the setup's sidecars and replaces the
    ePSF manifest, so an ePSF light curve (and its user export) of a target outside the new selection describes
    fits that no longer exist - remove it. Returns the removed catalog ids."""
    from epsf_export import find_epsf_export

    lc_dir = Path(photometry_dir) / "lightcurves"
    removed = []
    for p in sorted(lc_dir.glob("lightcurve_*_psf.csv")):
        cid = p.name[len("lightcurve_"): -len("_psf.csv")]
        if cid in target_ids:
            continue
        p.unlink()
        exp = find_epsf_export(photometry_dir, cid, setup)
        if exp is not None:
            exp.unlink()
        removed.append(cid)
    return removed


def run_epsf_stage(
    params: Any,
    paths: EpsfStagePaths | dict[str, Any],
    cfg: Any,
    progress_cb: ProgressCb | None = None,
    *,
    db: Any = None,
    draft_id: int | None = None,
    do_build: bool = True,
    do_fit_merge: bool = True,
    do_lc: bool = True,
    max_frames: int | None = None,
    dry_run: bool = False,
    pipeline_meta: dict[str, Any] | None = None,
    targets: str | None = None,
) -> dict[str, Any]:
    """Build ePSF, fit+merge psf_* into proc sidecars, write internal PSF LCs.

    ``targets``: which targets are fitted (``epsf_science_set.build_epsf_fit_set`` selection: ``ALL`` or
    comma-separated VSX types / names / ids, ``*`` wildcards); None -> ``cfg.epsf_targets``.

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

    out: dict[str, Any] = {
        "skipped": False,
        "platesolve_dir": str(ps),
        "frames_root": str(frames),
        "build": None,
        "merge": None,
        "lc": None,
        "export": None,
        "epsf_model_sha256": "",
        "n_stars": -1,
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
        from psf_photometry import select_epsf_build_stars

        _p(progress_cb, "Step ePSF-1: select the build stars on the MASTERSTAR catalogue")
        sel = select_epsf_build_stars(
            masterstar_fits_path=ms_fits,
            masterstars_csv_path=ms_csv,
            db=db,
            draft_id=int(did),
            min_stars=EPSF_MIN_BUILD_STARS,
        )
        out["build"] = {"n_build_stars": sel["n_stars"], "fwhm_px": sel["fwhm_px"], "cutout_size": sel["cutout_size"]}
        out["n_stars"] = int(sel["n_stars"])
        _p(progress_cb, f"ePSF build stars: {sel['n_stars']}")

    from epsf_science_set import build_epsf_fit_set

    selection = targets if targets is not None else str(getattr(cfg, "epsf_targets", "ALL") or "ALL")
    fit_set = build_epsf_fit_set(ps, selection)
    out["fit_set"] = fit_set.to_meta_dict()
    _p(progress_cb, f"fit set '{fit_set.selection}': {len(fit_set.target_ids)} targets + "
       f"{len(fit_set.comp_ids - fit_set.target_ids)} comparison stars"
       + (f"; not matched: {', '.join(fit_set.unmatched_tokens)}" if fit_set.unmatched_tokens else ""))
    if (do_fit_merge or do_lc) and not fit_set.catalog_ids:
        _p(progress_cb, f"PSF fit skipped: {fit_set.empty_reason}")
        do_fit_merge = do_lc = False

    if do_fit_merge:
        from epsf_psf_merge import run_epsf_psf_merge_job

        _p(progress_cb, "Step ePSF-2/3: per-frame PSF fit + guarded sidecar merge")

        def _merge_cb(i: int, total: int, msg: str) -> None:
            _p(progress_cb, f"ePSF fit {i}/{total}: {msg}")

        merge_out = run_epsf_psf_merge_job(
            frames_root=frames,
            platesolve_dir=ps,
            app_config=cfg,
            progress_cb=_merge_cb,
            fit_set=fit_set,
            pipeline_meta=pipeline_meta,
            max_frames=max_frames,
        )
        out["merge"] = {
            "written": merge_out.get("written"),
            "frames_total": merge_out.get("frames_total"),
            "fit_set": merge_out.get("fit_set"),
            "spatial": merge_out.get("spatial"),
            "epsf_job_summary": merge_out.get("epsf_job_summary"),
        }

    if do_fit_merge:
        out["removed_stale"] = remove_stale_epsf_products(phot, ps.name, fit_set.target_ids)
        if out["removed_stale"]:
            _p(progress_cb, f"removed the ePSF light curves of {len(out['removed_stale'])} target(s) outside this "
               "run's selection (their PSF fits were replaced)")

    if do_lc:
        from psf_internal_lc import write_internal_psf_lightcurves

        _p(progress_cb, "Step ePSF-4: internal PSF light curves")
        lc_out = write_internal_psf_lightcurves(
            platesolve_dir=ps,
            frames_root=frames,
            photometry_dir=phot,
            target_ids=sorted(fit_set.target_ids),
        )
        out["lc"] = {
            "n_written": lc_out.get("n_written"),
            "n_skipped": lc_out.get("n_skipped"),
        }
        if bool(getattr(cfg, "epsf_export_auto", True)):
            from epsf_export import write_epsf_user_lightcurves

            exp = write_epsf_user_lightcurves(ps, sorted(fit_set.target_ids), setup=ps.name, photometry_dir=phot,
                                              observer_code=str(getattr(cfg, "observer_code", "") or ""))
            out["export"] = {"n_written": exp["n_written"], "skipped": exp["skipped"]}
            _p(progress_cb, f"ePSF light-curve files: {exp['n_written']} in {phot / 'lightcurves_reports' / 'epsf'}")
        else:
            _p(progress_cb, "ePSF light-curve files not written (epsf_export_auto=false): ePSF tab button")

    from psf_internal_lc import load_epsf_build_meta

    out["epsf_model_sha256"] = str(load_epsf_build_meta(ps).get("model_sha256") or "")

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
