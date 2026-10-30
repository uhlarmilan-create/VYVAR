"""Moved from pipeline.py (CONSOLIDATE-01E6b). Facade re-exports this name.

export_per_frame_catalogs: per-frame catalog CSV export with MP pool.
Workers, initializer, and _EXPORT_PER_FRAME_WORKER_STATE live in
pipeline_catalog (E6a); imported E3 here.
detect_stars_and_match_catalog moves to catalog_match.py (C1);
imported via call-time facade import inside the body.
"""
from __future__ import annotations

import json
import logging
import math
import pickle
from pathlib import Path
from typing import Any, Sequence

from astropy.io import fits
import pandas as pd

from aperture_policy import fwhm_from_header_vy_fwhm
from config import AppConfig
from pipeline_constants import DAO_FWHM_FALLBACK_PX
from infolog import log_event
from utils import iter_fits_paths_recursive as _iter_fits_recursive
from pipeline_astrometry import _equipment_saturate_adu_from_db
from proc_frame_store import _vyvar_df_to_csv
from pipeline_catalog import (
    _estimate_catalog_frame_hw,
    _export_per_frame_disk_worker_task,
    _export_per_frame_ram_worker_task,
    _finalize_hybrid_bkg_fallback_sidecar,
    _init_export_per_frame_worker,
    find_qc_metrics_csv,
)

LOGGER = logging.getLogger("pipeline")

#: per_frame_catalog_index.csv: one row per aligned frame (OUTPUT-REVIEW-2026-10 step 6).
PER_FRAME_INDEX_COLUMNS: tuple[str, ...] = ("file", "status", "csv", "n_detected", "n_matched")


def write_per_frame_index(rows: Sequence[dict[str, Any]], path: Path | str) -> Path:
    """Write ``per_frame_catalog_index.csv`` (the index columns of the per-frame result rows)."""
    df = pd.DataFrame([{c: r.get(c) for c in PER_FRAME_INDEX_COLUMNS} for r in rows],
                      columns=list(PER_FRAME_INDEX_COLUMNS))
    _vyvar_df_to_csv(df, Path(path))
    return Path(path)


def _ef_setup(*, app_config, equipment_id, masterstar_fits, platesolve_dir):
    """Run config, night median FWHM, frame-QC FWHM per frame and the photometric gain / read noise of the MASTERSTAR."""

    _cfg_ap = app_config

    ps = Path(platesolve_dir) if platesolve_dir is not None else None
    _qc_fwhm_by_name: dict[str, float] = {}
    _fwhm_night_median_px: float | None = None
    try:
        from aperture_policy import load_qc_fwhm_map  # noqa: PLC0415

        _qc_root = None
        if ps is not None:
            try:
                _qc_root = Path(ps).resolve().parents[1]
            except IndexError:
                _qc_root = None
        _qc_csv = find_qc_metrics_csv(_qc_root) if _qc_root is not None else None
        _qc_fwhm_by_name, _fwhm_night_median_px = load_qc_fwhm_map(_qc_csv)
    except Exception as _qc_exc:  # noqa: BLE001
        LOGGER.warning("[APERTURE-01] QC FWHM map not loaded: %s", _qc_exc)

    # GAIN-ONE-2026-10: the run's gain / read noise (MASTERSTAR header, EQUIPMENTS), not the raw
    # config defaults; used by the background-error ratio of the per-frame catalogs
    from rig_photometry import photometric_gain_read_noise  # noqa: PLC0415

    _ms_hdr_fe = None
    if masterstar_fits is not None and Path(masterstar_fits).is_file():
        try:
            _ms_hdr_fe = fits.getheader(str(masterstar_fits), 0)
        except OSError:
            _ms_hdr_fe = None
    _pg_fe = photometric_gain_read_noise(_ms_hdr_fe, _cfg_ap, equipment_id)
    _gain_fe, _rn_fe = float(_pg_fe.gain), float(_pg_fe.read_noise)
    return _cfg_ap, _fwhm_night_median_px, _gain_fe, _qc_fwhm_by_name, _rn_fe, ps


def _ef_sat_diag(*, _cfg_ap, _msp, equipment_id, equipment_saturate_adu, files, master_tab, ps, work_ram=None):
    """SAT-DIAG context of the draft (measured saturation level) and the drift reference star."""
    LOGGER.info(
        "Per-frame catalog: MASTERSTAR lock (single catalog: %s, %s rows)",
        _msp.name,
        len(master_tab),
    )

    _sat_diag_ctx: Any | None = None
    _sat_diag_archive: str = ""
    _ref_ra_deg: float | None = None
    _ref_dec_deg: float | None = None
    _drift_ref_catalog_id: str | None = None
    from sat_diag import (  # noqa: PLC0415
        draft_archive_from_platesolve,
        run_sat_diag,
    )

    _arch = draft_archive_from_platesolve(ps)
    if _arch is not None:
        _sat_diag_archive = str(_arch)
        _eq_sat = equipment_saturate_adu
        if _eq_sat is None and equipment_id is not None:
            _eq_sat = _equipment_saturate_adu_from_db(int(equipment_id), _cfg_ap)
        # G05-02: with the RAM handoff the aligned frames are not on disk yet - the header comes from
        # memory (a missing file or a stale one of an earlier run was read before)
        _ref_hdr = fits.Header()
        if work_ram:
            _ref_hdr = work_ram[0][1]
        elif files:
            with fits.open(files[0], memmap=False) as _rh:
                _ref_hdr = _rh[0].header
        _sat_diag_ctx = run_sat_diag(_arch, equipment_adu=_eq_sat, hdr=_ref_hdr)
        if _sat_diag_ctx.sat_adu is not None:
            equipment_saturate_adu = float(_sat_diag_ctx.sat_adu)
        from sat_diag import resolve_drift_ref_sky_deg  # noqa: PLC0415

        _frame_hint = Path(files[0]).name if files else None
        _ref_ra_deg, _ref_dec_deg, _drift_ref_catalog_id = resolve_drift_ref_sky_deg(
            ps, frame_name_hint=_frame_hint
        )
        if _ref_ra_deg is None and master_tab is not None and not getattr(master_tab, "empty", True):
            if "ra_deg" in master_tab.columns and "dec_deg" in master_tab.columns:
                _mra = pd.to_numeric(master_tab["ra_deg"], errors="coerce")
                _mde = pd.to_numeric(master_tab["dec_deg"], errors="coerce")
                _flux_col = "flux" if "flux" in master_tab.columns else None
                if _flux_col:
                    _ord = pd.to_numeric(master_tab[_flux_col], errors="coerce").fillna(0)
                    _j = int(_ord.idxmax())
                else:
                    _j = 0
                if math.isfinite(float(_mra.iloc[_j])) and math.isfinite(float(_mde.iloc[_j])):
                    _ref_ra_deg = float(_mra.iloc[_j])
                    _ref_dec_deg = float(_mde.iloc[_j])
        if _ref_ra_deg is not None:
            LOGGER.info(
                "[SAT-DIAG] drift reference sky (%.5f, %.5f) from %s",
                _ref_ra_deg,
                _ref_dec_deg,
                _frame_hint or "platesolve",
            )
        LOGGER.info(
            "[SAT-DIAG] sat_adu=%s source=%s lin_adu=%s (archive %s)",
            _sat_diag_ctx.sat_adu,
            _sat_diag_ctx.sat_source,
            _sat_diag_ctx.lin_adu,
            _sat_diag_archive,
        )
    return _drift_ref_catalog_id, _ref_dec_deg, _ref_ra_deg, _sat_diag_archive, _sat_diag_ctx, equipment_saturate_adu


def _ef_detection_setup(*, dao_fwhm_px, files, masterstar_fits, total, use_ram_inputs, work_ram):
    """Gaussian FWHM override (MASTERSTAR VY_FWHM), DAO FWHM and the worker count of the per-frame export."""
    _gauss_override: float | None = None
    try:
        if masterstar_fits is not None:
            _ms_gauss = Path(masterstar_fits)
            if _ms_gauss.is_file():
                with fits.open(_ms_gauss, memmap=False) as _gfh:
                    _ghdr = _gfh[0].header
                    # FWHM-ONE-2026-10: VY_FWHM is the elliptical Gaussian FWHM of frame QC
                    # (MASTERSTAR also carried a copy under an over-long name until 2026-10).
                    _gauss_override = fwhm_from_header_vy_fwhm(_ghdr)  # G03-29: the one reader
    except Exception:  # noqa: BLE001
        _gauss_override = None
    if _gauss_override is not None:
        log_event(
            f"[PHOT] gaussian_fwhm_px_override = {float(_gauss_override):.4f}px "
            "(MASTERSTAR VY_FWHM)"
        )
    else:
        log_event("[PHOT] gaussian_fwhm_px_override = None -> fallback to momentx0.619 per frame")

    _dao_fw_export = (
        float(dao_fwhm_px)
        if dao_fwhm_px is not None
        else DAO_FWHM_FALLBACK_PX
    )
    from parallel_frames import frame_bytes, stage_workers  # noqa: PLC0415

    _ny, _nx = _estimate_catalog_frame_hw(work_ram if use_ram_inputs else None, files)
    n_workers = stage_workers(total, frame_bytes(_ny, _nx), stage="per-frame catalogs")

    deferred_csv_writes: list[tuple[Path, pd.DataFrame]] = []
    return _dao_fw_export, _gauss_override, deferred_csv_writes, n_workers


def _ef_run_frames(
    *,
    _catalog_worker_state,
    _debug_logged_once,
    defer_disk_writes,
    deferred_csv_writes,
    files,
    n_workers,
    progress_cb,
    root,
    total,
    use_ram_inputs,
    work_ram,
    write_sidecar_csv_next_to_fits,
):
    """Run the per-frame catalogue workers (RAM handoff or disk) and collect their rows."""
    from parallel_frames import map_frames  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    ws = _catalog_worker_state()
    # EXPORT-RAM-PEAK-2026-10: a RAM-handoff frame is serialised only when its task is submitted.
    # Packing every frame up front held a second copy of all aligned frames next to the RAM
    # buffer (2 x the night in memory): Brno 534, 97 frames 3552x2664, was killed out of memory
    # on a 7 GB machine even with one worker.
    if use_ram_inputs and work_ram is not None:
        keys = [str(it[0]) for it in work_ram]
        _ram_items = work_ram

        def _pack(i: int) -> Any:
            _name, _hdr, _arr = _ram_items[i]
            d = np.ascontiguousarray(_arr, dtype=np.float32)
            return (str((root / _name).resolve()), pickle.dumps(_hdr.copy()), d.tobytes(), int(d.shape[0]), int(d.shape[1]))

        task_fn: Any = _export_per_frame_ram_worker_task
    else:
        keys = [fp.name for fp in files]
        _disk_packs = [str(fp.resolve()) for fp in files]

        def _pack(i: int) -> Any:
            return _disk_packs[i]

        task_fn = _export_per_frame_disk_worker_task

    by_key: dict[str, dict[str, Any]] = {}
    done = 0

    def _collect(key: str, res: dict[str, Any]) -> None:
        nonlocal _debug_logged_once, done
        if (not _debug_logged_once) and isinstance(res, dict) and res.get("debug_pixel_match") is not None:
            _debug_logged_once = True
            try:
                log_event(
                    "DEBUG: per-frame debug_pixel_match (first frame): "
                    + json.dumps(res.get("debug_pixel_match"), ensure_ascii=False, default=str)
                )
            except Exception:  # noqa: BLE001
                log_event(f"DEBUG: per-frame debug_pixel_match (first frame): {res.get('debug_pixel_match')}")
        for msg in res.get("infolog_messages") or []:
            if msg:
                log_event(str(msg))
        for p_str, dff in res.get("deferred_writes") or []:
            deferred_csv_writes.append((Path(p_str), dff))
        by_key[key] = res
        done += 1
        if progress_cb is not None:
            progress_cb(done, total, f"Catalog: {key}")

    # The pool keeps at most two frames per worker in flight, so the packed RAM frames are not
    # all alive at once; results come back in frame order.
    for key, res in zip(
        keys,
        map_frames(task_fn, range(len(keys)), workers=n_workers, initializer=_init_export_per_frame_worker,
                   ctx=ws, pack=_pack, stage="per-frame catalogs"),
        strict=True,
    ):
        _collect(key, res)
    rows_out = []
    frame_rows: dict[str, dict[str, Any]] = {}
    for key in keys:
        rr = dict(by_key[key])
        rr.pop("deferred_writes", None)
        rr.pop("infolog_messages", None)
        _fv = rr.pop("frame_values", None)
        if _fv and rr.get("csv"):
            frame_rows[str(rr["csv"])] = _fv
        rows_out.append(rr)
    if not defer_disk_writes and write_sidecar_csv_next_to_fits and frame_rows:
        from proc_frame_store import write_frame_tables  # noqa: PLC0415

        write_frame_tables(frame_rows)
    return frame_rows, rows_out


def _ef_finish(
    *,
    _gain_fe,
    _rn_fe,
    _sat_diag_archive,
    _sat_diag_ctx,
    defer_disk_writes,
    ps,
    root,
    rows_out,
    write_sidecar_csv_next_to_fits,
):
    """Write the per-frame index / frame tables, the hybrid background summary and commit the SAT-DIAG provenance."""
    index_path = ps / "per_frame_catalog_index.csv"
    if not defer_disk_writes:
        write_per_frame_index(rows_out, index_path)
    n_ok = sum(1 for r in rows_out if r.get("status") == "ok")
    _hybrid_stats: dict[str, Any] = {}
    if not defer_disk_writes:
        _hybrid_stats = _finalize_hybrid_bkg_fallback_sidecar(
            root,
            write_sidecar=bool(write_sidecar_csv_next_to_fits),
            gain=_gain_fe,
            read_noise=_rn_fe,
            setup_label=str(root.name),
        )
    if _sat_diag_ctx is not None and _sat_diag_archive:
        try:
            from sat_diag import commit_sat_diag_provenance  # noqa: PLC0415

            # FIX-RAWPEAKS-2026-10: the per-frame rows now carry raw_peaks_used (sat_diag.json said
            # false although every frame got its raw peaks: the flag was never in the rows)
            _placed_raw = any(bool(r.get("raw_peaks_used")) for r in rows_out)
            commit_sat_diag_provenance(
                _sat_diag_ctx,
                _sat_diag_archive,
                placed_aperture_used=_placed_raw,
            )
        except Exception as _sd_write_exc:  # noqa: BLE001
            LOGGER.warning("[SAT-DIAG] provenance commit skipped: %s", _sd_write_exc)
    return _hybrid_stats, index_path, n_ok


def export_per_frame_catalogs(
    *,
    frames_root: Path,
    platesolve_dir: Path,
    catalog_match_max_sep_arcsec: float = 25.0,
    dao_fwhm_px: float | None = None,
    write_sidecar_csv_next_to_fits: bool = True,
    progress_cb: "callable | None" = None,
    masterstars_csv: Path | str | None = None,
    masterstar_fits: Path | str | None = None,
    equipment_saturate_adu: float | None = None,
    aligned_ram: "Sequence[tuple[str, fits.Header, Any]] | None" = None,
    aligned_target_dir: Path | str | None = None,
    defer_disk_writes: bool = False,
    app_config: AppConfig,
    draft_id: int | None = None,
    equipment_id: int | None = None,
) -> dict[str, Any]:
    """For each FITS under ``frames_root`` with WCS: DAO + catalog table, write one CSV per frame.

    **MASTERSTAR lock (MASTER-LOCK-2026-10):** every frame is matched only to the MASTERSTAR catalogue
    (``masterstars_csv`` + ``masterstar_fits``, required): catalogue identities come from MASTERSTAR,
    positions and fluxes from the frame (DAO + forced photometry at the MASTERSTAR positions).

    ``<platesolve_dir>/per_frame_catalog_index.csv`` lists every file and CSV path.

    **Performance:** each frame still runs DAO + catalog match + disk CSV write (dominant cost for many lights).
    Parallelism: :func:`parallel_frames.stage_workers` / :func:`parallel_frames.map_frames` (the run's
    RAM and CPU budget, frames in order). RAM handoff ``aligned_ram`` sends serialized headers + float32
    pixels to the workers.

    **RAM handoff:** pass ``aligned_ram`` as ``(filename, header, ndarray)`` tuples plus ``aligned_target_dir`` to
    run catalog matching **without** re-reading aligned FITS from disk. With ``defer_disk_writes=True``, sidecar
    CSV (and optional flat mirror) are returned in ``deferred_csv_writes`` for the caller to flush after FITS.

    """
    _cfg_ap, _fwhm_night_median_px, _gain_fe, _qc_fwhm_by_name, _rn_fe, ps = _ef_setup(
        app_config=app_config,
        equipment_id=equipment_id,
        masterstar_fits=masterstar_fits,
        platesolve_dir=platesolve_dir,
    )

    use_ram_inputs = aligned_ram is not None
    if use_ram_inputs and aligned_target_dir is None:
        raise ValueError("export_per_frame_catalogs: aligned_target_dir is required when aligned_ram is set")

    if not write_sidecar_csv_next_to_fits and not defer_disk_writes:
        write_sidecar_csv_next_to_fits = True

    root = Path(aligned_target_dir) if use_ram_inputs else Path(frames_root)
    if ps is None:
        ps = Path(platesolve_dir)

    work_ram: list[tuple[str, fits.Header, Any]] | None = None
    if use_ram_inputs:
        root.mkdir(parents=True, exist_ok=True)
        work_ram = sorted(list(aligned_ram), key=lambda t: t[0])
        files = [root / name for name, _, _ in work_ram]
    else:
        files = sorted(_iter_fits_recursive(root))

    if not files:
        return {
            "written": 0,
            "per_frame_dir": str(root),
            "per_frame_csv_mode": "sidecar" if write_sidecar_csv_next_to_fits else "none",
            "index_csv": "",
            "frames": [],
            "deferred_csv_writes": [],
        }

    total = len(files)
    _debug_logged_once = False
    master_tab: pd.DataFrame | None = None
    _msp = Path(masterstars_csv) if masterstars_csv is not None else None
    _msf = Path(masterstar_fits) if masterstar_fits is not None else None
    if _msp is None or not _msp.is_file() or _msf is None or not _msf.is_file():
        raise RuntimeError(
            "Per-frame catalog: MASTERSTAR lock needs masterstars_full_match.csv and MASTERSTAR.fits "
            f"(got {masterstars_csv!s}, {masterstar_fits!s})."
        )
    _mt = pd.read_csv(_msp, low_memory=False, dtype={"catalog_id": str, "name": str})
    if _mt.empty or "ra_deg" not in _mt.columns or "dec_deg" not in _mt.columns:
        raise RuntimeError(f"Per-frame catalog: {_msp.name} has no ra_deg/dec_deg rows.")
    master_tab = _mt
    _drift_ref_catalog_id, _ref_dec_deg, _ref_ra_deg, _sat_diag_archive, _sat_diag_ctx, equipment_saturate_adu = _ef_sat_diag(
        _cfg_ap=_cfg_ap,
        _msp=_msp,
        equipment_id=equipment_id,
        equipment_saturate_adu=equipment_saturate_adu,
        files=files,
        master_tab=master_tab,
        ps=ps,
        work_ram=work_ram,
    )

    _dao_fw_export, _gauss_override, deferred_csv_writes, n_workers = _ef_detection_setup(
        dao_fwhm_px=dao_fwhm_px,
        files=files,
        masterstar_fits=masterstar_fits,
        total=total,
        use_ram_inputs=use_ram_inputs,
        work_ram=work_ram,
    )

    def _catalog_worker_state() -> dict[str, Any]:
        return {
            "master_tab": master_tab,
            "catalog_match_max_sep_arcsec": float(catalog_match_max_sep_arcsec),
            "dao_fwhm_px": float(_dao_fw_export),
            "equipment_saturate_adu": equipment_saturate_adu,
            "sat_diag_ctx_dict": (
                _sat_diag_ctx.to_json_dict() if _sat_diag_ctx is not None else None
            ),
            # raw-frame peaks (sat_diag.apply_raw_peaks_to_proc_df) read the draft archive through this key
            "sat_diag_archive": _sat_diag_archive,
            "sat_diag_ref_ra": _ref_ra_deg,
            "sat_diag_ref_dec": _ref_dec_deg,
            "sat_diag_ref_catalog_id": _drift_ref_catalog_id,
            "write_sidecar_csv_next_to_fits": write_sidecar_csv_next_to_fits,
            "defer_disk_writes": defer_disk_writes,
            # CONFIG-ONE-2026-10: the run config itself (pickled once per worker), not copies of keys
            "cfg": _cfg_ap,
            "draft_id": int(draft_id) if draft_id is not None else None,
            "gaussian_fwhm_px_override": _gauss_override,
            "fwhm_night_median_px": _fwhm_night_median_px,
            "qc_fwhm_by_name": dict(_qc_fwhm_by_name),
        }

    # PER-FRAME-ONE-PATH-2026-10 (Milan 2026-10-13): one per-frame catalog function for any
    # number of workers (``_export_per_frame_run_catalog_core``). The former one-worker copy
    # (_run_one_catalog) lacked the MASTERSTAR join (zone, bp_rp, phot_g_mean_mag, ...), the
    # forced photometry and the ePSF records -> on a 1-2 core machine comparison-star selection failed
    # ('numpy.float64' object has no attribute 'isna' in the comp pool admission).
    frame_rows, rows_out = _ef_run_frames(
        _catalog_worker_state=_catalog_worker_state,
        _debug_logged_once=_debug_logged_once,
        defer_disk_writes=defer_disk_writes,
        deferred_csv_writes=deferred_csv_writes,
        files=files,
        n_workers=n_workers,
        progress_cb=progress_cb,
        root=root,
        total=total,
        use_ram_inputs=use_ram_inputs,
        work_ram=work_ram,
        write_sidecar_csv_next_to_fits=write_sidecar_csv_next_to_fits,
    )

    _hybrid_stats, index_path, n_ok = _ef_finish(
        _gain_fe=_gain_fe,
        _rn_fe=_rn_fe,
        _sat_diag_archive=_sat_diag_archive,
        _sat_diag_ctx=_sat_diag_ctx,
        defer_disk_writes=defer_disk_writes,
        ps=ps,
        root=root,
        rows_out=rows_out,
        write_sidecar_csv_next_to_fits=write_sidecar_csv_next_to_fits,
    )
    return {
        "written": int(n_ok),
        "per_frame_dir": str(root),
        "per_frame_csv_mode": "sidecar" if write_sidecar_csv_next_to_fits else "none",
        "index_csv": str(index_path),
        "frames": rows_out,
        "deferred_csv_writes": list(deferred_csv_writes) if defer_disk_writes else [],
        "frame_rows": frame_rows,
        "hybrid_bkg_fallback": _hybrid_stats,
        "photometric_gain": _gain_fe,
        "photometric_read_noise": _rn_fe,
    }
