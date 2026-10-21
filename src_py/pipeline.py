"""Core processing pipeline for FITS observations."""

from __future__ import annotations

import logging
import multiprocessing
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Sequence

from astropy.io import fits
import pandas as pd

from config import AppConfig
from database import (
    VyvarDatabase,
)
from fits_suffixes import path_suffix_is_fits
from infolog import log_exception
from calibration import (
    filter_light_paths_for_calibration_db,
)
from cal_diag import (
    CalDiagSession,
    dark_np_for_cal_diag,
    gate_result_for_frame,
    is_obs_group_aborted,
    run_cal_diag_pregate,
)



from fits_meta import (
    _safe_filter_token,
    log_lights_binning_from_headers_preflight,
    generate_observation_hash,
    fits_metadata_from_primary_header,
    extract_fits_metadata,
)


LOGGER = logging.getLogger(__name__)

from pipeline_constants import N_STARS_DIAG_K, SAT_LIMIT_CONTAINER_CLIP_ADU, SAT_LIMIT_NO_KNEE_FRAC


from masterstar_gaia_accounting import _dao_xy_binned_to_full  # noqa: E402,F401


















def _analyze_calibrated_qc_one(src: Path) -> dict[str, Any]:
    import numpy as np

    src = Path(src)
    try:
        with fits.open(src, memmap=False) as hdul:
            hdr = hdul[0].header
            data = np.array(hdul[0].data, dtype=np.float32, copy=True)
        qc = _qc_fwhm_elongation(data)
        finite = np.isfinite(data)
        arr = data[finite]
        return {
            "src": str(src),
            "filter": _safe_filter_token(str(hdr.get("FILTER") or hdr.get("FILT") or "NoFilter")),
            "fwhm_px": qc.get("fwhm_px"),
            "elongation": qc.get("elongation"),
            "n_sources": qc.get("n_sources"),
            "n_stars_detected": qc.get("n_stars_detected"),
            "bg_median": float(np.nanmedian(arr)) if arr.size else None,
            "p50": float(np.nanpercentile(arr, 50)) if arr.size else None,
            "p99": float(np.nanpercentile(arr, 99)) if arr.size else None,
            "max": float(np.nanmax(arr)) if arr.size else None,
        }
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0439] [SILENT-DROP] analyze calibrated QC one-frame helper failed: %s', exc)
        return {"src": str(src), "status": f"error: {exc}"}


from cal_stage import _header_has_vy_skysf  # noqa: E402,F401  # re-export; survivor in cal_stage.py


def analyze_calibrated_qc(
    *,
    calibrated_root: Path,
    max_frames: int | None = None,
    progress_cb: Callable[..., None] | None = None,
    only_paths: Sequence[Path | str] | None = None,
) -> pd.DataFrame:
    """Analyze calibrated frames (QC) without writing /processed outputs.

    Runs FWHM/elongation QC on calibrated data in memory and returns a QC dataframe.

    If ``max_frames`` is None, every light FITS under ``calibrated_root`` is analyzed.

    Parallelism: jednotny pocet workerov (auto CPU/RAM alebo env, pozri :func:`_vyvar_parallel_worker_count`);
    integer ``>1`` uses a process pool by default (``VYVAR_PARALLEL_BACKEND=thread`` for threads).
    """
    calibrated_root = Path(calibrated_root)
    files = _filter_light_paths_maybe(_iter_light_fits(calibrated_root), only_paths)
    if max_frames is not None:
        files = files[: max(0, int(max_frames))]
    total = len(files)
    if total > 0:
        _mh = estimate_memory_from_fits_headers(files)
        _peak = int(float(_mh["bytes_float32_max_frame"]) * 6.0)
        LOGGER.info(
            "VYVAR QC analyze: %s frames; odhad spicky RAM ~%s (float32 + docasne polia)",
            total,
            format_memory_bytes(_peak),
        )
    n_workers = _vyvar_qc_preprocess_workers()
    if n_workers > 1 and total > 1:
        LOGGER.info(
            "VYVAR QC analyze: parallel_workers=%s (paralelne; ~%sx RAM na snimok oproti 1 vlaknu)",
            n_workers,
            n_workers,
        )

    if n_workers > 1 and total > 1:
        with _vyvar_parallel_pool(n_workers) as ex:
            futs = {
                ex.submit(_analyze_calibrated_qc_one, src): src
                for src in files
            }
            by_src: dict[Path, dict[str, Any]] = {}
            done = 0
            for fut in as_completed(futs):
                src = futs[fut]
                by_src[src] = fut.result()
                done += 1
                if progress_cb is not None:
                    progress_cb(done, total, f"Analyzing {src.name}")
            rows = [by_src[s] for s in files]
    else:
        rows = []
        for i, src in enumerate(files, start=1):
            if progress_cb is not None:
                progress_cb(i, total, f"Analyzing {src.name}")
            rows.append(_analyze_calibrated_qc_one(src))

    return pd.DataFrame(rows)


class AstroPipeline:
    """Skeleton for the modular variable-star processing workflow."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig()
        self.db = VyvarDatabase(self.config.database_path)
        self.db._archive_root_override = Path(self.config.archive_root)

    def calibrate(self, session_path: Path | str) -> Any:
        """Calibrate raw images and build nightly masters.

        Planned implementation:
        - library masters: mean stack (dark), median stack (flat); calibration re-use library
        """
        LOGGER.info("Calibrate requested for session: %s", session_path)
        return None

    def quick_calibrate_last_import(
        self,
        *,
        archive_path: Path | str,
        master_dark_path: Path | None,
        masterflat_by_filter: dict[str, Path | None],
        progress_cb: "callable | None" = None,
        equipment_id: int | None = None,
        draft_id: int | None = None,
        observation_id: str | None = None,
        masterflat_by_obs_key: dict[str, str | Path | None] | None = None,
        master_dark_by_obs_key: dict[str, str | Path | None] | None = None,
        roundness_reject_above: float | None = 1.25,
    ) -> dict[str, Any]:
        """Calibrate imported draft/raw lights into `calibrated/` under archive_path."""
        ap = Path(archive_path)
        # Accept both draft root (.../draft_xxx) and direct non_calibrated path (.../draft_xxx/non_calibrated).
        ap_root = ap.parent if ap.name.casefold() == "non_calibrated" else ap
        LOGGER.info("Kalibracia archivu: %s", ap)
        # Support Draft structure: <archive>/Raw/lights and optional <archive>/non_calibrated/lights.
        # Both sources are calibrated/passthrough-written into the single target: <archive>/calibrated/lights.
        outputs: dict[str, Any] = {"archive_path": str(ap_root), "results": {}, "perf10_qsum": {}}
        merged_perf10: dict[str, dict[str, Any]] = {}

        draft_lights = ap_root / "non_calibrated" / "lights"
        if not draft_lights.exists() and ap.name.casefold() == "non_calibrated":
            draft_lights = ap / "lights"
        raw_lights = ap_root / "Raw" / "lights"
        # In non_calibrated draft mode always prefer Draft/non_calibrated/lights.
        if draft_lights.exists():
            out_root = ap_root / "calibrated" / "lights"
            outputs["results"]["non_calibrated"] = calibrate_lights_to_calibrated(
                lights_root=draft_lights,
                calibrated_root=out_root,
                master_dark_path=master_dark_path,
                masterflat_by_filter=masterflat_by_filter,
                progress_cb=progress_cb,
                pipeline_config=self.config,
                equipment_id=equipment_id,
                draft_id=draft_id,
                observation_id=observation_id,
                masterflat_by_obs_key=masterflat_by_obs_key,
                master_dark_by_obs_key=master_dark_by_obs_key,
            )
        elif raw_lights.exists():
            out_root = ap_root / "calibrated" / "lights"
            outputs["results"]["Raw"] = calibrate_lights_to_calibrated(
                lights_root=raw_lights,
                calibrated_root=out_root,
                master_dark_path=master_dark_path,
                masterflat_by_filter=masterflat_by_filter,
                progress_cb=progress_cb,
                pipeline_config=self.config,
                equipment_id=equipment_id,
                draft_id=draft_id,
                observation_id=observation_id,
                masterflat_by_obs_key=masterflat_by_obs_key,
                master_dark_by_obs_key=master_dark_by_obs_key,
            )

        for _sec_stats in (outputs.get("results") or {}).values():
            if isinstance(_sec_stats, dict):
                _p10 = _sec_stats.get("perf10_qc_results")
                if isinstance(_p10, dict):
                    merged_perf10.update(_p10)

        if merged_perf10 and draft_id is not None:
            outputs["perf10_qsum"] = apply_perf10_dao_qc_to_obs_files(
                db=self.db,
                draft_id=int(draft_id),
                archive_path=ap_root,
                perf10_qc_results=merged_perf10,
                roundness_reject_above=roundness_reject_above,
            )

        LOGGER.info("Kalibracia dokoncena (sekcii vystupu: %s)", list((outputs.get("results") or {}).keys()))
        if equipment_id is not None:
            cal_root = ap_root / "calibrated" / "lights"
            if cal_root.is_dir():
                osc_out = run_osc_channel_extraction_for_archive(
                    calibrated_lights_root=cal_root,
                    db=self.db,
                    equipment_id=int(equipment_id),
                    app_config=self.config,
                    progress_cb=progress_cb,
                )
                outputs["osc_extraction"] = osc_out
        return outputs

    def calibrate_batch(
        self,
        *,
        light_paths: Sequence[Path | str],
        lights_root: Path | str,
        calibrated_root: Path | str,
        master_dark_path: Path | str | None,
        masterflat_by_filter: dict[str, Path | str | None],
        max_workers: int | None = None,
        progress_cb: "callable | None" = None,
        equipment_id: int | None = None,
        draft_id: int | None = None,
        observation_id: str | None = None,
    ) -> dict[str, Any]:
        """Apply master dark/flat to many lights using ``ProcessPoolExecutor`` (``spawn``).

        Output layout matches :func:`calibrate_lights_to_calibrated`: for each input file
        ``dst = calibrated_root / Path(light).relative_to(lights_root)``.

        Returns a dict with:

        - ``output_paths``: list aligned with ``light_paths`` - calibrated FITS path or ``None`` on failure
        - ``results``: list of per-file ``dict``\\ s from workers (``src``, ``dst``, ``ok``, ``error``)
        - ``stats``: processed / ok / failed counts
        """
        import numpy as np

        lr = Path(lights_root).resolve()
        cr = Path(calibrated_root)
        cr.mkdir(parents=True, exist_ok=True)

        light_paths = filter_light_paths_for_calibration_db(
            [Path(lp) for lp in light_paths],
            database_path=self.config.database_path,
            draft_id=draft_id,
            observation_id=observation_id,
        )

        mf_paths: dict[str, Path | None] = {}
        for k, v in (masterflat_by_filter or {}).items():
            if v is None or str(v).strip() == "":
                mf_paths[str(k)] = None
            else:
                mf_paths[str(k)] = Path(v)

        mf_serial: dict[str, str | None] = {
            k: str(p.resolve()) if p is not None else None for k, p in mf_paths.items()
        }

        _md_log = Path(master_dark_path) if master_dark_path is not None else None
        _log_calibration_io_preflight(
            calibrated_root=cr,
            master_dark_path=_md_log,
            masterflat_by_filter=mf_paths,
        )

        md_init: str | None = None
        if master_dark_path is not None:
            md_p = Path(master_dark_path)
            if md_p.is_file():
                md_init = str(md_p.resolve())

        qc_pack = _qc_pack_from_config(
            self.config, draft_id=draft_id, observation_id=observation_id
        )

        items: list[
            tuple[
                str,
                str,
                str | None,
                dict[str, str | None],
                dict[str, Any] | None,
            ]
        ] = []
        for lp in light_paths:
            src_p = Path(lp).resolve()
            rel = src_p.relative_to(lr)
            dst_p = (cr / rel).resolve()
            items.append((str(src_p), str(dst_p), md_init, mf_serial, qc_pack))

        n = len(items)
        if n == 0:
            return {
                "output_paths": [],
                "results": [],
                "stats": {"n_input": 0, "ok": 0, "failed": 0},
            }

        nw = (
            max_workers
            if max_workers is not None
            else max(1, min(32, int(self.config.qc_preprocess_workers)))
        )
        nw = max(1, min(int(nw), n))
        if not _vyvar_calibrate_multiprocessing_enabled():
            nw = 1

        _native_b = None  # CAL-2026-10: master binning from each master header

        cal_diag_session = CalDiagSession()
        db_cal = _db_for_calibration_tasks(qc_pack)
        _lpaths = [Path(it[0]) for it in items]
        if _lpaths and md_init:
            cal_diag_session = run_cal_diag_pregate(
                _lpaths,
                obs_group_key_from_path=_obs_group_key_from_light_path,
                resolve_dark_path=lambda fp, og, lb: Path(md_init) if md_init else None,
                light_binning_from_path=_light_binning_from_path,
                master_binning=_native_b,
                match_and_crop_pair=_match_and_crop_pair,
                saturation_for_light=lambda fp: _saturation_adu_for_cal_diag(
                    fits.getheader(fp, 0),
                    db=db_cal,
                    equipment_id=equipment_id,
                ),
                ui_error=_pipeline_ui_error,
            )
        cal_diag_worker_blob = _cal_diag_export_for_workers(cal_diag_session)

        rows: list[dict[str, Any]]
        if nw <= 1:
            md_pre: Any = None
            if md_init:
                with fits.open(md_init, memmap=False) as h:
                    md_pre = np.array(h[0].data, dtype=np.float32, copy=True)
            flat_cache: dict[str, Any] = {}
            flat_med: dict[str, float] = {}
            db_main = _db_for_calibration_tasks(qc_pack)
            rows = []
            for i, it in enumerate(items):
                src_s, dst_s, md_s, mf_map, _qopt = it
                src_p = Path(src_s)
                dst_p = Path(dst_s)
                try:
                    _ok = _obs_group_key_from_light_path(src_p)
                    if is_obs_group_aborted(cal_diag_session, _ok):
                        if dst_p.exists():
                            dst_p.unlink(missing_ok=True)
                        rows.append(
                            {
                                "src": src_s,
                                "dst": dst_s,
                                "ok": True,
                                "skipped": True,
                                "error": None,
                                "qc_summary": None,
                                "traceback": None,
                            }
                        )
                        if progress_cb is not None:
                            progress_cb(i + 1, n, f"CAL-DIAG skip {src_p.name}")
                        continue
                    light_bx = _light_binning_from_path(src_p)
                    gr = gate_result_for_frame(
                        cal_diag_session,
                        obs_group_key=_ok,
                        dark_path=Path(md_s) if md_s else None,
                        light_binning=light_bx,
                    )
                    md_np = md_pre
                    if md_s:
                        with fits.open(src_p, memmap=False) as hdul:
                            lshape = (int(hdul[0].data.shape[0]), int(hdul[0].data.shape[1]))
                        md_np = dark_np_for_cal_diag(
                            cal_diag_session,
                            master_binning=_native_b,
                            dark_path=Path(md_s),
                            light_binning=light_bx,
                            light_shape=lshape,
                            light_filename=src_p.name,
                            gate_result=gr,
                        )
                    mf = {str(k): Path(v) if v else None for k, v in mf_map.items()}
                    _ud, _uf, qc_sum, _cf, _p10 = _calibrate_one_light_disk(
                        src=src_p,
                        dst=dst_p,
                        master_dark_path=Path(md_s) if md_s else None,
                        masterflat_by_filter=mf,
                        flat_cache=flat_cache,
                        flat_median_scale=flat_med,
                        md_data_preload=md_np,
                        db=db_main,
                        qc_pack=_qopt,
                        calibration_master_native_binning=_native_b,
                        cal_diag_gate_result=gr,
                    )
                    rows.append(
                        {
                            "src": src_s,
                            "dst": dst_s,
                            "ok": True,
                            "error": None,
                            "qc_summary": qc_sum,
                            "traceback": None,
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    _tb_cb = traceback.format_exc()
                    LOGGER.error("calibrate_batch: %s -> %s\n%s", src_s, exc, _tb_cb)
                    log_exception(f"CHYBA WORKERA: {Path(src_s).name}", exc)
                    rows.append(
                        {
                            "src": src_s,
                            "dst": None,
                            "ok": False,
                            "error": str(exc),
                            "traceback": _tb_cb,
                        }
                    )
                if progress_cb is not None:
                    progress_cb(i + 1, n, f"Calibrating {Path(src_s).name}")
        else:
            ctx = multiprocessing.get_context("spawn")
            rows = [None] * n  # type: ignore[misc]
            with ProcessPoolExecutor(
                max_workers=nw,
                mp_context=ctx,
                initializer=_init_calibrate_batch_worker,
                initargs=(_native_b, cal_diag_worker_blob),
            ) as ex:
                future_map = {ex.submit(_calibrate_batch_process_one, it): idx for idx, it in enumerate(items)}
                done = 0
                for fut in as_completed(future_map):
                    idx = future_map[fut]
                    rows[idx] = fut.result()
                    done += 1
                    if progress_cb is not None:
                        progress_cb(done, n, f"Calibrating batch {done}/{n}")

        out_paths: list[str | None] = []
        ok_c = 0
        fail_c = 0
        for r in rows:
            if r.get("ok"):
                ok_c += 1
                out_paths.append(str(r["dst"]) if r.get("dst") else None)
            else:
                fail_c += 1
                out_paths.append(None)

        return {
            "output_paths": out_paths,
            "results": rows,
            "stats": {
                "n_input": n,
                "ok": ok_c,
                "failed": fail_c,
                "max_workers": nw,
            },
        }

    def quick_preprocess_last_import(
        self,
        *,
        archive_path: Path | str,
        run: bool = True,
        reject_fwhm_px: float | None = None,
        reject_elongation: float | None = None,
        use_gpu_if_available: bool = False,
    ) -> dict[str, Any]:
        ap = Path(archive_path)
        ap_root = ap.parent if ap.name.casefold() == "non_calibrated" else ap
        src_cal = ap_root / "calibrated" / "lights"
        src_noncal = ap_root / "non_calibrated" / "lights"
        source_dir = src_cal if src_cal.exists() else src_noncal
        out: dict[str, Any] = {
            "archive_path": str(ap_root),
            "processed": {},
            "qc_suggestions": {},
            "checked": {
                "archive_exists": bool(ap_root.exists()),
                "source_path": str(source_dir),
                "calibrated_path": str(src_cal),
                "non_calibrated_path": str(src_noncal),
                "processed_path": str(ap_root / "processed" / "lights"),
                "detrended_path": str(ap_root / "detrended" / "lights"),
            },
        }

        if source_dir.exists():
            if run:
                df = preprocess_calibrated_to_processed(
                    calibrated_root=source_dir,
                    progress_cb=None,
                    app_config=self.config,
                )
            else:
                qc_csv_existing = source_dir / "qc_metrics.csv"
                if not qc_csv_existing.exists():
                    qc_csv_existing = ap_root / "detrended" / "lights" / "qc_metrics.csv"
                try:
                    df = pd.read_csv(qc_csv_existing) if qc_csv_existing.exists() else pd.DataFrame()
                except Exception:  # noqa: BLE001
                    df = pd.DataFrame()
            out["processed"]["source"] = {
                "lights_root": str(source_dir),
                "rows": int(len(df)),
                "rejected": int((df["status"].astype(str).str.startswith("rejected")).sum()) if not df.empty else 0,
                "source_dir": str(source_dir),
            }
            try:
                qc_csv = source_dir / "qc_metrics.csv"
                out["processed"]["source"]["qc_csv"] = str(qc_csv)
            except Exception as exc:  # noqa: BLE001
                LOGGER.debug("[PIPELINE] Cleanup step failed (non-critical): %s", exc)
            out["qc_suggestions"]["source"] = _qc_suggest_thresholds(df)

        if not out["processed"]:
            out["warning"] = "No calibrated lights found under this archive_path. Run calibration first (Run quick pipeline step)."
        return out

    def quick_analyze_last_import(
        self,
        *,
        archive_path: Path | str,
        max_frames: int | None = None,
    ) -> dict[str, Any]:
        ap = Path(archive_path)
        ap_root = ap.parent if ap.name.casefold() == "non_calibrated" else ap
        out: dict[str, Any] = {
            "archive_path": str(ap_root),
            "analysis": {},
            "qc_suggestions": {},
        }
        cal = ap_root / "calibrated" / "lights"
        if cal.exists():
            df = analyze_calibrated_qc(
                calibrated_root=cal,
                max_frames=max_frames,
            )
            out["analysis"]["calibrated"] = {"rows": int(len(df))}
            out["qc_suggestions"]["calibrated"] = _qc_suggest_thresholds(df)
            try:
                qc_csv = ap_root / "calibrated" / "lights" / "qc_analysis.csv"
                df.to_csv(qc_csv, index=False)
                out["analysis"]["calibrated"]["qc_csv"] = str(qc_csv)
            except Exception as exc:  # noqa: BLE001
                logging.error('[EXC-0444] `fetch_draft_light_rows_for_quality` failure logs WARNING and returns empty jump result...: %s', exc)
                LOGGER.debug("[PIPELINE] Cleanup step failed (non-critical): %s", exc)
        else:
            out["warning"] = "No calibrated lights found. Run calibration first."
        return out

    @staticmethod
    def _first_fits_file(session_path: Path | str) -> Path:
        session = Path(session_path)
        search_roots = [session, session / "Raw"]
        for root in search_roots:
            if not root.exists():
                continue
            for fp in sorted(root.rglob("*")):
                if fp.is_file() and path_suffix_is_fits(fp):
                    return fp
        raise FileNotFoundError(f"No FITS file found in session path: {session}")

    def prepare_observation_from_session(
        self,
        session_path: Path | str,
        *,
        id_equipment: int | None = None,
        id_telescope: int | None = None,
        id_location: int = 1,
    ) -> dict[str, Any]:
        """Read first FITS and build prefilled OBSERVATION payload."""
        first_fits = self._first_fits_file(session_path)
        metadata = extract_fits_metadata(first_fits, db=self.db)
        scanning_id = self.db.derive_scanning_id(metadata)

        if id_equipment is None or id_telescope is None:
            raise ValueError(
                "prepare_observation_from_session: id_equipment a id_telescope su povinne "
                "(vyberte kameru a dalekohlad v Session Upload)."
            )
        equipment_id = int(id_equipment)
        telescope_id = int(id_telescope)

        observation_payload = {
            "id_equipments": equipment_id,
            "id_telescope": telescope_id,
            "id_location": int(id_location),
            "id_scanning": scanning_id,
            "center_of_field_ra": float(metadata["ra"]),
            "center_of_field_de": float(metadata["dec"]),
            "observation_start_jd": float(metadata["jd_start"]),
        }
        return {
            "fits_file": str(first_fits),
            "metadata": metadata,
            "observation_payload": observation_payload,
            "missing_telescope": not bool(metadata.get("telescope")),
            "missing_camera": not bool(metadata.get("camera")),
        }

from pipeline_calibrate import (
    SkySurfaceOrderConflictError,
    _cal_diag_export_for_workers,
    _cal_diag_session_from_export,
    _calibrate_batch_process_one,
    _calibrate_one_light_disk,
    _db_for_calibration_tasks,
    _decide_preprocess_sky_action,
    _effective_saturation_limit,
    _filter_light_paths_maybe,
    _fit_subtract_preprocess_sky_surface,
    _has_valid_wcs,
    _infer_sat_limit_from_bitpix,
    _init_calibrate_batch_worker,
    _iter_light_fits,
    _light_binning_from_path,
    _log_calibration_io_preflight,
    _match_and_crop_pair,
    _obs_group_key_from_light_path,
    _pipeline_ui_error,
    _qc_enrich_calibrated_in_place,
    _qc_enrich_one_frame,
    _qc_fwhm_elongation,
    _qc_pack_from_config,
    _saturate_limit_adu_from_header,
    _saturation_adu_for_cal_diag,
    _sync_obs_calibration_state_with_retry,
    _vyvar_calibrate_multiprocessing_enabled,
    _vyvar_parallel_pool,
    _vyvar_qc_preprocess_workers,
    apply_perf10_dao_qc_to_obs_files,
    calibrate_lights_to_calibrated,
    estimate_archive_memory_profile,
    estimate_memory_from_fits_headers,
    format_memory_bytes,
    norm_fits_path_key,
    run_draft_ram_calibration_qc_to_obs_files,
    run_osc_channel_extraction_for_archive,
    scan_calibrated_lights_pointing,
)
from pipeline_preprocess import (
    _archive_raw_to_calibrated_light,
    _qc_suggest_thresholds,
    build_prefilter_rejected_map,
    calibrated_paths_for_draft_apply_filters,
    filter_files_by_qc_metrics_allowlist,
    load_qc_metrics_status_by_path,
    preprocess_calibrated_to_processed,
    qc_enrich_calibrated_lights_in_place,
    resolve_obs_file_to_processed_fits,
    resolve_preprocess_target_coordinates,
)


from pipeline_astrometry import (
    _dao_targeted_pass2_unmatched_gaia,
    _equipment_saturate_adu_from_db,
    _merge_vsx_exoplanet_variable_targets,
    _pass2_sibling_wcs_recovery,
    _path_segments_forbidden_for_masterstar_physical_source,
    _plate_solve_input_bundle,
    _query_vsx_local_frame_bbox,
    _resolve_best_effort_path_under,
    _vyvar_df_to_csv,
    astrometry_align_and_build_masterstar,
    draft_is_multi_group_obs,
    draft_obs_group_count,
    get_masterstar_candidates,
    resolve_masterstar_input_root,
    select_comparison_stars_spatial_grid,
    write_photometry_plan_files,
)
from pipeline_catalog import (
    _EXPORT_PER_FRAME_WORKER_STATE,
    _MASTERSTAR_ZONE_LOG_ONCE,
    _annotate_masterstars_flux_zones,
    _apply_dao_centroid_wcs_guard,
    _box_peak_max_adu,
    _build_exoplanet_promotion_rows_from_masterstars,
    _cfg_from_export_worker_state,
    _compute_airmass_from_altaz,
    _dao_auto_binning_factor,
    _dao_convolved_background_rms_adu,
    _dao_detection_threshold_adu,
    _dao_noise_sigma_adu,
    _detect_empirical_clip_level_adu,
    _effective_field_catalog_cone_radius_deg,
    _exo_host_annotation_arrays,
    _extract_airmass_from_header,
    _finalize_hybrid_bkg_fallback_sidecar,
    _init_export_per_frame_worker,
    _lock_matched_centroids_to_master_grid,
    _masterstar_zone_linear_threshold,
    _mean_bin2d_for_dao,
    _pixel_noise_sigma_pp_adu,
    _prefilter_dao_table_brightest,
    _proc_deduplicate_matched_catalog_rows,
    _query_exoplanet_local,
    _query_gaia_local,
    _query_vsx_local,
    _resolve_peak_saturation_limit_adu,
    detect_stars_match_master_reference,
    find_qc_metrics_csv,
    inv_sat_limit_peak_test_adu,
)

from catalog_match import detect_stars_and_match_catalog  # noqa: E402,F401
from frame_export import export_per_frame_catalogs  # noqa: E402,F401
from masterstar_build import generate_masterstar_and_catalog  # noqa: E402,F401

from pipeline_ui_helpers import (  # noqa: E402,F401
    _resolve_light_fits_for_quality_inspection,
    resolve_masterstars_metadata_csv,
)


