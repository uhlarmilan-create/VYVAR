"""Moved from photometry_core.py (CONSOLIDATE-01E3). Facade re-exports these names."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import logging
import math
import time
from astropy.io import fits as astrofits
import numpy as np
import pandas as pd
from config import (
    AppConfig,
    classify_field_density,
    compute_field_density,
)
from catalog_db import query_local_gaia
from gaia_catalog_id import normalize_gaia_source_id
from infolog import log_event
from photometry_comp import (
    _batch_enrich_targets_bp_rp_from_gaia_db,
    _enrich_target_bp_rp_from_gaia_db,
    _normalize_id_series,
    _normalize_id_value,
    _target_selection_effective_frame_hw_px,
    _read_field_density_inputs,
    _refresh_variable_targets_xy,
    _resolve_frame_hw_px_from_masterstar,
    _write_suspected_variables,
    build_global_comp_pool,
    select_active_targets,
    select_comparison_stars_per_target,
)
import photometry_comp as _photometry_comp
from photometry_provenance import merge_photometry_pipeline_meta
from photometry_shared import _angular_distance_deg, build_gs11_summary
from proc_frame_store import PROC_CSV_GLOB, PROC_STORE_COLS, ProcFrameStore

from photometry_core import (
    _GAIA_ID_DTYPE,
)


from comp_smart import N_RESERVE as _N_RESERVE  # noqa: E402

#: COMP-SMART-2026-10: reserve comparison candidates per target (aperture-photometry replacements)


def _comp_safe_bbox(
    variable_targets_csv: Path,
    csv_paths: list[Path],
    cfg: Any,
    safe_bbox: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float] | None:
    """safe_bbox re-shrunk by the night's largest sky-annulus radius (see COMP-EDGE-2026-10).
    Raw (alignment-intersection) bbox = safe_bbox + safe_bbox_r_out_px of photometry_plan.json;
    per-frame FWHM = VY_FWHM of the aligned frames (the value aperture photometry sizes the annulus with)."""
    if safe_bbox is None:
        return None
    try:
        import json as _json  # noqa: PLC0415

        from astropy.io import fits as _fits  # noqa: PLC0415

        plan = _json.loads((Path(variable_targets_csv).parent / "photometry_plan.json").read_text(encoding="utf-8"))
        r0 = float(plan.get("safe_bbox_r_out_px") or float("nan"))
        if not math.isfinite(r0):
            return safe_bbox
        from aperture_pertarget import frame_stem_from_source  # noqa: PLC0415
        from aperture_policy import clamp_fwhm_px  # noqa: PLC0415

        fws: list[float] = []
        for cp in csv_paths:
            fp = Path(cp).with_name(frame_stem_from_source(Path(cp).name) + ".fits")
            if fp.is_file():
                # same clamp as the aperture-photometry grid (measure_night_grid / _fwhm_from_fits_header)
                v = clamp_fwhm_px(_fits.getheader(fp).get("VY_FWHM"))
                if v is not None:
                    fws.append(float(v))
        if not fws:
            return safe_bbox
        r_max = float(getattr(cfg, "annulus_outer_fwhm", 5.2)) * max(fws)
        if r_max <= r0:
            return safe_bbox
        d = r_max - r0
        x0, y0, x1, y1 = safe_bbox
        sb = (x0 + d, y0 + d, x1 - d, y1 - d)
        if sb[2] <= sb[0] or sb[3] <= sb[1]:
            logging.warning("[COMP-EDGE] comp bbox empty after shrinking by %.1f px - safe_bbox used", d)
            return safe_bbox
        logging.info("[COMP-EDGE] comp bbox shrunk by %.1f px (r_out max %.1f px at FWHM %.2f vs %.1f px)",
                     d, r_max, max(fws), r0)
        return sb
    except Exception as exc:  # noqa: BLE001
        logging.warning("[COMP-EDGE] comp bbox not derived (%s) - safe_bbox used", exc)
        return safe_bbox


def _split_comp_reserve(comps: pd.DataFrame, ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """COMP-SMART-2026-10: rows marked ``comp_role == "reserve"`` (separate, larger selection,
    see photometry_comp._append_reserve_rows) are the reserve; the rest are the admitted comps
    of the unchanged ``n_comp_max`` selection."""
    comps = comps.reset_index(drop=True)
    if "comp_role" not in comps.columns:
        return comps, comps.iloc[0:0].copy()
    is_res = comps["comp_role"].astype(str).str.strip().eq("reserve")
    prim = comps.loc[~is_res].drop(columns=["comp_role", "reserve_rank"], errors="ignore").copy()
    resv = comps.loc[is_res].copy()
    return prim.reset_index(drop=True), resv.reset_index(drop=True)


def run_target_and_comp_selection(
    variable_targets_csv: Path,
    masterstars_csv: Path,
    per_frame_csv_dir: Path,
    output_dir: Path,
    *,
    fwhm_px: float = 3.7,
    frame_w_px: int = 2082,
    frame_h_px: int = 1397,
    plate_scale_arcsec_px: float | None = None,
    n_comp_min: int = 3,
    n_comp_max: int = 7,
    max_comp_rms: float = 0.1,
    min_dist_arcsec: float = 60.0,
    min_frames_frac: float = 0.3,
    flux_col: str = "dao_flux",
    comp_max_delta_bprp: float = 0.5,
    cfg: AppConfig | None = None,
    progress_cb: Any = None,
    draft_id: int | None = None,
    db: Any = None,
) -> dict[str, Any]:
    """Run target selection + comparison-star selection and save the outputs.

    Outputs (saved to output_dir):
      active_targets.csv              - VSX targets + ``zone_flag`` / ``skip_photometry`` (saturated)
      comparison_stars_per_target.csv - comparison stars for each target
      suspected_variables.csv         - candidate new variables (high RMS, not in VSX)

    Returns:
        dict with keys:
          n_active_targets, n_comparison_pairs,
          active_targets_csv, comparison_stars_csv, suspected_variables_csv,
          targets_without_comps (list catalog_id)

    Edge rule: a star is used when its sky annulus fits on the chip - inside the annulus-aware
    safe box of photometry_plan.json (alignment intersection shrunk by r_out), else at least
    annulus_outer_fwhm x MASTERSTAR FWHM from the frame edge.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # MASTERSTAR FWHM (VY_FWHM): used for the D3 astrometric gate. Neighbour geometry
    # (isolation, separation) uses the group night median (NIGHT-FWHM-2026-10, below).
    _ms_fits_fwhm = Path(variable_targets_csv).resolve().parent / "MASTERSTAR.fits"
    try:
        if _ms_fits_fwhm.is_file():
            from astropy.io import fits as _astrofits_fwhm  # noqa: PLC0415

            with _astrofits_fwhm.open(_ms_fits_fwhm, memmap=False) as _hdul_fw:
                _hdr_fw = _hdul_fw[0].header
            for _k_fw in ("VY_FWHM", "VY_FWHM_GAUSS", "VY_FWHM_GAUSSIAN"):
                _v_fw = _hdr_fw.get(_k_fw)
                if _v_fw is None:
                    continue
                _fv_fw = float(_v_fw)
                if 0.5 < _fv_fw < 30.0:
                    fwhm_px = _fv_fw
                    break
    except Exception as exc:  # noqa: BLE001
        logging.error(
            "[EXC-FWHM-P01] MASTERSTAR FWHM resolve failed - keeping fwhm_px=%.3f: %s",
            float(fwhm_px),
            exc,
        )
    logging.info(
        "[TARGET-SELECT] MASTERSTAR fwhm_px=%.3f (D3 astrometric gate)",
        float(fwhm_px),
    )

    def _p(msg: str) -> None:
        if progress_cb is None:
            return
        try:
            progress_cb(str(msg))
        except UnicodeEncodeError:
            # Some Windows consoles use legacy encodings (cp1252) and crash on diacritics.
            # Use ASCII escapes so printing never raises again.
            progress_cb(str(msg).encode("ascii", "backslashreplace").decode("ascii"))

    _cfg_base = cfg if cfg is not None else AppConfig()
    # CODE-REVIEW step 7B: the edge margin is the sky-annulus radius (was a fixed 50 px key).
    chip_interior_margin_px = int(math.ceil(float(_cfg_base.annulus_outer_fwhm) * float(fwhm_px)))
    _ms_density = Path(variable_targets_csv).resolve().parent / "MASTERSTAR.fits"
    _fw_in, _fh_in = int(frame_w_px), int(frame_h_px)
    frame_w_px, frame_h_px, _frame_hw_src = _resolve_frame_hw_px_from_masterstar(
        _ms_density,
        frame_w_px=_fw_in,
        frame_h_px=_fh_in,
        db=db,
        draft_id=draft_id,
    )
    if _frame_hw_src != "caller_default":
        logging.info(
            "[TARGET-SELECT] Frame dimensions %dx%d px from %s (caller default %dx%d)",
            int(frame_w_px),
            int(frame_h_px),
            _frame_hw_src,
            _fw_in,
            _fh_in,
        )
    _n_field, _cw_fd, _ch_fd, _nsrc_fd, _vy_ndao_raw = _read_field_density_inputs(
        _ms_density,
        Path(masterstars_csv),
        int(frame_w_px),
        int(frame_h_px),
    )
    _density = compute_field_density(_n_field, _cw_fd, _ch_fd)
    _d_class = classify_field_density(float(_density), 300.0, 1000.0)  # diagnostic label only
    logging.info(
        "[FIELD DENSITY] %.0f stars/Mpx -> class: %s (n_stars=%d, chip=%dx%dpx, n_src=%s)",
        float(_density),
        _d_class,
        int(_n_field),
        int(_cw_fd),
        int(_ch_fd),
        _nsrc_fd,
    )
    # COMP-RULES-2026-10: the comparison-star rules are the same for every field.
    # Field density is a diagnostic only; recorded in pipeline_meta["field_density"]
    # (OUTPUT-REVIEW-2026-10: no field_density.json).
    merge_photometry_pipeline_meta(
        output_dir,
        {
            "field_density": {
                "density_h_star_per_mpx": round(float(_density), 4),
                "density_class": _d_class,
                "n_stars": int(_n_field),
                "n_stars_dao_raw": int(_vy_ndao_raw)
                if _vy_ndao_raw is not None and int(_vy_ndao_raw) > 0
                else None,
                "n_stars_source": _nsrc_fd,
                "chip_w_px": int(_cw_fd),
                "chip_h_px": int(_ch_fd),
            }
        },
    )

    _wcs_scale_ok = True
    # PLATE-SCALE-2026-10: the plate scale of the solved MASTERSTAR (photometry_core); no config
    # value (a fixed 1.3"/px was silently wrong on every rig)
    if not (plate_scale_arcsec_px is not None and math.isfinite(float(plate_scale_arcsec_px))
            and float(plate_scale_arcsec_px) > 0):
        raise ValueError("Target and comparison-star selection: plate scale not derivable from the MASTERSTAR WCS")
    _ps_p01 = float(plate_scale_arcsec_px)
    _expected_scale = _ps_p01
    if _ms_density.is_file():
        try:
            from astropy.wcs import WCS as _WCS_check  # noqa: PLC0415

            with astrofits.open(_ms_density, memmap=False) as _hdul_wcs:
                _wcs_check = _WCS_check(_hdul_wcs[0].header)
            _psm_chk = np.asarray(_wcs_check.pixel_scale_matrix, dtype=np.float64)
            _actual_scale = float(np.sqrt(np.abs(np.linalg.det(_psm_chk))) * 3600.0)
            _scale_ratio = abs(_actual_scale - _expected_scale) / max(_expected_scale, 1e-9)
            if _scale_ratio > 0.20:
                log_event(
                    f"[WCS SANITY] Scale {_actual_scale:.3f}\"/px deviates "
                    f"{_scale_ratio * 100.0:.1f}% from expected "
                    f"{_expected_scale:.3f}\"/px - using pixel-distance fallback"
                )
                _wcs_scale_ok = False
        except Exception as _wcs_exc:  # noqa: BLE001
            logging.error('[EXC-0210] WCS scale sanity exception assumes scale OK - comp matching uses ra/dec haversine when ...: %s', _wcs_exc)
            logging.warning(
                "[WCS SANITY] check failed (non-fatal): %s - skipping check, assuming WCS scale OK "
                "(radec-haversine distance mode).",
                _wcs_exc,
            )
    log_event(
        f"[COMP SELECT] Distance mode: "
        f"{'pixel-fallback' if not _wcs_scale_ok else 'radec-haversine'}"
    )

    # -- target selection --
    _p("Target selection: active targets from VSX...")
    logging.info("[TARGET-SELECT] Selecting active targets...")
    _cfg_p01 = _cfg_base
    # Load annulus-aware safe bbox from photometry_plan.json (if available).
    _safe_bbox: tuple[float, float, float, float] | None = None
    try:
        plan_path = Path(variable_targets_csv).parent / "photometry_plan.json"
        if plan_path.is_file():
            import json as _json  # noqa: PLC0415

            _plan = _json.loads(plan_path.read_text(encoding="utf-8"))
            sb = _plan.get("safe_bbox_px")
            if isinstance(sb, (list, tuple)) and len(sb) == 4:
                x0b, y0b, x1b, y1b = sb
                _safe_bbox = (float(x0b), float(y0b), float(x1b), float(y1b))
    except Exception:  # noqa: BLE001
        _safe_bbox = None
    _ms_fits_p01 = Path(variable_targets_csv).resolve().parent / "MASTERSTAR.fits"
    _masterstar_wcs: Any = None
    if _ms_fits_p01.is_file():
        try:
            import warnings

            from astropy.wcs import FITSFixedWarning, WCS

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FITSFixedWarning)
                with astrofits.open(_ms_fits_p01, memmap=False) as hdul:
                    _masterstar_wcs = WCS(hdul[0].header)
        except Exception as exc:  # noqa: BLE001
            logging.warning("[VT REFRESH] could not load MASTERSTAR WCS: %s - x/y in variable_targets.csv unchanged", exc)
            _masterstar_wcs = None

    _vt_p01 = Path(variable_targets_csv)
    if _masterstar_wcs is not None and _vt_p01.is_file():
        _refresh_variable_targets_xy(
            variable_targets_csv=_vt_p01,
            wcs=_masterstar_wcs,
            chip_w=int(frame_w_px),
            chip_h=int(frame_h_px),
        )

    active = select_active_targets(
        variable_targets_csv,
        masterstars_csv,
        frame_w_px=frame_w_px,
        frame_h_px=frame_h_px,
        edge_margin_px=int(chip_interior_margin_px),
        safe_bbox=_safe_bbox,
        gaia_db_path=str(_cfg_p01.gaia_db_path or ""),
        cfg=_cfg_p01,
    )
    active_csv = output_dir / "active_targets.csv"
    try:
        from gaia_catalog_id import normalize_gaia_source_id_series  # noqa: PLC0415

        if "catalog_id" in active.columns:
            active = active.copy()
            active["catalog_id"] = normalize_gaia_source_id_series(active["catalog_id"])
    except Exception as exc:  # noqa: BLE001
        # EXC-0212: T3 -- ProcFrameStore not stored in Streamlit session_state - UI perf cache miss only (EXCEPT-BULK-2 2026-07-08)
        logging.error('[EXC-0211] active_targets.csv catalog_id normalization fails - float-truncated IDs written to disk: %s', exc)
        pass
    active.to_csv(active_csv, index=False)
    logging.info(f"[TARGET-SELECT] Saved: {active_csv} ({len(active)} targets)")
    _excluded = _photometry_comp.LAST_EXCLUDED_TARGETS
    if _excluded is not None and not _excluded.empty:
        excluded_csv = output_dir / "excluded_targets.csv"
        _excluded.to_csv(excluded_csv, index=False)
        logging.info(f"[TARGET-SELECT] Saved: {excluded_csv} ({len(_excluded)} excluded)")
    _p(f"Target selection done: {len(active)} active targets")

    if active.empty:
        return {
            "n_active_targets": 0,
            "n_comparison_pairs": 0,
            "active_targets_csv": str(active_csv),
            "comparison_stars_csv": None,
            "suspected_variables_csv": None,
            "targets_without_comps": [],
            "field_density_h_star_per_mpx": float(_density),
            "field_density_class": str(_d_class),
            "field_density_n_stars": int(_n_field),
        }

    # Read as strings to prevent Gaia ID precision loss (float64/scientific notation).
    ms_df = pd.read_csv(masterstars_csv, low_memory=False, dtype=_GAIA_ID_DTYPE)
    # Normalize Gaia ID to string
    for _id_col in ("catalog_id", "name"):
        if _id_col in ms_df.columns:
            ms_df[_id_col] = _normalize_id_series(ms_df[_id_col])

    # -- comparison-star selection - per target --
    all_comp_rows: list[pd.DataFrame] = []
    all_reserve_rows: list[pd.DataFrame] = []
    targets_without_comps: list[str] = []

    # PERF-5: unified ProcFrameStore - one disk read per proc_*.csv frame
    _pfc_dir = Path(per_frame_csv_dir)
    _proc_glob = PROC_CSV_GLOB
    if not any(_pfc_dir.glob(_proc_glob)):
        if any(_pfc_dir.glob("*_cal.csv")):
            _proc_glob = "*_cal.csv"
    _proc_store = ProcFrameStore.build(
        _pfc_dir,
        glob_pattern=_proc_glob,
        extra_cols=[flux_col] if flux_col not in PROC_STORE_COLS else None,
    )
    shared_csv_cache = _proc_store
    csv_paths = [Path(k) for k in _proc_store.keys()]
    # NIGHT-FWHM-2026-10: neighbour geometry (isolation, target-comp separation) uses the
    # night median FWHM of THIS group (typical frame; Irwin 2007, Kepler per-quarter
    # crowding), not the MASTERSTAR (best frame). MASTERSTAR FWHM stays for the D3
    # astrometric gate, which is measured on the MASTERSTAR image itself.
    from photometry_comp import night_median_fwhm_px  # noqa: PLC0415

    fwhm_night_px, _fwn_src = night_median_fwhm_px(csv_paths, float(fwhm_px))
    logging.info(
        "[COMP-SELECT] neighbour-geometry FWHM %.3f px (%s); MASTERSTAR %.3f px; iso %.1f px",
        float(fwhm_night_px),
        _fwn_src,
        float(fwhm_px),
        float(fwhm_night_px) * float(getattr(cfg or AppConfig(), "comp_isolation_fwhm", 3.0) or 3.0),
    )
    _p(f"Comparison-star selection: {len(_proc_store)} per-frame catalogs, {len(active)} targets...")

    try:
        import streamlit as st  # noqa: PLC0415

        if hasattr(st, "session_state"):
            st.session_state["proc_frame_store"] = _proc_store
            logging.debug("[PERF-6] ProcFrameStore stored in st.session_state")
    except Exception:  # noqa: BLE001
        pass

    _cfg_gaia_targets = _cfg_base
    _gaia_db_targets = str(_cfg_gaia_targets.gaia_db_path or "").strip()
    _vsx_db_targets = str(_cfg_gaia_targets.objects_db_path or "").strip() or None

    _vt_chip = pd.read_csv(variable_targets_csv, low_memory=False, dtype=_GAIA_ID_DTYPE)
    _vt_cid_exclude: frozenset[str] | None = None
    try:
        if "catalog_id" in _vt_chip.columns:
            from gaia_catalog_id import normalize_gaia_id_set  # noqa: PLC0415

            _vx = normalize_gaia_id_set(
                _vt_chip["catalog_id"].tolist(),
                log_label="variable_targets.csv (comp_selection exclude)",
            )
            _vt_cid_exclude = _vx or None
    except Exception:  # noqa: BLE001
        _vt_cid_exclude = None
    _fw_chip, _fh_chip = _target_selection_effective_frame_hw_px(
        _vt_chip,
        ms_df,
        frame_w_px=int(frame_w_px),
        frame_h_px=int(frame_h_px),
        edge_margin_px=int(chip_interior_margin_px),
    )

    # COMP-EDGE-2026-10 (Milan 2026-10-12; EDGE-ANNULUS-01): a comp's sky annulus must lie on the
    # chip in EVERY frame. safe_bbox was shrunk by the MASTERSTAR (best-frame) r_out; comps use
    # the night's largest r_out (annulus_outer_fwhm x max per-frame FWHM) - draft 526 S4 Cyg g:
    # comp 29 px from the edge, r_out 27.7 px at the median FWHM, 40 px in poor seeing.
    _comp_bbox = _comp_safe_bbox(variable_targets_csv, csv_paths, _cfg_base, _safe_bbox)

    _sat_may_exclude = True
    try:
        _sd_path = Path(per_frame_csv_dir).resolve().parent.parent.parent / "sat_diag.json"
        if not _sd_path.is_file():
            _sd_path = Path(output_dir).resolve().parent.parent / "sat_diag.json"
        if _sd_path.is_file():
            from sat_diag import load_sat_diag_json  # noqa: PLC0415

            _sd_ctx = load_sat_diag_json(_sd_path)
            if _sd_ctx is not None:
                _sat_may_exclude = bool(_sd_ctx.may_exclude_saturation())
                logging.info(
                    "[SAT-DIAG] comp_selection sat_may_exclude=%s source=%s sat_adu=%s",
                    _sat_may_exclude,
                    _sd_ctx.sat_source,
                    _sd_ctx.sat_adu,
                )
                # SAT-LIN-2026-10: linearity check from the science frames (report only).
                try:
                    import infolog as _infolog_lin  # noqa: PLC0415
                    from sat_diag import linearity_check, linearity_check_infolog_lines  # noqa: PLC0415

                    _lin = linearity_check(csv_paths, _sd_ctx.sat_adu)
                    for _ln in linearity_check_infolog_lines(_lin):
                        _infolog_lin.log_event(_ln)
                    try:
                        _sd_js = json.loads(_sd_path.read_text(encoding="utf-8"))
                        _sd_js["linearity_check"] = _lin
                        _sd_path.write_text(json.dumps(_sd_js, indent=2), encoding="ascii")
                    except Exception as _lin_w_exc:  # noqa: BLE001
                        logging.debug("[LINEARITY] sat_diag.json update skipped: %s", _lin_w_exc)
                except Exception as _lin_exc:  # noqa: BLE001
                    logging.warning("[LINEARITY] check failed: %s", _lin_exc)
    except Exception as _sd_p1_exc:  # noqa: BLE001
        logging.debug("[SAT-DIAG] comp_selection context load skipped: %s", _sd_p1_exc)

    _cfg_gp = _cfg_base
    _global_pool_df: pd.DataFrame | None = None
    # CONSOLIDATE-01D: global pool is always on (COMP-POOL-01); False branch deleted.
    try:
        _global_pool_df = build_global_comp_pool(
            masterstars_df=ms_df,
            per_frame_csv_paths=csv_paths,
            csv_cache=shared_csv_cache,
            variable_target_catalog_ids=_vt_cid_exclude or frozenset(),
            safe_bbox=_comp_bbox,
            chip_fw=int(_fw_chip),
            chip_fh=int(_fh_chip),
            chip_interior_margin_px=int(chip_interior_margin_px),
            max_comp_rms=float(max_comp_rms),
            cfg=_cfg_gp,
            flux_col=flux_col,
            min_frames_frac=float(min_frames_frac),
            fwhm_px=float(fwhm_px),
            photometry_dir_for_meta=Path(output_dir),
            obs_group=Path(variable_targets_csv).resolve().parent.name,
            masterstar_fits=Path(variable_targets_csv).resolve().parent / "MASTERSTAR.fits",
            field_catalog_csv=Path(variable_targets_csv).resolve().parent / "field_catalog_cone.csv",
            sat_may_exclude=bool(_sat_may_exclude),
        )
        if _global_pool_df is None or getattr(_global_pool_df, "empty", True):
            _global_pool_df = None
    except Exception as _gcp_exc:  # noqa: BLE001
        from invariants_runtime import PopulationEmptiedError  # noqa: PLC0415

        if isinstance(_gcp_exc, PopulationEmptiedError):
            raise
        logging.warning(
            "[GLOBAL COMP POOL] build failed: %s - falling back to per-target masterstars",
            _gcp_exc,
        )
        _global_pool_df = None

    if _global_pool_df is not None and "catalog_id" in _global_pool_df.columns:
        _global_pool_df = _global_pool_df.sort_values("catalog_id", kind="mergesort").reset_index(
            drop=True
        )

    _gaia_batch: dict[str, dict[str, Any]] = {}
    if _gaia_db_targets:
        _cids_batch = [
            str(normalize_gaia_source_id(r.get("catalog_id") or ""))
            for _, r in active.iterrows()
        ]
        _gaia_batch = _batch_enrich_targets_bp_rp_from_gaia_db(_cids_batch, _gaia_db_targets)
        logging.info(
            "[COMP-SELECT] Gaia batch lookup: %d/%d targets enriched",
            len(_gaia_batch),
            int(len(active)),
        )

    # PERF-3: prefetch Gaia bp_rp + teff for masterstars comp pool (before per-target loop).
    _comp_gaia_prefetch: dict[str, dict[str, Any]] = {}
    _comp_source_ids_n = 0
    try:
        _comp_id_seen: set[str] = set()
        _comp_source_ids: list[str] = []
        for _pool_df in (ms_df, _global_pool_df):
            if _pool_df is None or getattr(_pool_df, "empty", True):
                continue
            for _id_col in ("catalog_id", "name"):
                if _id_col not in _pool_df.columns:
                    continue
                for raw in _pool_df[_id_col].dropna().unique():
                    g = normalize_gaia_source_id(raw)
                    if not g or not g.isdigit() or g in _comp_id_seen:
                        continue
                    _comp_id_seen.add(g)
                    _comp_source_ids.append(g)
        _comp_source_ids_n = len(_comp_source_ids)
        if _comp_source_ids and _gaia_db_targets:
            _comp_gaia_prefetch = _batch_enrich_targets_bp_rp_from_gaia_db(
                _comp_source_ids,
                _gaia_db_targets,
            )
            logging.info(
                "[PERF-3] Comp Gaia prefetch: %d source_ids -> %d hits",
                _comp_source_ids_n,
                len(_comp_gaia_prefetch),
            )
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0213] Comp Gaia bp_rp prefetch batch fails - comp_selection comp selection hits DB per-star instead o...: %s', exc)
        logging.warning("[PERF-3] Comp Gaia prefetch failed (non-fatal): %s", exc)

    _t_comp_selection = time.time()
    _n_active = int(len(active))
    _n_oos_skipped = 0
    # PHASE1-SKIP / TARGET-RULES-2026-10: no comparison selection for targets whose target-selection skip is
    # final (saturated, S/N below snr_min, VSX type out of scope) - they are never measured.
    from photometry_core import FINAL_SKIP_REASONS  # noqa: PLC0415

    _n_comp_selection = int(
        (~active.get("skip_reason", pd.Series("", index=active.index)).fillna("").astype(str).str.strip()
         .isin(list(FINAL_SKIP_REASONS))).sum()
    )
    _i_comp_selection = 0
    # COMP-RULES-2026-10: minimum target-comparison separation from geometry, not a fixed
    # 60 arcsec: the comparison aperture must lie outside the target's sky annulus
    # (d >= r_out + r_ap = (annulus_outer_fwhm + k_max) x FWHM).
    try:
        from aperture_pertarget import DEFAULT_APERTURE_F_GRID  # noqa: PLC0415

        _scale_md = locals().get("_actual_scale") if bool(locals().get("_wcs_scale_ok", False)) else None
        if not (isinstance(_scale_md, float) and math.isfinite(_scale_md) and _scale_md > 0):
            _scale_md = float(_expected_scale)
        _k_md = float(getattr(_cfg_base, "annulus_outer_fwhm", 5.2)) + float(max(DEFAULT_APERTURE_F_GRID))
        _md_phys = float(_k_md * float(fwhm_night_px) * float(_scale_md))
        if math.isfinite(_md_phys) and _md_phys > 0:
            logging.info(
                "[COMP-RULES] min target-comp separation %.1f arcsec = %.2f FWHM x %.2f px x %.3f \"/px",
                _md_phys,
                _k_md,
                float(fwhm_night_px),
                float(_scale_md),
            )
            min_dist_arcsec = _md_phys
    except Exception as _md_exc:  # noqa: BLE001
        logging.warning("[COMP-RULES] physical min separation not resolved (%s); using %.1f arcsec",
                        _md_exc, float(min_dist_arcsec))
    for _i_active, (active_idx, target_row) in enumerate(active.iterrows(), start=1):
        try:
            _skip_reason = str(target_row.get("skip_reason") or "").strip()
            if _skip_reason in FINAL_SKIP_REASONS:
                _n_oos_skipped += 1
                continue
            _i_comp_selection += 1
            if progress_cb is not None:
                _tid = str(target_row.get("vsx_name") or target_row.get("catalog_id", ""))[:48]
                _p(f"Comparison-star selection: target {_i_comp_selection}/{_n_comp_selection}: {_tid}")
            tr_enriched = _enrich_target_bp_rp_from_gaia_db(
                target_row,
                gaia_db_path=_gaia_db_targets,
                gaia_prefetch=_gaia_batch,
            )
            if "bp_rp" in active.columns:
                active.loc[active_idx, "bp_rp"] = tr_enriched.get("bp_rp", active.loc[active_idx, "bp_rp"])
            comps = select_comparison_stars_per_target(
                tr_enriched,
                ms_df,
                csv_paths,
                csv_cache=shared_csv_cache,
                global_comp_pool_df=_global_pool_df,
                fwhm_px=fwhm_night_px,
                fwhm_d3_px=float(fwhm_px),
                n_comp_min=n_comp_min,
                # COMP-SMART-2026-10: rank N_RESERVE more candidates as the reserve that
                # aperture photometry uses to replace suspect comps (split below).
                n_comp_max=int(n_comp_max),
                n_reserve=_N_RESERVE,
                max_comp_rms=max_comp_rms,
                min_dist_arcsec=min_dist_arcsec,
                min_frames_frac=min_frames_frac,
                flux_col=flux_col,
                chip_fw=_fw_chip,
                chip_fh=_fh_chip,
                chip_interior_margin_px=int(chip_interior_margin_px),
                gaia_db_path=str(_cfg_gaia_targets.gaia_db_path or "").strip() or None,
                gaia_prefetch=_comp_gaia_prefetch,
                variable_target_catalog_ids=_vt_cid_exclude,
                cfg=_cfg_gp,
                plate_scale_arcsec=_ps_p01,
                use_pixel_dist=not _wcs_scale_ok,
                sat_may_exclude=_sat_may_exclude,
            )
            if comps is not None and not comps.empty:
                comps, _resv = _split_comp_reserve(comps)
                if _resv is not None and not _resv.empty:
                    all_reserve_rows.append(_resv)
            if comps is None or comps.empty:
                targets_without_comps.append(str(tr_enriched.get("catalog_id", "")))
            else:
                all_comp_rows.append(comps)
        except Exception as exc:  # noqa: BLE001
            # EXC-0215: T3 -- Prefetch coverage stats log after comp selection suppressed (EXCEPT-BULK-2 2026-07-08)
            logging.warning(
                "[COMP-SELECT] %s: unexpected error, skipping: %s",
                str(target_row.get("catalog_id", "?")),
                exc,
            )
            targets_without_comps.append(str(target_row.get("catalog_id", "") or ""))
            continue

    if _n_oos_skipped:
        logging.info(
            "Comparison-star selection: %d targets with a final skip (VSX type, S/N, depth) - no comp selection",
            _n_oos_skipped,
        )

    try:
        active.to_csv(active_csv, index=False)
        logging.info("[COMP-SELECT] active_targets.csv rewritten after filling target bp_rp (Gaia DB).")
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0214] active_targets.csv rewrite after comp_selection bp_rp enrichment fails - disk copy lacks update...: %s', exc)
        log_event(f"active_targets.csv write after comparison-star selection failed: {exc!s}")

    comp_df = pd.concat(all_comp_rows, ignore_index=True) if all_comp_rows else pd.DataFrame()
    if "target_catalog_id" in comp_df.columns and "catalog_id" in comp_df.columns:
        _before = len(comp_df)
        comp_df = comp_df.drop_duplicates(
            subset=["target_catalog_id", "catalog_id"], keep="first"
        )
        _after = len(comp_df)
        if _before != _after:
            log_event(
                f"comparison_stars_per_target: removed {_before - _after} "
                f"duplicate (target_catalog_id, catalog_id) rows"
            )
    if _comp_gaia_prefetch and not comp_df.empty and "catalog_id" in comp_df.columns:
        try:
            _sel_cids = {
                normalize_gaia_source_id(x)
                for x in comp_df["catalog_id"].tolist()
                if normalize_gaia_source_id(x)
            }
            _n_pref_hit = sum(1 for c in _sel_cids if c in _comp_gaia_prefetch)
            logging.info(
                "[PERF-3] Selected comp stars covered by prefetch: %d/%d "
                "(pool prefetch %d ids, %d DB hits)",
                _n_pref_hit,
                len(_sel_cids),
                _comp_source_ids_n,
                len(_comp_gaia_prefetch),
            )
        except Exception:  # noqa: BLE001
            pass
    # Safety: even when no comps found (or all targets failed), keep a stable schema so CSV isn't empty.
    if comp_df is None or len(list(comp_df.columns)) == 0:
        comp_df = pd.DataFrame(
            columns=[
                "catalog_id",
                "name",
                "ra_deg",
                "dec_deg",
                "x",
                "y",
                "mag",
                "bp_rp",
                "comp_rms",
                "comp_score",
                "comp_n_frames",
                "target_catalog_id",
                "target_vsx_name",
                "target_bp_rp",
                "delta_bprp_abs",
                "comp_tier",
                "color_tier_src",
                "comp_weight",
                "selection_note",
                "selected_tier",
                "tier4_warning",
                "n_tier1",
                "n_tier2",
                "n_tier3",
                "n_tier4",
            ]
        )

    # Fallback: fill bp_rp for COMP stars without a Gaia colour from the local Gaia DB (sky box around RA/Dec).
    try:
        if (
            not comp_df.empty
            and "bp_rp" in comp_df.columns
            and "ra_deg" in comp_df.columns
            and "dec_deg" in comp_df.columns
        ):
            gaia_db_path = str(_cfg_base.gaia_db_path or "").strip() or None

            bp_nan = pd.to_numeric(comp_df["bp_rp"], errors="coerce").isna()
            ra_ok = pd.to_numeric(comp_df["ra_deg"], errors="coerce").apply(lambda v: math.isfinite(float(v)))
            dec_ok = pd.to_numeric(comp_df["dec_deg"], errors="coerce").apply(lambda v: math.isfinite(float(v)))
            needs = comp_df[bp_nan & ra_ok & dec_ok].copy()

            n_nan = int(len(needs))
            n_found = 0
            if n_nan > 0 and gaia_db_path:
                if "gaia_bp_rp_source" not in comp_df.columns:
                    comp_df["gaia_bp_rp_source"] = ""

                # Magnitude column for matching Gaia photometry (prefer "mag", fallback to "phot_g_mean_mag").
                mag_col = "mag" if "mag" in comp_df.columns else ("phot_g_mean_mag" if "phot_g_mean_mag" in comp_df.columns else None)

                radius_deg = 0.001  # ~3.6 arcsec
                for i, row in needs.iterrows():
                    ra0 = float(pd.to_numeric(row.get("ra_deg"), errors="coerce"))
                    dec0 = float(pd.to_numeric(row.get("dec_deg"), errors="coerce"))
                    if not (math.isfinite(ra0) and math.isfinite(dec0)):
                        continue

                    mag_comp = float("nan")
                    if mag_col is not None:
                        try:
                            mag_comp = float(pd.to_numeric(row.get(mag_col), errors="coerce"))
                        except Exception:  # noqa: BLE001
                            # EXC-0216: T4 -- Bad ra/dec on one Gaia fallback row skipped in comp bp_rp nearest-neighbor search (EXCEPT-BULK-2 2026-07-08)
                            mag_comp = float("nan")
                    if not math.isfinite(mag_comp):
                        continue

                    dec_min = max(-90.0, dec0 - radius_deg)
                    dec_max = min(90.0, dec0 + radius_deg)

                    # Handle RA wrap at 0/360 for tiny windows.
                    ra_min = ra0 - radius_deg
                    ra_max = ra0 + radius_deg
                    gaia_rows: list[dict[str, Any]] = []
                    if ra_min < 0.0:
                        gaia_rows.extend(
                            query_local_gaia(
                                ra_min=360.0 + ra_min,
                                ra_max=360.0,
                                dec_min=dec_min,
                                dec_max=dec_max,
                                db_path=gaia_db_path,
                                mag_limit=max(20.0, mag_comp + 2.0),
                                max_rows=200,
                            )
                        )
                        gaia_rows.extend(
                            query_local_gaia(
                                ra_min=0.0,
                                ra_max=ra_max,
                                dec_min=dec_min,
                                dec_max=dec_max,
                                db_path=gaia_db_path,
                                mag_limit=max(20.0, mag_comp + 2.0),
                                max_rows=200,
                            )
                        )
                    elif ra_max > 360.0:
                        gaia_rows.extend(
                            query_local_gaia(
                                ra_min=ra_min,
                                ra_max=360.0,
                                dec_min=dec_min,
                                dec_max=dec_max,
                                db_path=gaia_db_path,
                                mag_limit=max(20.0, mag_comp + 2.0),
                                max_rows=200,
                            )
                        )
                        gaia_rows.extend(
                            query_local_gaia(
                                ra_min=0.0,
                                ra_max=ra_max - 360.0,
                                dec_min=dec_min,
                                dec_max=dec_max,
                                db_path=gaia_db_path,
                                mag_limit=max(20.0, mag_comp + 2.0),
                                max_rows=200,
                            )
                        )
                    else:
                        gaia_rows = query_local_gaia(
                            ra_min=ra_min,
                            ra_max=ra_max,
                            dec_min=dec_min,
                            dec_max=dec_max,
                            db_path=gaia_db_path,
                            mag_limit=max(20.0, mag_comp + 2.0),
                            max_rows=200,
                        )

                    if not gaia_rows:
                        continue

                    best = None
                    best_d = float("inf")
                    for gr in gaia_rows:
                        try:
                            g_mag = float(gr.get("g_mag"))
                        except Exception:  # noqa: BLE001
                            g_mag = float("nan")
                        if not (math.isfinite(g_mag) and abs(g_mag - mag_comp) < 1.0):
                            continue
                        try:
                            ra_g = float(gr.get("ra"))
                            dec_g = float(gr.get("dec"))
                        except Exception:  # noqa: BLE001
                            continue
                        if not (math.isfinite(ra_g) and math.isfinite(dec_g)):
                            continue
                        d = _angular_distance_deg(ra0, dec0, ra_g, dec_g)
                        if math.isfinite(d) and d < best_d:
                            best_d = d
                            best = gr

                    if best is None:
                        continue
                    bprp = best.get("bp_rp")
                    try:
                        bprp_f = float(bprp)
                    except Exception:  # noqa: BLE001
                        bprp_f = float("nan")
                    if not math.isfinite(bprp_f):
                        continue

                    comp_df.loc[i, "bp_rp"] = bprp_f
                    comp_df.loc[i, "gaia_bp_rp_source"] = "gaia_db_fallback"
                    n_found += 1

            if n_nan > 0:
                log_event(f"COMP bp_rp fallback: {n_found}/{n_nan} stars filled from Gaia DB")
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0217] Whole comp bp_rp Gaia-DB fallback block fails - comps export with NaN bp_rp and wrong t...: %s', exc)
        pass

    comp_csv = output_dir / "comparison_stars_per_target.csv"
    # OUTPUT-REVIEW-2026-10 step 5: one table - the admitted comps and the COMP-SMART reserve
    # (p1_role / p1_rank); aperture photometry marks the stars of the final ensembles (used_role / used_rank).
    from comp_table import LEGACY_COMP_SELECTION_CSV, LEGACY_RESERVE_CSV, comp_selection_table, write_comp_table  # noqa: PLC0415

    _res_all = pd.concat(all_reserve_rows, ignore_index=True) if all_reserve_rows else None
    write_comp_table(comp_selection_table(comp_df, _res_all), comp_csv)
    logging.info("[COMP-SMART] reserve: %d rows (in %s)", 0 if _res_all is None else len(_res_all), comp_csv.name)
    for _old in (LEGACY_COMP_SELECTION_CSV, LEGACY_RESERVE_CSV):  # files of the earlier 3-table format
        try:
            (output_dir / _old).unlink(missing_ok=True)
        except OSError as _ul_exc:
            logging.warning("[COMP-TABLE] %s not removed: %s", _old, _ul_exc)
    logging.info(
        f"[COMP-SELECT] Saved: {comp_csv} "
        f"({len(comp_df)} rows, {len(all_comp_rows)} targets with comparison stars)"
    )
    logging.info(f"[COMP-SELECT] Time (comp selection): {time.time() - _t_comp_selection:.1f}s")
    _ps_p1 = _ps_p01
    _gs11_p1 = build_gs11_summary(
        [],
        _cfg_base,
        plate_scale_arcsec=_ps_p1,
    )
    merge_photometry_pipeline_meta(output_dir, {"gs11_summary": _gs11_p1})

    # -- Suspected variables --
    # Stars with high RMS (>3 sigma above the median) that are neither VSX nor active targets
    _p("Suspected variables (new candidates)...")
    suspected_csv = output_dir / "suspected_variables.csv"
    _active_ids: set[str] = set()
    for _ax in active["catalog_id"].tolist():
        _nx = _normalize_id_value(_ax)
        if _nx:
            _active_ids.add(_nx)

    _m_sus = float(chip_interior_margin_px)
    _box_sus = _comp_bbox if _comp_bbox is not None else (
        _m_sus, _m_sus, float(_fw_chip) - _m_sus, float(_fh_chip) - _m_sus
    )

    _write_suspected_variables(
        ms_df=ms_df,
        csv_paths=csv_paths,
        active_target_ids=_active_ids,
        output_path=suspected_csv,
        min_frames_frac=min_frames_frac,
        outlier_sigma=3.0,
        interior_bbox=_box_sus,
        csv_cache=shared_csv_cache,
    )
    # Best-effort: repair Gaia IDs in suspected_variables.csv via RA/DEC + local Gaia DB.
    try:
        from repair_catalog_ids import repair_csv_catalog_ids_from_gaia_db  # noqa: PLC0415

        _gdb = str(_cfg_base.gaia_db_path or "").strip()
        if _gdb:
            gdbp = Path(_gdb)
            if gdbp.is_file() and suspected_csv.is_file():
                _ = repair_csv_catalog_ids_from_gaia_db(
                    csv_path=suspected_csv,
                    gaia_db_path=gdbp,
                    id_col="catalog_id",
                    backup=False,
                    max_sep_arcsec=10.0,
                    log_fn=lambda _m: None,
                )
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0219] suspected_variables.csv catalog_id auto-repair from Gaia DB fails silently: %s', exc)
        pass

    _p(f"Target and comparison-star selection done: {int(len(active))} targets, {int(len(comp_df))} target-comparison pairs")
    return {
        "n_active_targets": int(len(active)),
        "n_comparison_pairs": int(len(comp_df)),
        "active_targets_csv": str(active_csv),
        "comparison_stars_csv": str(comp_csv),
        "suspected_variables_csv": str(suspected_csv),
        "targets_without_comps": targets_without_comps,
        "field_density_h_star_per_mpx": float(_density),
        "field_density_class": str(_d_class),
        "field_density_n_stars": int(_n_field),
        "proc_store": _proc_store,
    }
