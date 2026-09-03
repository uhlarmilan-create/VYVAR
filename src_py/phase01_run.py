"""Moved from photometry_core.py (CONSOLIDATE-01E3). Facade re-exports these names."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import copy
import json
import logging
import math
import time
from astropy.io import fits as astrofits
import numpy as np
import pandas as pd
from config import (
    AppConfig,
    DENSITY_OVERRIDES,
    apply_crowding_overrides,
    apply_density_overrides,
    classify_field_density,
    compute_field_density,
)
from database import query_local_gaia
from gaia_catalog_id import normalize_gaia_source_id
from infolog import log_event
from photometry_comp import (
    _attach_predicted_dilution_report,
    _batch_enrich_targets_bp_rp_from_gaia_db,
    _enrich_target_bp_rp_from_gaia_db,
    _normalize_id_series,
    _normalize_id_value,
    _phase0_effective_frame_hw_px,
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
from photometry_shared import _angular_distance_deg, _resolve_plate_scale_arcsec_per_px, build_gs11_summary
from proc_frame_store import PROC_CSV_GLOB, PROC_STORE_COLS, ProcFrameStore

from photometry_core import (
    _GAIA_ID_DTYPE,
)

def run_phase0_and_phase1(
    variable_targets_csv: Path,
    masterstars_csv: Path,
    per_frame_csv_dir: Path,
    output_dir: Path,
    *,
    fwhm_px: float = 3.7,
    frame_w_px: int = 2082,
    frame_h_px: int = 1397,
    chip_interior_margin_px: int = 100,
    plate_scale_arcsec_px: float | None = None,
    max_dist_deg: float = 1.0,
    max_mag_diff: float = 0.25,
    max_mag_diff_t1: float = 0.50,
    max_mag_diff_t2: float = 1.00,
    max_mag_diff_t3: float = 1.50,
    max_mag_diff_t4: float = 2.00,
    n_comp_min: int = 3,
    n_comp_max: int = 7,
    max_comp_rms: float = 0.1,
    min_dist_arcsec: float = 60.0,
    min_frames_frac: float = 0.3,
    rms_outlier_sigma: float = 3.0,
    exclude_gaia_nss: bool = True,
    exclude_gaia_extobj: bool = True,
    mag_bright_threshold: float = 12.0,
    max_mag_diff_bright_floor: float = 0.0,
    max_psf_chi2: float = 3.0,
    max_fwhm_factor: float = 1.5,
    isolation_radius_px: float = 25.0,
    flux_col: str = "dao_flux",
    comp_max_delta_bprp: float = 0.5,
    cfg: AppConfig | None = None,
    progress_cb: Any = None,
    draft_id: int | None = None,
    db: Any = None,
) -> dict[str, Any]:
    """Spusti Fazu 0 + Fazu 1 a ulozi vystupy.

    Vystupy (ulozene do output_dir):
      active_targets.csv              - VSX ciele + ``zone_flag`` / ``skip_photometry`` (saturovane)
      comparison_stars_per_target.csv - porovnavacie hviezdy pre kazdy ciel
      suspected_variables.csv         - kandidati na nove premenne (vysoky RMS, nie VSX)

    Returns:
        dict s klucmi:
          n_active_targets, n_comparison_pairs,
          active_targets_csv, comparison_stars_csv, suspected_variables_csv,
          targets_without_comps (list catalog_id)

    Args:
        chip_interior_margin_px: Min. pocet pixelov od okraja cipu pre **vsetky** kroky Fazy 0+1
            (aktivne ciele, porovnavacky, suspected). ``0`` = bez priestoroveho orezania.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prefer night seeing (VY_FWHM) over the 3.7 default / Gaussian core for
    # Phase-1 blend isolation (COMP-ASSIGN-03 uses snr_cog_isolation_fwhm x FWHM).
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
        "[PHASE 0+1] fwhm_px=%.3f (single-source iso=%.1f px at %.2f FWHM)",
        float(fwhm_px),
        float(fwhm_px) * float(getattr(cfg or AppConfig(), "snr_cog_isolation_fwhm", 3.0) or 3.0),
        float(getattr(cfg or AppConfig(), "snr_cog_isolation_fwhm", 3.0) or 3.0),
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
            "[PHASE 0+1] Frame dimensions %dx%d px from %s (caller default %dx%d)",
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
    _d_class = classify_field_density(
        float(_density),
        float(_cfg_base.field_density_sparse_threshold),
        float(_cfg_base.field_density_dense_threshold),
    )
    logging.info(
        "[FIELD DENSITY] %.0f hviezd/Mpx -> trieda: %s (n_stars=%d, chip=%dx%dpx, n_src=%s)",
        float(_density),
        _d_class,
        int(_n_field),
        int(_cw_fd),
        int(_ch_fd),
        _nsrc_fd,
    )
    _adaptive_on = bool(_cfg_base.field_density_adaptive_enabled)
    _cfg_for_2a = copy.copy(_cfg_base)
    # -- [CROWDING-CLASSIFIER] signal-based comp overrides (gated, default OFF) --
    # Replaces the detection/scale-locked stars/Mpx class with detection-independent
    # crowding_index signals. Additive sidecar (crowding_index.json); never overwrites
    # field_density.json. Falls back to the stars/Mpx path on any failure.
    _crowding_applied = False
    if bool(getattr(_cfg_base, "crowding_classifier_enabled", False)) and db is not None and draft_id is not None:
        try:
            from crowding_index import compute_crowding_index as _compute_ci  # noqa: PLC0415
            from database import get_gaia_db_max_g_mag as _get_gmax  # noqa: PLC0415

            _ps_dir = _ms_density.parent
            _gmax = float(_get_gmax(_cfg_base.gaia_db_path))
            _ci_res, _ = _compute_ci(
                _ps_dir.parent.parent, _ps_dir.name, db, int(draft_id), gaia_db_max_g=_gmax
            )
            _blend = _ci_res.get("blend_frac_1fwhm")
            _avail = _ci_res.get("n_gaia_below_eff_limit")
            _bottleneck = bool(_ci_res.get("catalog_is_bottleneck"))
            # SAMPLING GATE: a high comp-RMS only signals real contamination when the PSF
            # is resolved. On under-sampled fields the comp-RMS tail is the field floor,
            # so tightening max_comp_rms there thins the ensemble and worsens the LC.
            _min_fwhm = float(getattr(_cfg_base, "crowding_tighten_min_fwhm_px", 3.0))
            _well_sampled = float(fwhm_px) >= _min_fwhm
            _blend_high = _blend is not None and float(_blend) >= float(
                _cfg_base.crowding_blend_tighten_threshold
            )
            _tighten = bool(_blend_high and _well_sampled)
            _loosen = _avail is not None and float(_avail) < float(
                _cfg_base.crowding_comp_availability_loosen_count
            )
            _cfg_for_2a, _md_delta = apply_crowding_overrides(
                copy.copy(_cfg_base),
                loosen=bool(_loosen),
                tighten=bool(_tighten),
                suppress_mag_loosen=_bottleneck,
            )
            max_mag_diff = float(_cfg_for_2a.phase01_comparison_max_mag_diff)
            n_comp_min = int(_cfg_for_2a.phase01_comparison_n_comp_min)
            comp_max_delta_bprp = float(_cfg_for_2a.comp_max_delta_bprp)
            max_comp_rms = float(_cfg_for_2a.phase01_comparison_max_comp_rms)
            min_dist_arcsec = float(_cfg_for_2a.phase01_comparison_min_dist_arcsec)
            if _loosen:
                max_dist_deg = float(max_dist_deg) + float(_md_delta)
            _crowding_applied = True
            logging.info(
                "[CROWDING CLASSIFIER] blend=%.4f (th=%.3f) fwhm=%.2fpx (gate>=%.1f->sampled=%s) "
                "avail=%s (th=%.0f) bottleneck=%s -> loosen=%s tighten=%s | legacy stars/Mpx class=%s",
                float(_blend) if _blend is not None else float("nan"),
                float(_cfg_base.crowding_blend_tighten_threshold),
                float(fwhm_px),
                _min_fwhm,
                bool(_well_sampled),
                _avail,
                float(_cfg_base.crowding_comp_availability_loosen_count),
                _bottleneck,
                bool(_loosen),
                bool(_tighten),
                _d_class,
            )
            try:
                (output_dir / "crowding_index.json").write_text(
                    json.dumps(
                        {
                            **_ci_res,
                            "classifier": {
                                "enabled": True,
                                "loosen": bool(_loosen),
                                "tighten": bool(_tighten),
                                "blend_high": bool(_blend_high),
                                "well_sampled": bool(_well_sampled),
                                "fwhm_px": float(fwhm_px),
                                "tighten_min_fwhm_px": float(_min_fwhm),
                                "suppress_mag_loosen": bool(_bottleneck),
                                "blend_tighten_threshold": float(
                                    _cfg_base.crowding_blend_tighten_threshold
                                ),
                                "comp_availability_loosen_count": float(
                                    _cfg_base.crowding_comp_availability_loosen_count
                                ),
                                "legacy_stars_mpx_class": _d_class,
                                "eff_max_mag_diff": float(max_mag_diff),
                                "eff_n_comp_min": int(n_comp_min),
                                "eff_comp_max_delta_bprp": float(comp_max_delta_bprp),
                                "eff_max_comp_rms": float(max_comp_rms),
                                "eff_min_dist_arcsec": float(min_dist_arcsec),
                                "eff_max_dist_deg": float(max_dist_deg),
                            },
                        },
                        indent=2,
                        ensure_ascii=False,
                        default=str,
                    ),
                    encoding="utf-8",
                )
            except Exception as exc:  # noqa: BLE001
                logging.error('[EXC-0208] crowding_signal.json diagnostic write fails during field-density adaptive step: %s', exc)
                pass
        except Exception as _exc:  # noqa: BLE001
            logging.warning(
                "[CROWDING CLASSIFIER] signal computation failed (%s) - falling back to stars/Mpx",
                _exc,
            )
            _crowding_applied = False
    if not _crowding_applied and _adaptive_on:
        _cfg_for_2a = apply_density_overrides(copy.copy(_cfg_base), _d_class)
        max_mag_diff = float(_cfg_for_2a.phase01_comparison_max_mag_diff)
        n_comp_min = int(_cfg_for_2a.phase01_comparison_n_comp_min)
        comp_max_delta_bprp = float(_cfg_for_2a.comp_max_delta_bprp)
        max_comp_rms = float(_cfg_for_2a.phase01_comparison_max_comp_rms)
        min_dist_arcsec = float(_cfg_for_2a.phase01_comparison_min_dist_arcsec)
        _md_extra = DENSITY_OVERRIDES.get(_d_class, {}).get("phase01_comparison_max_dist_deg")
        if _md_extra is not None:
            max_dist_deg = float(max_dist_deg) + float(_md_extra)
    try:
        _fd_adaptive_applied = bool(
            not _crowding_applied and _adaptive_on and _d_class in ("sparse", "dense")
        )
        (output_dir / "field_density.json").write_text(
            json.dumps(
                {
                    "density_h_star_per_mpx": round(float(_density), 4),
                    "density_class": _d_class,
                    "n_stars": int(_n_field),
                    "n_stars_dao_raw": int(_vy_ndao_raw)
                    if _vy_ndao_raw is not None and int(_vy_ndao_raw) > 0
                    else None,
                    "n_stars_source": _nsrc_fd,
                    "chip_w_px": int(_cw_fd),
                    "chip_h_px": int(_ch_fd),
                    "field_density_adaptive_applied": _fd_adaptive_applied,
                    "crowding_classifier_applied": bool(_crowding_applied),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0209] field_density.json write fails - downstream Phase 2A lacks stored density metadata: %s', exc)
        pass

    _wcs_scale_ok = True
    _expected_scale = (
        float(plate_scale_arcsec_px)
        if plate_scale_arcsec_px is not None
        and math.isfinite(float(plate_scale_arcsec_px))
        and float(plate_scale_arcsec_px) > 0
        else float(_cfg_base.phase01_plate_scale_arcsec_per_px or 1.3)
    )
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

    # -- FAZA 0 --
    _p("Faza 0: vyber aktivnych cielov z VSX...")
    logging.info("[FAZA 0] Vyber aktivnych cielov...")
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
    _ms_for_catalog_only = Path(variable_targets_csv).resolve().parent / "MASTERSTAR.fits"
    _masterstar_wcs: Any = None
    if _ms_for_catalog_only.is_file():
        try:
            import warnings

            from astropy.wcs import FITSFixedWarning, WCS

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FITSFixedWarning)
                with astrofits.open(_ms_for_catalog_only, memmap=False) as hdul:
                    _masterstar_wcs = WCS(hdul[0].header)
        except Exception as exc:  # noqa: BLE001
            logging.warning("[VT REFRESH] MASTERSTAR WCS sa nepodarilo nacitat: %s - x/y v variable_targets.csv bez zmeny", exc)
            _masterstar_wcs = None

    _vt_p01 = Path(variable_targets_csv)
    if _masterstar_wcs is not None and _vt_p01.is_file():
        _refresh_variable_targets_xy(
            variable_targets_csv=_vt_p01,
            wcs=_masterstar_wcs,
            chip_w=int(frame_w_px),
            chip_h=int(frame_h_px),
        )

    if (
        plate_scale_arcsec_px is not None
        and math.isfinite(float(plate_scale_arcsec_px))
        and float(plate_scale_arcsec_px) > 0
    ):
        _plate_scale_p01 = float(plate_scale_arcsec_px)
    elif _ms_for_catalog_only.is_file():
        _plate_scale_p01 = _resolve_plate_scale_arcsec_per_px(_cfg_p01, _ms_for_catalog_only)
    else:
        _plate_scale_p01 = _resolve_plate_scale_arcsec_per_px(_cfg_p01)

    # TARGET-DEPTH-02: derive population depth from MASTERSTAR zone (single-frame copy; factor=1).
    _target_depth_g: float | None = None
    _depth_payload: dict[str, Any] = {}
    try:
        from comp_pool_noise import derive_target_depth_from_masterstar  # noqa: PLC0415

        _ms_depth = pd.read_csv(masterstars_csv, low_memory=False)
        _depth = derive_target_depth_from_masterstar(_ms_depth, masterstar_n_combine=1)
        _target_depth_g = _depth.target_depth_g
        _depth_payload = {
            "target_depth_g": _depth.target_depth_g,
            "linear_frac_thr": _depth.linear_frac_thr,
            "n_stars": _depth.n_stars,
            "mode": _depth.mode,
            "masterstar_n_combine": _depth.masterstar_n_combine,
            "snr_scale_factor": _depth.snr_scale_factor,
            "mag_offset": _depth.mag_offset,
            "rule": _depth.rule,
            "bin_rows": _depth.bin_rows,
        }
        if _target_depth_g is not None:
            logging.info(
                "[TARGET-DEPTH-02] derived target_depth_g=%.3f (linear_frac_thr=%.3f, n_stars=%d, n_combine=%d)",
                float(_target_depth_g),
                float(_depth.linear_frac_thr),
                int(_depth.n_stars),
                int(_depth.masterstar_n_combine),
            )
        else:
            logging.info("[TARGET-DEPTH-02] depth not derived: %s", _depth.rule)
    except Exception as exc:  # noqa: BLE001
        logging.error("[TARGET-DEPTH-02] depth derivation failed - continuing without depth gate: %s", exc)
        _target_depth_g = None

    active = select_active_targets(
        variable_targets_csv,
        masterstars_csv,
        frame_w_px=frame_w_px,
        frame_h_px=frame_h_px,
        edge_margin_px=int(chip_interior_margin_px),
        safe_bbox=_safe_bbox,
        gaia_db_path=str(_cfg_p01.gaia_db_path or ""),
        vsx_local_db_path=str(_cfg_p01.vsx_local_db_path or "").strip() or None,
        masterstar_fits_path=_ms_for_catalog_only if _ms_for_catalog_only.is_file() else None,
        plate_scale_arcsec_px=_plate_scale_p01,  # TODO-23
        cfg=_cfg_p01,
        target_depth_g=_target_depth_g,
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
    active = _attach_predicted_dilution_report(active, _cfg_p01)
    active.to_csv(active_csv, index=False)
    logging.info(f"[FAZA 0] Ulozene: {active_csv} ({len(active)} cielov)")
    if _depth_payload:
        try:
            _n_depth = (
                int((active["skip_reason"].astype(str) == "below_target_depth").sum())
                if (not active.empty and "skip_reason" in active.columns)
                else 0
            )
            _depth_payload = {
                **_depth_payload,
                "n_active_targets": int(len(active)),
                "n_masked_below_target_depth": _n_depth,
            }
            (output_dir / "target_depth.json").write_text(
                json.dumps(_depth_payload, indent=2, ensure_ascii=True),
                encoding="ascii",
                errors="replace",
            )
        except Exception as exc:  # noqa: BLE001
            logging.error("[TARGET-DEPTH-01] target_depth.json write failed: %s", exc)
    _excluded = _photometry_comp.LAST_EXCLUDED_TARGETS
    if _excluded is not None and not _excluded.empty:
        excluded_csv = output_dir / "excluded_targets.csv"
        _excluded.to_csv(excluded_csv, index=False)
        logging.info(f"[FAZA 0] Ulozene: {excluded_csv} ({len(_excluded)} excluded)")
    _p(f"Faza 0 hotova: {len(active)} aktivnych cielov")

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
            "field_density_adaptive_applied": bool(_adaptive_on and _d_class in ("sparse", "dense")),
            "field_density_n_stars": int(_n_field),
            "cfg_effective_for_photometry": _cfg_for_2a if _adaptive_on else None,
        }

    # Read as strings to prevent Gaia ID precision loss (float64/scientific notation).
    ms_df = pd.read_csv(masterstars_csv, low_memory=False, dtype=_GAIA_ID_DTYPE)
    # Normalizuj Gaia ID na string
    for _id_col in ("catalog_id", "name"):
        if _id_col in ms_df.columns:
            ms_df[_id_col] = _normalize_id_series(ms_df[_id_col])

    # -- FAZA 1 - per target --
    all_comp_rows: list[pd.DataFrame] = []
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
    _p(f"Faza 1: ProcFrameStore {len(_proc_store)} per-frame CSV - vyber porovnavaciek ({len(active)} cielov)...")

    try:
        import streamlit as st  # noqa: PLC0415

        if hasattr(st, "session_state"):
            st.session_state["proc_frame_store"] = _proc_store
            logging.debug("[PERF-6] ProcFrameStore stored in st.session_state")
    except Exception:  # noqa: BLE001
        pass

    _cfg_gaia_targets = _cfg_base
    _gaia_db_targets = str(_cfg_gaia_targets.gaia_db_path or "").strip()
    _vsx_db_targets = str(_cfg_gaia_targets.vsx_local_db_path or "").strip() or None

    _vt_chip = pd.read_csv(variable_targets_csv, low_memory=False, dtype=_GAIA_ID_DTYPE)
    _vt_cid_exclude: frozenset[str] | None = None
    try:
        if "catalog_id" in _vt_chip.columns:
            from gaia_catalog_id import normalize_gaia_id_set  # noqa: PLC0415

            _vx = normalize_gaia_id_set(
                _vt_chip["catalog_id"].tolist(),
                log_label="variable_targets.csv (phase1 exclude)",
            )
            _vt_cid_exclude = _vx or None
    except Exception:  # noqa: BLE001
        _vt_cid_exclude = None
    _fw_chip, _fh_chip = _phase0_effective_frame_hw_px(
        _vt_chip,
        ms_df,
        frame_w_px=int(frame_w_px),
        frame_h_px=int(frame_h_px),
        edge_margin_px=int(chip_interior_margin_px),
    )

    _cfg_gp = _cfg_for_2a if _adaptive_on else _cfg_base
    _global_pool_df: pd.DataFrame | None = None
    # CONSOLIDATE-01D: global pool is always on (COMP-POOL-01); False branch deleted.
    try:
        _global_pool_df = build_global_comp_pool(
            masterstars_df=ms_df,
            per_frame_csv_paths=csv_paths,
            csv_cache=shared_csv_cache,
            variable_target_catalog_ids=_vt_cid_exclude or frozenset(),
            safe_bbox=_safe_bbox,
            chip_fw=int(_fw_chip),
            chip_fh=int(_fh_chip),
            chip_interior_margin_px=int(chip_interior_margin_px),
            max_comp_rms=float(max_comp_rms),
            cfg=_cfg_gp,
            flux_col=flux_col,
            min_frames_frac=float(min_frames_frac),
            fwhm_px=float(fwhm_px),
            max_psf_chi2=float("inf"),  # global pool: skip PSF chi^2 (per-target filter unchanged)
            max_fwhm_factor=float(max_fwhm_factor),
            admission_artifact_dir=Path(output_dir),
            photometry_dir_for_meta=Path(output_dir),
        )
        if _global_pool_df is None or getattr(_global_pool_df, "empty", True):
            _global_pool_df = None
    except Exception as _gcp_exc:  # noqa: BLE001
        from comp_pool_noise import CompPoolAdmissionError  # noqa: PLC0415
        from invariants_runtime import PopulationEmptiedError  # noqa: PLC0415

        if isinstance(_gcp_exc, (CompPoolAdmissionError, PopulationEmptiedError)):
            raise
        logging.warning(
            "[GLOBAL COMP POOL] zostavenie zlyhalo: %s - fallback na per-target masterstars",
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
            "[PHASE 1] Gaia batch lookup: %d/%d targets enriched",
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
        logging.error('[EXC-0213] Comp Gaia bp_rp prefetch batch fails - phase1 comp selection hits DB per-star instead o...: %s', exc)
        logging.warning("[PERF-3] Comp Gaia prefetch failed (non-fatal): %s", exc)

    _t_phase1 = time.time()
    _gs11_comp_rejects_acc: list[int] = [0]
    _n_active = int(len(active))
    _n_oos_skipped = 0
    _n_phase1 = int(
        (active.get("skip_reason", pd.Series(dtype=str)).astype(str) != "vsx_type_out_of_scope").sum()
    )
    _i_phase1 = 0
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
                    "[SAT-DIAG] phase1 sat_may_exclude=%s source=%s sat_adu=%s",
                    _sat_may_exclude,
                    _sd_ctx.sat_source,
                    _sd_ctx.sat_adu,
                )
    except Exception as _sd_p1_exc:  # noqa: BLE001
        logging.debug("[SAT-DIAG] phase1 context load skipped: %s", _sd_p1_exc)

    for _i_active, (active_idx, target_row) in enumerate(active.iterrows(), start=1):
        try:
            _skip_reason = str(target_row.get("skip_reason") or "").strip()
            # Only vsx_type_out_of_scope is deterministic/final here. Do NOT skip
            # saturated/zone_flag targets - PER-FRAME-SAT can revive them in Phase 2A.
            if _skip_reason == "vsx_type_out_of_scope":
                _n_oos_skipped += 1
                continue
            _i_phase1 += 1
            if progress_cb is not None:
                _tid = str(target_row.get("vsx_name") or target_row.get("catalog_id", ""))[:48]
                _p(f"Phase 1: target {_i_phase1}/{_n_phase1}: {_tid}")
            tr_enriched = _enrich_target_bp_rp_from_gaia_db(
                target_row,
                gaia_db_path=_gaia_db_targets,
                vsx_local_db_path=_vsx_db_targets,
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
                fwhm_px=fwhm_px,
                max_dist_deg=max_dist_deg,
                max_mag_diff=max_mag_diff,
                max_mag_diff_t1=max_mag_diff_t1,
                max_mag_diff_t2=max_mag_diff_t2,
                max_mag_diff_t3=max_mag_diff_t3,
                max_mag_diff_t4=max_mag_diff_t4,
                n_comp_min=n_comp_min,
                n_comp_max=n_comp_max,
                max_comp_rms=max_comp_rms,
                min_dist_arcsec=min_dist_arcsec,
                min_frames_frac=min_frames_frac,
                rms_outlier_sigma=rms_outlier_sigma,
                exclude_gaia_nss=exclude_gaia_nss,
                exclude_gaia_extobj=exclude_gaia_extobj,
                mag_bright_threshold=mag_bright_threshold,
                max_mag_diff_bright_floor=max_mag_diff_bright_floor,
                max_psf_chi2=float("inf"),  # DAO-era proc CSV: chi^2 not ePSF yet
                max_fwhm_factor=max_fwhm_factor,
                isolation_radius_px=isolation_radius_px,
                flux_col=flux_col,
                chip_fw=_fw_chip,
                chip_fh=_fh_chip,
                chip_interior_margin_px=int(chip_interior_margin_px),
                max_delta_bprp=float(comp_max_delta_bprp),
                vsx_local_db_path=str(_cfg_gaia_targets.vsx_local_db_path or "").strip() or None,
                gaia_db_path=str(_cfg_gaia_targets.gaia_db_path or "").strip() or None,
                gaia_prefetch=_comp_gaia_prefetch,
                variable_target_catalog_ids=_vt_cid_exclude,
                cfg=_cfg_gp,
                plate_scale_arcsec=float(
                    plate_scale_arcsec_px
                    if plate_scale_arcsec_px is not None
                    and math.isfinite(float(plate_scale_arcsec_px))
                    and float(plate_scale_arcsec_px) > 0
                    else (
                        float(_cfg_gaia_targets.phase01_plate_scale_arcsec_per_px)
                        or 1.3
                    )
                ),
                use_pixel_dist=not _wcs_scale_ok,
                gs11_comp_rejects_acc=_gs11_comp_rejects_acc,
                sat_may_exclude=_sat_may_exclude,
            )
            if comps is None or comps.empty:
                targets_without_comps.append(str(tr_enriched.get("catalog_id", "")))
            else:
                all_comp_rows.append(comps)
        except Exception as exc:  # noqa: BLE001
            # EXC-0215: T3 -- Prefetch coverage stats log after comp selection suppressed (EXCEPT-BULK-2 2026-07-08)
            logging.warning(
                "[PHASE1] %s: neocakavana chyba, preskakujem: %s",
                str(target_row.get("catalog_id", "?")),
                exc,
            )
            targets_without_comps.append(str(target_row.get("catalog_id", "") or ""))
            continue

    if _n_oos_skipped:
        logging.info(
            "Faza 1: %d out-of-scope targets skipped (no comp selection)",
            _n_oos_skipped,
        )

    try:
        active.to_csv(active_csv, index=False)
        logging.info("[FAZA 0-1] active_targets.csv prepisane po doplneni bp_rp targetov (Gaia DB).")
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0214] active_targets.csv rewrite after phase1 bp_rp enrichment fails - disk copy lacks update...: %s', exc)
        log_event(f"active_targets.csv zapis po Faze 1 zlyhal: {exc!s}")

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
                "contamination_idx",
                "comp_n_frames",
                "target_catalog_id",
                "target_vsx_name",
                "target_bp_rp",
                "delta_bprp_abs",
                "comp_tier",
                "color_tier_src",
                "comp_weight",
                "selection_note",
                "used_mag_tol",
                "selected_tier",
                "tier4_warning",
                "n_tier1",
                "n_tier2",
                "n_tier3",
                "n_tier4",
            ]
        )

    # Fallback: dopln bp_rp pre COMP hviezdy bez Gaia farby pomocou lokalnej Gaia DB (sky-box okolo RA/Dec).
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
                log_event(f"COMP bp_rp fallback: {n_found}/{n_nan} hviezd doplnenych z Gaia DB")
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0217] Whole comp bp_rp Gaia-DB fallback block fails - comps export with NaN bp_rp and wrong t...: %s', exc)
        pass

    comp_csv = output_dir / "comparison_stars_per_target.csv"
    try:
        from gaia_catalog_id import normalize_gaia_source_id_series  # noqa: PLC0415

        if "catalog_id" in comp_df.columns:
            comp_df = comp_df.copy()
            comp_df["catalog_id"] = normalize_gaia_source_id_series(comp_df["catalog_id"])
        if "target_catalog_id" in comp_df.columns:
            comp_df = comp_df.copy()
            comp_df["target_catalog_id"] = normalize_gaia_source_id_series(comp_df["target_catalog_id"])
    except Exception as exc:  # noqa: BLE001
        logging.error('[EXC-0218] comparison_stars_per_target.csv catalog_id normalization fails - corrupted IDs in comp ...: %s', exc)
        pass
    from sky_separation import persist_dist_deg_column  # noqa: PLC0415

    persist_dist_deg_column(comp_df)
    comp_df.to_csv(comp_csv, index=False)
    _sparse_target_n = 0
    if "comp_path" in comp_df.columns and "target_catalog_id" in comp_df.columns:
        try:
            _cp = (
                comp_df.groupby(comp_df["target_catalog_id"].astype(str).str.strip())["comp_path"]
                .first()
                .astype(str)
                .str.strip()
                .str.lower()
            )
            _sparse_target_n = int((_cp == "sparse_fallback").sum())
        except Exception:  # noqa: BLE001
            _sparse_target_n = 0
    merge_photometry_pipeline_meta(
        output_dir,
        {"comp_sparse_fallback_target_count": int(_sparse_target_n)},
    )
    try:
        from pinned_ensembles import record_pinned_provenance_meta  # noqa: PLC0415

        record_pinned_provenance_meta(output_dir)
    except Exception as _pin_meta_exc:  # noqa: BLE001
        logging.warning("[PIN] pipeline_meta provenance record failed: %s", _pin_meta_exc)
    logging.info(
        f"[FAZA 1] Ulozene: {comp_csv} "
        f"({len(comp_df)} riadkov, {len(all_comp_rows)} targetov s porovnavackami)"
    )
    logging.info(f"[FAZA 1] Cas (comp selection): {time.time() - _t_phase1:.1f}s")
    _ps_p1 = float(
        plate_scale_arcsec_px
        if plate_scale_arcsec_px is not None
        and math.isfinite(float(plate_scale_arcsec_px))
        and float(plate_scale_arcsec_px) > 0
        else (float(_cfg_base.phase01_plate_scale_arcsec_per_px) or 1.3)
    )
    _gs11_p1 = build_gs11_summary(
        [],
        _cfg_base,
        comps_gs11_rejected=int(_gs11_comp_rejects_acc[0]),
        plate_scale_arcsec=_ps_p1,
    )
    merge_photometry_pipeline_meta(output_dir, {"gs11_summary": _gs11_p1})

    # -- Suspected variables --
    # Hviezdy s vysokym RMS (>3sigma nad medianom) ktore nie su VSX ani active targets
    _p("Faza 1: suspected variables (nove kandidaty)...")
    suspected_csv = output_dir / "suspected_variables.csv"
    _active_ids: set[str] = set()
    for _ax in active["catalog_id"].tolist():
        _nx = _normalize_id_value(_ax)
        if _nx:
            _active_ids.add(_nx)

    _margin_sus: int | None = None if int(chip_interior_margin_px) <= 0 else int(chip_interior_margin_px)

    _write_suspected_variables(
        ms_df=ms_df,
        csv_paths=csv_paths,
        active_target_ids=_active_ids,
        output_path=suspected_csv,
        min_frames_frac=min_frames_frac,
        outlier_sigma=3.0,
        interior_fw=_fw_chip,
        interior_fh=_fh_chip,
        interior_margin_px=_margin_sus,
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

    _p(f"Faza 0+1 hotovo: {int(len(active))} cielov, {int(len(comp_df))} parov porovnavaciek")
    return {
        "n_active_targets": int(len(active)),
        "n_comparison_pairs": int(len(comp_df)),
        "active_targets_csv": str(active_csv),
        "comparison_stars_csv": str(comp_csv),
        "suspected_variables_csv": str(suspected_csv),
        "targets_without_comps": targets_without_comps,
        "field_density_h_star_per_mpx": float(_density),
        "field_density_class": str(_d_class),
        "field_density_adaptive_applied": bool(_adaptive_on and _d_class in ("sparse", "dense")),
        "field_density_n_stars": int(_n_field),
        "cfg_effective_for_photometry": _cfg_for_2a if _adaptive_on else None,
        "proc_store": _proc_store,
    }
