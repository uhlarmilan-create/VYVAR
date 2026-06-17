"""Unified Settings dashboard: paths, QC, photometry, phase 0+1, alignment, tools + rich help."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from config import AppConfig, save_config_json
import ui_dao_stars as ui_dao_stars
import ui_photometry as ui_photometry
from masterstar_context import (
    load_masterstar_context,
    masterstar_context_markdown,
    resolve_masterstar_fits_path,
)


def _detail_help(title: str, *, phase: str, used_in: str, compute: str | None = None) -> None:
    with st.expander(f"❓ {title}", expanded=False):
        st.markdown(f"**Phase / process:** {phase}")
        st.markdown(f"**Where and how it is used:** {used_in}")
        if compute:
            st.markdown(f"**Derivation / computation:** {compute}")


def render_settings_dashboard(
    cfg: AppConfig,
    pipeline: Any,
    *,
    draft_dir_override: Path | None = None,
) -> None:
    st.subheader("Settings")
    st.caption(
        "Values in tabs **Overview … Phase 0+1** are saved with one button **Save main settings** "
        "to `config.json`. Parallelism (QC, preprocess, alignment, per-frame catalog) is unified — derived from CPU "
        "and RAM; environment variable `VYVAR_PARALLEL_WORKERS` overrides the default worker count."
    )

    draft_id = st.session_state.get("vyvar_last_draft_id")
    ms_path = resolve_masterstar_fits_path(
        cfg=cfg, db=getattr(pipeline, "db", None), draft_id=draft_id, draft_dir_override=draft_dir_override
    )
    ms_ctx = load_masterstar_context(ms_path)

    tab_ov, tab_paths, tab_cal, tab_qc, tab_aln, tab_ap, tab_p01, tab_tools = st.tabs(
        [
            "Overview",
            "Paths and catalogs",
            "Calibration",
            "Quality (QC)",
            "Alignment",
            "Photometry (aperture)",
            "Phase 0+1",
            "Tools",
        ]
    )

    with tab_ov:
        st.markdown("### Active draft and MASTERSTAR")
        if draft_id is None:
            st.info("Enter a draft in Pipeline (number or path) — MASTERSTAR context will appear here.")
        else:
            st.caption(f"Draft ID: **{int(draft_id)}**")
        if draft_dir_override is not None:
            st.caption(f"Folder override: `{draft_dir_override}`")
        st.markdown(masterstar_context_markdown(ms_ctx))
        _detail_help(
            "What the MASTERSTAR block above means",
            phase="After MASTERSTAR processing (pipeline / platesolve step).",
            used_in="Informational summary for scale, FWHM, and WCS; part of pipeline derives FOV / sep from FITS+WCS instead of manual JSON.",
            compute="`VY_FWHM` written by pipeline (median DAO FWHM from set or fit). Scale: `astropy.wcs.utils.proj_plane_pixel_scales` → mean in arcsec/px. Center: `pixel_to_world` at chip center.",
        )
        st.markdown("### Effective values (from `config.json`)")
        st.markdown(
            f"- Aperture: factor **{cfg.aperture_fwhm_factor:.2f}×FWHM**, annulus **{cfg.annulus_inner_fwhm:.2f}–{cfg.annulus_outer_fwhm:.2f}×FWHM**\n"
            f"- QC after calibration: **{'on' if cfg.qc_after_calibrate_enabled else 'off'}**, max HFR **{cfg.qc_max_hfr:.1f}**, min stars **{cfg.qc_min_stars}**\n"
            f"- Alignment: max **{cfg.alignment_max_stars}** stars, detection σ **{cfg.alignment_detection_sigma:.2f}**\n"
            f"- Phase 0+1: max Δmag **{cfg.phase01_comparison_max_mag_diff:.2f}**, min frame **{100 * cfg.phase01_comparison_min_frames_frac:.0f}%** of frames"
        )

    with tab_paths:
        st.markdown("### Paths and library")
        archive_root = st.text_input(
            "archive_root",
            value=str(cfg.archive_root),
            help="Archive root (Drafts, …).",
        )
        _detail_help(
            "archive_root",
            phase="Import, drafts, most disk outputs.",
            used_in="`AppConfig.archive_root` — base for `Drafts/draft_XXXXXX`, cache, and exports.",
            compute="None; must be a valid absolute path.",
        )
        calib_root = st.text_input(
            "calibration_library_root",
            value=str(cfg.calibration_library_root),
            help="Master dark/flat/bias library.",
        )
        _detail_help(
            "calibration_library_root",
            phase="Calibration (dark/flat stack), Calibration Library UI.",
            used_in="Find masters by validity and filter; write new masters.",
            compute="None.",
        )
        db_path = st.text_input(
            "database_path",
            value=str(cfg.database_path),
            help="SQLite DB for drafts, QC, MASTERSTAR paths.",
        )
        _detail_help(
            "database_path",
            phase="Entire app and pipeline run.",
            used_in="All draft tables, QC hash, paths to `MASTERSTAR.fits` (`get_obs_draft_masterstar_path`).",
            compute="None.",
        )

        st.markdown("---")
        st.subheader("GAIA DR3 (VYVAR Local Catalog)")
        gaia_db_path = st.text_input(
            "GAIA_DB_PATH (SQLite .db)",
            value=str(getattr(cfg, "gaia_db_path", "") or ""),
        )
        _detail_help(
            "GAIA_DB_PATH",
            phase="Per-frame and MASTERSTAR catalog (cone query), blind index.",
            used_in="Local cone search instead of online VizieR; requires table `gaia_dr3`.",
            compute="None — external DB (import/build script).",
        )
        st.session_state["GAIA_DB_PATH"] = str(gaia_db_path).strip()
        col_g1, col_g2 = st.columns([1, 3])
        with col_g1:
            if st.button("🔍 Test Connection", key="vyvar_test_gaia_db"):
                try:
                    import sqlite3

                    p = Path(str(gaia_db_path).strip())
                    if not p.is_file() or p.suffix.lower() != ".db":
                        raise FileNotFoundError("GAIA_DB_PATH must be an existing .db file.")
                    con = sqlite3.connect(str(p))
                    try:
                        cur = con.execute(
                            "SELECT name FROM sqlite_master WHERE type='table' AND name='gaia_dr3' LIMIT 1;"
                        )
                        if cur.fetchone() is None:
                            raise ValueError("Table `gaia_dr3` missing in DB.")
                    finally:
                        con.close()
                    st.success("OK: DB exists and table `gaia_dr3` is available.")
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))
        with col_g2:
            st.caption("Expect SQLite DB with table `gaia_dr3` and indexes `idx_ra`, `idx_dec`.")

        blind_index_fine_path = st.text_input(
            "BLIND_INDEX_FINE_PATH (.pkl)",
            value=str(getattr(cfg, "blind_index_fine_path", "") or ""),
            key="vyvar_blind_index_fine_path",
        )
        _detail_help(
            "BLIND_INDEX_FINE_PATH",
            phase="Blind astrometry / triangle matching (narrow rigs).",
            used_in="Fine triangle index for Newton-scale fields (~1.3″/px).",
            compute="Precomputed index — built by script #2 (`build_blind_index.py`).",
        )
        col_f1, col_f2 = st.columns([1, 3])
        with col_f1:
            if st.button("🔍 Test index", key="vyvar_test_blind_index_fine"):
                try:
                    import pickle

                    p = Path(str(blind_index_fine_path).strip())
                    if not p.is_file() or p.suffix.lower() != ".pkl":
                        raise FileNotFoundError("BLIND_INDEX_FINE_PATH must be an existing .pkl file.")
                    with open(p, "rb") as f:
                        data = pickle.load(f)
                    required = {"tree", "metadata", "mag_limit"}
                    missing = required - set(data.keys())
                    if missing:
                        raise ValueError(f"Index missing keys: {sorted(missing)}")
                    st.success(f"OK: fine index loads ({int(data.get('n_stars', 0)):,} stars).")
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))
        with col_f2:
            st.caption(
                "Triangle index for narrow rigs (Newton ~1.3″/px), mag14. "
                "Built by script #2. Wrong or missing path → blind solve fails on narrow fields."
            )

        blind_index_wide_path = st.text_input(
            "BLIND_INDEX_WIDE_PATH (.pkl)",
            value=str(getattr(cfg, "blind_index_wide_path", "") or ""),
            key="vyvar_blind_index_wide_path",
        )
        _detail_help(
            "BLIND_INDEX_WIDE_PATH",
            phase="Blind astrometry / triangle matching (wide rigs).",
            used_in="Wide triangle index for Carl-Zeiss-scale fields (~9.77″/px).",
            compute="Precomputed index — built by script #2 (`build_blind_index.py`).",
        )
        col_w1, col_w2 = st.columns([1, 3])
        with col_w1:
            if st.button("🔍 Test index", key="vyvar_test_blind_index_wide"):
                try:
                    import pickle

                    p = Path(str(blind_index_wide_path).strip())
                    if not p.is_file() or p.suffix.lower() != ".pkl":
                        raise FileNotFoundError("BLIND_INDEX_WIDE_PATH must be an existing .pkl file.")
                    with open(p, "rb") as f:
                        data = pickle.load(f)
                    required = {"tree", "metadata", "mag_limit"}
                    missing = required - set(data.keys())
                    if missing:
                        raise ValueError(f"Index missing keys: {sorted(missing)}")
                    st.success(f"OK: wide index loads ({int(data.get('n_stars', 0)):,} stars).")
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))
        with col_w2:
            st.caption(
                "Triangle index for wide rigs (Carl-Zeiss ~9.77″/px), mag14. "
                "Wrong or missing path → blind solve fails on wide fields."
            )

        st.markdown("---")
        st.subheader("VSX local database")
        vsx_db_path = st.text_input(
            "VSX_LOCAL_DB_PATH (SQLite .db, table `vsx_data`)",
            value=str(getattr(cfg, "vsx_local_db_path", "") or ""),
            key="vyvar_vsx_local_db_path",
        )
        _detail_help(
            "VSX_LOCAL_DB_PATH",
            phase="Variable targets export, MASTERSTAR QA, suspected LC.",
            used_in="Local VSX cone query (oid, ra_deg, dec_deg, …).",
            compute="None — import from VizieR.",
        )
        col_v1, col_v2 = st.columns([1, 3])
        with col_v1:
            if st.button("🔍 Test Connection", key="vyvar_test_vsx_local_db"):
                try:
                    from database import validate_vsx_local_db_schema

                    ok, code = validate_vsx_local_db_schema(str(vsx_db_path).strip())
                    if not ok:
                        _msgs = {
                            "missing_file": "File does not exist or path is empty.",
                            "missing_table_vsx_data": "Table `vsx_data` missing in DB.",
                        }
                        if str(code).startswith("missing_columns:"):
                            st.error(f"Missing columns: {code.split(':', 1)[1]} (required: oid, ra_deg, dec_deg).")
                        else:
                            st.error(_msgs.get(str(code), str(code)))
                    else:
                        st.success("OK: file exists, table `vsx_data` has required columns.")
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))
        with col_v2:
            st.caption(
                "SQLite from VizieR B/vsx/vsx import (columns oid, name, ra_deg, dec_deg, var_type, mag_max, mag_min). "
                "Pipeline use follows this path."
            )

        vsx_mag_limit_save = st.number_input(
            "Mag limit for Variable Targets export (VSX)",
            min_value=0.0,
            max_value=21.0,
            value=float(getattr(cfg, "vsx_variable_targets_mag_limit", 13.0) or 13.0),
            step=0.5,
            help="Cut VSX by mag_max.",
        )
        _detail_help(
            "vsx_variable_targets_mag_limit",
            phase="Export variable targets (VSX cone).",
            used_in="Keeps rows with `mag_max` ≤ limit (or without `mag_max`). Value 0 = no cut.",
            compute="Filter in SQL / pandas after query — not a physical computation.",
        )

    with tab_cal:
        st.markdown("### Calibration")
        new_dark = st.slider(
            "masterdark_validity_days",
            min_value=1,
            max_value=3650,
            value=int(cfg.masterdark_validity_days),
            help="Days after which master dark is considered stale.",
        )
        _detail_help(
            "masterdark_validity_days",
            phase="Select master dark from Calibration Library.",
            used_in="Compare observation date vs. master date; out of validity the master is not selected.",
            compute="Date difference in days vs. threshold.",
        )
        new_flat = st.slider(
            "masterflat_validity_days",
            min_value=1,
            max_value=3650,
            value=int(cfg.masterflat_validity_days),
            help="Master flat validity in days.",
        )
        _detail_help(
            "masterflat_validity_days",
            phase="Same as dark — flat selection.",
            used_in="Calibrate lights before later steps.",
            compute="Date difference vs. threshold.",
        )
        _cln_none = st.checkbox(
            "calibration_library_native_binning: read from each master FITS (JSON null)",
            value=cfg.calibration_library_native_binning is None,
            key="vyvar_settings_cl_bin_null",
        )
        new_cl_bin = st.number_input(
            "calibration_library_native_binning (1–16, if not “from FITS”)",
            min_value=1,
            max_value=16,
            value=int(cfg.calibration_library_native_binning or 1),
            disabled=bool(_cln_none),
            key="vyvar_settings_cl_bin",
        )
        _detail_help(
            "calibration_library_native_binning",
            phase="Match masters to frame (same binning).",
            used_in="If null, binning is read from master FITS header; else fixed value.",
            compute="None — either from header or constant from JSON.",
        )

    with tab_qc:
        st.markdown("### QC")
        qc_after_cal = st.checkbox(
            "qc_after_calibrate_enabled",
            value=bool(cfg.qc_after_calibrate_enabled),
            help="After calibration: QC metrics on flattened lights.",
        )
        _detail_help(
            "qc_after_calibrate_enabled",
            phase="Right after calibration (flattened lights).",
            used_in="Compute HFR, star count, background — write to DB (`OBS_FILES`) and limits in later steps (analyze, preprocess, MASTERSTAR pipeline selection).",
            compute="HFR/DAO metrics from `photutils`/pipeline QC modules (not a simple JSON formula).",
        )
        st.caption(
            "Threshold `qc_max_background_rms` is advanced — stays in `config.json` only (usually `null`); not set here."
        )
        qc_hfr = st.slider(
            "qc_max_hfr",
            min_value=0.5,
            max_value=20.0,
            value=float(cfg.qc_max_hfr),
            step=0.1,
            help="Worse HFR than this threshold → reject / flag (per pipeline logic).",
        )
        _detail_help(
            "qc_max_hfr",
            phase="QC after calibration and checks in later phases (analyze, preprocess).",
            used_in="Compare measured HFR (radius or equivalent) to threshold; too high HFR → reject or flag in DB per pipeline logic.",
            compute="HFR from frame analysis; threshold is constant from `config.json`.",
        )
        qc_stars = st.slider(
            "qc_min_stars",
            min_value=0,
            max_value=500,
            value=int(cfg.qc_min_stars),
            step=1,
            help="Minimum detected stars for successful QC.",
        )
        _detail_help(
            "qc_min_stars",
            phase="QC after calibration and related pipeline limits.",
            used_in="If detected star count is below threshold, frame is suspicious or rejected in QC.",
            compute="Count from DAO detection above threshold — threshold from this slider.",
        )
    with tab_aln:
        st.markdown("### Frame alignment (astroalign + DAO)")
        aln_max = st.slider(
            "alignment_max_stars",
            min_value=10,
            max_value=5000,
            value=int(cfg.alignment_max_stars),
            step=10,
            help="Max. brightest control points per frame.",
        )
        _detail_help(
            "alignment_max_stars",
            phase="Alignment phase (between calibration and stack / per-frame).",
            used_in="Sort stars by flux, trim to N for matching reference frame.",
            compute="N = min(detected, alignment_max_stars).",
        )
        aln_sig = st.slider(
            "alignment_detection_sigma",
            min_value=0.5,
            max_value=20.0,
            value=float(cfg.alignment_detection_sigma),
            step=0.25,
            help="DAO detection threshold for stars during alignment.",
        )
        _detail_help(
            "alignment_detection_sigma",
            phase="Alignment (DAO find).",
            used_in="Tied to QC detection style — higher σ = fewer faint stars, more robust to noise.",
            compute="Threshold in sigma above local background (standard DAO logic).",
        )

    with tab_ap:
        st.subheader("Detector — photometric parameters")
        st.caption(
            "Gain and Read Noise are used for SNR-optimal aperture "
            "(TODO-21) and photometric error calculation (Phase 2A). "
            "Values are stored per camera in EQUIPMENTS table."
        )
        _db_eq = getattr(pipeline, "db", None)
        if _db_eq is None:
            st.info("Database unavailable — gain/RN cannot be edited.")
        else:
            _eq_rows = _db_eq.get_equipments(active_only=False)
            if not _eq_rows:
                st.info("No equipment in DB. Add a camera via Database Explorer.")
            else:
                for _eq in _eq_rows:
                    _eq_id = int(_eq["ID"])
                    _eq_name = (
                        str(_eq.get("CAMERANAME") or _eq.get("ALIAS") or "").strip()
                        or f"ID={_eq_id}"
                    )
                    with st.expander(f"📷 {_eq_name}", expanded=False):
                        _g_cur, _rn_cur = _db_eq.get_equipment_cosmic_params(_eq_id)
                        _g_disp = float(_g_cur) if _g_cur is not None else 0.0
                        _rn_disp = float(_rn_cur) if _rn_cur is not None else 0.0
                        _gain_ui_max = 50.0
                        _rn_ui_max = 500.0
                        if _g_disp > _gain_ui_max or _g_disp < 0.0:
                            st.warning(
                                f"Stored gain {_g_disp:.3f} e⁻/ADU is outside the editor range "
                                f"0–{_gain_ui_max:.0f}; clamped for display — correct and Save."
                            )
                        if _rn_disp > 50.0:
                            st.warning(
                                f"Stored read noise {_rn_disp:.1f} e⁻ is unusually high "
                                f"(typical 3–15 e⁻) — verify units and Save if corrected."
                            )
                        _g_val = min(max(_g_disp, 0.0), _gain_ui_max)
                        _rn_val = min(max(_rn_disp, 0.0), _rn_ui_max)
                        _col1, _col2 = st.columns(2)
                        with _col1:
                            _gain_new = st.number_input(
                                "Gain [e⁻/ADU]",
                                min_value=0.0,
                                max_value=_gain_ui_max,
                                value=_g_val,
                                step=0.01,
                                format="%.3f",
                                key=f"vyvar_gain_{_eq_id}",
                                help="Typically 0.5–5.0 e⁻/ADU. QHY294MM: ~3.17",
                            )
                        with _col2:
                            _rn_new = st.number_input(
                                "Read Noise [e⁻]",
                                min_value=0.0,
                                max_value=_rn_ui_max,
                                value=_rn_val,
                                step=0.1,
                                format="%.1f",
                                key=f"vyvar_rn_{_eq_id}",
                                help="Typically 3–15 e⁻. QHY294MM: ~7.6",
                            )
                        if st.button("💾 Save", key=f"vyvar_save_eq_{_eq_id}"):
                            _db_eq.set_equipment_cosmic_params(_eq_id, _gain_new, _rn_new)
                            st.success(
                                f"Saved: gain={float(_gain_new):.3f} e⁻/ADU, "
                                f"RN={float(_rn_new):.1f} e⁻"
                            )
                        if _g_cur is None or _rn_cur is None:
                            st.warning(
                                "⚠️ Gain or Read Noise not set — "
                                "Phase 2A will use fallback (gain=1.0, RN=10.0). "
                                "Set values for accurate SNR calculations."
                            )
        st.markdown("---")
        st.markdown("### Photometry (aperture and annulus)")
        st.caption("Aperture vs. DAO and PSF toggles are under **Tools → Photometry (mode)**.")
        ap_fwhm = st.slider(
            "aperture_fwhm_factor",
            min_value=0.5,
            max_value=6.0,
            value=float(cfg.aperture_fwhm_factor),
            step=0.1,
            help="Aperture radius = factor × measured FWHM.",
        )
        _detail_help(
            "aperture_fwhm_factor",
            phase="Phase 2 / per-frame aperture photometry (if enabled).",
            used_in="Circle radius in pixels: `r_ap = factor × FWHM` (FWHM from header / frame measurement).",
            compute="Multiply local FWHM by constant from config.",
        )
        ann_in = st.slider(
            "annulus_inner_fwhm",
            min_value=1.0,
            max_value=10.0,
            value=float(cfg.annulus_inner_fwhm),
            step=0.25,
        )
        ann_out = st.slider(
            "annulus_outer_fwhm",
            min_value=1.5,
            max_value=12.0,
            value=float(cfg.annulus_outer_fwhm),
            step=0.25,
        )
        _detail_help(
            "annulus_inner_fwhm / annulus_outer_fwhm",
            phase="Aperture photometry — background subtraction.",
            used_in="Annulus between `r_inner = inner×FWHM` and `r_outer = outer×FWHM` around the star.",
            compute="Area-weighted mean between circles → `annulus_median` for sky; flux_v = sum(aperture) − sky×area.",
        )
        if ann_out <= ann_in:
            st.warning("annulus_outer_fwhm must be greater than annulus_inner_fwhm — will be adjusted on save.")
        st.markdown("**Role-aware aperture (TODO-44)**")
        col_var, col_comp = st.columns(2)
        with col_var:
            aperture_variable_factor = st.slider(
                "Variable target factor",
                0.5,
                2.0,
                float(cfg.aperture_variable_factor or 1.0),
                0.05,
                help="Aperture scale for variable targets (1.0 = SNR-optimal)",
            )
        with col_comp:
            aperture_comp_factor = st.slider(
                "Comp/check factor",
                0.5,
                2.0,
                float(cfg.aperture_comp_factor or 1.1),
                0.05,
                help="Aperture scale for comparison stars (1.1 = +10% for S/N)",
            )
        with st.expander("🔬 Advanced algorithms (ALG-2/3/4/5)"):
            st.caption("ALG-3 runs before ensemble normalization; ALG-2/4 run after airmass detrend.")
            col_a, col_b = st.columns(2)
            with col_a:
                temporal_binning_enabled = st.toggle(
                    "Temporal Binning (ALG-3)",
                    value=bool(cfg.temporal_binning_enabled),
                    help=(
                        "Hartley & Wilson 2023 MNRAS: smooth comp light curves before ensemble. "
                        "Default OFF — per-frame ensemble preserves common-mode cancellation."
                    ),
                )
                pytics_enabled = st.toggle(
                    "PyTICS weights (ALG-5)",
                    value=bool(cfg.pytics_enabled),
                    help="RASTI 2026: iterative comp star intercalibration",
                )
            with col_b:
                savgol_detrend_enabled = st.toggle(
                    "Savitzky-Golay detrend (ALG-2)",
                    value=bool(cfg.savgol_detrend_enabled),
                    help="Removes slow systematic trends after airmass detrend",
                )
                democratic_detrend_enabled = st.toggle(
                    "Democratic Detrender (ALG-4)",
                    value=bool(cfg.democratic_detrend_enabled),
                    help="arXiv 2026: 3-model marginalization + err_inflation column",
                )
            comp_slope_sig_k = st.slider(
                "comp_slope_significance_k",
                min_value=0.0,
                max_value=10.0,
                value=float(getattr(cfg, "comp_slope_significance_k", 3.0) or 3.0),
                step=0.5,
                help="Comp stability: min |slope|/stderr (σ) on common-mode-removed residual to exclude a comp.",
            )
            _detail_help(
                "comp_slope_significance_k",
                phase="Phase-2A comp stability (after common-mode detrend).",
                used_in="Slope exclusion requires both |slope| > comp_max_slope_mmag_hr and σ ≥ this threshold.",
                compute="Honeycutt/Broeg common-mode removed first; insignificant residual slopes are kept.",
            )
        nl_pct = st.slider(
            "nonlinearity_peak_percentile",
            min_value=0.0,
            max_value=50.0,
            value=float(cfg.nonlinearity_peak_percentile),
            step=0.5,
        )
        nl_ratio = st.slider(
            "nonlinearity_fwhm_ratio",
            min_value=1.01,
            max_value=3.0,
            value=float(cfg.nonlinearity_fwhm_ratio),
            step=0.01,
        )
        _detail_help(
            "nonlinearity_peak_percentile / nonlinearity_fwhm_ratio",
            phase="QC / nonlinearity flagging at aperture.",
            used_in="Finds brightness peak percentile and compares profile width to FWHM expectation.",
            compute="Pipeline heuristic: peak above percentile and FWHM ratio > threshold → suspect saturation/nonlinearity.",
        )
        st.markdown("---")
        st.markdown("### Data quality & validation")
        st.caption("LC-quality classification thresholds (Phase 2A summary `lc_quality_flag`).")
        lc_q_min = st.slider(
            "lc_quality_min_frames",
            min_value=3,
            max_value=500,
            value=int(getattr(cfg, "lc_quality_min_frames", 20) or 20),
            help="Frame floor for full good/noisy LC-quality verdict.",
        )
        lc_q_short = st.slider(
            "lc_quality_short_min_frames",
            min_value=2,
            max_value=100,
            value=int(getattr(cfg, "lc_quality_short_min_frames", 3) or 3),
            help="Below this -> no_data; [short, min) -> short_baseline (YELLOW, exportable).",
        )
        lc_q_frac = st.slider(
            "lc_quality_min_normal_frac",
            min_value=0.1,
            max_value=1.0,
            value=float(getattr(cfg, "lc_quality_min_normal_frac", 0.5) or 0.5),
            step=0.05,
            help="Minimum unsaturated/normal frame fraction for short_baseline or good/noisy.",
        )
        comp_trust_min = st.slider(
            "comp_trust_min_comps",
            min_value=3,
            max_value=20,
            value=int(getattr(cfg, "comp_trust_min_comps", 5) or 5),
            help="Trust RED floor (n_clean below); Phase-1 selection min stays separate.",
        )
        chk_min_epochs = st.slider(
            "check_star_min_epochs",
            min_value=3,
            max_value=50,
            value=int(getattr(cfg, "check_star_min_epochs", 5) or 5),
            help="Trust gate: check-star scatter ignored below this epoch count.",
        )
        if int(lc_q_short) > int(lc_q_min):
            st.warning("lc_quality_short_min_frames will be clamped to lc_quality_min_frames on save.")

        st.markdown("---")
        st.caption(
            "Frame-quality gate (Round-2 B.2): rejects transparency/PSF-collapsed frames in "
            "Phase 2A. Default OFF -> baseline byte-identical."
        )
        frame_quality_gate_enabled = st.toggle(
            "frame_quality_gate_enabled",
            value=bool(getattr(cfg, "frame_quality_gate_enabled", False)),
            help="Drop whole frames whose PSF-concentration (flux_large/flux) is a robust outlier "
            "and FWHM >= median. Targets cloud/dawn collapse; spares clear-but-faint frames.",
        )
        frame_quality_ratio_k = st.slider(
            "frame_quality_ratio_k",
            min_value=2.0,
            max_value=20.0,
            value=float(getattr(cfg, "frame_quality_ratio_k", 5.0) or 5.0),
            step=0.5,
            help="Robust z-score cut on per-frame flux_large/flux: reject if z=(ratio-median)/(1.4826*MAD) > k.",
        )
        frame_quality_fwhm_factor = st.slider(
            "frame_quality_fwhm_factor",
            min_value=0.8,
            max_value=3.0,
            value=float(getattr(cfg, "frame_quality_fwhm_factor", 1.0) or 1.0),
            step=0.05,
            help="Guard: reject a ratio-outlier only if its FWHM >= factor*median-FWHM (spares sharp frames).",
        )
        frame_quality_min_keep_frames = st.slider(
            "frame_quality_min_keep_frames",
            min_value=3,
            max_value=200,
            value=int(getattr(cfg, "frame_quality_min_keep_frames", 10) or 10),
            help="Safety floor: skip the gate entirely if it would keep fewer than this many frames.",
        )

    with tab_p01:
        st.markdown("### Phase 0+1 — star matching / stability")
        st.caption(
            "Filter parameters when matching catalog between frames (Gaia + custom rules). Details in `config.py` under `phase01_*` fields."
        )
        p01_md = st.slider(
            "phase01_comparison_max_dist_deg",
            min_value=0.05,
            max_value=10.0,
            value=float(cfg.phase01_comparison_max_dist_deg),
            step=0.05,
        )
        _detail_help(
            "phase01_comparison_max_dist_deg",
            phase="Phase 0+1 — spatial match between frames.",
            used_in="Max. angular distance between candidates for the same star.",
            compute="Great-circle or projection in pipeline; threshold in degrees from config.",
        )
        p01_mm = st.slider(
            "phase01_comparison_max_mag_diff",
            min_value=0.05,
            max_value=5.0,
            value=float(cfg.phase01_comparison_max_mag_diff),
            step=0.05,
        )
        p01_mag_b = st.slider(
            "phase01_comparison_mag_bright_threshold",
            min_value=6.0,
            max_value=18.0,
            value=float(cfg.phase01_comparison_mag_bright_threshold),
            step=0.25,
        )
        p01_mag_bf = st.slider(
            "phase01_comparison_max_mag_diff_bright_floor",
            min_value=0.0,
            max_value=4.0,
            value=float(cfg.phase01_comparison_max_mag_diff_bright_floor),
            step=0.05,
        )
        p01_mag_abs = st.slider(
            "Max Δmag absolute ceiling",
            min_value=1.5,
            max_value=5.0,
            value=float(getattr(cfg, "phase01_comparison_max_mag_diff_absolute", 3.0) or 3.0),
            step=0.25,
        )
        _detail_help(
            "phase01_comparison_max_mag_diff (+ bright threshold)",
            phase="Phase 0+1 — photometric match between frames.",
            used_in="If |Δmag| between frames > threshold, match is rejected. For bright stars (mag < threshold) at least `max_mag_diff_bright_floor` applies.",
            compute="Dynamic threshold: `max(|Δmag|, floor for bright)` per `config.py` logic.",
        )
        st.subheader("Color filter — Gaia BP-RP")
        tier1_bprp = st.slider(
            "Tier1 |ΔBP-RP| limit",
            min_value=0.05,
            max_value=0.50,
            value=float(getattr(cfg, "comp_tier1_bprp_limit", 0.25) or 0.25),
            step=0.01,
            help="Tier1 comp stars: |BP-RP(comp) − BP-RP(target)| ≤ this value.",
        )
        tier2_bprp = st.slider(
            "Tier2 |ΔBP-RP| limit",
            min_value=0.10,
            max_value=0.80,
            value=float(getattr(cfg, "comp_tier2_bprp_limit", 0.48) or 0.48),
            step=0.01,
        )
        tier3_bprp = st.slider(
            "Tier3 |ΔBP-RP| limit",
            min_value=0.20,
            max_value=1.20,
            value=float(getattr(cfg, "comp_tier3_bprp_limit", 0.79) or 0.79),
            step=0.01,
        )
        tier4_bprp = st.slider(
            "Tier4 |ΔBP-RP| limit (informational bound)",
            min_value=0.50,
            max_value=2.00,
            value=float(getattr(cfg, "comp_tier4_bprp_limit", 1.10) or 1.10),
            step=0.05,
        )
        comp_dbprp = st.slider(
            "Max |ΔBP-RP| (hard filter)",
            min_value=0.20,
            max_value=2.00,
            value=float(getattr(cfg, "comp_max_delta_bprp", 0.79) or 0.79),
            step=0.01,
        )
        st.caption("BP-RP (Gaia) is the colour filter for comparison star tiering and hard |ΔBP-RP| cut.")
        tier1_w = st.slider(
            "comp_tier1_weight",
            min_value=0.50,
            max_value=1.00,
            value=float(getattr(cfg, "comp_tier1_weight", 1.00) or 1.00),
            step=0.05,
        )
        tier2_w = st.slider(
            "comp_tier2_weight",
            min_value=0.50,
            max_value=1.00,
            value=float(getattr(cfg, "comp_tier2_weight", 0.85) or 0.85),
            step=0.05,
        )
        tier3_w = st.slider(
            "comp_tier3_weight",
            min_value=0.10,
            max_value=0.75,
            value=float(getattr(cfg, "comp_tier3_weight", 0.50) or 0.50),
            step=0.05,
        )
        tier4_w = st.slider(
            "comp_tier4_weight",
            min_value=0.05,
            max_value=0.50,
            value=float(getattr(cfg, "comp_tier4_weight", 0.25) or 0.25),
            step=0.05,
        )
        p01_ncmin = st.slider(
            "phase01_comparison_n_comp_min",
            min_value=2,
            max_value=12,
            value=int(cfg.phase01_comparison_n_comp_min),
        )
        p01_ncmax = st.slider(
            "phase01_comparison_n_comp_max",
            min_value=3,
            max_value=20,
            value=int(cfg.phase01_comparison_n_comp_max),
        )
        p01_rms = st.slider(
            "phase01_comparison_max_comp_rms",
            min_value=0.01,
            max_value=0.5,
            value=float(cfg.phase01_comparison_max_comp_rms),
            step=0.01,
        )
        p01_sparse_fb = st.checkbox(
            "comp_sparse_fallback_enabled",
            value=bool(getattr(cfg, "comp_sparse_fallback_enabled", True)),
            help=(
                "Per-target sparse fallback only: when default comp selection is starved, "
                "run generous pool + iterative CM-residual clip (default OFF)."
            ),
        )
        p01_sparse_fb_min = st.number_input(
            "comp_sparse_fallback_min",
            min_value=2,
            max_value=int(cfg.phase01_comparison_n_comp_max),
            value=int(getattr(cfg, "comp_sparse_fallback_min", 0) or cfg.phase01_comparison_n_comp_min),
            step=1,
            disabled=not p01_sparse_fb,
            help="Trigger fallback when default yields fewer comps than this (0 in config → n_comp_min).",
        )
        p01_clip_sigma = st.slider(
            "comp_clip_sigma",
            min_value=3.0,
            max_value=10.0,
            value=float(getattr(cfg, "comp_clip_sigma", 5.0) or 5.0),
            step=0.5,
            disabled=not p01_sparse_fb,
        )
        p01_mind = st.slider(
            "phase01_comparison_min_dist_arcsec",
            min_value=0.0,
            max_value=600.0,
            value=float(cfg.phase01_comparison_min_dist_arcsec),
            step=5.0,
        )
        p01_mff = st.slider(
            "phase01_comparison_min_frames_frac",
            min_value=0.05,
            max_value=0.95,
            value=float(cfg.phase01_comparison_min_frames_frac),
            step=0.05,
        )
        _detail_help(
            "phase01_comparison_min_frames_frac",
            phase="Phase 0+1 — stability across frames.",
            used_in="Star must appear in at least this fraction of frames, else dropped from comparison set.",
            compute="matched_frame_count / total_count ≥ threshold.",
        )
        p01_ex_nss = st.checkbox(
            "phase01_comparison_exclude_gaia_nss",
            value=bool(cfg.phase01_comparison_exclude_gaia_nss),
        )
        p01_ex_ext = st.checkbox(
            "phase01_comparison_exclude_gaia_extobj",
            value=bool(cfg.phase01_comparison_exclude_gaia_extobj),
        )
        _detail_help(
            "exclude_gaia_nss / exclude_gaia_extobj",
            phase="Phase 0+1 — catalog cleaning.",
            used_in="Removes binaries/NSS or extended objects from Gaia columns so they do not spoil star matching.",
            compute="Boolean filter on rows before matching.",
        )
        p01_chip = st.slider(
            "phase01_chip_interior_margin_px",
            min_value=0,
            max_value=2000,
            value=int(cfg.phase01_chip_interior_margin_px),
            step=5,
        )
        _detail_help(
            "phase01_chip_interior_margin_px",
            phase="Phase 0+1 and suspected LC — chip edge.",
            used_in="Stars closer than margin px to edge are ignored in matching / suspected calculations.",
            compute="Pixel coordinates: x < margin or x > W−margin (similarly y).",
        )

    with tab_tools:
        st.caption("Separate save: each tool has its own Save button.")
        tdao, tphot, tqual = st.tabs(["DAO-STARS / MASTERSTAR", "Photometry (mode)", "Photometry — diagnostics"])
        with tdao:
            ui_dao_stars.render_dao_stars_dashboard(
                cfg, pipeline=pipeline, draft_dir_override=draft_dir_override
            )
        with tphot:
            ui_photometry.render_photometry_dashboard(cfg)
        with tqual:
            from ui_photometry_quality import render_photometry_quality_diagnostic

            render_photometry_quality_diagnostic(
                pipeline=pipeline,
                draft_id=st.session_state.get("vyvar_last_draft_id"),
            )

    st.markdown("---")
    if st.button("Save main settings to config.json", type="primary", key="vyvar_settings_master_save"):
        cfg.archive_root = Path(archive_root)
        cfg.calibration_library_root = Path(calib_root)
        cfg.database_path = Path(db_path)
        cfg.masterdark_validity_days = int(new_dark)
        cfg.masterflat_validity_days = int(new_flat)
        cfg.calibration_library_native_binning = None if _cln_none else int(new_cl_bin)
        cfg.gaia_db_path = str(gaia_db_path).strip()
        cfg.blind_index_fine_path = str(blind_index_fine_path).strip()
        cfg.blind_index_wide_path = str(blind_index_wide_path).strip()
        cfg.blind_index_path = cfg.blind_index_fine_path
        cfg.vsx_local_db_path = str(vsx_db_path).strip()
        cfg.vsx_variable_targets_mag_limit = float(vsx_mag_limit_save)
        cfg.qc_max_hfr = float(qc_hfr)
        cfg.qc_min_stars = int(qc_stars)
        cfg.qc_after_calibrate_enabled = bool(qc_after_cal)
        cfg.alignment_max_stars = int(max(10, min(5000, aln_max)))
        det_sig = float(aln_sig)
        cfg.alignment_detection_sigma = det_sig if det_sig > 0 else 5.0
        cfg.aperture_fwhm_factor = float(max(0.5, min(6.0, ap_fwhm)))
        cfg.annulus_inner_fwhm = float(max(1.0, min(10.0, ann_in)))
        cfg.annulus_outer_fwhm = float(max(1.5, min(12.0, ann_out)))
        if cfg.annulus_outer_fwhm <= cfg.annulus_inner_fwhm:
            cfg.annulus_outer_fwhm = cfg.annulus_inner_fwhm + 1.0
        cfg.nonlinearity_peak_percentile = float(max(0.0, min(50.0, nl_pct)))
        cfg.nonlinearity_fwhm_ratio = float(max(1.01, min(3.0, nl_ratio)))
        cfg.aperture_variable_factor = float(max(0.5, min(2.0, aperture_variable_factor)))
        cfg.aperture_comp_factor = float(max(0.5, min(2.0, aperture_comp_factor)))
        cfg.temporal_binning_enabled = bool(temporal_binning_enabled)
        cfg.pytics_enabled = bool(pytics_enabled)
        cfg.savgol_detrend_enabled = bool(savgol_detrend_enabled)
        cfg.democratic_detrend_enabled = bool(democratic_detrend_enabled)
        cfg.comp_slope_significance_k = float(max(0.0, min(10.0, comp_slope_sig_k)))
        cfg.lc_quality_short_min_frames = int(max(2, min(100, lc_q_short)))
        if cfg.lc_quality_short_min_frames > cfg.lc_quality_min_frames:
            cfg.lc_quality_short_min_frames = cfg.lc_quality_min_frames
        cfg.lc_quality_min_normal_frac = float(max(0.1, min(1.0, lc_q_frac)))
        cfg.comp_trust_min_comps = int(max(3, min(20, comp_trust_min)))
        if cfg.comp_trust_min_comps > cfg.phase01_comparison_n_comp_max:
            cfg.comp_trust_min_comps = int(cfg.phase01_comparison_n_comp_max)
        cfg.check_star_min_epochs = int(max(3, min(50, chk_min_epochs)))
        cfg.frame_quality_gate_enabled = bool(frame_quality_gate_enabled)
        cfg.frame_quality_ratio_k = float(max(2.0, min(20.0, frame_quality_ratio_k)))
        cfg.frame_quality_fwhm_factor = float(max(0.8, min(3.0, frame_quality_fwhm_factor)))
        cfg.frame_quality_min_keep_frames = int(max(3, min(100000, frame_quality_min_keep_frames)))

        cfg.phase01_comparison_max_dist_deg = float(max(0.05, min(10.0, p01_md)))
        cfg.phase01_comparison_max_mag_diff = float(max(0.05, min(5.0, p01_mm)))
        cfg.phase01_comparison_mag_bright_threshold = float(max(6.0, min(18.0, p01_mag_b)))
        cfg.phase01_comparison_max_mag_diff_bright_floor = float(max(0.0, min(4.0, p01_mag_bf)))
        cfg.phase01_comparison_max_mag_diff_absolute = float(max(1.5, min(5.0, p01_mag_abs)))
        cfg.comp_tier1_bprp_limit = float(max(0.05, min(0.50, tier1_bprp)))
        cfg.comp_tier2_bprp_limit = float(max(0.10, min(0.80, tier2_bprp)))
        cfg.comp_tier3_bprp_limit = float(max(0.20, min(1.20, tier3_bprp)))
        cfg.comp_tier4_bprp_limit = float(max(0.50, min(2.00, tier4_bprp)))
        cfg.comp_max_delta_bprp = float(max(0.20, min(2.0, comp_dbprp)))
        cfg.comp_tier1_weight = float(max(0.50, min(1.00, tier1_w)))
        cfg.comp_tier2_weight = float(max(0.50, min(1.00, tier2_w)))
        cfg.comp_tier3_weight = float(max(0.10, min(0.75, tier3_w)))
        cfg.comp_tier4_weight = float(max(0.05, min(0.50, tier4_w)))
        cfg.phase01_comparison_n_comp_min = int(max(2, min(12, p01_ncmin)))
        cfg.phase01_comparison_n_comp_max = int(max(3, min(20, p01_ncmax)))
        if cfg.phase01_comparison_n_comp_max < cfg.phase01_comparison_n_comp_min:
            cfg.phase01_comparison_n_comp_max = cfg.phase01_comparison_n_comp_min
        cfg.phase01_comparison_max_comp_rms = float(max(0.01, min(0.5, p01_rms)))
        cfg.comp_sparse_fallback_enabled = bool(p01_sparse_fb)
        cfg.comp_iterative_clip_enabled = bool(p01_sparse_fb)
        _fb_min_raw = int(p01_sparse_fb_min)
        cfg.comp_sparse_fallback_min = 0 if _fb_min_raw == int(cfg.phase01_comparison_n_comp_min) else _fb_min_raw
        cfg.comp_clip_sigma = float(max(3.0, min(10.0, p01_clip_sigma)))
        cfg.phase01_comparison_min_dist_arcsec = float(max(0.0, min(600.0, p01_mind)))
        cfg.phase01_comparison_min_frames_frac = float(max(0.05, min(0.95, p01_mff)))
        cfg.phase01_comparison_exclude_gaia_nss = bool(p01_ex_nss)
        cfg.phase01_comparison_exclude_gaia_extobj = bool(p01_ex_ext)
        cfg.phase01_chip_interior_margin_px = int(max(0, min(2000, p01_chip)))

        save_config_json(cfg.project_root, cfg.to_json())
        cfg.ensure_base_dirs()
        st.success("Saved to `config.json`. Refreshing UI…")
        st.rerun()

    st.caption(
        "Plate-solve FOV, part of scale, and per-frame match separations may be derived from **FITS + WCS + DB** "
        "(not always in JSON) — see MASTERSTAR block in Overview."
    )
