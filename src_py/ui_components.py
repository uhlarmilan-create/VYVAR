"""Reusable Streamlit UI components."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import pandas as pd
import sqlite3
import streamlit as st

from pipeline import (
    AstroPipeline,
    list_best_processed_light_paths_for_masterstar,
)
from infolog import log_event
import plotly.express as px

_log = logging.getLogger(__name__)

DRAFT_CENTER_RA_STATE_KEY = "cur_draft_ra"
DRAFT_CENTER_DE_STATE_KEY = "cur_draft_de"
DRAFT_FOCAL_MM_STATE_KEY = "cur_draft_focal_mm"
DRAFT_PIXEL_UM_STATE_KEY = "cur_draft_pixel_um"

# NOTE: render_masterstar_selection_qc / render_photometric_grid_qa are not wired into app.py (inactive).
# persist_draft_center_on_change and draft center state keys remain active (app.py).
# Last audit: 2026-05-18


def persist_draft_center_on_change(
    db: Any,
    draft_id: int | None,
    *,
    ra_key: str = DRAFT_CENTER_RA_STATE_KEY,
    de_key: str = DRAFT_CENTER_DE_STATE_KEY,
    focal_key: str = DRAFT_FOCAL_MM_STATE_KEY,
    pixel_key: str = DRAFT_PIXEL_UM_STATE_KEY,
) -> None:
    """Persist status-panel values from session state to ``draft manifest`` on widget change."""
    if draft_id is None:
        return
    try:
        ra = float(st.session_state.get(ra_key, float("nan")))
        de = float(st.session_state.get(de_key, float("nan")))
    except (TypeError, ValueError):
        return
    if not (math.isfinite(ra) and math.isfinite(de)):
        return
    focal_v: float | None = None
    pixel_v: float | None = None
    try:
        _foc_raw = st.session_state.get(focal_key, None)
        if _foc_raw is not None:
            _foc = float(_foc_raw)
            if math.isfinite(_foc) and _foc > 0:
                focal_v = float(_foc)
    except (TypeError, ValueError):
        focal_v = None
    try:
        _pix_raw = st.session_state.get(pixel_key, None)
        if _pix_raw is not None:
            _pix = float(_pix_raw)
            if math.isfinite(_pix) and _pix > 0:
                pixel_v = float(_pix)
    except (TypeError, ValueError):
        pixel_v = None
    try:
        db.update_obs_draft_status_panel_values(
            int(draft_id),
            center_ra_deg=float(ra),
            center_de_deg=float(de),
            focal_mm=focal_v,
            pixel_um=pixel_v,
        )
        st.session_state["center_ra"] = float(ra)
        st.session_state["center_de"] = float(de)
        st.session_state["vyvar_last_saved_draft_center_sig"] = f"{int(draft_id)}|{ra:.9f}|{de:.9f}"
    except Exception as exc:  # noqa: BLE001
        # EXC-0507: T3 -- UI diagnostic/plot only (st.session_state['center_de'] = float(de) / st.session_state['... (EXCEPT-BULK 2026-07-08)
        log_event(f"Draft center on_change save skipped: {exc!s}")


def render_masterstar_selection_qc(
    *,
    pipeline: AstroPipeline,
    draft_id: int | None,
    archive_path: Path | None = None,
    take_n: int = 3,
) -> dict[str, Any]:
    """Tabulka: ``take_n`` (2-5) najlepsich FITS z ``processed/lights`` podla FWHM; cesty pre MASTERSTAR tlacidla."""
    did = int(draft_id) if draft_id is not None else None
    tn = max(2, min(5, int(take_n)))

    _arch = archive_path
    if _arch is None or not _arch.is_dir():
        if did is not None:
            try:
                _drow_a = pipeline.db.fetch_obs_draft_by_id(int(did))
                if _drow_a is not None:
                    _raw_ap = str(_drow_a.get("ARCHIVE_PATH") or "").strip()
                    if _raw_ap:
                        _try_p = Path(_raw_ap)
                        if _try_p.is_dir():
                            _arch = _try_p
            except (sqlite3.Error, OSError, AttributeError, TypeError, ValueError) as exc:
                _log.debug(
                    "Archive path probe skipped for draft %s: %s",
                    did,
                    exc,
                )

    if _arch is None or not _arch.is_dir():
        st.info("Enter a valid **archive path** above (or import with ARCHIVE_PATH in the draft) to find FITS in `processed/lights`.")
        return {
            "masterstar_candidate_paths": [],
            "masterstar_candidates_table": pd.DataFrame(),
            "masterstar_candidates_n": 0,
            "masterstar_processed_total": 0,
        }

    from pipeline import _iter_fits_recursive, resolve_masterstar_input_root

    try:
        _root = resolve_masterstar_input_root(_arch, setup_name=None)
        _total_proc = len(list(_iter_fits_recursive(_root))) if _root is not None else 0
    except Exception:  # noqa: BLE001
        _total_proc = 0

    ranked = list_best_processed_light_paths_for_masterstar(
        _arch,
        setup_name=None,
        draft_id=did,
        app_config=pipeline.config,
        take_n=tn,
    )

    st.info(
        f"Showing **{len(ranked)}** best files from **`processed`** (lowest FWHM in header or from DB), "
        f"**{_total_proc}** FITS total in the folder."
    )
    st.caption(
        "Order matches **FITS QA** selection and quality metrics (FWHM, etc.), not legacy image-stacking modes."
    )

    cand_resolved = [str(p.resolve()) if p.exists() else str(p) for p in ranked]

    show = pd.DataFrame()
    if ranked:
        show = pd.DataFrame({"File (processed)": [p.name for p in ranked]})
        st.dataframe(show, width="stretch", hide_index=True)
    else:
        st.warning("No FITS in `processed/lights` - run **MAKE MASTERSTAR** after calibration.")

    return {
        "masterstar_candidate_paths": cand_resolved,
        "masterstar_candidates_table": show,
        "masterstar_candidates_n": int(len(ranked)),
        "masterstar_processed_total": int(_total_proc),
    }


def render_photometric_grid_qa(*, pipeline: AstroPipeline, draft_id: int | None) -> None:
    st.subheader("Photometric Grid QA")
    did = int(draft_id) if draft_id is not None else None
    if did is None:
        st.info("Draft ID is not set.")
        return
    from masterstars_enrichment import ENRICHMENT_COLUMNS, grid_qa_dataframe_from_masterstars_csv
    from utils import resolve_draft_dir

    draft_dir = resolve_draft_dir(pipeline.db, did)
    if draft_dir is None:
        st.info("Draft directory not found.")
        return
    csv_path: Path | None = None
    ps_root = Path(draft_dir) / "platesolve"
    if ps_root.is_dir():
        for setup in sorted(ps_root.iterdir()):
            cand = setup / "masterstars_full_match.csv"
            if cand.is_file():
                csv_path = cand
                break
    if csv_path is None:
        st.info("masterstars_full_match.csv not found. Run **MAKE MASTERSTAR** in VAR-STREM first.")
        return
    try:
        raw = pd.read_csv(csv_path, low_memory=False, dtype={"catalog_id": str})
    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))
        return
    missing = [c for c in ENRICHMENT_COLUMNS if c not in raw.columns]
    if missing:
        st.info(
            "Photometric grid enrichment not available for this draft (pre-retirement). "
            f"Missing columns: {', '.join(missing)}. Re-run MAKE MASTERSTAR to populate."
        )
        return
    df_norm = grid_qa_dataframe_from_masterstars_csv(raw)
    if df_norm is None:
        st.info("Photometric grid enrichment not available for this draft (pre-retirement).")
        return
    df = df_norm
    # Normalize cols
    for c in ("G_MAG", "BP_RP", "STRESS_RMS"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["IS_SAFE_COMP"] = pd.to_numeric(df.get("IS_SAFE_COMP"), errors="coerce").fillna(0).astype(int)
    df["SAFE_OVERRIDE"] = pd.to_numeric(df.get("SAFE_OVERRIDE"), errors="coerce").fillna(0).astype(int)
    df["EXCLUSION_REASON"] = df.get("EXCLUSION_REASON").astype(str) if "EXCLUSION_REASON" in df.columns else ""
    for _fl in ("LIKELY_NONLINEAR", "ON_BAD_COLUMN"):
        if _fl in df.columns:
            df[_fl] = pd.to_numeric(df[_fl], errors="coerce").fillna(0).astype(int)

    st.caption("Heatmap: count of stars with `is_safe_comp=1` in the matrix (mag bin x color bin).")
    safe = df[df["IS_SAFE_COMP"] == 1].copy()
    # Parse bins from PHOT_CATEGORY if available.
    if "PHOT_CATEGORY" in df.columns:
        safe_pc = safe["PHOT_CATEGORY"].astype(str)
        safe["mag_bin"] = pd.to_numeric(safe_pc.str.extract(r"_mag_([0-9]+\\.[0-9])")[0], errors="coerce")
        safe["col_bin"] = pd.to_numeric(safe_pc.str.extract(r"_col_([0-9]+\\.[0-9][0-9])")[0], errors="coerce")
    else:
        safe["mag_bin"] = (safe["G_MAG"] * 2.0).round() / 2.0
        safe["col_bin"] = (safe["BP_RP"] * 4.0).round() / 4.0

    safe = safe.dropna(subset=["mag_bin", "col_bin"])
    if safe.empty:
        st.caption("No heatmap data (no valid mag/color bins).")
        return
    grp = safe.groupby(["mag_bin", "col_bin"], dropna=False).size().reset_index(name="count")
    if grp.empty:
        return
    try:
        fig = px.density_heatmap(
            grp,
            x="col_bin",
            y="mag_bin",
            z="count",
            color_continuous_scale="Viridis",
            title="Safe comparison stars (count per bin)",
        )
        st.plotly_chart(fig, width="stretch")
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Heatmap: {exc}")
        st.dataframe(grp, width="stretch", hide_index=True)
