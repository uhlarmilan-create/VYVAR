"""Reusable Streamlit UI components."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from pipeline import (
    AstroPipeline,
    list_best_processed_light_paths_for_masterstar,
)
from infolog import log_event
import plotly.express as px

DRAFT_CENTER_RA_STATE_KEY = "cur_draft_ra"
DRAFT_CENTER_DE_STATE_KEY = "cur_draft_de"
DRAFT_FOCAL_MM_STATE_KEY = "cur_draft_focal_mm"
DRAFT_PIXEL_UM_STATE_KEY = "cur_draft_pixel_um"


def persist_draft_center_on_change(
    db: Any,
    draft_id: int | None,
    *,
    ra_key: str = DRAFT_CENTER_RA_STATE_KEY,
    de_key: str = DRAFT_CENTER_DE_STATE_KEY,
    focal_key: str = DRAFT_FOCAL_MM_STATE_KEY,
    pixel_key: str = DRAFT_PIXEL_UM_STATE_KEY,
) -> None:
    """Persist status-panel values from session state to ``OBS_DRAFT`` on widget change."""
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
        log_event(f"Draft center on_change save skipped: {exc!s}")


def render_masterstar_selection_qc(
    *,
    pipeline: AstroPipeline,
    draft_id: int | None,
    archive_path: Path | None = None,
    take_n: int = 3,
) -> dict[str, Any]:
    """Tabuľka: ``take_n`` (2–5) najlepších FITS z ``processed/lights`` podľa FWHM; cesty pre MASTERSTAR tlačidlá."""
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
            except Exception:  # noqa: BLE001
                pass

    if _arch is None or not _arch.is_dir():
        st.info("Zadaj platnú **cestu k archívu** vyššie (alebo import s ARCHIVE_PATH v drafte), aby sa našli FITS v `processed/lights`.")
        return {
            "masterstar_candidate_paths": [],
            "masterstar_candidates_table": pd.DataFrame(),
            "masterstar_candidates_n": 0,
            "masterstar_processed_total": 0,
        }

    from pipeline import _iter_fits_recursive, resolve_masterstar_input_root

    try:
        _root = resolve_masterstar_input_root(_arch, setup_name=None)
        _total_proc = len(list(_iter_fits_recursive(_root)))
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
        f"Zobrazených **{len(ranked)}** najlepších súborov z **`processed`** (najnižší FWHM v hlavičke alebo z DB), "
        f"v priečinku je spolu **{_total_proc}** FITS."
    )
    st.caption(
        "Poradie zodpovedá výberu vo **FITS QA** a metrikám kvality (FWHM a pod.), nie starým režimom skladania snímok."
    )

    cand_resolved = [str(p.resolve()) if p.exists() else str(p) for p in ranked]

    show = pd.DataFrame()
    if ranked:
        show = pd.DataFrame({"Súbor (processed)": [p.name for p in ranked]})
        st.dataframe(show, use_container_width=True, hide_index=True)
    else:
        st.warning("V `processed/lights` nie sú žiadne FITS — spusti **MAKE MASTERSTAR** po kalibrácii.")

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
        st.info("Draft ID nie je nastavené.")
        return
    try:
        rows = pipeline.db.fetch_master_sources_for_draft(int(did))
    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))
        return
    if not rows:
        st.info("MASTER_SOURCES je prázdne. Najprv spusti **MAKE MASTERSTAR** v VAR-STREM.")
        return
    df = pd.DataFrame(rows)
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

    st.caption("Heatmap: počet hviezd s `is_safe_comp=1` v matici (mag bin × color bin).")
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
        st.caption("Žiadne dáta pre heatmap (bez platných mag/color binov).")
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
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Heatmap: {exc}")
        st.dataframe(grp, use_container_width=True, hide_index=True)
