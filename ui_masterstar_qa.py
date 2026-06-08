"""MASTERSTAR QA: projekcia masterstars cez WCS, metriky a náhľad mapy (DAO / MATCH / Gaia)."""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from warnings import catch_warnings, simplefilter

import numpy as np
import pandas as pd
import streamlit as st

from platesolve_ui_paths import cone_csv_path, default_bundle_dir, masterstars_csv_in_dir, platesolve_bundle_dirs
from config import AppConfig
from gaia_catalog_id import normalize_gaia_source_id, read_vyvar_csv
from pipeline import AstroPipeline
from utils import resolve_draft_dir


@st.cache_data(show_spinner="Loading VSX from local database…")
def _cached_msqa_vsx_chip_table(
    _fits_mtime: float,
    _vsx_mtime: float,
    fits_path: str,
    vsx_db_path: str,
    plate_solve_fov_deg: float,
) -> pd.DataFrame:
    """VSX v kuželi ako pri katalógu + world_to_pixel do súradníc snímku.

    ``_fits_mtime`` / ``_vsx_mtime`` sú súčasť kľúča cache (Streamlit), nie použité v tele funkcie.
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astropy.io import fits
    from astropy.wcs import FITSFixedWarning, WCS

    from pipeline import _effective_field_catalog_cone_radius_deg, _query_vsx_local

    fp = Path(fits_path)
    vp = Path(vsx_db_path)
    if not fp.is_file() or not vp.is_file():
        return pd.DataFrame()
    with fits.open(fp, memmap=False) as hdul:
        hdr = hdul[0].header
        raw = hdul[0].data
        h0, w0 = int(raw.shape[0]), int(raw.shape[1])
    with catch_warnings():
        simplefilter("ignore", FITSFixedWarning)
        w = WCS(hdr)
    if not getattr(w, "has_celestial", False):
        return pd.DataFrame()
    center, r_deg = _effective_field_catalog_cone_radius_deg(
        w, h0, w0, float(plate_solve_fov_deg), fits_header=hdr
    )
    vdf = _query_vsx_local(center=center, radius_deg=float(r_deg), vsx_db_path=vp)
    if vdf is None or vdf.empty:
        return pd.DataFrame()
    vdf = vdf.copy()
    coo = SkyCoord(
        ra=pd.to_numeric(vdf["ra_deg"], errors="coerce").astype(float).values * u.deg,
        dec=pd.to_numeric(vdf["dec_deg"], errors="coerce").astype(float).values * u.deg,
        frame="icrs",
    )
    xp, yp = w.world_to_pixel(coo)
    vdf["x"] = np.asarray(xp, dtype=np.float64)
    vdf["y"] = np.asarray(yp, dtype=np.float64)
    fin = np.isfinite(vdf["x"].to_numpy()) & np.isfinite(vdf["y"].to_numpy())
    xn, yn = vdf["x"].to_numpy(), vdf["y"].to_numpy()
    inb = (xn >= 0) & (yn >= 0) & (xn < float(w0)) & (yn < float(h0))
    vdf = vdf.loc[fin & inb].reset_index(drop=True)
    mmax = pd.to_numeric(vdf.get("mag_max"), errors="coerce")
    mmin = pd.to_numeric(vdf.get("mag_min"), errors="coerce")
    vdf["mag_eff"] = mmax.where(mmax.notna(), mmin)
    return vdf


def _masterstar_reference_basename(
    setup_dir: Path,
    *,
    ap_path: Path | None,
    draft_id: int | None,
    pipeline: AstroPipeline | None,
) -> tuple[str | None, str]:
    """Resolve the light-frame basename used to build ``MASTERSTAR.fits``."""
    ref: str | None = None
    via: list[str] = []

    if draft_id is not None and pipeline is not None:
        try:
            mp = pipeline.db.get_obs_draft_masterstar_source_path(int(draft_id))
            if mp:
                ref = Path(str(mp)).name
                via.append("db")
        except Exception:  # noqa: BLE001
            pass

    ms_fits = setup_dir / "MASTERSTAR.fits"
    if ref is None and ms_fits.is_file():
        try:
            from astropy.io import fits

            with fits.open(ms_fits, memmap=False) as hdul:
                hdr = hdul[0].header
            for key in (
                "VY_REFFILE",
                "VY_SRCFILE",
                "VY_SRCPATH",
                "ORIGNAME",
                "ORIGFILE",
                "FILENAME",
            ):
                raw = hdr.get(key)
                if raw is None:
                    continue
                if isinstance(raw, tuple):
                    raw = raw[0]
                s = str(raw).strip()
                if s:
                    ref = Path(s).name
                    via.append(f"header:{key}")
                    break
        except Exception:  # noqa: BLE001
            pass

    cand_csv = setup_dir / "masterstar_candidates.csv"
    if ref is None and cand_csv.is_file():
        try:
            cdf = pd.read_csv(cand_csv, low_memory=False)
            if not cdf.empty:
                for sc in ("rank", "RANK", "qc_rank", "overall_rank", "score", "SCORE"):
                    if sc in cdf.columns:
                        cdf = cdf.sort_values(sc, ascending=str(sc).lower() in ("rank", "qc_rank"))
                        break
                for pc in ("FILE_PATH", "file_path", "path", "FILE", "filename", "basename", "processed_path"):
                    if pc in cdf.columns:
                        s0 = str(cdf.iloc[0].get(pc) or "").strip()
                        if s0:
                            ref = Path(s0).name
                            via.append("masterstar_candidates.csv")
                        break
        except Exception:  # noqa: BLE001
            pass

    if ref is None and ap_path is not None:
        infologs = sorted(ap_path.glob("infolog_*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if infologs:
            try:
                for line in infologs[0].read_text(encoding="utf-8", errors="replace").splitlines():
                    if "MASTERSTAR: čistá kópia" in line and "zdroj" in line:
                        m = re.search(r"zdroj\s+(\S+\.fits)", line, flags=re.IGNORECASE)
                        if m:
                            ref = m.group(1).strip()
                            via.append("infolog")
                            break
            except Exception:  # noqa: BLE001
                pass

    note = "+".join(via) if via else "unresolved"
    return ref, note


def _resolve_masterstar_proc_overlay_csv(
    setup_dir: Path,
    *,
    ap_path: Path | None,
    draft_id: int | None,
    pipeline: AstroPipeline | None,
) -> tuple[Path | None, str]:
    """Return ``proc_*.csv`` for the MASTERSTAR reference frame (per-frame catalog)."""
    ref_bn, ref_via = _masterstar_reference_basename(
        setup_dir, ap_path=ap_path, draft_id=draft_id, pipeline=pipeline
    )
    idx_path = setup_dir / "per_frame_catalog_index.csv"
    if idx_path.is_file():
        try:
            idf = pd.read_csv(idx_path, low_memory=False)
            if not idf.empty and "csv" in idf.columns:
                files = idf["file"].astype(str).str.strip() if "file" in idf.columns else pd.Series(dtype=str)
                if ref_bn:
                    ref_l = ref_bn.strip().lower()
                    ref_stem = Path(ref_bn).stem.lower()
                    hit = files.str.lower().eq(ref_l)
                    if not bool(hit.any()):
                        hit = files.str.lower().str.endswith(ref_stem + ".fits") | files.str.lower().eq(
                            "proc_" + ref_stem + ".fits"
                        )
                    if bool(hit.any()):
                        row = idf.loc[hit].iloc[0]
                        csv_p = Path(str(row["csv"]).strip())
                        if csv_p.is_file():
                            return csv_p, f"per_frame_catalog_index ({ref_via})"
                ms_hit = files.str.upper().str.contains("MASTERSTAR")
                if bool(ms_hit.any()):
                    row = idf.loc[ms_hit].iloc[0]
                    csv_p = Path(str(row["csv"]).strip())
                    if csv_p.is_file():
                        return csv_p, "per_frame_catalog_index (MASTERSTAR row)"
        except Exception:  # noqa: BLE001
            pass

    if ref_bn and ap_path is not None:
        stem = Path(ref_bn).stem
        proc_stem = stem if stem.lower().startswith("proc_") else f"proc_{stem}"
        for root in (
            ap_path / "detrended_aligned" / "lights" / setup_dir.name,
            ap_path / "detrended_aligned" / "lights",
        ):
            if not root.is_dir():
                continue
            for name in (f"{proc_stem}.csv", f"{stem}.csv"):
                p = root / name
                if p.is_file():
                    return p, f"aligned lights ({ref_via})"
    return None, ref_via


def _infer_setup_name_from_masterstar_candidate_path(p: Path) -> str | None:
    """Infer setup name (e.g. R_60_1) from a selected MASTERSTAR candidate light frame path."""
    try:
        parts = [pp for pp in p.parts]
        for i in range(len(parts) - 1):
            if parts[i].lower() == "lights":
                cand = parts[i + 1]
                if re.fullmatch(r"[A-Za-z]+_\d+_\d+", cand):
                    return cand
        # Fallback: any segment that looks like SETUP
        for seg in parts:
            if re.fullmatch(r"[A-Za-z]+_\d+_\d+", seg):
                return seg
    except Exception:  # noqa: BLE001
        return None
    return None


def render_masterstar_qa(
    *,
    cfg: AppConfig,
    draft_id: int | None,
    pipeline: AstroPipeline,
    draft_dir_override: Path | None = None,
) -> None:
    if st.session_state.pop("vyvar_masterstar_qa_force_refresh", False):
        st.rerun()
    st.subheader("MASTERSTARS Diagnostic")
    st.caption(
        "**Goal:** verify that the **MASTERSTAR catalog** (masterstars_full_match.csv) projects reliably "
        "through WCS from **MASTERSTAR.fits** — record counts, match rate, and Gaia query radius check."
    )

    # Draft root: override → Drafts/draft_{id} → session archives (MASTERSTAR QA order).
    _session_draft_id = st.session_state.get("vyvar_last_draft_id")
    _did = _session_draft_id if _session_draft_id is not None else draft_id
    ap_path: Path | None = None
    if draft_dir_override is not None and Path(draft_dir_override).is_dir():
        ap_path = Path(draft_dir_override).resolve()
    else:
        _resolved = resolve_draft_dir(
            None,
            _did,
            cfg.archive_root,
            drafts_before_session=True,
        )
        if _resolved:
            ap_path = Path(_resolved)
        if ap_path is None or not ap_path.is_dir():
            last_res = st.session_state.get("vyvar_last_import_result")
            if last_res and getattr(last_res, "archive_path", None):
                ap_path = Path(str(last_res.archive_path))

    if ap_path is not None and ap_path.is_dir():
        st.caption(f"Active draft: `{ap_path}`")
    else:
        default_ap = ""
        _resolved_default = resolve_draft_dir(
            draft_dir_override,
            _did,
            cfg.archive_root,
            drafts_before_session=True,
        )
        if _resolved_default:
            default_ap = _resolved_default
        elif _did is not None:
            try:
                default_ap = str(
                    (Path(cfg.archive_root) / "Drafts" / f"draft_{int(_did):06d}").resolve()
                )
            except (TypeError, ValueError):
                default_ap = ""
        ap_manual = st.text_input(
            "Archive path (draft)",
            value=default_ap,
            key="vyvar_masterstar_ap",
        )
        if not ap_manual.strip():
            st.info(
                "No active draft. Load a draft above (**Load draft**), run **RUN VYVAR**, "
                "or enter a path to `.../Archive/Drafts/draft_XXXXXX`."
            )
            return
        ap_path = Path(ap_manual.strip())
    ps_root = ap_path / "platesolve"
    bundles = platesolve_bundle_dirs(ps_root)
    if not bundles:
        st.warning(
            f"No complete MASTERSTAR bundle in `{ps_root}` "
            "(``MASTERSTAR.fits`` + ``masterstars_full_match.csv`` / ``masterstars.csv`` in ``platesolve/`` or ``platesolve/<filter>/``). "
            "In step 3 enable MASTERSTAR and run, or use **MASTERSTAR only**."
        )
        return

    # If Quality Dashboard selected a MASTERSTAR candidate, prefer that setup here.
    preferred_setup: str | None = None
    masterstar_candidate_path: Path | None = None
    if draft_id is not None:
        try:
            mp = pipeline.db.get_obs_draft_masterstar_source_path(int(draft_id))
            if mp:
                masterstar_candidate_path = Path(str(mp))
                preferred_setup = _infer_setup_name_from_masterstar_candidate_path(masterstar_candidate_path)
        except Exception:  # noqa: BLE001
            preferred_setup = None

    if masterstar_candidate_path is not None and masterstar_candidate_path.is_file():
        st.caption(f"MASTERSTAR selected in Quality Dashboard: `{masterstar_candidate_path.name}`")
        if preferred_setup:
            st.caption(f"Preferred platesolve setup: **{preferred_setup}**")

    if len(bundles) > 1:
        names = [p.name for p in bundles]
        pref = default_bundle_dir(ps_root, preferred_name=preferred_setup)
        pref_nm = pref.name if pref is not None else names[0]
        ix = names.index(pref_nm) if pref_nm in names else 0
        pick_nm = st.selectbox(
            "Filter / group (platesolve):",
            options=names,
            index=ix,
            key="vyvar_msqa_platesolve_setup",
        )
        setup_dir = ps_root / pick_nm
    else:
        setup_dir = bundles[0]
        st.caption(f"Platesolve setup: **{setup_dir.name}**")

    fits_path_ms = setup_dir / "MASTERSTAR.fits"
    csv_path_ms = masterstars_csv_in_dir(setup_dir)
    if not fits_path_ms.is_file():
        st.warning(
            f"Missing `{fits_path_ms}`. In step 3 enable MASTERSTAR and run, or use **MASTERSTAR only**."
        )
        return

    if csv_path_ms is None or not csv_path_ms.is_file():
        st.warning(f"Missing masterstars CSV in `{setup_dir}`. Run MASTERSTAR catalog build.")
        return

    try:
        from astropy.io import fits
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        from astropy.wcs import FITSFixedWarning, WCS

        ms_df = read_vyvar_csv(csv_path_ms, low_memory=False)
        if ms_df.empty or "ra_deg" not in ms_df.columns or "dec_deg" not in ms_df.columns:
            st.warning("MASTERSTAR CSV has no ra_deg/dec_deg — mapping not possible.")
            return

        with fits.open(fits_path_ms, memmap=False) as hdul:
            raw = np.asarray(hdul[0].data, dtype=np.float32)
            hdr = hdul[0].header
        h0, w0 = int(raw.shape[0]), int(raw.shape[1])

        with catch_warnings():
            simplefilter("ignore", FITSFixedWarning)
            w = WCS(hdr)
        if not getattr(w, "has_celestial", False):
            st.warning("Current FITS has no usable WCS — mapping not possible.")
            return

        try:
            scales = w.celestial.proj_plane_pixel_scales()
            sx = abs(float(scales[0].to(u.arcsec).value))
            sy = abs(float(scales[1].to(u.arcsec).value))
            diag_deg = math.hypot(float(w0) * sx, float(h0) * sy) / 3600.0
            min_radius_deg = 0.5 * float(diag_deg)
            q_radius_deg = float(hdr.get("VY_GAIR", 0.0) or 0.0)
            if q_radius_deg > 0 and q_radius_deg < min_radius_deg:
                st.error("GAIA QUERY RADIUS TOO SMALL - EDGES WILL FAIL!")
                st.caption(
                    f"Query r={q_radius_deg:.3f}° < half-diagonal={min_radius_deg:.3f}° "
                    f"(diag≈{diag_deg:.3f}°)."
                )
        except Exception:  # noqa: BLE001
            pass

        coo = SkyCoord(
            ra=ms_df["ra_deg"].astype(float).values * u.deg,
            dec=ms_df["dec_deg"].astype(float).values * u.deg,
            frame="icrs",
        )
        xp, yp = w.celestial.world_to_pixel(coo)
        xp = np.asarray(xp, dtype=np.float64)
        yp = np.asarray(yp, dtype=np.float64)

        finite_map = np.isfinite(xp) & np.isfinite(yp)
        ms_all = ms_df.loc[finite_map].copy().reset_index(drop=True)

        n_ok = (
            int(ms_all["catalog_id"].map(normalize_gaia_source_id).astype(str).str.strip().ne("").sum())
            if "catalog_id" in ms_all.columns
            else 0
        )
        n_all = int(len(ms_all))
        mpix = float(max(1.0, (float(w0) * float(h0)) / 1_000_000.0))
        ref_density = float(n_ok) / mpix
        match_rate = (100.0 * float(n_ok) / float(n_all)) if n_all > 0 else 0.0
        _cone_csv = setup_dir / "field_catalog_cone.csv"
        _cone_rows: int | None = None
        if _cone_csv.is_file():
            try:
                _cone_rows = len(read_vyvar_csv(_cone_csv, low_memory=False))
            except Exception:  # noqa: BLE001
                pass
        # TODO-25: pipeline-computed value (same formula as detect_stars_and_match_catalog)
        _gaia_dao_pct: float | None = None
        _n_gaia_undetected: int | None = None
        _meta_catalog_rows: int | None = None
        _meta_json = setup_dir / "photometry" / "pipeline_meta.json"
        if _meta_json.is_file():
            try:
                _meta = json.loads(_meta_json.read_text(encoding="utf-8"))
                _pct_raw = _meta.get("gaia_dao_completeness_pct")
                if _pct_raw is not None:
                    _gaia_dao_pct = float(_pct_raw)
                _nu_raw = _meta.get("n_gaia_undetected")
                if _nu_raw is not None:
                    _n_gaia_undetected = int(_nu_raw)
                _cr_raw = _meta.get("catalog_rows")
                if _cr_raw is not None:
                    _meta_catalog_rows = int(_cr_raw)
            except Exception:  # noqa: BLE001
                pass
        if _gaia_dao_pct is None and _cone_rows and _cone_rows > 0:
            _n_dao_detected_fb = n_ok
            _gaia_dao_pct = 100.0 * float(_n_dao_detected_fb) / float(_cone_rows)
            _n_gaia_undetected = int(_cone_rows) - int(_n_dao_detected_fb)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Stars mapped in frame", n_all)
        with m2:
            st.metric("DAO→Gaia Match (%)", f"{match_rate:.2f}")
        with m3:
            if _gaia_dao_pct is not None:
                _denom = _meta_catalog_rows if _meta_catalog_rows else _cone_rows
                _undetected = (
                    int(_n_gaia_undetected)
                    if _n_gaia_undetected is not None and _denom
                    else (int(_denom) - int(n_ok) if _denom else 0)
                )
                _help_detected = (
                    int(_denom) - _undetected if _denom and _n_gaia_undetected is not None else int(n_ok)
                )
                st.metric(
                    "Gaia→DAO Completeness (%)",
                    f"{_gaia_dao_pct:.1f}",
                    help=(
                        f"{_help_detected} of {_denom or '?'} Gaia catalog stars detected by DAO\n"
                        f"Undetected (catalog_only): {_undetected}"
                    ),
                )
            else:
                st.metric("Gaia→DAO Completeness (%)", "—")
        with m4:
            st.metric("Reference Star Density (stars/MPix)", f"{ref_density:.1f}")
            if ref_density > 1500.0:
                st.markdown(
                    f"<div style='color:#39FF14;font-weight:800;'>Reference Star Density: {ref_density:.1f} (MASTERSTAR LOCK)</div>",
                    unsafe_allow_html=True,
                )
        st.caption(
            f"With **catalog_id**: **{n_ok}** · without **catalog_id**: **{max(0, n_all - n_ok)}** "
            f"(all rows with finite WCS projection)."
        )
        if _gaia_dao_pct is not None:
            if _gaia_dao_pct >= 90.0:
                st.caption("✅ Completeness: EXCELLENT (≥90% Gaia stars detected)")
            elif _gaia_dao_pct >= 80.0:
                st.caption("⚠️ Completeness: GOOD (80–90%)")
            else:
                st.caption("❌ Completeness: LOW (<80%) — consider TODO-13 2-pass DAO")
        elif _cone_rows and _cone_rows > 0:
            st.caption("Gaia→DAO: pipeline_meta.json missing — recomputed from MASTERSTAR rows")
        else:
            st.caption("Gaia→DAO: field_catalog_cone.csv unavailable")
        if match_rate >= 90.0:
            st.markdown(
                "<div style='color:#39FF14;font-weight:900;font-size:1.2rem;'>Astrometry quality: EXCELLENT (Lock OK)</div>",
                unsafe_allow_html=True,
            )

        in_frame = finite_map & (xp >= 0) & (yp >= 0) & (xp < w0) & (yp < h0)
        xp2 = xp[in_frame]
        yp2 = yp[in_frame]
        ms2 = ms_df.loc[in_frame].copy().reset_index(drop=True)
        if "x" in ms2.columns and "y" in ms2.columns:
            ms2["x_meas"] = pd.to_numeric(ms2["x"], errors="coerce")
            ms2["y_meas"] = pd.to_numeric(ms2["y"], errors="coerce")
        else:
            ms2["x_meas"] = np.nan
            ms2["y_meas"] = np.nan
        ms2["x"] = xp2
        ms2["y"] = yp2
        ms2["matched"] = (
            ms2["catalog_id"].map(normalize_gaia_source_id).astype(str).str.strip().ne("")
            if "catalog_id" in ms2.columns
            else False
        )
        if "name" in ms2.columns:
            ms2["name"] = ms2["name"].fillna("").astype(str)
        else:
            ms2["name"] = ""

        proc_csv, proc_note = _resolve_masterstar_proc_overlay_csv(
            setup_dir,
            ap_path=ap_path,
            draft_id=draft_id,
            pipeline=pipeline,
        )
        ms_plot = ms2
        if proc_csv is not None and proc_csv.is_file():
            try:
                odf = read_vyvar_csv(proc_csv, low_memory=False)
                if not odf.empty and "x" in odf.columns and "y" in odf.columns:
                    ox = pd.to_numeric(odf["x"], errors="coerce").to_numpy(dtype=np.float64)
                    oy = pd.to_numeric(odf["y"], errors="coerce").to_numpy(dtype=np.float64)
                    o_ok = np.isfinite(ox) & np.isfinite(oy)
                    o_in = o_ok & (ox >= 0) & (oy >= 0) & (ox < float(w0)) & (oy < float(h0))
                    ms_plot = odf.loc[o_in].copy().reset_index(drop=True)
                    ms_plot["x"] = ox[o_in]
                    ms_plot["y"] = oy[o_in]
                    if "x_meas" not in ms_plot.columns:
                        ms_plot["x_meas"] = ms_plot["x"]
                        ms_plot["y_meas"] = ms_plot["y"]
                    if "source_type" not in ms_plot.columns:
                        ms_plot["source_type"] = ""
                    stype = ms_plot["source_type"].fillna("").astype(str).str.strip().str.upper()
                    ms_plot["matched"] = stype.eq("GAIA_MATCHED") | (
                        ms_plot["catalog_id"].fillna("").astype(str).str.strip().ne("")
                        & ~stype.eq("FORCED_APERTURE")
                    )
            except Exception as _proc_exc:  # noqa: BLE001
                st.warning(f"Layer proc CSV read failed ({proc_csv.name}): {_proc_exc}")
                ms_plot = ms2
                proc_note = f"read_error ({proc_note})"
        else:
            proc_note = proc_note or "unresolved"

        st.markdown("### Layer display")
        st.caption(
            "Diagnostic map: **MASTERSTAR.fits** + per-frame **proc CSV** overlay (not masterstars reprojection). "
            "**Green** = `GAIA_MATCHED`, **cyan** = `FORCED_APERTURE`, **red** = `DAO_ONLY` / unmatched (expect 0 after TODO-13). "
            "Summary metrics above still use **masterstars_full_match.csv**. "
            "**VSX** (below) = yellow squares from local SQLite."
        )
        if proc_csv is not None and proc_csv.is_file():
            st.caption(f"Layer source: `{proc_csv.name}` — resolved via **{proc_note}**.")
        else:
            st.warning(
                f"Layer fallback: masterstars WCS projection (proc CSV not found; {proc_note}). "
                "Re-export per-frame catalogs after TODO-13."
            )
        cone_path_ui = cone_csv_path(setup_dir)
        has_field_cat = cone_path_ui.is_file()
        use_meas_xy = st.checkbox(
            "DAO/MATCH from measured x,y in CSV (not ra/dec reprojection)",
            value=False,
            key="vyvar_msqa_dao_measured_xy",
        )
        l1, l2, l3, l4 = st.columns(4)
        with l1:
            show_dao = st.checkbox("Show detections (DAO)", value=True, key="vyvar_msqa_show_dao")
        with l2:
            show_gaia = st.checkbox("Show catalog (GAIA)", value=True, key="vyvar_msqa_show_gaia")
        with l3:
            show_match = st.checkbox("Show matches (MATCH)", value=True, key="vyvar_msqa_show_match")
        with l4:
            show_labels = st.checkbox("Show ID/magnitude", value=False, key="vyvar_msqa_show_labels")

        st.markdown("#### VSX on frame (local DB)")
        from config import AppConfig

        _cfg_msqa = AppConfig()
        _vsx_p = str(getattr(_cfg_msqa, "vsx_local_db_path", "") or "").strip()
        _vsx_path_ok = bool(_vsx_p) and Path(_vsx_p).is_file()
        if not _vsx_path_ok:
            st.caption(
                "Set SQLite VSX path in **Settings** (`vsx_local_db_path`, table `vsx_data`) and test the connection."
            )
            show_vsx = False
            vsx_filt: pd.DataFrame | None = None
            mag_slider = 14.0
        else:
            show_vsx = st.checkbox(
                "Show VSX on frame (yellow squares)",
                value=False,
                key="vyvar_msqa_show_vsx",
            )
            c_v1, c_v2 = st.columns([4, 1])
            with c_v1:
                mag_slider = st.slider(
                    "Limiting magnitude (higher = fainter stars included; max/min from VSX)",
                    min_value=6.0,
                    max_value=18.0,
                    value=14.0,
                    step=0.25,
                    key="vyvar_msqa_vsx_mag_limit",
                    help="Shows records where derived magnitude (mag_max or mag_min) is **≤** this value. "
                    "Entries without magnitude in DB are always shown.",
                )
            with c_v2:
                st.write("")
                st.write("")
                if st.button("Refresh VSX from DB", key="vyvar_msqa_vsx_refresh"):
                    _cached_msqa_vsx_chip_table.clear()
                    st.rerun()
            vsx_chip_all = pd.DataFrame()
            if show_vsx:
                try:
                    _ft = float(fits_path_ms.stat().st_mtime)
                    _vt = float(Path(_vsx_p).stat().st_mtime)
                    vsx_chip_all = _cached_msqa_vsx_chip_table(
                        _ft,
                        _vt,
                        str(fits_path_ms.resolve()),
                        str(Path(_vsx_p).resolve()),
                        float(getattr(_cfg_msqa, "plate_solve_fov_deg", 1.0) or 1.0),
                    )
                except Exception as _vx_exc:  # noqa: BLE001
                    st.warning(f"VSX query failed: {_vx_exc}")
                    vsx_chip_all = pd.DataFrame()
            if show_vsx and not vsx_chip_all.empty and "mag_eff" in vsx_chip_all.columns:
                me = pd.to_numeric(vsx_chip_all["mag_eff"], errors="coerce")
                vsx_filt = vsx_chip_all.loc[me.isna() | (me <= float(mag_slider))].copy()
            elif show_vsx:
                vsx_filt = vsx_chip_all.copy() if not vsx_chip_all.empty else None
            else:
                vsx_filt = None
            if show_vsx:
                st.caption(
                    f"VSX in field (before mag filter): **{len(vsx_chip_all)}** · after slider: **{len(vsx_filt) if vsx_filt is not None else 0}**"
                )

        if "source_type" in ms_plot.columns:
            st_up = ms_plot["source_type"].fillna("").astype(str).str.strip().str.upper()
            n_gaia_m = int(st_up.eq("GAIA_MATCHED").sum())
            n_forced = int(st_up.eq("FORCED_APERTURE").sum())
            n_dao_only = int((~st_up.isin({"GAIA_MATCHED", "FORCED_APERTURE"})).sum())
            st.caption(
                f"Layer in frame (**{len(ms_plot)}** rows): "
                f"GAIA_MATCHED **{n_gaia_m}** · FORCED_APERTURE **{n_forced}** · DAO_ONLY **{n_dao_only}**"
            )
        else:
            n_ok_frame = int(ms_plot["matched"].sum()) if "matched" in ms_plot.columns else 0
            n_in_frame = int(len(ms_plot))
            st.caption(
                f"Green (Gaia matched, in frame): **{n_ok_frame}** · "
                f"Red (DAO only, in frame): **{max(0, n_in_frame - n_ok_frame)}**"
            )

        from masterstar_qa_plot import (
            build_msqa_vsx_plotly_figure,
            build_starfield_qa_png_mapping,
            downsample_array_2d,
            msqa_prepare_vsx_plotly_series,
            percentile_stretch_rgb,
        )

        png_bytes, scx_q, scy_q, note = build_starfield_qa_png_mapping(
            raw,
            hdr,
            ms_plot,
            max_side=1600,
            mark_r=5.0,
            show_labels=bool(show_labels),
            invert=False,
            stretch_lo=1.0,
            stretch_hi=99.0,
            crosshair=False,
            overlay_field_cat=True,
            field_cat_path=cone_path_ui if has_field_cat else None,
            field_cat_mtime=(
                float(cone_path_ui.stat().st_mtime)
                if (has_field_cat and cone_path_ui.is_file())
                else None
            ),
            show_dao=bool(show_dao),
            show_gaia=bool(show_gaia),
            show_match=bool(show_match),
            dao_match_xy_source="measured" if use_meas_xy else "reproj",
            vsx_chip_df=vsx_filt if (show_vsx and vsx_filt is not None and not vsx_filt.empty) else None,
        )
        cap = f"{note}(map Δx×{scx_q:.4f}, Δy×{scy_q:.4f}) — {fits_path_ms.name}"
        _img_key = f"ms_map_{os.path.getmtime(str(csv_path_ms)):.6f}"
        try:
            if has_field_cat and cone_path_ui.is_file():
                _img_key += f"_{os.path.getmtime(str(cone_path_ui)):.6f}"
        except OSError:
            pass
        _img_key += f"_d{int(show_dao)}g{int(show_gaia)}m{int(show_match)}l{int(show_labels)}q{int(use_meas_xy)}"
        _img_key += f"_vx{int(show_vsx)}m{float(mag_slider):.2f}"
        try:
            st.image(png_bytes, caption=cap, width="stretch", key=_img_key)
        except TypeError:
            st.image(png_bytes, caption=cap, width="stretch")

        if show_vsx and vsx_filt is not None and not vsx_filt.empty:
            st.caption("**Interactive VSX preview** — tooltip on hover over yellow square (same downsampling as map above).")
            _disp, _scx_p, _scy_p = downsample_array_2d(raw, 1600)
            _rgb = percentile_stretch_rgb(_disp, 1.0, 99.0)
            _xs, _ys, _nm, _mg, _vt = msqa_prepare_vsx_plotly_series(vsx_filt, _scx_p, _scy_p)
            _fig = build_msqa_vsx_plotly_figure(_rgb, _xs, _ys, names=_nm, mag_labels=_mg, var_types=_vt)
            st.plotly_chart(_fig, width="stretch")
        elif show_vsx and _vsx_path_ok:
            st.info("No VSX stars in this field at the current magnitude limit (or DB empty in cone).")

        with st.expander("WCS / DAO diagnostics (DAO residuals vs Gaia→pixel)", expanded=False):
            st.caption(
                "Same logic as ``scripts/diagnose_masterstar_wcs_dao.py``: round-trip WCS from CSV and "
                "compare centroids to Gaia positions from ``field_catalog_cone.csv`` (if present)."
            )
            worst_n = st.number_input(
                "Worst rows in table",
                min_value=0,
                max_value=100,
                value=12,
                key="vyvar_ms_diag_worst_n",
            )
            if st.button("Run diagnostic", key="vyvar_ms_diag_run"):
                try:
                    from masterstar_wcs_dao_diagnostic import run_masterstar_wcs_dao_diagnostic

                    _report = run_masterstar_wcs_dao_diagnostic(
                        fits_path_ms,
                        csv_path_ms,
                        cone_path=cone_path_ui if cone_path_ui.is_file() else None,
                        worst_n=int(worst_n),
                    )
                    st.code(_report, language=None)
                except Exception as _exc:  # noqa: BLE001
                    st.error(str(_exc))
    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))
