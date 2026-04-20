"""MASTERSTAR QA: projekcia masterstars cez WCS, metriky a náhľad mapy (DAO / MATCH / Gaia)."""

from __future__ import annotations

import math
import os
from pathlib import Path
from warnings import catch_warnings, simplefilter

import numpy as np
import pandas as pd
import streamlit as st

from platesolve_ui_paths import cone_csv_path, default_bundle_dir, masterstars_csv_in_dir, platesolve_bundle_dirs


@st.cache_data(show_spinner="Načítavam VSX z lokálnej databázy…")
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


def render_masterstar_qa() -> None:
    if st.session_state.pop("vyvar_masterstar_qa_force_refresh", False):
        st.rerun()
    st.subheader("MASTERSTARS Diagnostic")
    st.caption(
        "**Cieľ:** overiť, či sa **MASTERSTAR katalóg** (masterstars_full_match.csv) dá spoľahlivo "
        "promítať cez WCS z **MASTERSTAR.fits** — počty záznamov, match rate a kontrola polomeru Gaia dotazu."
    )

    last_res = st.session_state.get("vyvar_last_import_result")
    default_ap = ""
    if last_res and getattr(last_res, "archive_path", None):
        default_ap = str(last_res.archive_path)
    ap = st.text_input("Cesta k archívu (draft)", value=default_ap, key="vyvar_masterstar_ap")
    if not ap.strip():
        st.info("Zadaj cestu k archívu, napr. `.../Archive/Drafts/draft_000029`.")
        return

    ap_path = Path(ap.strip())
    ps_root = ap_path / "platesolve"
    bundles = platesolve_bundle_dirs(ps_root)
    if not bundles:
        st.warning(
            f"V `{ps_root}` nie je žiadny kompletný MASTERSTAR balík "
            "(``MASTERSTAR.fits`` + ``masterstars_full_match.csv`` / ``masterstars.csv`` v ``platesolve/`` alebo v ``platesolve/<filter>/``). "
            "V kroku 3 zapni MASTERSTAR a spusti, alebo použi **Len MASTERSTAR**."
        )
        return

    if len(bundles) > 1:
        names = [p.name for p in bundles]
        pref = default_bundle_dir(ps_root)
        pref_nm = pref.name if pref is not None else names[0]
        ix = names.index(pref_nm) if pref_nm in names else 0
        pick_nm = st.selectbox(
            "Filter / skupina (platesolve):",
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
            f"Chýba `{fits_path_ms}`. V kroku 3 zapni MASTERSTAR a spusti, alebo použi **Len MASTERSTAR**."
        )
        return

    if csv_path_ms is None or not csv_path_ms.is_file():
        st.warning(f"Chýba masterstars CSV v `{setup_dir}`. Spusti tvorbu MASTERSTAR katalógu.")
        return

    try:
        from astropy.io import fits
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        from astropy.wcs import FITSFixedWarning, WCS

        ms_df = pd.read_csv(csv_path_ms)
        if ms_df.empty or "ra_deg" not in ms_df.columns or "dec_deg" not in ms_df.columns:
            st.warning("MASTERSTAR CSV nemá ra_deg/dec_deg — mapping nie je možný.")
            return

        with fits.open(fits_path_ms, memmap=False) as hdul:
            raw = np.asarray(hdul[0].data, dtype=np.float32)
            hdr = hdul[0].header
        h0, w0 = int(raw.shape[0]), int(raw.shape[1])

        with catch_warnings():
            simplefilter("ignore", FITSFixedWarning)
            w = WCS(hdr)
        if not getattr(w, "has_celestial", False):
            st.warning("Aktuálny FITS nemá použiteľné WCS — mapping nie je možný.")
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

        n_ok = int(ms_all["catalog_id"].fillna("").astype(str).str.strip().ne("").sum()) if "catalog_id" in ms_all.columns else 0
        n_all = int(len(ms_all))
        mpix = float(max(1.0, (float(w0) * float(h0)) / 1_000_000.0))
        ref_density = float(n_ok) / mpix
        match_rate = (100.0 * float(n_ok) / float(n_all)) if n_all > 0 else 0.0
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Mapované hviezdy v snímku", n_all)
        with m2:
            st.metric("Match Rate (%)", f"{match_rate:.2f}")
        with m3:
            st.metric("Reference Star Density (stars/MPix)", f"{ref_density:.1f}")
            if ref_density > 1500.0:
                st.markdown(
                    f"<div style='color:#39FF14;font-weight:800;'>Reference Star Density: {ref_density:.1f} (MASTERSTAR LOCK)</div>",
                    unsafe_allow_html=True,
                )
        st.caption(
            f"S prideleným **catalog_id**: **{n_ok}** · bez **catalog_id**: **{max(0, n_all - n_ok)}** "
            f"(všetky riadky s konečnou WCS projekciou)."
        )
        if match_rate >= 90.0:
            st.markdown(
                "<div style='color:#39FF14;font-weight:900;font-size:1.2rem;'>Kvalita astrometrie: VYNIKAJÚCA (Lock OK)</div>",
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
            ms2["catalog_id"].fillna("").astype(str).str.strip().ne("")
            if "catalog_id" in ms2.columns
            else False
        )
        if "name" in ms2.columns:
            ms2["name"] = ms2["name"].fillna("").astype(str)
        else:
            ms2["name"] = ""

        st.markdown("### Zobrazenie vrstiev")
        st.caption(
            "Diagnostická mapa: **MASTERSTAR.fits** + DAO (červené obrysy) + MATCH/ Gaia matched (zelené) + katalógový kužeľ (modré). "
            "Zelená = riadok s catalog_id; červená samostatne = DAO bez zhody (len ak sú zapnuté obe vrstvy, zelená je navrch). "
            "**VSX** (nižšie) = žlté štvorce z lokálnej SQLite. "
            "**Reprojekcia** = world_to_pixel(ra_deg,dec_deg) z CSV (test WCS vs uložené nebeské súradnice); **merané x,y** = centroidy z CSV pred prepísaním."
        )
        cone_path_ui = cone_csv_path(setup_dir)
        has_field_cat = cone_path_ui.is_file()
        use_meas_xy = st.checkbox(
            "DAO/MATCH z meraných x,y v CSV (nie reprojekcia ra/dec)",
            value=False,
            key="vyvar_msqa_dao_measured_xy",
        )
        l1, l2, l3, l4 = st.columns(4)
        with l1:
            show_dao = st.checkbox("Zobraziť detekcie (DAO)", value=True, key="vyvar_msqa_show_dao")
        with l2:
            show_gaia = st.checkbox("Zobraziť katalóg (GAIA)", value=True, key="vyvar_msqa_show_gaia")
        with l3:
            show_match = st.checkbox("Zobraziť zhody (MATCH)", value=True, key="vyvar_msqa_show_match")
        with l4:
            show_labels = st.checkbox("Zobraziť ID/Jasnosť", value=False, key="vyvar_msqa_show_labels")

        st.markdown("#### VSX na snímku (lokálna DB)")
        from config import AppConfig

        _cfg_msqa = AppConfig()
        _vsx_p = str(getattr(_cfg_msqa, "vsx_local_db_path", "") or "").strip()
        _vsx_path_ok = bool(_vsx_p) and Path(_vsx_p).is_file()
        if not _vsx_path_ok:
            st.caption(
                "Nastav cestu k SQLite VSX v **Settings** (`vsx_local_db_path`, tabuľka `vsx_data`) a otestuj pripojenie."
            )
            show_vsx = False
            vsx_filt: pd.DataFrame | None = None
            mag_slider = 14.0
        else:
            show_vsx = st.checkbox(
                "Zobraziť VSX na snímku (žlté štvorce)",
                value=False,
                key="vyvar_msqa_show_vsx",
            )
            c_v1, c_v2 = st.columns([4, 1])
            with c_v1:
                mag_slider = st.slider(
                    "Limitná magnitúda (väčšia = pribudnú slabšie hviezdy; max/min z VSX)",
                    min_value=6.0,
                    max_value=18.0,
                    value=14.0,
                    step=0.25,
                    key="vyvar_msqa_vsx_mag_limit",
                    help="Zobrazia sa záznamy, kde odvodená magnitúda (mag_max alebo mag_min) je **≤** tejto hodnoty. "
                    "Bez magnitude v DB sa zobrazia vždy.",
                )
            with c_v2:
                st.write("")
                st.write("")
                if st.button("Obnoviť VSX z DB", key="vyvar_msqa_vsx_refresh"):
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
                    st.warning(f"VSX dotaz zlyhal: {_vx_exc}")
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
                    f"VSX v poli (pred filtrom mag): **{len(vsx_chip_all)}** · po slidery: **{len(vsx_filt) if vsx_filt is not None else 0}**"
                )

        n_ok_frame = int(ms2["matched"].sum()) if "matched" in ms2.columns else 0
        n_in_frame = int(len(ms2))
        st.caption(
            f"Zelené (Gaia matched, v rámci snímku): **{n_ok_frame}** · "
            f"Červené (DAO only v rámci snímku): **{max(0, n_in_frame - n_ok_frame)}**"
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
            ms2,
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
            st.image(png_bytes, caption=cap, use_container_width=True, key=_img_key)
        except TypeError:
            st.image(png_bytes, caption=cap, use_container_width=True)

        if show_vsx and vsx_filt is not None and not vsx_filt.empty:
            st.caption("**Interaktívny náhľad VSX** — tooltip pri hoveri nad žltým štvorcom (rovnaký downsampling ako mapa vyššie).")
            _disp, _scx_p, _scy_p = downsample_array_2d(raw, 1600)
            _rgb = percentile_stretch_rgb(_disp, 1.0, 99.0)
            _xs, _ys, _nm, _mg, _vt = msqa_prepare_vsx_plotly_series(vsx_filt, _scx_p, _scy_p)
            _fig = build_msqa_vsx_plotly_figure(_rgb, _xs, _ys, names=_nm, mag_labels=_mg, var_types=_vt)
            st.plotly_chart(_fig, use_container_width=True)
        elif show_vsx and _vsx_path_ok:
            st.info("V tomto poli nie sú žiadne VSX hviezdy pri aktuálnom limite magnitúdy (alebo DB je prázdna v kuželi).")

        with st.expander("WCS / DAO diagnostika (reziduály DAO vs Gaia→pixel)", expanded=False):
            st.caption(
                "Rovnaká logika ako skript ``scripts/diagnose_masterstar_wcs_dao.py``: round-trip WCS z CSV a "
                "porovnanie centroidov s polohou Gaie z ``field_catalog_cone.csv`` (ak existuje)."
            )
            worst_n = st.number_input(
                "Najhorších riadkov v tabuľke",
                min_value=0,
                max_value=100,
                value=12,
                key="vyvar_ms_diag_worst_n",
            )
            if st.button("Spustiť diagnostiku", key="vyvar_ms_diag_run"):
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
