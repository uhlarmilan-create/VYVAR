"""Streamlit dashboard: photometry-related ``AppConfig`` fields."""

from __future__ import annotations

import streamlit as st

from config import AppConfig, save_config_json


def render_photometry_dashboard(cfg: AppConfig) -> None:
    st.subheader("Fotometria")
    st.caption(
        "Apertúra, annulus, saturácia, BPM z master darku a kontrola nelinearity. Hodnoty sa ukladajú do `config.json`."
    )

    aperture_on = st.checkbox(
        "Apertúrna fotometria (photutils kruh + annulus namiesto DAO flux)",
        value=bool(cfg.aperture_photometry_enabled),
        key="vyvar_aperture_photometry_enabled",
    )
    ap_fwhm = st.slider(
        "Priemer apertúry: násobiteľ FWHM",
        min_value=0.5,
        max_value=6.0,
        value=float(cfg.aperture_fwhm_factor),
        step=0.1,
        key="vyvar_aperture_fwhm_factor",
    )
    ann_in = st.slider(
        "Annulus vnútorný okraj (násobiteľ FWHM)",
        min_value=1.0,
        max_value=10.0,
        value=float(cfg.annulus_inner_fwhm),
        step=0.25,
        key="vyvar_annulus_inner_fwhm",
    )
    ann_out = st.slider(
        "Annulus vonkajší okraj (násobiteľ FWHM)",
        min_value=1.5,
        max_value=12.0,
        value=float(cfg.annulus_outer_fwhm),
        step=0.25,
        key="vyvar_annulus_outer_fwhm",
    )
    if ann_out <= ann_in:
        st.warning("Vonkajší annulus musí byť väčší ako vnútorný — pri uložení sa upraví automaticky.")

    st.markdown("---")
    st.subheader("Saturácia")
    pfs = float(cfg.photometry_fallback_saturate_adu) if cfg.photometry_fallback_saturate_adu is not None else 0.0
    photometry_fb_sat = st.number_input(
        "Fallback SATURATE (ADU), ak v FITS chýba",
        min_value=0.0,
        max_value=1.0e9,
        value=pfs,
        step=1.0,
        key="vyvar_photometry_fallback_saturate_adu",
        help="0 = vypnuté. Typicky 65535 pre 16-bit; 16383 pre 14-bit v 16-bit kontajneri.",
    )

    st.markdown("---")
    st.subheader("Nelinearita (QC)")
    nl_pct = st.slider(
        "Horné percentily podľa peak_max_adu (%)",
        min_value=0.0,
        max_value=50.0,
        value=float(cfg.nonlinearity_peak_percentile),
        step=0.5,
        key="vyvar_nonlinearity_peak_percentile",
    )
    nl_ratio = st.slider(
        "Pomer FWHM vs medián poľa (prah)",
        min_value=1.01,
        max_value=3.0,
        value=float(cfg.nonlinearity_fwhm_ratio),
        step=0.01,
        key="vyvar_nonlinearity_fwhm_ratio",
    )

    st.markdown("---")
    st.subheader("BPM (master dark)")
    bpm_sigma = st.slider(
        "MAD násobiteľ pre `*_dark_bpm.json`",
        min_value=2.0,
        max_value=12.0,
        value=float(cfg.bpm_dark_mad_sigma),
        step=0.25,
        key="vyvar_bpm_dark_mad_sigma",
    )

    if st.button("Uložiť fotometriu", type="primary", key="vyvar_save_photometry"):
        cfg.aperture_photometry_enabled = bool(aperture_on)
        cfg.aperture_fwhm_factor = float(max(0.5, min(6.0, ap_fwhm)))
        cfg.annulus_inner_fwhm = float(max(1.0, min(10.0, ann_in)))
        cfg.annulus_outer_fwhm = float(max(1.5, min(12.0, ann_out)))
        if cfg.annulus_outer_fwhm <= cfg.annulus_inner_fwhm:
            cfg.annulus_outer_fwhm = cfg.annulus_inner_fwhm + 1.0
        cfg.photometry_fallback_saturate_adu = (
            float(photometry_fb_sat) if float(photometry_fb_sat) > 0 else None
        )
        cfg.nonlinearity_peak_percentile = float(max(0.0, min(50.0, nl_pct)))
        cfg.nonlinearity_fwhm_ratio = float(max(1.01, min(3.0, nl_ratio)))
        cfg.bpm_dark_mad_sigma = float(max(2.0, min(12.0, bpm_sigma)))
        save_config_json(cfg.project_root, cfg.to_json())
        cfg.ensure_base_dirs()
        st.success("Uložené do `config.json`. Obnovujem UI…")
        st.rerun()
