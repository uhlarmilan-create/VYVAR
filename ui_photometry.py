"""Streamlit dashboard: photometry-related ``AppConfig`` fields (výstupy a prepínače)."""

from __future__ import annotations

import streamlit as st

from config import AppConfig, save_config_json


def render_photometry_dashboard(cfg: AppConfig) -> None:
    st.subheader("Fotometria")
    st.caption(
        "Apertúra, annulus a nelinearita sa upravujú v **Settings → Fotometria**. "
        "Tu sú prepínače režimu výstupov."
    )

    aperture_on = st.checkbox(
        "Apertúrna fotometria (photutils kruh + annulus namiesto DAO flux)",
        value=bool(cfg.aperture_photometry_enabled),
        key="vyvar_aperture_photometry_enabled",
    )
    save_png = st.checkbox(
        "Ukladať PNG lightcurve / field map (Fáza 2A)",
        value=bool(cfg.save_lightcurve_png),
        key="vyvar_save_lightcurve_png",
    )
    psf_on = st.checkbox(
        "PSF fotometria (experimentálne; vyžaduje masterstar_epsf.fits)",
        value=bool(cfg.psf_photometry_enabled),
        key="vyvar_psf_photometry_enabled",
    )

    st.caption(
        "Prah saturácie: kľúčové slová v FITS (`SATURATE`, `MAXLIN`, …), `DATAMAX` / `MAXPIX`, "
        "prípadne `EQUIPMENTS.SATURATE_ADU` pri drafte so zariadením."
    )

    if st.button("Uložiť fotometriu", type="primary", key="vyvar_save_photometry"):
        cfg.aperture_photometry_enabled = bool(aperture_on)
        cfg.save_lightcurve_png = bool(save_png)
        cfg.psf_photometry_enabled = bool(psf_on)
        save_config_json(cfg.project_root, cfg.to_json())
        cfg.ensure_base_dirs()
        st.success("Uložené do `config.json`. Obnovujem UI…")
        st.rerun()
