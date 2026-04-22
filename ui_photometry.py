"""Streamlit dashboard: photometry-related ``AppConfig`` fields (výstupy a prepínače)."""

from __future__ import annotations

import streamlit as st

from config import AppConfig, save_config_json


def _detail_help(title: str, *, phase: str, used_in: str, compute: str | None = None) -> None:
    with st.expander(f"❓ {title}", expanded=False):
        st.markdown(f"**Fáza / proces:** {phase}")
        st.markdown(f"**Kde a ako sa používa:** {used_in}")
        if compute:
            st.markdown(f"**Odvodenie / výpočet:** {compute}")


def render_photometry_dashboard(cfg: AppConfig) -> None:
    st.subheader("Fotometria")
    st.caption(
        "Apertúra, annulus a nelinearita sú v **Nastavenia → Fotometria (apertúra)**. "
        "Tu sú prepínače režimu výstupov (uloženie vlastným tlačidlom nižšie)."
    )

    aperture_on = st.checkbox(
        "Apertúrna fotometria (photutils kruh + annulus namiesto DAO flux)",
        value=bool(cfg.aperture_photometry_enabled),
        key="vyvar_aperture_photometry_enabled",
        help="Zapína kruhovú apertúru + sky annulus podľa FWHM faktorov z config.",
    )
    _detail_help(
        "aperture_photometry_enabled",
        phase="Fáza 2 / lightcurve — výpočet fluxu na hviezde.",
        used_in="Pipeline volí medzi DAO sumárny flux a `photutils` CircularAperture / CircularAnnulus podľa tejto voľby.",
        compute="Polomery apertúry a annulusu: `aperture_fwhm_factor`, `annulus_*_fwhm` × lokálne FWHM (Nastavenia).",
    )
    save_png = st.checkbox(
        "Ukladať PNG lightcurve / field map (Fáza 2A)",
        value=bool(cfg.save_lightcurve_png),
        key="vyvar_save_lightcurve_png",
        help="Export grafov na disk po behu fázy 2A.",
    )
    _detail_help(
        "save_lightcurve_png",
        phase="Fáza 2A — vizualizácia výsledkov.",
        used_in="Ukladá PNG pre lightcurve / mapu poľa (záťaž na disk, užitočné na kontrolu).",
        compute="Žiadny numerický výpočet — len voliteľný export matplotlib/plot pipeline.",
    )
    psf_on = st.checkbox(
        "PSF fotometria (experimentálne; vyžaduje masterstar_epsf.fits)",
        value=bool(cfg.psf_photometry_enabled),
        key="vyvar_psf_photometry_enabled",
        help="EPSF fit z MASTERSTAR; náročnejšie, vhodné pri prekrývajúcich sa hviezdách.",
    )
    _detail_help(
        "psf_photometry_enabled",
        phase="Experimentálna PSF fotometria (ak je v pipeline zapojená).",
        used_in="Vyžaduje `masterstar_epsf.fits` z MASTERSTAR workflow; fitting PSF modelu namiesto jednoduchej apertúry.",
        compute="EPSF / PSF photometry z knižníc v pipeline — parametre súvisia s MASTERSTAR výstupom.",
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
