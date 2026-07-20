"""Streamlit dashboard: photometry-related ``AppConfig`` fields (vystupy a prepinace)."""

from __future__ import annotations

import streamlit as st

from config import AppConfig, save_config_json, ui_config_persist


def _detail_help(title: str, *, phase: str, used_in: str, compute: str | None = None) -> None:
    with st.expander(f"? {title}", expanded=False):
        st.markdown(f"**Phase / process:** {phase}")
        st.markdown(f"**Where and how it is used:** {used_in}")
        if compute:
            st.markdown(f"**Derivation / computation:** {compute}")


def render_photometry_dashboard(cfg: AppConfig) -> None:
    st.subheader("Photometry")
    st.caption(
        "Aperture, annulus, and nonlinearity are in **Settings -> Photometry (aperture)**. "
        "Output mode toggles are here (save with the button below)."
    )

    st.subheader("Photometry Mode")
    _mode_options = ["aperture", "epsf", "both"]
    _mode_labels = {
        "aperture": "[blue] Aperture only",
        "epsf": "[red] ePSF only",
        "both": "[purple] Both (comparison mode)",
    }
    _current_mode = str(getattr(cfg, "photometry_mode", "both"))
    if _current_mode not in _mode_options:
        _current_mode = "both"

    selected_mode = st.radio(
        "Photometry method",
        options=_mode_options,
        format_func=lambda x: _mode_labels[x],
        index=_mode_options.index(_current_mode),
        horizontal=True,
        key="ui_photometry_mode",
        help=(
            "Aperture: fast, robust, best for isolated stars. "
            "ePSF: PSF-fitting, better for crowded fields. "
            "Both: run both and compare (default)."
        ),
    )

    if selected_mode != _current_mode:
        cfg.photometry_mode = selected_mode
        with ui_config_persist():
            save_config_json(cfg.project_root, cfg.to_json())
        st.info(
            f"Photometry mode set to '{selected_mode}'. "
            "This takes effect on the next RUN VYVAR."
        )

    aperture_on = st.checkbox(
        "Aperture photometry (photutils circle + annulus instead of DAO flux)",
        value=bool(cfg.aperture_photometry_enabled),
        key="vyvar_aperture_photometry_enabled",
        help="Enables circular aperture + sky annulus per FWHM factors from config.",
    )
    _detail_help(
        "aperture_photometry_enabled",
        phase="Phase 2 / lightcurve - flux on the star.",
        used_in="Pipeline chooses between DAO summed flux and `photutils` CircularAperture / CircularAnnulus per this setting.",
        compute="Aperture and annulus radii: `aperture_fwhm_factor`, `annulus_*_fwhm` x local FWHM (Settings).",
    )
    save_png = st.checkbox(
        "Save PNG light curve / field map (Phase 2A)",
        value=bool(cfg.save_lightcurve_png),
        key="vyvar_save_lightcurve_png",
        help="Export charts to disk after Phase 2A run.",
    )
    _detail_help(
        "save_lightcurve_png",
        phase="Phase 2A - result visualization.",
        used_in="Saves PNG for light curve / field map (disk load, useful for QA).",
        compute="No numeric computation - optional matplotlib/plot pipeline export.",
    )
    psf_on = st.checkbox(
        "PSF photometry (experimental; requires masterstar_epsf.fits)",
        value=bool(cfg.psf_photometry_enabled),
        key="vyvar_psf_photometry_enabled",
        help="EPSF fit from MASTERSTAR; heavier, suited to overlapping stars.",
    )
    _detail_help(
        "psf_photometry_enabled",
        phase="Experimental PSF photometry (if enabled in pipeline).",
        used_in="Requires `masterstar_epsf.fits` from MASTERSTAR workflow; PSF model fit instead of simple aperture.",
        compute="EPSF / PSF photometry from pipeline libraries - parameters tied to MASTERSTAR output.",
    )

    st.caption(
        "Saturation threshold: FITS keywords (`SATURATE`, `MAXLIN`, ...), `DATAMAX` / `MAXPIX`, "
        "or `EQUIPMENTS.SATURATE_ADU` when the draft has equipment assigned."
    )

    if st.button("Save photometry", type="primary", key="vyvar_save_photometry"):
        cfg.photometry_mode = str(selected_mode)
        cfg.aperture_photometry_enabled = bool(aperture_on)
        cfg.save_lightcurve_png = bool(save_png)
        cfg.psf_photometry_enabled = bool(psf_on)
        with ui_config_persist():
            save_config_json(cfg.project_root, cfg.to_json())
        cfg.ensure_base_dirs()
        st.success("Saved to `config.json`. Refreshing UI...")
        st.rerun()
