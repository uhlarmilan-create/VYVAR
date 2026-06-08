"""DAO-STARS: úprava hlavných MASTERSTAR parametrov detekcie / SIP (config.json)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import streamlit as st

from config import AppConfig, save_config_json
from masterstar_context import load_masterstar_context, resolve_masterstar_fits_path
from platesolve_ui_paths import masterstars_csv_in_dir


def _detail_help(title: str, *, phase: str, used_in: str, compute: str | None = None) -> None:
    with st.expander(f"❓ {title}", expanded=False):
        st.markdown(f"**Phase / process:** {phase}")
        st.markdown(f"**Where and how it is used:** {used_in}")
        if compute:
            st.markdown(f"**Derivation / computation:** {compute}")


def _masterstar_detection_count(fits_path: Path, setup_dir: Path | None) -> int | None:
    if setup_dir is not None:
        csv_p = masterstars_csv_in_dir(setup_dir)
        if csv_p is not None and csv_p.is_file():
            try:
                import pandas as pd

                df = pd.read_csv(csv_p, low_memory=False)
                if not df.empty:
                    return int(len(df))
            except Exception:  # noqa: BLE001
                pass
    try:
        from astropy.io import fits

        with fits.open(str(fits_path), memmap=False) as hdul:
            hdr = hdul[0].header
            for key in ("VY_NSTAR", "VY_NSTARS", "VY_NDET", "VY_NDETECT"):
                if key in hdr:
                    try:
                        v = int(float(hdr[key]))
                        if v > 0:
                            return v
                    except (TypeError, ValueError):
                        continue
    except Exception:  # noqa: BLE001
        pass
    return None


def _compute_masterstar_suggestions(
    *,
    fwhm_px: float,
    n_stars: int | None,
    chip_w: int | None,
    chip_h: int | None,
    pixel_scale_arcsec: float | None,
) -> dict[str, float]:
    fwhm = float(fwhm_px) if math.isfinite(float(fwhm_px)) and float(fwhm_px) > 0 else 2.5
    suggest_ap = float(min(2.0, max(1.5, 2.5 / fwhm)))
    n = int(n_stars) if n_stars is not None and int(n_stars) > 0 else 500
    if n > 800:
        suggest_dao = 2.0
    elif n > 400:
        suggest_dao = 2.5
    else:
        suggest_dao = 3.0
    suggest_pre = float(max(0.5, suggest_dao - 0.5))
    ps = float(pixel_scale_arcsec) if pixel_scale_arcsec is not None and math.isfinite(pixel_scale_arcsec) else 1.0
    fw = int(chip_w) if chip_w and int(chip_w) > 0 else 4000
    fh = int(chip_h) if chip_h and int(chip_h) > 0 else 4000
    diag_deg = math.hypot(float(fw) * ps, float(fh) * ps) / 3600.0
    suggest_dist = float(round(diag_deg * 0.6, 2))
    return {
        "aperture_fwhm_factor": suggest_ap,
        "masterstar_dao_threshold_sigma": suggest_dao,
        "masterstar_prematch_peak_sigma_floor": suggest_pre,
        "phase01_comparison_max_dist_deg": suggest_dist,
    }


def render_dao_stars_dashboard(
    cfg: AppConfig,
    *,
    pipeline: Any = None,
    draft_dir_override: Path | None = None,
) -> None:
    st.subheader("DAO-STARS")
    st.caption(
        "**MASTERSTAR** reference frame is chosen in **FITS QA**; detection thresholds and **SIP** "
        "plate-solve range are configured here. Saved to **config.json**."
    )

    cur_sf = float(getattr(cfg, "masterstar_prematch_peak_sigma_floor", 3.2))
    cur_ds = float(getattr(cfg, "masterstar_dao_threshold_sigma", 1.8))

    st.markdown("#### Pre-filter peak (SNR before matching)")
    st.caption(
        "Keeps detections with local peak above **median + k×σ**. **Lower k** → more faint stars before Gaia matching. "
        "Range **0.5–6.0**."
    )
    s_floor = st.slider(
        "masterstar_prematch_peak_sigma_floor (k)",
        min_value=0.5,
        max_value=6.0,
        value=min(max(cur_sf, 0.5), 6.0),
        step=0.1,
        key="vyvar_dao_stars_prematch_k",
        help="SNR filter before Gaia matching: peak > median + k×σ of local background.",
    )
    _detail_help(
        "masterstar_prematch_peak_sigma_floor",
        phase="MASTERSTAR — pre-filter detections before Gaia match.",
        used_in="Reduces faint artifacts before spatial matching; affects prematch speed and stability.",
        compute="Per peak: compare to median + k×σ in the neighborhood.",
    )

    st.markdown("#### DAO threshold (plate-solve + catalog)")
    st.caption(
        "**DAOStarFinder:** threshold = **k × RMS**. **Lower k** → more candidates (including noise). "
        "Same **k** is used in the solver and the subsequent catalog. Range **0.1–6.0**."
    )
    dao_sig = st.slider(
        "masterstar_dao_threshold_sigma (k)",
        min_value=0.1,
        max_value=6.0,
        value=min(max(cur_ds, 0.1), 6.0),
        step=0.05,
        key="vyvar_dao_stars_dao_sigma",
        help="DAOStarFinder: threshold = k × frame RMS.",
    )
    _detail_help(
        "masterstar_dao_threshold_sigma",
        phase="MASTERSTAR plate-solve and star catalog build.",
        used_in="Same sensitivity in solver and WCS/catalog star list; lower k = more detections.",
        compute="DAO threshold as a multiple of local RMS (photutils DAOStarFinder).",
    )

    st.info(
        "With many detections, relax **k** and **DAO σ** only when **WCS is already good** "
        "(low px RMS in Infolog). Poor plate-solve stretches matching to a large **match_sep** — "
        "that skews diagnostics."
    )

    draft_id = st.session_state.get("vyvar_last_draft_id")
    db = getattr(pipeline, "db", None) if pipeline is not None else None
    ms_path = resolve_masterstar_fits_path(
        cfg=cfg, db=db, draft_id=draft_id, draft_dir_override=draft_dir_override
    )
    if st.button("Suggest from MASTERSTAR", key="vyvar_dao_stars_suggest_ms"):
        if ms_path is None or not ms_path.is_file():
            st.warning("No MASTERSTAR.fits for the active draft — run MASTERSTAR / plate-solve first.")
        else:
            ctx = load_masterstar_context(ms_path)
            fwhm = ctx.vy_fwhm_gauss_px or ctx.vy_fwhm_px
            if fwhm is None or not math.isfinite(float(fwhm)):
                try:
                    from ui_aperture_photometry import _load_fwhm

                    fwhm = float(_load_fwhm(ms_path))
                except Exception:  # noqa: BLE001
                    fwhm = float(getattr(cfg, "aperture_fwhm_factor", 1.7)) * 2.0
            setup_dir = ms_path.parent
            n_stars = _masterstar_detection_count(ms_path, setup_dir)
            sug = _compute_masterstar_suggestions(
                fwhm_px=float(fwhm),
                n_stars=n_stars,
                chip_w=ctx.chip_width,
                chip_h=ctx.chip_height,
                pixel_scale_arcsec=ctx.pixel_scale_arcsec,
            )
            st.session_state["vyvar_dao_ms_suggestions"] = sug
            st.session_state["vyvar_dao_ms_suggest_meta"] = {
                "fwhm": float(fwhm),
                "n_stars": int(n_stars) if n_stars is not None else 0,
            }

    _sug = st.session_state.get("vyvar_dao_ms_suggestions")
    _meta = st.session_state.get("vyvar_dao_ms_suggest_meta") or {}
    if isinstance(_sug, dict) and _sug:
        fwhm_d = float(_meta.get("fwhm", float("nan")))
        n_d = int(_meta.get("n_stars", 0))
        st.info(
            f"""Suggested from MASTERSTAR (FWHM={fwhm_d:.2f}px, N={n_d} stars):
· aperture_fwhm_factor: {float(_sug['aperture_fwhm_factor']):.2f}
· dao_threshold_sigma: {float(_sug['masterstar_dao_threshold_sigma']):.1f}
· prematch_peak_sigma: {float(_sug['masterstar_prematch_peak_sigma_floor']):.1f}
· max_dist_deg: {float(_sug['phase01_comparison_max_dist_deg']):.2f}°

Apply these values manually in the sliders above, or click **Apply suggestions** to write to config.json."""
        )
        if st.button("Apply suggestions", key="vyvar_dao_stars_apply_suggestions"):
            cfg.aperture_fwhm_factor = float(_sug["aperture_fwhm_factor"])
            cfg.masterstar_dao_threshold_sigma = float(_sug["masterstar_dao_threshold_sigma"])
            cfg.masterstar_prematch_peak_sigma_floor = float(_sug["masterstar_prematch_peak_sigma_floor"])
            cfg.phase01_comparison_max_dist_deg = float(_sug["phase01_comparison_max_dist_deg"])
            save_config_json(cfg.project_root, cfg.to_json())
            st.success("Suggestions saved to `config.json`. Refresh the page to reload sliders.")
            st.rerun()

    st.markdown("#### SIP at plate-solve (MASTERSTAR)")
    st.caption(
        "Solver tries **from higher order downward** to the lower bound (e.g. 5→4→3). **Min** must not exceed **max**."
    )
    _sip_opts = [2, 3, 4, 5]
    _cur_hi = min(max(int(getattr(cfg, "masterstar_platesolve_sip_max_order", 5)), 2), 5)
    _cur_lo = min(max(int(getattr(cfg, "masterstar_platesolve_sip_min_order", 3)), 2), 5)
    sc1, sc2 = st.columns(2)
    with sc1:
        sip_hi = st.select_slider(
            "masterstar_platesolve_sip_max_order",
            options=_sip_opts,
            value=_cur_hi,
            key="vyvar_dao_sip_max",
        )
    with sc2:
        sip_lo = st.select_slider(
            "masterstar_platesolve_sip_min_order",
            options=_sip_opts,
            value=_cur_lo,
            key="vyvar_dao_sip_min",
        )
    _detail_help(
        "masterstar_platesolve_sip_max_order / min_order",
        phase="MASTERSTAR — SIP distortion during WCS solve (Astrometry.net / solve-field).",
        used_in="Solver tries SIP orders from max down to min until RMS/validity pass; higher order = more flexible, risk of overfit.",
        compute="SIP polynomial order iteration; RMS limits may be in `config.json` (`masterstar_platesolve_prewrite_*`).",
    )

    st.caption(
        "Expected plate-solve scale is derived **automatically from the DB** "
        "(EQUIPMENTS + TELESCOPE + binning from FITS). "
        "Solver RMS limits remain in **config.json** (defaults / null)."
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Save to config.json", type="primary", key="vyvar_dao_stars_save"):
            cfg.masterstar_prematch_peak_sigma_floor = float(s_floor)
            cfg.masterstar_dao_threshold_sigma = float(dao_sig)
            _shi, _slo = int(sip_hi), int(sip_lo)
            if _slo > _shi:
                _slo = _shi
            cfg.masterstar_platesolve_sip_max_order = max(2, min(5, _shi))
            cfg.masterstar_platesolve_sip_min_order = max(2, min(5, _slo))
            save_config_json(cfg.project_root, cfg.to_json())
            st.success(
                "Saved. The next **MASTERSTARS** / worker run will use the new values "
                "(this session still holds the previous `cfg` in memory)."
            )
    with c2:
        st.caption(
            f"In memory: prematch **k={cfg.masterstar_prematch_peak_sigma_floor:.2f}**, "
            f"DAO **σ={cfg.masterstar_dao_threshold_sigma:.2f}**, "
            f"SIP **{cfg.masterstar_platesolve_sip_max_order}→{cfg.masterstar_platesolve_sip_min_order}**."
        )
