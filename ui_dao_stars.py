"""DAO-STARS: úprava hlavných MASTERSTAR parametrov detekcie / SIP (config.json)."""

from __future__ import annotations

import streamlit as st

from config import AppConfig, save_config_json


def render_dao_stars_dashboard(cfg: AppConfig) -> None:
    st.subheader("DAO-STARS")
    st.caption(
        "**MASTERSTAR** referenčný snímok sa vyberá vo **FITS QA**; tu sú prahy detekcie a rozsah **SIP** pri plate-solve. "
        "Ukladá sa do **config.json**."
    )

    cur_sf = float(getattr(cfg, "masterstar_prematch_peak_sigma_floor", 3.2))
    cur_ds = float(getattr(cfg, "masterstar_dao_threshold_sigma", 1.8))

    st.markdown("#### Predzáber peak (SNR pred matchom)")
    st.caption(
        "Zostanú detekcie s lokálnym peakom nad **median + k×σ**. **Nižšie k** → viac slabších hviezd pred párovaním s Gaia. "
        "Rozsah **0.5–6.0**."
    )
    s_floor = st.slider(
        "masterstar_prematch_peak_sigma_floor (k)",
        min_value=0.5,
        max_value=6.0,
        value=min(max(cur_sf, 0.5), 6.0),
        step=0.1,
        key="vyvar_dao_stars_prematch_k",
    )

    st.markdown("#### DAO prah (plate-solve + katalóg)")
    st.caption(
        "**DAOStarFinder:** threshold = **k × RMS**. **Nižšie k** → viac kandidátov (aj šum). Rovnaké **k** ide do solvera aj do následného katalógu. "
        "Rozsah **0.1–6.0**."
    )
    dao_sig = st.slider(
        "masterstar_dao_threshold_sigma (k)",
        min_value=0.1,
        max_value=6.0,
        value=min(max(cur_ds, 0.1), 6.0),
        step=0.05,
        key="vyvar_dao_stars_dao_sigma",
    )

    st.info(
        "Pri **veľkom počte** detekcií uvoľni **k** aj **DAO σ** len ak je **WCS už dobrý** (nízky px RMS v Infologu). "
        "Pri slabom plate-solve sa párovanie často natiahne na veľký **match_sep** — skresľuje to diagnostiku."
    )

    st.markdown("#### SIP pri plate-solve (MASTERSTAR)")
    st.caption(
        "Solver skúša **od vyššieho rádu nadol** po spodnú hranicu (napr. 5→4→3). **Min** nesmie byť väčší ako **max**."
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

    st.caption(
        "Očakávaná mierka plate-solve sa odvodzuje **automaticky z DB** (EQUIPMENTS + TELESCOPE + binning z FITS). "
        "Technické RMS limity solvera ostávajú v **config.json** (predvolené / null)."
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Zapísať do config.json", type="primary", key="vyvar_dao_stars_save"):
            cfg.masterstar_prematch_peak_sigma_floor = float(s_floor)
            cfg.masterstar_dao_threshold_sigma = float(dao_sig)
            _shi, _slo = int(sip_hi), int(sip_lo)
            if _slo > _shi:
                _slo = _shi
            cfg.masterstar_platesolve_sip_max_order = max(2, min(5, _shi))
            cfg.masterstar_platesolve_sip_min_order = max(2, min(5, _slo))
            save_config_json(cfg.project_root, cfg.to_json())
            st.success(
                "Uložené. Ďalší beh **MASTERSTARS** / worker použije nové hodnoty (aktuálna session má `cfg` už v pamäti)."
            )
    with c2:
        st.caption(
            f"V pamäti: prematch **k={cfg.masterstar_prematch_peak_sigma_floor:.2f}**, DAO **σ={cfg.masterstar_dao_threshold_sigma:.2f}**, "
            f"SIP **{cfg.masterstar_platesolve_sip_max_order}→{cfg.masterstar_platesolve_sip_min_order}**."
        )
