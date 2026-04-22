"""Unified Settings dashboard: paths, QC, photometry, phase 0+1, alignment, tools + rich help."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from config import AppConfig, save_config_json
import ui_dao_stars as ui_dao_stars
import ui_photometry as ui_photometry
from masterstar_context import (
    load_masterstar_context,
    masterstar_context_markdown,
    resolve_masterstar_fits_path,
)


def _detail_help(title: str, *, phase: str, used_in: str, compute: str | None = None) -> None:
    with st.expander(f"❓ {title}", expanded=False):
        st.markdown(f"**Fáza / proces:** {phase}")
        st.markdown(f"**Kde a ako sa používa:** {used_in}")
        if compute:
            st.markdown(f"**Odvodenie / výpočet:** {compute}")


def render_settings_dashboard(
    cfg: AppConfig,
    pipeline: Any,
    *,
    draft_dir_override: Path | None = None,
) -> None:
    st.subheader("Nastavenia")
    st.caption(
        "Hodnoty v záložkách **Prehľad … Fáza 0+1** sa ukladajú jedným tlačidlom **Uložiť hlavné nastavenia** "
        "do `config.json`. Paralelizmus (QC, preprocess, alignment, per-frame katalóg) je jednotný — odvodený z CPU "
        "a RAM; premenná prostredia `VYVAR_PARALLEL_WORKERS` prepíše predvolený počet workerov."
    )

    draft_id = st.session_state.get("vyvar_last_draft_id")
    ms_path = resolve_masterstar_fits_path(
        cfg=cfg, db=getattr(pipeline, "db", None), draft_id=draft_id, draft_dir_override=draft_dir_override
    )
    ms_ctx = load_masterstar_context(ms_path)

    tab_ov, tab_paths, tab_cal, tab_qc, tab_aln, tab_ap, tab_p01, tab_tools = st.tabs(
        [
            "Prehľad",
            "Cesty a katalógy",
            "Kalibrácia",
            "Kvalita (QC)",
            "Zarovnanie",
            "Fotometria (apertúra)",
            "Fáza 0+1",
            "Nástroje",
        ]
    )

    with tab_ov:
        st.markdown("### Aktívny draft a MASTERSTAR")
        if draft_id is None:
            st.info("V Pipeline zadaj draft (číslo alebo cesta) — tu sa zobrazí kontext MASTERSTAR.")
        else:
            st.caption(f"Draft ID: **{int(draft_id)}**")
        if draft_dir_override is not None:
            st.caption(f"Override priečinka: `{draft_dir_override}`")
        st.markdown(masterstar_context_markdown(ms_ctx))
        _detail_help(
            "Čo znamená blok MASTERSTAR vyššie",
            phase="Po spracovaní MASTERSTAR (krok pipeline / platesolve).",
            used_in="Informačný prehľad pre mierku, FWHM a WCS; časť pipeline odvodzuje FOV / sep z FITS+WCS namiesto ručného JSON.",
            compute="`VY_FWHM` zapisuje pipeline (medián DAO FWHM zo sady alebo fit). Mierka: `astropy.wcs.utils.proj_plane_pixel_scales` → priemer v arcsec/px. Stred: `pixel_to_world` na stred čipu.",
        )
        st.markdown("### Efektívne hodnoty (z `config.json`)")
        st.markdown(
            f"- Apertúra: faktor **{cfg.aperture_fwhm_factor:.2f}×FWHM**, annulus **{cfg.annulus_inner_fwhm:.2f}–{cfg.annulus_outer_fwhm:.2f}×FWHM**\n"
            f"- QC po kalibrácii: **{'zap.' if cfg.qc_after_calibrate_enabled else 'vyp.'}**, max HFR **{cfg.qc_max_hfr:.1f}**, min hviezd **{cfg.qc_min_stars}**\n"
            f"- Zarovnanie: max **{cfg.alignment_max_stars}** hviezd, detekcia σ **{cfg.alignment_detection_sigma:.2f}**\n"
            f"- Fáza 0+1: max Δmag **{cfg.phase01_comparison_max_mag_diff:.2f}**, min rámec **{100 * cfg.phase01_comparison_min_frames_frac:.0f}%** snímok"
        )

    with tab_paths:
        st.markdown("### Cesty a knižnica")
        archive_root = st.text_input(
            "archive_root",
            value=str(cfg.archive_root),
            help="Koreň archívu (Drafts, …).",
        )
        _detail_help(
            "archive_root",
            phase="Import, drafty, väčšina výstupov na disk.",
            used_in="`AppConfig.archive_root` — základ pre `Drafts/draft_XXXXXX`, cache a exporty.",
            compute="Žiadny; musí byť platná absolútna cesta.",
        )
        calib_root = st.text_input(
            "calibration_library_root",
            value=str(cfg.calibration_library_root),
            help="Knižnica master dark/flat/bias.",
        )
        _detail_help(
            "calibration_library_root",
            phase="Kalibrácia (dark/flat stack), Calibration Library UI.",
            used_in="Hľadanie masterov podľa validity a filtra; zápis nových masterov.",
            compute="Žiadny.",
        )
        db_path = st.text_input(
            "database_path",
            value=str(cfg.database_path),
            help="SQLite DB pre drafty, QC, cesty k MASTERSTAR.",
        )
        _detail_help(
            "database_path",
            phase="Celý beh aplikácie a pipeline.",
            used_in="Všetky tabuľky draftov, QC hash, cesty k `MASTERSTAR.fits` (`get_obs_draft_masterstar_path`).",
            compute="Žiadny.",
        )

        st.markdown("---")
        st.subheader("GAIA DR3 (VYVAR Local Catalog)")
        gaia_db_path = st.text_input(
            "GAIA_DB_PATH (SQLite .db)",
            value=str(getattr(cfg, "gaia_db_path", "") or ""),
        )
        _detail_help(
            "GAIA_DB_PATH",
            phase="Per-frame a MASTERSTAR katalóg (cone query), blind index.",
            used_in="Lokálny cone search namiesto online VizieR; vyžaduje tabuľku `gaia_dr3`.",
            compute="Žiadny — externá DB (import/build skript).",
        )
        st.session_state["GAIA_DB_PATH"] = str(gaia_db_path).strip()
        col_g1, col_g2 = st.columns([1, 3])
        with col_g1:
            if st.button("🔍 Test Connection", key="vyvar_test_gaia_db"):
                try:
                    import sqlite3

                    p = Path(str(gaia_db_path).strip())
                    if not p.is_file() or p.suffix.lower() != ".db":
                        raise FileNotFoundError("GAIA_DB_PATH musí byť existujúci .db súbor.")
                    con = sqlite3.connect(str(p))
                    try:
                        cur = con.execute(
                            "SELECT name FROM sqlite_master WHERE type='table' AND name='gaia_dr3' LIMIT 1;"
                        )
                        if cur.fetchone() is None:
                            raise ValueError("V DB chýba tabuľka `gaia_dr3`.")
                    finally:
                        con.close()
                    st.success("OK: DB existuje a tabuľka `gaia_dr3` je dostupná.")
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))
        with col_g2:
            st.caption("Očakáva sa SQLite DB s tabuľkou `gaia_dr3` a indexmi `idx_ra`, `idx_dec`.")

        blind_index_path = st.text_input(
            "BLIND_INDEX_PATH (.pkl)",
            value=str(getattr(cfg, "blind_index_path", "") or ""),
            key="vyvar_blind_index_path",
        )
        _detail_help(
            "BLIND_INDEX_PATH",
            phase="Blind astrometry / triangle matching (ak je zapnuté v pipeline).",
            used_in="Cesta k `gaia_triangles.pkl` z `build_gaia_blind_index.py`.",
            compute="Predpočítaný index — žiadny runtime výpočet okrem načítania.",
        )
        st.caption("Cesta k súboru gaia_triangles.pkl (generovaný skriptom build_gaia_blind_index.py)")

        st.markdown("---")
        st.subheader("VSX lokálna databáza")
        vsx_db_path = st.text_input(
            "VSX_LOCAL_DB_PATH (SQLite .db, tabuľka `vsx_data`)",
            value=str(getattr(cfg, "vsx_local_db_path", "") or ""),
            key="vyvar_vsx_local_db_path",
        )
        _detail_help(
            "VSX_LOCAL_DB_PATH",
            phase="Variable targets export, MASTERSTAR QA, suspected LC.",
            used_in="Lokálny dotaz do VSX podľa kužela (oid, ra_deg, dec_deg, …).",
            compute="Žiadny — import z VizieR.",
        )
        col_v1, col_v2 = st.columns([1, 3])
        with col_v1:
            if st.button("🔍 Test Connection", key="vyvar_test_vsx_local_db"):
                try:
                    from database import validate_vsx_local_db_schema

                    ok, code = validate_vsx_local_db_schema(str(vsx_db_path).strip())
                    if not ok:
                        _msgs = {
                            "missing_file": "Súbor neexistuje alebo cesta je prázdna.",
                            "missing_table_vsx_data": "V DB chýba tabuľka `vsx_data`.",
                        }
                        if str(code).startswith("missing_columns:"):
                            st.error(f"Chýbajú stĺpce: {code.split(':', 1)[1]} (potrebné: oid, ra_deg, dec_deg).")
                        else:
                            st.error(_msgs.get(str(code), str(code)))
                    else:
                        st.success("OK: súbor existuje, tabuľka `vsx_data` má požadované stĺpce.")
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))
        with col_v2:
            st.caption(
                "SQLite z importu VizieR B/vsx/vsx (stĺpce oid, name, ra_deg, dec_deg, var_type, mag_max, mag_min). "
                "Použitie v pipeline sa riadi touto cestou."
            )

        vsx_mag_limit_save = st.number_input(
            "Mag limit pre Variable Targets export (VSX)",
            min_value=0.0,
            max_value=21.0,
            value=float(getattr(cfg, "vsx_variable_targets_mag_limit", 13.0) or 13.0),
            step=0.5,
            help="Režne VSX podľa mag_max.",
        )
        _detail_help(
            "vsx_variable_targets_mag_limit",
            phase="Export variable targets (VSX cone).",
            used_in="Zachová riadky s `mag_max` ≤ limit (alebo bez `mag_max`). Hodnota 0 = bez rezu.",
            compute="Filtrácia v SQL / pandase po dotaze — nie fyzikálny výpočet.",
        )

    with tab_cal:
        st.markdown("### Kalibrácia")
        new_dark = st.slider(
            "masterdark_validity_days",
            min_value=1,
            max_value=3650,
            value=int(cfg.masterdark_validity_days),
            help="Po koľkých dňoch sa master dark považuje za starý.",
        )
        _detail_help(
            "masterdark_validity_days",
            phase="Výber master darku z Calibration Library.",
            used_in="Porovnanie dátumu pozorovania vs. dátum masteru; mimo platnosti sa master nevyberie.",
            compute="Rozdiel dátumov v dňoch oproti prahu.",
        )
        new_flat = st.slider(
            "masterflat_validity_days",
            min_value=1,
            max_value=3650,
            value=int(cfg.masterflat_validity_days),
            help="Platnosť master flatu v dňoch.",
        )
        _detail_help(
            "masterflat_validity_days",
            phase="Rovnako ako dark — výber flatu.",
            used_in="Kalibrácia lightov pred ďalšími krokmi.",
            compute="Rozdiel dátumov vs. prah.",
        )
        _cln_none = st.checkbox(
            "calibration_library_native_binning: čítať z každého master FITS (JSON null)",
            value=cfg.calibration_library_native_binning is None,
            key="vyvar_settings_cl_bin_null",
        )
        new_cl_bin = st.number_input(
            "calibration_library_native_binning (1–16, ak nie je „z FITS“)",
            min_value=1,
            max_value=16,
            value=int(cfg.calibration_library_native_binning or 1),
            disabled=bool(_cln_none),
            key="vyvar_settings_cl_bin",
        )
        _detail_help(
            "calibration_library_native_binning",
            phase="Matching masterov k snímku (rovnaký binning).",
            used_in="Ak null, binning sa číta z FITS hlavičky masteru; inak pevná hodnota.",
            compute="Žiadny — buď z hlavičky, alebo konštanta z JSON.",
        )

    with tab_qc:
        st.markdown("### QC")
        qc_after_cal = st.checkbox(
            "qc_after_calibrate_enabled",
            value=bool(cfg.qc_after_calibrate_enabled),
            help="Po kalibrácii: QC metriky na flatovaných lightoch.",
        )
        _detail_help(
            "qc_after_calibrate_enabled",
            phase="Hneď po kalibrácii (flatované lights).",
            used_in="Spočítanie HFR, počtu hviezd, pozadia — zápis do DB (`OBS_FILES`) a limity pri ďalších krokoch (analyze, preprocess, výber do MASTERSTAR pipeline).",
            compute="HFR/DAO metriky z `photutils`/pipeline QC modulov (nie jednoduchý vzorec v JSON).",
        )
        st.caption(
            "Prah `qc_max_background_rms` je pokročilý — ostáva len v `config.json` (väčšinou `null`); tu sa nenastavuje."
        )
        qc_hfr = st.slider(
            "qc_max_hfr",
            min_value=0.5,
            max_value=20.0,
            value=float(cfg.qc_max_hfr),
            step=0.1,
            help="Horší HFR ako tento prah → zamietnutie / flag (podľa logiky pipeline).",
        )
        _detail_help(
            "qc_max_hfr",
            phase="QC po kalibrácii a pri kontrolách v ďalších fázach (analyze, preprocess).",
            used_in="Porovnanie meraného HFR (polomer alebo ekvivalent) s prahom; príliš vysoký HFR → zamietnutie alebo flag v DB podľa logiky pipeline.",
            compute="HFR z analýzy snímku; prah je konštanta z `config.json`.",
        )
        qc_stars = st.slider(
            "qc_min_stars",
            min_value=0,
            max_value=500,
            value=int(cfg.qc_min_stars),
            step=1,
            help="Minimum detekovaných hviezd pre úspešný QC.",
        )
        _detail_help(
            "qc_min_stars",
            phase="QC po kalibrácii a pri súvisiacich limitoch v pipeline.",
            used_in="Ak počet detekovaných hviezd pod prahom, snímok je podozrivý alebo zamietnutý v QC.",
            compute="Počet z DAO detekcie nad prahom — prah je z tohto slidera.",
        )
        cosmic_on = st.checkbox("cosmic_clean_enabled", value=bool(cfg.cosmic_clean_enabled))
        _detail_help(
            "cosmic_clean_enabled",
            phase="Preprocess / kalibrované alebo processed lights (LACosmic).",
            used_in="Zapína odstránenie kozmických čiar pred ďalšou photometriou.",
            compute="LACosmic s parametrami nižšie.",
        )
        cosmic_sig = st.slider(
            "cosmic_sigclip",
            min_value=2.0,
            max_value=12.0,
            value=float(cfg.cosmic_sigclip),
            step=0.1,
        )
        _detail_help(
            "cosmic_sigclip",
            phase="LACosmic pri `cosmic_clean_enabled`.",
            used_in="Sigma clipping prah pre outlier pixel/hviezdy.",
            compute="štandardná LACosmic logika: odchylka vs. lokálny šum × sigclip.",
        )
        cosmic_obj = st.slider(
            "cosmic_objlim",
            min_value=1.0,
            max_value=20.0,
            value=float(cfg.cosmic_objlim),
            step=0.25,
        )
        _detail_help(
            "cosmic_objlim",
            phase="LACosmic.",
            used_in="Obmedzenie počtu iterácií / objektov na snímku (ochrana hviezdnej fotosféry).",
            compute="Podľa implementácie astroscrappy/LACosmic v pipeline.",
        )

    with tab_aln:
        st.markdown("### Zarovnanie snímok (astroalign + DAO)")
        aln_max = st.slider(
            "alignment_max_stars",
            min_value=10,
            max_value=5000,
            value=int(cfg.alignment_max_stars),
            step=10,
            help="Max. počet najjasnejších kontrolných bodov na rámec.",
        )
        _detail_help(
            "alignment_max_stars",
            phase="Fáza zarovnania (medzi kalibráciou a stackom / per-frame).",
            used_in="Zoradenie hviezd podľa fluxu, orez na N pre párovanie s referenčným rámcom.",
            compute="N = min(detekovaných, alignment_max_stars).",
        )
        aln_sig = st.slider(
            "alignment_detection_sigma",
            min_value=0.5,
            max_value=20.0,
            value=float(cfg.alignment_detection_sigma),
            step=0.25,
            help="DAO/ detekčný prah pre hviezdy pri zarovnaní.",
        )
        _detail_help(
            "alignment_detection_sigma",
            phase="Zarovnanie (DAO find).",
            used_in="Spája sa s QC štýlom detekcie — vyšší σ = menej slabých hviezd, robustnejšie na šum.",
            compute="Prah v sigma nad lokálnym pozadím (štandardná DAO logika).",
        )

    with tab_ap:
        st.markdown("### Fotometria (apertúra a annulus)")
        st.caption("Prepínače apertúry vs. DAO a PSF sú v záložke **Nástroje → Fotometria (režim)**.")
        ap_fwhm = st.slider(
            "aperture_fwhm_factor",
            min_value=0.5,
            max_value=6.0,
            value=float(cfg.aperture_fwhm_factor),
            step=0.1,
            help="Polomer apertúry = faktor × merané FWHM.",
        )
        _detail_help(
            "aperture_fwhm_factor",
            phase="Fáza 2 / per-frame apertúrna fotometria (ak je zapnutá).",
            used_in="Polomer kruhu v pixeloch: `r_ap = factor × FWHM` (FWHM z hlavičky / merania snímku).",
            compute="Násobenie lokálneho FWHM konštantou z config.",
        )
        ann_in = st.slider(
            "annulus_inner_fwhm",
            min_value=1.0,
            max_value=10.0,
            value=float(cfg.annulus_inner_fwhm),
            step=0.25,
        )
        ann_out = st.slider(
            "annulus_outer_fwhm",
            min_value=1.5,
            max_value=12.0,
            value=float(cfg.annulus_outer_fwhm),
            step=0.25,
        )
        _detail_help(
            "annulus_inner_fwhm / annulus_outer_fwhm",
            phase="Apertúrna fotometria — odčítanie pozadia.",
            used_in="Medzikružie medzi `r_inner = inner×FWHM` a `r_outer = outer×FWHM` okolo hviezdy.",
            compute="Plošný priemer medzi kruhmi → `annulus_median` na odhad sky; flux_v = sum(apertúra) − sky×plocha.",
        )
        if ann_out <= ann_in:
            st.warning("annulus_outer_fwhm musí byť väčší ako annulus_inner_fwhm — pri uložení sa upraví.")
        nl_pct = st.slider(
            "nonlinearity_peak_percentile",
            min_value=0.0,
            max_value=50.0,
            value=float(cfg.nonlinearity_peak_percentile),
            step=0.5,
        )
        nl_ratio = st.slider(
            "nonlinearity_fwhm_ratio",
            min_value=1.01,
            max_value=3.0,
            value=float(cfg.nonlinearity_fwhm_ratio),
            step=0.01,
        )
        _detail_help(
            "nonlinearity_peak_percentile / nonlinearity_fwhm_ratio",
            phase="QC / flagovanie nelinearít pri apertúre.",
            used_in="Nájde percentil špičky jasu a porovná šírku profilu s očakávaním podľa FWHM.",
            compute="Heuristika v pipeline: ak je peak nad percentilom a FWHM ratio > prah → podozrivá saturácia/nelinearita.",
        )

    with tab_p01:
        st.markdown("### Fáza 0+1 — porovnanie hviezd / stabilita")
        st.caption(
            "Parametre filtra pri párovaní katalógu medzi snímkami (Gaia + vlastné pravidlá). Podrobnosti v `config.py` pri poliach `phase01_*`."
        )
        p01_md = st.slider(
            "phase01_comparison_max_dist_deg",
            min_value=0.05,
            max_value=10.0,
            value=float(cfg.phase01_comparison_max_dist_deg),
            step=0.05,
        )
        _detail_help(
            "phase01_comparison_max_dist_deg",
            phase="Fáza 0+1 — spatial match medzi snímkami.",
            used_in="Max. uhlová vzdialenosť medzi kandidátmi na rovnakú hviezdu.",
            compute="Great-circle alebo projekcia v pipeline; prah v stupňoch z config.",
        )
        p01_mm = st.slider(
            "phase01_comparison_max_mag_diff",
            min_value=0.05,
            max_value=5.0,
            value=float(cfg.phase01_comparison_max_mag_diff),
            step=0.05,
        )
        p01_mag_b = st.slider(
            "phase01_comparison_mag_bright_threshold",
            min_value=6.0,
            max_value=18.0,
            value=float(cfg.phase01_comparison_mag_bright_threshold),
            step=0.25,
        )
        p01_mag_bf = st.slider(
            "phase01_comparison_max_mag_diff_bright_floor",
            min_value=0.0,
            max_value=4.0,
            value=float(cfg.phase01_comparison_max_mag_diff_bright_floor),
            step=0.05,
        )
        _detail_help(
            "phase01_comparison_max_mag_diff (+ bright threshold)",
            phase="Fáza 0+1 — fotometrická zhoda medzi snímkami.",
            used_in="Ak |Δmag| medzi snímkami > prah, párovanie sa zamietne. Pre jasné hviezdy (mag < threshold) sa použije aspoň `max_mag_diff_bright_floor`.",
            compute="Dynamický prah: `max(|Δmag|, floor pre bright)` podľa `config.py` logiky.",
        )
        p01_bv = st.slider(
            "phase01_comparison_max_bv_diff",
            min_value=0.02,
            max_value=3.0,
            value=float(cfg.phase01_comparison_max_bv_diff),
            step=0.02,
        )
        p01_ncmin = st.slider(
            "phase01_comparison_n_comp_min",
            min_value=2,
            max_value=12,
            value=int(cfg.phase01_comparison_n_comp_min),
        )
        p01_ncmax = st.slider(
            "phase01_comparison_n_comp_max",
            min_value=3,
            max_value=20,
            value=int(cfg.phase01_comparison_n_comp_max),
        )
        p01_rms = st.slider(
            "phase01_comparison_max_comp_rms",
            min_value=0.01,
            max_value=0.5,
            value=float(cfg.phase01_comparison_max_comp_rms),
            step=0.01,
        )
        p01_mind = st.slider(
            "phase01_comparison_min_dist_arcsec",
            min_value=0.0,
            max_value=600.0,
            value=float(cfg.phase01_comparison_min_dist_arcsec),
            step=5.0,
        )
        p01_mff = st.slider(
            "phase01_comparison_min_frames_frac",
            min_value=0.05,
            max_value=0.95,
            value=float(cfg.phase01_comparison_min_frames_frac),
            step=0.05,
        )
        _detail_help(
            "phase01_comparison_min_frames_frac",
            phase="Fáza 0+1 — stabilita cez snímky.",
            used_in="Hviezda musí byť prítomná aspoň v danej frakcii snímok, inak sa vyhodí z porovnávacieho setu.",
            compute="počet_snímok_s_matchom / celkový_počet ≥ prah.",
        )
        p01_ex_nss = st.checkbox(
            "phase01_comparison_exclude_gaia_nss",
            value=bool(cfg.phase01_comparison_exclude_gaia_nss),
        )
        p01_ex_ext = st.checkbox(
            "phase01_comparison_exclude_gaia_extobj",
            value=bool(cfg.phase01_comparison_exclude_gaia_extobj),
        )
        _detail_help(
            "exclude_gaia_nss / exclude_gaia_extobj",
            phase="Fáza 0+1 — čistenie katalógu.",
            used_in="Vyraďuje binárky/NSS alebo extended objekty z Gaia stĺpcov, aby nekazili porovnanie hviezd.",
            compute="Boolean filter na riadky pred párovaním.",
        )
        p01_chip = st.slider(
            "phase01_chip_interior_margin_px",
            min_value=0,
            max_value=2000,
            value=int(cfg.phase01_chip_interior_margin_px),
            step=5,
        )
        _detail_help(
            "phase01_chip_interior_margin_px",
            phase="Fáza 0+1 a suspected LC — okraj čipu.",
            used_in="Hviezdy bližšie ako margin px od okraja sa ignorujú pri porovnaní / suspected výpočtoch.",
            compute="Pixelové súradnice: x < margin alebo x > W−margin (podobne y).",
        )

    with tab_tools:
        st.caption("Samostatné uloženie: každý nástroj má vlastné tlačidlo „Uložiť“.")
        tdao, tphot, tqual = st.tabs(["DAO-STARS / MASTERSTAR", "Fotometria (režim)", "Fotometria — diagnostika"])
        with tdao:
            ui_dao_stars.render_dao_stars_dashboard(cfg)
        with tphot:
            ui_photometry.render_photometry_dashboard(cfg)
        with tqual:
            from ui_photometry_quality import render_photometry_quality_diagnostic

            render_photometry_quality_diagnostic(
                pipeline=pipeline,
                draft_id=st.session_state.get("vyvar_last_draft_id"),
            )

    st.markdown("---")
    if st.button("Uložiť hlavné nastavenia do config.json", type="primary", key="vyvar_settings_master_save"):
        cfg.archive_root = Path(archive_root)
        cfg.calibration_library_root = Path(calib_root)
        cfg.database_path = Path(db_path)
        cfg.masterdark_validity_days = int(new_dark)
        cfg.masterflat_validity_days = int(new_flat)
        cfg.calibration_library_native_binning = None if _cln_none else int(new_cl_bin)
        cfg.gaia_db_path = str(gaia_db_path).strip()
        cfg.blind_index_path = str(blind_index_path).strip()
        cfg.vsx_local_db_path = str(vsx_db_path).strip()
        cfg.vsx_variable_targets_mag_limit = float(vsx_mag_limit_save)
        cfg.qc_max_hfr = float(qc_hfr)
        cfg.qc_min_stars = int(qc_stars)
        cfg.qc_after_calibrate_enabled = bool(qc_after_cal)
        cfg.cosmic_clean_enabled = bool(cosmic_on)
        cfg.cosmic_sigclip = float(cosmic_sig)
        cfg.cosmic_objlim = float(cosmic_obj)
        cfg.alignment_max_stars = int(max(10, min(5000, aln_max)))
        det_sig = float(aln_sig)
        cfg.alignment_detection_sigma = det_sig if det_sig > 0 else 5.0
        cfg.aperture_fwhm_factor = float(max(0.5, min(6.0, ap_fwhm)))
        cfg.annulus_inner_fwhm = float(max(1.0, min(10.0, ann_in)))
        cfg.annulus_outer_fwhm = float(max(1.5, min(12.0, ann_out)))
        if cfg.annulus_outer_fwhm <= cfg.annulus_inner_fwhm:
            cfg.annulus_outer_fwhm = cfg.annulus_inner_fwhm + 1.0
        cfg.nonlinearity_peak_percentile = float(max(0.0, min(50.0, nl_pct)))
        cfg.nonlinearity_fwhm_ratio = float(max(1.01, min(3.0, nl_ratio)))

        cfg.phase01_comparison_max_dist_deg = float(max(0.05, min(10.0, p01_md)))
        cfg.phase01_comparison_max_mag_diff = float(max(0.05, min(5.0, p01_mm)))
        cfg.phase01_comparison_mag_bright_threshold = float(max(6.0, min(18.0, p01_mag_b)))
        cfg.phase01_comparison_max_mag_diff_bright_floor = float(max(0.0, min(4.0, p01_mag_bf)))
        cfg.phase01_comparison_max_bv_diff = float(max(0.02, min(3.0, p01_bv)))
        cfg.phase01_comparison_n_comp_min = int(max(2, min(12, p01_ncmin)))
        cfg.phase01_comparison_n_comp_max = int(max(3, min(20, p01_ncmax)))
        if cfg.phase01_comparison_n_comp_max < cfg.phase01_comparison_n_comp_min:
            cfg.phase01_comparison_n_comp_max = cfg.phase01_comparison_n_comp_min
        cfg.phase01_comparison_max_comp_rms = float(max(0.01, min(0.5, p01_rms)))
        cfg.phase01_comparison_min_dist_arcsec = float(max(0.0, min(600.0, p01_mind)))
        cfg.phase01_comparison_min_frames_frac = float(max(0.05, min(0.95, p01_mff)))
        cfg.phase01_comparison_exclude_gaia_nss = bool(p01_ex_nss)
        cfg.phase01_comparison_exclude_gaia_extobj = bool(p01_ex_ext)
        cfg.phase01_chip_interior_margin_px = int(max(0, min(2000, p01_chip)))

        save_config_json(cfg.project_root, cfg.to_json())
        cfg.ensure_base_dirs()
        st.success("Uložené do `config.json`. Obnovujem UI…")
        st.rerun()

    st.caption(
        "Plate-solve FOV, časť mierky a per-frame match separácie sa môžu odvádzať z **FITS + WCS + DB** "
        "(nie sú vždy v JSON) — pozri blok MASTERSTAR v Prehľade."
    )
