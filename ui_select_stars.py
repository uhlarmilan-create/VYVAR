"""Select Stars dashboard — Fáza 0+1: výber aktívnych premenných a porovnávacích hviezd."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import streamlit as st

from photometry import run_phase0_and_phase1
from platesolve_ui_paths import default_bundle_dir, masterstars_csv_in_dir
from vyvar_ui_status import vyvar_footer_idle, vyvar_footer_running

if TYPE_CHECKING:
    from config import AppConfig
    from pipeline import AstroPipeline


# ---------------------------------------------------------------------------
# Pomocné funkcie
# ---------------------------------------------------------------------------


def _find_phase01_setups(cfg: "AppConfig", draft_id: int | None) -> dict[str, dict[str, Path | None]]:
    """Všetky platesolve setupy s ``per_frame_catalog_index.csv`` a cesty pre Fázu 0+1."""
    if draft_id is None:
        return {}
    try:
        archive = Path(cfg.archive_root)
        draft_dir = archive / "Drafts" / f"draft_{int(draft_id):06d}"
        ps_dir = draft_dir / "platesolve"
        aligned_root = draft_dir / "detrended_aligned" / "lights"
        if not ps_dir.is_dir():
            return {}

        out: dict[str, dict[str, Path | None]] = {}

        def _add_setup(obs_group_dir: Path) -> None:
            name = obs_group_dir.name
            per_frame_dir = (aligned_root / name) if (aligned_root / name).is_dir() else None
            ms_csv = masterstars_csv_in_dir(obs_group_dir)
            out[name] = {
                "variable_targets_csv": obs_group_dir / "variable_targets.csv",
                "masterstars_csv": ms_csv,
                "per_frame_csv_dir": per_frame_dir,
                "output_dir": obs_group_dir / "photometry",
                "obs_group_dir": obs_group_dir,
                "masterstar_fits": obs_group_dir / "MASTERSTAR.fits",
            }

        for subdir in sorted(ps_dir.iterdir()):
            if subdir.is_dir() and (subdir / "per_frame_catalog_index.csv").is_file():
                _add_setup(subdir)

        return out
    except Exception:  # noqa: BLE001
        return {}


def _default_setup_name(setups: dict[str, dict[str, Path | None]], cfg: "AppConfig", draft_id: int) -> str:
    if not setups:
        return ""
    ps = Path(cfg.archive_root) / "Drafts" / f"draft_{int(draft_id):06d}" / "platesolve"
    pick = default_bundle_dir(ps)
    if pick is not None and pick.name in setups:
        return pick.name
    r_first = next((k for k in sorted(setups) if k.upper().startswith("R_")), None)
    return r_first or sorted(setups)[0]


def _load_fwhm_from_masterstar(masterstar_fits: Path | None) -> float:
    """Načítaj VY_FWHM z MASTERSTAR.fits hlavičky."""
    if masterstar_fits is None or not masterstar_fits.is_file():
        return 3.7
    try:
        from astropy.io import fits as astrofits

        with astrofits.open(masterstar_fits, memmap=False) as hdul:
            v = float(hdul[0].header.get("VY_FWHM", 3.7))
            if 1.0 < v < 15.0:
                return round(v, 3)
    except Exception:  # noqa: BLE001
        pass
    return 3.7


def _results_exist(output_dir: Path | None) -> bool:
    if output_dir is None:
        return False
    return (
        (output_dir / "active_targets.csv").exists()
        and (output_dir / "comparison_stars_per_target.csv").exists()
    )


def _results_timestamp(output_dir: Path | None) -> str:
    if output_dir is None:
        return ""
    p = output_dir / "active_targets.csv"
    if p.exists():
        import datetime

        ts = datetime.datetime.fromtimestamp(p.stat().st_mtime)
        return ts.strftime("%d.%m.%Y %H:%M")
    return ""


# ---------------------------------------------------------------------------
# Subtaby výsledkov
# ---------------------------------------------------------------------------


def _render_targets_tab(active_df: pd.DataFrame) -> None:
    """Tab: Premenné hviezdy (active targets)."""
    st.markdown(f"**{len(active_df)} aktívnych cieľov**")

    show_cols = [
        c
        for c in [
            "vsx_name",
            "vsx_type",
            "mag",
            "vsx_period",
            "ra_deg",
            "dec_deg",
            "match_dist_arcsec",
            "zone",
        ]
        if c in active_df.columns
    ]

    display = active_df[show_cols].copy()
    if "mag" in display.columns:
        display["mag"] = pd.to_numeric(display["mag"], errors="coerce").round(3)
    if "match_dist_arcsec" in display.columns:
        display["match_dist_arcsec"] = pd.to_numeric(
            display["match_dist_arcsec"], errors="coerce"
        ).round(2)
    if "ra_deg" in display.columns:
        display["ra_deg"] = pd.to_numeric(display["ra_deg"], errors="coerce").round(5)
    if "dec_deg" in display.columns:
        display["dec_deg"] = pd.to_numeric(display["dec_deg"], errors="coerce").round(5)

    st.dataframe(display, use_container_width=True, hide_index=True)

    # Upozornenie na noisy1 hviezdy
    if "zone" in active_df.columns:
        noisy1 = active_df[active_df["zone"] == "noisy1"]
        if len(noisy1) > 0:
            st.warning(
                f"⚠️ {len(noisy1)} hviezd v zóne **noisy1** — slabší signál, "
                f"možná premenná ale fotometria bude menej presná."
            )


def _render_comparison_tab(comp_df: pd.DataFrame) -> None:
    """Tab: Porovnávacie hviezdy per target."""
    if comp_df.empty:
        st.info("Žiadne porovnávacie hviezdy.")
        return

    # Dropdown na výber targetu
    target_options = []
    if "target_vsx_name" in comp_df.columns:
        target_options = sorted(comp_df["target_vsx_name"].dropna().unique().tolist())
    elif "target_catalog_id" in comp_df.columns:
        target_options = sorted(comp_df["target_catalog_id"].dropna().unique().tolist())

    if not target_options:
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        return

    selected = st.selectbox(
        "Vyber premennú hviezdu:",
        options=target_options,
        key="select_stars_target_dropdown",
    )

    filter_col = "target_vsx_name" if "target_vsx_name" in comp_df.columns else "target_catalog_id"
    sub = comp_df[comp_df[filter_col] == selected].copy()

    st.markdown(f"**{len(sub)} porovnávačiek** pre `{selected}`")

    show_cols = [
        c
        for c in [
            "catalog_id",
            "name",
            "mag",
            "b_v",
            "bp_rp",
            "comp_tier",
            "_dist_deg",
            "comp_rms",
            "comp_n_frames",
            "zone",
        ]
        if c in sub.columns
    ]

    display = sub[show_cols].copy()
    for col in ("mag", "b_v", "bp_rp", "comp_rms"):
        if col in display.columns:
            display[col] = pd.to_numeric(display[col], errors="coerce").round(4)
    if "_dist_deg" in display.columns:
        display = display.rename(columns={"_dist_deg": "dist_deg"})
        display["dist_deg"] = pd.to_numeric(display["dist_deg"], errors="coerce").round(4)
    if "comp_tier" in display.columns:
        def _tier_css(v: object) -> str:
            s = str(v or "").strip()
            key = s.split("_", 1)[0].upper()
            return {
                "TIER1": "background-color:rgba(34,197,94,0.25);font-weight:600;",
                "TIER2": "background-color:rgba(59,130,246,0.25);font-weight:600;",
                "TIER3": "background-color:rgba(234,179,8,0.25);font-weight:600;",
                "TIER4": "background-color:rgba(239,68,68,0.25);font-weight:600;color:rgba(127,29,29,1.0);",
            }.get(key, "")

        st.dataframe(
            display.style.applymap(_tier_css, subset=["comp_tier"]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.dataframe(display, use_container_width=True, hide_index=True)

    # RMS histogram
    if "comp_rms" in sub.columns and len(sub) > 1:
        st.markdown("**RMS distribúcia porovnávačiek:**")
        rms_vals = sub["comp_rms"].dropna().tolist()
        rms_df = pd.DataFrame({"RMS": rms_vals})
        st.bar_chart(rms_df["RMS"])


def _render_suspected_tab(suspected_df: pd.DataFrame) -> None:
    """Tab: Suspected new variables."""
    if suspected_df.empty:
        st.success("Žiadni kandidáti na nové premenné hviezdy.")
        return

    st.markdown(
        f"**{len(suspected_df)} kandidátov** na nové premenné hviezdy "
        f"(vysoký RMS scatter, nie v VSX katalógu)"
    )

    show_cols = [
        c
        for c in [
            "catalog_id",
            "mag",
            "comp_rms",
            "n_frames",
            "ra_deg",
            "dec_deg",
            "zone",
        ]
        if c in suspected_df.columns
    ]

    display = suspected_df[show_cols].copy()
    for col in ("mag", "comp_rms", "ra_deg", "dec_deg"):
        if col in display.columns:
            display[col] = pd.to_numeric(display[col], errors="coerce").round(4)

    sort_col = "comp_rms" if "comp_rms" in display.columns else display.columns[0]
    st.dataframe(
        display.sort_values(sort_col, ascending=False, na_position="last"),
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------------------------
# Hlavný render
# ---------------------------------------------------------------------------


def render_select_stars(
    cfg: "AppConfig",
    draft_id: int | None,
    pipeline: "AstroPipeline",
) -> None:
    """Hlavná funkcia pre Select Stars tab."""
    _ = pipeline
    st.header("Select Stars")
    st.caption("Fáza 0+1: Výber aktívnych premenných hviezd a porovnávacích hviezd.")

    if draft_id is None:
        st.info("Žiadny aktívny draft. Najprv spusti platesolve.")
        return

    setups = _find_phase01_setups(cfg, draft_id)
    if not setups:
        st.warning(
            "Nenájdené vstupné súbory. Najprv spusti platesolve (Fáza plate-solve musí byť dokončená)."
        )
        return

    setup_names = sorted(setups)
    if len(setup_names) > 1:
        default_nm = _default_setup_name(setups, cfg, int(draft_id))
        sel_ix = setup_names.index(default_nm) if default_nm in setup_names else 0
        chosen = st.selectbox(
            "Filter / skupina (platesolve):",
            options=setup_names,
            index=sel_ix,
            key="select_stars_platesolve_setup",
        )
    else:
        chosen = setup_names[0]
        st.caption(f"Platesolve setup: **{chosen}**")

    paths = setups[chosen]
    vt_csv = paths.get("variable_targets_csv")
    ms_csv = paths.get("masterstars_csv")
    per_frame_dir = paths.get("per_frame_csv_dir")
    output_dir = paths.get("output_dir")
    ms_fits = paths.get("masterstar_fits")

    # Skontroluj či vstupné súbory existujú
    missing = []
    if vt_csv is None or not vt_csv.exists():
        missing.append("variable_targets.csv")
    if ms_csv is None or not ms_csv.exists():
        missing.append("masterstars_full_match.csv")
    if per_frame_dir is None or not per_frame_dir.exists():
        missing.append("per-frame CSV adresár (detrended_aligned)")

    if missing:
        st.error(f"Chýbajú súbory: {', '.join(missing)}")
        return

    fwhm_px = _load_fwhm_from_masterstar(ms_fits if isinstance(ms_fits, Path) else None)
    exists = _results_exist(output_dir)

    with st.expander("Pravidlá výberu porovnávacích hviezd (Fáza 1) — z `config.json`", expanded=False):
        st.markdown(
            "Účinné hodnoty z **AppConfig** (kľúče v zátvorkách). Pri riedkom poli typicky zväčši "
            "**`phase01_comparison_max_mag_diff`** a **`phase01_comparison_max_dist_deg`**, prípadne "
            "zvýš **`phase01_comparison_max_comp_rms`** alebo zníž **`phase01_comparison_min_frames_frac`**. "
            "Pri **jasných cieľoch** (napr. R~9 mag) nastav **`phase01_comparison_max_mag_diff_bright_floor`** "
            "(min. |Δmag| pás; ``0`` vypne) a prípadne **`phase01_comparison_mag_bright_threshold`**."
        )
        st.code(
            f"max_dist_deg = {cfg.phase01_comparison_max_dist_deg}\n"
            f"max_mag_diff = {cfg.phase01_comparison_max_mag_diff}\n"
            f"mag_bright_threshold = {cfg.phase01_comparison_mag_bright_threshold}\n"
            f"max_mag_diff_bright_floor = {cfg.phase01_comparison_max_mag_diff_bright_floor}\n"
            f"max_bv_diff = {cfg.phase01_comparison_max_bv_diff}\n"
            f"n_comp_min / max = {cfg.phase01_comparison_n_comp_min} / {cfg.phase01_comparison_n_comp_max}\n"
            f"max_comp_rms = {cfg.phase01_comparison_max_comp_rms}\n"
            f"min_dist_arcsec = {cfg.phase01_comparison_min_dist_arcsec}\n"
            f"min_frames_frac = {cfg.phase01_comparison_min_frames_frac}\n"
            f"exclude_gaia_nss = {cfg.phase01_comparison_exclude_gaia_nss}\n"
            f"exclude_gaia_extobj = {cfg.phase01_comparison_exclude_gaia_extobj}",
            language="text",
        )

    run_again = False
    first_run = False

    # ── Stavový panel ──
    if exists:
        ts = _results_timestamp(output_dir)
        st.success(f"✅ Fáza 0+1 prebehla: {ts}")
        col1, col2 = st.columns(2)
        with col1:
            run_again = st.button(
                "🔄 Spustiť znova",
                key="select_stars_run_again",
                type="secondary",
            )
        with col2:
            st.caption(f"FWHM: {fwhm_px} px | Per-frame dir: {per_frame_dir.name if per_frame_dir else '?'}")
    else:
        st.info("Fáza 0+1 ešte nebehala pre tento draft.")
        col1, col2 = st.columns(2)
        with col1:
            first_run = st.button(
                "▶ Spustiť Fázu 0+1",
                key="select_stars_first_run",
                type="primary",
            )
        with col2:
            st.caption(f"FWHM: {fwhm_px} px")

    should_run = (exists and run_again) or (not exists and first_run)

    # ── Spustenie jobu ──
    if should_run:
        try:
            vyvar_footer_running("Fáza 0+1", "Štartujem výber cieľov a porovnávačiek…")

            def _phase01_ui(msg: str) -> None:
                vyvar_footer_running("Fáza 0+1", msg)

            result = run_phase0_and_phase1(
                variable_targets_csv=vt_csv,
                masterstars_csv=ms_csv,
                per_frame_csv_dir=per_frame_dir,
                output_dir=output_dir,
                fwhm_px=fwhm_px,
                max_dist_deg=float(cfg.phase01_comparison_max_dist_deg),
                max_mag_diff=float(cfg.phase01_comparison_max_mag_diff),
                mag_bright_threshold=float(cfg.phase01_comparison_mag_bright_threshold),
                max_mag_diff_bright_floor=float(cfg.phase01_comparison_max_mag_diff_bright_floor),
                max_bv_diff=float(cfg.phase01_comparison_max_bv_diff),
                n_comp_min=int(cfg.phase01_comparison_n_comp_min),
                n_comp_max=int(cfg.phase01_comparison_n_comp_max),
                max_comp_rms=float(cfg.phase01_comparison_max_comp_rms),
                min_dist_arcsec=float(cfg.phase01_comparison_min_dist_arcsec),
                min_frames_frac=float(cfg.phase01_comparison_min_frames_frac),
                exclude_gaia_nss=bool(cfg.phase01_comparison_exclude_gaia_nss),
                exclude_gaia_extobj=bool(cfg.phase01_comparison_exclude_gaia_extobj),
                progress_cb=_phase01_ui,
            )
            st.success(
                f"✅ Hotovo: {result['n_active_targets']} premenných, "
                f"{result['n_comparison_pairs']} porovnávacích párov."
            )
            if result.get("targets_without_comps"):
                st.warning(
                    f"⚠️ {len(result['targets_without_comps'])} cieľov bez porovnávačiek: "
                    f"{', '.join(result['targets_without_comps'][:5])}"
                )
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f"❌ Chyba: {exc}")
            logging.exception("Select Stars Fáza 0+1 zlyhala")
        finally:
            vyvar_footer_idle()
        return

    # ── Zobraz výsledky ──
    if not exists:
        return

    try:
        active_df = pd.read_csv(output_dir / "active_targets.csv")
        comp_df = pd.read_csv(output_dir / "comparison_stars_per_target.csv")
        suspected_path = output_dir / "suspected_variables.csv"
        suspected_df = pd.read_csv(suspected_path) if suspected_path.exists() else pd.DataFrame()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Chyba pri načítaní výsledkov: {exc}")
        return

    # ── Súhrn ──
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Aktívne ciele", len(active_df))
    with c2:
        avg_comp = round(len(comp_df) / max(len(active_df), 1), 1)
        st.metric("Porovnávacie páry", len(comp_df), help=f"Priemer {avg_comp}/cieľ")
    with c3:
        st.metric("Suspected variables", len(suspected_df))

    st.divider()

    # ── Subtaby ──
    result_tabs = st.tabs(["🌟 Premenné hviezdy", "⚖️ Porovnávacie hviezdy", "🔍 Suspected Variables"])

    with result_tabs[0]:
        _render_targets_tab(active_df)

    with result_tabs[1]:
        _render_comparison_tab(comp_df)

    with result_tabs[2]:
        _render_suspected_tab(suspected_df)
