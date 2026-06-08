"""Select Stars dashboard — Fáza 0+1: výber aktívnych premenných a porovnávacích hviezd."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import streamlit as st

from photometry import run_phase0_and_phase1
from platesolve_ui_paths import default_bundle_dir, masterstars_csv_in_dir
from vyvar_ui_status import (
    is_bv_related_phase01_ui_column,
    log_if_ui_hiding_bv_for_bprp_primary,
    vyvar_footer_idle,
    vyvar_footer_running,
)

if TYPE_CHECKING:
    from config import AppConfig
    from pipeline import AstroPipeline

# Gaia ID musí byť str — float64 stráca cifry
_GAIA_ID_DTYPE: dict[str, type] = {"catalog_id": str, "name": str}

# NOTE: render_select_stars is not wired into app.py (inactive entry point).
# _sanitize_suspected_variables_df is still imported by ui_suspected_lightcurves.py.
# Functions are preserved for potential future use.
# Last audit: 2026-05-18


# ---------------------------------------------------------------------------
# Pomocné funkcie
# ---------------------------------------------------------------------------


def _find_phase01_setups(
    cfg: "AppConfig",
    draft_id: int | None,
    *,
    draft_dir_override: Path | None = None,
) -> dict[str, dict[str, Path | None]]:
    """Všetky platesolve setupy s ``per_frame_catalog_index.csv`` a cesty pre Fázu 0+1."""
    if draft_id is None and draft_dir_override is None:
        return {}
    try:
        archive = Path(cfg.archive_root)
        if draft_dir_override is not None and draft_dir_override.is_dir():
            draft_dir = draft_dir_override.resolve()
        elif draft_id is not None:
            draft_dir = (archive / "Drafts" / f"draft_{int(draft_id):06d}").resolve()
        else:
            return {}
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


def _default_setup_name(
    setups: dict[str, dict[str, Path | None]],
    cfg: "AppConfig",
    draft_id: int,
    *,
    draft_dir_override: Path | None = None,
) -> str:
    if not setups:
        return ""
    if draft_dir_override is not None:
        ps = draft_dir_override / "platesolve"
    elif draft_id > 0:
        ps = Path(cfg.archive_root) / "Drafts" / f"draft_{int(draft_id):06d}" / "platesolve"
    else:
        ps = Path()
    pick = default_bundle_dir(ps) if ps.is_dir() else None
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
    st.markdown(f"**{len(active_df)} active targets**")

    show_cols = [
        c
        for c in [
            "vsx_name",
            "vsx_type",
            "mag",
            "vsx_period",
            "ra_deg",
            "dec_deg",
            "zone_flag",
        ]
        if c in active_df.columns
    ]

    display = active_df[show_cols].copy()
    if "mag" in display.columns:
        display["mag"] = pd.to_numeric(display["mag"], errors="coerce").round(3)
    if "ra_deg" in display.columns:
        display["ra_deg"] = pd.to_numeric(display["ra_deg"], errors="coerce").round(5)
    if "dec_deg" in display.columns:
        display["dec_deg"] = pd.to_numeric(display["dec_deg"], errors="coerce").round(5)

    st.dataframe(display, width="stretch", hide_index=True)

    # Upozornenie na noisy1 hviezdy
    if "zone_flag" in active_df.columns:
        noisy1 = active_df[active_df["zone_flag"] == "noisy1"]
        if len(noisy1) > 0:
            st.warning(
                f"⚠️ {len(noisy1)} stars in **noisy1** zone — weaker signal, "
                f"possibly variable but photometry will be less precise."
            )


def _render_comparison_tab(comp_df: pd.DataFrame, *, use_bprp_primary: bool) -> None:
    """Tab: Porovnávacie hviezdy per target."""
    if comp_df.empty:
        st.info("No comparison stars.")
        return

    has_bv_cols = any(is_bv_related_phase01_ui_column(c) for c in comp_df.columns)
    if bool(use_bprp_primary) and has_bv_cols:
        log_if_ui_hiding_bv_for_bprp_primary(bprp_primary_ui_active=True)
    else:
        log_if_ui_hiding_bv_for_bprp_primary(bprp_primary_ui_active=False)

    # Dropdown na výber targetu
    target_options = []
    if "target_vsx_name" in comp_df.columns:
        target_options = sorted(comp_df["target_vsx_name"].dropna().unique().tolist())
    elif "target_catalog_id" in comp_df.columns:
        target_options = sorted(comp_df["target_catalog_id"].dropna().unique().tolist())

    if not target_options:
        if use_bprp_primary:
            drop_c = [c for c in comp_df.columns if is_bv_related_phase01_ui_column(c)]
            show_full = comp_df.drop(columns=drop_c, errors="ignore").copy()
        else:
            show_full = comp_df
        st.dataframe(show_full, width="stretch", hide_index=True)
        return

    selected = st.selectbox(
        "Select variable star:",
        options=target_options,
        key="select_stars_target_dropdown",
    )

    filter_col = "target_vsx_name" if "target_vsx_name" in comp_df.columns else "target_catalog_id"
    sub = comp_df[comp_df[filter_col] == selected].copy()

    st.markdown(f"**{len(sub)} comparison stars** for `{selected}`")

    if use_bprp_primary:
        preferred = [
            "catalog_id",
            "name",
            "mag",
            "bp_rp",
            "delta_bprp_abs",
            "color_tier_src",
            "comp_tier",
            "_dist_deg",
            "comp_rms",
            "comp_n_frames",
            "zone",
        ]
    else:
        preferred = [
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
    show_cols = [c for c in preferred if c in sub.columns]
    for c in sub.columns:
        if c in show_cols:
            continue
        if use_bprp_primary and is_bv_related_phase01_ui_column(c):
            continue
        show_cols.append(c)

    display = sub[show_cols].copy()
    for col in ("mag", "b_v", "bp_rp", "comp_rms", "delta_bprp_abs"):
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
            width="stretch",
            hide_index=True,
        )
    else:
        st.dataframe(display, width="stretch", hide_index=True)

    # RMS histogram
    if "comp_rms" in sub.columns and len(sub) > 1:
        st.markdown("**Comparison-star RMS distribution:**")
        rms_vals = sub["comp_rms"].dropna().tolist()
        rms_df = pd.DataFrame({"RMS": rms_vals})
        st.bar_chart(rms_df["RMS"])


def _sanitize_suspected_variables_df(df: pd.DataFrame) -> pd.DataFrame:
    """Odstráni poškodené / hlavičkové riadky zo starších CSV alebo zlých joinov."""
    if df.empty or "catalog_id" not in df.columns:
        return df
    from photometry_core import _normalize_id_value

    s = df.copy()
    nid = s["catalog_id"].map(_normalize_id_value)
    ok = nid.astype(str).str.len() > 0
    ok &= ~nid.astype(str).str.lower().isin(("none", "nan", "catalog_id", "name"))
    if "comp_rms" in s.columns:
        cr = pd.to_numeric(s["comp_rms"], errors="coerce")
        ok &= cr.notna() & cr.lt(500.0) & cr.gt(0.0)
    if "n_frames" in s.columns:
        nf = pd.to_numeric(s["n_frames"], errors="coerce")
        ok &= nf.notna() & nf.le(500_000) & nf.ge(1)
    if "zone" in s.columns:
        zt = s["zone"].astype(str)
        ok &= ~zt.str.contains(r"catalog_id.*linear", case=False, na=False, regex=True)
    return s.loc[ok].reset_index(drop=True)


def _render_suspected_tab(suspected_df: pd.DataFrame) -> None:
    """Tab: Suspected new variables."""
    if suspected_df.empty:
        st.success("No candidates for new variable stars.")
        return

    st.markdown(
        f"**{len(suspected_df)} candidates** for new variable stars "
        f"(high RMS scatter, not in VSX catalog)"
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
        width="stretch",
        hide_index=True,
    )


# ---------------------------------------------------------------------------
# Hlavný render
# ---------------------------------------------------------------------------


def render_select_stars(
    cfg: "AppConfig",
    draft_id: int | None,
    pipeline: "AstroPipeline",
    *,
    draft_dir_override: Path | None = None,
) -> None:
    """Hlavná funkcia pre Select Stars tab."""
    _ = pipeline
    st.header("Select Stars")
    st.caption("Phase 0+1: Select active variable stars and comparison stars.")

    if draft_id is None and draft_dir_override is None:
        st.info("No active draft. Run platesolve or load a draft above first.")
        return

    setups = _find_phase01_setups(cfg, draft_id, draft_dir_override=draft_dir_override)
    if not setups:
        st.warning(
            "Input files not found. Run platesolve first (plate-solve phase must be complete)."
        )
        return

    setup_names = sorted(setups)
    if len(setup_names) > 1:
        default_nm = _default_setup_name(
            setups,
            cfg,
            int(draft_id) if draft_id is not None else 0,
            draft_dir_override=draft_dir_override,
        )
        sel_ix = setup_names.index(default_nm) if default_nm in setup_names else 0
        chosen = st.selectbox(
            "Filter / group (platesolve):",
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
        missing.append("per-frame CSV directory (detrended_aligned)")

    if missing:
        st.error(f"Missing files: {', '.join(missing)}")
        return

    fwhm_px = _load_fwhm_from_masterstar(ms_fits if isinstance(ms_fits, Path) else None)
    exists = _results_exist(output_dir)

    with st.expander("Comparison-star selection rules (Phase 1) — from `config.json`", expanded=False):
        st.markdown(
            "Effective values from **AppConfig** (keys in parentheses). For sparse fields, typically increase "
            "**`phase01_comparison_max_mag_diff`** and **`phase01_comparison_max_dist_deg`**, or "
            "raise **`phase01_comparison_max_comp_rms`** or lower **`phase01_comparison_min_frames_frac`**. "
            "For **bright targets** (e.g. R~9 mag), set **`phase01_comparison_max_mag_diff_bright_floor`** "
            "(min. |Δmag| band; ``0`` disables) and optionally **`phase01_comparison_mag_bright_threshold`**. "
            "**Chip margin (same for variables, comps, and suspected):** "
            "``phase01_chip_interior_margin_px`` — px from edge; ``0`` = off."
        )
        _chip_line = (
            f"phase01_chip_interior_margin_px = 0  # no spatial clipping"
            if int(cfg.phase01_chip_interior_margin_px) <= 0
            else f"phase01_chip_interior_margin_px = {int(cfg.phase01_chip_interior_margin_px)}"
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
            f"exclude_gaia_extobj = {cfg.phase01_comparison_exclude_gaia_extobj}\n"
            f"{_chip_line}",
            language="text",
        )

    run_again = False
    first_run = False

    # ── Stavový panel ──
    if exists:
        ts = _results_timestamp(output_dir)
        st.success(f"✅ Phase 0+1 completed: {ts}")
        col1, col2 = st.columns(2)
        with col1:
            run_again = st.button(
                "🔄 Run again",
                key="select_stars_run_again",
                type="secondary",
            )
        with col2:
            st.caption(f"FWHM: {fwhm_px} px | Per-frame dir: {per_frame_dir.name if per_frame_dir else '?'}")
        if len(setup_names) > 1:
            st.caption("This button runs Phase **0+1 for all** filters / setups at once.")
    else:
        st.info("Phase 0+1 has not run for this draft yet.")
        col1, col2 = st.columns(2)
        with col1:
            first_run = st.button(
                "▶ Run Phase 0+1",
                key="select_stars_first_run",
                type="primary",
            )
        with col2:
            st.caption(f"FWHM: {fwhm_px} px")
        if len(setup_names) > 1:
            st.caption("This button runs Phase **0+1 for all** filters / setups at once.")

    should_run = (exists and run_again) or (not exists and first_run)

    # ── Spustenie jobu ──
    if should_run:
        try:
            vyvar_footer_running("Phase 0+1", "Starting target and comparison-star selection…")

            def _phase01_ui(msg: str) -> None:
                vyvar_footer_running("Phase 0+1", msg)

            setups_to_run = list(setup_names)
            errors: list[str] = []
            last_result: dict | None = None
            n_ok = 0
            for nm in setups_to_run:
                pth = setups[nm]
                _vt = pth.get("variable_targets_csv")
                _ms = pth.get("masterstars_csv")
                _pf = pth.get("per_frame_csv_dir")
                _out = pth.get("output_dir")
                _msfits = pth.get("masterstar_fits")
                miss: list[str] = []
                if _vt is None or not _vt.exists():
                    miss.append("variable_targets.csv")
                if _ms is None or not _ms.exists():
                    miss.append("masterstars_full_match.csv")
                if _pf is None or not _pf.exists():
                    miss.append("per-frame CSV directory (detrended_aligned)")
                if miss:
                    errors.append(f"{nm}: missing {', '.join(miss)}")
                    continue
                _fwhm = _load_fwhm_from_masterstar(_msfits if isinstance(_msfits, Path) else None)
                try:
                    _phase01_ui(f"Phase 0+1: {nm} …")
                    last_result = run_phase0_and_phase1(
                        variable_targets_csv=_vt,
                        masterstars_csv=_ms,
                        per_frame_csv_dir=_pf,
                        output_dir=_out,
                        fwhm_px=_fwhm,
                        cfg=cfg,
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
                        chip_interior_margin_px=int(cfg.phase01_chip_interior_margin_px),
                        progress_cb=_phase01_ui,
                    )
                    n_ok += 1
                except Exception as exc_nm:  # noqa: BLE001
                    errors.append(f"{nm}: {exc_nm}")
                    logging.exception("Select Stars Fáza 0+1 zlyhala pre %s", nm)

            if n_ok == len(setups_to_run) and not errors and last_result is not None:
                st.success(
                    f"✅ Done for all filters ({len(setups_to_run)}): last block "
                    f"{last_result['n_active_targets']} variables, "
                    f"{last_result['n_comparison_pairs']} pairs."
                )
                if last_result.get("targets_without_comps"):
                    st.warning(
                        f"⚠️ {len(last_result['targets_without_comps'])} targets without comparison stars "
                        f"(last setup): {', '.join(last_result['targets_without_comps'][:5])}"
                    )
            elif n_ok > 0:
                st.success(f"✅ Phase 0+1: {n_ok}/{len(setups_to_run)} setups succeeded.")
                if last_result and last_result.get("targets_without_comps"):
                    st.warning(
                        f"⚠️ {len(last_result['targets_without_comps'])} targets without comparison stars "
                        f"(last successful setup): {', '.join(last_result['targets_without_comps'][:5])}"
                    )
            if errors:
                (st.error if n_ok == 0 else st.warning)(
                    "Issues on some filters:\n" + "\n".join(errors)
                )
            if n_ok:
                st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f"❌ Error: {exc}")
            logging.exception("Select Stars Fáza 0+1 zlyhala")
        finally:
            vyvar_footer_idle()
        return

    # ── Zobraz výsledky ──
    if not exists:
        return

    try:
        active_df = pd.read_csv(output_dir / "active_targets.csv", dtype=_GAIA_ID_DTYPE)
        comp_df = pd.read_csv(
            output_dir / "comparison_stars_per_target.csv",
            dtype={**_GAIA_ID_DTYPE, "target_catalog_id": str},
        )
        suspected_path = output_dir / "suspected_variables.csv"
        suspected_df = (
            pd.read_csv(suspected_path, dtype=_GAIA_ID_DTYPE) if suspected_path.exists() else pd.DataFrame()
        )
        suspected_df = _sanitize_suspected_variables_df(suspected_df)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Error loading results: {exc}")
        return

    # ── Súhrn ──
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Active targets", len(active_df))
    with c2:
        avg_comp = round(len(comp_df) / max(len(active_df), 1), 1)
        st.metric("Comparison pairs", len(comp_df), help=f"Average {avg_comp}/target")
    with c3:
        st.metric("Suspected variables", len(suspected_df))

    st.divider()

    # ── Subtaby ──
    result_tabs = st.tabs(["🌟 Variable stars", "⚖️ Comparison stars", "🔍 Suspected Variables"])

    with result_tabs[0]:
        _render_targets_tab(active_df)

    with result_tabs[1]:
        _render_comparison_tab(
            comp_df,
            use_bprp_primary=bool(getattr(cfg, "phase01_use_bprp_primary", True)),
        )

    with result_tabs[2]:
        _render_suspected_tab(suspected_df)
