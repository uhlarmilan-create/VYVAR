"""ePSF photometry tab (EPSF-TARGETS-2026-10, Milan 2026-10-19).

RUN ePSF runs after the aperture photometry of a draft: automatically when ``epsf_auto_run`` is on
(night run / full flow), otherwise from this tab. The user chooses the targets (``ALL`` or VSX types,
names, catalog ids - ``epsf_science_set.build_epsf_fit_set``); the tab then shows the PSF light curve
of each measured target, optionally over its aperture light curve.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import streamlit as st

from epsf_science_set import parse_epsf_selection
from utils import resolve_draft_dir_path

if TYPE_CHECKING:
    from config import AppConfig

_TARGETS_HELP = (
    "ALL = every VSX target that has aperture photometry (types excluded in config.json "
    "vsx_out_of_scope_types have none). Or a comma-separated list of VSX types (EA, EB, EW), star names "
    "(V0393 Cyg) or catalog ids; * is a wildcard (E* = every type starting with E, ZTF* = every ZTF name). "
    "The comparison stars of the chosen targets are measured with them."
)


def psf_lightcurve_targets(platesolve_dir: Path) -> pd.DataFrame:
    """Targets with an internal PSF light curve: catalog_id, name, vsx_type, mag (from active_targets)."""
    lc_dir = Path(platesolve_dir) / "photometry" / "lightcurves"
    ids = sorted(p.name[len("lightcurve_") : -len("_psf.csv")] for p in lc_dir.glob("lightcurve_*_psf.csv"))
    out = pd.DataFrame({"catalog_id": ids})
    if out.empty:
        return out.assign(name=[], vsx_type=[], mag=[])
    from epsf_science_set import epsf_target_candidates

    cand = epsf_target_candidates(platesolve_dir).set_index("catalog_id")
    out["name"] = [str(cand["name"].get(c, c)) if c in cand.index else c for c in ids]
    out["vsx_type"] = [str(cand["vsx_type"].get(c, "")) if c in cand.index else "" for c in ids]
    out["mag"] = [cand["mag"].get(c, np.nan) if c in cand.index else np.nan for c in ids]
    return out.sort_values(["mag", "name"], na_position="last").reset_index(drop=True)


def psf_lc_figure(df: pd.DataFrame, label: str, *, show_aperture: bool):
    """PSF light curve of one target; with ``show_aperture`` the aperture light curve under it.

    ``df`` is ``epsf_lc_compare.load_pair``. Both curves are differential (the ePSF ensemble is the aperture
    ensemble without comparison stars that have no ePSF measurement); their ensembles weight the stars by each method's own scatter, so the levels differ by a
    constant - the PSF curve is shifted onto the aperture level (median over the common epochs).
    """
    import plotly.graph_objects as go

    ap = df.loc[df["ap_ok"]]
    ps = df.loc[df["psf_ok"]]
    both = df.loc[df["ap_ok"] & df["psf_ok"]]
    shift = float(np.median(both["psf_mag"] - both["ap_mag"])) if show_aperture and len(both) else 0.0
    t0 = float(np.floor(np.nanmin(df["bjd"]))) if np.isfinite(df["bjd"]).any() else 0.0
    fig = go.Figure()
    if show_aperture:
        fig.add_trace(go.Scatter(
            x=ap["bjd"] - t0, y=ap["ap_mag"], mode="markers", name="aperture", customdata=ap["bjd"],
            marker=dict(size=5, color="rgba(42,120,214,0.55)"),
            error_y=dict(type="data", array=ap["ap_err"], thickness=1, width=0, color="rgba(42,120,214,0.3)"),
            hovertemplate="BJD %{customdata:.5f}<br>aperture %{y:.4f} mag<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=ps["bjd"] - t0, y=ps["psf_mag"] - shift, mode="markers", name="ePSF", customdata=ps["bjd"],
        marker=dict(size=7, color="#eb6834", symbol="diamond"),
        error_y=dict(type="data", array=ps["psf_err"], thickness=1, width=0, color="rgba(235,104,52,0.45)"),
        hovertemplate="BJD %{customdata:.5f}<br>ePSF %{y:.4f} mag<extra></extra>",
    ))
    fig.update_layout(
        title=label, height=480, margin=dict(t=60, b=40), legend=dict(orientation="h", y=1.08),
        xaxis_title=f"BJD_TDB - {t0:.0f}", yaxis_title="differential magnitude",
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickformat=".3f")
    return fig


@dataclass(frozen=True)
class EpsfSetupRun:
    """One setup (filter / obs group) of the draft in a RUN ePSF: its paths, its fit set and what is missing."""

    setup: str
    platesolve_dir: Path | None
    masterstar_fits: Path | None
    masterstars_csv: Path | None
    frames_dir: Path | None
    fit: Any
    missing: tuple[str, ...]

    @property
    def runnable(self) -> bool:
        return bool(self.fit.target_ids) and not self.missing

    def pending_entry(self) -> dict[str, str]:
        return {
            "setup_name": self.setup,
            "masterstar_fits_path": str(self.masterstar_fits),
            "masterstars_csv_path": str(self.masterstars_csv),
            "per_frame_csv_dir": str(self.frames_dir),
            "platesolve_dir": str(self.platesolve_dir),
        }


def epsf_run_plan(all_setups: dict[str, dict[str, Any]], selection: str) -> list[EpsfSetupRun]:
    """EPSF-MULTISETUP-2026-10 (Milan 2026-10-19): RUN ePSF measures every setup of the draft with the same
    target selection - each setup has its own targets with aperture photometry and its own comparison stars."""
    from epsf_science_set import EpsfFitSet, build_epsf_fit_set

    plan = []
    for setup, paths in all_setups.items():
        og = paths.get("obs_group_dir")
        ps = Path(og) if og is not None and Path(og).is_dir() else None
        ms_fits = Path(paths["masterstar_fits"]) if paths.get("masterstar_fits") else None
        ms_csv = ps / "masterstars_full_match.csv" if ps is not None else None
        frames = Path(paths["per_frame_csv_dir"]) if paths.get("per_frame_csv_dir") else None
        missing = [lbl for lbl, f in (("MASTERSTAR.fits", ms_fits), ("masterstars_full_match.csv", ms_csv))
                   if f is None or not f.is_file()]
        if frames is None or not frames.is_dir():
            missing.append("aligned frames directory")
        if ps is None:
            fit = EpsfFitSet(frozenset(), frozenset(), selection=selection, empty_reason="setup directory not found")
            missing.insert(0, "setup directory")
        else:
            fit = build_epsf_fit_set(ps, selection)
        plan.append(EpsfSetupRun(setup, ps, ms_fits, ms_csv, frames, fit, tuple(missing)))
    return plan


def _render_target_selection(cfg: "AppConfig", all_setups: dict[str, dict[str, Any]], draft_key: str) -> list[EpsfSetupRun]:
    selection = st.text_input(
        "Targets for ePSF photometry",
        value=str(getattr(cfg, "epsf_targets", "ALL") or "ALL"),
        key=f"epsf_targets_{draft_key}",
        help=_TARGETS_HELP,
    )
    plan = epsf_run_plan(all_setups, selection)
    rows = []
    for r in plan:
        if r.missing:
            status = "missing: " + ", ".join(r.missing)
        elif not r.fit.target_ids:
            status = f"nothing to fit: {r.fit.empty_reason}"
        else:
            status = "will run"
        rows.append({"setup": r.setup, "targets": len(r.fit.target_ids), "of candidates": r.fit.n_candidates,
                     "comparison stars": len(r.fit.comp_ids - r.fit.target_ids), "status": status})
    n_run = sum(r.runnable for r in plan)
    st.caption(f"RUN ePSF measures every setup of the night with this selection: {n_run} of {len(plan)} setup(s).")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    unmatched = set.intersection(*(set(r.fit.unmatched_tokens) for r in plan)) if plan else set()
    if unmatched:
        st.warning("Not matched in any setup (check the spelling): "
                   + ", ".join(t for t in parse_epsf_selection(selection) if t in unmatched))
    from epsf_science_set import epsf_target_candidates

    for r in plan:
        if not r.fit.target_ids:
            continue
        with st.expander(f"Selected targets - {r.setup} ({len(r.fit.target_ids)})", expanded=False):
            cand = epsf_target_candidates(r.platesolve_dir)
            sel = cand[cand["catalog_id"].isin(r.fit.target_ids)].sort_values("mag")
            st.dataframe(
                sel.rename(columns={"name": "star", "vsx_type": "VSX type", "mag": "G"})[["star", "VSX type", "G"]],
                hide_index=True, width="stretch",
                column_config={"G": st.column_config.NumberColumn("G", format="%.2f")},
            )
    return plan


def generate_epsf_files(all_setups: dict[str, dict[str, Any]], observer_code: str) -> dict[str, int]:
    """EPSF-EXPORT-2026-10: write the user light-curve files of every setup from its existing ePSF light
    curves (``epsf_export.write_epsf_user_lightcurves``). Returns {setup: files written}."""
    from epsf_export import write_epsf_user_lightcurves

    out: dict[str, int] = {}
    for setup, paths in all_setups.items():
        og = paths.get("obs_group_dir")
        if og is None or not Path(og).is_dir():
            continue
        ids = psf_lightcurve_targets(Path(og))["catalog_id"].tolist()
        if ids:
            out[setup] = int(write_epsf_user_lightcurves(Path(og), ids, setup=setup,
                                                         observer_code=observer_code)["n_written"])
    return out


def _render_export_button(cfg: "AppConfig", all_setups: dict[str, dict[str, Any]], draft_key: str) -> None:
    auto = bool(getattr(cfg, "epsf_export_auto", True))
    st.caption(
        "ePSF light-curve files for your own use (bjd_tdb, delta_mag, err + header; not for AAVSO / VarAstro) go to "
        "`photometry/lightcurves_reports/epsf` next to the `aavso` and `varastro` folders. "
        + ("RUN ePSF writes them (`epsf_export_auto` in config.json)." if auto
           else "`epsf_export_auto` is off in config.json: write them with the button.")
    )
    if st.button("Generate ePSF light-curve files", key=f"epsf_export_btn_{draft_key}",
                 help="Writes the files of every setup from its existing ePSF light curves (replaces older files)."):
        done = generate_epsf_files(all_setups, str(getattr(cfg, "observer_code", "") or ""))
        if done:
            st.success("ePSF light-curve files written: " + ", ".join(f"{k} {v}" for k, v in done.items()))
        else:
            st.warning("No ePSF light curves in this draft - run ePSF photometry first.")


def _render_light_curves(ps_dir: Path, setup: str) -> None:
    from epsf_lc_compare import compare_draft, frame_fwhm_map, load_pair

    tg = psf_lightcurve_targets(ps_dir)
    if tg.empty:
        st.info("No ePSF light curves yet for this setup - run ePSF photometry above.")
        return
    labels = {r.catalog_id: f"{r.name} ({r.vsx_type or '?'}, G {r.mag:.2f})" if np.isfinite(r.mag) else r.name
              for r in tg.itertuples()}
    c1, c2 = st.columns([3, 1])
    with c1:
        cid = st.selectbox("Target", options=tg["catalog_id"].tolist(), format_func=lambda c: labels.get(c, c),
                           key=f"epsf_lc_target_{setup}")
    with c2:
        show_ap = st.checkbox("Aperture LC", value=True, key=f"epsf_lc_ap_{setup}",
                              help="Show the aperture light curve of the same target for comparison")
    lc_dir = ps_dir / "photometry" / "lightcurves"
    df = load_pair(lc_dir, str(cid), frame_fwhm_map(ps_dir))
    st.plotly_chart(psf_lc_figure(df, labels.get(cid, cid), show_aperture=show_ap), width="stretch")
    st.caption(
        "Both curves are differential; the ePSF ensemble is the aperture ensemble without the comparison stars "
        "that have no ePSF measurement on the night. The ePSF curve is drawn at the aperture "
        "level (the two ensembles weight their stars by each method's own scatter, so the levels differ by a "
        "constant). AAVSO / VarAstro exports are aperture photometry."
    )
    from epsf_export import find_epsf_export

    p = find_epsf_export(ps_dir / "photometry", str(cid), setup)
    if p is not None:
        st.download_button(f"Download ePSF light curve ({p.name})", data=p.read_bytes(), file_name=p.name,
                           mime="text/plain", key=f"epsf_dl_{setup}_{cid}")

    with st.expander("All measured targets: ePSF vs aperture scatter", expanded=False):
        table = compare_draft(ps_dir, tg["catalog_id"].tolist())
        if table.empty:
            st.caption("No pairs to compare.")
        else:
            table.insert(1, "star", [labels.get(c, c) for c in table["catalog_id"]])
            table["psf_over_ap"] = table["psf_p2p_mmag"] / table["ap_p2p_mmag"]
            st.caption(
                "p2p = robust epoch-to-epoch scatter; p2p/err = 1 when the quoted errors describe the scatter. "
                "Both on the epochs measured by both methods."
            )
            st.dataframe(
                table[["star", "n_both", "ap_p2p_mmag", "psf_p2p_mmag", "psf_over_ap", "ap_p2p_over_err",
                       "psf_p2p_over_err"]],
                hide_index=True, width="stretch",
                column_config={
                    "n_both": st.column_config.NumberColumn("epochs", format="%d"),
                    "ap_p2p_mmag": st.column_config.NumberColumn("aperture p2p [mmag]", format="%.1f"),
                    "psf_p2p_mmag": st.column_config.NumberColumn("ePSF p2p [mmag]", format="%.1f"),
                    "psf_over_ap": st.column_config.NumberColumn("ePSF / aperture", format="%.2f"),
                    "ap_p2p_over_err": st.column_config.NumberColumn("aperture p2p/err", format="%.2f"),
                    "psf_p2p_over_err": st.column_config.NumberColumn("ePSF p2p/err", format="%.2f"),
                },
            )
    with st.expander("Per-frame ePSF models", expanded=False):
        _render_frame_models(ps_dir)


def _render_frame_models(platesolve_dir: Path) -> None:
    """EPSF-FRAME-2026-10: FWHM and build of the per-frame ePSF models (interactive)."""
    from psf_photometry import EPSF_FRAMES_DIR, EPSF_FRAMES_MANIFEST

    man_p = Path(platesolve_dir) / EPSF_FRAMES_DIR / EPSF_FRAMES_MANIFEST
    if not man_p.is_file():
        st.caption("No per-frame ePSF models yet.")
        return
    man = json.loads(man_p.read_text(encoding="utf-8"))
    fr = pd.DataFrame(man.get("frames") or [])
    if fr.empty:
        return
    import plotly.graph_objects as go

    fs = man.get("fit_set") or {}
    st.caption(
        f"Last run: targets '{fs.get('selection', '?')}', {fs.get('n_targets', '?')} targets + "
        f"{fs.get('n_comps', '?')} comparison stars; {len(fr)} frames, "
        f"{int((~fr['converged'].astype(bool)).sum())} models not converged."
    )
    colors = np.where(fr["converged"].astype(bool), "#2a78d6", "#d9534f")
    fig = go.Figure(go.Scatter(
        x=list(range(1, len(fr) + 1)), y=fr["fwhm_px"], mode="lines+markers",
        marker=dict(size=6, color=colors), line=dict(width=1, color="#9bb7d4"),
        customdata=np.c_[fr["frame"], fr["n_stars_used"], fr["iterations"]],
        hovertemplate="%{customdata[0]}<br>FWHM %{y:.2f} px<br>build stars %{customdata[1]}"
        "<br>iterations %{customdata[2]}<extra></extra>",
    ))
    fig.update_layout(xaxis_title="frame", yaxis_title="ePSF FWHM [px]", height=280, margin=dict(t=20, b=40))
    st.plotly_chart(fig, width="stretch")


def render_epsf_dashboard(
    draft_dir: Path | None,
    cfg: "AppConfig",
    *,
    draft_id: int | None = None,
) -> None:
    """ePSF tab: target selection + RUN ePSF, then the PSF light curves."""
    st.header("ePSF Photometry")
    resolved = resolve_draft_dir_path(draft_dir, draft_id, cfg.archive_root, drafts_before_session=True)
    if resolved is None:
        st.info("No active draft. Load a draft above or run VAR-STREM.")
        return
    draft_path = Path(resolved)

    from platesolve_ui_paths import find_photometry_setups

    all_setups = find_photometry_setups(cfg, draft_id, draft_dir_override=draft_path)
    if not all_setups:
        st.warning("No platesolve setups with per-frame catalogs found for this draft.")
        return
    if bool(getattr(cfg, "epsf_auto_run", False)):
        st.caption(f"ePSF is ON in config.json (epsf_auto_run): it runs after the aperture photometry for "
                   f"targets '{getattr(cfg, 'epsf_targets', 'ALL')}'. Run it again here with another selection.")
    else:
        st.caption("ePSF is OFF in config.json (epsf_auto_run): run it here after the aperture photometry.")

    plan = _render_target_selection(cfg, all_setups, draft_path.name)
    runnable = [r for r in plan if r.runnable]
    run = st.button("RUN ePSF photometry", type="primary", key=f"epsf_run_btn_{draft_path.name}",
                    disabled=not runnable,
                    help="For every setup of the night: builds the ePSF of every frame and fits the selected "
                    "targets with their comparison stars; writes their ePSF light curves. It replaces the previous "
                    "ePSF run: light curves of targets outside the new selection are removed.")
    if run:
        st.session_state["vyvar_pending_job"] = {
            "kind": "run_epsf",
            "label": "RUN ePSF Photometry...",
            "archive_path": str(draft_path),
            "draft_id": int(draft_id) if draft_id is not None else None,
            "epsf_targets": runnable[0].fit.selection,
            "setups": [r.pending_entry() for r in runnable],
        }
        st.rerun()

    last = st.session_state.get("vyvar_last_job_output")
    if isinstance(last, dict) and last.get("job_kind") == "run_epsf" and last.get("archive_path") == str(draft_path):
        st.success(str(last.get("message")))

    st.divider()
    st.subheader("Light curves")
    _render_export_button(cfg, all_setups, draft_path.name)
    setup = str(st.selectbox("Platesolve setup:", options=list(all_setups.keys()), key="epsf_setup_select"))
    og_dir = (all_setups.get(setup) or {}).get("obs_group_dir")
    if og_dir is None or not Path(og_dir).is_dir():
        st.error("Platesolve setup directory not found.")
        return
    _render_light_curves(Path(og_dir), setup)
