#!/usr/bin/env python3
"""Read-only Gaia<->DAO reconciliation diagnostic (footprint reference + Fleming fit).

Usage:
  python scripts/dao_reconcile_diag.py --draft 424
  python scripts/dao_reconcile_diag.py --all-drafts
  python scripts/dao_reconcile_diag.py --setup "C:/.../platesolve/NoFilter_60_2"
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from astropy.io import fits  # noqa: E402
from astropy.wcs import WCS  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: E402,F401  (repo layout: src_py + dev on sys.path)
_ROOT = _bootstrap.REPO_ROOT
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import AppConfig  # noqa: E402
from dao_reconcile import (  # noqa: E402
    ReferencePopulationMismatch,
    compute_gaia_dao_reconcile,
    fleming_completeness,
    resolve_effective_match_depth,
)
from gaia_catalog_id import read_vyvar_csv  # noqa: E402
from masterstar_context import header_core_fwhm_px, load_masterstar_context  # noqa: E402


def _resolve_setup_dir(cfg: AppConfig, draft: int | None, setup: str | None) -> Path:
    if setup:
        p = Path(setup).resolve()
        if not p.is_dir():
            raise SystemExit(f"Setup directory not found: {p}")
        return p
    if draft is None:
        raise SystemExit("Provide --draft N, --all-drafts, or --setup PATH")
    p = (Path(cfg.archive_root) / "Drafts" / f"draft_{int(draft):06d}" / "platesolve").resolve()
    if not p.is_dir():
        raise SystemExit(f"Platesolve dir not found: {p}")
    subs = sorted(d for d in p.iterdir() if d.is_dir())
    if len(subs) == 1:
        return subs[0]
    for d in subs:
        if (d / "MASTERSTAR.fits").is_file():
            return d
    if subs:
        return subs[0]
    raise SystemExit(f"No setup subdir under {p}")


def _enumerate_draft_setups(cfg: AppConfig) -> list[tuple[str, Path, str]]:
    """Return (draft_label, setup_dir, setup_name) for each solved MASTERSTAR bundle."""
    out: list[tuple[str, Path, str]] = []
    drafts_root = Path(cfg.archive_root) / "Drafts"
    if not drafts_root.is_dir():
        return out
    for draft_dir in sorted(drafts_root.glob("draft_*")):
        if not draft_dir.is_dir():
            continue
        ps = draft_dir / "platesolve"
        if not ps.is_dir():
            continue
        for setup_dir in sorted(ps.iterdir()):
            if not setup_dir.is_dir():
                continue
            if not (setup_dir / "MASTERSTAR.fits").is_file():
                continue
            if not (setup_dir / "masterstars_full_match.csv").is_file():
                continue
            out.append((draft_dir.name, setup_dir, setup_dir.name))
    return out


def _load_inputs(setup_dir: Path) -> dict:
    ms_path = setup_dir / "masterstars_full_match.csv"
    fits_path = setup_dir / "MASTERSTAR.fits"
    cone_path = setup_dir / "field_catalog_cone.csv"
    if not ms_path.is_file():
        raise FileNotFoundError(f"Missing masterstars_full_match.csv in {setup_dir}")
    if not fits_path.is_file():
        raise FileNotFoundError(f"Missing MASTERSTAR.fits in {setup_dir}")

    det = read_vyvar_csv(ms_path, low_memory=False, dtype={"catalog_id": str})
    cone = (
        read_vyvar_csv(cone_path, low_memory=False, dtype={"catalog_id": str})
        if cone_path.is_file()
        else None
    )

    fwhm_px = 3.5
    plate_scale = None
    wcs = None
    naxis1 = naxis2 = 0
    ctx = load_masterstar_context(fits_path)
    fwhm_px = ctx.vy_fwhm_gauss_px or ctx.vy_fwhm_px or fwhm_px
    plate_scale = ctx.pixel_scale_arcsec
    with fits.open(fits_path, memmap=False) as hdul:
        hdr = hdul[0].header
        wcs = WCS(hdr)
        naxis1 = int(hdr.get("NAXIS1") or 0)
        naxis2 = int(hdr.get("NAXIS2") or 0)
        if fwhm_px is None or fwhm_px <= 0:
            fwhm_px = header_core_fwhm_px(hdr) or 3.5

    meta_path = setup_dir / "photometry" / "pipeline_meta.json"
    mag_limit = 18.0
    match_sep = 8.0
    pipeline_meta: dict = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        pipeline_meta = meta
        if meta.get("dao_fwhm_px") is not None:
            fwhm_px = float(meta["dao_fwhm_px"])
        match_depth_info = resolve_effective_match_depth(meta, is_masterstar=True)
        if match_depth_info.get("match_depth") is not None:
            mag_limit = float(match_depth_info["match_depth"])
        elif meta.get("faintest_mag_limit") is not None:
            mag_limit = float(meta["faintest_mag_limit"])
        match_sep = float(
            meta.get("match_sep_arcsec_effective")
            or meta.get("match_sep_arcsec_requested")
            or match_sep
        )
    else:
        match_depth_info = resolve_effective_match_depth(None, is_masterstar=True)

    return {
        "detections": det,
        "cone": cone,
        "fwhm_px": float(fwhm_px),
        "plate_scale": plate_scale,
        "wcs": wcs,
        "naxis1": naxis1,
        "naxis2": naxis2,
        "mag_limit": mag_limit,
        "match_sep_arcsec": match_sep,
        "fits_path": fits_path,
        "pipeline_meta": pipeline_meta,
        "match_depth_info": match_depth_info,
    }


def _save_missed_g_histogram(report: dict, labeled: pd.DataFrame, out_dir: Path) -> str | None:
    if labeled is None or labeled.empty or "_bucket" not in labeled.columns:
        return None
    missed = labeled.loc[labeled["_bucket"] == "genuinely_missed"]
    mags = pd.to_numeric(missed.get("_mag", missed.get("mag")), errors="coerce").dropna()
    if mags.empty:
        return None
    g50 = report.get("g_lim_50")
    g90 = report.get("g_lim_90")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(mags, bins=30, color="#E45756", edgecolor="white", alpha=0.9)
    if g50 is not None:
        label50 = report.get("g_lim_50_display") or f"G_lim_50={g50}"
        ax.axvline(float(g50), color="#72B7B2", linestyle=":", linewidth=2, label=label50)
    if g90 is not None:
        label90 = report.get("g_lim_90_display") or f"G_lim_90={g90}"
        ax.axvline(float(g90), color="#F58518", linestyle=":", linewidth=2, label=label90)
    ax.set_xlabel("Gaia G (genuinely missed)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Missed-G histogram (below_g90={report.get('n_missed_below_g90', '?')} "
        f"fadezone={report.get('n_missed_fadezone', '?')})"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    png = out_dir / "g_hist_missed_g90.png"
    fig.savefig(png, dpi=120)
    plt.close(fig)
    return str(png)


def _format_g_lim(report: dict, key: str) -> str:
    display = report.get(f"{key}_display")
    if display:
        return str(display)
    val = report.get(key)
    censored = report.get(f"{key}_censored")
    if val is None:
        return "?"
    if censored:
        return f">= {float(val):.1f} (censored)"
    return f"{float(val):.2f}"


def _print_cross_draft_table(results: list[dict]) -> None:
    if not results:
        return
    print("\n=== Cross-draft summary (R-2b) ===")
    hdr = (
        f"{'draft':<14} {'setup':<14} {'G_lim_50':<22} {'G_lim_90':<22} "
        f"{'compl_50':>8} {'miss@G90':>8} {'fadezone':>8} {'match_depth':>11}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(results, key=lambda x: (x.get("draft_label", ""), x.get("setup_name", ""))):
        print(
            f"{r.get('draft_label', '?'):<14} "
            f"{r.get('setup_name', '?'):<14} "
            f"{_format_g_lim(r, 'g_lim_50'):<22} "
            f"{_format_g_lim(r, 'g_lim_90'):<22} "
            f"{r.get('gaia_dao_completeness_pct') or '?':>8} "
            f"{r.get('n_missed_below_g90', '?'):>8} "
            f"{r.get('n_missed_fadezone', '?'):>8} "
            f"{r.get('match_depth', '?'):>11}"
        )


def _save_histograms(labeled: pd.DataFrame, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    buckets = [
        ("matched", "Matched"),
        ("below_limit", "Below limit (G > G_lim_50)"),
        ("blended", "Blended"),
        ("genuinely_missed", "Genuinely missed"),
        ("off_frame", "Off-frame"),
    ]
    for key, title in buckets:
        if "_bucket" not in labeled.columns:
            continue
        sub = labeled.loc[labeled["_bucket"] == key]
        mags = pd.to_numeric(sub.get("_mag", sub.get("mag")), errors="coerce").dropna()
        if mags.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(mags, bins=40, color="#4C78A8", edgecolor="white", alpha=0.9)
        ax.set_xlabel("Gaia G")
        ax.set_ylabel("Count")
        ax.set_title(f"{title} (n={len(mags)})")
        fig.tight_layout()
        png = out_dir / f"g_hist_{key}.png"
        fig.savefig(png, dpi=120)
        plt.close(fig)
        paths.append(str(png))
    return paths


def _save_completeness_curve_png(report: dict, out_dir: Path) -> str | None:
    curve = report.get("completeness_curve") or []
    if not curve:
        return None
    centers = [float(c["bin_center"]) for c in curve]
    fracs = [float(c["completeness_frac"]) for c in curve]
    g50 = report.get("g_lim_50")
    g90 = report.get("g_lim_90")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(centers, fracs, "o-", color="#4C78A8", label="binned")
    params = report.get("fleming_fit_params") or {}
    if report.get("fit_method") == "fleming1995_erf" and params.get("sigma_mag"):
        grid = np.linspace(min(centers), max(centers), 100)
        model = fleming_completeness(grid, float(g50), float(params["sigma_mag"]))
        ax.plot(grid, model, "--", color="#E45756", label="Fleming fit")
    if g50 is not None:
        ax.axvline(float(g50), color="#72B7B2", linestyle=":", label=f"G_lim_50={g50}")
    if g90 is not None:
        ax.axvline(float(g90), color="#F58518", linestyle=":", label=f"G_lim_90={g90}")
    ax.set_xlabel("Gaia G")
    ax.set_ylabel("Completeness fraction")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.set_title("Completeness curve")
    fig.tight_layout()
    png = out_dir / "completeness_curve.png"
    fig.savefig(png, dpi=120)
    plt.close(fig)
    return str(png)


def _report_to_jsonable(report: dict) -> dict:
    out = {k: v for k, v in report.items() if k not in ("labeled_reference", "labeled_cone")}
    return out


def _print_report(report: dict) -> None:
    print("=== Gaia<->DAO reconciliation (R-2b footprint) ===")
    print(
        f"G_lim_50: {_format_g_lim(report, 'g_lim_50')} | "
        f"G_lim_90: {_format_g_lim(report, 'g_lim_90')} | fit: {report.get('fit_method')}"
    )
    if report.get("completeness_50_label"):
        print(f"Headline: {report.get('completeness_50_label')}")
    print(f"match_depth: {report.get('match_depth')} ({report.get('match_depth_source', '?')})")
    print(f"n_ref_in_frame: {report.get('n_ref_in_frame')} | off-frame: {report.get('n_gaia_off_frame')}")
    print(f"  matched:         {report.get('n_gaia_matched')}")
    print(f"  below-limit:     {report.get('n_gaia_below_limit')}")
    print(f"  blended:         {report.get('n_gaia_blended')}")
    print(f"  genuinely-missed:{report.get('n_gaia_missed')} "
          f"(below_g90={report.get('n_missed_below_g90')} fadezone={report.get('n_missed_fadezone')})")
    print(f"completeness_50: {report.get('gaia_dao_completeness_pct')}%")
    if report.get("gaia_dao_completeness_raw_pct") is not None:
        print(f"raw (cone legacy): {report.get('gaia_dao_completeness_raw_pct')}%")
    ud = report.get("unmatched_dao") or {}
    print(
        f"Unmatched DAO: {ud.get('n_dao_unmatched')} "
        f"(faint-real={ud.get('n_now_matched_to_faint')} "
        f"artifact={ud.get('n_artifact_candidates')} "
        f"unexplained={ud.get('n_unexplained')})"
    )


def run_diagnostic(
    setup_dir: Path,
    cfg: AppConfig,
    *,
    draft_label: str | None = None,
) -> dict:
    inp = _load_inputs(setup_dir)
    gaia_db = str(cfg.gaia_db_path or "").strip()
    if not gaia_db:
        raise RuntimeError("GAIA_DB_PATH not configured")

    report = compute_gaia_dao_reconcile(
        inp["detections"],
        gaia_db_path=gaia_db,
        wcs=inp["wcs"],
        naxis1=int(inp["naxis1"]),
        naxis2=int(inp["naxis2"]),
        fwhm_px=float(inp["fwhm_px"]),
        plate_scale_arcsec=inp["plate_scale"],
        mag_limit=float(inp["mag_limit"]),
        match_sep_arcsec=float(inp["match_sep_arcsec"]),
        cone_df=inp["cone"],
    )
    md = inp.get("match_depth_info") or resolve_effective_match_depth(inp.get("pipeline_meta"), is_masterstar=True)
    report["match_depth"] = md.get("match_depth")
    report["match_depth_source"] = md.get("match_depth_source")
    report["faintest_mag_limit_config"] = md.get("faintest_mag_limit_config")

    label = draft_label or setup_dir.name
    out_dir = _ROOT / "tmp" / "dao_reconcile" / label / setup_dir.name
    labeled = report.get("labeled_reference")
    hist_paths = _save_histograms(labeled, out_dir) if labeled is not None else []
    missed_png = _save_missed_g_histogram(report, labeled, out_dir) if labeled is not None else None
    curve_png = _save_completeness_curve_png(report, out_dir)
    payload = _report_to_jsonable(report)
    payload["setup_dir"] = str(setup_dir)
    payload["draft_label"] = label
    payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["histogram_pngs"] = hist_paths
    if missed_png:
        payload["missed_g_histogram_png"] = missed_png
    if curve_png:
        payload["completeness_curve_png"] = curve_png
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "report.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["report_json"] = str(json_path)
    _print_report(report)
    print(f"Report JSON: {json_path}")
    return payload


def run_all_drafts(cfg: AppConfig) -> tuple[list[dict], list[dict]]:
    results: list[dict] = []
    skipped: list[dict] = []
    for draft_label, setup_dir, setup_name in _enumerate_draft_setups(cfg):
        key = f"{draft_label}/{setup_name}"
        print(f"\n--- {key} ---")
        try:
            rep = run_diagnostic(setup_dir, cfg, draft_label=draft_label)
            rep["setup_name"] = setup_name
            results.append(rep)
        except (ReferencePopulationMismatch, FileNotFoundError, RuntimeError) as exc:
            skipped.append({"draft": draft_label, "setup": setup_name, "reason": str(exc)})
            print(f"SKIP {key}: {exc}")
        except Exception as exc:  # noqa: BLE001
            skipped.append({"draft": draft_label, "setup": setup_name, "reason": str(exc)})
            print(f"SKIP {key}: {exc}")

    summary_path = _ROOT / "tmp" / "dao_reconcile" / "cross_draft_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps({"results": results, "skipped": skipped}, indent=2),
        encoding="utf-8",
    )
    _print_cross_draft_table(results)
    print(f"\nCross-draft summary: {summary_path}")
    return results, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Gaia<->DAO footprint reconciliation diagnostic.")
    parser.add_argument("--draft", type=int, default=None, help="Draft number (e.g. 424).")
    parser.add_argument("--all-drafts", action="store_true", help="Run on all archive drafts with MASTERSTAR.")
    parser.add_argument("--setup", type=str, default="", help="Explicit platesolve setup directory.")
    args = parser.parse_args()

    cfg = AppConfig()
    if args.all_drafts:
        run_all_drafts(cfg)
        return
    setup_dir = _resolve_setup_dir(cfg, args.draft, args.setup.strip() or None)
    draft_label = f"draft_{int(args.draft):06d}" if args.draft is not None else setup_dir.parent.parent.name
    run_diagnostic(setup_dir, cfg, draft_label=draft_label)


if __name__ == "__main__":
    main()
