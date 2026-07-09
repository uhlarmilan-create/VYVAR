#!/usr/bin/env python3
"""Read-only Gaia<->DAO reconciliation diagnostic for any draft setup directory.

Usage:
  python scripts/dao_reconcile_diag.py --draft 424
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

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import AppConfig  # noqa: E402
from dao_reconcile import compute_gaia_dao_reconcile  # noqa: E402
from gaia_catalog_id import read_vyvar_csv  # noqa: E402
from masterstar_context import header_core_fwhm_px, load_masterstar_context  # noqa: E402


def _resolve_setup_dir(cfg: AppConfig, draft: int | None, setup: str | None) -> Path:
    if setup:
        p = Path(setup).resolve()
        if not p.is_dir():
            raise SystemExit(f"Setup directory not found: {p}")
        return p
    if draft is None:
        raise SystemExit("Provide --draft N or --setup PATH")
    p = (Path(cfg.archive_root) / "Drafts" / f"draft_{int(draft):06d}" / "platesolve").resolve()
    if not p.is_dir():
        raise SystemExit(f"Platesolve dir not found: {p}")
    subs = sorted(d for d in p.iterdir() if d.is_dir())
    if len(subs) == 1:
        return subs[0]
    if subs:
        # Prefer bundle with MASTERSTAR.fits
        for d in subs:
            if (d / "MASTERSTAR.fits").is_file():
                return d
        return subs[0]
    raise SystemExit(f"No setup subdir under {p}")


def _load_inputs(setup_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, float, float | None, WCS | None, Path]:
    cone_path = setup_dir / "field_catalog_cone.csv"
    ms_path = setup_dir / "masterstars_full_match.csv"
    fits_path = setup_dir / "MASTERSTAR.fits"
    for req, label in ((cone_path, "field_catalog_cone.csv"), (ms_path, "masterstars_full_match.csv")):
        if not req.is_file():
            raise SystemExit(f"Missing {label} in {setup_dir}")

    cone = read_vyvar_csv(cone_path, low_memory=False, dtype={"catalog_id": str})
    det = read_vyvar_csv(ms_path, low_memory=False, dtype={"catalog_id": str})

    fwhm_px: float | None = None
    plate_scale: float | None = None
    wcs: WCS | None = None

    if fits_path.is_file():
        ctx = load_masterstar_context(fits_path)
        fwhm_px = ctx.vy_fwhm_gauss_px or ctx.vy_fwhm_px
        plate_scale = ctx.pixel_scale_arcsec
        with fits.open(fits_path, memmap=False) as hdul:
            wcs = WCS(hdul[0].header)
            if fwhm_px is None:
                fwhm_px = header_core_fwhm_px(hdul[0].header)

    meta_path = setup_dir / "photometry" / "pipeline_meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if fwhm_px is None:
            raw = meta.get("dao_fwhm_px")
            if raw is not None:
                fwhm_px = float(raw)

    if fwhm_px is None or not np.isfinite(fwhm_px) or fwhm_px <= 0:
        fwhm_px = 3.5

    return cone, det, float(fwhm_px), plate_scale, wcs, fits_path


def _save_g_histograms(labeled: pd.DataFrame, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    buckets = [
        ("matched", "Matched (DAO hit)"),
        ("below_limit", "Below limit (G > G_lim)"),
        ("blended", "Blended (<=1.5xFWHM of matched)"),
        ("genuinely_missed", "Genuinely missed"),
    ]
    for key, title in buckets:
        sub = labeled.loc[labeled["_bucket"] == key]
        mags = pd.to_numeric(sub.get("_mag"), errors="coerce").dropna()
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


def _report_to_jsonable(report: dict) -> dict:
    out = {k: v for k, v in report.items() if k != "labeled_cone"}
    return out


def _print_report(report: dict) -> None:
    gs = report["g_lim_stats"]
    print("=== Gaia<->DAO reconciliation ===")
    print(f"G_lim (p{gs['percentile']:.0f} matched G): {report['g_lim_est']}")
    print(f"  matched G distribution: p50={gs['p50']} p90={gs['p90']} p95={gs['p95']} max={gs['max']}")
    print(f"FWHM={report['fwhm_px']} px | blend radius={report['blend_radius_px']} px", end="")
    if report.get("blend_radius_arcsec") is not None:
        print(f" ({report['blend_radius_arcsec']:.3f}\")", end="")
    print()
    print(f"Cone rows: {report['catalog_rows']}")
    print(f"  matched:        {report['n_gaia_matched']}")
    print(f"  below-limit:    {report['n_gaia_below_limit']}")
    print(f"  blended:        {report['n_gaia_blended']}")
    print(f"  genuinely-missed:{report['n_gaia_missed']}")
    print(
        f"Corrected completeness: {report['gaia_dao_completeness_pct']}% "
        f"= matched / (matched + genuinely_missed)"
    )
    print(f"Raw completeness (legacy): {report['gaia_dao_completeness_raw_pct']}%")
    ud = report["unmatched_dao"]
    print(f"Unmatched DAO detections: {ud['n_dao_unmatched']}")
    print(f"  artifact candidates: {ud['n_artifact_candidates']}")
    print(f"  unexplained: {ud['n_unexplained']}")
    col = ud["collinearity"]
    print(
        f"  collinearity probe: n_collinear={col['n_collinear']} "
        f"consistent_with_line={col['consistent_with_line']}"
    )
    if ud["peak_dao"].get("n"):
        pk = ud["peak_dao"]
        print(f"  peak_dao: p50={pk['p50']} p90={pk['p90']} max={pk['max']}")
    if ud["flux"].get("n"):
        fl = ud["flux"]
        print(f"  flux: p50={fl['p50']} p90={fl['p90']} max={fl['max']}")


def run_diagnostic(
    setup_dir: Path,
    *,
    g_lim_percentile: float = 95.0,
    draft_label: str | None = None,
) -> dict:
    cone, det, fwhm_px, plate_scale, wcs, _fits = _load_inputs(setup_dir)
    report = compute_gaia_dao_reconcile(
        cone,
        det,
        fwhm_px=fwhm_px,
        plate_scale_arcsec=plate_scale,
        wcs=wcs,
        g_lim_percentile=g_lim_percentile,
    )
    label = draft_label or setup_dir.name
    out_dir = _ROOT / "tmp" / "dao_reconcile" / label
    hist_paths = _save_g_histograms(report["labeled_cone"], out_dir)
    payload = _report_to_jsonable(report)
    payload["setup_dir"] = str(setup_dir)
    payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["histogram_pngs"] = hist_paths
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "report.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["report_json"] = str(json_path)
    _print_report(report)
    print(f"Report JSON: {json_path}")
    for hp in hist_paths:
        print(f"Histogram: {hp}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Gaia<->DAO field accounting diagnostic (read-only).")
    parser.add_argument("--draft", type=int, default=None, help="Draft number (e.g. 424).")
    parser.add_argument("--setup", type=str, default="", help="Explicit platesolve setup directory.")
    parser.add_argument("--g-lim-pct", type=float, default=95.0, help="G_lim percentile (default 95).")
    args = parser.parse_args()

    cfg = AppConfig()
    setup_dir = _resolve_setup_dir(cfg, args.draft, args.setup.strip() or None)
    draft_label = f"draft_{int(args.draft):06d}" if args.draft is not None else setup_dir.name
    run_diagnostic(setup_dir, g_lim_percentile=float(args.g_lim_pct), draft_label=draft_label)


if __name__ == "__main__":
    main()
