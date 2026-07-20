#!/usr/bin/env python3
"""
TODO-LC-TREND diagnostic: rising/falling trends in ROT light curves (read-only).

Usage:
    python scripts/lc_trend_diagnostic.py --draft 342
    python scripts/lc_trend_diagnostic.py --draft draft_000342 --setup NoFilter_60_2
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr

MIN_NORMAL_FRAMES = 20
AIRMASS_R_THRESH = 0.4
CT_DELTA_MMAG_THRESH = 5.0
SLOPE_CT_CHANGE_MMAG_HR = 1.0
TREND_ALG3_MMAG_HR = 2.0
TREND_ASTRO_MMAG_HR = 5.0


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_draft_photometry_dir(
    draft: str | int,
    setup: str = "NoFilter_60_2",
    archive_root: Path | None = None,
) -> Path:
    """Mirror photometry_core draft layout: Archive/Drafts/draft_XXXXXX/platesolve/<setup>/photometry."""
    root = archive_root or (_repo_root() / "Archive")
    m = re.search(r"(\d+)", str(draft))
    if not m:
        raise ValueError(f"Cannot parse draft id from: {draft!r}")
    did = int(m.group(1))
    return root / "Drafts" / f"draft_{did:06d}" / "platesolve" / setup / "photometry"


def _parse_draft_id(draft: str | int) -> int:
    m = re.search(r"(\d+)", str(draft))
    if not m:
        raise ValueError(f"Cannot parse draft id from: {draft!r}")
    return int(m.group(1))


def load_rot_catalog_ids(phot_dir: Path) -> pd.DataFrame:
    """ROT targets from photometry_summary.csv (vsx_type contains ROT)."""
    summary_path = phot_dir / "photometry_summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing {summary_path}")
    summ = pd.read_csv(summary_path, dtype={"catalog_id": str})
    if "vsx_type" not in summ.columns:
        raise ValueError(f"No vsx_type column in {summary_path}")
    vt = summ["vsx_type"].fillna("").astype(str).str.upper()
    rot = summ[vt.str.contains("ROT", na=False)].copy()
    return rot


def _normal_mask(flags: pd.Series) -> np.ndarray:
    return flags.fillna("").astype(str).str.strip().str.lower().eq("normal").to_numpy()


def _slope_mmag_per_hour(time_days: np.ndarray, mag: np.ndarray) -> float:
    """Linear slope in mmag/hour (time in BJD days)."""
    ok = np.isfinite(time_days) & np.isfinite(mag)
    if int(ok.sum()) < 2:
        return float("nan")
    lr = linregress(time_days[ok], mag[ok])
    # linregress slope is mag per BJD day -> mmag per hour
    return float(lr.slope * 1000.0 / 24.0)


def _airmass_residual_r(time_days: np.ndarray, mag: np.ndarray, airmass: np.ndarray) -> float:
    ok = np.isfinite(time_days) & np.isfinite(mag) & np.isfinite(airmass)
    if int(ok.sum()) < 3:
        return float("nan")
    t = time_days[ok]
    m = mag[ok]
    am = airmass[ok]
    lr = linregress(t, m)
    resid = m - (lr.slope * t + lr.intercept)
    if float(np.std(resid)) < 1e-12 or float(np.std(am)) < 1e-12:
        return float("nan")
    r, _ = pearsonr(resid, am)
    return float(r)


def _primary_suspect(
    *,
    airmass_r: float,
    ct_ok: bool,
    ct_delta_mmag: float,
    slope_mmag_hr: float,
) -> str:
    if np.isfinite(airmass_r) and abs(airmass_r) > AIRMASS_R_THRESH:
        return "AIRMASS"
    if ct_ok and abs(ct_delta_mmag) > CT_DELTA_MMAG_THRESH:
        return "CT"
    if (
        np.isfinite(slope_mmag_hr)
        and abs(slope_mmag_hr) > TREND_ASTRO_MMAG_HR
        and (not np.isfinite(airmass_r) or abs(airmass_r) <= AIRMASS_R_THRESH)
    ):
        return "ASTROPHYSICAL"
    if np.isfinite(slope_mmag_hr) and abs(slope_mmag_hr) > TREND_ALG3_MMAG_HR:
        return "ALG3_COMP"
    return "LOW_TREND"


def diagnose_rot_target(
    catalog_id: str,
    lc_path: Path,
    summary_row: pd.Series | None,
) -> dict[str, object] | None:
    if not lc_path.is_file():
        return None
    lc = pd.read_csv(lc_path)
    if "flag" not in lc.columns:
        return None

    mask = _normal_mask(lc["flag"])
    n_normal = int(mask.sum())
    if n_normal < MIN_NORMAL_FRAMES:
        return None

    time_col = "bjd" if "bjd" in lc.columns else "jd"
    if time_col not in lc.columns:
        return None

    t = lc.loc[mask, time_col].to_numpy(dtype=float)
    mag_pre = lc.loc[mask, "mag_calib"].to_numpy(dtype=float) if "mag_calib" in lc.columns else np.full(n_normal, np.nan)
    mag_post = (
        lc.loc[mask, "mag_calib_ct"].to_numpy(dtype=float)
        if "mag_calib_ct" in lc.columns
        else mag_pre.copy()
    )
    am = lc.loc[mask, "airmass"].to_numpy(dtype=float) if "airmass" in lc.columns else np.full(n_normal, np.nan)

    slope_post = _slope_mmag_per_hour(t, mag_post)
    airmass_r = _airmass_residual_r(t, mag_post, am)

    ct_ok_val = bool(lc["ct_ok"].iloc[0]) if "ct_ok" in lc.columns else False
    if ct_ok_val and "ct_correction" in lc.columns:
        ct_corr = pd.to_numeric(lc["ct_correction"], errors="coerce").dropna()
        ct_delta_mmag = float(ct_corr.median() * 1000.0) if not ct_corr.empty else 0.0
    else:
        ct_delta_mmag = 0.0

    if "mag_calib_raw" in lc.columns:
        raw = pd.to_numeric(lc.loc[mask, "mag_calib_raw"], errors="coerce")
        cal = pd.to_numeric(lc.loc[mask, "mag_calib"], errors="coerce")
        airmass_detrend_mmag = float((cal.median() - raw.median()) * 1000.0)
    else:
        airmass_detrend_mmag = float("nan")

    ct_c1 = float("nan")
    if "ct_c1" in lc.columns:
        c1v = pd.to_numeric(lc.loc[mask, "ct_c1"], errors="coerce")
        if c1v.notna().any():
            ct_c1 = float(c1v.dropna().median())

    am_slope_summary = float("nan")
    if summary_row is not None and "am_slope" in summary_row.index:
        am_slope_summary = pd.to_numeric(summary_row.get("am_slope"), errors="coerce")
        am_slope_summary = float(am_slope_summary) if pd.notna(am_slope_summary) else float("nan")

    ac_correction_med = float("nan")
    if "ac_correction" in lc.columns:
        acv = pd.to_numeric(lc.loc[mask, "ac_correction"], errors="coerce")
        if acv.notna().any():
            ac_correction_med = float(acv.dropna().median())

    suspect = _primary_suspect(
        airmass_r=airmass_r,
        ct_ok=ct_ok_val,
        ct_delta_mmag=ct_delta_mmag,
        slope_mmag_hr=slope_post,
    )

    return {
        "catalog_id": catalog_id,
        "n_frames": n_normal,
        "ct_ok": ct_ok_val,
        "slope_mmag_hr": slope_post,
        "airmass_r": airmass_r,
        "ct_delta_mmag": ct_delta_mmag if ct_ok_val else float("nan"),
        "airmass_detrend_mmag": airmass_detrend_mmag,
        "primary_suspect": suspect,
        "am_slope_summary": am_slope_summary,
        "ct_c1": ct_c1,
        "ac_correction_med": ac_correction_med,
        "vsx_name": (
            str(summary_row.get("vsx_name", "")) if summary_row is not None else ""
        ),
    }


def _markdown_table(rows: list[dict[str, object]]) -> str:
    headers = [
        "catalog_id",
        "n_frames",
        "ct_ok",
        "slope_mmag_hr",
        "airmass_r",
        "airmass_detrend_mmag",
        "ct_delta_mmag",
        "primary_suspect",
    ]

    def _fmt(v: object) -> str:
        if isinstance(v, bool):
            return str(v)
        if isinstance(v, float):
            if not np.isfinite(v):
                return "-"
            if abs(v) >= 100:
                return f"{v:.2f}"
            return f"{v:.3f}"
        return str(v)

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(h, "")) for h in headers) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ROT LC trend diagnostic (TODO-LC-TREND)")
    parser.add_argument("--draft", default="342", help="Draft number or draft_000342")
    parser.add_argument("--setup", default="NoFilter_60_2", help="Platesolve setup folder name")
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=None,
        help="Override Archive root (default: <repo>/Archive)",
    )
    args = parser.parse_args(argv)

    phot_dir = resolve_draft_photometry_dir(
        args.draft, setup=args.setup, archive_root=args.archive_root
    )
    lc_dir = phot_dir / "lightcurves"
    if not lc_dir.is_dir():
        print(f"ERROR: lightcurves dir not found: {lc_dir}", file=sys.stderr)
        return 1

    rot_df = load_rot_catalog_ids(phot_dir)
    summary_by_cid = rot_df.set_index("catalog_id", drop=False)

    print(f"# TODO-LC-TREND diagnostic - draft_{_parse_draft_id(args.draft):06d}")
    print(f"Photometry: `{phot_dir}`")
    print(f"ROT candidates (vsx_type contains ROT): **{len(rot_df)}**")
    print(f"Minimum unflagged (`normal`) frames: **{MIN_NORMAL_FRAMES}**")
    print()

    results: list[dict[str, object]] = []
    skipped = 0
    for _, row in rot_df.iterrows():
        cid = str(row["catalog_id"])
        lc_path = lc_dir / f"lightcurve_{cid}.csv"
        summ_row = summary_by_cid.loc[cid] if cid in summary_by_cid.index else None
        diag = diagnose_rot_target(cid, lc_path, summ_row)
        if diag is None:
            skipped += 1
            continue
        results.append(diag)

    results.sort(
        key=lambda r: (
            0
            if r["primary_suspect"] == "AIRMASS"
            else 1
            if r["primary_suspect"] == "CT"
            else 2
            if r["primary_suspect"] == "ALG3_COMP"
            else 3
            if r["primary_suspect"] == "ASTROPHYSICAL"
            else 9,
            -abs(float(r["slope_mmag_hr"]) if np.isfinite(float(r["slope_mmag_hr"])) else 0.0),
        )
    )

    print(f"Analyzed: **{len(results)}** | Skipped (<{MIN_NORMAL_FRAMES} normal or missing LC): **{skipped}**")
    print()

    if not results:
        print("No ROT targets met analysis criteria.")
        return 0

    print(_markdown_table(results))
    print()

    # Suspect counts
    counts: dict[str, int] = {}
    for r in results:
        k = str(r["primary_suspect"])
        counts[k] = counts.get(k, 0) + 1
    print("## Primary suspect counts")
    for k in sorted(counts, key=lambda x: (-counts[x], x)):
        print(f"- **{k}**: {counts[k]}")
    print()

    print("## Per-target airmass / color-term metadata")
    print("(LC has no `ac_slope`; `am_slope` from photometry_summary; `ct_c1` / `ac_correction` from LC)")
    print()
    for r in results:
        cid = r["catalog_id"]
        name = r.get("vsx_name") or ""
        ams = r["am_slope_summary"]
        c1 = r["ct_c1"]
        ac = r["ac_correction_med"]
        print(
            f"- `{cid}`"
            + (f" ({name})" if name else "")
            + f" - am_slope(summary)={ams if np.isfinite(float(ams)) else '-'}, "
            f"ct_c1(median)={c1 if np.isfinite(float(c1)) else '-'}, "
            f"ac_correction(median)={ac if np.isfinite(float(ac)) else '-'}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
