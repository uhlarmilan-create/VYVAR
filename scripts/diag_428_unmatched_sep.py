#!/usr/bin/env python3
"""F-428 A3 v2: separation diagnostic for unmatched DET_* rows (read-only).

Uses masterstars ra_deg/dec_deg (never pixel x/y) and field-bounded Gaia DB nearest-neighbor
with cos(dec)-scaled RA separation. Self-check gates before reporting.
"""
from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import AppConfig
from gaia_catalog_id import normalize_gaia_source_id
from scripts.repair_catalog_ids import _pick_gaia_table, _sep_arcsec

# Phase-0 census from draft_428 infolog (2026-07-15 UI run)
PHASE0_CENSUS = {
    "linear": 115,
    "noisy1": 11,
    "noisy2": 11,
    "noisy3": 41,
    "saturated_active": 2,
    "no_catalog_id": 1,
    "out_of_frame": 47,
    "excluded_no_dao_match": 17,
}


def _sep_arcsec_vec(ra: float, dec: float, ra_arr: np.ndarray, dec_arr: np.ndarray) -> np.ndarray:
    """Vectorized small-angle separation (arcsec) with cos(dec) on RA."""
    dra = (ra_arr - float(ra)) * math.cos(math.radians(float(dec)))
    ddec = dec_arr - float(dec)
    return np.sqrt(dra * dra + ddec * ddec) * 3600.0


def _resolve_mag(row: pd.Series) -> tuple[float, str]:
    for col in ("mag", "phot_g_mean_mag", "vsx_mag_max", "catalog_mag"):
        if col in row.index:
            val = pd.to_numeric(row.get(col), errors="coerce")
            if pd.notna(val) and math.isfinite(float(val)):
                return float(val), col
    return float("nan"), "(none)"


def _load_gaia_field(con: sqlite3.Connection, table: str, ms: pd.DataFrame, pad_deg: float) -> pd.DataFrame:
    ra = pd.to_numeric(ms["ra_deg"], errors="coerce")
    dec = pd.to_numeric(ms["dec_deg"], errors="coerce")
    ra_min, ra_max = float(ra.min()), float(ra.max())
    dec_min, dec_max = float(dec.min()), float(dec.max())
    q = f"""
        SELECT source_id, ra, dec
        FROM {table}
        WHERE ra BETWEEN ? AND ?
          AND dec BETWEEN ? AND ?
    """
    params = (ra_min - pad_deg, ra_max + pad_deg, dec_min - pad_deg, dec_max + pad_deg)
    gdf = pd.read_sql_query(q, con, params=params)
    return gdf, (ra_min, ra_max, dec_min, dec_max), params


def _nearest_gaia_brute(ra: float, dec: float, gra: np.ndarray, gdec: np.ndarray) -> tuple[float, int | None]:
    if gra.size == 0 or not (math.isfinite(ra) and math.isfinite(dec)):
        return float("nan"), None
    sep = _sep_arcsec_vec(ra, dec, gra, gdec)
    idx = int(np.argmin(sep))
    return float(sep[idx]), idx


def _classify_excluded_vt_row(
    row: pd.Series,
    *,
    frame_w: int,
    frame_h: int,
    edge_margin: int,
    active_ids: set[str],
    ms_by_cid: dict[str, pd.Series],
) -> str:
    cid = str(row.get("_cid_norm", "") or "")
    x = pd.to_numeric(row.get("x"), errors="coerce")
    y = pd.to_numeric(row.get("y"), errors="coerce")
    if pd.notna(x) and pd.notna(y):
        in_frame = (
            float(x) >= edge_margin
            and float(x) <= frame_w - edge_margin
            and float(y) >= edge_margin
            and float(y) <= frame_h - edge_margin
        )
        if not in_frame:
            return "out_of_frame"
    if not cid:
        return "no_catalog_id"
    if cid in ms_by_cid:
        ms_row = ms_by_cid[cid]
        if str(ms_row.get("zone_flag", ms_row.get("zone", ""))).strip().lower() == "saturated":
            return "saturated"
        if bool(pd.to_numeric(ms_row.get("skip_photometry"), errors="coerce")) if "skip_photometry" in ms_row.index else False:
            return "saturated"
    if cid not in active_ids:
        return "no_dao_gaia_match"
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser(description="F-428 A3 v2 unmatched DET_* / excluded-target diagnostic")
    ap.add_argument(
        "--masterstars",
        type=Path,
        default=_ROOT / "Archive/Drafts/draft_000428/platesolve/NoFilter_60_2/masterstars_full_match.csv",
    )
    ap.add_argument(
        "--variable-targets",
        type=Path,
        default=_ROOT / "Archive/Drafts/draft_000428/platesolve/NoFilter_60_2/variable_targets.csv",
    )
    ap.add_argument(
        "--active-targets",
        type=Path,
        default=_ROOT / "Archive/Drafts/draft_000428/platesolve/NoFilter_60_2/photometry/active_targets.csv",
    )
    ap.add_argument("--gaia-db", type=Path, default=None)
    ap.add_argument("--fwhm-px", type=float, default=2.3976)
    ap.add_argument("--plate-scale-arcsec-px", type=float, default=2.6)
    ap.add_argument("--frame-w", type=int, default=2082)
    ap.add_argument("--frame-h", type=int, default=1397)
    ap.add_argument("--edge-margin", type=int, default=50)
    ap.add_argument("--self-match-max-arcsec", type=float, default=2.0)
    ap.add_argument("--median-self-check-arcsec", type=float, default=15.0)
    ap.add_argument("--out", type=Path, default=_ROOT / "tmp/f428_unmatched_sep_v2.txt")
    args = ap.parse_args()

    ms_path = Path(args.masterstars)
    vt_path = Path(args.variable_targets)
    at_path = Path(args.active_targets)
    for p, label in ((ms_path, "masterstars"), (vt_path, "variable_targets"), (at_path, "active_targets")):
        if not p.is_file():
            print(f"Missing {label}: {p}", file=sys.stderr)
            return 2

    cfg = AppConfig()
    gdb = Path(args.gaia_db or cfg.gaia_db_path or "")
    if not gdb.is_file():
        print(f"Missing Gaia DB: {gdb}", file=sys.stderr)
        return 2

    ms = pd.read_csv(ms_path, low_memory=False, dtype={"catalog_id": str, "name": str})
    vt = pd.read_csv(vt_path, low_memory=False, dtype={"catalog_id": str})
    at = pd.read_csv(at_path, low_memory=False, dtype={"catalog_id": str})

    ms["cid_norm"] = ms["catalog_id"].map(normalize_gaia_source_id)
    vt["_cid_norm"] = vt["catalog_id"].map(normalize_gaia_source_id)
    at["_cid_norm"] = at["catalog_id"].map(normalize_gaia_source_id)

    # Unmatched DET_*: name matches DET_#### AND empty catalog_id (pipeline placeholder rows)
    det_mask = ms["name"].astype(str).str.match(r"^DET_\d+$", na=False) & (
        ms["catalog_id"].fillna("").astype(str).str.strip().isin(("", "nan", "None"))
    )
    unmatched = ms.loc[det_mask].copy()
    matched_ms = ms.loc[~det_mask].copy()

    con = sqlite3.connect(str(gdb))
    table = _pick_gaia_table(con)
    total_gaia = int(con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    gaia_field, (ra_min, ra_max, dec_min, dec_max), bbox_params = _load_gaia_field(con, table, ms, pad_deg=0.02)
    con.close()

    gra = gaia_field["ra"].to_numpy(dtype=float)
    gdec = gaia_field["dec"].to_numpy(dtype=float)

    lines: list[str] = []
    lines.append("# F-428 A3 diagnostic v2")
    lines.append(f"gaia_db: {gdb.resolve()}")
    lines.append(f"gaia_table: {table} (total rows in DB: {total_gaia})")
    lines.append(
        f"gaia_field_query_bbox: RA [{bbox_params[0]:.5f}, {bbox_params[1]:.5f}] "
        f"DEC [{bbox_params[2]:.5f}, {bbox_params[3]:.5f}] (pad=0.02 deg)"
    )
    lines.append(f"gaia_rows_in_bbox: {len(gaia_field)}")
    lines.append(f"masterstars: {ms_path}")
    lines.append(f"coord_columns_used: masterstars/variable_targets -> ra_deg, dec_deg (NOT x/y pixels)")
    lines.append(
        f"unmatched_DET_rows: {len(unmatched)} / {len(ms)} "
        f"(name~'^DET_\\d+$' AND empty catalog_id)"
    )

    fwhm_arcsec = float(args.fwhm_px) * float(args.plate_scale_arcsec_px)
    lines.append(f"FWHM={args.fwhm_px}px × {args.plate_scale_arcsec_px}\"/px = {fwhm_arcsec:.3f}\"")

    gaia_seps: list[float] = []
    for _, row in unmatched.iterrows():
        ra = float(row["ra_deg"])
        dec = float(row["dec_deg"])
        sep, _ = _nearest_gaia_brute(ra, dec, gra, gdec)
        if math.isfinite(sep):
            gaia_seps.append(sep)

    if gaia_seps:
        arr = np.asarray(gaia_seps, dtype=float)
        for p in (50, 75, 90, 95, 99):
            lines.append(f"nearest_gaia_sep_p{p}: {float(np.percentile(arr, p)):.3f}\"")
        lines.append(f"nearest_gaia_sep_max: {float(np.max(arr)):.3f}\"")
        for mult, label in ((1.0, "1x_fwhm"), (1.5, "1.5x_fwhm"), (2.0, "2x_fwhm")):
            thr = mult * fwhm_arcsec
            cnt = int((arr <= thr).sum())
            lines.append(f"within_{label}: {cnt}/{len(unmatched)}")
    else:
        lines.append("nearest_gaia_sep: no finite values")

    # Self-match: VT/MS rows with same catalog_id
    lines.append("")
    lines.append("# Self-match check (catalog_id present in masterstars)")
    ms_by_cid: dict[str, pd.Series] = {}
    for _, mrow in ms.iterrows():
        cid = str(mrow.get("cid_norm", "") or "")
        if cid:
            ms_by_cid[cid] = mrow

    self_violations: list[str] = []
    check_ids = set(vt["_cid_norm"].dropna()) | set(at["_cid_norm"].dropna())
    for cid in sorted(check_ids):
        if not cid or cid not in ms_by_cid:
            continue
        vt_rows = vt[vt["_cid_norm"] == cid]
        if vt_rows.empty:
            continue
        vrow = vt_rows.iloc[0]
        mrow = ms_by_cid[cid]
        ra_v = float(vrow["ra_deg"])
        dec_v = float(vrow["dec_deg"])
        ra_m = float(mrow["ra_deg"])
        dec_m = float(mrow["dec_deg"])
        sep = _sep_arcsec(ra_v, dec_v, ra_m, dec_m)
        if sep > float(args.self_match_max_arcsec):
            name = str(vrow.get("vsx_name") or vrow.get("name") or cid)
            self_violations.append(
                f"  VIOLATION {name} cid={cid} vt-ms sep={sep:.3f}\" "
                f"(vt ra/dec={ra_v:.6f},{dec_v:.6f}; ms ra/dec={ra_m:.6f},{dec_m:.6f})"
            )
    lines.append(f"self_match_checked: {len(check_ids)} catalog_ids with vt row + ms row")
    lines.append(f"self_match_violations: {len(self_violations)} (threshold {args.self_match_max_arcsec}\")")
    lines.extend(self_violations[:30])
    if len(self_violations) > 30:
        lines.append(f"  ... {len(self_violations) - 30} more violations")

    # Excluded population: vt catalog_id NOT IN active_targets (dedup by cid)
    active_ids = set(at["_cid_norm"].dropna().astype(str))
    ex = vt[~vt["_cid_norm"].astype(str).isin(active_ids)].drop_duplicates(subset=["_cid_norm"])
    lines.append("")
    lines.append("# Excluded VSX population (variable_targets catalog_id NOT IN active_targets, dedup)")
    lines.append(f"variable_targets_rows: {len(vt)} (unique catalog_id: {vt['_cid_norm'].nunique()})")
    lines.append(f"active_targets_rows: {len(at)} (unique catalog_id: {at['_cid_norm'].nunique()})")
    lines.append(f"excluded_set_size: {len(ex)}")
    lines.append(
        "phase0_census_reference: "
        f"linear={PHASE0_CENSUS['linear']} noisy1={PHASE0_CENSUS['noisy1']} "
        f"noisy2={PHASE0_CENSUS['noisy2']} noisy3={PHASE0_CENSUS['noisy3']} "
        f"saturated={PHASE0_CENSUS['saturated_active']} no_catalog_id={PHASE0_CENSUS['no_catalog_id']} "
        f"out_of_frame={PHASE0_CENSUS['out_of_frame']} "
        f"excluded_no_dao_match={PHASE0_CENSUS['excluded_no_dao_match']}"
    )
    census_excluded = (
        PHASE0_CENSUS["out_of_frame"]
        + PHASE0_CENSUS["excluded_no_dao_match"]
        + PHASE0_CENSUS["no_catalog_id"]
        + PHASE0_CENSUS["saturated_active"]
    )
    lines.append(
        f"phase0_excluded_sum(out_of_frame+no_dao+no_catalog_id+saturated): {census_excluded} "
        f"(active {PHASE0_CENSUS['linear'] + PHASE0_CENSUS['noisy1'] + PHASE0_CENSUS['noisy2'] + PHASE0_CENSUS['noisy3'] + PHASE0_CENSUS['saturated_active']} "
        f"+ excluded = {PHASE0_CENSUS['linear'] + PHASE0_CENSUS['noisy1'] + PHASE0_CENSUS['noisy2'] + PHASE0_CENSUS['noisy3'] + PHASE0_CENSUS['saturated_active'] + census_excluded} vs vt {len(vt)})"
    )

    reason_counts: dict[str, int] = {}
    for _, row in ex.iterrows():
        reason = _classify_excluded_vt_row(
            row,
            frame_w=int(args.frame_w),
            frame_h=int(args.frame_h),
            edge_margin=int(args.edge_margin),
            active_ids=active_ids,
            ms_by_cid=ms_by_cid,
        )
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    lines.append("excluded_class_breakdown (vt-not-in-active heuristic):")
    for k, v in sorted(reason_counts.items()):
        lines.append(f"  {k}: {v}")

    lines.append("")
    lines.append("# Excluded target detail (nearest Gaia + nearest masterstar match by cid when available)")
    for _, row in ex.iterrows():
        name = str(row.get("vsx_name") or row.get("name") or "")
        cid = str(row.get("_cid_norm", "") or "")
        mag, mag_col = _resolve_mag(row)
        ra = float(row["ra_deg"])
        dec = float(row["dec_deg"])
        gsep, _ = _nearest_gaia_brute(ra, dec, gra, gdec)
        if cid and cid in ms_by_cid:
            mrow = ms_by_cid[cid]
            dsep = _sep_arcsec(ra, dec, float(mrow["ra_deg"]), float(mrow["dec_deg"]))
            dname = str(mrow.get("name") or cid)
        else:
            # nearest matched masterstar by sky position (for context only)
            mra = pd.to_numeric(matched_ms["ra_deg"], errors="coerce").to_numpy(float)
            mdec = pd.to_numeric(matched_ms["dec_deg"], errors="coerce").to_numpy(float)
            if mra.size:
                sep_arr = _sep_arcsec_vec(ra, dec, mra, mdec)
                idx = int(np.argmin(sep_arr))
                dsep = float(sep_arr[idx])
                dname = str(matched_ms.iloc[idx].get("name") or matched_ms.iloc[idx].get("catalog_id") or "")
            else:
                dsep, dname = float("nan"), ""
        reason = _classify_excluded_vt_row(
            row,
            frame_w=int(args.frame_w),
            frame_h=int(args.frame_h),
            edge_margin=int(args.edge_margin),
            active_ids=active_ids,
            ms_by_cid=ms_by_cid,
        )
        lines.append(
            f"  {name} cid={cid} mag={mag:.3f} (col={mag_col}) reason={reason} "
            f"nearest_gaia={gsep:.3f}\" ms_match={dname} sep={dsep:.3f}\""
        )

    # Sanity gates
    fail_reasons: list[str] = []
    if gaia_seps:
        med = float(np.median(np.asarray(gaia_seps)))
        if med >= float(args.median_self_check_arcsec):
            fail_reasons.append(
                f"median nearest-Gaia sep {med:.3f}\" >= {args.median_self_check_arcsec}\" "
                f"(unmatched DET_* positions are systematically offset from catalog; not a query bug)"
            )
    if self_violations:
        fail_reasons.append(f"{len(self_violations)} self-match violations (threshold {args.self_match_max_arcsec}\")")

    ex_sum = sum(reason_counts.get(k, 0) for k in ("out_of_frame", "no_dao_gaia_match", "no_catalog_id", "saturated"))
    if ex_sum != len(ex):
        fail_reasons.append(
            f"excluded class sum {ex_sum} != excluded_set_size {len(ex)} (other/unclassified present)"
        )
    if len(ex) != census_excluded and abs(len(ex) - census_excluded) > 5:
        fail_reasons.append(
            f"excluded_set_size {len(ex)} differs from phase0 census excluded sum {census_excluded} by >5"
        )

    lines.append("")
    if fail_reasons:
        lines.append("DIAG SELF-CHECK FAIL")
        for fr in fail_reasons:
            lines.append(f"  - {fr}")
    else:
        lines.append("DIAG SELF-CHECK PASS")

    text = "\n".join(lines) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(text)
    return 1 if fail_reasons else 0


if __name__ == "__main__":
    raise SystemExit(main())
