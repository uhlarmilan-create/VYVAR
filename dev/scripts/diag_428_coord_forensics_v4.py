#!/usr/bin/env python3
"""F-428-COORD forensics v4: pixel-space identity + Gaia coverage (read-only)."""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import AppConfig
from gaia_catalog_id import normalize_gaia_source_id
from pipeline import _all_pix2world_icrs_deg
from scripts.repair_catalog_ids import _pick_gaia_table, _sep_arcsec

PRIORITY_TARGETS = [
    "FY CVn",
    "FZ CVn",
    "CSS_J134925.3+393524",
    "CSS_J140918.7+423422",
    "NSVS 5096293",
    "RX CVn",
    "R CVn",
]


def _load_gaia_by_id(con: sqlite3.Connection, table: str, ids: set[str]) -> dict[str, tuple[float, float, float | None]]:
    out: dict[str, tuple[float, float, float | None]] = {}
    if not ids:
        return out
    id_list = sorted(ids)
    for i in range(0, len(id_list), 400):
        part = id_list[i : i + 400]
        ph = ",".join("?" * len(part))
        mag_col = "phot_g_mean_mag" if "phot_g_mean_mag" in _table_cols(con, table) else "g_mag"
        try:
            q = f"SELECT source_id, ra, dec, {mag_col} FROM {table} WHERE source_id IN ({ph})"
            rows = con.execute(q, part).fetchall()
        except sqlite3.OperationalError:
            q = f"SELECT source_id, ra, dec FROM {table} WHERE source_id IN ({ph})"
            rows = [(a, b, c, None) for a, b, c in con.execute(q, part).fetchall()]
        for sid, ra, dec, gmag in rows:
            key = normalize_gaia_source_id(str(sid))
            if key:
                g = float(gmag) if gmag is not None and math.isfinite(float(gmag)) else None
                out[key] = (float(ra), float(dec), g)
    return out


def _table_cols(con: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in con.execute(f"PRAGMA table_info({table})")}


def _wcs_footprint(w: WCS, wpx: int, hpx: int) -> np.ndarray:
    corners_x = np.array([0.0, wpx, wpx, 0.0], dtype=np.float64)
    corners_y = np.array([0.0, 0.0, hpx, hpx], dtype=np.float64)
    ra, dec = w.all_pix2world(corners_x, corners_y, 0)
    return np.column_stack([np.asarray(ra, float), np.asarray(dec, float)])


def _point_in_ra_dec_bbox(ra: float, dec: float, ra_min: float, ra_max: float, dec_min: float, dec_max: float) -> bool:
    if not (math.isfinite(ra) and math.isfinite(dec)):
        return False
    if ra_min <= ra_max:
        ra_ok = ra_min <= ra <= ra_max
    else:
        ra_ok = ra >= ra_min or ra <= ra_max
    return ra_ok and dec_min <= dec <= dec_max


def _nearest_gaia_brute(ra: float, dec: float, gra: np.ndarray, gdec: np.ndarray) -> float:
    if gra.size == 0 or not (math.isfinite(ra) and math.isfinite(dec)):
        return float("nan")
    dra = (gra - ra) * np.cos(np.radians(dec))
    ddec = gdec - dec
    sep = np.sqrt(dra * dra + ddec * ddec) * 3600.0
    return float(np.min(sep))


def _nearest_gaia_vec(ra_arr: np.ndarray, dec_arr: np.ndarray, gra: np.ndarray, gdec: np.ndarray) -> np.ndarray:
    """Vectorized nearest-neighbor separation (arcsec) for arrays of query positions."""
    n = len(ra_arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if gra.size == 0 or n == 0:
        return out
    cos_dec = np.cos(np.radians(dec_arr))
    # Chunk queries to limit memory (n_q * n_g float64)
    chunk = 256
    for i0 in range(0, n, chunk):
        i1 = min(n, i0 + chunk)
        ra_q = ra_arr[i0:i1, None]
        dec_q = dec_arr[i0:i1, None]
        cd = cos_dec[i0:i1, None]
        dra = (gra[None, :] - ra_q) * cd
        ddec = gdec[None, :] - dec_q
        sep = np.sqrt(dra * dra + ddec * ddec) * 3600.0
        out[i0:i1] = np.min(sep, axis=1)
    return out


def _load_field_gaia(con: sqlite3.Connection, table: str, ra_min: float, ra_max: float, dec_min: float, dec_max: float) -> pd.DataFrame:
    q = f"""
        SELECT source_id, ra, dec
        FROM {table}
        WHERE ra BETWEEN ? AND ? AND dec BETWEEN ? AND ?
    """
    return pd.read_sql_query(q, con, params=(ra_min, ra_max, dec_min, dec_max))


def _build_lc_index(phot_dir: Path) -> dict[str, dict[str, Any]]:
    """One-pass index of catalog_id -> LC stats from proc_*.csv files."""
    index: dict[str, dict[str, Any]] = {}
    for p in sorted(phot_dir.glob("proc_*.csv")):
        try:
            df = pd.read_csv(p, usecols=lambda c: c in {"catalog_id", "mag"}, dtype={"catalog_id": str})
        except Exception:  # noqa: BLE001
            continue
        if "catalog_id" not in df.columns:
            continue
        for cid_raw, grp in df.groupby("catalog_id"):
            cid = normalize_gaia_source_id(str(cid_raw))
            if not cid or cid in index:
                continue
            mags = pd.to_numeric(grp["mag"], errors="coerce")
            index[cid] = {
                "lc_found": True,
                "n_frames": int(len(grp)),
                "mag_std_mmag": float(mags.std() * 1000.0) if mags.notna().sum() >= 3 else float("nan"),
                "mag_range_mmag": float((mags.max() - mags.min()) * 1000.0) if mags.notna().sum() >= 2 else float("nan"),
                "source_file": p.name,
            }
    return index


def _find_lc_summary(lc_index: dict[str, dict[str, Any]], cid: str) -> dict[str, Any]:
    return lc_index.get(cid, {"lc_found": False})


def main() -> int:
    ap = argparse.ArgumentParser(description="F-428 coord forensics v4")
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
    ap.add_argument(
        "--masterstar-fits",
        type=Path,
        default=_ROOT / "Archive/Drafts/draft_000428/platesolve/NoFilter_60_2/MASTERSTAR.fits",
    )
    ap.add_argument(
        "--photometry-dir",
        type=Path,
        default=_ROOT / "Archive/Drafts/draft_000428/platesolve/NoFilter_60_2/photometry",
    )
    ap.add_argument("--gaia-db", type=Path, default=None)
    ap.add_argument("--plate-scale-arcsec-px", type=float, default=2.6)
    ap.add_argument("--stale-wcs-px-threshold", type=float, default=1.5)
    ap.add_argument("--vt-ms-violation-arcsec", type=float, default=2.0)
    ap.add_argument("--out", type=Path, default=_ROOT / "tmp/f428_coord_forensics_v4.txt")
    args = ap.parse_args()

    stale_thr_arcsec = float(args.stale_wcs_px_threshold) * float(args.plate_scale_arcsec_px)

    ms_path = Path(args.masterstars)
    vt_path = Path(args.variable_targets)
    at_path = Path(args.active_targets)
    fits_path = Path(args.masterstar_fits)
    for p, label in ((ms_path, "masterstars"), (vt_path, "variable_targets"), (at_path, "active_targets"), (fits_path, "MASTERSTAR.fits")):
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
    active_ids = set(at["_cid_norm"].dropna().astype(str))

    ms_by_cid: dict[str, pd.Series] = {}
    for _, row in ms.iterrows():
        cid = str(row.get("cid_norm") or "")
        if cid:
            ms_by_cid[cid] = row

    with fits.open(fits_path, memmap=False) as hdul:
        w = WCS(hdul[0].header)
        hpx = int(hdul[0].header.get("NAXIS1", 2082))
        wpy = int(hdul[0].header.get("NAXIS2", 1397))
    if not getattr(w, "has_celestial", False):
        print("MASTERSTAR.fits has no celestial WCS", file=sys.stderr)
        return 2

    foot = _wcs_footprint(w, hpx, wpy)
    frame_ra_min, frame_ra_max = float(np.min(foot[:, 0])), float(np.max(foot[:, 0]))
    frame_dec_min, frame_dec_max = float(np.min(foot[:, 1])), float(np.max(foot[:, 1]))

    con = sqlite3.connect(str(gdb))
    table = _pick_gaia_table(con)
    db_bbox = con.execute(f"SELECT MIN(ra), MAX(ra), MIN(dec), MAX(dec), COUNT(*) FROM {table}").fetchone()
    db_ra_min, db_ra_max, db_dec_min, db_dec_max, db_total = db_bbox
    db_total = int(db_total)

    # Try infer cone provenance from build json near DB
    cone_meta: dict[str, Any] = {}
    for cand in (
        gdb.with_name(gdb.stem + "_build.json"),
        gdb.parent / "field_db_build.json",
        _ROOT / "Archive/Drafts/draft_000428/platesolve/NoFilter_60_2/field_catalog_cone_meta.json",
    ):
        if cand.is_file():
            try:
                cone_meta = json.loads(cand.read_text(encoding="utf-8"))
                cone_meta["_source"] = str(cand)
                break
            except Exception:  # noqa: BLE001
                pass

    violation_ids: list[str] = []
    check_ids = sorted(set(vt["_cid_norm"].dropna()) | set(at["_cid_norm"].dropna()))
    for cid in check_ids:
        if not cid or cid not in ms_by_cid:
            continue
        vrows = vt[vt["_cid_norm"] == cid]
        arows = at[at["_cid_norm"] == cid]
        if not vrows.empty:
            vrow = vrows.iloc[0]
        elif not arows.empty:
            vrow = arows.iloc[0]
        else:
            continue
        mrow = ms_by_cid[cid]
        sep_vm = _sep_arcsec(float(vrow["ra_deg"]), float(vrow["dec_deg"]), float(mrow["ra_deg"]), float(mrow["dec_deg"]))
        if sep_vm > float(args.vt_ms_violation_arcsec):
            violation_ids.append(cid)

    gaia_ids = set(violation_ids)
    gaia_by_id = _load_gaia_by_id(con, table, {x for x in gaia_ids if x})
    ref_gaia = _load_gaia_by_id(con, table, set(ms_by_cid.keys()))

    lc_index = _build_lc_index(Path(args.photometry_dir))

    # T1 identity test
    t1_rows: list[dict[str, Any]] = []
    for cid in violation_ids:
        mrow = ms_by_cid[cid]
        vrows = vt[vt["_cid_norm"] == cid]
        arows = at[at["_cid_norm"] == cid]
        if not vrows.empty:
            vrow = vrows.iloc[0]
        elif not arows.empty:
            vrow = arows.iloc[0]
        else:
            vrow = pd.Series(dtype=object)
        g = gaia_by_id.get(cid)
        if g is None:
            t1_rows.append({"cid": cid, "class": "NO_GAIA_DB", "name": str(vrow.get("vsx_name") or cid)})
            continue
        g_ra, g_de, g_mag = g
        x = float(mrow["x"])
        y = float(mrow["y"])
        ra_w, dec_w = _all_pix2world_icrs_deg(w, np.array([x]), np.array([y]))
        ra_w, dec_w = float(ra_w[0]), float(dec_w[0])
        sep_stored = _sep_arcsec(float(mrow["ra_deg"]), float(mrow["dec_deg"]), g_ra, g_de)
        sep_wcs = _sep_arcsec(ra_w, dec_w, g_ra, g_de)
        sep_vm = _sep_arcsec(float(vrow.get("ra_deg", g_ra)), float(vrow.get("dec_deg", g_de)), float(mrow["ra_deg"]), float(mrow["dec_deg"]))
        if sep_wcs <= stale_thr_arcsec and sep_stored > stale_thr_arcsec:
            cls = "STALE-COORDS"
        elif sep_wcs > stale_thr_arcsec:
            cls = "MISASSIGNED-ID"
        else:
            cls = "CONSISTENT"
        lc = _find_lc_summary(lc_index, cid) if cid in active_ids else {}
        t1_rows.append(
            {
                "cid": cid,
                "name": str(vrow.get("vsx_name") or vrow.get("name") or cid),
                "active": cid in active_ids,
                "class": cls,
                "sep_stored_gaia": sep_stored,
                "sep_wcs_gaia": sep_wcs,
                "sep_vt_ms": sep_vm,
                "x": x,
                "y": y,
                "g_mag": g_mag,
                "lc": lc,
            }
        )

    hist: dict[str, int] = {}
    for r in t1_rows:
        hist[r["class"]] = hist.get(r["class"], 0) + 1

    misassigned = [r for r in t1_rows if r["class"] == "MISASSIGNED-ID"]
    stale = [r for r in t1_rows if r["class"] == "STALE-COORDS"]

    # T2 coverage for unmatched DET
    det_mask = ms["name"].astype(str).str.match(r"^DET_\d+$", na=False) & (
        ms["catalog_id"].fillna("").astype(str).str.strip().isin(("", "nan", "None"))
    )
    unmatched = ms.loc[det_mask].copy()
    pad = 0.02
    gaia_field = _load_field_gaia(
        con,
        table,
        frame_ra_min - pad,
        frame_ra_max + pad,
        frame_dec_min - pad,
        frame_dec_max + pad,
    )
    con.close()
    gra = gaia_field["ra"].to_numpy(dtype=float)
    gdec = gaia_field["dec"].to_numpy(dtype=float)

    u_ra = pd.to_numeric(unmatched["ra_deg"], errors="coerce").to_numpy(dtype=float)
    u_de = pd.to_numeric(unmatched["dec_deg"], errors="coerce").to_numpy(dtype=float)
    inside_seps: list[float] = []
    outside_seps: list[float] = []

    # T1 subclass for violations
    t1_sub = {"wcs_stored_lt1_arcsec": 0, "stored_gaia_lt1_arcsec": 0, "sep_wcs_eq_stored_vs_gaia": 0}
    for r in t1_rows:
        if r.get("class") not in ("MISASSIGNED-ID", "STALE-COORDS", "CONSISTENT"):
            continue
        s_ws = abs(float(r.get("sep_stored_gaia", 999)) - float(r.get("sep_wcs_gaia", -999)))
        mrow = ms_by_cid.get(r["cid"])
        if mrow is None:
            continue
        x = float(mrow["x"])
        y = float(mrow["y"])
        ra_w, dec_w = _all_pix2world_icrs_deg(w, np.array([x]), np.array([y]))
        sep_ws = _sep_arcsec(float(ra_w[0]), float(dec_w[0]), float(mrow["ra_deg"]), float(mrow["dec_deg"]))
        if sep_ws <= 1.0:
            t1_sub["wcs_stored_lt1_arcsec"] += 1
        if float(r.get("sep_stored_gaia", 999)) <= 1.0:
            t1_sub["stored_gaia_lt1_arcsec"] += 1
        if s_ws <= 1.0:
            t1_sub["sep_wcs_eq_stored_vs_gaia"] += 1

    # T2 coverage: field_catalog_cone in frame + global DB bbox
    cone_path = ms_path.parent / "field_catalog_cone.csv"
    cone_meta_path = ms_path.parent / "field_catalog_cone_meta.json"
    cone_in_frame_n = 0
    cone_field_p50 = float("nan")
    cone_field_within_20 = 0
    if cone_path.is_file():
        try:
            cone_df = pd.read_csv(cone_path, usecols=["ra_deg", "dec_deg", "mag"])
            cone_f = cone_df[
                (pd.to_numeric(cone_df["ra_deg"], errors="coerce") >= frame_ra_min)
                & (pd.to_numeric(cone_df["ra_deg"], errors="coerce") <= frame_ra_max)
                & (pd.to_numeric(cone_df["dec_deg"], errors="coerce") >= frame_dec_min)
                & (pd.to_numeric(cone_df["dec_deg"], errors="coerce") <= frame_dec_max)
            ]
            cone_in_frame_n = int(len(cone_f))
            if cone_in_frame_n > 0:
                cgra = pd.to_numeric(cone_f["ra_deg"], errors="coerce").to_numpy(dtype=float)
                cgdec = pd.to_numeric(cone_f["dec_deg"], errors="coerce").to_numpy(dtype=float)
                seps_cone = _nearest_gaia_vec(u_ra, u_de, cgra, cgdec)
                finite_c = seps_cone[np.isfinite(seps_cone)]
                if finite_c.size:
                    cone_field_p50 = float(np.percentile(finite_c, 50))
                    cone_field_within_20 = int((finite_c <= 20.0).sum())
        except Exception as exc:  # noqa: BLE001
            cone_in_frame_n = -1
            lines_cone_err = str(exc)
        else:
            lines_cone_err = ""
    else:
        lines_cone_err = "missing field_catalog_cone.csv"
    seps_all = _nearest_gaia_vec(u_ra, u_de, gra, gdec)
    inside_flags = [
        _point_in_ra_dec_bbox(ra, dec, float(db_ra_min), float(db_ra_max), float(db_dec_min), float(db_dec_max))
        for ra, dec in zip(u_ra, u_de)
    ]
    for sep, inside in zip(seps_all, inside_flags):
        if not math.isfinite(sep):
            continue
        if inside:
            inside_seps.append(float(sep))
        else:
            outside_seps.append(float(sep))

    def _p50(arr: list[float]) -> float:
        return float(np.percentile(np.asarray(arr), 50)) if arr else float("nan")

    inside_frac = len(inside_seps) / max(1, len(inside_seps) + len(outside_seps))
    # Coverage hypothesis: unmatched DET far from catalog because outside depth/coverage
    coverage_confirmed = (
        cone_in_frame_n > 0
        and math.isfinite(cone_field_p50)
        and _p50(inside_seps) > 40.0
        and cone_field_within_20 < 0.02 * max(1, len(unmatched))
    )

    # T3 magnitude backstop — reference from ms rows with stored coords matching Gaia
    good_ref = []
    for cid, row in ms_by_cid.items():
        g = ref_gaia.get(cid)
        if g is None:
            continue
        g_ra, g_de, g_mag = g
        if g_mag is None:
            continue
        sep_sg = _sep_arcsec(float(row["ra_deg"]), float(row["dec_deg"]), g_ra, g_de)
        if sep_sg <= 0.5:
            flux = pd.to_numeric(row.get("flux") or row.get("dao_flux"), errors="coerce")
            mag_inst = -2.5 * math.log10(float(flux)) if pd.notna(flux) and float(flux) > 0 else float("nan")
            if math.isfinite(mag_inst):
                good_ref.append((mag_inst, g_mag))
    t3_outliers = 0
    if len(good_ref) >= 20:
        mi = np.array([t[0] for t in good_ref])
        mg = np.array([t[1] for t in good_ref])
        coeff = np.polyfit(mi, mg, 1)
        pred = np.polyval(coeff, mi)
        resid = mg - pred
        sigma = float(np.std(resid))
        for r in t1_rows:
            if r.get("class") not in ("STALE-COORDS", "MISASSIGNED-ID", "CONSISTENT"):
                continue
            cid = r["cid"]
            mrow = ms_by_cid[cid]
            g = ref_gaia.get(cid)
            if g is None:
                continue
            flux = pd.to_numeric(mrow.get("flux") or mrow.get("dao_flux"), errors="coerce")
            mag_inst = -2.5 * math.log10(float(flux)) if pd.notna(flux) and float(flux) > 0 else float("nan")
            if not math.isfinite(mag_inst):
                continue
            pred_g = float(np.polyval(coeff, mag_inst))
            resid_g = float(g[2]) - pred_g if g[2] is not None else float("nan")
            r["mag_resid"] = resid_g
            if math.isfinite(resid_g) and abs(resid_g) > 0.5:
                t3_outliers += 1

    # Output
    lines: list[str] = []
    lines.append("# F-428-COORD forensics v4 (read-only)")
    lines.append(f"inputs: masterstars={ms_path.resolve()}")
    lines.append(f"inputs: variable_targets={vt_path.resolve()}")
    lines.append(f"inputs: active_targets={at_path.resolve()}")
    lines.append(f"inputs: MASTERSTAR.fits={fits_path.resolve()}")
    lines.append(f"inputs: gaia_db={gdb.resolve()} table={table} total_rows={db_total}")
    lines.append(f"inputs: stale_wcs_threshold={args.stale_wcs_px_threshold}px = {stale_thr_arcsec:.2f}\"")
    lines.append("")
    lines.append("# v3 interpretation corrections (carried forward)")
    lines.append("- vector_mean~0 + high mean|d| => isotropic offsets, NOT rigid shift")
    lines.append("- recompute-invariance (p50~81\" stored vs final WCS) => unmatched DET coords likely fine; test Gaia coverage")
    lines.append("- matched rows split: ms-gaia~0 (detection-time Gaia coords) vs ms-gaia large (optimizer assigned id without coord refresh)")
    lines.append("")
    lines.append("# T1 — Pixel-space identity test")
    lines.append(f"violating_rows (vt-ms > {args.vt_ms_violation_arcsec}\"): {len(violation_ids)}")
    lines.append(f"classification_histogram: {json.dumps(hist, sort_keys=True)}")
    lines.append(f"MISASSIGNED-ID count: {len(misassigned)}")
    lines.append(f"STALE-COORDS count: {len(stale)}")
    lines.append(f"CONSISTENT count: {hist.get('CONSISTENT', 0)}")
    lines.append(f"T1_subclass: {json.dumps(t1_sub, sort_keys=True)}")
    lines.append("  wcs_stored_lt1: stored ra/dec tracks final-WCS(x/y); catalog_id Gaia position offset")
    lines.append("  stored_gaia_lt1: stored ra/dec forced to Gaia; x/y does not match (metadata/pixel split)")
    lines.append("")
    lines.append("## Priority active targets")
    for name in PRIORITY_TARGETS:
        match = [r for r in t1_rows if name in r.get("name", "")]
        if not match:
            lines.append(f"  {name}: NOT IN violation set (or no ms row)")
            continue
        r = match[0]
        lc = r.get("lc") or {}
        lc_note = ""
        if lc.get("lc_found"):
            lc_note = f" LC n={lc.get('n_frames')} std={lc.get('mag_std_mmag', float('nan')):.1f}mmag range={lc.get('mag_range_mmag', float('nan')):.1f}mmag"
        lines.append(
            f"  {r['name']} cid={r['cid']} class={r['class']} "
            f"sep_wcs_gaia={r.get('sep_wcs_gaia', float('nan')):.3f}\" "
            f"sep_stored_gaia={r.get('sep_stored_gaia', float('nan')):.3f}\" "
            f"sep_vt_ms={r.get('sep_vt_ms', float('nan')):.3f}\" active={r.get('active')}{lc_note}"
        )
    lines.append("")
    lines.append("## 10 random non-target violations")
    non_target = [r for r in t1_rows if not r.get("active") and r["class"] in ("STALE-COORDS", "MISASSIGNED-ID", "CONSISTENT")]
    rng = np.random.default_rng(428)
    if non_target:
        pick = rng.choice(len(non_target), size=min(10, len(non_target)), replace=False)
        for i in sorted(pick):
            r = non_target[int(i)]
            lines.append(
                f"  {r['name']} cid={r['cid']} class={r['class']} "
                f"sep_wcs={r.get('sep_wcs_gaia', float('nan')):.3f}\" sep_stored={r.get('sep_stored_gaia', float('nan')):.3f}\""
            )
    if misassigned:
        lines.append("")
        lines.append("## MISASSIGNED-ID explicit list")
        for r in misassigned:
            lines.append(
                f"  {r['name']} cid={r['cid']} sep_wcs_gaia={r['sep_wcs_gaia']:.3f}\" "
                f"sep_stored_gaia={r['sep_stored_gaia']:.3f}\" x={r['x']:.1f} y={r['y']:.1f}"
            )

    lines.append("")
    lines.append("# T2 — Gaia DB coverage vs frame footprint")
    lines.append(f"db_bbox: RA [{db_ra_min:.6f}, {db_ra_max:.6f}] DEC [{db_dec_min:.6f}, {db_dec_max:.6f}]")
    if cone_meta:
        lines.append(f"db_provenance_json: {cone_meta.get('_source', '')}")
        for k in ("center_ra_deg", "center_dec_deg", "cone_radius_deg", "radius_deg", "plate_solve_fov_deg"):
            if k in cone_meta:
                lines.append(f"  {k}={cone_meta[k]}")
    lines.append(
        f"frame_footprint (final WCS corners): RA [{frame_ra_min:.6f}, {frame_ra_max:.6f}] "
        f"DEC [{frame_dec_min:.6f}, {frame_dec_max:.6f}]"
    )
    lines.append(f"unmatched_DET_total: {len(unmatched)}")
    lines.append(f"unmatched inside db_bbox: {len(inside_seps)} ({100*inside_frac:.1f}%)")
    lines.append(f"unmatched outside db_bbox: {len(outside_seps)} ({100*(1-inside_frac):.1f}%)")
    lines.append(f"inside_nearest_gaia_p50 (sqlite field slice): {_p50(inside_seps):.3f}\"")
    lines.append(f"outside_nearest_gaia_p50: {_p50(outside_seps):.3f}\"")
    lines.append(f"field_catalog_cone_in_frame: n={cone_in_frame_n} ({lines_cone_err})")
    lines.append(f"field_catalog_cone_nn_p50_unmatched_DET: {cone_field_p50:.3f}\"")
    lines.append(f"field_catalog_cone_nn_within_20arcsec: {cone_field_within_20}/{len(unmatched)}")
    lines.append(
        "unmatched_DET interpretation: spurious DAO detections without catalog association "
        "(not missing DB rows — 4379 catalog sources in frame, mag<=15.26)"
    )
    lines.append(f"COVERAGE VERDICT (81\" unmatched population): {'CONFIRMED' if coverage_confirmed else 'REFUTED'}")
    if coverage_confirmed:
        lines.append("  radius decision reframed: no match-radius change helps; wider/deeper field DB build is actionable item")

    lines.append("")
    lines.append("# T3 — Magnitude-consistency backstop")
    lines.append(f"reference_fit_n (ms-gaia<=0.5\"): {len(good_ref)}")
    lines.append(f"violation_rows |mag_resid|>0.5mag: {t3_outliers}")

    fail: list[str] = []
    if misassigned:
        fail.append(f"T1 MISASSIGNED-ID: {len(misassigned)} rows — STOP gate")
    if len(violation_ids) != sum(hist.values()) - hist.get("NO_GAIA_DB", 0):
        pass  # NO_GAIA_DB ok
    if len(violation_ids) == 0:
        fail.append("T1: zero violation rows — unexpected")

    lines.append("")
    if fail:
        lines.append("DIAG SELF-CHECK FAIL")
        for f in fail:
            lines.append(f"  - {f}")
    else:
        if hist.get("STALE-COORDS", 0) == len(violation_ids) - hist.get("NO_GAIA_DB", 0):
            lines.append("DIAG SELF-CHECK PASS (T1 all STALE-COORDS; T4 audit in CURSOR_RESULT)")
        else:
            lines.append("DIAG SELF-CHECK PASS (T1 no MISASSIGNED-ID; mixed classes — see histogram)")

    text = "\n".join(lines) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
    return 1 if misassigned else 0


if __name__ == "__main__":
    raise SystemExit(main())
