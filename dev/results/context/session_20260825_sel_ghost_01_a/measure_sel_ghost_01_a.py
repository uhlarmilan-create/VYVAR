"""SEL-GHOST-01 Part A: measure g_60_4 association (read-only on live draft 520)."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

ROOT = Path(r"C:\ASTRO\python\VYVAR")
sys.path.insert(0, str(ROOT / "src_py"))

from gaia_catalog_id import normalize_gaia_source_id  # noqa: E402
from wcs_invertibility import post_match_pixel_sep  # noqa: E402

SESSION = ROOT / "dev" / "results" / "session_20260825_sel_ghost_01_a"
G60 = ROOT / "Archive" / "Drafts" / "draft_000520" / "platesolve" / "g_60_4"
INFOLOG = ROOT / "Archive" / "Drafts" / "draft_000520" / "infolog_20260824_204055.txt"
MS_CSV = G60 / "masterstars_full_match.csv"
MS_FITS = G60 / "MASTERSTAR.fits"
CONE = G60 / "field_catalog_cone.csv"
CENSUS = G60 / "gaia_source_state_census.csv"
CALIB = G60 / "dao_gaia_calibration.json"
EPSF_516 = ROOT / "Archive" / "Drafts" / "draft_000516" / "platesolve" / "NoFilter_60_2" / "masterstar_epsf.fits"

GHOST_IDS = [
    "1112112413285008896",
    "1112115024625070720",
    "1111930718988511616",
    "1112119250872867200",
    "1112110042463052928",
    "1111931371821079552",
    "1111737823417422464",
    "1111922300852743808",
]

LOG_PATTERNS = [
    re.compile(r"Catalog match", re.I),
    re.compile(r"post_match_identity_gate"),
    re.compile(r"WCS refine", re.I),
    re.compile(r"opakovanie"),
    re.compile(r"zuzenie"),
    re.compile(r"refine zamietnuty"),
    re.compile(r"refine skipped", re.I),
    re.compile(r"Gaia/pixel NN"),
    re.compile(r"match_sep_arcsec"),
    re.compile(r"MATCH STATS"),
    re.compile(r"Astrometry optimizer"),
    re.compile(r"MASTERSTAR: VYVAR pary"),
    re.compile(r"Match rate"),
    re.compile(r"binning DAO"),
    re.compile(r"SNR filter"),
    re.compile(r"Gaia->DAO"),
]


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def nonempty_cid(series: pd.Series) -> pd.Series:
    return series.map(normalize_gaia_source_id).astype(str).str.strip().replace({"nan": "", "None": ""})


def extract_infolog(path: Path) -> list[dict]:
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()
    # g_60_4 MASTERSTAR block is around the first "binning DAO" after platesolve.
    rows = []
    for i, line in enumerate(lines, start=1):
        if any(p.search(line) for p in LOG_PATTERNS):
            rows.append({"line": i, "text": line.rstrip()})
    return rows, lines


def load_cone_coords(cone_path: Path) -> dict[str, tuple[float, float, float]]:
    cone = pd.read_csv(cone_path, low_memory=False, dtype={"catalog_id": str, "source_id": str})
    id_col = "catalog_id" if "catalog_id" in cone.columns else "source_id"
    ra_col = "ra_deg" if "ra_deg" in cone.columns else "ra"
    de_col = "dec_deg" if "dec_deg" in cone.columns else "dec"
    mag_col = None
    for c in ("phot_g_mean_mag", "g_mag", "mag"):
        if c in cone.columns:
            mag_col = c
            break
    out: dict[str, tuple[float, float, float]] = {}
    for _, r in cone.iterrows():
        k = normalize_gaia_source_id(r.get(id_col))
        if not k:
            continue
        ra = float(pd.to_numeric(r.get(ra_col), errors="coerce"))
        de = float(pd.to_numeric(r.get(de_col), errors="coerce"))
        mag = float(pd.to_numeric(r.get(mag_col), errors="coerce")) if mag_col else float("nan")
        if math.isfinite(ra) and math.isfinite(de):
            out[k] = (ra, de, mag)
    return out


def main() -> None:
    SESSION.mkdir(parents=True, exist_ok=True)
    summary: dict = {}

    ms_sha = sha256_file(MS_CSV)
    epsf_sha = sha256_file(EPSF_516)
    summary["masterstars_full_match_sha256"] = ms_sha
    summary["epsf_516_sha256"] = epsf_sha
    summary["epsf_516_expected"] = "172f95403beae36dc9c7b35e4758f37996bb661e3d96d180d1444ded71369a20"
    summary["epsf_516_unchanged"] = epsf_sha == summary["epsf_516_expected"]

    # --- 2.1 infolog ---
    log_rows, all_lines = extract_infolog(INFOLOG)
    with (SESSION / "log_forensics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["line", "text"])
        w.writeheader()
        w.writerows(log_rows)

    # Restrict to g_60_4 MASTERSTAR neighborhood: first DAO binning through optimizer write.
    # Find first "binning DAO" then take until next platesolve set or 80 lines of hits.
    dao_hits = [r for r in log_rows if "binning DAO" in r["text"]]
    summary["dao_binning_log_lines"] = dao_hits
    gate_lines = [r for r in log_rows if "post_match_identity_gate" in r["text"]]
    refine_cat = [
        r
        for r in log_rows
        if ("Catalog match: WCS refine" in r["text"])
        or ("Gaia/pixel NN WCS refine" in r["text"])
        or ("refine zamietnuty" in r["text"])
        or ("refine skipped" in r["text"].lower())
    ]
    widen_lines = [r for r in log_rows if "opakovanie" in r["text"] or "zuzenie" in r["text"]]
    summary["n_post_match_identity_gate_lines"] = len(gate_lines)
    summary["gate_lines"] = gate_lines
    summary["catalog_wcs_refine_lines"] = refine_cat
    summary["widen_tighten_lines"] = widen_lines
    skipped = [r for r in log_rows if "post_match_identity_gate skipped:" in r["text"]]
    summary["gate_skipped_lines"] = skipped

    # --- table + WCS ---
    df = pd.read_csv(MS_CSV, low_memory=False, dtype={"catalog_id": str, "name": str})
    cid = nonempty_cid(df["catalog_id"] if "catalog_id" in df.columns else pd.Series([""] * len(df)))
    df["_cid"] = cid
    matched = cid.ne("")
    n_rows = int(len(df))
    n_matched = int(matched.sum())
    summary["n_ms_rows"] = n_rows
    summary["n_matched_catalog_id"] = n_matched
    summary["nunique_catalog_id"] = int(cid[matched].nunique())
    summary["columns"] = list(df.columns)
    summary["match_sep_arcsec_column_present"] = "match_sep_arcsec" in df.columns
    if "vy_match_mode" in df.columns:
        summary["vy_match_mode_counts"] = (
            df.loc[matched, "vy_match_mode"].fillna("").astype(str).value_counts().to_dict()
        )

    with fits.open(MS_FITS) as hdul:
        hdr = hdul[0].header
        data_shape = hdul[0].data.shape if hdul[0].data is not None else None
        wcs = WCS(hdr)
    vy_fwhm = float(hdr.get("VY_FWHM") or hdr.get("DAO_FWHM") or float("nan"))
    xbin = hdr.get("XBINNING")
    summary["header_VY_FWHM"] = vy_fwhm
    summary["header_XBINNING"] = xbin
    summary["masterstar_shape"] = list(data_shape) if data_shape else None
    scales = proj_plane_pixel_scales(wcs)
    plate_scale = float(np.mean(np.asarray(scales, dtype=np.float64)) * 3600.0)
    summary["plate_scale_arcsec_px_wcs"] = plate_scale
    if CALIB.exists():
        calib = json.loads(CALIB.read_text(encoding="utf-8"))
        summary["dao_gaia_calibration_keys"] = sorted(calib.keys()) if isinstance(calib, dict) else None
        for k in ("plate_scale_arcsec_px", "pixscale_arcsec", "arcsec_per_pixel", "fwhm_px"):
            if isinstance(calib, dict) and k in calib:
                summary[f"calib_{k}"] = calib[k]
        # nested
        if isinstance(calib, dict):
            for k, v in calib.items():
                if isinstance(v, (int, float, str)) and any(
                    t in str(k).lower() for t in ("scale", "fwhm", "bin")
                ):
                    summary[f"calib_{k}"] = v

    # Governing _fwhm_used: pipeline.py:9084 max(1.2, _base_fw / bfac)
    # Log: FWHM=1.25px, binning DAO=2x; header VY_FWHM=2.5 -> 2.5/2 = 1.25
    bfac = 2.0
    base_fw = float(vy_fwhm) if math.isfinite(vy_fwhm) and vy_fwhm > 0 else 2.5
    fwhm_used = float(max(1.2, base_fw / bfac))
    fail_px = 3.0 * fwhm_used
    warn_px = 1.5 * fwhm_used
    summary["_base_fw"] = base_fw
    summary["bfac"] = bfac
    summary["_fwhm_used"] = fwhm_used
    summary["fail_threshold_px"] = fail_px
    summary["warn_threshold_px"] = warn_px
    summary["fail_threshold_arcsec"] = fail_px * plate_scale

    cone_map = load_cone_coords(CONE)
    summary["cone_n"] = len(cone_map)

    census = pd.read_csv(CENSUS, low_memory=False, dtype=str)
    census_id_col = "catalog_id" if "catalog_id" in census.columns else "source_id"
    gmag_col = None
    for c in ("phot_g_mean_mag", "g_mag", "mag"):
        if c in census.columns:
            gmag_col = c
            break
    census["_cid"] = census[census_id_col].map(normalize_gaia_source_id)
    census["_g"] = pd.to_numeric(census[gmag_col], errors="coerce") if gmag_col else np.nan
    g12 = census.loc[census["_g"].lt(12.0) & census["_cid"].ne(""), "_cid"].tolist()
    # Prefer DETECTED if present
    if "source_state" in census.columns:
        det = census["source_state"].astype(str).str.startswith("DETECTED")
        g12_det = census.loc[census["_g"].lt(12.0) & det & census["_cid"].ne(""), "_cid"].tolist()
        if g12_det:
            g12 = g12_det
    g12 = sorted(set(g12), key=lambda k: cone_map.get(k, (0, 0, 99.0))[2])
    summary["g12_ids"] = g12
    summary["n_g12"] = len(g12)

    def row_metrics(r: pd.Series) -> dict:
        k = str(r["_cid"])
        x = float(pd.to_numeric(r.get("x"), errors="coerce"))
        y = float(pd.to_numeric(r.get("y"), errors="coerce"))
        g = cone_map.get(k)
        gaia_ra = g[0] if g else float("nan")
        gaia_dec = g[1] if g else float("nan")
        gmag = g[2] if g else float(pd.to_numeric(r.get("phot_g_mean_mag"), errors="coerce"))
        if math.isfinite(gaia_ra) and math.isfinite(gaia_dec):
            verdict, dpx = post_match_pixel_sep(
                x, y, gaia_ra, gaia_dec, wcs, fwhm_px=fwhm_used, fail_factor=3.0
            )
            gx, gy = wcs.world_to_pixel_values(gaia_ra, gaia_dec)
            ra_det, de_det = wcs.pixel_to_world_values(x, y)
            dra = (float(ra_det) - gaia_ra) * math.cos(math.radians(gaia_dec)) * 3600.0
            dde = (float(de_det) - gaia_dec) * 3600.0
            sky_sep = math.hypot(dra, dde)
        else:
            verdict, dpx = "fail", float("nan")
            gx = gy = float("nan")
            sky_sep = float("nan")
        mode = str(r.get("vy_match_mode") or "")
        return {
            "catalog_id": k,
            "G": gmag,
            "x": x,
            "y": y,
            "x_gaia": float(gx) if math.isfinite(float(gx)) else float("nan"),
            "y_gaia": float(gy) if math.isfinite(float(gy)) else float("nan"),
            "vy_match_mode": mode,
            "vy_dao_pass": r.get("vy_dao_pass"),
            "source_state": r.get("source_state"),
            "source_type": r.get("source_type"),
            "ambiguous_owner": r.get("ambiguous_owner"),
            "match_sep_arcsec": r.get("match_sep_arcsec") if "match_sep_arcsec" in df.columns else "column absent",
            "name": r.get("name"),
            "d_px": dpx,
            "sky_sep_arcsec": sky_sep,
            "gate_verdict": verdict,
            "name_is_gaia_id": bool(re.fullmatch(r"\d{12,22}", str(r.get("name") or "").strip())),
            "name_equals_catalog_id": str(r.get("name") or "").strip() == k,
        }

    ghost_rows = []
    for gid in GHOST_IDS:
        sub = df.loc[df["_cid"] == gid]
        if sub.empty:
            ghost_rows.append({"catalog_id": gid, "present": False})
            continue
        rec = row_metrics(sub.iloc[0])
        rec["present"] = True
        rec["set"] = "ghost"
        ghost_rows.append(rec)
    g12_rows = []
    for gid in g12:
        sub = df.loc[df["_cid"] == gid]
        if sub.empty:
            g12_rows.append({"catalog_id": gid, "present": False, "set": "g12"})
            continue
        rec = row_metrics(sub.iloc[0])
        rec["present"] = True
        rec["set"] = "g12"
        g12_rows.append(rec)

    prov = pd.DataFrame(ghost_rows + g12_rows)
    prov.to_csv(SESSION / "provenance_ghosts_g12.csv", index=False)

    # --- 2.3 fieldwide gate replay ---
    ok = warn = fail = no_coords = 0
    would_lose = 0
    dpx_all = []
    for idx in df.index[matched]:
        r = df.loc[idx]
        k = str(r["_cid"])
        g = cone_map.get(k)
        if g is None:
            no_coords += 1
            continue
        x = float(pd.to_numeric(r.get("x"), errors="coerce"))
        y = float(pd.to_numeric(r.get("y"), errors="coerce"))
        verdict, dpx = post_match_pixel_sep(x, y, g[0], g[1], wcs, fwhm_px=fwhm_used, fail_factor=3.0)
        dpx_all.append(dpx)
        if verdict == "ok":
            ok += 1
        elif verdict == "warn":
            warn += 1
        else:
            fail += 1
            would_lose += 1
    honest_n = ok + warn  # survive fail_factor 3.0 (warn kept)
    summary["gate_replay_ok"] = ok
    summary["gate_replay_warn"] = warn
    summary["gate_replay_fail"] = fail
    summary["gate_replay_no_gaia_coords"] = no_coords
    summary["n_would_lose_catalog_id"] = would_lose
    summary["dpx_p50"] = float(np.nanmedian(dpx_all)) if dpx_all else float("nan")
    summary["dpx_p95"] = float(np.nanpercentile(dpx_all, 95)) if dpx_all else float("nan")

    # retained DAO rows from log (SNR 692/719)
    n_retained = 692
    summary["n_retained_dao_log"] = n_retained
    summary["pipeline_match_rate_final_table"] = n_matched / float(n_retained)
    summary["pipeline_match_rate_final_over_nrows"] = n_matched / float(max(1, n_rows))
    summary["honest_match_rate_dpx_le_3fwhm"] = honest_n / float(n_retained)
    summary["honest_n_pairs"] = honest_n

    ghost_verdicts = [r.get("gate_verdict") for r in ghost_rows if r.get("present")]
    g12_verdicts = [r.get("gate_verdict") for r in g12_rows if r.get("present")]
    summary["ghost_n_present"] = sum(1 for r in ghost_rows if r.get("present"))
    summary["ghost_n_fail"] = sum(1 for v in ghost_verdicts if v == "fail")
    summary["ghost_n_lose_id"] = summary["ghost_n_fail"]
    summary["g12_n_present"] = sum(1 for r in g12_rows if r.get("present"))
    summary["g12_n_fail"] = sum(1 for v in g12_verdicts if v == "fail")
    summary["g12_n_lose_id"] = summary["g12_n_fail"]

    ghost_modes = [str(r.get("vy_match_mode") or "") for r in ghost_rows if r.get("present")]
    summary["ghost_vy_match_modes"] = ghost_modes
    summary["p_a3_all_locked"] = all(m == "locked" for m in ghost_modes) and len(ghost_modes) == 8

    sky = [float(r["sky_sep_arcsec"]) for r in ghost_rows if r.get("present") and math.isfinite(float(r.get("sky_sep_arcsec") or "nan"))]
    thr_as = fail_px * plate_scale
    summary["ghost_sky_sep_arcsec"] = sky
    summary["p_a4_all_gt_3fwhm_scale_le_96"] = bool(sky) and all((s > thr_as) and (s <= 96.0) for s in sky)
    summary["p_a4_n_gt_18"] = sum(1 for s in sky if s > 18.0)
    summary["last_logged_match_sep_arcsec"] = 18.0  # only logged widen; silent 0.95 loop unlogged

    # name restore diagnostic on final table (post-optimizer; both set)
    name_gaia = df["name"].map(lambda v: bool(re.fullmatch(r"\d{12,22}", str(v or "").strip())))
    summary["n_name_looks_like_gaia_id"] = int(name_gaia.sum())
    summary["n_name_gaia_and_catalog_id_set"] = int((name_gaia & matched).sum())

    (SESSION / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in [
        "n_ms_rows", "n_matched_catalog_id", "nunique_catalog_id",
        "_fwhm_used", "fail_threshold_px", "plate_scale_arcsec_px_wcs",
        "gate_replay_ok", "gate_replay_warn", "gate_replay_fail",
        "n_would_lose_catalog_id", "pipeline_match_rate_final_table",
        "honest_match_rate_dpx_le_3fwhm", "ghost_n_fail", "g12_n_fail",
        "p_a3_all_locked", "p_a4_all_gt_3fwhm_scale_le_96", "p_a4_n_gt_18",
        "n_g12", "epsf_516_unchanged", "match_sep_arcsec_column_present",
        "n_post_match_identity_gate_lines",
    ]}, indent=2))


if __name__ == "__main__":
    main()
