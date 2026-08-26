# -*- coding: ascii -*-
"""C6-3b measure: era04 vs live pixels (M1) and era03 vs era04 AC/ZP pool (M2).

Measure only. No config change. No era04 lock. No live mutation.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

ROOT = Path(r"C:\ASTRO\python\VYVAR")
sys.path.insert(0, str(ROOT / "src_py"))

from comp_rms_loo import (  # noqa: E402
    COMP_RMS_LOO_PHOTON_K_DEFAULT,
    LN10_OVER_2P5,
    compute_loo_mag_rms_map,
)
from d3_comparison_candidacy import apply_d3_comparison_candidacy  # noqa: E402
from gaia_catalog_id import normalize_gaia_source_id  # noqa: E402

ERA03 = ROOT / "Archive" / "Drafts" / "draft_000516_snapshot_era03_20260820"
ERA04 = ROOT / "Archive" / "Drafts" / "draft_000516_snapshot_era04_20260826"
LIVE = ROOT / "Archive" / "Drafts" / "draft_000516"
SETUP = "NoFilter_60_2"
SESSION = Path(__file__).resolve().parent
BO = "1498613634033133184"
FW = "1497343732462852864"
GH = "1498804639818507904"
PROBE_FRAMES = ("010", "050", "109")
DARK_NAME = "Dark_60s_Dark_0G_-10deg_Bin1_20260422.fits"
FLAT_NAME = "Flat_0.15s_NoFilter_0G_-10.5deg_Bin1_20260422.fits"
SQLITE = ROOT / "vyvar.sqlite3"
K_LOO = COMP_RMS_LOO_PHOTON_K_DEFAULT


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def nid(x: object) -> str:
    return normalize_gaia_source_id(x) or str(x).strip()


def id_series(df: pd.DataFrame) -> pd.Series:
    if "name" in df.columns:
        n = df["name"].map(nid)
        if float(n.str.fullmatch(r"\d{12,22}").mean()) > 0.5:
            return n
    if "catalog_id" in df.columns:
        return df["catalog_id"].map(nid)
    return df.iloc[:, 0].astype(str).map(nid)


ID_DTYPE = {
    "catalog_id": str,
    "name": str,
    "target_catalog_id": str,
    "comp_id": str,
}


def read_csv_ids(p: Path, **kwargs: object) -> pd.DataFrame:
    return pd.read_csv(p, dtype=ID_DTYPE, **kwargs)


def aligned_dir(root: Path) -> Path:
    return root / "detrended_aligned" / "lights" / SETUP


def phot(root: Path) -> Path:
    return root / "platesolve" / SETUP / "photometry"


def ms_path(root: Path) -> Path:
    return root / "platesolve" / SETUP / "masterstars_full_match.csv"


def lc_path(root: Path, tid: str) -> Path:
    return phot(root) / "lightcurves" / f"lightcurve_{tid}.csv"


def pct(a: np.ndarray, q: float) -> float:
    b = np.asarray(a, dtype=np.float64)
    b = b[np.isfinite(b)]
    if b.size == 0:
        return float("nan")
    return float(np.percentile(b, q))


def g_dist(s: pd.Series) -> dict[str, float]:
    a = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    return {"n": int(np.isfinite(a).sum()), "p10": pct(a, 10), "p50": pct(a, 50), "p90": pct(a, 90)}


def bprp_p50(s: pd.Series) -> float:
    return pct(pd.to_numeric(s, errors="coerce").to_numpy(dtype=float), 50)


def git_out(args: list[str]) -> str:
    r = subprocess.run(
        ["git", *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    return (r.stdout or "").strip()


def load_fits_data(p: Path) -> np.ndarray:
    with fits.open(p, memmap=True) as hdul:
        return np.asarray(hdul[0].data, dtype=np.float64)


def stamp_dpix(a: np.ndarray, b: np.ndarray, x: float, y: float, r: float) -> dict[str, float]:
    h, w = a.shape
    r = max(1.0, float(r))
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(r)):
        return {"n_pix": 0, "max_abs": float("nan"), "median": float("nan")}
    x0 = int(max(0, math.floor(x - r - 1)))
    x1 = int(min(w, math.ceil(x + r + 2)))
    y0 = int(max(0, math.floor(y - r - 1)))
    y1 = int(min(h, math.ceil(y + r + 2)))
    yy, xx = np.ogrid[y0:y1, x0:x1]
    m = (xx - x) ** 2 + (yy - y) ** 2 <= r * r
    d = (b[y0:y1, x0:x1] - a[y0:y1, x0:x1])[m]
    if d.size == 0:
        return {"n_pix": 0, "max_abs": float("nan"), "median": float("nan")}
    return {"n_pix": int(d.size), "max_abs": float(np.max(np.abs(d))), "median": float(np.median(d))}


def sqlite_library_rows(paths: list[Path]) -> list[dict[str, object]]:
    if not SQLITE.is_file():
        return []
    conn = sqlite3.connect(str(SQLITE))
    conn.row_factory = sqlite3.Row
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(CALIBRATION_LIBRARY)").fetchall()]
        out: list[dict[str, object]] = []
        for p in paths:
            rows = conn.execute(
                "SELECT * FROM CALIBRATION_LIBRARY WHERE FILE_PATH = ? OR FILE_PATH LIKE ?",
                (str(p), f"%{p.name}"),
            ).fetchall()
            if not rows:
                out.append({"file": str(p), "n_records": 0, "columns": cols})
                continue
            for row in rows:
                rec = {k: row[k] for k in row.keys()}
                rec["_queried_file"] = str(p)
                out.append(rec)
        return out
    except sqlite3.Error as exc:
        return [{"error": str(exc)}]
    finally:
        conn.close()


def master_identity(p: Path) -> dict[str, object]:
    exists = p.is_file()
    st = p.stat() if exists else None
    return {
        "path": str(p),
        "exists": exists,
        "sha256": sha256_file(p) if exists else None,
        "size": int(st.st_size) if st else None,
        "mtime_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
        if st
        else None,
    }


def cal_diag_dark(root: Path) -> dict[str, object]:
    p = root / "cal_diag.json"
    if not p.is_file():
        return {"path": str(p), "exists": False}
    js = json.loads(p.read_text(encoding="utf-8"))
    keys = js.get("keys") or {}
    first = next(iter(keys.values()), {}) if keys else {}
    return {
        "cal_diag_path": str(p),
        "n_keys": int(len(keys)),
        "key_names": list(keys.keys()),
        "dark_path": first.get("dark_path"),
        "status": first.get("status"),
        "spec_version": js.get("spec_version"),
    }


def manifest_calib(root: Path) -> str:
    p = root / "draft_manifest.json"
    if not p.is_file():
        return ""
    js = json.loads(p.read_text(encoding="utf-8"))
    return str((js.get("calib") if isinstance(js, dict) else None) or js.get("fields", {}).get("calib") or "")


def parse_manifest_calib(s: str) -> dict[str, str]:
    out: dict[str, str] = {}
    if "dark=" in s:
        rest = s.split("dark=", 1)[1]
        out["dark"] = rest.split(";")[0].strip()
    if "Flat_" in s or "flat_by_filter" in s:
        for token in s.replace("\\\\", "\\").split("'"):
            if "Flat_" in token and token.endswith(".fits"):
                out["flat"] = token
                break
    return out


def ensemble_for(root: Path, tid: str) -> pd.DataFrame:
    p = phot(root) / "comparison_stars_per_target.csv"
    df = read_csv_ids(p)
    cid = df["target_catalog_id"].map(nid)
    return df.loc[cid.eq(tid)].copy()


def measure_m1() -> dict[str, object]:
    t0 = time.perf_counter()
    dark = ROOT / "CalibrationLibrary" / DARK_NAME
    flat = ROOT / "CalibrationLibrary" / FLAT_NAME
    dark_id = master_identity(dark)
    flat_id = master_identity(flat)
    lib_rows = sqlite_library_rows([dark, flat])

    cal_e4 = cal_diag_dark(ERA04)
    cal_live = cal_diag_dark(LIVE)
    man_e4 = parse_manifest_calib(manifest_calib(ERA04))
    man_live = parse_manifest_calib(manifest_calib(LIVE))
    man_e3 = parse_manifest_calib(manifest_calib(ERA03))

    ms = read_csv_ids(ms_path(ERA04))
    bo_row = ms.loc[id_series(ms).eq(BO)]
    bo_x = float(pd.to_numeric(bo_row["x"].iloc[0], errors="coerce")) if not bo_row.empty else float("nan")
    bo_y = float(pd.to_numeric(bo_row["y"].iloc[0], errors="coerce")) if not bo_row.empty else float("nan")
    lc = pd.read_csv(lc_path(ERA04, BO))
    ap_r = float(pd.to_numeric(lc["aperture_r_px"], errors="coerce").median())

    frames_out: list[dict[str, object]] = []
    sha_all_equal = True
    for fr in PROBE_FRAMES:
        name = f"BO_CVn_Light_{fr}.fits"
        pe4 = aligned_dir(ERA04) / name
        plv = aligned_dir(LIVE) / name
        rec: dict[str, object] = {
            "frame": fr,
            "name": name,
            "era04_exists": pe4.is_file(),
            "live_exists": plv.is_file(),
        }
        if pe4.is_file() and plv.is_file():
            s4 = sha256_file(pe4)
            sl = sha256_file(plv)
            rec["sha_era04"] = s4
            rec["sha_live"] = sl
            rec["sha_equal"] = s4 == sl
            if s4 != sl:
                sha_all_equal = False
            a = load_fits_data(pe4)
            b = load_fits_data(plv)
            d = b - a
            rec["naxis"] = [int(a.shape[1]), int(a.shape[0])]
            rec["max_abs_dpix"] = float(np.max(np.abs(d)))
            rec["median_dpix"] = float(np.median(d))
            rec["mean_dpix"] = float(np.mean(d))
            rec["frac_nonzero"] = float(np.mean(d != 0))
            rec["aperture"] = stamp_dpix(a, b, bo_x, bo_y, ap_r)
        frames_out.append(rec)

    align_log = git_out(
        [
            "log",
            "--since=2026-08-20",
            "--pretty=format:%h %ad %s",
            "--date=short",
            "--",
            "src_py/pipeline.py",
            "src_py/vyvar_alignment_frame.py",
            "src_py/osc_align.py",
        ]
    )
    head = git_out(["rev-parse", "--short", "HEAD"])
    live_mtime = None
    e4_mtime = None
    p_live109 = aligned_dir(LIVE) / "BO_CVn_Light_109.fits"
    p_e4109 = aligned_dir(ERA04) / "BO_CVn_Light_109.fits"
    if p_live109.is_file():
        live_mtime = datetime.fromtimestamp(p_live109.stat().st_mtime, tz=timezone.utc).isoformat()
    if p_e4109.is_file():
        e4_mtime = datetime.fromtimestamp(p_e4109.stat().st_mtime, tz=timezone.utc).isoformat()

    dark_same = (cal_e4.get("dark_path") == cal_live.get("dark_path")) and (
        man_e4.get("dark") == man_live.get("dark")
    )
    flat_same = man_e4.get("flat") == man_live.get("flat")
    masters_equal = bool(dark_same and flat_same and dark_id["sha256"] and flat_id["sha256"])

    # Per-epoch BO dmag vs aperture dpix. If probe SHAs equal, skip 134-frame FITS load.
    lc3 = pd.read_csv(lc_path(ERA03, BO))
    lc4 = pd.read_csv(lc_path(ERA04, BO))
    lc3["sf"] = lc3["source_file"].astype(str).map(lambda s: Path(s).name)
    lc4["sf"] = lc4["source_file"].astype(str).map(lambda s: Path(s).name)
    j = lc3.merge(lc4, on="sf", suffixes=("_e3", "_e4"))
    j["dmag_mmag"] = (j["mag_calib_e4"] - j["mag_calib_e3"]) * 1000.0
    j["dmag_final_mmag"] = (j["mag_calib_final_e4"] - j["mag_calib_final_e3"]) * 1000.0

    epoch_rows: list[dict[str, object]] = []
    if sha_all_equal:
        for _, row in j.iterrows():
            epoch_rows.append(
                {
                    "source_file": row["sf"],
                    "dmag_mmag": float(row["dmag_mmag"]),
                    "dmag_final_mmag": float(row["dmag_final_mmag"]),
                    "max_abs_dpix_ap": 0.0,
                    "median_dpix_ap": 0.0,
                    "sha_equal_assumed": True,
                }
            )
        corr = float("nan")
        corr_note = "probe frames 010/050/109 SHA-equal; aperture dpix taken as 0; correlation undefined"
    else:
        for _, row in j.iterrows():
            bn = str(row["sf"])
            # source_file may be proc csv; map to aligned FITS basename
            stem = bn.replace("proc_", "").replace(".csv", ".fits")
            if not stem.endswith(".fits"):
                stem = Path(bn).stem
                if stem.startswith("proc_"):
                    stem = stem[5:]
                stem = stem + ".fits"
            pe4 = aligned_dir(ERA04) / stem
            plv = aligned_dir(LIVE) / stem
            rec = {
                "source_file": bn,
                "fits": stem,
                "dmag_mmag": float(row["dmag_mmag"]),
                "dmag_final_mmag": float(row["dmag_final_mmag"]),
            }
            if pe4.is_file() and plv.is_file():
                a = load_fits_data(pe4)
                b = load_fits_data(plv)
                st = stamp_dpix(a, b, bo_x, bo_y, ap_r)
                rec["max_abs_dpix_ap"] = st["max_abs"]
                rec["median_dpix_ap"] = st["median"]
            else:
                rec["max_abs_dpix_ap"] = float("nan")
                rec["median_dpix_ap"] = float("nan")
            epoch_rows.append(rec)
        xs = np.array([r["max_abs_dpix_ap"] for r in epoch_rows], dtype=float)
        ys = np.array([r["dmag_mmag"] for r in epoch_rows], dtype=float)
        m = np.isfinite(xs) & np.isfinite(ys)
        if int(m.sum()) >= 5 and float(np.std(xs[m])) > 0:
            corr = float(np.corrcoef(xs[m], ys[m])[0, 1])
            corr_note = "pearson corr(max|dpix|_aperture, dmag_calib)"
        else:
            corr = float("nan")
            corr_note = "insufficient dpix variance or n"

    pixels_differ = not sha_all_equal
    preproc_named: str | None = None
    if pixels_differ and not masters_equal:
        preproc_named = "master_identity_library"
    elif pixels_differ and masters_equal:
        preproc_named = "alignment_or_resample_code"
    else:
        preproc_named = None

    out = {
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "compared": "era04 aligned FITS vs live 516 aligned FITS; masters from CalibrationLibrary vs cal_diag/manifest",
        "bo_xy": {"x": bo_x, "y": bo_y, "aperture_r_px": ap_r},
        "frames": frames_out,
        "sha_all_probe_equal": sha_all_equal,
        "pixels_differ": pixels_differ,
        "masters": {
            "dark_file": dark_id,
            "flat_file": flat_id,
            "cal_diag_era04": cal_e4,
            "cal_diag_live": cal_live,
            "manifest_era04": man_e4,
            "manifest_live": man_live,
            "manifest_era03": man_e3,
            "dark_path_equal_era04_vs_live": dark_same,
            "flat_path_equal_era04_vs_live": flat_same,
            "masters_equal": masters_equal,
            "library_records": lib_rows,
        },
        "alignment": {
            "HEAD": head,
            "live_Light_109_mtime_utc": live_mtime,
            "era04_Light_109_mtime_utc": e4_mtime,
            "git_log_since_20260820": align_log.splitlines() if align_log else [],
        },
        "bo_dmag_vs_dpix": {
            "n_epochs": int(len(epoch_rows)),
            "corr": corr,
            "note": corr_note,
            "dmag_calib_median_mmag": float(np.nanmedian(j["dmag_mmag"].to_numpy(dtype=float))),
            "dmag_final_median_mmag": float(np.nanmedian(j["dmag_final_mmag"].to_numpy(dtype=float))),
        },
        "preproc_named_input": preproc_named,
    }
    pd.DataFrame(frames_out).to_csv(SESSION / "c63b_m1_frames.csv", index=False)
    pd.DataFrame(epoch_rows).to_csv(SESSION / "c63b_m1_bo_dmag_dpix.csv", index=False)
    return out


def static_pool_pre_d3(ms: pd.DataFrame) -> pd.DataFrame:
    """Replay build_global_comp_pool static filters up to D3 (no RMS)."""
    pool = ms.copy()
    if "x" not in pool.columns or "y" not in pool.columns:
        return pool.iloc[0:0].copy()
    if "zone" in pool.columns:
        z = pool["zone"].astype(str).str.strip().str.lower()
        pool = pool.loc[~z.isin(["saturated", "nonlinear"])].copy()

    def _b(col: str) -> pd.Series:
        if col not in pool.columns:
            return pd.Series(False, index=pool.index)
        v = pool[col]
        if v.dtype == bool:
            return v.fillna(False)
        return v.astype(str).str.strip().str.lower().isin(["1", "true", "yes"])

    cand = ~_b("is_saturated") & ~_b("vsx_known_variable") & ~_b("likely_saturated")
    return pool.loc[cand].copy()


def measure_m2() -> dict[str, object]:
    t0 = time.perf_counter()
    ms3 = read_csv_ids(ms_path(ERA03))
    ms4 = read_csv_ids(ms_path(ERA04))
    cs3 = read_csv_ids(ERA03 / "platesolve" / SETUP / "comparison_stars.csv")
    cs4 = read_csv_ids(ERA04 / "platesolve" / SETUP / "comparison_stars.csv")

    def pool_stats(df: pd.DataFrame, label: str) -> dict[str, object]:
        gcol = "phot_g_mean_mag" if "phot_g_mean_mag" in df.columns else "mag"
        bcol = "bp_rp" if "bp_rp" in df.columns else None
        ids = id_series(df)
        return {
            "label": label,
            "n": int(len(df)),
            "n_unique_id": int(ids.nunique()),
            "G": g_dist(df[gcol]),
            "bp_rp_p50": bprp_p50(df[bcol]) if bcol else float("nan"),
        }

    pools = {
        "era03_comparison_stars": pool_stats(cs3, "era03_comparison_stars"),
        "era04_comparison_stars": pool_stats(cs4, "era04_comparison_stars"),
        "era03_masterstars": pool_stats(ms3, "era03_masterstars"),
        "era04_masterstars": pool_stats(ms4, "era04_masterstars"),
    }

    # D3 split on era04 (the 1860)
    meta_p = phot(ERA04) / "pipeline_meta.json"
    fwhm = 3.7
    solve_rms = None
    if meta_p.is_file():
        pj = json.loads(meta_p.read_text(encoding="utf-8"))
        inp = pj.get("match_sep_formula_inputs") or {}
        if isinstance(inp, dict) and inp.get("solve_rms_px") is not None:
            solve_rms = float(inp["solve_rms_px"])
        fw = pj.get("fwhm_px") or pj.get("fwhm_dao_px")
        if fw is not None:
            try:
                fwhm = float(fw)
            except (TypeError, ValueError):
                pass

    pre = static_pool_pre_d3(ms4)
    d3_mask, d3_meta = apply_d3_comparison_candidacy(
        pre, fwhm_dao_px=fwhm, solve_rms_px=solve_rms, log_label="c63b"
    )
    d3_in = pre.copy()
    d3_in["_d3"] = d3_mask.to_numpy()
    st = d3_in["source_state"].astype(str).str.strip()
    from masterstar_gaia_accounting import SOURCE_DETECTED_P1, SOURCE_DETECTED_P2

    m_state = st.isin({SOURCE_DETECTED_P1, SOURCE_DETECTED_P2})
    snr = pd.to_numeric(d3_in["snr_ap_pixscaled"], errors="coerce")
    m_snr = snr.ge(10.0)
    dropped = d3_in.loc[~d3_in["_d3"]].copy()
    drop_state = dropped.loc[~m_state.reindex(dropped.index).fillna(False)].copy()
    drop_snr = dropped.loc[m_state.reindex(dropped.index).fillna(False) & ~m_snr.reindex(dropped.index).fillna(False)].copy()
    # residual/gate should be 0 from C6-1 log
    gate = d3_in["vy_identity_gate"].astype(str).str.strip().str.lower()
    resid = pd.to_numeric(d3_in["gaia_dao_resid_px"], errors="coerce")
    from d3_comparison_candidacy import d3_resid_ceiling_px

    ceil = d3_resid_ceiling_px(fwhm_dao_px=fwhm, solve_rms_px=solve_rms)
    m_gate = gate.ne("fail")
    m_resid = resid.le(ceil)
    drop_gate = dropped.loc[
        m_state.reindex(dropped.index).fillna(False)
        & ~m_gate.reindex(dropped.index).fillna(False)
    ].copy()
    drop_resid = dropped.loc[
        m_state.reindex(dropped.index).fillna(False)
        & m_gate.reindex(dropped.index).fillna(False)
        & ~m_resid.reindex(dropped.index).fillna(False)
    ].copy()

    survivors = d3_in.loc[d3_in["_d3"]].copy()

    # LOO on D3 survivors (pool that can hit k*photon)
    proc_dir = aligned_dir(ERA04)
    proc_paths = sorted(proc_dir.glob("proc_*.csv"))
    csv_cache: dict[str, pd.DataFrame] = {}
    for p in proc_paths:
        csv_cache[str(p)] = read_csv_ids(p)
    cand_ids = {i for i in id_series(survivors).tolist() if i}
    loo_map, _basis = compute_loo_mag_rms_map(
        cand_ids=cand_ids,
        per_frame_csv_paths=proc_paths,
        csv_cache=csv_cache,
        flux_col="dao_flux",
        min_frames_frac=0.3,
    )
    survivors = survivors.copy()
    survivors["_nid"] = id_series(survivors)
    survivors["loo_rms"] = survivors["_nid"].map(lambda i: float(loo_map.get(i, float("nan"))))
    survivors["photon"] = LN10_OVER_2P5 / pd.to_numeric(survivors["snr_ap_pixscaled"], errors="coerce")
    survivors["r"] = survivors["loo_rms"] / survivors["photon"]
    k_ceil = np.minimum(0.1, float(K_LOO) * survivors["photon"].to_numpy(dtype=float))
    survivors["k_photon_ceil"] = k_ceil
    survivors["fail_k_photon"] = survivors["loo_rms"].to_numpy(dtype=float) > k_ceil
    fail_k = survivors.loc[survivors["fail_k_photon"]].copy()

    g_surv = pd.to_numeric(survivors["phot_g_mean_mag"], errors="coerce")
    r_surv = pd.to_numeric(survivors["r"], errors="coerce")
    # G at which median r crosses 5: bin by G, find brightest G where median r >= 5
    order = np.argsort(g_surv.to_numpy(dtype=float))
    g_arr = g_surv.to_numpy(dtype=float)[order]
    r_arr = r_surv.to_numpy(dtype=float)[order]
    mfin = np.isfinite(g_arr) & np.isfinite(r_arr)
    g_arr, r_arr = g_arr[mfin], r_arr[mfin]
    g_cross = float("nan")
    # rolling median in G-sorted windows of 40 stars; first (bright) crossing of r=5
    win = 40
    if g_arr.size >= win:
        for i in range(0, int(g_arr.size) - win + 1):
            med_r = float(np.median(r_arr[i : i + win]))
            g_mid = float(np.median(g_arr[i : i + win]))
            if med_r >= 5.0:
                g_cross = g_mid
                break
    # also: among brightest decile, median r
    n_s = int(g_arr.size)
    bright_decile_r = float("nan")
    if n_s >= 10:
        n10 = max(1, n_s // 10)
        bright_decile_r = float(np.median(r_arr[:n10]))

    r_vs_g_rows = []
    if g_arr.size:
        # 0.5-mag bins
        gmin, gmax = float(np.min(g_arr)), float(np.max(g_arr))
        edges = np.arange(math.floor(gmin * 2) / 2.0, math.ceil(gmax * 2) / 2.0 + 0.51, 0.5)
        for i in range(len(edges) - 1):
            sel = (g_arr >= edges[i]) & (g_arr < edges[i + 1])
            if int(sel.sum()) < 3:
                continue
            r_vs_g_rows.append(
                {
                    "G_lo": float(edges[i]),
                    "G_hi": float(edges[i + 1]),
                    "n": int(sel.sum()),
                    "r_p50": float(np.median(r_arr[sel])),
                    "r_p90": float(np.percentile(r_arr[sel], 90)),
                    "frac_r_gt_5": float(np.mean(r_arr[sel] > 5.0)),
                }
            )
    pd.DataFrame(r_vs_g_rows).to_csv(SESSION / "c63b_m2_r_vs_g.csv", index=False)

    def grp_g(df: pd.DataFrame) -> dict[str, object]:
        if df is None or df.empty:
            return {"n": 0, "G": g_dist(pd.Series(dtype=float)), "bp_rp_p50": float("nan")}
        return {
            "n": int(len(df)),
            "G": g_dist(df["phot_g_mean_mag"] if "phot_g_mean_mag" in df.columns else df.get("mag", pd.Series(dtype=float))),
            "bp_rp_p50": bprp_p50(df["bp_rp"]) if "bp_rp" in df.columns else float("nan"),
        }

    drop_split = {
        "n_pre_d3": int(len(d3_in)),
        "n_d3_out": int(int(d3_meta.get("n_out", 0))),
        "n_removed_1860_expected": int(len(d3_in)) - int(d3_meta.get("n_out", 0)),
        "d3_meta": d3_meta,
        "source_state": grp_g(drop_state),
        "snr_ap_pixscaled_lt_10": grp_g(drop_snr),
        "vy_identity_gate": grp_g(drop_gate),
        "gaia_dao_resid": grp_g(drop_resid),
        "k_photon_of_1860": {"n": 0, "note": "k*photon is after D3; none of the 1860 D3 drops are k*photon"},
        "k_photon_among_d3_survivors": grp_g(fail_k),
        "d3_survivors": grp_g(survivors),
    }

    # Per-target ensembles and CT replay
    replay_rows: list[dict[str, object]] = []
    for tid, name in ((BO, "BO"), (FW, "FW"), (GH, "GH")):
        e3 = pd.read_csv(lc_path(ERA03, tid))
        e4 = pd.read_csv(lc_path(ERA04, tid))
        e3["sf"] = e3["source_file"].astype(str).map(lambda s: Path(s).name)
        e4["sf"] = e4["source_file"].astype(str).map(lambda s: Path(s).name)
        jj = e3.merge(e4, on="sf", suffixes=("_e3", "_e4"))
        ens3 = ensemble_for(ERA03, tid)
        ens4 = ensemble_for(ERA04, tid)
        ids3 = set(id_series(ens3))
        ids4 = set(id_series(ens4))
        c1 = float(e4["ct_c1"].iloc[0])
        tgt_bprp = float(e4["ct_bp_rp_target"].iloc[0])
        ac4 = float(e4["ac_correction"].iloc[0])
        med_e3 = float(e3["ct_bp_rp_comp_med"].iloc[0])
        med_e4 = float(e4["ct_bp_rp_comp_med"].iloc[0])
        # Replay: era04 mag_calib + era03 CT membership (era03 bp_rp_comp_med) + era04 AC
        ct_replay = c1 * (tgt_bprp - med_e3)
        mcf_replay = jj["mag_calib_e4"] + ct_replay + ac4
        d_obs = (jj["mag_calib_final_e4"] - jj["mag_calib_final_e3"]) * 1000.0
        d_rep = (mcf_replay.to_numpy(dtype=float) - jj["mag_calib_final_e3"].to_numpy(dtype=float)) * 1000.0
        # Weighted BP-RP of era03 membership using era04 Gaia bp_rp
        ms4_idx = ms4.copy()
        ms4_idx["_nid"] = id_series(ms4_idx)
        bps = []
        wts = []
        for cid in ids3:
            sub = ens3.loc[id_series(ens3).eq(cid)]
            w = float(pd.to_numeric(sub["comp_weight"].iloc[0], errors="coerce")) if not sub.empty and "comp_weight" in ens3.columns else 1.0
            rowm = ms4_idx.loc[ms4_idx["_nid"].eq(cid)]
            bp = float(pd.to_numeric(rowm["bp_rp"].iloc[0], errors="coerce")) if not rowm.empty else float("nan")
            if math.isfinite(bp) and math.isfinite(w) and w > 0:
                bps.append(bp)
                wts.append(w)
        if bps:
            bp_w = float(np.sum(np.array(wts) * np.array(bps)) / np.sum(wts))
        else:
            bp_w = float("nan")
        rec = {
            "target": name,
            "catalog_id": tid,
            "n_ens_e3": int(len(ids3)),
            "n_ens_e4": int(len(ids4)),
            "ids_swapped": sorted(ids3.symmetric_difference(ids4)),
            "ct_c1": c1,
            "ct_bp_rp_target": tgt_bprp,
            "ct_bp_rp_comp_med_e3": med_e3,
            "ct_bp_rp_comp_med_e4": med_e4,
            "d_bp_rp_comp_med": med_e4 - med_e3,
            "ac_e3": float(e3["ac_correction"].iloc[0]),
            "ac_e4": ac4,
            "dmag_calib_median_mmag": float(np.nanmedian((jj["mag_calib_e4"] - jj["mag_calib_e3"]) * 1000.0)),
            "dmag_final_obs_median_mmag": float(np.nanmedian(d_obs)),
            "dmag_final_replay_era03_ctmed_median_mmag": float(np.nanmedian(d_rep)),
            "ct_replay": ct_replay,
            "wmean_bp_rp_era03_ids_on_era04_ms": bp_w,
            "ens3_bp_rp_p50": bprp_p50(ens3["bp_rp"]) if "bp_rp" in ens3.columns else float("nan"),
            "ens4_bp_rp_p50": bprp_p50(ens4["bp_rp"]) if "bp_rp" in ens4.columns else float("nan"),
            "ens3_G": g_dist(ens3["phot_g_mean_mag"] if "phot_g_mean_mag" in ens3.columns else ens3["mag"]),
            "ens4_G": g_dist(ens4["phot_g_mean_mag"] if "phot_g_mean_mag" in ens4.columns else ens4["mag"]),
        }
        # dump ensemble members
        for era, ens in (("e3", ens3), ("e4", ens4)):
            rec[f"{era}_members"] = [
                {
                    "catalog_id": nid(r["name"]) if "name" in ens.columns else nid(r["catalog_id"]),
                    "bp_rp": float(pd.to_numeric(r.get("bp_rp"), errors="coerce")),
                    "G": float(pd.to_numeric(r.get("phot_g_mean_mag", r.get("mag")), errors="coerce")),
                    "comp_weight": float(pd.to_numeric(r.get("comp_weight"), errors="coerce"))
                    if "comp_weight" in ens.columns
                    else float("nan"),
                    "comp_tier": int(r["comp_tier"]) if "comp_tier" in ens.columns and pd.notna(r.get("comp_tier")) else None,
                }
                for _, r in ens.iterrows()
            ]
        replay_rows.append(rec)

    pd.json_normalize(replay_rows).to_csv(SESSION / "c63b_m2_replay.csv", index=False)

    # comparison_stars id overlap
    id3 = set(id_series(cs3))
    id4 = set(id_series(cs4))
    removed_cs = id3 - id4
    added_cs = id4 - id3
    ms4i = ms4.copy()
    ms4i["_nid"] = id_series(ms4i)
    rem_df = ms4i.loc[ms4i["_nid"].isin(removed_cs)]
    # G of comparison_stars-era03 members missing in era04 file
    ms3i = ms3.copy()
    ms3i["_nid"] = id_series(ms3i)
    rem_from_e3 = ms3i.loc[ms3i["_nid"].isin(removed_cs)]

    pools["era03_cs_ids_not_in_era04_cs"] = {
        "n": int(len(removed_cs)),
        "G_from_era03_ms": g_dist(rem_from_e3["phot_g_mean_mag"]) if not rem_from_e3.empty else g_dist(pd.Series(dtype=float)),
        "bp_rp_p50_era03": bprp_p50(rem_from_e3["bp_rp"]) if not rem_from_e3.empty else float("nan"),
        "G_from_era04_ms": g_dist(rem_df["phot_g_mean_mag"]) if not rem_df.empty else g_dist(pd.Series(dtype=float)),
    }
    pools["era04_cs_ids_not_in_era03_cs"] = {"n": int(len(added_cs))}

    bo_replay = next(r for r in replay_rows if r["target"] == "BO")
    collapse = abs(float(bo_replay["dmag_final_replay_era03_ctmed_median_mmag"])) < 8.0

    out = {
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "compared": "era03 vs era04 comparison_stars.csv (AC/ZP export) and D3 global_comp_pool on era04 masterstars; mag_calib_final replay uses era03 ct_bp_rp_comp_med on era04 mag_calib+ac",
        "pools": pools,
        "drop_split": drop_split,
        "loo": {
            "k": K_LOO,
            "n_d3_survivors_with_loo": int(np.isfinite(survivors["r"]).sum()),
            "median_r": float(np.nanmedian(survivors["r"])),
            "bright_decile_median_r": bright_decile_r,
            "G_where_median_r_crosses_5": g_cross,
            "n_fail_k_photon": int(survivors["fail_k_photon"].sum()),
            "fail_k_G": g_dist(fail_k["phot_g_mean_mag"]) if not fail_k.empty else g_dist(pd.Series(dtype=float)),
            "bright_end_r_exceeds_5": bool(math.isfinite(bright_decile_r) and bright_decile_r > 5.0),
        },
        "replay": replay_rows,
        "plus59_collapses_to_mmag": collapse,
        "plus59_cause": "pool_gate_ct_bprp_med" if collapse else "not_pool_gate_alone",
        "fwhm_px": fwhm,
        "solve_rms_px": solve_rms,
    }
    # compact CSV for drop groups
    rows = []
    for key in ("source_state", "snr_ap_pixscaled_lt_10", "vy_identity_gate", "gaia_dao_resid", "k_photon_among_d3_survivors"):
        g = drop_split[key]
        rows.append({"group": key, "n": g["n"], **{f"G_{k}": v for k, v in g["G"].items()}, "bp_rp_p50": g["bp_rp_p50"]})
    pd.DataFrame(rows).to_csv(SESSION / "c63b_m2_drop_split.csv", index=False)
    return out


def main() -> None:
    t_all = time.perf_counter()
    m1 = measure_m1()
    m2 = measure_m2()
    summary = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "HEAD": git_out(["rev-parse", "HEAD"]),
        "m1": m1,
        "m2": m2,
        "elapsed_s_total": round(time.perf_counter() - t_all, 3),
    }
    (SESSION / "c63b_m1.json").write_text(json.dumps(m1, indent=2, default=str) + "\n", encoding="ascii")
    (SESSION / "c63b_m2.json").write_text(json.dumps(m2, indent=2, default=str) + "\n", encoding="ascii")
    (SESSION / "c63b_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="ascii")
    print("M1 elapsed_s", m1["elapsed_s"], "sha_equal", m1["sha_all_probe_equal"], "pixels_differ", m1["pixels_differ"])
    print("M2 elapsed_s", m2["elapsed_s"], "collapse", m2["plus59_collapses_to_mmag"], "cause", m2["plus59_cause"])
    print("D3 n_in", m2["drop_split"]["n_pre_d3"], "n_out", m2["drop_split"]["n_d3_out"])
    print("G_cross_r5", m2["loo"]["G_where_median_r_crosses_5"], "bright_decile_r", m2["loo"]["bright_decile_median_r"])
    print("wrote", SESSION / "c63b_summary.json")


if __name__ == "__main__":
    main()
