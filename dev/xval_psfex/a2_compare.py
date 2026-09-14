# -*- coding: ascii -*-
"""EPSF-XVAL-A2-COMPARE: PSFEx vs VYVAR ePSF (dev-only; no src_py imports).

Linux OUT_DIR on disk is session_.../a2/out/ (run_log / work / deg3),
not session_.../a2/{run_log,work,deg3}. Bound here; not improvised
silently. a2/ is gitignored (Milan 2026-09-14 local-only).
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

REPO = Path(__file__).resolve().parents[2]
SESSION = REPO / "dev" / "results" / "context" / "session_20260907_epsfxval_a2"
# Actual Linux OUT_DIR (expected a2/ in the task; disk has a2/out/).
A2_OUT = SESSION / "a2" / "out"
VYREF = SESSION / "vyvar_reference"
OUT = SESSION / "a2_compare"
KIT = Path(__file__).resolve().parent
TARGETS_CSV = KIT / "targets.csv"
SNAP_LIGHTS = (
    REPO
    / "Archive"
    / "Drafts"
    / "draft_000516_snapshot_era04_20260826"
    / "detrended_aligned"
    / "lights"
    / "NoFilter_60_2"
)
MS_PATH = (
    REPO
    / "Archive"
    / "Drafts"
    / "draft_000516"
    / "platesolve"
    / "NoFilter_60_2"
    / "masterstars_full_match.csv"
)
PINS_CSV = REPO / "dev" / "validation" / "pinned_ensembles.csv"
G4_PATHS = {
    "csv": REPO
    / "Archive"
    / "Drafts"
    / "draft_000516"
    / "platesolve"
    / "NoFilter_60_2"
    / "masterstars_full_match.csv",
    "fits": REPO
    / "Archive"
    / "Drafts"
    / "draft_000516"
    / "platesolve"
    / "NoFilter_60_2"
    / "MASTERSTAR.fits",
    "epsf": REPO
    / "Archive"
    / "Drafts"
    / "draft_000516"
    / "platesolve"
    / "NoFilter_60_2"
    / "masterstar_epsf.fits",
}
G4_EXPECT = {"csv": "bfa24039", "fits": "13e77cf8", "epsf": "172f9540"}

TARGET_CID = "1498613634033133184"
CHECK_CID = "1497613731286514432"
ENS_IDS = [
    "1497771992240531712",
    "1499200223486564608",
    "1497974027502858240",
    "1497368849430107904",
]
R_MATCH = 2.0
PROBE_STEMS = [
    "BO_CVn_Light_001",
    "BO_CVn_Light_037",
    "BO_CVn_Light_076",
    "BO_CVn_Light_109",
    "BO_CVn_Light_148",
]
PASS2_COLS = [
    "X_IMAGE",
    "Y_IMAGE",
    "XPSF_IMAGE",
    "YPSF_IMAGE",
    "FLUX_PSF",
    "FLUXERR_PSF",
    "MAG_PSF",
    "MAGERR_PSF",
    "CHI2_PSF",
    "FLAGS",
]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def g4_live_516() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for key, path in G4_PATHS.items():
        digest = _sha256_file(path)
        exp = G4_EXPECT[key]
        out[key] = {
            "path": str(path),
            "sha256": digest,
            "prefix": digest[:8],
            "verdict": "PASS" if digest.startswith(exp) else "FAIL",
        }
    return out


def flux_to_inst_mag(flux: np.ndarray) -> np.ndarray:
    """Same as run_xval_a1.py _flux_to_inst_mag: -2.5 log10(F), no ZP."""
    f = np.asarray(flux, dtype=np.float64)
    out = np.full(f.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(f) & (f > 0)
    out[ok] = -2.5 * np.log10(f[ok])
    return out


def rms_after_median(d: np.ndarray) -> float:
    v = np.asarray(d, dtype=np.float64)
    ok = np.isfinite(v)
    if int(ok.sum()) < 2:
        return float("nan")
    v = v[ok] - float(np.median(v[ok]))
    return float(np.sqrt(np.mean(v * v)))


def rebuild_delta(
    target_flux: np.ndarray,
    comp_flux: dict[str, np.ndarray],
    comp_ids: list[str],
) -> np.ndarray:
    """AIJ tot_C_cnts: delta = m_t - (-2.5 log10(sum F_c)).

    Replicates ``dev/xval_pythonphot/run_xval_a1.py:449-478``
    (``rebuild_delta``) and ``photometry_lightcurve.ensemble_normalize``
    ``:677-679`` / ``:765-774`` (plain flux sum; weights do not enter).
    Full pinned membership or NaN.
    """
    ft = np.asarray(target_flux, dtype=np.float64)
    n = len(ft)
    delta = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if not (math.isfinite(float(ft[i])) and float(ft[i]) > 0):
            continue
        csum = 0.0
        missing = False
        for cid in comp_ids:
            arr = comp_flux.get(cid)
            if arr is None or i >= len(arr):
                missing = True
                break
            fv = float(arr[i])
            if not (math.isfinite(fv) and fv > 0):
                missing = True
                break
            csum += fv
        if missing or csum <= 0:
            continue
        delta[i] = -2.5 * math.log10(float(ft[i]) / csum)
    return delta


def read_targets() -> pd.DataFrame:
    t = pd.read_csv(TARGETS_CSV, comment="#", dtype={"catalog_id": str})
    t["catalog_id"] = t["catalog_id"].astype(str).str.strip()
    return t


def read_g_map() -> dict[str, float]:
    ms = pd.read_csv(MS_PATH, dtype={"catalog_id": str})
    ms["catalog_id"] = ms["catalog_id"].astype(str).str.strip()
    out: dict[str, float] = {}
    gcol = "phot_g_mean_mag" if "phot_g_mean_mag" in ms.columns else "catalog_mag"
    for _, r in ms.iterrows():
        cid = str(r["catalog_id"])
        g = float(pd.to_numeric(r.get(gcol), errors="coerce"))
        out[cid] = g
    return out


def verify_pins() -> None:
    pins = pd.read_csv(PINS_CSV, dtype=str)
    tid_col = pins.columns[0]
    mem_col = pins.columns[1]
    sub = pins[pins[tid_col].astype(str).str.strip() == TARGET_CID]
    members = [str(x).strip() for x in sub[mem_col].tolist()]
    if sorted(members) != sorted(ENS_IDS):
        raise SystemExit(
            f"STOP: pinned ensemble for {TARGET_CID} is {members}, expected {ENS_IDS}"
        )
    side = json.loads((VYREF / "ensemble_sidecar.json").read_text(encoding="utf-8"))
    if str(side.get("ensemble_source")) != "pinned":
        raise SystemExit(f"STOP: ensemble_source={side.get('ensemble_source')!r} not pinned")
    if [str(x) for x in side.get("ensemble_ids", [])] != ENS_IDS:
        raise SystemExit(f"STOP: sidecar ensemble_ids={side.get('ensemble_ids')}")


def read_pass2(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        comment="#",
        sep=r"\s+",
        header=None,
        names=PASS2_COLS,
        engine="python",
    )
    for c in PASS2_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_wcs(fits_path: Path) -> WCS:
    with fits.open(fits_path) as hdul:
        hdr = hdul[0].header
        # Never use SCALE; WCS (PC + CDELT) is authoritative.
        _ = hdr.get("SCALE")
        w = WCS(hdr)
    scales = proj_plane_pixel_scales(w)
    if scales is None or not np.isfinite(float(scales[0])):
        raise SystemExit(f"STOP: WCS plate scale unusable on {fits_path.name}")
    return w


def world_to_pix(w: WCS, ra: np.ndarray, dec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """FITS 1-indexed pixels (SExtractor X_IMAGE / XPSF_IMAGE convention)."""
    xy = np.asarray(w.wcs_world2pix(np.column_stack([ra, dec]), 1), dtype=np.float64)
    return xy[:, 0], xy[:, 1]


def list_stems(work: Path) -> list[str]:
    return sorted(p.name for p in work.iterdir() if p.is_dir())


def match_frame(
    cat: pd.DataFrame,
    x_pred: np.ndarray,
    y_pred: np.ndarray,
    catalog_ids: list[str],
    r_match: float,
) -> pd.DataFrame:
    """Nearest neighbour on (XPSF_IMAGE, YPSF_IMAGE). Ambiguities flagged, not dropped."""
    xs = cat["XPSF_IMAGE"].to_numpy(dtype=np.float64)
    ys = cat["YPSF_IMAGE"].to_numpy(dtype=np.float64)
    valid = np.isfinite(xs) & np.isfinite(ys)
    xs_v = xs[valid]
    ys_v = ys[valid]
    idx_v = np.flatnonzero(valid)
    rows: list[dict[str, object]] = []
    claimed: dict[int, list[int]] = {}
    for i, cid in enumerate(catalog_ids):
        rec: dict[str, object] = {
            "catalog_id": cid,
            "x_pred": float(x_pred[i]),
            "y_pred": float(y_pred[i]),
            "matched": False,
            "ambiguous": False,
            "ambiguity_kind": "",
            "nn_dist_px": float("nan"),
            "n_inside_r": 0,
            "pass2_row": -1,
            "XPSF_IMAGE": float("nan"),
            "YPSF_IMAGE": float("nan"),
            "FLUX_PSF": float("nan"),
            "MAG_PSF": float("nan"),
            "FLAGS": float("nan"),
            "flux_nonfinite": False,
        }
        if not (math.isfinite(float(x_pred[i])) and math.isfinite(float(y_pred[i]))):
            rec["ambiguity_kind"] = "pred_nan"
            rows.append(rec)
            continue
        if xs_v.size == 0:
            rows.append(rec)
            continue
        d = np.hypot(xs_v - float(x_pred[i]), ys_v - float(y_pred[i]))
        j = int(np.argmin(d))
        dist = float(d[j])
        n_in = int((d <= r_match).sum())
        rec["nn_dist_px"] = dist
        rec["n_inside_r"] = n_in
        if n_in >= 2:
            rec["ambiguous"] = True
            rec["ambiguity_kind"] = "two_pass2_in_r"
        if dist <= r_match:
            rec["matched"] = True
            prow = int(idx_v[j])
            rec["pass2_row"] = prow
            claimed.setdefault(prow, []).append(i)
            hit = cat.iloc[prow]
            rec["XPSF_IMAGE"] = float(hit["XPSF_IMAGE"])
            rec["YPSF_IMAGE"] = float(hit["YPSF_IMAGE"])
            flux = float(hit["FLUX_PSF"])
            rec["FLUX_PSF"] = flux
            rec["MAG_PSF"] = float(hit["MAG_PSF"])
            rec["FLAGS"] = float(hit["FLAGS"])
            rec["flux_nonfinite"] = not (math.isfinite(flux) and flux > 0)
        rows.append(rec)
    for prow, owners in claimed.items():
        if len(owners) < 2:
            continue
        dists = [float(rows[i]["nn_dist_px"]) for i in owners]
        keep = owners[int(np.argmin(dists))]
        for i in owners:
            rows[i]["ambiguous"] = True
            rows[i]["ambiguity_kind"] = (
                "shared_pass2_kept" if i == keep else "shared_pass2_loser"
            )
            if i != keep:
                rows[i]["matched"] = False
    return pd.DataFrame(rows)


def probe_tolerance(work: Path, targets: pd.DataFrame) -> dict[str, float]:
    stems = list_stems(work)
    have = [s for s in PROBE_STEMS if s in stems]
    if len(have) < 5:
        have = stems[:5]
    if len(have) < 5:
        raise SystemExit(f"STOP: fewer than 5 frames for NN probe ({have})")
    ids = targets["catalog_id"].tolist()
    ra = targets["ra"].to_numpy(dtype=np.float64)
    dec = targets["dec"].to_numpy(dtype=np.float64)
    dists: list[float] = []
    n_amb = 0
    n_att = 0
    per_frame: list[dict[str, object]] = []
    for stem in have:
        cat = read_pass2(work / stem / f"{stem}_pass2.cat")
        w = load_wcs(SNAP_LIGHTS / f"{stem}.fits")
        xp, yp = world_to_pix(w, ra, dec)
        m = match_frame(cat, xp, yp, ids, R_MATCH)
        n_att += int(len(m))
        n_amb += int(m["ambiguous"].sum())
        true_d = m.loc[m["matched"], "nn_dist_px"].to_numpy(dtype=np.float64)
        dists.extend([float(x) for x in true_d if math.isfinite(x)])
        per_frame.append(
            {
                "stem": stem,
                "n_matched": int(m["matched"].sum()),
                "n_ambiguous": int(m["ambiguous"].sum()),
                "nn_med": float(np.median(true_d)) if true_d.size else float("nan"),
                "nn_p95": float(np.percentile(true_d, 95)) if true_d.size else float("nan"),
            }
        )
    arr = np.asarray(dists, dtype=np.float64)
    summary = {
        "n_probe_frames": float(len(have)),
        "n_true_match_dists": float(arr.size),
        "nn_median_px": float(np.median(arr)) if arr.size else float("nan"),
        "nn_p95_px": float(np.percentile(arr, 95)) if arr.size else float("nan"),
        "ambiguity_rate": float(n_amb) / float(n_att) if n_att else float("nan"),
        "n_ambiguous": float(n_amb),
        "n_attempts": float(n_att),
    }
    pd.DataFrame(per_frame).to_csv(OUT / "match_probe_5frames.csv", index=False)
    return summary


def run_side(label: str, work: Path, targets: pd.DataFrame, gmap: dict[str, float]) -> dict[str, object]:
    ids = targets["catalog_id"].tolist()
    ra = targets["ra"].to_numpy(dtype=np.float64)
    dec = targets["dec"].to_numpy(dtype=np.float64)
    stems = list_stems(work)
    match_rows: list[pd.DataFrame] = []
    for stem in stems:
        cat_path = work / stem / f"{stem}_pass2.cat"
        cat = read_pass2(cat_path)
        missing = [c for c in PASS2_COLS if c not in cat.columns]
        if missing:
            raise SystemExit(f"STOP: pass2 columns missing {missing} on {cat_path}")
        w = load_wcs(SNAP_LIGHTS / f"{stem}.fits")
        xp, yp = world_to_pix(w, ra, dec)
        m = match_frame(cat, xp, yp, ids, R_MATCH)
        m["stem"] = stem
        m["source_file"] = f"proc_{stem}.csv"
        match_rows.append(m)
    matches = pd.concat(match_rows, ignore_index=True)
    matches.to_csv(OUT / f"match_rows_{label}.csv", index=False)

    census_rows = []
    for cid in ids:
        sub = matches[matches["catalog_id"] == cid]
        n_matched = int(sub["matched"].sum())
        n_unmatched = int((~sub["matched"]).sum())
        n_flagged = int(
            ((sub["matched"]) & (pd.to_numeric(sub["FLAGS"], errors="coerce") > 0)).sum()
        )
        n_nonfinite = int(sub["flux_nonfinite"].sum())
        n_amb = int(sub["ambiguous"].sum())
        census_rows.append(
            {
                "catalog_id": cid,
                "role": str(targets.loc[targets["catalog_id"] == cid, "role"].iloc[0])
                if "role" in targets.columns
                else "",
                "phot_g_mean_mag": gmap.get(cid, float("nan")),
                "n_frames": int(len(sub)),
                "n_matched": n_matched,
                "n_unmatched": n_unmatched,
                "n_flagged": n_flagged,
                "n_nonfinite": n_nonfinite,
                "n_ambiguous": n_amb,
            }
        )
    census = pd.DataFrame(census_rows)
    census.to_csv(OUT / f"match_census_{label}.csv", index=False)
    return {"matches": matches, "census": census, "stems": stems}


def vyvar_flux_table() -> pd.DataFrame:
    proc = pd.read_csv(VYREF / "proc_psf_flux.csv", comment="#", dtype={"catalog_id": str})
    proc["catalog_id"] = proc["catalog_id"].astype(str).str.strip()
    proc["psf_flux"] = pd.to_numeric(proc["psf_flux"], errors="coerce")
    return proc


def m1_table(
    matches: pd.DataFrame,
    proc: pd.DataFrame,
    targets: pd.DataFrame,
    gmap: dict[str, float],
    label: str,
) -> pd.DataFrame:
    """d_si = m_psf_VYVAR - m_PSFEx; per-star median removed; RMS over epochs.

    Population: every targets.csv star (n=65: 60 PSF-LC + 4 ensemble + check).
    VYVAR mag from proc_psf_flux.csv psf_flux. PSFEx MAG_PSF as-is.
    Same residual construction as A1 M1 (run_xval_a1.py:646-669).
    """
    rows = []
    for cid in targets["catalog_id"].tolist():
        msub = matches[(matches["catalog_id"] == cid) & (matches["matched"])]
        vsub = proc[proc["catalog_id"] == cid][["source_file", "psf_flux"]]
        merged = msub.merge(vsub, on="source_file", how="outer")
        m_v = flux_to_inst_mag(pd.to_numeric(merged["psf_flux"], errors="coerce").to_numpy())
        m_p = pd.to_numeric(merged["MAG_PSF"], errors="coerce").to_numpy()
        if "flux_nonfinite" in merged.columns:
            flux_bad = merged["flux_nonfinite"].astype("boolean").fillna(True).to_numpy()
        else:
            flux_bad = np.ones(len(merged), dtype=bool)
        m_p = np.where(np.asarray(flux_bad, dtype=bool), np.nan, m_p)
        d = m_v - m_p
        ok = np.isfinite(d)
        rms = rms_after_median(d)
        rows.append(
            {
                "catalog_id": cid,
                "phot_g_mean_mag": gmap.get(cid, float("nan")),
                "role": str(targets.loc[targets["catalog_id"] == cid, "role"].iloc[0])
                if "role" in targets.columns
                else "",
                "n_ok": int(ok.sum()),
                "n_nan": int((~ok).sum()),
                "rms_mag": rms,
                "rms_mmag": rms * 1000.0 if math.isfinite(rms) else float("nan"),
                "median_d_mag": float(np.nanmedian(d)) if int(ok.sum()) else float("nan"),
            }
        )
    m1 = pd.DataFrame(rows).sort_values(
        ["phot_g_mean_mag", "catalog_id"], na_position="last"
    )
    m1.to_csv(OUT / f"m1_per_star{label}.csv", index=False)
    return m1


def flux_series(
    matches: pd.DataFrame,
    proc: pd.DataFrame,
    cid: str,
    stems: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Aligned PSFEx / VYVAR fluxes on the stem list. PSFEx uses FLUX_PSF."""
    n = len(stems)
    f_p = np.full(n, np.nan)
    f_v = np.full(n, np.nan)
    reasons = [""] * n
    msub = matches[matches["catalog_id"] == cid]
    vsub = proc[proc["catalog_id"] == cid]
    for i, stem in enumerate(stems):
        mh = msub[msub["stem"] == stem]
        if mh.empty or not bool(mh.iloc[0]["matched"]):
            reasons[i] = "psfex_unmatched"
        elif bool(mh.iloc[0]["flux_nonfinite"]):
            reasons[i] = "psfex_nonfinite"
        else:
            f_p[i] = float(mh.iloc[0]["FLUX_PSF"])
        vh = vsub[vsub["source_file"] == f"proc_{stem}.csv"]
        if vh.empty:
            reasons[i] = (reasons[i] + "+vyvar_absent").lstrip("+")
        else:
            fv = float(pd.to_numeric(vh.iloc[0]["psf_flux"], errors="coerce"))
            if math.isfinite(fv) and fv > 0:
                f_v[i] = fv
            else:
                reasons[i] = (reasons[i] + "+vyvar_nonfinite").lstrip("+")
    return f_p, f_v, reasons


def m2_product(
    matches: pd.DataFrame,
    proc: pd.DataFrame,
    stems: list[str],
    label: str,
) -> dict[str, object]:
    """Differential LC on the identical epoch set (target + 4 ensemble valid both sides)."""
    need = [TARGET_CID, CHECK_CID] + ENS_IDS
    series: dict[str, dict[str, np.ndarray]] = {}
    loss: dict[str, list[str]] = {}
    for cid in need:
        fp, fv, reasons = flux_series(matches, proc, cid, stems)
        series[cid] = {"psfex": fp, "vyvar": fv}
        loss[cid] = reasons

    n = len(stems)
    both_ok = np.ones(n, dtype=bool)
    lost_rows = []
    for i, stem in enumerate(stems):
        parts = []
        for cid in [TARGET_CID] + ENS_IDS:
            ok_p = math.isfinite(float(series[cid]["psfex"][i])) and float(series[cid]["psfex"][i]) > 0
            ok_v = math.isfinite(float(series[cid]["vyvar"][i])) and float(series[cid]["vyvar"][i]) > 0
            if not ok_p:
                parts.append(f"{cid}:psfex={loss[cid][i] or 'invalid'}")
            if not ok_v:
                parts.append(f"{cid}:vyvar={loss[cid][i] or 'invalid'}")
        if parts:
            both_ok[i] = False
            lost_rows.append({"stem": stem, "reason": ";".join(parts)})
    pd.DataFrame(lost_rows).to_csv(OUT / f"m2_frames_lost_{label}.csv", index=False)

    def _masked(arr: np.ndarray) -> np.ndarray:
        out = np.array(arr, dtype=np.float64, copy=True)
        out[~both_ok] = np.nan
        return out

    vy_comp = {cid: _masked(series[cid]["vyvar"]) for cid in ENS_IDS}
    px_comp = {cid: _masked(series[cid]["psfex"]) for cid in ENS_IDS}
    vy_tgt = rebuild_delta(_masked(series[TARGET_CID]["vyvar"]), vy_comp, ENS_IDS)
    px_tgt = rebuild_delta(_masked(series[TARGET_CID]["psfex"]), px_comp, ENS_IDS)
    vy_chk = rebuild_delta(_masked(series[CHECK_CID]["vyvar"]), vy_comp, ENS_IDS)
    px_chk = rebuild_delta(_masked(series[CHECK_CID]["psfex"]), px_comp, ENS_IDS)

    # Sidecar cross-check (target only): rebuilt VYVAR vs frozen psf_delta_mag.
    lc = pd.read_csv(
        VYREF / "lightcurves" / f"lightcurve_{TARGET_CID}_psf.csv",
        comment="#",
        dtype={"source_file": str},
    )
    side = np.full(n, np.nan)
    for i, stem in enumerate(stems):
        hit = lc[lc["source_file"] == f"proc_{stem}.csv"]
        if hit.empty:
            continue
        side[i] = float(pd.to_numeric(hit.iloc[0]["psf_delta_mag"], errors="coerce"))
    # Compare on the same both-ok set.
    side_m = np.array(side, dtype=np.float64)
    side_m[~both_ok] = np.nan
    d_ref = vy_tgt - side_m
    rms_ref = rms_after_median(d_ref)
    # Also raw RMS without median (expected ~0).
    ok_ref = np.isfinite(d_ref)
    rms_ref_raw = (
        float(np.sqrt(np.mean(d_ref[ok_ref] ** 2))) if int(ok_ref.sum()) else float("nan")
    )

    def _pack(name: str, vy: np.ndarray, px: np.ndarray) -> pd.DataFrame:
        d = vy - px
        med = float(np.nanmedian(d[np.isfinite(d)])) if np.isfinite(d).any() else float("nan")
        resid = d - med
        ep = pd.DataFrame(
            {
                "stem": stems,
                "in_identical_set": both_ok,
                "diff_vyvar": vy,
                "diff_psfex": px,
                "d_e": d,
                "resid_after_median": resid,
            }
        )
        fname = f"m2_epochs_{name}.csv" if label == "deg2" else f"m2_epochs_{name}_deg3.csv"
        ep.to_csv(OUT / fname, index=False)
        return ep

    ep_t = _pack("target", vy_tgt, px_tgt)
    ep_c = _pack("check", vy_chk, px_chk)
    rms_t = rms_after_median(ep_t["d_e"].to_numpy())
    rms_c = rms_after_median(ep_c["d_e"].to_numpy())

    def _top(ep: pd.DataFrame) -> list[dict[str, object]]:
        sub = ep[ep["in_identical_set"] & np.isfinite(ep["resid_after_median"])]
        sub = sub.reindex(sub["resid_after_median"].abs().sort_values(ascending=False).index)
        out = []
        for _, r in sub.head(8).iterrows():
            out.append(
                {
                    "stem": r["stem"],
                    "resid_mmag": float(r["resid_after_median"]) * 1000.0,
                }
            )
        return out

    return {
        "n_stems": n,
        "n_identical": int(both_ok.sum()),
        "n_lost": int((~both_ok).sum()),
        "n_finite_target": int(np.isfinite(ep_t.loc[ep_t["in_identical_set"], "d_e"]).sum()),
        "n_finite_check": int(np.isfinite(ep_c.loc[ep_c["in_identical_set"], "d_e"]).sum()),
        "rms_target_mmag": rms_t * 1000.0 if math.isfinite(rms_t) else float("nan"),
        "rms_check_mmag": rms_c * 1000.0 if math.isfinite(rms_c) else float("nan"),
        "sidecar_rms_after_median_mmag": rms_ref * 1000.0 if math.isfinite(rms_ref) else float("nan"),
        "sidecar_rms_raw_mmag": rms_ref_raw * 1000.0 if math.isfinite(rms_ref_raw) else float("nan"),
        "sidecar_n": int(ok_ref.sum()),
        "top_target": _top(ep_t),
        "top_check": _top(ep_c),
    }


def reading(rms_t: float, rms_c: float) -> str:
    vals = [rms_t, rms_c]
    if any(not math.isfinite(v) for v in vals):
        return "R-A2-3: > 10.0 mmag (RMS non-finite) -> VYVAR-side root cause; first suspect EPSF-SHAPE-01."
    mx = max(vals)
    if rms_t <= 3.0 and rms_c <= 3.0:
        return "R-A2-1: target AND check RMS <= 3.0 mmag -> AIJ-class validation of the VYVAR ePSF path."
    if mx <= 10.0:
        return "R-A2-2: 3.0-10.0 mmag -> open investigation, no closure claim."
    return "R-A2-3: > 10.0 mmag -> VYVAR-side root cause; first suspect EPSF-SHAPE-01."


def premise_layout() -> dict[str, object]:
    log = A2_OUT / "run_log.txt"
    last = log.read_text(encoding="utf-8", errors="replace").splitlines()[-1] if log.is_file() else ""
    work = A2_OUT / "work"
    deg3 = A2_OUT / "deg3" / "work"
    return {
        "expected_a2_root": str(SESSION / "a2"),
        "actual_linux_outdir": str(A2_OUT),
        "layout_note": "task expected a2/{run_log,work,deg3}; disk is a2/out/{run_log,work,deg3} plus a2/draft_... snapshot copy",
        "run_log_last": last,
        "n_work": len(list_stems(work)) if work.is_dir() else 0,
        "n_deg3": len(list_stems(deg3)) if deg3.is_dir() else 0,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    layout = premise_layout()
    print("[a2] layout", json.dumps(layout, indent=2))
    if layout["run_log_last"] != "done n_frames=134 n_fail=0":
        print("[a2] WARN run_log last line:", layout["run_log_last"])
    if int(layout["n_work"]) != 134 or int(layout["n_deg3"]) != 134:
        raise SystemExit(f"STOP: stem counts {layout}")

    targets = read_targets()
    if len(targets) != 65:
        raise SystemExit(f"STOP: targets.csv n={len(targets)} expected 65")
    gmap = read_g_map()
    verify_pins()
    proc = vyvar_flux_table()
    if len(proc) != 7795:
        print(f"[a2] WARN proc_psf_flux rows={len(proc)} expected 7795")

    probe = probe_tolerance(A2_OUT / "work", targets)
    print("[a2] NN probe deg2", probe)
    if float(probe["nn_p95_px"]) > 1.5:
        raise SystemExit(
            f"STOP: NN p95 of true matches {probe['nn_p95_px']:.4f} px > 1.5; "
            "tolerance premise is wrong"
        )
    if float(probe["ambiguity_rate"]) > 0.01:
        raise SystemExit(
            f"STOP: ambiguity rate {probe['ambiguity_rate']:.4f} > 1%; "
            "tolerance premise is wrong"
        )

    results: dict[str, object] = {"layout": layout, "probe_deg2": probe}
    for label, work in (("deg2", A2_OUT / "work"), ("deg3", A2_OUT / "deg3" / "work")):
        side = run_side(label, work, targets, gmap)
        suffix = "" if label == "deg2" else "_deg3"
        m1 = m1_table(side["matches"], proc, targets, gmap, suffix)
        m2 = m2_product(side["matches"], proc, side["stems"], label)
        results[label] = {
            "census_n_stars": int(len(side["census"])),
            "m1_n": int(len(m1)),
            "m1_target_mmag": float(
                m1.loc[m1["catalog_id"] == TARGET_CID, "rms_mmag"].iloc[0]
            ),
            "m2": m2,
        }
        print(f"[a2] {label} M1 target RMS mmag", results[label]["m1_target_mmag"])
        print(f"[a2] {label} M2", {k: m2[k] for k in m2 if k not in ("top_target", "top_check")})

    m2d = results["deg2"]["m2"]  # type: ignore[index]
    if float(m2d["sidecar_rms_raw_mmag"]) > 0.05:  # 0.05 mmag ~ float/rounding
        raise SystemExit(
            f"STOP: rebuilt VYVAR diff LC vs frozen sidecar RMS raw "
            f"{m2d['sidecar_rms_raw_mmag']} mmag (expected ~0)"
        )

    results["reading_deg2"] = reading(
        float(m2d["rms_target_mmag"]), float(m2d["rms_check_mmag"])
    )
    results["scale"] = {
        "aperture_vs_aij_mmag": 1.9503,
        "a1_target_mmag": 34.34,
        "a1_check_mmag": 31.79,
        "g3_internal_bound_mmag": 12.5,
    }
    results["g4"] = g4_live_516()
    (OUT / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("[a2] reading", results["reading_deg2"])
    print("[a2] g4", results["g4"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
